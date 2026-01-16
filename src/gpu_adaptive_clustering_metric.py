import logging
import time
from typing import Optional

from cuml.cluster import KMeans as cuKMeans
from cupyx import scatter_max as cupy_scatter_max
from sklearn.cluster import KMeans

from src import _HAS_GPU, Array, xp
from src.logging_config import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)


def generate_synthetic_points_batched(
    X: Array, starts: Array, counts: Array, MIN_PTS_PER_CLUSTER: int
) -> tuple[Optional[Array], Optional[Array]]:
    """
    Batch generate synthetic (convex-combination) points for clusters smaller than N_min,
    over all clusters. This is intended to augment small clusters with additional points
    to stabilize error-estimation without changing the original data.

    For each cluster k with count m_k < MIN_PTS_PER_CLUSTER, we create n_needed[k] = MIN_PTS_PER_CLUSTER - m_k
    synthetic points. Each synthetic point is a convex combination of two randomly
    chosen points from that same cluster:
        x_synth = a * x_i + (1 - a) * x_j,   where a ~ Uniform(0,1),
    and i, j are random indices within the cluster segment.

    Args:
        X: (N, d) Array of data points sorted by cluster labels.
        starts: (K,) Array of starting indices for each cluster.
        counts: (K,) Array of counts for each cluster.
        MIN_PTS_PER_CLUSTER: Minimum number of points required per cluster.

    Returns:
        X_synth: (M, d) synthetic points, where M = sum_k max(0, MIN_PTS_PER_CLUSTER -
        counts[k]), or None if no synthesis is needed.
        cluster_ids_synth: (M,) integer cluster id for each synthetic point (so you can
            associate each synthetic point back to its cluster), or None if none.
    """
    # Identify how many synthetic points are needed per cluster.
    n_needed = xp.maximum(0, MIN_PTS_PER_CLUSTER - counts)  # (K,)
    total_synth = int(xp.sum(n_needed))

    # If no synthesis needed, return None.
    if total_synth == 0:
        return None, None

    # Clusters that actually need synthetic points.
    small_cluster_indices = xp.nonzero(n_needed)[0]  # (K_small,)
    needed_counts = n_needed[small_cluster_indices]  # (K_small,)

    # For each synthetic point, assign it to which cluster it belongs.
    # E.g., if C0: 3 points, C1: 0 points, C2: 2 points, then
    # cluster_ids_synth = [0,0,0,2,2].
    cluster_ids_synth = repeats(small_cluster_indices, needed_counts)

    # For each synthetic point, pick two random indices within that cluster.
    synth_starts = starts[cluster_ids_synth]  # (M,)
    synth_counts = counts[cluster_ids_synth]  # (M,)

    # Generate random offsets.
    offsets1 = xp.floor(xp.random.rand(total_synth) * synth_counts).astype(xp.int32)
    offsets2 = xp.floor(xp.random.rand(total_synth) * synth_counts).astype(xp.int32)

    # Convert offsets to absolute indices into X.
    idx1 = synth_starts + offsets1
    idx2 = synth_starts + offsets2

    # Form convex combinations.
    alpha = xp.random.rand(total_synth, 1).astype(X.dtype)  # (M, 1)
    X_synth = alpha * X[idx1] + (1.0 - alpha) * X[idx2]  # (M, d)

    return X_synth, cluster_ids_synth


def batched_estimate_hessian_symmetric(
    DeltaX: Array, DeltaG: Array, starts: Array
) -> Array:
    """
    Batch estimate the symmetric Hessian matrices for multiple clusters.
    Equations (per cluster k):
        X = [Δx_1 ... Δx_m] ∈ R^{d x m}
        G = [Δg_1 ... Δg_m] ∈ R^{d x m}

        C = X X^T = Σ Δx Δx^T
        E = X G^T = Σ Δx Δg^T
        Solve: H C + C H = E + E^T via C = U D U^T

    Args:
        DeltaX: (N, d) *Centered* array of data points sorted by cluster labels,
            i.e. rows are Δx_i^T = (x_i - μ_k)^T for the cluster that point i belongs to.
        DeltaG: (N, d) *Centered* array of gradients corresponding to DeltaX,
            i.e. rows are Δg_i^T = (g_i - g(μ_k))^T for the cluster that point i belongs
            to.
        starts: (K,) Array of starting indices for each cluster.

    Returns:
        H: (K, d, d) Batch of symmetric Hessians for each cluster
    """
    # Build per-point outer products:
    # XX_all[i] = Δx_i Δx_i^T
    # XG_all[i] = Δx_i Δg_i^T
    XX_all = DeltaX[:, :, None] * DeltaX[:, None, :]  # (N, d, d)
    XG_all = DeltaX[:, :, None] * DeltaG[:, None, :]  # (N, d, d)

    # Segment sums (cluster-wise):
    # C_k = Σ Δx Δx^T
    # E_k = Σ Δx Δg^T
    C = xp.add.reduceat(XX_all, starts)  # (K, d, d)
    E = xp.add.reduceat(XG_all, starts)  # (K, d, d)

    # RHS S_k = E_k + E_k^T
    S = E + E.transpose(0, 2, 1)  # (K, d, d)

    # Batch eigen-decomposition of each C_k = U_k D_k U_k^T.
    dvals, U = xp.linalg.eigh((C + C.transpose(0, 2, 1)) * 0.5)
    dvals = xp.maximum(dvals, 1e-18)

    # Gtilde = U^T S U.
    Gtilde = U.transpose(0, 2, 1) @ S @ U

    # F_ij = Gtilde_ij / (d_i + d_j).
    denom = dvals[:, :, None] + dvals[:, None, :]
    F = Gtilde / denom

    # H = U F U^T.
    H = U @ F @ U.transpose(0, 2, 1)

    # Ensure symmetry.
    return (H + H.transpose(0, 2, 1)) * 0.5


def batched_sqrt_abs_psd(H: Array) -> Array:
    """
    Batch compute the square root of the absolute value of symmetric PSD matrices over
    all clusters.
    Equations (per matrix/cluster k):
        If H_k is symmetric, it has eigendecomposition:
            H_k = U_k diag(λ_k) U_k^T.
        Then the square root of the absolute value is:
            |H_k|^{1/2} = U_k diag(sqrt(|λ_k|)) U_k^T.

    Args:
        H: (K, d, d) Batch of symmetric matrices (corresponding to Hessians of clusters).

    Returns:
        M: (K, d, d) Batch of square-rooted absolute value matrices, where
            M_k M_k = |H_k|.
    """
    # Batch eigen-decomposition of each H_k = U_k D_k U_k^T.
    # w: (K, d) eigenvalues, U: (K, d, d) eigenvectors.
    w, U = xp.linalg.eigh(H)

    # Compute sqrt(|w|) elementwise.
    w_abs_sqrt = xp.sqrt(xp.abs(w))

    M = (U * w_abs_sqrt[:, None, :]) @ U.transpose(0, 2, 1)  # (K, d, d)
    return (M + M.transpose(0, 2, 1)) * 0.5


def batched_top_eigvec(Cov: Array) -> Array:
    """
    Batch compute the dominant (top) eigenvector of symmetric covariance matrices over
    all clusters.

    Args:
        Cov: (K, d, d) Batch of symmetric covariance matrices.

    Returns:
        v: (K, d) Batch of top eigenvectors, where v[k] is the eigenvector corresponding
            to the largest eigenvalue of Cov[k] matrix.
    """
    # Batch eigen-decomposition of symmetric matrices.
    # w: (K, d) eigenvalues (ascending), V: (K, d, d) eigenvectors.
    w, V = xp.linalg.eigh((Cov + Cov.transpose(0, 2, 1)) * 0.5)

    # Since eigenvalues are ascending, the last column of V is the top eigenvector.
    return V[:, :, -1]  # (K, d)


def batched_median_split_decision(
    scores: Array, cluster_id: Array, starts: Array, counts: Array
) -> Array:
    """
    Batch compute a median split decision from per-point scores over all clusters.
    Given per-point scores s_i and cluster ids k(i), this returns:
        decision_i = (s_i >= median_k(i))
    where median_k is the median score within cluster k, and decision_i (True/False)
    represents whether point i goes to the "right" (True) or "left" (False) child cluster.

    Args:
        scores: (N,) Array of per-point scores.
        cluster_id: (N,) Array of cluster ids for each point.
        starts: (K,) Array of starting indices for each cluster.
        counts: (K,) Array of counts for each cluster.

    Returns:
        split_decision: (N,) boolean array; True means "right child", False "left child".

    Notes:
        This performs a global radix-like sort:
            1) sort by score
            2) sort by cluster_id
        which yields (cluster_id asc, score asc within cluster).
    """
    N = int(scores.shape[0])
    if N == 0:
        return scores >= 0

    # Sort by score first.
    idx_by_score = xp.argsort(scores)  # (N,)
    cluster_sorted_by_score = cluster_id[idx_by_score]

    # Sort by cluster id, while preserving score-order within each cluster.
    idx_group = xp.argsort(cluster_sorted_by_score)  # (N,)
    order = idx_by_score[idx_group]

    # Scores sorted within clusters.
    scores_sorted = scores[order]  # (N,)

    # For each cluster segment [starts[k] : starts[k] + counts[k]),
    # median is average of middle element(s).
    i0 = starts + (counts - 1) // 2  # (K,)
    i1 = starts + counts // 2  # (K,)

    median = 0.5 * (scores_sorted[i0] + scores_sorted[i1])  # (K,)
    median_expanded = median[cluster_id]  # (N,)
    split_decision = scores >= median_expanded  # (N,)
    return split_decision


def batched_calculate_Q(
    DeltaX: Array, starts: Array, counts: Array, K: int, eps: float = 1e-12
) -> Array:
    """
    Batch compute PCA-aligned trust ellipsoid matrices Q_k over all clusters.
    This constructs, for every cluster k, a PSD matrix Q_k such that the ellipsoid:
        { y : y^T Q_k y <= 1 },
    is aligned with the cluster's PCA directions, and its axis lengths are chosen from
    max-absolute projections of points onto those PCA axes.
    Equations (per cluster k):
        Let Y_k be the centered points of cluster k (rows), and let Δx_i be rows of
        DeltaX.
        1) Compute cluster-wise covariance (scale doesn't matter here (no 1/m) since we
        only need eigenvectors):
            Cov_k = Σ Δx Δx^T    ∈ R^{d x d}

        2) PCA directions via eigendecomposition:
            Cov_k = U_k Λ_k U_k^T

        3) Axis extents via max-absolute projection:
            d_{k,j} = max | u_{k,j}^T Δx_i |    for j=1..d

        4) Ellipsoid matrix:
            Q_k = U_k diag(1/d_{k,1}^2, ..., 1/d_{k,d}^2) U_k^T

    Args:
        DeltaX: (N, d) *Centered* array of data points sorted by cluster labels,
            i.e. rows are Δx_i^T = (x_i - μ_k)^T for the cluster that point i belongs to.
        starts: (K,) Array of starting indices for each cluster.
        counts: (K,) Array of counts for each cluster.
        K: Number of clusters.
        eps: Small positive number to avoid divide-by-zero if a cluster has ~0 extent
            along some PCA axis.

    Returns:
        Q: (K, d, d) Batch of PSD trust-ellipsoid matrices Q_k.

    """
    # Build per-point outer products:
    # XX_all[i] = Δx_i Δx_i^T
    XX_all = DeltaX[:, :, None] * DeltaX[:, None, :]  # (N, d, d)
    # Segment sums (cluster-wise) to get Covariance matrices.
    Cov = xp.add.reduceat(XX_all, starts)  # (K, d, d)

    # Batch eigen-decomposition of each Cov_k = U_k D_k U_k^T.
    _, U = xp.linalg.eigh((Cov + Cov.transpose(0, 2, 1)) * 0.5)

    # Projections of points onto PCA axes.
    map_n_to_k = repeats(xp.arange(K), counts)

    # Expand U so every point gets its cluster's eigenvector matrix.
    U_expanded = U[map_n_to_k]  # (N, d, d)

    # Compute projections of points onto their cluster's PCA axes.
    proj = U_expanded.transpose(0, 2, 1) @ DeltaX[:, :, None]  # (N, d, 1)
    proj = proj[:, :, 0]  # (N, d)

    # Compute per-axis max |projection| per cluster.
    proj_abs = xp.abs(proj)
    d_axes = xp.zeros((K, proj_abs.shape[1]), dtype=proj_abs.dtype)
    if xp.__name__ == "array_api_compat.numpy":
        xp.maximum.at(d_axes, map_n_to_k, proj_abs)
    else:
        cupy_scatter_max(d_axes, map_n_to_k, proj_abs)

    # Avoid divide-by-zero.
    d_axes = xp.maximum(d_axes, eps)  # (K, d)

    # Build Q_k matrices.
    Dinv2 = 1.0 / (d_axes**2)  # (K, d)
    Q_inner = U * Dinv2[:, None, :]  # (K, d, d)
    Q = Q_inner @ U.transpose(0, 2, 1)  # (K, d, d)
    return (Q + Q.transpose(0, 2, 1)) * 0.5


def allocate_output_buffers(
    MAX_CLUSTERS_EST: int, d: int, dtype
) -> tuple[dict[str, Array], int]:
    """
    Allocate output buffers for adaptive clustering.

    Args:
        MAX_CLUSTERS_EST: Estimated maximum number of clusters.
        d: Data dimensionality.
        dtype: Data type for floating-point arrays.

    Returns:
        buffers: Dictionary of allocated output buffers.
            Keys: "centroids", "f", "g", "Q", "E_abs", "E_rel", "real_counts", "counts"
        capacity: The allocated capacity (number of clusters)
    """
    capacity = max(1, MAX_CLUSTERS_EST)
    buffers = {
        "centroids": xp.empty((capacity, d), dtype=dtype),
        "f": xp.empty((capacity,), dtype=dtype),
        "g": xp.empty((capacity, d), dtype=dtype),
        "Q": xp.empty((capacity, d, d), dtype=dtype),
        "E_abs": xp.empty((capacity,), dtype=dtype),
        "E_rel": xp.empty((capacity,), dtype=dtype),
        "real_counts": xp.empty((capacity,), dtype=xp.int64),
        "counts": xp.empty((capacity,), dtype=xp.int64),
    }
    return buffers, capacity


def adjust_buffer_capacity(
    buffers: dict[str, Array],
    current_cluster_count: int,
    current_capacity: int,
    required_total: int,
) -> tuple[dict[str, Array], int]:
    """
    Ensure that output buffers have enough capacity for required_total clusters.

    Args:
        buffers: Current output buffers.
        current_cluster_count: Current number of clusters stored.
        current_capacity: Current capacity of buffers.
        required_total: Required total capacity.

    Returns:
        new_buffers: Updated output buffers with sufficient capacity.
        new_capacity: New capacity of buffers.
    """
    if required_total <= current_capacity:
        # No need to resize.
        return buffers, current_capacity

    d = int(buffers["centroids"].shape[1])
    new_capacity = max(required_total, max(1, current_capacity) * 2)
    dtype = buffers["centroids"].dtype
    logger.warning(
        f"Resizing output buffers: {current_capacity} -> {new_capacity} clusters"
    )

    new_buffers = {
        "centroids": xp.empty((new_capacity, d), dtype=dtype),
        "f": xp.empty((new_capacity,), dtype=dtype),
        "g": xp.empty((new_capacity, d), dtype=dtype),
        "Q": xp.empty((new_capacity, d, d), dtype=dtype),
        "E_abs": xp.empty((new_capacity,), dtype=dtype),
        "E_rel": xp.empty((new_capacity,), dtype=dtype),
        "real_counts": xp.empty((new_capacity,), dtype=buffers["real_counts"].dtype),
        "counts": xp.empty((new_capacity,), dtype=buffers["counts"].dtype),
    }
    if current_cluster_count > 0:
        for k in new_buffers:
            new_buffers[k][:current_cluster_count] = buffers[k][:current_cluster_count]

    del buffers
    return new_buffers, int(new_capacity)


def adaptive_clustering(
    X: Array,
    *,
    f_fn,
    g_fn,
    ATOL: float = 1e-2,
    RTOL: float = 1e-2,
    NUM_INITIAL_CLUSTERS: int = 100,
    MIN_PTS_PER_CLUSTER: int = 100,
    MAX_CLUSTERS_EST: int = 100_000,
    SEED: int = 0,
    LOG_MEM_STATS: bool = False,
) -> dict[str, Array]:
    """
    Adaptive, GPU-native clustering driven by first-order Taylor error, with
    Hessian-informed splits for clusters that violate error tolerances.

    This routine constructs a piecewise-local surrogate of a scalar function f(x) over
    a point cloud X by forming clusters whose members are well-approximated by a
    first-order Taylor model about the cluster centroid μ_k:

        f(x) ≈ f(μ_k) + (x - μ_k)^T g(μ_k),

    where g(·) is the gradient of f. A cluster is deemed acceptable if its maximum
    pointwise linearization error is below either an absolute tolerance or a global
    relative tolerance:

        E_abs,k = max_{x in cluster k} |f(x) - f_lin(x; μ_k)|,
        E_rel,k = E_abs,k / max_x |f(x)|,

        acceptable if (E_abs,k < ATOL) OR (E_rel,k < RTOL).

    Clusters that fail are split using a second-order-informed direction derived from
    a symmetric Hessian estimate H_k (fitted from Δx and Δg within that cluster). The
    split direction is computed in the transformed coordinates z = |H_k|^{1/2} Δx, and
    then choosing the dominant PCA direction of Cov(z). Points are partitioned by a
    within-cluster median split along that direction.

    Small clusters are augmented with synthetic convex-combination points to stabilize
    covariance / ellipse estimation without altering the set of *real* points that the
    algorithm ultimately partitions (synthetic points are only used for local geometry
    estimation and are never marked inactive).

    Args:
        X: (N, d) Array of data points to cluster.
        f_fn: Callable mapping X (N, d) -> (N, 1) function values.
        g_fn: Callable mapping X (N, d) -> (N, 1, d) function gradients.
        ATOL: Absolute tolerance for max linearization error within a cluster.
        RTOL: Relative tolerance (w.r.t. global max |f|) for max error within a cluster.
        NUM_INITIAL_CLUSTERS: Number of initial KMeans clusters (cuML KMeans).
        MIN_PTS_PER_CLUSTER: Minimum points per cluster for stable geometry estimates.
            Clusters below this are augmented with synthetic points.
        MAX_CLUSTERS_EST: Initial allocation capacity for output buffers (auto-grows).
        SEED: RNG seed for KMeans and synthetic point generation.
        LOG_MEM_STATS: If True, log GPU memory stats at key points.

    Returns:
        outputs: Dictionary of arrays, truncated to the number of accepted clusters:
            - "centroids": (K_final, d) cluster centroids μ_k
            - "f":         (K_final,)   f(μ_k)
            - "g":         (K_final, d) g(μ_k)
            - "Q":         (K_final, d, d) PSD trust-ellipsoid matrices aligned with PCA
            - "E_abs":     (K_final,)   per-cluster max absolute linearization error
            - "E_rel":     (K_final,)   per-cluster max relative error (global scaling)
            - "counts":    (K_final,)   number of *real* points assigned to each cluster

    Notes:
        - Synthetic points (if generated) are labeled with global index -1 and are
          never used to deactivate points; only real points are deactivated.
        - Labels are updated only for real points on failed clusters; the split rule is
          derived from the combined (real + synth) batch to make the split direction
          more stable in under-populated clusters.
    """
    # Ensure data is on GPU.
    X = xp.asarray(X)  # (N, d)
    N, d = X.shape

    if LOG_MEM_STATS:
        log_mem_stats("Start: Adaptive Clustering")

    # Log run parameters.
    msg = f"""Adaptive Clustering Parameters:
      {ATOL=}
      {RTOL=}
      {NUM_INITIAL_CLUSTERS=:,}
      {MIN_PTS_PER_CLUSTER=:,}
      {MAX_CLUSTERS_EST=:,}
      {SEED=}
      Data Points: {X.shape[0]:,} (Dim: {X.shape[1]:,})"""
    logging.debug(msg)

    # -------------------------------------------------------------------------
    # 1) Initial clustering (coarse partition) via cuML KMeans on the GPU.
    #    The adaptive loop will refine this partition by splitting "bad" clusters.
    # -------------------------------------------------------------------------
    _start_init = time.perf_counter()
    if _HAS_GPU:
        kmeans = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
        labels = kmeans.fit_predict(X)
    else:
        kmeans = KMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
        labels = kmeans.fit_predict(X)
    logging.debug(f"Runtime - Initial KMeans: {time.perf_counter() - _start_init:.4f}s")

    # Allocate outputs with an estimated capacity; will grow if needed.
    buffers, capacity = allocate_output_buffers(MAX_CLUSTERS_EST, d, X.dtype)
    valid_cluster_count = 0

    if LOG_MEM_STATS:
        log_mem_stats("After: Initial KMeans")

    # -------------------------------------------------------------------------
    # 2) Precompute f and g for all REAL points once.
    #    We will gather these values later when constructing each active batch.
    # -------------------------------------------------------------------------
    _start_f_grad = time.perf_counter()
    F_all = f_fn(X).ravel()  # (N,)
    G_all = g_fn(X)[:, 0, :]  # (N, d)
    logging.debug(
        f"Runtime - Initial f/g eval: {time.perf_counter() - _start_f_grad:.4f}s"
    )

    assert F_all.shape == (N,), f"Expected F_all to be (N,), got {F_all.shape}"
    assert G_all.shape == (N, d), f"Expected G_all to be (N, d), got {G_all.shape}"

    # Global scaling for relative errors: use max |f| over the currently-evaluated set.
    F_inf = float(xp.max(xp.abs(F_all))) + 1e-30

    # Active-mask tracks REAL points that still need to be assigned to an acceptable cluster.
    active_mask = xp.ones(N, dtype=bool)  # (N,)

    if LOG_MEM_STATS:
        log_mem_stats("After: Precompute f/g for all points")

    # -------------------------------------------------------------------------
    # 3) Adaptive refinement loop:
    #    - batch active points by current labels
    #    - augment small clusters with synthetic points
    #    - accept clusters that satisfy error tolerances and output their metadata
    #    - split clusters that fail using Hessian-informed median split.
    # -------------------------------------------------------------------------
    level = 0
    while True:
        if LOG_MEM_STATS:
            log_mem_stats(f"Start: L{level:02}")

        n_active = int(xp.sum(active_mask))
        if n_active == 0:
            # All REAL points have been assigned to acceptable clusters.
            break
        _start_loop = time.perf_counter()

        # Extract active REAL points and their current labels.
        active_indices = xp.nonzero(active_mask)[0]  # (N_active,)
        active_labels = labels[active_indices]  # (N_active,)

        # ---------------------------------------------------------------------
        # 3a) Sort active REAL points by label so we can do segmented reductions.
        # ---------------------------------------------------------------------
        sort_order_real = xp.argsort(active_labels)  # (N_active,)
        sorted_indices_real = active_indices[sort_order_real]
        sorted_labels_real = active_labels[sort_order_real]
        sorted_X_real = X[sorted_indices_real]  # (N_active, d)

        # Compute per-cluster segmentation for REAL points (within active set).
        unique_cluster_ids, starts_real, counts_real = xp.unique(  # type: ignore
            sorted_labels_real, return_index=True, return_counts=True
        )
        K_batch = int(unique_cluster_ids.shape[0])

        # ---------------------------------------------------------------------
        # 3b) Augmentation: Pad clusters smaller than N_MIN_PER_CLUSTER
        #     with synthetic convex-combination points.
        # ---------------------------------------------------------------------
        X_synth, ids_synth = generate_synthetic_points_batched(
            sorted_X_real, starts_real, counts_real, MIN_PTS_PER_CLUSTER
        )

        # ---------------------------------------------------------------------
        # 3c) Build a "combined" batch = (sorted real points) + (synthetic points)
        #     and re-sort by label so cluster segments remain contiguous.
        #     Track type via I_batch:
        #         I_batch[i] >= 0  => global index of REAL point
        #         I_batch[i] == -1 => synthetic point
        # ---------------------------------------------------------------------
        if X_synth is not None:
            # Global ids for synth points are -1.
            global_ids_synth = xp.full(
                X_synth.shape[0], -1, dtype=sorted_indices_real.dtype
            )

            # Map ids_synth back to original cluster labels.
            labels_synth = unique_cluster_ids[ids_synth]

            # Concatenate real + synth, then sort by label to restore segmented layout.
            X_combined = xp.vstack([sorted_X_real, X_synth])
            L_combined = xp.concatenate([sorted_labels_real, labels_synth])
            I_combined = xp.concatenate([sorted_indices_real, global_ids_synth])

            # Sort the combined data.
            sort_order_comb = xp.argsort(L_combined)
            X_batch = X_combined[sort_order_comb]
            L_batch = L_combined[sort_order_comb]
            I_batch = I_combined[sort_order_comb]

            # Recompute segmentation on the combined batch (starts/counts change).
            _, starts_batch, counts_batch = xp.unique(  # type: ignore
                L_batch, return_index=True, return_counts=True
            )
            is_real_pt = I_batch >= 0
            real_counts_batch = xp.add.reduceat(
                is_real_pt.astype(xp.int64), starts_batch
            )  # (K_batch,)

            # Build F_batch/G_batch aligned with X_batch.
            # Real points are gathered from precomputed F_all/G_all; synthetic are evaluated.
            F_real = F_all[sorted_indices_real]
            G_real = G_all[sorted_indices_real]
            F_synth = f_fn(X_synth).ravel()
            G_synth = g_fn(X_synth)[:, 0, :]

            F_comb = xp.concatenate([F_real, F_synth])  # (N_real + N_synth,)
            G_comb = xp.vstack([G_real, G_synth])  # (N_real + N_synth, d)

            F_batch = F_comb[sort_order_comb]
            G_batch = G_comb[sort_order_comb]

        else:
            # No augmentation needed, batch is just the sorted real points.
            X_batch = sorted_X_real
            L_batch = sorted_labels_real
            I_batch = sorted_indices_real
            starts_batch = starts_real
            counts_batch = counts_real
            real_counts_batch = counts_real
            F_batch = F_all[sorted_indices_real]
            G_batch = G_all[sorted_indices_real]

        # Update global scaling for relative errors.
        F_inf = float(xp.max(xp.abs(F_batch))) + 1e-30

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Batch Preparation")

        # ---------------------------------------------------------------------
        # 3d) Batched first-order model and error computation:
        #     - compute centroids μ_k, f(μ_k), g(μ_k) for each cluster segment.
        # ---------------------------------------------------------------------
        sums = xp.add.reduceat(X_batch, starts_batch)  # (K_batch, d)
        mus = sums / counts_batch[:, None]  # (K_batch, d)

        f_mus = f_fn(mus)[:, 0]  # (K_batch,)
        g_mus = g_fn(mus)[:, 0, :]  # (K_batch, d)

        # Map each point in the batch to a local cluster index k in [0, K_batch).
        map_n_to_k = repeats(xp.arange(K_batch), counts_batch)

        # Center points around their cluster centroids.
        DeltaX = X_batch - mus[map_n_to_k]  # (N_batch, d)

        # First-order Taylor prediction around μ_k for each point.
        f_linear = f_mus[map_n_to_k] + xp.sum(
            DeltaX * g_mus[map_n_to_k], axis=1
        )  # (N_batch,)

        # Pointwise absolute linearization error.
        err_point = xp.abs(F_batch - f_linear)  # (N_batch,)

        # Cluster-wise max error
        E_abs = xp.zeros(K_batch, dtype=err_point.dtype)  # (K_batch,)
        if xp.__name__ == "array_api_compat.numpy":
            xp.maximum.at(E_abs, map_n_to_k, err_point)
        else:
            cupy_scatter_max(E_abs, map_n_to_k, err_point)

        # Relative error scaled by max |f| in the batch.
        E_rel = E_abs / F_inf

        # Cluster accept/reject criterion (good/bad)
        is_good = (E_abs < ATOL) | (E_rel < RTOL)  # (K_batch,)

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Error Computation")

        # ---------------------------------------------------------------------
        # 3e) Finalize acceptable clusters:
        #     - compute Q_k trust ellipsoids from centered points (using combined data)
        #     - append to output buffers
        #     - deactivate only REAL points belonging to good clusters.
        # ---------------------------------------------------------------------
        good_indices = xp.nonzero(is_good)[0]  # (K_good,)
        if good_indices.size > 0:
            cluster_good_mask = is_good
            point_good_mask = cluster_good_mask[map_n_to_k]  # (N_batch,)

            # Use centered points from accepted clusters for ellipsoid estimation.
            good_DeltaX = DeltaX[point_good_mask]

            # Rebuild per-good-cluster segmentation (contiguous because batch is label-sorted).
            good_counts_total = counts_batch[good_indices]
            good_counts_real = real_counts_batch[good_indices]
            good_starts = xp.r_[0, xp.cumsum(good_counts_total)[:-1]]

            Qs = batched_calculate_Q(
                good_DeltaX, good_starts, good_counts_total, int(good_indices.size)
            )

            # Ensure output buffers can hold the newly accepted clusters.
            n_good = int(good_indices.size)
            end_out = valid_cluster_count + n_good
            buffers, capacity = adjust_buffer_capacity(
                buffers, valid_cluster_count, capacity, end_out
            )

            # Commit cluster metadata to outputs.
            buffers["centroids"][valid_cluster_count:end_out] = mus[good_indices]
            buffers["f"][valid_cluster_count:end_out] = f_mus[good_indices]
            buffers["g"][valid_cluster_count:end_out] = g_mus[good_indices]
            buffers["Q"][valid_cluster_count:end_out] = Qs
            buffers["E_abs"][valid_cluster_count:end_out] = E_abs[good_indices]
            buffers["E_rel"][valid_cluster_count:end_out] = E_rel[good_indices]
            buffers["real_counts"][valid_cluster_count:end_out] = good_counts_real
            buffers["counts"][valid_cluster_count:end_out] = good_counts_total
            valid_cluster_count = int(end_out)

            # Deactivate only REAL points (synthetic points have index -1).
            point_indices_in_batch = I_batch[point_good_mask]
            real_point_mask = point_indices_in_batch >= 0
            global_indices_good = point_indices_in_batch[real_point_mask]
            active_mask[global_indices_good] = False

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Finalize Good Clusters")
        # ---------------------------------------------------------------------
        # 3f) Split unacceptable clusters (bad clusters):
        #     Estimate symmetric Hessian H_k via (Δx, Δg), compute M_k = |H_k|^{1/2},
        #     transform points z = M_k Δx, and split by median along dominant PCA
        #     direction of Cov(z).
        # ---------------------------------------------------------------------
        bad_indices = xp.nonzero(~is_good)[0]  # (K_bad,)
        if bad_indices.size == 0:
            # No more bad clusters to split; we're done.
            break

        cluster_bad_mask = ~is_good
        point_bad_mask = cluster_bad_mask[map_n_to_k]  # (N_batch,)
        bad_DeltaX = DeltaX[point_bad_mask]  # (N_bad, d)

        # Build centered gradient differences Δg = g(x) - g(μ_k) for Hessian fitting.
        bad_G_points = G_batch[point_bad_mask]  # (N_bad, d)
        bad_g_mus = g_mus[map_n_to_k][point_bad_mask]  # (N_bad, d)
        bad_DeltaG = bad_G_points - bad_g_mus  # (N_bad, d)

        # Rebuild segmentation for the bad subset (still contiguous due to global sorting).
        bad_counts = counts_batch[bad_indices]
        bad_starts = xp.r_[0, xp.cumsum(bad_counts)[:-1]]
        K_bad = int(bad_indices.size)

        # Hessian estimation and absolute-square-root transform for each bad cluster.
        H = batched_estimate_hessian_symmetric(
            bad_DeltaX, bad_DeltaG, bad_starts
        )  # (K_bad, d, d)
        M = batched_sqrt_abs_psd(H)  # (K_bad, d, d)

        # Expand M_k to each point in the bad subset, then compute z = M_k Δx.
        map_bad_n_to_k = repeats(xp.arange(K_bad), bad_counts)  # (N_bad,)
        M_expanded = M[map_bad_n_to_k]  # (N_bad, d, d)
        Z = (M_expanded @ bad_DeltaX[:, :, None])[:, :, 0]  # (N_bad, d)

        # Cluster-wise covariance of Z for PCA split direction.
        ZZ_all = Z[:, :, None] * Z[:, None, :]  # (N_bad, d, d)
        Cov_Z = xp.add.reduceat(ZZ_all, bad_starts)  # (K_bad, d, d)

        # Dominant eigenvector per cluster gives the split direction in Z-space.
        v = batched_top_eigvec(Cov_Z)  # (K_bad, d)
        v_expanded = v[map_bad_n_to_k]  # (N_bad, d)

        # Scalar projection scores and within-cluster median split.
        scores = xp.sum(Z * v_expanded, axis=1)  # (N_bad,)
        split_decision = batched_median_split_decision(
            scores, map_bad_n_to_k, bad_starts, bad_counts
        )  # (N_bad,)

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Bad Cluster Splitting")

        # ---------------------------------------------------------------------
        # 3g) Update labels for REAL points only:
        #     - align decisions back to global indices via I_batch
        #     - new label = old*2 + {0,1} so the tree structure is preserved.
        # ---------------------------------------------------------------------
        indices_in_bad_subset = I_batch[point_bad_mask]  # (N_bad,)
        is_real_in_bad = indices_in_bad_subset >= 0
        global_indices_bad_real = indices_in_bad_subset[is_real_in_bad]

        decisions_real = split_decision[is_real_in_bad]
        old_ids = labels[global_indices_bad_real]
        new_ids = old_ids * 2 + decisions_real.astype(xp.int32)
        labels[global_indices_bad_real] = new_ids

        # Bookkeeping / debug logs.
        level += 1
        n_synth_count = X_synth.shape[0] if X_synth is not None else 0
        logger.debug(
            f"Level {level}: Active {n_active:,} (+{n_synth_count:,} synth) -> "
            f"{K_batch:,} clusters (Good: {good_indices.size:,}, Bad: {bad_indices.size:,})"
        )
        logger.debug(
            f"Runtime - Level {level}: {time.perf_counter() - _start_loop:.4f}s"
        )

    # -------------------------------------------------------------------------
    # 4) Truncate outputs to the number of finalized clusters and return.
    # -------------------------------------------------------------------------
    final_cluster_count = int(valid_cluster_count)
    logger.debug(
        f"Adaptive Clustering Complete: {final_cluster_count:,} final clusters."
    )
    logger.debug(f"Runtime - Total: {time.perf_counter() - _start_init:.4f}s")
    return {k: buffers[k][:final_cluster_count] for k in buffers}


if __name__ == "__main__":
    from src.logging_config import setup_logging
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging(console_level=logging.DEBUG)
    xp.random.seed(0)
    _N = 100_000
    d = 2
    print(f"Generating {_N:,} points (d={d})...")
    PROB_LIMS = (-1.0, 3)
    X = xp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(_N, d))

    TOL = 1e-3
    NUM_INITIAL_CLUSTERS = 100
    MIN_PTS_PER_CLUSTER = 1000

    print("Starting Adaptive GPU Clustering...")
    results = adaptive_clustering(
        X,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        ATOL=TOL,
        RTOL=TOL,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        MAX_CLUSTERS_EST=100_000,
    )

    print("-" * 40)
    print(f"Final Cluster Count: {results['centroids'].shape[0]:,}")

    E_abs_vals = results["E_abs"]
    E_rel_vals = results["E_rel"]
    counts = results["counts"]
    real_counts = results["real_counts"]
    print(
        "E_abs stats: min/mean/max =",
        xp.round(E_abs_vals.min(), 3),
        xp.round(E_abs_vals.mean(), 3),
        xp.round(E_abs_vals.max(), 3),
    )
    print(
        "E_rel stats: min/mean/max =",
        xp.round(E_rel_vals.min(), 6),
        xp.round(E_rel_vals.mean(), 6),
        xp.round(E_rel_vals.max(), 6),
    )
    print(
        "Counts stats: min/mean/max =",
        int(counts.min()),
        f"{counts.mean():.1f}",
        int(counts.max()),
    )
    print(
        "Real Counts stats: min/mean/max =",
        int(real_counts.min()),
        f"{real_counts.mean():.1f}",
        int(real_counts.max()),
    )

    if d == 2 and len(results["centroids"]) <= 3000:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Ellipse

        if xp.__name__ == "array_api_compat.cupy":
            results = {k: xp.asnumpy(v) for k, v in results.items()}

        # 1 plot 2 subplots (1X2)
        # Subplot 1: scatter of final cluster centroids on contourf
        # Subplot 2: scatter of final cluster centroids with trust ellipses on contourf

        x = np.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        y = np.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        X_grid, Y_grid = np.meshgrid(x, y)
        Z_grid = rosenbrock(np.vstack([X_grid.ravel(), Y_grid.ravel()]).T).reshape(
            X_grid.shape
        )

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        cs = axs[0].contourf(
            X_grid,
            Y_grid,
            Z_grid,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(cs, ax=axs[0])
        axs[0].set_title("Rosenbrock Function Contour")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].set_aspect("equal")

        # Subplot 2: with ellipses
        cs2 = axs[1].contourf(
            X_grid,
            Y_grid,
            Z_grid,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(cs2, ax=axs[1])
        axs[1].set_title("Adaptive Clustering with Trust Ellipses")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].set_aspect("equal")
        axs[1].scatter(
            results["centroids"][:, 0],
            results["centroids"][:, 1],
            c="red",
            s=10,
            label="Centroids",
        )

        # Draw ellipses
        level = 1.0
        for i in range(results["centroids"].shape[0]):
            mu = results["centroids"][i]
            Q = results["Q"][i]

            # Eigen-decomposition of Q
            w, V = np.linalg.eigh(Q)
            w = np.maximum(w, 1e-12)

            order = np.argsort(w)[::-1]
            w = w[order]
            V = V[:, order]

            a = np.sqrt(level / w[0])
            b = np.sqrt(level / w[1])

            angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))

            ellipse = Ellipse(
                xy=(mu[0], mu[1]),
                width=2 * a,
                height=2 * b,
                angle=angle,
                edgecolor="red",
                facecolor="none",
                lw=1,
                alpha=0.5,
            )
            axs[1].add_patch(ellipse)
        axs[1].legend()
        plt.tight_layout()
        fig.suptitle(
            f"Adaptive Clustering: Rosenbrock (TOL={TOL}, N_INIT={NUM_INITIAL_CLUSTERS}, N_MIN={MIN_PTS_PER_CLUSTER}), Final Clusters: {results['centroids'].shape[0]:,}",
            fontsize=16,
        )

        plt.subplots_adjust(top=0.88)
        fname = f"adaptive_clustering_rosenbrock_tol{int(TOL*1e4)}_init{NUM_INITIAL_CLUSTERS}_min{MIN_PTS_PER_CLUSTER}.png"
        plt.savefig(fname, dpi=300)
        print(f"Saved figure to {fname}")
    else:
        print("Skipping plot (d != 2 or too many clusters).")
