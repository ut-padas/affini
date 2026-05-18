import logging
import time
from typing import Any, Callable, Literal, Optional

import matplotlib.pyplot as plt
from cuml.cluster import KMeans as cuKMeans
from cupyx import scatter_max as cupy_scatter_max
from knndbscan import knn_dbscan
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans

from src import IS_CUPY_BACKEND, Array, xp
from src.logging_utils import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)

TINY_EPS = 1e-12


def scatter_maximum_1d(out: Array, idx: Array, vals: Array) -> None:
    """
    1D segmented max scatter: ``out[idx[i]] = max(out[idx[i]], vals[i])``.
    Uses `cupyx.scatter_max` on GPU and `numpy.maximum.at` semantics on CPU.
    Eg:
        out = [0, 0, 0]
        idx = [0, 1, 0, 2, 1]
        vals = [1, 2, 3, 4, 0]
        scatter_maximum_1d(out, idx, vals)
        # out is now [3, 2, 4]

    Args:
        out: (K,) output array updated in-place.
        idx: (N,) integer indices in [0, K).
        vals: (N,) values to reduce via max per index.
    """
    if IS_CUPY_BACKEND:
        cupy_scatter_max(out, idx, vals)
    else:
        xp.maximum.at(out, idx, vals)


def compute_label_segments(sorted_labels: Array) -> tuple[Array, Array, Array]:
    """Compute segmentation metadata for a label-sorted 1D label array.

    Args:
        sorted_labels: (N,) labels sorted in nondecreasing order.

    Returns:
        uniq: (K,) unique label values (ascending).
        starts: (K,) start indices of each label-run in `sorted_labels`.
        counts: (K,) run lengths.
    """
    uniq, starts, counts = xp.unique(
        sorted_labels, return_index=True, return_counts=True
    )
    return uniq, starts, counts


def generate_sythetic_points(
    X_sorted: Array,
    starts: Array,
    counts: Array,
    uniq_labels: Array,
    MIN_PTS_PER_CLUSTER: int,
    *,
    SEED: int,
    level: int,
) -> tuple[Optional[Array], Optional[Array]]:
    """
    Batch generate synthetic point generation via convex combinations within clusters.

    For each cluster whose active count is below `MIN_PTS_PER_CLUSTER`, generate the
    required number of synthetic points via convex combinations of *two distinct*
    points from the same cluster:

        x_synth = a * x_i + (1 - a) * x_j,    a ~ Uniform(0, 1)

    This stabilizes covariance / Hessian-estimation steps in later geometry, while the
    pool design makes synthetic points persist across levels.

    Args:
        X_sorted: (N_active, d) active points sorted by label.
        starts: (K,) cluster segment starts within `X_sorted`.
        counts: (K,) active counts per cluster.
        uniq_labels: (K,) label values for each segment.
        MIN_PTS_PER_CLUSTER: minimum desired points per cluster.
        SEED: base RNG seed.
        level: current refinement level; used to derive deterministic RNG stream.

    Returns:
        X_synth: (M, d) synthetic points, or None if no padding needed.
        labels_synth: (M,) *global* label values associated to each synthetic point,
            or None if no padding needed.
    """
    # If any cluster has <3 points, convex-combo padding is ill-defined (degenerate).
    if xp.any(counts < 3):
        bad = uniq_labels[xp.nonzero(counts < 3)[0][:10]]
        raise RuntimeError(
            f"Level {level}: cluster(s) with <3 points detected. "
            f"Example labels: {bad.tolist()} (showing up to 10)."
        )

    # How many synthetic points we need per cluster.
    need = xp.maximum(0, MIN_PTS_PER_CLUSTER - counts)  # (K,)
    total_synth = int(xp.sum(need))
    if total_synth == 0:
        return None, None

    rng = xp.random.default_rng(SEED + level)
    rand = rng.random

    # Map each synthetic sample to a local segment id k in [0..K-1].
    k_synth = repeats(xp.arange(int(need.shape[0])), need)  # (M,)
    labels_synth = uniq_labels[k_synth]  # (M,)

    # For each synth sample, choose two DISTINCT points inside its cluster segment.
    m = counts[k_synth]  # (M,)
    s0 = starts[k_synth]  # (M,)

    # i ~ Uniform{0,...,m-1}
    i = (xp.floor(rand(total_synth) * m)).astype(xp.int64)
    # j ~ Uniform{0,...,m-2}, then shift to skip i
    j = (xp.floor(rand(total_synth) * (m - 1))).astype(xp.int64)
    j = j + (j >= i)

    idx_i = s0 + i
    idx_j = s0 + j

    a = rand((total_synth, 1)).astype(X_sorted.dtype)  # (M,1)
    X_synth = a * X_sorted[idx_i] + (1.0 - a) * X_sorted[idx_j]  # (M,d)
    return X_synth, labels_synth.astype(xp.int64)


def calculate_scalar_metrics(
    DeltaX: Array,
    G_points: Array,
    G_centroids: Array,
    map_n_to_k: Array,
    K: int,
) -> tuple[Array, Array]:
    """
    Computes scalar trust radius and observed Lipschitz constant per cluster.

    Args:
        DeltaX: (N, d) centered points.
        G_points: (N, fd, d) gradients at points.
        G_centroids: (K, fd, d) gradients at centroids.
        map_n_to_k: (N,) mapping.

    Returns:
        radii: (K,) max Euclidean distance from centroid.
        L_obs: (K,) max observed curvature |dg|/|dx| across all output dimensions.
    """
    # 1. Radius: max(|dx|)
    dist_x = xp.linalg.norm(DeltaX, axis=1)  # (N,)
    radii = xp.zeros((K,), dtype=DeltaX.dtype)
    scatter_maximum_1d(radii, map_n_to_k, dist_x)

    # 2. Gradient Diff: |dg| per component.
    # Expand centroids: (N, fd, d)
    g_center_exp = G_centroids[map_n_to_k]
    diff_g_tensor = G_points - g_center_exp  # (N, fd, d)

    # Norm per component per point: (N, fd)
    diff_g_norms = xp.linalg.norm(diff_g_tensor, axis=2)

    # 3. Ratios per component: (N, fd)
    valid_mask = dist_x > TINY_EPS
    ratios = xp.zeros_like(diff_g_norms)
    if xp.any(valid_mask):
        ratios[valid_mask] = diff_g_norms[valid_mask] / dist_x[valid_mask][:, None]

    # 4. L_obs: Max over points AND components
    # First max over components for each point
    max_ratio_per_point = xp.max(ratios, axis=1)  # (N,)

    L_obs = xp.zeros((K,), dtype=DeltaX.dtype)
    scatter_maximum_1d(L_obs, map_n_to_k, max_ratio_per_point)

    return radii, L_obs


def calculate_Q(
    DeltaX: Array,
    starts: Array,
    counts: Array,
    K: int,
    *,
    Q_SHRINK: float = 0.0,
) -> tuple[Array, Array]:
    """
    PCA-aligned trust ellipsoid $Q_k$ using max-abs projections, with covariance shrinkage.

    This constructs, for each cluster k, a PSD matrix Q_k defining an ellipsoid:
        { y : y^T Q_k y <= 1 }
    aligned with PCA directions of the cluster, with axis lengths chosen by maximum
    absolute projection of centered points onto those axes.

    Args:
        DeltaX: (N, d) centered points for a *packed* batch (cluster segments contiguous).
        starts: (K,) segment starts within `DeltaX`.
        counts: (K,) points per segment.
        K: number of segments/clusters.
        Q_SHRINK: covariance shrinkage factor in [0,1]. When >0:
            Cov <- (1-a) Cov + a * (tr(Cov)/d) I.

    Returns:
        Q: (K, d, d) PSD ellipsoid matrices.
    """
    # Cov_k = sum_i (dx_i dx_i^T) for points belonging to cluster k.
    XX_all = DeltaX[:, :, None] * DeltaX[:, None, :]  # (N,d,d)
    Cov = xp.add.reduceat(XX_all, starts)  # (K,d,d)
    Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

    if Q_SHRINK > 0.0:
        # Sigma = (1-a)Cov + a*(tr(Cov)/d)*I, per cluster
        d = int(Cov.shape[1])
        tr = xp.trace(Cov, axis1=1, axis2=2)  # (K,)
        scale = tr / max(d, 1)
        I = xp.eye(d, dtype=Cov.dtype)[None, :, :]  # (1,d,d)
        Cov = (1.0 - Q_SHRINK) * Cov + Q_SHRINK * (scale[:, None, None] * I)
        Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

        # PCA directions U (columns).
        Lambda, U = xp.linalg.eigh(Cov)

        # Invert Eigenvalues to get shape.
        # Since Cov is PSD and shrunk, Lambda > 0.
        Lambda_inv = 1.0 / xp.maximum(Lambda, TINY_EPS)  # (K,d)

        # dist^2 = x^T S x = x^T (U Lambda_inv U^T) x
        map_n_to_k = repeats(xp.arange(K), counts)  # (N,)
        U_exp = U[map_n_to_k]  # (N, d, d)
        L_inv_exp = Lambda_inv[map_n_to_k]  # (N, d)

        # Projection: p = U^T x
        proj = (U_exp.transpose(0, 2, 1) @ DeltaX[:, :, None])[:, :, 0]  # (N,d)

        # Distance squared relative to eigen-shape
        dist_sq = xp.sum((proj * proj) * L_inv_exp, axis=1)  # (N,)

        # Find max distance per cluster to scale the ellipsoid
        # scale = 1.0 / max(dist_sq)
        max_dist_sq = xp.zeros((K,), dtype=proj.dtype)
        scatter_maximum_1d(max_dist_sq, map_n_to_k, dist_sq)

        # Avoid degenerate axes
        max_dist_sq = xp.maximum(max_dist_sq, TINY_EPS)

        scale = 1.0 / max_dist_sq  # (K,)

        # Q = scale * (U Lambda_inv U^T)
        # Combine scale into Lambda_inv
        final_D = scale[:, None] * Lambda_inv  # (K, d)

        # Q = U diag(final_D) U^T
        Q = (U * final_D[:, None, :]) @ U.transpose(0, 2, 1)  # (K,d,d)
    else:
        # PCA directions U (columns).
        _, U = xp.linalg.eigh(Cov)

        # Projection scores onto PCA axes: proj_i = U_k^T dx_i.
        map_n_to_k = repeats(xp.arange(K), counts)  # (N,)
        U_exp = U[map_n_to_k]  # (N,d,d)
        proj = (U_exp.transpose(0, 2, 1) @ DeltaX[:, :, None])[:, :, 0]  # (N,d)
        proj_abs = xp.abs(proj)

        # Axis extents: d_axes[k,j] = max_i |proj[i,j]| within cluster k.
        d_axes = xp.zeros((K, proj_abs.shape[1]), dtype=proj_abs.dtype)
        scatter_maximum_1d(d_axes, map_n_to_k, proj_abs)

        d_axes = xp.maximum(d_axes, TINY_EPS)
        Dinv2 = 1.0 / (d_axes * d_axes)  # (K,d)

        # Q = U diag(Dinv2) U^T
        Q = (U * Dinv2[:, None, :]) @ U.transpose(0, 2, 1)

    return 0.5 * (Q + Q.transpose(0, 2, 1)), Cov


def compute_good_covariance_hessian(
    DeltaX: Array,
    Fvals_linear: Array,
    Fvals_exact: Array,
    starts: Array,
) -> Array:
    """
    Estimates the Hessian using the covariance formulation.
    It can be shown that by taking level curves of the linear error,
    the Hessian can be approximated by the inverse of the covariance matrix of these points.
    We make sure each level curve has enough points to make sure matrix is well-conditioned.
    """
    # len(starts) is the number of ellipses we are computing for
    points_in_level_curve = xp.full(len(DeltaX), False)
    Ndx, d = DeltaX.shape

    # Compute errors
    errors = Fvals_linear - Fvals_exact

    # Pick points from each such that we have floor(2d*log(d)) points in each level curve
    # I am using a for loop because I don't know what I am doing otherwise (need to vectorize)
    for j in range(len(starts)):
        eps = 1e-3
        errors_current = errors[starts[j] : starts[j + 1]]
        within_tol = errors_current < eps
        if len(within_tol) <= 2 * d * xp.log(d) + 1:
            within_tol = errors_current > eps
        else:
            while xp.sum(within_tol) < xp.floor(2 * d * xp.log(d)):
                eps = eps * 2
                within_tol = errors_current < eps

        points_in_level_curve[starts[j] : starts[j + 1]] = within_tol.flatten()

    # error_blocks = np.split(errors, starts[1:])
    # sorted_error_blocks = [np.sort(b) for b in error_blocks]
    # Compute the covariances
    DeltaX_within_tol = DeltaX[points_in_level_curve]
    XX_all = DeltaX_within_tol[:, :, None] * DeltaX_within_tol[:, None, :]
    C = xp.add.reduceat(XX_all, starts)

    # Take the absolute value, inverse square root (use eigendecomposition)
    dvals, U = xp.linalg.eigh(C)
    abs_dvals = np.abs(dvals)
    inv_abs_sqrt_dvals = abs_dvals ** (-0.5)
    M = (U * inv_abs_sqrt_dvals[:, None, :]) @ U.transpose(0, 2, 1)
    return 0.5 * (M + M.transpose(0, 2, 1))


def estimate_hessian_sqrt_symmetric(
    DeltaX: Array,
    DeltaG: Array,
    starts: Array,
) -> Array:
    """
    Scalar Symmetric Hessian estimation.
    For vector-valued adaptation, the caller must provide DeltaG corresponding
    to the specific *active component* for each cluster segment.

    Args:
        DeltaX: (N, d) centered points.
        DeltaG: (N, d) centered gradients (of the active component).
        starts: (K,) segment starts.

    Returns:
        M: (K, d, d) |H|^1/2.
    """
    XX_all = DeltaX[:, :, None] * DeltaX[:, None, :]
    XG_all = DeltaX[:, :, None] * DeltaG[:, None, :]

    C = xp.add.reduceat(XX_all, starts)
    E = xp.add.reduceat(XG_all, starts)

    C = 0.5 * (C + C.transpose(0, 2, 1))
    S = E + E.transpose(0, 2, 1)

    dvals, U = xp.linalg.eigh(C)
    dvals = xp.maximum(dvals, TINY_EPS)

    Gtilde = U.transpose(0, 2, 1) @ S @ U
    denom = dvals[:, :, None] + dvals[:, None, :]
    F = Gtilde / denom

    h_eigvals, V = xp.linalg.eigh(F)
    Z = U @ V
    w_abs_sqrt = xp.sqrt(xp.abs(h_eigvals))

    M = (Z * w_abs_sqrt[:, None, :]) @ Z.transpose(0, 2, 1)
    return 0.5 * (M + M.transpose(0, 2, 1))


def get_dominant_eigenvector(Cov: Array) -> Array:
    """Dominant eigenvector of symmetric covariance matrices.

    Args:
        Cov: (K, d, d) symmetric matrices.

    Returns:
        v: (K, d) top eigenvector per matrix.
    """
    # w asc => last is largest
    w, V = xp.linalg.eigh(Cov)
    return V[:, :, -1]


def compute_median_split_decision(
    scores: Array,
    cluster_id: Array,
    starts: Array,
    counts: Array,
) -> Array:
    """
    Compute per-cluster median split decisions for a packed (clustered) score vector.

    Given per-point scores `scores` and a per-point cluster id `cluster_id` (local ids
    like 0..K-1), return a boolean decision:
        decision[i] = scores[i] >= median(scores within that cluster).

    Implementation note:
        This uses a two-stage argsort (by score, then by cluster id) to obtain scores
        sorted *within clusters* without explicit Python loops.
    """
    N = int(scores.shape[0])
    if N == 0:
        return scores >= 0

    idx_by_score = xp.argsort(scores)
    cid_by_score = cluster_id[idx_by_score]

    idx_group = xp.argsort(cid_by_score)
    order = idx_by_score[idx_group]

    scores_sorted = scores[order]

    i0 = starts + (counts - 1) // 2
    i1 = starts + counts // 2
    median = 0.5 * (scores_sorted[i0] + scores_sorted[i1])  # (K,)

    return scores >= median[cluster_id]


def allocate_output_buffers(
    MAX_CLUSTERS_EST: int,
    d: int,
    fd: int,  # Function output dimension
    dtype: Any,
    STORE_HESSIAN_SQRT_PER_CLUSTER: bool,
    STORE_TREE_DATA: bool,
) -> tuple[dict[str, Array], int]:
    """Allocate output buffers for finalized clusters.

    Args:
        MAX_CLUSTERS_EST: initial capacity estimate for finalized clusters.
        d: dimensionality of the input space.
        dtype: floating dtype for outputs.

    Returns:
        buffers: dict of arrays sized to `cap`.
        cap: allocated capacity (number of clusters).

    Notes:
        This matches the contract of `adaptive_clustering` return value.
    """
    cap = max(1, int(MAX_CLUSTERS_EST))
    buffers = {
        "centroids": xp.empty((cap, d), dtype=dtype),
        "f": xp.empty((cap, fd), dtype=dtype),  # Vector valued
        "g": xp.empty((cap, fd, d), dtype=dtype),  # Tensor valued
        "radii": xp.empty((cap,), dtype=dtype),
        "L_obs": xp.empty((cap,), dtype=dtype),
        "E_cluster": xp.empty((cap,), dtype=dtype),
        "counts": xp.empty((cap,), dtype=xp.int64),
        "Q": xp.empty((cap, d, d), dtype=dtype),
    }
    if STORE_HESSIAN_SQRT_PER_CLUSTER:
        buffers["H_sqrt"] = xp.empty((cap, d, d), dtype=dtype)
    if STORE_TREE_DATA:
        buffers["tree_data"] = {}
    return buffers, cap


def adjust_buffer_capacity(
    buffers: dict[str, Array],
    current_count: int,
    current_cap: int,
    required_total: int,
    STORE_HESSIAN_SQRT_PER_CLUSTER: bool,
    STORE_TREE_DATA: bool,
) -> tuple[dict[str, Array], int]:
    """Grow output buffers if the finalized cluster count exceeds current capacity.

    Args:
        buffers: current output buffers.
        current_count: number of clusters currently written.
        current_cap: current capacity.
        required_total: required capacity after the next write.

    Returns:
        new_buffers: potentially reallocated buffers.
        new_cap: updated capacity.
    """
    if required_total <= current_cap:
        return buffers, current_cap

    d = int(buffers["centroids"].shape[1])
    fd = int(buffers["f"].shape[1])
    new_cap = max(required_total, max(1, current_cap) * 2)
    dtype = buffers["centroids"].dtype

    logger.warning(f"Resizing output buffers: {current_cap:,} -> {new_cap:,} clusters")

    new_buffers = {
        "centroids": xp.empty((new_cap, d), dtype=dtype),
        "f": xp.empty((new_cap, fd), dtype=dtype),
        "g": xp.empty((new_cap, fd, d), dtype=dtype),
        "radii": xp.empty((new_cap,), dtype=dtype),
        "L_obs": xp.empty((new_cap,), dtype=dtype),
        "E_cluster": xp.empty((new_cap,), dtype=dtype),
        "counts": xp.empty((new_cap,), dtype=xp.int64),
        "Q": xp.empty((new_cap, d, d), dtype=dtype),
    }
    if current_count > 0:
        for k in new_buffers:
            new_buffers[k][:current_count] = buffers[k][:current_count]

    if STORE_HESSIAN_SQRT_PER_CLUSTER:
        new_buffers["H_sqrt"] = xp.empty((new_cap, d, d), dtype=dtype)
        if current_count > 0:
            new_buffers["H_sqrt"][:current_count] = buffers["H_sqrt"][:current_count]

    if STORE_TREE_DATA:
        new_buffers["tree_data"] = {}
        if "tree_data" in buffers and buffers["tree_data"]:
            for key, value in buffers["tree_data"].items():
                new_buffers["tree_data"][key] = value

    # Delete old buffers to free memory.
    del buffers
    return new_buffers, int(new_cap)


def adaptive_clustering(
    X: Array,
    *,
    f_fn: Optional[Callable[[Array], Array]] = None,
    g_fn: Optional[Callable[[Array], Array]] = None,
    oracle_fn: Optional[Callable[[Array], tuple[Array, Array]]] = None,
    TOL: float,
    ERROR_METRIC: Literal["L2", "Linf-global", "Linf-local"] = "Linf-global",
    INITIAL_CLUSTERING_METHOD: Literal["kmeans", "knndbscan"],
    NUM_INITIAL_CLUSTERS: int = 100,
    MIN_PTS_PER_CLUSTER: int = 100,
    MAX_CLUSTERS_EST: int = 100_000,
    Q_SHRINK: float = 0.05,
    SEED: int = 0,
    LOG_MEM_STATS: bool = False,
    # Pool sizing
    SYNTH_BUDGET: Optional[int] = None,
    SYNTH_BUDGET_FACTOR: float = 50.0,
    MAX_LEVELS: int = 50,
    # Hessian informed splitting
    USE_HESSIAN_INFORMED_SPLIT=True,
    USE_COVARIANCE_HESSIAN=False,
    # Testing flags
    IGNORE_GRADIENT: bool = False,
    STORE_HESSIAN_SQRT_PER_CLUSTER: bool = False,
    STORE_TREE_DATA: bool = False,
) -> dict[str, Array]:
    """
    Adaptive clustering for Vector-Valued functions F: R^d -> R^m.
    Supports component-wise relative error metrics and Hessian-informed splitting
    via Worst-Case Active Component selection.
    """
    # Validate function arguments
    if oracle_fn is not None:
        if f_fn is not None or g_fn is not None:
            raise ValueError(
                "Cannot specify both oracle_fn and f_fn/g_fn. Use either oracle_fn OR both f_fn and g_fn."
            )
    else:
        if f_fn is None or g_fn is None:
            raise ValueError("Must specify either oracle_fn OR both f_fn and g_fn.")

    _start_time = time.perf_counter()

    total_f_evals = 0

    X0 = xp.asarray(X)  # (N0, d)
    N0, d = X0.shape

    if SYNTH_BUDGET is None:
        SYNTH_BUDGET = int(max(0, SYNTH_BUDGET_FACTOR * N0))
    capacity = int(N0 + SYNTH_BUDGET)
    if LOG_MEM_STATS:
        log_mem_stats("Start: Adaptive Clustering (pool alloc)")

    # ---- 1. Initialize Pool ----
    X_pool = xp.empty((capacity, d), dtype=X0.dtype)
    X_pool[:N0] = X0
    n_total = int(N0)

    # Eval initial F/G to determine m (output dimension)
    _t0 = time.perf_counter()
    if oracle_fn is not None:
        F0_val, G0_val = oracle_fn(X_pool[:N0])
        total_f_evals += N0
    else:
        F0_val = f_fn(X_pool[:N0])  # type: ignore
        total_f_evals += N0
        if IGNORE_GRADIENT:
            G0_val = 0.0
        else:
            G0_val = g_fn(X_pool[:N0])  # type: ignore

    fd = F0_val.shape[1]  # Output dimension (fd)

    F_pool = xp.empty((capacity, fd), dtype=X0.dtype)
    G_pool = xp.empty((capacity, fd, d), dtype=X0.dtype)

    F_pool[:N0] = F0_val
    G_pool[:N0] = G0_val

    logger.info(f"Runtime - Initial f/g: {time.perf_counter() - _t0:.3f}s")
    logger.info(f"Function: Input d={d}, Output fd={fd}")

    # Global scale per component for Linf-global
    # S_j = max_{x \in \Omega} |f_j(x)|
    F_max_global = xp.max(xp.abs(F_pool[:N0]), axis=0)  # (fd,)
    synth_added_prev_level = False

    # ---- 2. Initial Clustering ----
    if INITIAL_CLUSTERING_METHOD == "kmeans":
        _t1 = time.perf_counter()
        if IS_CUPY_BACKEND:
            km = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
            labels0 = km.fit_predict(X_pool[:N0])
        else:
            km = KMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
            labels0 = km.fit_predict(xp.asarray(X_pool[:N0]))

        labels_pool = xp.empty((capacity,), dtype=xp.int64)
        labels_pool[:N0] = labels0.astype(xp.int64)
        logger.info(f"Runtime - Initial KMeans: {time.perf_counter() - _t1:.3f}s")
    elif INITIAL_CLUSTERING_METHOD == "knndbscan":
        _device = "gpu" if IS_CUPY_BACKEND else "cpu"
        _t1 = time.perf_counter()
        labels0 = knn_dbscan(
            X_pool[:N0],
            eps=0.1,
            min_samples=MIN_PTS_PER_CLUSTER,
            n_jobs=-1,
            device=_device,
        )
        labels_pool = xp.empty((capacity,), dtype=xp.int64)
        labels_pool[:N0] = labels0.astype(xp.int64)

        # Includes noise label (-1) in count
        num_initial_clusters = int(xp.unique(labels_pool[:N0]).shape[0])
        logger.info(f"Runtime - Initial KNN-DBSCAN: {time.perf_counter() - _t1:.3f}s")
        logger.info(f"Initial KNN-DBSCAN found {num_initial_clusters} clusters.")
    else:
        raise ValueError(
            f"Invalid INITIAL_CLUSTERING_METHOD: {INITIAL_CLUSTERING_METHOD}. "
            f"Choose 'kmeans' or 'knndbscan'."
        )

    active = xp.zeros((capacity,), dtype=bool)
    active[:N0] = True

    # Outputs
    buffers, out_cap = allocate_output_buffers(
        MAX_CLUSTERS_EST,
        d,
        fd,
        X0.dtype,
        STORE_HESSIAN_SQRT_PER_CLUSTER,
        STORE_TREE_DATA,
    )
    out_count = 0

    # ---- 3. Main Loop ----
    level = 1
    while level <= MAX_LEVELS:
        t_level = time.perf_counter()
        if LOG_MEM_STATS:
            log_mem_stats(f"Start: L{level:02}")

        # Update Global Scale if synthetic points added previously
        if synth_added_prev_level:
            F_max_global = xp.max(xp.abs(F_pool[:n_total]), axis=0)

        # Basic Check
        if xp.all(F_max_global < TINY_EPS):
            logger.warning(f"Level {level}: F is near zero everywhere.")

        # Active indices within the currently-used pool prefix [0..n_total).
        idx_active = xp.nonzero(active[:n_total])[0]
        n_active = int(idx_active.shape[0])
        if n_active == 0:
            logger.info(f"Level {level:03}: no active points remain, terminating.")
            break

        synth_added_this_level = False

        # Sort and Segment
        label_active = labels_pool[idx_active]
        perm = xp.argsort(label_active)
        idx_sorted = idx_active[perm]
        label_sorted = label_active[perm]
        uniq_labels, starts, counts = compute_label_segments(label_sorted)
        K_batch = int(uniq_labels.shape[0])

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Setup")

        # --- Padding Step ---
        X_sorted = X_pool[idx_sorted]
        X_synth, labels_synth = generate_sythetic_points(
            X_sorted,
            starts,
            counts,
            uniq_labels,
            MIN_PTS_PER_CLUSTER,
            SEED=SEED,
            level=level,
        )

        if X_synth is not None:
            M_synth = int(X_synth.shape[0])
            n_new = n_total + M_synth
            if n_new > capacity:
                raise MemoryError(
                    f"Pool capacity exceeded: need {n_new} > {capacity}. "
                    f"Increase SYNTH_BUDGET / SYNTH_BUDGET_FACTOR. "
                    f"Level {level:03}, good clusters so far: {out_count:,} "
                    f"Active clusters this level: {K_batch:,}."
                )

            # Write new synthetic points into the pool tail and mark them active.
            X_pool[n_total:n_new] = X_synth
            labels_pool[n_total:n_new] = labels_synth
            active[n_total:n_new] = True

            # Evaluate F/G at new synthetic points
            if oracle_fn is not None:
                F_new, G_new = oracle_fn(X_synth)
                total_f_evals += X_synth.shape[0]
            else:
                F_new = f_fn(X_synth)  # type: ignore
                total_f_evals += X_synth.shape[0]
                if IGNORE_GRADIENT:
                    G_new = 0.0
                else:
                    G_new = g_fn(X_synth)  # type: ignore
            F_pool[n_total:n_new] = F_new
            G_pool[n_total:n_new] = G_new

            n_total = n_new
            synth_added_this_level = True

            # Re-sort active set
            idx_active = xp.nonzero(active[:n_total])[0]
            label_active = labels_pool[idx_active]
            perm = xp.argsort(label_active)
            idx_sorted = idx_active[perm]
            label_sorted = label_active[perm]
            uniq_labels, starts, counts = compute_label_segments(label_sorted)
            K_batch = int(uniq_labels.shape[0])
            X_sorted = X_pool[idx_sorted]  # Refresh views

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Padding")

        # --- Error Computation (Vector-Valued) ---
        F_sorted = F_pool[idx_sorted]  # (N_act, fd)
        G_sorted = G_pool[idx_sorted]  # (N_act, fd, d)

        # Centroids
        mus = xp.add.reduceat(X_sorted, starts) / counts[:, None]  # (K, d)

        # Eval f/g at centroids
        if oracle_fn is not None:
            f_mus, g_mus = oracle_fn(mus)
            total_f_evals += mus.shape[0]
        else:
            f_mus = f_fn(mus)  # type: ignore
            total_f_evals += mus.shape[0]
            if IGNORE_GRADIENT:
                g_mus = xp.zeros((K_batch, fd, d), dtype=mus.dtype)
            else:
                g_mus = g_fn(mus)  # type: ignore

        map_n_to_k = repeats(xp.arange(K_batch), counts)
        dx = X_sorted - mus[map_n_to_k]  # (N_act, d)

        # Linear Surrogate: f_lin = f(mu) + g(mu) * dx
        # g(mu) shape (K, fd, d) -> expand -> (N, fd, d)
        # dx shape (N, d) -> (N, d, 1)
        # result (N, fd)
        g_mus_exp = g_mus[map_n_to_k]  # (N, fd, d)
        linear_term = (g_mus_exp @ dx[:, :, None])[:, :, 0]  # (N, fd)
        f_lin = f_mus[map_n_to_k] + linear_term

        diff = F_sorted - f_lin  # (N, fd)
        abs_diff = xp.abs(diff)

        # Calculate E_cluster based on metric
        if ERROR_METRIC == "L2":
            # L2 Error: Total Frobenius Norm of the error matrix for the cluster.
            # Matches "Old Code" behavior (Total L2 Norm) extended to vectors.

            # 1. Square errors: (N, fd)
            sq_err = diff**2

            # 2. Sum squared errors per cluster per component: (K, fd)
            sum_sq_per_cluster_comp = xp.add.reduceat(sq_err, starts)

            # 3. Sum over components to get total squared error per cluster: (K,)
            # (If fd=1, this is a no-op which matches scalar behavior)
            total_sq_error = xp.sum(sum_sq_per_cluster_comp, axis=1)

            # 4. Square root to get L2 norm
            E_cluster = xp.sqrt(total_sq_error)

            # Standard L2 check
            is_good = E_cluster < TOL

        elif ERROR_METRIC == "Linf-global":
            # e_j(C) = max_{x} |diff_j|
            # Need max per cluster, per component.
            # We iterate columns of m to use scatter_max
            E_rel_max = xp.zeros((K_batch,), dtype=diff.dtype)

            for j in range(fd):
                max_diff_j = xp.zeros((K_batch,), dtype=diff.dtype)
                scatter_maximum_1d(max_diff_j, map_n_to_k, abs_diff[:, j])

                # Normalize by Global S_j
                scale = F_max_global[j] + TINY_EPS
                rel_err_j = max_diff_j / scale

                # Accumulate max relative error across components
                E_rel_max = xp.maximum(E_rel_max, rel_err_j)

            E_cluster = E_rel_max
            is_good = E_cluster < TOL

        elif ERROR_METRIC == "Linf-local":
            E_rel_max = xp.zeros((K_batch,), dtype=diff.dtype)
            E_abs_max = xp.zeros((K_batch,), dtype=diff.dtype)  # <--- Track Abs Error
            abs_F = xp.abs(F_sorted)

            for j in range(fd):
                max_diff_j = xp.zeros((K_batch,), dtype=diff.dtype)
                scatter_maximum_1d(max_diff_j, map_n_to_k, abs_diff[:, j])

                # Accumulate Max Absolute Error across components
                E_abs_max = xp.maximum(E_abs_max, max_diff_j)  # <--- Update

                # Local Scale L_j = max_{x in C} |f_j|
                local_max_F = xp.zeros((K_batch,), dtype=diff.dtype)
                scatter_maximum_1d(local_max_F, map_n_to_k, abs_F[:, j])

                scale = local_max_F + TINY_EPS
                rel_err_j = max_diff_j / scale

                E_rel_max = xp.maximum(E_rel_max, rel_err_j)

            E_cluster = E_rel_max

            # Hybrid Acceptance: Relative < TOL  OR  Absolute < TOL
            # This prevents over-refining near f(x) = 0
            is_good = (E_cluster < TOL) | (E_abs_max < TOL)
        else:
            raise ValueError(f"Unknown ERROR_METRIC: {ERROR_METRIC}")

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Error Computation")
        # ---- 4. Finalize Good Clusters ----
        good_idx = xp.nonzero(is_good)[0]
        n_good = int(good_idx.shape[0])

        if n_good > 0:
            all_radii, all_L_obs = calculate_scalar_metrics(
                dx, G_sorted, g_mus, map_n_to_k, K_batch
            )

            # Pack good points for Q calculation
            mask_good_pts = is_good[map_n_to_k]
            dx_good = dx[mask_good_pts]
            good_counts = counts[good_idx]
            good_starts = xp.r_[0, xp.cumsum(good_counts)[:-1]]

            Qs, _ = calculate_Q(
                dx_good,
                good_starts,
                good_counts,
                n_good,
                Q_SHRINK=Q_SHRINK,
            )

            # Store results
            end_out = out_count + n_good
            buffers, out_cap = adjust_buffer_capacity(
                buffers,
                out_count,
                out_cap,
                end_out,
                STORE_HESSIAN_SQRT_PER_CLUSTER,
                STORE_TREE_DATA,
            )

            # Optional: Store Hessian for good clusters (using their active component)
            if STORE_HESSIAN_SQRT_PER_CLUSTER:
                # Need to determine active component for good clusters too?
                # Prompt only specified it for Splitting.
                # We will pick component 0 or dominant just for storage,
                # OR re-run selection. Let's run selection for consistency.

                # 1. Compute L2 relative error per component
                diff_good = diff[mask_good_pts]  # (N_good, fd)
                F_good = F_sorted[mask_good_pts]

                sq_err = xp.add.reduceat(diff_good**2, good_starts)  # (K_good, fd)
                sq_norm = xp.add.reduceat(F_good**2, good_starts)

                l2_err_num = xp.sqrt(sq_err)
                l2_err_den = xp.sqrt(sq_norm) + TINY_EPS
                rel_l2 = l2_err_num / l2_err_den

                active_comps = xp.argmax(rel_l2, axis=1)  # (K_good,)

                # 2. Extract Gradients
                G_good_all = G_sorted[mask_good_pts]  # (N_good, fd, d)
                g_mu_good_all = g_mus[good_idx]  # (K_good, fd, d)

                # We need to slice G per point based on the cluster's active component
                # Expand active_comps to points
                map_good_n_to_k = repeats(xp.arange(n_good), good_counts)
                pt_active = active_comps[map_good_n_to_k]  # (N_good,)

                # Fancy indexing to grab (N_good, d)
                # G[i, pt_active[i], :]
                idx_n = xp.arange(int(diff_good.shape[0]))
                G_sel = G_good_all[idx_n, pt_active, :]
                g_mu_sel = g_mu_good_all[map_good_n_to_k, pt_active, :]  # approx

                dG_sel = G_sel - g_mu_sel
                H_sqrt = estimate_hessian_sqrt_symmetric(dx_good, dG_sel, good_starts)
                buffers["H_sqrt"][out_count:end_out] = H_sqrt

            buffers["centroids"][out_count:end_out] = mus[good_idx]
            buffers["f"][out_count:end_out] = f_mus[good_idx]
            buffers["g"][out_count:end_out] = g_mus[good_idx]
            buffers["radii"][out_count:end_out] = all_radii[good_idx]
            buffers["L_obs"][out_count:end_out] = all_L_obs[good_idx]
            buffers["Q"][out_count:end_out] = Qs
            buffers["E_cluster"][out_count:end_out] = E_cluster[good_idx]
            buffers["counts"][out_count:end_out] = counts[good_idx]

            out_count = end_out

            if STORE_TREE_DATA:
                buffers["tree_data"][level] = {}
                points_in_clusters = []
                for i, idx in enumerate(good_idx):
                    cluster_points = X_sorted[map_n_to_k == idx]
                    points_in_clusters.append(cluster_points)
                buffers["tree_data"][level] = {
                    "centroids": mus[good_idx],
                    "f": f_mus[good_idx],
                    "g": g_mus[good_idx],
                    "radii": all_radii[good_idx],
                    "L_obs": all_L_obs[good_idx],
                    "E_cluster": E_cluster[good_idx],
                    "counts": counts[good_idx],
                    "Q": Qs,
                    "points_in_clusters": points_in_clusters,
                }

            # Deactivate all points (real + synthetic) in accepted clusters.
            active[idx_sorted[mask_good_pts]] = False

        # ---- 5. Hessian-Informed Split for Bad Clusters ----
        bad_idx = xp.nonzero(~is_good)[0]
        n_bad_clusters = int(bad_idx.shape[0])

        if n_bad_clusters == 0:
            synth_added_prev_level = synth_added_this_level
            logger.info(f"Level {level:03}: no bad clusters remain, terminating.")
            break

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Finalize good clusters")
        mask_bad_pts = (~is_good)[map_n_to_k]
        dx_bad = dx[mask_bad_pts]
        bad_counts = counts[bad_idx]
        bad_starts = xp.r_[0, xp.cumsum(bad_counts)[:-1]]

        if IGNORE_GRADIENT or not USE_HESSIAN_INFORMED_SPLIT:
            M_split = xp.eye(d)[None, :, :].repeat(n_bad_clusters, axis=0)
        else:
            # --- Worst-Case Active Component Selection ---
            # 1. Compute L2 Relative Error per component for BAD clusters
            diff_bad = diff[mask_bad_pts]  # (N_bad, m)
            F_bad_pts = F_sorted[mask_bad_pts]

            # Sum squares per cluster segment
            sum_sq_err = xp.add.reduceat(diff_bad**2, bad_starts)  # (K_bad, fd)
            sum_sq_val = xp.add.reduceat(F_bad_pts**2, bad_starts)  # (K_bad, fd)

            err_l2 = xp.sqrt(sum_sq_err)
            norm_l2 = xp.sqrt(sum_sq_val) + TINY_EPS
            scores = err_l2 / norm_l2  # (K_bad, fd)

            # Select winner
            active_comps = xp.argmax(scores, axis=1)  # (K_bad,)

            # 2. Extract specific Gradients
            if not USE_COVARIANCE_HESSIAN:
                G_bad_all = G_sorted[mask_bad_pts]  # (N_bad, fd, d)
                g_mu_bad_all = g_mus[bad_idx]  # (K_bad, fd, d)

                # Map cluster choice to points
                map_bad_n_to_k = repeats(xp.arange(n_bad_clusters), bad_counts)
                pt_active_comp = active_comps[map_bad_n_to_k]  # (N_bad,)

                # Slice: G_sel[i] = G_bad_all[i, pt_active_comp[i], :]
                range_N = xp.arange(int(dx_bad.shape[0]))
                G_sel = G_bad_all[range_N, pt_active_comp, :]  # (N_bad, d)

                # Centroid grad for that component (expanded)
                g_mu_sel = g_mu_bad_all[map_bad_n_to_k, pt_active_comp, :]

                dG_bad = G_sel - g_mu_sel

                # 3. Compute H^1/2
                M_split = estimate_hessian_sqrt_symmetric(dx_bad, dG_bad, bad_starts)

            else:
                # Instead of directly computing Hessian via least squares,
                # we take points in good error, compute covariance,
                # and use the inverse as the approximate Hessian.
                # It can be shown that if we have enough points,
                # this is close to the Hessian, and requires much less computation.
                # Do not use gradients at all.
                F_bad_all = F_sorted[mask_bad_pts]  # (N_bad, fd)

                M_split = compute_good_covariance_hessian(
                    dx_bad, F_bad_all, f_lin, bad_starts
                )

        # Transform and Split
        map_bad_n_to_k = repeats(xp.arange(n_bad_clusters), bad_counts)
        M_exp = M_split[map_bad_n_to_k]
        Z = (M_exp @ dx_bad[:, :, None])[:, :, 0]  # (N_bad, d)

        # Cov(Z)
        ZZ_all = Z[:, :, None] * Z[:, None, :]
        CovZ = xp.add.reduceat(ZZ_all, bad_starts)
        CovZ = 0.5 * (CovZ + CovZ.transpose(0, 2, 1))

        # Eigenvector
        v = get_dominant_eigenvector(CovZ)
        v_exp = v[map_bad_n_to_k]
        proj_scores = xp.sum(Z * v_exp, axis=1)

        split_right = compute_median_split_decision(
            proj_scores, map_bad_n_to_k, bad_starts, bad_counts
        )

        # Update labels
        idx_bad_pool = idx_sorted[mask_bad_pts]
        old_labels = uniq_labels[bad_idx]
        old_per_point = old_labels[map_bad_n_to_k]

        new_labels = old_per_point * 2 + split_right.astype(xp.int64)
        labels_pool[idx_bad_pool] = new_labels

        # End-of-level bookkeeping
        synth_added_prev_level = synth_added_this_level
        logger.info(
            f"Level {level:03}: Clusters={K_batch:,}, active={n_active:,}, "
            f"good={int(n_good):,}, bad={n_bad_clusters:,}, "
            f"synth={n_total - N0:,}, "
            f"pool_used={n_total:,}/{capacity:,} ({n_total / capacity:.0%}) , "
            f"F_in_max={xp.max(F_max_global):.3e}, "
        )
        logger.info(f"Runtime - Level {level:03}: {time.perf_counter() - t_level:.3f}s")

        level += 1

    # ---- 6. Finalize ----
    if level > MAX_LEVELS:
        logger.warning(f"Reached MAX_LEVELS={MAX_LEVELS}, terminating.")
    final_count = int(out_count)
    logger.info(f"Adaptive clustering complete: {final_count:,} final clusters.")
    logger.info(f"Runtime - Total: {time.perf_counter() - _start_time:.3f}s")

    out_dict = {k: buffers[k][:final_count] for k in buffers if k != "tree_data"}
    if STORE_TREE_DATA:
        out_dict["tree_data"] = buffers["tree_data"]
    out_dict["num_levels"] = level
    out_dict["num_generated_pts"] = n_total - N0
    out_dict["total_f_evals"] = int(total_f_evals)
    return out_dict


if __name__ == "__main__":
    import argparse
    from functools import partial

    from src import FIGURES_DIR
    from src.logging_utils import setup_logging
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging(console_level=logging.DEBUG)

    AC_PLOTS_DIR = FIGURES_DIR / "adaptive_clustering_rosenbrock"

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Adaptive GPU Clustering Example with Rosenbrock Function"
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Dimensionality of the problem"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-2, help="Tolerance for training and evaluation"
    )
    parser.add_argument(
        "--n-train", type=float, default=100_000, help="Number of training samples"
    )
    parser.add_argument(
        "--n-init-clusters",
        type=int,
        default=25,
        help="Number of initial clusters for adaptive clustering",
    )
    parser.add_argument(
        "--n-min-per-cluster",
        type=int,
        default=500,
        help="Minimum points per cluster for adaptive clustering",
    )
    parser.add_argument(
        "--error-metric",
        type=str,
        default="Linf-global",
        choices=["L2", "Linf-global", "Linf-local"],
        help="Error metric for adaptive clustering",
    )
    parser.add_argument(
        "--initial-clustering-method",
        type=str,
        default="kmeans",
        choices=["kmeans", "knndbscan"],
        help="Method for initial clustering",
    )
    parser.add_argument(
        "--rosenbrock-a",
        type=float,
        default=100.0,
        help="Rosenbrock function parameter 'a'",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="",
        help="Subdirectory for output files",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-execution even if output file exists",
    )

    args = parser.parse_args()
    DIM = args.dim
    TOL = args.tol
    N_TRAIN = int(args.n_train)
    INITIAL_CLUSTERING_METHOD = args.initial_clustering_method
    NUM_INITIAL_CLUSTERS = args.n_init_clusters
    MIN_PTS_PER_CLUSTER = args.n_min_per_cluster
    ERROR_METRIC = args.error_metric
    A_ROSENBROCK = args.rosenbrock_a
    OUTPUT_SUBDIR = args.output_subdir
    FORCE_REEXEC = args.force

    # Check if file already exists
    AC_PLOTS_DIR = AC_PLOTS_DIR / OUTPUT_SUBDIR if OUTPUT_SUBDIR else AC_PLOTS_DIR
    AC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    img_fpath = (
        AC_PLOTS_DIR
        / f"adaptive_clustering_rosenbrock_{A_ROSENBROCK}_tol{TOL}_error_metric_{ERROR_METRIC}_init{NUM_INITIAL_CLUSTERS}_min{MIN_PTS_PER_CLUSTER}_initmethod_{INITIAL_CLUSTERING_METHOD}.png"
    )
    if DIM == 2 and img_fpath.exists() and not FORCE_REEXEC:
        print(f"Plot file {img_fpath} already exists. Skipping execution.")
        exit(0)

    # Creat partial Rosenbrock functions
    rosenbrock = partial(rosenbrock, a=A_ROSENBROCK)
    rosenbrock_grad = partial(rosenbrock_grad, a=A_ROSENBROCK)

    xp.random.seed(0)
    print(f"Generating {N_TRAIN:,} points (d={DIM})...")
    PROB_LIMS = (-1.0, 3)
    X = xp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(N_TRAIN, DIM))

    print("Starting Adaptive Clustering...")
    results = adaptive_clustering(
        X,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        TOL=TOL,
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        ERROR_METRIC=ERROR_METRIC,
        SYNTH_BUDGET_FACTOR=1000,
    )

    print("-" * 40)
    print(f"Final Cluster Count: {results['centroids'].shape[0]:,}")

    E_cluster_vals = results["E_cluster"]
    counts = results["counts"]
    print(
        "E_cluster stats: min/mean/max =",
        xp.round(E_cluster_vals.min(), 6),
        xp.round(E_cluster_vals.mean(), 6),
        xp.round(E_cluster_vals.max(), 6),
    )
    print(
        "Counts stats: min/mean/max =",
        int(counts.min()),
        f"{counts.mean():.1f}",
        int(counts.max()),
    )

    if DIM == 2 and len(results["centroids"]) <= 10_000:
        import numpy as np
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Ellipse

        if IS_CUPY_BACKEND:
            results = {k: xp.asnumpy(v) for k, v in results.items()}

        # 1 plot 2 subplots (1X2)
        # Subplot 1: scatter of final cluster centroids on contourf
        # Subplot 2: scatter of final cluster centroids with trust ellipsoids on contourf as well as circles corresponding to radii

        x = np.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        y = np.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        X_grid, Y_grid = np.meshgrid(x, y)

        # Ensure input to rosenbrock is (N, 2)
        grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
        Z_grid = rosenbrock(grid_pts)
        if hasattr(Z_grid, "get"):
            Z_grid = Z_grid.get()  # Handle cupy return if needed
        Z_grid = Z_grid.reshape(X_grid.shape)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # -- Subplot 1: Contours + Centroids --
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
        axs[0].scatter(
            results["centroids"][:, 0],
            results["centroids"][:, 1],
            c="red",
            s=5,
            label="Centroids",
        )

        # -- Subplot 2: Contours + OBBs --
        cs2 = axs[1].contourf(
            X_grid,
            Y_grid,
            Z_grid,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(cs2, ax=axs[1])
        axs[1].set_title("Adaptive Clustering with Oriented Bounding Boxes")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].set_aspect("equal")

        # Plot Centroids
        axs[1].scatter(
            results["centroids"][:, 0],
            results["centroids"][:, 1],
            c="red",
            s=5,
            zorder=10,
        )

        # Draw Ellipsoids
        centroids = results["centroids"]
        Qs = results["Q"]
        radii = results["radii"]

        for i in range(centroids.shape[0]):
            center = centroids[i]
            Q = Qs[i]
            radius = radii[i]

            # Eigen-decomposition of Q
            eigvals, eigvecs = np.linalg.eigh(Q)
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            a, b = 2.0 * np.sqrt(1.0 / eigvals)  # 2-sigma lengths
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ellip = Ellipse(
                xy=center,
                width=a,
                height=b,
                angle=angle,
                edgecolor="blue",
                facecolor="none",
                lw=0.5,
                alpha=0.7,
            )
            axs[1].add_patch(ellip)

            # Draw radius circle with dashed line
            circ = Circle(
                xy=center,
                radius=radius,
                edgecolor="black",
                facecolor="none",
                lw=0.5,
                alpha=0.7,
                linestyle="dashed",
            )
            axs[1].add_patch(circ)

        # Update legend to include ellipses and circles
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Centroids",
                markerfacecolor="red",
                markersize=5,
            ),
            Line2D([0], [0], color="blue", lw=0.5, label="Trust Ellipsoid"),
            Line2D(
                [0],
                [0],
                color="black",
                lw=0.5,
                linestyle="dashed",
                label="Trust Radius",
            ),
        ]
        axs[1].legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        fig.suptitle(
            f"Rosenbrock (a={A_ROSENBROCK}) (TOL={TOL}, ERROR_METRIC={ERROR_METRIC}), N_CLUSTERS={len(centroids):,}, INITIAL_CLUSTERING_METHOD={INITIAL_CLUSTERING_METHOD}",
            fontsize=11,
        )

        plt.subplots_adjust(top=0.88)
        plt.savefig(img_fpath, dpi=300)
        print(f"Saved figure to {img_fpath}")
    else:
        print("Skipping plot (d != 2 or too many clusters).")

    # Save npz file of results
    npz_fpath = img_fpath.with_suffix(".npz")
    if IS_CUPY_BACKEND:
        results = {k: xp.asnumpy(v) for k, v in results.items()}
    xp.savez_compressed(npz_fpath, **results)
