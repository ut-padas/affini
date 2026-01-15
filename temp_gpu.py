import time
from typing import Optional

import cupy as cp
import numpy as np
from cuml.cluster import KMeans as cuKMeans
from cupyx import scatter_max


def median_split_decision_batched(
    scores: cp.ndarray,
    cluster_id: cp.ndarray,
    starts: cp.ndarray,
    counts: cp.ndarray,
) -> cp.ndarray:
    """
    Compute a per-cluster median split decision for packed/segmented data.

    Given per-point scores s_i and cluster ids k(i), this returns:
        decision_i = (s_i >= median_k(i))
    where median_k is the median score within cluster k.

    Args:
        scores: (N,) per-point scalar scores.
        cluster_id: (N,) integer cluster ids in [0, K-1] for each point.
        starts: (K,) start offsets for each cluster *in the packed order*
                (typically starts = [0, cumsum(counts)[:-1]]).
        counts: (K,) cluster sizes.

    Returns:
        split_decision: (N,) boolean array; True means "right child", False "left child".

    Notes:
        This performs a global radix-like sort:
            1) stable sort by score
            2) stable sort by cluster_id
        which yields (cluster_id asc, score asc within cluster).
    """
    N = int(scores.shape[0])
    if N == 0:
        return scores >= 0  # empty, doesn't matter

    # --- radix sort: secondary key = score, primary key = cluster_id ---
    # stable sort by score first
    idx_by_score = cp.argsort(scores, kind="stable")  # (N,)
    cluster_sorted_by_score = cluster_id[idx_by_score.get()]

    # stable sort by cluster id (preserves score-order within each cluster)
    idx_group = cp.argsort(cluster_sorted_by_score, kind="stable")  # (N,)
    order = idx_by_score[idx_group]  # (N,)

    # scores sorted within clusters
    scores_sorted = scores[order]  # (N,)

    # --- median per cluster in sorted array ---
    # For each cluster segment [starts[k] : starts[k] + counts[k]),
    # median is average of middle element(s).
    i0 = starts + (counts - 1) // 2  # (K,)
    i1 = starts + counts // 2  # (K,)

    med = 0.5 * (scores_sorted[i0] + scores_sorted[i1])  # (K,)

    # --- assign median threshold back to original point order ---
    # threshold for point i is med[cluster_id[i]]
    split_decision = scores >= med[cluster_id]  # (N,)

    return split_decision


def generate_synthetic_points_batched(
    X: cp.ndarray, starts: cp.ndarray, counts: cp.ndarray, N_min: int
) -> tuple[Optional[cp.ndarray], Optional[cp.ndarray]]:
    """
    Batch generate synthetic (convex-combination) points for clusters smaller than N_min,
    over all clusters. This is intended to pad small clusters with additional points to
    stabilize error-estimation without changing the original data.

    For each cluster k with count m_k < N_min, we create n_needed[k] = N_min - m_k
    synthetic points. Each synthetic point is a convex combination of two randomly
    chosen points from that same cluster:
        x_synth = a * x_i + (1 - a) * x_j,   where a ~ Uniform(0,1),
    and i, j are random indices within the cluster segment.

    Args:
        X: (N, d) Array of data points sorted by cluster labels.
        starts: (K,) Array of starting indices for each cluster.
        counts: (K,) Array of counts for each cluster.
        N_min: Minimum number of points required per cluster.

    Returns:
        X_synth: (M, d) synthetic points, where M = sum_k max(0, N_min - counts[k]),
            or None if no synthesis is needed.
        cluster_ids_synth: (M,) integer cluster id for each synthetic point (so you can
            associate each synthetic point back to its cluster), or None if none.
    """
    # Identify how many synthetic points are needed per cluster
    n_needed = cp.maximum(0, N_min - counts)  # (K,)
    total_synth = int(cp.sum(n_needed))

    # If no synthesis needed, return None
    if total_synth == 0:
        return None, None

    # Clusters that actually need synthetic points
    small_cluster_indices = cp.nonzero(n_needed)[0]  # (K_small,)
    needed_counts = n_needed[small_cluster_indices]  # (K_small,)

    # For each synthetic point, assign it to which cluster it belongs
    # E.g., if C0: 3 points, C1: 0 points, C2: 2 points, then
    # cluster_ids_synth = [0,0,0,2,2]
    cluster_ids_synth = cp.repeat(small_cluster_indices.get(), needed_counts.get())

    # For each synthetic point, pick two random indices within that cluster
    synth_starts = starts[cluster_ids_synth]  # (M,)
    synth_counts = counts[cluster_ids_synth]  # (M,)

    # Generate random offsets
    offsets1 = cp.floor(cp.random.rand(total_synth) * synth_counts).astype(cp.int32)
    offsets2 = cp.floor(cp.random.rand(total_synth) * synth_counts).astype(cp.int32)

    # Convert offsets to absolute indices into X
    idx1 = synth_starts + offsets1
    idx2 = synth_starts + offsets2

    # Form convex combinations
    alpha = cp.random.rand(total_synth, 1).astype(X.dtype)  # (M, 1)
    X_synth = alpha * X[idx1] + (1.0 - alpha) * X[idx2]  # (M, d)

    return X_synth, cluster_ids_synth


def batched_estimate_hessian_symmetric(
    DeltaX: cp.ndarray, DeltaG: cp.ndarray, starts: cp.ndarray
) -> cp.ndarray:
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
    C = cp.add.reduceat(XX_all, starts)  # (K, d, d)
    E = cp.add.reduceat(XG_all, starts)  # (K, d, d)

    # RHS S_k = E_k + E_k^T
    S = E + E.transpose(0, 2, 1)  # (K, d, d)

    # Batch eigen-decomposition of each C_k = U_k D_k U_k^T
    dvals, U = cp.linalg.eigh((C + C.transpose(0, 2, 1)) * 0.5)
    dvals = cp.maximum(dvals, 1e-18)

    # Gtilde = U^T S U
    Gtilde = U.transpose(0, 2, 1) @ S @ U

    # F_ij = Gtilde_ij / (d_i + d_j)
    denom = dvals[:, :, None] + dvals[:, None, :]
    F = Gtilde / denom

    # H = U F U^T
    H = U @ F @ U.transpose(0, 2, 1)

    # Ensure symmetry
    return (H + H.transpose(0, 2, 1)) * 0.5


def batched_sqrt_abs_psd(H: cp.ndarray) -> cp.ndarray:
    """
    Batch compute the square root of the absolute value of symmetric PSD matrices over
    all clusters.
    Equations (per matrix/cluster k):
        If H_k is symmetric, it has eigendecomposition:
            H_k = U_k diag(λ_k) U_k^T
        Then the square root of the absolute value is:
            |H_k|^{1/2} = U_k diag(sqrt(|λ_k|)) U_k^T

    Args:
        H: (K, d, d) Batch of symmetric matrices (corresponding to Hessians of clusters)

    Returns:
        M: (K, d, d) Batch of square-rooted absolute value matrices, where
            M_k M_k = |H_k|.
    """
    # Batch eigen-decomposition of each H_k = U_k D_k U_k^T
    # w: (K, d) eigenvalues, U: (K, d, d) eigenvectors
    w, U = cp.linalg.eigh(H)

    # Compute sqrt(|w|) elementwise
    w_abs_sqrt = cp.sqrt(cp.abs(w))

    M = (U * w_abs_sqrt[:, None, :]) @ U.transpose(0, 2, 1)  # (K, d, d)
    return (M + M.transpose(0, 2, 1)) * 0.5


def batched_top_eigvec(Cov: cp.ndarray) -> cp.ndarray:
    """
    Batch compute the dominant (top) eigenvector of symmetric covariance matrices over
    all clusters.

    Args:
        Cov: (K, d, d) Batch of symmetric covariance matrices.

    Returns:
        v: (K, d) Batch of top eigenvectors, where v[k] is the eigenvector corresponding
            to the largest eigenvalue of Cov[k] matrix.
    """
    # Batch eigen-decomposition of symmetric matrices
    # w: (K, d) eigenvalues (ascending), V: (K, d, d) eigenvectors
    w, V = cp.linalg.eigh((Cov + Cov.transpose(0, 2, 1)) * 0.5)

    # Since eigenvalues are ascending, the last column of V is the top eigenvector
    return V[:, :, -1]  # (K, d)


def batched_calculate_Q(
    DeltaX: cp.ndarray,
    starts: cp.ndarray,
    counts: cp.ndarray,
    K: int,
    eps: float = 1e-12,
) -> cp.ndarray:
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
    # Segment sums (cluster-wise) to get Covariance matrices
    Cov = cp.add.reduceat(XX_all, starts)  # (K, d, d)

    # Batch eigen-decomposition of each Cov_k = U_k D_k U_k^T
    _, U = cp.linalg.eigh((Cov + Cov.transpose(0, 2, 1)) * 0.5)

    # Projections of points onto PCA axes
    map_n_to_k = cp.repeat(np.arange(K), counts.get())

    # Expand U so every point gets its cluster's eigenvector matrix
    U_expanded = U[map_n_to_k]  # (N, d, d)

    # Compute projections of points onto their cluster's PCA axes
    proj = U_expanded.transpose(0, 2, 1) @ DeltaX[:, :, None]  # (N, d, 1)
    proj = proj[:, :, 0]  # (N, d)

    # Compute per-axis max |projection| per cluster
    proj_abs = cp.abs(proj)
    d_axes = cp.zeros((K, proj_abs.shape[1]), dtype=proj_abs.dtype)
    scatter_max(d_axes, map_n_to_k, proj_abs)

    # Avoid divide-by-zero
    d_axes = cp.maximum(d_axes, eps)  # (K, d)

    # Build Q_k matrices
    Dinv2 = 1.0 / (d_axes**2)  # (K, d)
    Q_inner = U * Dinv2[:, None, :]  # (K, d, d)
    Q = Q_inner @ U.transpose(0, 2, 1)  # (K, d, d)
    return (Q + Q.transpose(0, 2, 1)) * 0.5


def _init_output_buffers(max_estimated_clusters, d, dtype):
    cap = max(1, int(max_estimated_clusters))
    buffers = {
        "centroids": cp.empty((cap, d), dtype=dtype),
        "f": cp.empty((cap,), dtype=dtype),
        "g": cp.empty((cap, d), dtype=dtype),
        "Q": cp.empty((cap, d, d), dtype=dtype),
        "E_abs": cp.empty((cap,), dtype=dtype),
        "E_rel": cp.empty((cap,), dtype=dtype),
        "counts": cp.empty((cap,), dtype=cp.int64),
    }
    return buffers, cap


def _ensure_capacity(buffers, out_n, cap, required_total):
    required_total = int(required_total)
    if required_total <= cap:
        return buffers, cap
    new_cap = max(required_total, max(1, cap) * 2)
    d = int(buffers["centroids"].shape[1])
    dtype = buffers["centroids"].dtype
    new_buffers = {
        "centroids": cp.empty((new_cap, d), dtype=dtype),
        "f": cp.empty((new_cap,), dtype=dtype),
        "g": cp.empty((new_cap, d), dtype=dtype),
        "Q": cp.empty((new_cap, d, d), dtype=dtype),
        "E_abs": cp.empty((new_cap,), dtype=dtype),
        "E_rel": cp.empty((new_cap,), dtype=dtype),
        "counts": cp.empty((new_cap,), dtype=buffers["counts"].dtype),
    }
    n = int(out_n)
    if n > 0:
        for k in new_buffers:
            new_buffers[k][:n] = buffers[k][:n]
    return new_buffers, int(new_cap)


def adaptive_clustering(
    X: cp.ndarray,
    *,
    f_fn,
    g_fn,
    ATOL: float = 1e-2,
    RTOL: float = 1e-2,
    N_INIT_CLUSTERS: int = 100,
    N_MIN_PER_CLUSTER: int = 100,
    MAX_EST_CLUSTERS: int = 100_000,
    VERBOSE: bool = True,
    SEED: int = 0,
):
    X = cp.asarray(X)

    if VERBOSE:
        print("Adaptive Clustering Parameters:")
        print(f"  ATOL: {ATOL}")
        print(f"  RTOL: {RTOL}")
        print(f"  N_INIT_CLUSTERS: {N_INIT_CLUSTERS:,}")
        print(f"  N_MIN_PER_CLUSTER: {N_MIN_PER_CLUSTER:,}")
        print(f"  MAX_EST_CLUSTERS: {MAX_EST_CLUSTERS:,}")
        print(f"  SEED: {SEED}")
        print(f"  Data Points: {X.shape[0]:,} (Dim: {X.shape[1]:,})")

    start_init = time.perf_counter()
    kmeans = cuKMeans(n_clusters=N_INIT_CLUSTERS, random_state=SEED)
    labels = kmeans.fit_predict(X)
    if VERBOSE:
        print(f"  > Initial KMeans: {time.perf_counter() - start_init:.4f}s")

    N, d = X.shape
    buffers, cap = _init_output_buffers(MAX_EST_CLUSTERS, d=d, dtype=X.dtype)
    out_n = 0

    F_all = f_fn(X).ravel()
    G_all = g_fn(X)[:, 0, :]
    F_inf = float(cp.max(cp.abs(F_all))) + 1e-30

    active_mask = cp.ones(N, dtype=bool)

    level = 0
    while True:
        n_active = int(cp.sum(active_mask))
        if n_active == 0:
            break
        t_start = time.perf_counter()

        # --- 1. Identify Active Real Points ---
        active_indices = cp.nonzero(active_mask)[0]
        active_labels = labels[active_indices]

        # Initial Sort of Real Points (Needed to generate synth properly)
        sort_order_real = cp.argsort(active_labels)
        sorted_indices_real = active_indices[sort_order_real]
        sorted_labels_real = active_labels[sort_order_real]
        sorted_X_real = X[sorted_indices_real]

        # Get counts of REAL points to determine synthesis needs
        unique_cluster_ids, starts_real, counts_real = cp.unique(  # type: ignore
            sorted_labels_real, return_index=True, return_counts=True
        )
        K_batch = int(unique_cluster_ids.shape[0])

        # --- 2. Generate Synthetic Points ---
        X_synth, ids_synth = generate_synthetic_points_batched(
            sorted_X_real, starts_real, counts_real, N_MIN_PER_CLUSTER
        )

        # --- 3. Merge & Resort ---
        if X_synth is not None:
            # Create "Combined" arrays
            # We track original indices: 0..N for real, -1 for synth
            global_ids_synth = cp.full(
                X_synth.shape[0], -1, dtype=sorted_indices_real.dtype
            )

            # Map synth IDs (which are 0..K index in batch) back to actual labels
            # ids_synth currently holds indices into unique_cluster_ids array
            # We need the actual label values to sort correctly
            labels_synth = unique_cluster_ids[ids_synth]

            X_combined = cp.vstack([sorted_X_real, X_synth])
            L_combined = cp.concatenate([sorted_labels_real, labels_synth])
            I_combined = cp.concatenate(
                [sorted_indices_real, global_ids_synth]
            )  # Track real/synth

            # THE COST: Sort the combined data
            sort_order_comb = cp.argsort(L_combined)

            X_batch = X_combined[sort_order_comb]
            L_batch = L_combined[sort_order_comb]
            I_batch = I_combined[sort_order_comb]  # Positive=Real, -1=Synth

            # Re-compute segmentation on combined data
            # Note: unique_cluster_ids should remain the same set, but starts/counts change
            _, starts_batch, counts_batch = cp.unique(  # type: ignore
                L_batch, return_index=True, return_counts=True
            )

            # We need function values for synth points
            # For real points, we could look them up, but for uniformity,
            # let's just evaluate f/g on the whole batch or gather.
            # Evaluating f/g on X_batch is cleaner but might re-evaluate real points.
            # Optimization: Gather F/G for real, Evaluate for Synth, then Concat.

            # Gather Real
            F_real = F_all[sorted_indices_real]
            G_real = G_all[sorted_indices_real]

            # Evaluate Synth
            F_synth = f_fn(X_synth).ravel()
            G_synth = g_fn(X_synth)[:, 0, :]

            F_comb = cp.concatenate([F_real, F_synth])
            G_comb = cp.vstack([G_real, G_synth])

            F_batch = F_comb[sort_order_comb]
            G_batch = G_comb[sort_order_comb]

        else:
            # No augmentation needed
            X_batch = sorted_X_real
            L_batch = sorted_labels_real
            I_batch = sorted_indices_real
            starts_batch = starts_real
            counts_batch = counts_real
            F_batch = F_all[sorted_indices_real]
            G_batch = G_all[sorted_indices_real]

        # Update F_inf in case synth changed scale
        F_inf = float(cp.max(cp.abs(F_batch))) + 1e-30

        # --- 4. Batched Linear Algebra (Unified) ---
        # Centroids
        sums = cp.add.reduceat(X_batch, starts_batch)
        mus = sums / counts_batch[:, None]

        f_mus = f_fn(mus)[:, 0]
        g_mus = g_fn(mus)[:, 0, :]

        # Error Check
        map_n_to_k = cp.repeat(np.arange(K_batch), counts_batch.get())
        DeltaX = X_batch - mus[map_n_to_k]
        f_linear = f_mus[map_n_to_k] + cp.sum(DeltaX * g_mus[map_n_to_k], axis=1)
        err_point = cp.abs(F_batch - f_linear)

        E_abs = cp.zeros(K_batch, dtype=err_point.dtype)
        scatter_max(E_abs, map_n_to_k, err_point)

        E_rel = E_abs / F_inf
        is_good = (E_abs < ATOL) | (E_rel < RTOL)

        # --- 5. Finalize Good ---
        good_indices = cp.nonzero(is_good)[0]
        if good_indices.size > 0:
            cluster_good_mask = is_good
            point_good_mask = cluster_good_mask[map_n_to_k]

            # Use Combined data for Q (better robust ellipse)
            good_DeltaX = DeltaX[point_good_mask]
            good_counts = counts_batch[good_indices]
            good_starts = cp.r_[0, cp.cumsum(good_counts)[:-1]]

            Qs = batched_calculate_Q(
                good_DeltaX, good_starts, good_counts, int(good_indices.size)
            )

            n_good = int(good_indices.size)
            end_out = out_n + n_good
            buffers, cap = _ensure_capacity(buffers, out_n, cap, end_out)

            buffers["centroids"][out_n:end_out] = mus[good_indices]
            buffers["f"][out_n:end_out] = f_mus[good_indices]
            buffers["g"][out_n:end_out] = g_mus[good_indices]
            buffers["Q"][out_n:end_out] = Qs
            buffers["E_abs"][out_n:end_out] = E_abs[good_indices]
            buffers["E_rel"][out_n:end_out] = E_rel[good_indices]
            buffers["counts"][out_n:end_out] = good_counts
            out_n = int(end_out)

            # Mark REAL points as inactive
            # I_batch has real indices >= 0, synth indices -1
            point_indices_in_batch = I_batch[point_good_mask]
            real_point_mask = point_indices_in_batch >= 0

            global_indices_good = point_indices_in_batch[real_point_mask]
            active_mask[global_indices_good] = False

        # --- 6. Split Bad ---
        bad_indices = cp.nonzero(~is_good)[0]
        if bad_indices.size == 0:
            break

        cluster_bad_mask = ~is_good
        point_bad_mask = cluster_bad_mask[map_n_to_k]

        bad_DeltaX = DeltaX[point_bad_mask]

        # Gradients for Hessian
        bad_G_points = G_batch[point_bad_mask]
        bad_g_mus = g_mus[map_n_to_k][point_bad_mask]
        bad_DeltaG = bad_G_points - bad_g_mus

        bad_counts = counts_batch[bad_indices]
        bad_starts = cp.r_[0, cp.cumsum(bad_counts)[:-1]]
        K_bad = int(bad_indices.size)

        # Standard Hessian on Unified Data
        H = batched_estimate_hessian_symmetric(bad_DeltaX, bad_DeltaG, bad_starts)
        M = batched_sqrt_abs_psd(H)

        map_bad_n_to_k = cp.repeat(np.arange(K_bad), bad_counts.get())
        M_expanded = M[map_bad_n_to_k]  # (N_bad, d, d)
        Z = (M_expanded @ bad_DeltaX[:, :, None])[:, :, 0]  # (N_bad, d)

        Z_starts = bad_starts
        # Build per-point outer products:
        # ZZ_all[i] = z_i z_i^T
        ZZ_all = Z[:, :, None] * Z[:, None, :]  # (N_bad, d, d)
        # Segment sums (cluster-wise) to get Covariance matrices for Z
        Cov_Z = cp.add.reduceat(ZZ_all, Z_starts)  # (K_bad, d, d)

        # Compute score projections of Z onto top eigenvector of Cov_Z
        v = batched_top_eigvec(Cov_Z)  # (K_bad, d)
        v_expanded = v[map_bad_n_to_k]  # (N_bad, d)
        scores = cp.sum(Z * v_expanded, axis=1)  # (N_bad,)
        split_decision = median_split_decision_batched(
            scores, map_bad_n_to_k, bad_starts, bad_counts
        )  # (N_bad,)

        # Update Labels for REAL points only
        # We must align the split decision back to the original points
        indices_in_bad_subset = I_batch[point_bad_mask]

        # Filter only real points
        is_real_in_bad = indices_in_bad_subset >= 0
        global_indices_bad_real = indices_in_bad_subset[is_real_in_bad]

        # Corresponding decisions
        decisions_real = split_decision[is_real_in_bad]

        old_ids = labels[global_indices_bad_real]
        new_ids = old_ids * 2 + decisions_real.astype(cp.int32)
        labels[global_indices_bad_real] = new_ids

        level += 1
        if VERBOSE:
            n_synth_count = X_synth.shape[0] if X_synth is not None else 0
            print(
                f"Level {level}: Active {n_active:,} (+{n_synth_count} synth) -> {K_batch} clusters (Good: {good_indices.size}, Bad: {bad_indices.size}) | {time.perf_counter() - t_start:.4f}s"
            )

    n_final = int(out_n)

    if VERBOSE:
        print(f"Adaptive Clustering Complete: {n_final:,} final clusters.")
        print(f"  > Total Time: {time.perf_counter() - start_init:.4f}s")
    return {k: buffers[k][:n_final] for k in buffers}


ROSENBROCK_SCALE = 100.0


def rosenbrock(X):
    x_i = X[:, :-1]
    x_next = X[:, 1:]
    return cp.sum(
        ROSENBROCK_SCALE * (x_next - x_i**2.0) ** 2.0 + (1.0 - x_i) ** 2.0,
        axis=1,
        keepdims=True,
    )


def rosenbrock_grad(X):
    n_pts, n_dim = X.shape
    J = cp.zeros((n_pts, 1, n_dim), dtype=X.dtype)
    J[:, 0, :-1] += 2 * (X[:, :-1] - 1) - 4 * ROSENBROCK_SCALE * X[:, :-1] * (
        X[:, 1:] - X[:, :-1] ** 2
    )
    J[:, 0, 1:] += 2 * ROSENBROCK_SCALE * (X[:, 1:] - X[:, :-1] ** 2)
    return J


if __name__ == "__main__":
    cp.random.seed(0)
    _N = 1_000_000
    d = 2
    print(f"Generating {_N:,} points (d={d})...")
    PROB_LIMS = (-1.0, 3)
    X = cp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(_N, d))

    TOL = 1e-2
    N_MIN_PER_CLUSTER = 100
    N_INIT_CLUSTERS = 100

    print("Starting Adaptive GPU Clustering...")
    results = adaptive_clustering(
        X,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        ATOL=TOL,
        RTOL=TOL,
        N_INIT_CLUSTERS=N_INIT_CLUSTERS,
        N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
        VERBOSE=True,
    )

    print("-" * 40)
    print(f"Final Cluster Count: {results['centroids'].shape[0]:,}")

    E_abs_vals = results["E_abs"]
    E_rel_vals = results["E_rel"]
    counts = results["counts"]
    print(
        "E_abs stats: min/mean/max =",
        cp.round(E_abs_vals.min(), 3),
        cp.round(E_abs_vals.mean(), 3),
        cp.round(E_abs_vals.max(), 3),
    )
    print(
        "E_rel stats: min/mean/max =",
        cp.round(E_rel_vals.min(), 6),
        cp.round(E_rel_vals.mean(), 6),
        cp.round(E_rel_vals.max(), 6),
    )
    print(
        "Counts stats: min/mean/max =",
        int(counts.min()),
        f"{counts.mean():.1f}",
        int(counts.max()),
    )

    if d == 2 and len(results["centroids"]) <= 3000:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Ellipse

        # 1 plot 2 subplots (1X2)
        # Subplot 1: scatter of final cluster centroids on contourf
        # Subplot 2: scatter of final cluster centroids with trust ellipses on contourf

        x = cp.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        y = cp.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        X_grid, Y_grid = cp.meshgrid(x, y)
        Z_grid = rosenbrock(cp.vstack([X_grid.ravel(), Y_grid.ravel()]).T).reshape(
            X_grid.shape
        )

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        cs = axs[0].contourf(
            X_grid.get(),
            Y_grid.get(),
            Z_grid.get(),
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
            X_grid.get(),
            Y_grid.get(),
            Z_grid.get(),
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
            results["centroids"][:, 0].get(),
            results["centroids"][:, 1].get(),
            c="red",
            s=10,
            label="Centroids",
        )

        # Draw ellipses
        level = 1.0
        for i in range(results["centroids"].shape[0]):
            mu = results["centroids"][i].get()
            Q = results["Q"][i].get()

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
            f"Adaptive Clustering: Rosenbrock (TOL={TOL}, N_INIT={N_INIT_CLUSTERS}, N_MIN={N_MIN_PER_CLUSTER}), Final Clusters: {results['centroids'].shape[0]:,}",
            fontsize=16,
        )

        plt.subplots_adjust(top=0.88)
        fname = f"adaptive_clustering_rosenbrock_tol{int(TOL*1e4)}_init{N_INIT_CLUSTERS}_min{N_MIN_PER_CLUSTER}.png"
        plt.savefig(fname, dpi=300)
        print(f"Saved figure to {fname}")
    else:
        print("Skipping plot (d != 2 or too many clusters).")
