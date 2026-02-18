import gc
import logging
import time
from typing import Any, Callable, Literal, Optional

import numpy as np
from sklearn.cluster import KMeans

from adaptive_clustering_utils import (
    CudaTimer,
    adjust_buffer_capacity_host,
    allocate_output_buffers_host,
    convert_to_2d_array,
    convert_to_3d_array,
    get_n_smallest_per_block,
    run_fused_metrics,
    run_fused_q_dist,
    run_fused_score,
    run_fused_stats,
    run_fused_taylor_comp_abs,
    run_fused_taylor_metric,
    scatter_maximum_1d,
)
from src import IS_CUPY_BACKEND, Array, _to_device, _to_host, xp
from src.logging_utils import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)
TINY_EPS = 1e-12


def adaptive_clustering(
    X: Array,
    *,
    f_fn: Optional[Callable[[Array], Array]] = None,
    g_fn: Optional[Callable[[Array], Array]] = None,
    oracle_fn: Optional[Callable[[Array], tuple[Array, Array]]] = None,
    TOL: float,
    ERROR_METRIC: Literal["L2", "Linf-global", "Linf-local"] = "Linf-global",
    INITIAL_CLUSTERING_METHOD: Literal["kmeans", "knndbscan"] = "kmeans",
    NUM_INITIAL_CLUSTERS: int = 100,
    MIN_PTS_PER_CLUSTER: int = 100,
    MAX_CLUSTERS_EST: int = 100_000,
    SEED: int = 0,
    Q_SHRINK: float = 0.05,
    MAX_LEVELS: int = 50,
    BATCH_SIZE: int = 10_000_000,
    CLUSTERING_MATRIX: Literal["Hessian_LSQ", "Cov_Lvl_curves", "None"] = "Hessian_LSQ",
    IGNORE_GRADIENT: bool = False,
    STORE_HESSIAN_SQRT_PER_CLUSTER: bool = False,
    STORE_TREE_DATA: bool = False,
    LOG_MEM_STATS: bool = False,
    PROFILE_TIMING: bool = False,
) -> dict[str, Any]:
    """
    Adaptive Clustering for Vector-Valued Functions (GPU Accelerated).

    Constructs a piecewise-local linear surrogate model for a function F: R^d \to R^fd.
    The algorithm recursively splits clusters where the local first-order Taylor approximation fails to meet the specified `TOL`.

    Architecture (Sort-Streaming):
    To handle datasets larger than GPU memory, this implementation uses a "Sort-Streaming" approach:
    1.  **Physical Sort:** Points are sorted by cluster label in VRAM, allowing zero-copy slicing.
    2.  **Streaming:** The current level is processed in batches on the GPU.
    3.  **Offloading:** Accepted ("Good") clusters are immediately moved to CPU host memory.
    4.  **Refinement:** Rejected ("Bad") clusters are split and streamed to a CPU staging buffer
        for the next level.

    **Error Metrics:**
    Let E(x) = F(x) - (F(μ_k) + ∇F(μ_k)(x - μ_k)).
    -   L2: RMSE-style check. √(∑_{x ∈ C_k} ‖E(x)‖_F^2) < TOL.
        (Strict total energy check).
    -   Linf-global: Max relative error normalized by global range.
        max_{x ∈ C_k} max_j |E_j(x)| / max_y |F_j(y)| < TOL.
    -   Linf-local: Max relative error normalized by local cluster range.
        max_{x ∈ C_k} max_j |E_j(x)| / max_{y ∈ C_k} |F_j(y)| < TOL.
        Includes hybrid acceptance (Absolute < TOL) to handle zero-crossings.

    Args:
        X: Input training points. Shape (N, d).
        f_fn: Callable returning function values. Output shape (N, fd).
        g_fn: Callable returning gradients/Jacobians. Output shape (N, fd, d).
        TOL: Error tolerance for cluster acceptance.
        ERROR_METRIC: Metric for acceptance ('L2', 'Linf-global', 'Linf-local').
        INITIAL_CLUSTERING_METHOD: Method for initial clustering ('kmeans' or 'knndbscan').
        NUM_INITIAL_CLUSTERS: Number of seeds for initial K-Means.
        MIN_PTS_PER_CLUSTER: Minimum points per cluster. If a cluster has fewer,
            synthetic points are generated via convex combinations to stabilize geometry.
        MAX_CLUSTERS_EST: Initial capacity for CPU output buffers (auto-expands).
        SEED: Random seed for initialization and synthetic generation.
        Q_SHRINK: Shrinkage factor a for covariance: Σ ← (1-a)Σ + a (Tr(Σ)/d)I.
        MAX_LEVELS: Maximum recursion depth for splitting.
        BATCH_SIZE: Maximum number of points processed in a single GPU kernel launch.
        USE_HESSIAN_INFORMED_SPLIT: If True, splits clusters along the dominant
        eigenvector of the local Hessian estimate. If False, splits along the principal
        axis (PCA) of points.
        IGNORE_GRADIENT: If True, assumes ∇F = 0 (Voronoi/Piecewise-Constant
        approximation).
        STORE_HESSIAN_SQRT_PER_CLUSTER: If True, computes and stores |H|^{1/2} for every
        accepted cluster (useful for anisotropic trust regions downstream).
        LOG_MEM_STATS: If True, prints VRAM usage at key steps.
        PROFILE_TIMING: If True, tracks detailed execution times for performance profiling.

    Returns:
        A dictionary containing the surrogate model parameters:
        -   `centroids`: (K, d) Cluster centers.
        -   `f`: (K, fd) Function values at centroids.
        -   `g`: (K, fd, d) Gradients/Jacobians at centroids.
        -   `Q`: (K, d, d) Trust region shape matrices (inverse covariance).
        -   `radii`: (K,) Trust region radii.
        -   `L_obs`: (K,) Observed Lipschitz constants.
        -   `E_cluster`: (K,) Max observed error per cluster.
        -   `counts`: (K,) Number of points support per cluster.
        -   `H_sqrt`: (K, d, d) (Optional) Symmetric Hessian square roots.
        -   `tree_data`: (dict) (Optional) Per-level cluster data keyed by level index, for visualization/debugging.
        -   `num_levels`: (int) Total levels refined.
        -   `num_generated_pts`: (int) Number of generated points.
        -  `total_f_evals`: (int) Total function evaluations performed.
        -  `profiling_stats`: (dict) Detailed timing and statistics if PROFILE_TIMING is True.
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

    if STORE_TREE_DATA:
        logger.warning(
            "STORE_TREE_DATA=True: This will store per-cluster point data for all finalized clusters, which can consume a lot of memory, and slow down the algorithm. Use for debugging/visualization purposes, but be cautious with large datasets or many levels of refinement."
        )

    _start_time = time.perf_counter()

    mempool = None
    if IS_CUPY_BACKEND:
        import cupy as cp

        mempool = cp.get_default_memory_pool()

    # Profiling Stats
    profiling_stats = {
        "N0": int(xp.asarray(X).shape[0]),
        "num_levels": 0,
        "num_generated_pts": 0,
        "total_f_evals": 0,
        "num_final_clusters": 0,
        "time_total": 0.0,
        "time_init": 0.0,
        "time_slicing": 0.0,
        "time_find_req_synth": 0.0,
        "time_synth_gen": 0.0,
        "time_error_calc": 0.0,
        "time_filter": 0.0,
        "time_good_Q": 0.0,
        "time_good_d2h": 0.0,
        "time_bad_split": 0.0,
        "time_bad_d2h": 0.0,
        "time_transition": 0.0,
        "time_gc": 0.0,
        "level_stats": [],
    }

    num_generated_pts = 0
    total_f_evals = 0
    tree_data: dict = {}

    # --- 1. Initialization Phase ---
    with CudaTimer("time_init", profiling_stats, PROFILE_TIMING):
        # Upload initial data to GPU
        X0_gpu = xp.asarray(X, dtype=xp.float64)  # Shape: (N, d)
        d = X0_gpu.shape[1]

        if LOG_MEM_STATS:
            log_mem_stats("Start Adaptive Clustering")

        # Evaluate initial F/G on GPU
        _t0 = time.perf_counter()
        if oracle_fn is not None:
            F_out, G_out = oracle_fn(X0_gpu)
            F0_gpu = convert_to_2d_array(F_out)
            fd = F0_gpu.shape[1]
            if IGNORE_GRADIENT:
                G0_gpu = xp.zeros((X0_gpu.shape[0], fd, d), dtype=xp.float64)
            else:
                G0_gpu = convert_to_3d_array(G_out)
        else:
            assert f_fn is not None
            F0_gpu = convert_to_2d_array(f_fn(X0_gpu))  # Shape: (N, fd)
            fd = F0_gpu.shape[1]

            if IGNORE_GRADIENT:
                G0_gpu = xp.zeros((X0_gpu.shape[0], fd, d), dtype=xp.float64)
            else:
                assert g_fn is not None
                G0_gpu = convert_to_3d_array(g_fn(X0_gpu))  # Shape: (N, fd, d)

        total_f_evals += X0_gpu.shape[0]

        logger.info(f"Runtime - Initial f/g: {time.perf_counter()-_t0:.3f}s")
        logger.info(f"Function: Input d={d}, Output fd={fd}")
        if LOG_MEM_STATS:
            log_mem_stats("After Initial F/G Evaluation")

        # Compute Global Scale for Linf-global metric: S_j = max_x |F_j(x)|
        F_max_global = xp.max(xp.abs(F0_gpu), axis=0)  # Shape: (fd,)

        # Initial Clustering
        _t1 = time.perf_counter()
        if INITIAL_CLUSTERING_METHOD == "kmeans":
            if IS_CUPY_BACKEND:
                from cuml.cluster import KMeans as cuKMeans

                km = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
                labels0 = km.fit_predict(X0_gpu)
            else:
                km = KMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
                labels0 = km.fit_predict(X0_gpu)
            logger.info(f"Runtime - Initial KMeans: {time.perf_counter()-_t1:.3f}s")
        elif INITIAL_CLUSTERING_METHOD == "knndbscan":
            from knndbscan import knn_dbscan

            _device = "gpu" if IS_CUPY_BACKEND else "cpu"
            labels0 = knn_dbscan(
                X0_gpu,
                eps=0.5,
                min_samples=MIN_PTS_PER_CLUSTER,
                device=_device,
                n_jobs=-1,
            ).astype(xp.int32)

            # Output labels can include -1 for noise.
            _uni_labels = xp.unique(labels0)
            _num_clusters = int(_uni_labels.shape[0])
            logger.info(f"Initial KNN-DBSCAN: Found {_num_clusters:,} clusters")

            _has_noise = int(xp.sum(_uni_labels == -1) > 0)
            if _has_noise:
                logger.info("Initial KNN-DBSCAN: Detected noise points (label -1)")
            else:
                logger.info("Initial KNN-DBSCAN: No noise points detected")

            logger.info(f"Runtime - Initial KNN-DBSCAN: {time.perf_counter()-_t1:.3f}s")
        else:
            raise ValueError(
                f"Unsupported INITIAL_CLUSTERING_METHOD: {INITIAL_CLUSTERING_METHOD}"
            )

        if LOG_MEM_STATS:
            log_mem_stats("After Initial KMeans Clustering")

        # --- Physical Sort ---
        # Group data by cluster in VRAM to enable contiguous slicing later
        sort_idx = xp.argsort(labels0)
        CLUSTER_PTS = X0_gpu[sort_idx]  # (N, d)
        CLUSTER_F = F0_gpu[sort_idx]  # (N, fd)
        CLUSTER_G = G0_gpu[sort_idx]  # (N, fd, d)

        uniq_labels, CLUSTER_SIZES = xp.unique(labels0[sort_idx], return_counts=True)
        CLUSTER_STARTS = xp.r_[0, xp.cumsum(CLUSTER_SIZES)[:-1]]

        # Free temp buffers
        del X0_gpu, F0_gpu, G0_gpu, labels0, sort_idx, uniq_labels
        if mempool is not None:
            mempool.free_all_blocks()

        # Allocate Output Buffers on CPU (Host) to save VRAM
        buffers, out_cap = allocate_output_buffers_host(
            MAX_CLUSTERS_EST, d, fd, STORE_HESSIAN_SQRT_PER_CLUSTER
        )
        out_count = 0

    # --- 2. Main Refinement Loop (BFS) ---
    level = 0
    while level <= MAX_LEVELS:
        _t_level = time.perf_counter()
        if LOG_MEM_STATS:
            log_mem_stats(f"Start: L{level:02}")

        num_clusters = len(CLUSTER_SIZES)
        if num_clusters == 0:
            logger.info(f"Level {level:02}: No clusters to process, terminating.")
            break

        # Collect Level Stats (irrespective of PROFILE_TIMING)
        # Note: calling .item() or float() on cupy scalar forces sync
        profiling_stats["level_stats"].append(
            {
                "level": int(level),
                "clusters_start": int(num_clusters),
                "size_min": int(float(xp.min(CLUSTER_SIZES))),
                "size_mean": float(xp.mean(CLUSTER_SIZES)),
                "size_median": int(float(xp.median(CLUSTER_SIZES))),
                "size_max": int(float(xp.max(CLUSTER_SIZES))),
            }
        )

        # Calculate augmentation needs (how many synthetic points per cluster?)
        with CudaTimer("time_find_req_synth", profiling_stats, PROFILE_TIMING):
            level_augmented_sizes = xp.maximum(CLUSTER_SIZES, MIN_PTS_PER_CLUSTER)
            level_total_aug_pts = int(xp.sum(level_augmented_sizes))

        # CPU view for batch scheduling
        aug_sizes_cpu = _to_host(level_augmented_sizes)
        aug_cumsum_cpu = np.cumsum(aug_sizes_cpu)

        # Host Staging Buffers for the NEXT level (Streaming architecture)
        next_level_X_list = []
        next_level_F_list = []
        next_level_G_list = []
        next_level_sizes_list = []

        # --- Batch Processing Loop ---
        # Process clusters in chunks that fit in GPU memory
        num_batches = level_total_aug_pts // BATCH_SIZE + 1
        start_cluster_idx = 0

        if LOG_MEM_STATS:
            log_mem_stats(f"Start: L{level:02} Batching")

        K_good = 0
        for b in range(num_batches):
            logger.info(f"Level {level:02}: Processing Batch {b+1}/{num_batches}")
            # Determine cluster range for this batch
            target_aug_count = (b + 1) * BATCH_SIZE
            end_cluster_idx = int(np.searchsorted(aug_cumsum_cpu, target_aug_count))
            end_cluster_idx = min(end_cluster_idx + 1, num_clusters)
            if start_cluster_idx >= end_cluster_idx:
                break

            # 1. Zero-Copy Slicing from GPU Arrays
            with CudaTimer("time_slicing", profiling_stats, PROFILE_TIMING):
                sizes = CLUSTER_SIZES[start_cluster_idx:end_cluster_idx]
                starts_local = CLUSTER_STARTS[start_cluster_idx:end_cluster_idx]
                slice_start = int(starts_local[0])
                slice_end = int(starts_local[-1] + sizes[-1])

                pts = CLUSTER_PTS[slice_start:slice_end]
                f_vals = CLUSTER_F[slice_start:slice_end]
                g_vals = CLUSTER_G[slice_start:slice_end]
                starts_local_shifted = starts_local - slice_start

            # 2. Augmentation (Synthetic Points via Convex Combinations)
            with CudaTimer("time_synth_gen", profiling_stats, PROFILE_TIMING):
                aug_sizes = level_augmented_sizes[start_cluster_idx:end_cluster_idx]
                aug_total = int(xp.sum(aug_sizes))
                K_batch = len(aug_sizes)

                X_batch = xp.empty((aug_total, d), dtype=xp.float64)
                F_batch = xp.empty((aug_total, fd), dtype=xp.float64)
                G_batch = xp.empty((aug_total, fd, d), dtype=xp.float64)

                # Map augmented indices to local cluster IDs
                # Eg: aug_sizes = [3,5,2] -> aug_ids = [0,0,0,1,1,1,1,1,2,2]
                # Aug starts: starting index of each cluster in augmented batch
                # Eg: aug_sizes = [3,5,2] -> aug_starts = [0,3,8,10]
                aug_starts = xp.r_[0, xp.cumsum(aug_sizes)[:-1]]
                aug_ids = repeats(xp.arange(K_batch), aug_sizes)

                # Fill Real Data
                pt_idxs_in_cluster = (
                    xp.arange(aug_total, dtype=xp.int32) - aug_starts[aug_ids]
                )
                is_real = pt_idxs_in_cluster < sizes[aug_ids]
                X_batch[is_real] = pts
                F_batch[is_real] = f_vals
                G_batch[is_real] = g_vals

                # Fill Synthetic Data
                is_synth = ~is_real
                num_synth = int(xp.sum(is_synth))
                num_generated_pts += num_synth
                if num_synth > 0:
                    if LOG_MEM_STATS:
                        log_mem_stats(
                            f" L{level:02} B{b:02}/{num_batches:02}: Start Augmentation"
                        )
                    synth_c_ids = aug_ids[is_synth]
                    rng = xp.random.default_rng(SEED + level + b)

                    # Vectorized random pair selection within clusters
                    c_starts = starts_local_shifted[synth_c_ids]
                    c_counts = sizes[synth_c_ids]
                    rand1 = (rng.random(num_synth) * c_counts).astype(xp.int64)
                    rand2 = (rng.random(num_synth) * (c_counts - 1)).astype(
                        xp.int64
                    ) + 1
                    idx1 = c_starts + rand1
                    idx2 = c_starts + (rand1 + rand2) % c_counts

                    p1 = pts[idx1]
                    p2 = pts[idx2]
                    alpha = rng.random((num_synth, 1)).astype(xp.float64)
                    X_synth_vals = p1 * alpha + p2 * (1.0 - alpha)
                    X_batch[is_synth] = X_synth_vals

                    # Evaluate f for synthetic points
                    if oracle_fn is not None:
                        F_syn, G_syn = oracle_fn(X_synth_vals)
                        F_batch[is_synth] = convert_to_2d_array(F_syn)
                        if IGNORE_GRADIENT:
                            G_batch[is_synth] = 0.0
                        else:
                            G_batch[is_synth] = convert_to_3d_array(G_syn)
                    else:
                        assert f_fn is not None
                        F_batch[is_synth] = convert_to_2d_array(f_fn(X_synth_vals))
                        if IGNORE_GRADIENT:
                            G_batch[is_synth] = 0.0
                        else:
                            assert g_fn is not None
                            G_batch[is_synth] = convert_to_3d_array(g_fn(X_synth_vals))
                    total_f_evals += X_synth_vals.shape[0]
                    del (
                        c_starts,
                        c_counts,
                        rand1,
                        rand2,
                        idx1,
                        idx2,
                        p1,
                        p2,
                        alpha,
                        X_synth_vals,
                    )

            if LOG_MEM_STATS:
                log_mem_stats(
                    f" L{level:02} B{b:02}/{num_batches:02}: After Augmentation"
                )

            # 3. Compute Centroids & Reference Gradients
            with CudaTimer("time_error_calc", profiling_stats, PROFILE_TIMING):
                mus = (
                    xp.add.reduceat(X_batch, aug_starts) / aug_sizes[:, None]
                )  # (K_batch, d)

                if oracle_fn is not None:
                    F_mus_val, G_mus_val = oracle_fn(mus)
                    f_mus = convert_to_2d_array(F_mus_val)
                    if IGNORE_GRADIENT:
                        g_mus = xp.zeros((K_batch, fd, d), dtype=xp.float64)
                    else:
                        g_mus = convert_to_3d_array(G_mus_val)
                else:
                    assert f_fn is not None
                    f_mus = convert_to_2d_array(f_fn(mus))  # (K_batch, fd)
                    if IGNORE_GRADIENT:
                        g_mus = xp.zeros((K_batch, fd, d), dtype=xp.float64)
                    else:
                        assert g_fn is not None
                        g_mus = convert_to_3d_array(g_fn(mus))  # (K_batch, fd, d)

                total_f_evals += mus.shape[0]

                # 4. Error Metrics & Acceptance Logic
                if ERROR_METRIC == "Linf-local":
                    # Acceptance (Rel < TOL || Abs < TOL)
                    E_cluster_rel_max = xp.zeros((K_batch,), dtype=xp.float64)
                    E_cluster_abs_max = xp.zeros((K_batch,), dtype=xp.float64)

                    # Loop over dimensions to avoid materializing (N, FD) diff tensor
                    for j in range(fd):
                        # Compute max |f_j| per cluster
                        F_max_j = xp.zeros((K_batch,), dtype=xp.float64)
                        scatter_maximum_1d(F_max_j, aug_ids, xp.abs(F_batch[:, j]))

                        # Taylor Error (Absolute)
                        pointwise_tay_err_j = run_fused_taylor_comp_abs(
                            X_batch, mus, aug_ids, F_batch, f_mus, g_mus, j
                        )  # (N_batch,)

                        # Max Error per cluster
                        E_abs_j = xp.zeros((K_batch,), dtype=xp.float64)
                        scatter_maximum_1d(E_abs_j, aug_ids, pointwise_tay_err_j)

                        # Normalize
                        scale = xp.maximum(F_max_j, TINY_EPS)
                        E_rel_j = E_abs_j / scale  # (K_batch,)

                        # Accumulate Max
                        E_cluster_rel_max = xp.maximum(E_cluster_rel_max, E_rel_j)
                        E_cluster_abs_max = xp.maximum(E_cluster_abs_max, E_abs_j)

                    is_good = (E_cluster_rel_max < TOL) | (E_cluster_abs_max < TOL)
                    E_cluster = E_cluster_rel_max

                else:
                    # Standard Metrics (L2 or Linf-global)
                    if ERROR_METRIC == "L2":
                        scales_ker = xp.ones((fd,), dtype=xp.float64)
                        metric_mode = 0  # Sum Squares
                    else:
                        scales_ker = F_max_global
                        metric_mode = 1  # Max Relative

                    # err_pts: (N_batch,) aggregated error per point
                    err_pts = run_fused_taylor_metric(
                        X_batch,
                        mus,
                        aug_ids,
                        F_batch,
                        f_mus,
                        g_mus,
                        scales_ker,
                        metric_mode,
                    )

                    if ERROR_METRIC == "L2":
                        # Sum squared errors over points -> sqrt (Frobenius Norm)
                        sum_sq = xp.add.reduceat(err_pts, aug_starts)  # (K_batch,)
                        E_cluster = xp.sqrt(sum_sq)
                    else:
                        # Max relative error over points
                        E_cluster = xp.zeros((K_batch,), dtype=xp.float64)  # (K_batch,)
                        scatter_maximum_1d(E_cluster, aug_ids, err_pts)

                    is_good = E_cluster < TOL  # (K_batch,)

            if LOG_MEM_STATS:
                log_mem_stats(
                    f" L{level:02} B{b:02}/{num_batches:02}: After Error Metrics"
                )

            # 5. Handle Accepted ("Good") Clusters
            with CudaTimer("time_filter", profiling_stats, PROFILE_TIMING):
                good_idx_local = xp.nonzero(is_good)[0]
                K_good = int(good_idx_local.shape[0])

            if K_good > 0:
                with CudaTimer("time_good_Q", profiling_stats, PROFILE_TIMING):
                    good_mask = is_good[aug_ids]
                    X_good = X_batch[good_mask]
                    G_good = G_batch[good_mask]
                    good_counts = aug_sizes[good_idx_local]
                    dense_labels_good = repeats(xp.arange(K_good), good_counts)

                    # Geometry Metrics (Radii, Lipschitz)
                    radii_pt, ratio_pt = run_fused_metrics(
                        X_good,
                        G_good,
                        mus[good_idx_local],
                        g_mus[good_idx_local],
                        dense_labels_good,
                    )
                    radii = xp.zeros((K_good,), dtype=radii_pt.dtype)
                    L_obs = xp.zeros((K_good,), dtype=ratio_pt.dtype)
                    scatter_maximum_1d(radii, dense_labels_good, radii_pt)
                    scatter_maximum_1d(L_obs, dense_labels_good, ratio_pt)

                    # Compute Trust Ellipsoid (Q) via Covariance
                    Cov, _ = run_fused_stats(
                        X_good,
                        mus[good_idx_local],
                        dense_labels_good,
                        K_good,
                        d,
                        fd,
                        xp.zeros(K_good, dtype=xp.int32),
                        calc_E=False,
                    )
                    Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

                    # Covariance Shrinkage: C <- (1-a)C + a*tr(C)/d*I
                    if Q_SHRINK > 0.0:
                        tr = xp.trace(Cov, axis1=1, axis2=2)
                        scale_tr = tr / max(d, 1)
                        I = xp.eye(d, dtype=Cov.dtype)[None, :, :]
                        Cov = (1.0 - Q_SHRINK) * Cov + Q_SHRINK * (
                            scale_tr[:, None, None] * I
                        )
                        Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

                    # Eigendecomposition for Q
                    Lambda, U = xp.linalg.eigh(Cov)
                    Lambda_inv = 1.0 / xp.maximum(Lambda, TINY_EPS)

                    # Scale Q by max distance in eigen-basis
                    dist_sq = run_fused_q_dist(
                        X_good, mus[good_idx_local], dense_labels_good, U, Lambda_inv
                    )
                    max_dist = xp.zeros((K_good,), dtype=dist_sq.dtype)
                    scatter_maximum_1d(max_dist, dense_labels_good, dist_sq)
                    scale_fac = 1.0 / xp.maximum(max_dist, TINY_EPS)

                    # Q = U * (scale * L_inv) * U^T
                    Qs = (
                        U * (scale_fac[:, None] * Lambda_inv)[:, None, :]
                    ) @ U.transpose(0, 2, 1)

                    # Optional: Hessian Storage
                    H_sqrt = None
                    if STORE_HESSIAN_SQRT_PER_CLUSTER:
                        # Identify active component for storage (based on L2 relative error)
                        g_mus_exp = g_mus[good_idx_local][dense_labels_good]
                        dx = X_good - mus[good_idx_local][dense_labels_good]
                        lin = xp.einsum("nmd,nd->nm", g_mus_exp, dx)
                        diff_vals = F_batch[good_mask] - (
                            f_mus[good_idx_local][dense_labels_good] + lin
                        )

                        good_starts = xp.r_[0, xp.cumsum(good_counts)[:-1]]
                        sq_err = xp.add.reduceat(diff_vals**2, good_starts)
                        sq_val = xp.add.reduceat(F_batch[good_mask] ** 2, good_starts)
                        scores = xp.sqrt(sq_err) / (xp.sqrt(sq_val) + TINY_EPS)
                        active_comps_good = xp.argmax(scores, axis=1).astype(xp.int32)

                        # Estimate Symmetric Hessian
                        C_h, E_h = run_fused_stats(
                            X_good,
                            mus[good_idx_local],
                            dense_labels_good,
                            K_good,
                            d,
                            fd,
                            active_comps_good,
                            G=G_good,
                            g_mus=g_mus[good_idx_local],
                            calc_E=True,
                        )
                        C_h = 0.5 * (C_h + C_h.transpose(0, 2, 1))
                        S_h = E_h + E_h.transpose(0, 2, 1)

                        # Solve Sylvester Eq via Eigendecomposition
                        dvals, U_h = xp.linalg.eigh(C_h)
                        dvals = xp.maximum(dvals, TINY_EPS)
                        Gtilde = U_h.transpose(0, 2, 1) @ S_h @ U_h
                        F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])
                        _, V_h = xp.linalg.eigh(F_mat)

                        # Reconstruct |H|^1/2
                        Z = U_h @ V_h
                        M = (Z * xp.sqrt(xp.abs(_))[:, None, :]) @ Z.transpose(0, 2, 1)
                        H_sqrt = 0.5 * (M + M.transpose(0, 2, 1))

                # Download Results to CPU
                with CudaTimer("time_good_d2h", profiling_stats, PROFILE_TIMING):
                    end_out = out_count + K_good
                    buffers, out_cap = adjust_buffer_capacity_host(
                        buffers,
                        out_count,
                        out_cap,
                        end_out,
                        STORE_HESSIAN_SQRT_PER_CLUSTER,
                    )
                    buffers["centroids"][out_count:end_out] = _to_host(
                        mus[good_idx_local]
                    )
                    buffers["f"][out_count:end_out] = _to_host(f_mus[good_idx_local])
                    buffers["g"][out_count:end_out] = _to_host(g_mus[good_idx_local])
                    buffers["Q"][out_count:end_out] = _to_host(Qs)
                    if H_sqrt is not None:
                        buffers["H_sqrt"][out_count:end_out] = _to_host(H_sqrt)
                    buffers["E_cluster"][out_count:end_out] = _to_host(
                        E_cluster[good_idx_local]
                    )
                    buffers["radii"][out_count:end_out] = _to_host(radii)
                    buffers["L_obs"][out_count:end_out] = _to_host(L_obs)
                    buffers["counts"][out_count:end_out] = _to_host(
                        good_counts.astype(xp.int64)
                    )
                    out_count = end_out

                if STORE_TREE_DATA:
                    _src = out_count - K_good
                    _td_entry = {
                        "centroids": buffers["centroids"][_src:out_count].copy(),
                        "f": buffers["f"][_src:out_count].copy(),
                        "g": buffers["g"][_src:out_count].copy(),
                        "radii": buffers["radii"][_src:out_count].copy(),
                        "L_obs": buffers["L_obs"][_src:out_count].copy(),
                        "E_cluster": buffers["E_cluster"][_src:out_count].copy(),
                        "counts": buffers["counts"][_src:out_count].copy(),
                        "Q": buffers["Q"][_src:out_count].copy(),
                        "points_in_clusters": [
                            _to_host(X_good[dense_labels_good == i])
                            for i in range(K_good)
                        ],
                    }
                    tree_data[level] = _td_entry

            if LOG_MEM_STATS:
                log_mem_stats(
                    f" L{level:02} B{b:02}/{num_batches:02}: After Good Clusters"
                )

            # 6. Handle Rejected ("Bad") Clusters -> Split
            with CudaTimer("time_filter", profiling_stats, PROFILE_TIMING):
                bad_idx_local = xp.nonzero(~is_good)[0]
                n_bad = int(bad_idx_local.shape[0])

            if n_bad > 0:
                with CudaTimer("time_bad_split", profiling_stats, PROFILE_TIMING):
                    bad_mask = (~is_good)[aug_ids]
                    X_bad = X_batch[bad_mask]  # (N_bad, d)
                    F_bad = F_batch[bad_mask]
                    G_bad = G_batch[bad_mask]
                    bad_counts = aug_sizes[bad_idx_local]
                    dense_labels_bad = repeats(xp.arange(n_bad), bad_counts)

                    if IGNORE_GRADIENT or CLUSTERING_MATRIX == "None":
                        # Fallback: PCA Split (Split along variance of X)
                        M_split = xp.eye(d, dtype=xp.float64)[None, :, :].repeat(
                            n_bad, axis=0
                        )
                        C, _ = run_fused_stats(
                            X_bad,
                            mus[bad_idx_local],
                            dense_labels_bad,
                            n_bad,
                            d,
                            fd,
                            xp.zeros(n_bad, dtype=xp.int32),
                            calc_E=False,
                        )
                        C = 0.5 * (C + C.transpose(0, 2, 1))
                        CovZ = C

                    elif CLUSTERING_MATRIX == "Hessian_LSQ":
                        # Hessian Split: Split along eigenvector of Hessian estimate
                        # 1. Determine "Active Component" (Output dim with highest error)
                        g_mus_exp_bad = g_mus[bad_idx_local][dense_labels_bad]
                        dx_bad = X_bad - mus[bad_idx_local][dense_labels_bad]
                        lin_bad = xp.einsum("nmd,nd->nm", g_mus_exp_bad, dx_bad)
                        diff_bad = F_bad - (
                            f_mus[bad_idx_local][dense_labels_bad] + lin_bad
                        )

                        bad_starts = xp.r_[0, xp.cumsum(bad_counts)[:-1]]
                        sq_err = xp.add.reduceat(diff_bad**2, bad_starts)
                        sq_val = xp.add.reduceat(F_bad**2, bad_starts)
                        scores = xp.sqrt(sq_err) / (xp.sqrt(sq_val) + TINY_EPS)
                        active_comps = xp.argmax(scores, axis=1).astype(xp.int32)

                        # 2. Accumulate Stats and Solve for Hessian
                        C, E = run_fused_stats(
                            X_bad,
                            mus[bad_idx_local],
                            dense_labels_bad,
                            n_bad,
                            d,
                            fd,
                            active_comps,
                            G=G_bad,
                            g_mus=g_mus[bad_idx_local],
                            calc_E=True,
                        )
                        C = 0.5 * (C + C.transpose(0, 2, 1))
                        S = E + E.transpose(0, 2, 1)

                        dvals, U = xp.linalg.eigh(C)
                        dvals = xp.maximum(dvals, TINY_EPS)
                        Gtilde = U.transpose(0, 2, 1) @ S @ U
                        F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])
                        _, V = xp.linalg.eigh(F_mat)

                        # M = |H|^1/2
                        Z = U @ V
                        M_split = (Z * xp.sqrt(xp.abs(_))[:, None, :]) @ Z.transpose(
                            0, 2, 1
                        )
                        M_split = 0.5 * (M_split + M_split.transpose(0, 2, 1))

                        # Covariance in transformed space Z = M * dx
                        CovZ = M_split @ C @ M_split.transpose(0, 2, 1)

                    elif CLUSTERING_MATRIX == "Cov_Lvl_curves":
                        # Approximate Hessian using inverse covariance formulation.
                        # 1. Determine points we will use to approximate Hessian.
                        # Use the errors we have already computed
                        # We want to have enough points (2d*log(d))
                        n_est_cov = 2 * d * xp.log(d)
                        E_bad = E_cluster[bad_mask]
                        cov_est_mask = get_n_smallest_per_block(
                            E_bad, dense_labels_bad, n_est_cov  # is this right?
                        )
                        X_est_cov = X_bad[cov_est_mask]

                        # 2. Use these points to compute covariances
                        C, _ = run_fused_stats(
                            X_est_cov,
                            mus[bad_idx_local],
                            dense_labels_bad[cov_est_mask],
                            n_bad,
                            d,
                            fd,
                            xp.zeros(n_bad, dtype=xp.int32),
                            calc_E=False,
                        )
                        C = 0.5 * (C + C.transpose(0, 2, 1))

                        # 2.5: Shrinkage?

                        # 3. Take inverses
                        CovZ = xp.linalg.inv(C)

                    # Get Split Direction (Dominant Eigenvector)
                    _, V_z = xp.linalg.eigh(CovZ)
                    v = V_z[:, :, -1]

                    # Project Points & Sort
                    scores_proj = run_fused_score(
                        X_bad, mus[bad_idx_local], dense_labels_bad, M_split, v
                    )
                    sort_order = xp.lexsort(xp.stack([scores_proj, dense_labels_bad]))

                with CudaTimer("time_bad_d2h", profiling_stats, PROFILE_TIMING):
                    # Collect Sorted Data for Next Level
                    next_level_X_list.append(_to_host(X_bad[sort_order]))
                    next_level_F_list.append(_to_host(F_bad[sort_order]))
                    next_level_G_list.append(_to_host(G_bad[sort_order]))

                    # Calculate Child Sizes (Binary Split)
                    half = bad_counts // 2
                    other = bad_counts - half
                    new_sz = xp.zeros(2 * n_bad, dtype=xp.int64)  # (2 * n_bad,)
                    new_sz[0::2] = half
                    new_sz[1::2] = other
                    next_level_sizes_list.append(_to_host(new_sz))

                with CudaTimer("time_gc", profiling_stats, PROFILE_TIMING):
                    del (
                        M_split,
                        C,
                        scores_proj,
                        sort_order,
                        bad_mask,
                        X_bad,
                        F_bad,
                        G_bad,
                        dense_labels_bad,
                        bad_counts,
                        half,
                        other,
                    )

            if LOG_MEM_STATS:
                log_mem_stats(
                    f" L{level:02} B{b:02}/{num_batches:02}: After Bad Clusters"
                )

            start_cluster_idx = end_cluster_idx
            del X_batch, F_batch, G_batch
            if mempool is not None:
                mempool.free_all_blocks()

        # --- 3. Level Transition ---
        if len(next_level_sizes_list) == 0:
            logger.info(f"Level {level:02}: No bad clusters to refine, terminating.")
            break

        # Consolidate Next Level Data
        with CudaTimer("time_transition", profiling_stats, PROFILE_TIMING):
            next_X = np.concatenate(next_level_X_list)
            next_F = np.concatenate(next_level_F_list)
            next_G = np.concatenate(next_level_G_list)
            next_sizes = np.concatenate(next_level_sizes_list)

            # Upload to GPU (Handling potential zero-size clusters from splits)
            valid = next_sizes > 0
            if not np.all(valid):
                # Some clusters are empty after split, filter them out
                keep = np.repeat(valid, next_sizes)
                CLUSTER_PTS = xp.asarray(next_X[keep])
                CLUSTER_F = xp.asarray(next_F[keep])
                CLUSTER_G = xp.asarray(next_G[keep])
                CLUSTER_SIZES = xp.asarray(next_sizes[valid])
            else:
                # All clusters valid
                CLUSTER_PTS = xp.asarray(next_X)
                CLUSTER_F = xp.asarray(next_F)
                CLUSTER_G = xp.asarray(next_G)
                CLUSTER_SIZES = xp.asarray(next_sizes)

            CLUSTER_STARTS = xp.r_[0, xp.cumsum(CLUSTER_SIZES)[:-1]]

        # Update Level Stats with end-of-level info
        if profiling_stats["level_stats"]:
            profiling_stats["level_stats"][-1].update(
                {
                    "clusters_level_end": int(len(CLUSTER_SIZES)),
                    "good_clusters": int(K_good),
                    "num_pts_next_level": int(CLUSTER_PTS.shape[0]),
                }
            )

        if LOG_MEM_STATS:
            log_mem_stats(f"End: L{level:02} Transition")

        # Cleanup Host Buffers
        with CudaTimer("time_gc", profiling_stats, PROFILE_TIMING):
            del (
                next_level_X_list,
                next_level_F_list,
                next_level_G_list,
                next_level_sizes_list,
                next_X,
                next_F,
                next_G,
                next_sizes,
            )
            gc.collect()

        logger.info(
            f"Level {level:02}: Clusters (start): {num_clusters:,}, "
            f"Clusters (end): {len(CLUSTER_SIZES):,}, "
            f"Good Clusters: {K_good:,}, "
            f"Num Pts (next level): {CLUSTER_PTS.shape[0]:,} "
        )
        logger.info(
            f"Level {level:02}: Cluster Sizes - min: {int(CLUSTER_SIZES.min())}, "
            f"mean: {CLUSTER_SIZES.mean():.1f}, median: {int(xp.median(CLUSTER_SIZES))}, "
            f"max: {int(CLUSTER_SIZES.max())}"
        )
        logger.info(
            f"Runtime - Level {level:02}: {time.perf_counter() - _t_level:.3f}s"
        )
        level += 1

    _total_time = time.perf_counter() - _start_time
    profiling_stats["time_total"] = _total_time

    # --- 4. Finalization ---
    final_count = int(out_count)

    profiling_stats["num_levels"] = level
    profiling_stats["num_generated_pts"] = num_generated_pts
    profiling_stats["total_f_evals"] = total_f_evals
    profiling_stats["num_final_clusters"] = final_count

    out_dict = {k: buffers[k][:final_count] for k in buffers}
    out_dict["profiling_stats"] = profiling_stats
    if STORE_TREE_DATA:
        out_dict["tree_data"] = tree_data

    logger.info(f"Adaptive clustering complete: {final_count:,} final clusters.")
    logger.info(f"Runtime - Total: {_total_time:.3f}s")
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
    NUM_INITIAL_CLUSTERS = args.n_init_clusters
    MIN_PTS_PER_CLUSTER = args.n_min_per_cluster
    ERROR_METRIC = args.error_metric
    INITIAL_CLUSTERING_METHOD = args.initial_clustering_method
    A_ROSENBROCK = args.rosenbrock_a
    OUTPUT_SUBDIR = args.output_subdir
    FORCE_REEXEC = args.force

    # Check if file already exists
    AC_PLOTS_DIR = AC_PLOTS_DIR / OUTPUT_SUBDIR if OUTPUT_SUBDIR else AC_PLOTS_DIR
    AC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    img_fpath = (
        AC_PLOTS_DIR
        / f"adaptive_clustering_rosenbrock_{A_ROSENBROCK}_tol{TOL}_error_metric_{ERROR_METRIC}_init{NUM_INITIAL_CLUSTERS}_min{MIN_PTS_PER_CLUSTER}.png"
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
    STORE_TREE_DATA = True
    results = adaptive_clustering(
        X,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        TOL=TOL,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS * DIM,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        ERROR_METRIC=ERROR_METRIC,
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        LOG_MEM_STATS=True,
        STORE_TREE_DATA=STORE_TREE_DATA,
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

    if STORE_TREE_DATA:
        tree_data = results["tree_data"]
        print("\nTree Data Summary:")
        for lvl, data in tree_data.items():
            print(f"Level {lvl:02}: Clusters: {data['centroids'].shape[0]:,}")

    if DIM == 2 and len(results["centroids"]) <= 10_000:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Ellipse

        # 1 plot 2 subplots (1X2)
        # Subplot 1: scatter of final cluster centroids on contourf
        # Subplot 2: scatter of final cluster centroids with trust ellipsoids on contourf as well as circles corresponding to radii

        results = {k: _to_host(v) for k, v in results.items()}
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
            f"Rosenbrock (a={A_ROSENBROCK}) (TOL={TOL}, ERROR_METRIC={ERROR_METRIC}), N_CLUSTERS={len(centroids):,}",
            fontsize=11,
        )

        plt.subplots_adjust(top=0.88)
        plt.savefig(img_fpath, dpi=300)
        print(f"Saved figure to {img_fpath}")
    else:
        print("Skipping plot (d != 2 or too many clusters).")

    # Save npz file of results
    npz_fpath = img_fpath.with_suffix(".npz")
    results = {k: _to_host(v) for k, v in results.items()}
    xp.savez_compressed(npz_fpath, **results)
    xp.savez_compressed(npz_fpath, **results)
