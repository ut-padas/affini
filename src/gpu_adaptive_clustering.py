import logging
import time
from typing import Callable

import cupy as cp
import numpy as np
from cuml.cluster import KMeans as cuKMeans
from cupy import cumsum, einsum, empty, float64, int32, maximum
from cupy.linalg import norm
from cupyx import scatter_max

logger = logging.getLogger(__name__)


def get_cluster_starts(cluster_sizes: cp.ndarray) -> cp.ndarray:
    """
    Compute starting indices for each cluster based on their sizes.

    Args:
        cluster_sizes (cp.ndarray): Number of points in each cluster.

    Returns:
        cp.ndarray: Starting indices for each cluster, shape (num_clusters,).

    Example:
    >>> cluster_sizes = cp.array([3, 1, 0, 4, 2])
    >>> get_cluster_starts(cluster_sizes)
    >>> cp.array([0, 3, 4, 4, 8])
    """
    return cp.r_[0, cumsum(cluster_sizes)[:-1]]


def repeat_cluster_indices(cluster_sizes: cp.ndarray) -> cp.ndarray:
    """
    Generate an array of cluster indices repeated according to cluster sizes.
    Uses searchsorted to handle zero-size clusters correctly.

    Args:
        cluster_sizes (cp.ndarray): Number of points in each cluster.

    Returns:
        cp.ndarray: Array of cluster indices, shape (total_points,).

    Example:
    >>> cluster_sizes = cp.array([3, 1, 4, 2, 0, 2])
    >>> repeat_cluster_indices(cluster_sizes)
    >>> cp.array([0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 5, 5])
    """
    total_pts = int(cp.sum(cluster_sizes))
    if total_pts == 0:
        return cp.array([], dtype=int32)

    # Prefix sum gives the upper boundaries of each segment
    cum_offsets = cumsum(cluster_sizes)

    # Search where each index falls in the bins
    cluster_idxs = cp.searchsorted(cum_offsets, cp.arange(total_pts), side="right")
    return cluster_idxs.astype(int32)


def compute_leading_eigenvectors(
    pts: cp.ndarray,
    cluster_starts: cp.ndarray,
    cluster_sizes: cp.ndarray,
    num_clusters: int,
    rng: cp.random.Generator,
    POWER_ITERS: int,
    EPS: float = 1e-9,
    DTYPE=float64,
    TILE_SIZE: int = 5_000_000,
) -> cp.ndarray:
    """
    Compute leading eigenvectors for multiple clusters using power iteration.

    Args:
        pts (cp.ndarray): Flattened points for all clusters.
        cluster_starts (cp.ndarray): Start indices for each cluster in points.
        cluster_sizes (cp.ndarray): Number of points in each cluster.
        num_clusters (int): Number of clusters.
        rng (cp.random.Generator): Random number generator for initialization.
        POWER_ITERS (int): Number of power iteration steps.
        EPS (float): Small value to avoid division by zero.
        DTYPE (cp.dtype): Data type for computations.
        TILE_SIZE (int): Number of points to process per tile to avoid OOM.

    Returns:
        cp.ndarray: Leading eigenvectors, shape (num_clusters, D).
    """
    # Compute cluster means
    pt_sums = cp.add.reduceat(pts, cluster_starts)
    cluster_sizes_safe = maximum(cluster_sizes[:, None], 1)
    centroids = pt_sums / cluster_sizes_safe
    del pt_sums

    # Initialize random eigenvectors
    eigenvectors = rng.standard_normal((num_clusters, pts.shape[1]), dtype=DTYPE)
    eigenvectors /= norm(eigenvectors, axis=1, keepdims=True)

    # Broadcast cluster indices
    cluster_idxs = repeat_cluster_indices(cluster_sizes)
    total_pts = pts.shape[0]

    # Power iteration loop
    for _ in range(POWER_ITERS):
        next_eigenvectors = cp.zeros_like(eigenvectors)

        # Tile-based processing to manage memory usage
        for i in range(0, total_pts, TILE_SIZE):
            end = min(i + TILE_SIZE, total_pts)

            # Slices for this tile
            pts_tile = pts[i:end]
            idx_tile = cluster_idxs[i:end]

            # Broadcast inputs
            means_tile = centroids[idx_tile]
            eig_tile = eigenvectors[idx_tile]

            # Center points and project onto eigenvectors
            centered_pts = pts_tile - means_tile
            proj = einsum("ij,ij->i", centered_pts, eig_tile)
            scaled = centered_pts * proj[:, None]

            # Accumulate contributions
            cp.add.at(next_eigenvectors, idx_tile, scaled)

            del pts_tile, idx_tile, means_tile, eig_tile, centered_pts, proj, scaled

        # Normalize safely to avoid division by zero
        eigenvector_norms = norm(next_eigenvectors, axis=1, keepdims=True)
        safe_norms = maximum(eigenvector_norms, EPS)
        eigenvectors = next_eigenvectors / safe_norms

    return eigenvectors


def split_clusters_along_eigenvectors(
    pts: cp.ndarray,
    cluster_sizes: cp.ndarray,
    directions: cp.ndarray,
):
    """Split clusters by projecting along eigenvectors and sorting.

    Args:
        pts (cp.ndarray): Flattened points for all clusters.
        cluster_sizes (cp.ndarray): Number of points in each cluster.
        directions (cp.ndarray): Eigenvectors for splitting, shape (num_clusters, d).

    Returns:
        tuple: (sorted_pts, left_sizes, right_sizes)
    """
    # Project all points onto directions
    cluster_idxs = repeat_cluster_indices(cluster_sizes)
    directions_expanded = directions[cluster_idxs]
    projection_scores = einsum("ij,ij->i", pts, directions_expanded)

    # Perform Lexicographical Sort: (Cluster Id, Projection Score)
    # Goal: Group points by Cluster ID, and order them by Score within the cluster.
    # Strategy: "Radix Sort" logic — sort by the secondary key first, then by the
    # primary key.
    idxs_sorted_by_score = cp.argsort(projection_scores)
    cluster_ids_ordered_by_score = cluster_idxs[idxs_sorted_by_score]
    idxs_grouping_clusters = cp.argsort(cluster_ids_ordered_by_score)
    final_order = idxs_sorted_by_score[idxs_grouping_clusters]
    sorted_pts = pts[final_order]

    # Calculate split points (Median Split)
    count_left_children = cluster_sizes // 2
    count_right_children = cluster_sizes - count_left_children
    return sorted_pts, count_left_children, count_right_children


def adaptive_cluster_algorithm_gpu(
    X_train: cp.ndarray,
    f: Callable,
    grad: Callable,
    *,
    DATA_DIM: int,
    F_DIM: int,
    NUM_INITIAL_CLUSTERS: int,
    MIN_PTS_PER_CLUSTER: int,
    ATOL: float,
    RTOL: float,
    MAX_DEPTH: int = 500,
    MAX_CLUSTERS_EST: int = 100_000,
    POWER_ITERS: int = 3,
    SEED: int = 0,
    DTYPE=float64,
    BATCH_SIZE: int = 10_000_000,
) -> tuple:
    """Run fully GPU-based adaptive clustering.

    Args:
        X_train (cp.ndarray): Training data, shape (N, D), with N points of D dimensions.
        f (Callable): Function to evaluate at cluster centroids. Should return (N, FD), when called with (N, D). Here, FD is the function output dimension.
        grad (Callable): Gradient of the function f. Should return (N, FD, D), when called with (N, D).
        DATA_DIM (int): Dimension of the input data D.
        F_DIM (int): Dimension of the function output FD.
        NUM_INITIAL_CLUSTERS (int): Number of initial clusters.
        MIN_PTS_PER_CLUSTER (int): Minimum points per cluster during augmentation.
        ATOL (float): Absolute tolerance for convergence.
        RTOL (float): Relative tolerance for convergence.
        MAX_DEPTH (int): Maximum depth for BFS splitting.
        MAX_CLUSTERS_EST (int): Estimated maximum number of clusters for pre-allocation.
        POWER_ITERS (int): Number of power iterations for eigenvector computation.
        SEED (int): Random seed for reproducibility.
        DTYPE (cp.dtype): Data type for computations.
        BATCH_SIZE (int): Max points to process per batch loop to prevent OOM.

    Returns:
        tuple:
            - cp.ndarray: Cluster centroids, shape (M, D).
            - cp.ndarray: Function values at centroids, shape (M, FD).
            - cp.ndarray: Jacobians at centroids, shape (M, FD, D).
            - cp.ndarray: Hessian proxies (2 * max_error / radius^2), shape (M,).
            - cp.ndarray: Radii of clusters, shape (M,).
            - cp.ndarray: Max errors per cluster, shape (M,).
            - cp.ndarray: Mean errors per cluster, shape (M,).
            - cp.ndarray: Sizes of clusters, shape (M,).
            - cp.ndarray: Depths of clusters, shape (M,).
    """
    cp.random.seed(SEED)
    rng = cp.random.default_rng(SEED)
    assert rng is not None, "RNG initialization failed."

    # Get default memory pool for explicit cleanup
    mempool = cp.get_default_memory_pool()

    start_time = time.perf_counter()
    logger.info(f"Starting GPU Clustering on {X_train.shape}")

    # Initial clustering
    start_time_init = time.perf_counter()
    kmeans = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=0)
    init_labels = kmeans.fit_predict(X_train)
    elapsed_init = time.perf_counter() - start_time_init
    logger.info(f"Initial KMeans done in {elapsed_init:.4f}s")

    # Sort data by label
    sort_idxs = cp.argsort(init_labels)
    CLUSTER_PTS = X_train[sort_idxs]

    _, CLUSTER_SIZES = cp.unique(init_labels[sort_idxs], return_counts=True)  # type: ignore
    CLUSTER_STARTS = get_cluster_starts(CLUSTER_SIZES)

    # Storage buffers
    final_idx = 0
    final_cluster_centroids = empty((MAX_CLUSTERS_EST, DATA_DIM), dtype=DTYPE)
    final_cluster_F = empty((MAX_CLUSTERS_EST, F_DIM), dtype=DTYPE)
    final_cluster_J = empty((MAX_CLUSTERS_EST, F_DIM, DATA_DIM), dtype=DTYPE)
    final_cluster_max_err = empty(MAX_CLUSTERS_EST, dtype=DTYPE)
    final_cluster_mean_err = empty(MAX_CLUSTERS_EST, dtype=DTYPE)
    final_cluster_radii = empty(MAX_CLUSTERS_EST, dtype=DTYPE)
    final_cluster_sizes = empty(MAX_CLUSTERS_EST, dtype=int32)
    final_cluster_depths = empty(MAX_CLUSTERS_EST, dtype=int32)

    # Breadth-First Clustering Loop
    current_depth = 0
    while len(CLUSTER_SIZES) > 0 and current_depth < MAX_DEPTH:
        num_clusters = len(CLUSTER_SIZES)
        total_pts_level = int(cp.sum(CLUSTER_SIZES))

        msg = f"L{current_depth}: {num_clusters} clusters, {total_pts_level} pts."
        logger.debug(msg)

        # Augment clusters with synthetic points
        # Compute expected per-cluster sizes to meet minimum points requirement
        level_augmented_sizes = maximum(CLUSTER_SIZES, MIN_PTS_PER_CLUSTER)
        level_total_aug_pts = int(cp.sum(level_augmented_sizes))

        # Calculate number of batches required to process all augmented points
        num_batches = (level_total_aug_pts + BATCH_SIZE - 1) // BATCH_SIZE

        # Precompute cumulative sums as host array for slice indexing
        aug_cumsum_host = cp.asnumpy(cumsum(level_augmented_sizes))

        start_cluster_idx = 0
        total_bad_clusters_level = 0

        # Buffers for next level
        next_level_pts_list = []
        next_level_sizes_list = []

        # Process batches
        for b in range(num_batches):
            # Determine how many augmented points to process in this batch
            target_aug_count = (b + 1) * BATCH_SIZE

            # Calculate how many clusters fit in this batch
            end_cluster_idx = int(np.searchsorted(aug_cumsum_host, target_aug_count))
            end_cluster_idx = min(end_cluster_idx + 1, num_clusters)

            # Break if no more clusters to process
            if start_cluster_idx >= end_cluster_idx:
                break

            # Slice clusters for this batch
            sizes = CLUSTER_SIZES[start_cluster_idx:end_cluster_idx]
            starts = CLUSTER_STARTS[start_cluster_idx:end_cluster_idx]

            # Map global point indices to local batch
            slice_start = int(starts[0])
            slice_end = int(starts[-1] + sizes[-1])
            pts = CLUSTER_PTS[slice_start:slice_end]
            starts_local = starts - slice_start

            # Augment clusters in this batch
            aug_sizes = level_augmented_sizes[start_cluster_idx:end_cluster_idx]
            aug_total = int(cp.sum(aug_sizes))

            aug_pts = empty((aug_total, DATA_DIM), dtype=DTYPE)
            aug_ids = repeat_cluster_indices(aug_sizes)
            aug_starts = get_cluster_starts(aug_sizes)

            # Mask for real vs synthetic points
            pt_idxs_in_cluster = cp.arange(aug_total, dtype=int32) - aug_starts[aug_ids]
            is_real_pt = pt_idxs_in_cluster < sizes[aug_ids]

            # Fill real points
            aug_pts[is_real_pt] = pts

            # Identify synthetic points
            is_synth_pt = ~is_real_pt
            num_synth_pts = int(cp.sum(is_synth_pt))

            # Fill synthetic points if required
            if num_synth_pts > 0:
                synth_cluster_ids = aug_ids[is_synth_pt]
                _starts_expanded = starts_local[synth_cluster_ids]
                _sizes_expanded = sizes[synth_cluster_ids]

                random_indices_0 = cp.random.rand(num_synth_pts, dtype=DTYPE)
                pt_idx_0 = (random_indices_0 * _sizes_expanded).astype(int32)

                random_idxs_1 = cp.random.rand(num_synth_pts, dtype=DTYPE)
                offset = (random_idxs_1 * (_sizes_expanded - 1)).astype(int32) + 1
                pt_idx_1 = (pt_idx_0 + offset) % _sizes_expanded

                pt_0 = pts[_starts_expanded + pt_idx_0]
                pt_1 = pts[_starts_expanded + pt_idx_1]

                interpolation_factor = cp.random.rand(num_synth_pts, 1, dtype=DTYPE)

                # Use views to avoid extra allocations
                view = aug_pts[is_synth_pt]

                # Compute synthetic points
                cp.multiply(pt_0, (1.0 - interpolation_factor), out=view)
                view += pt_1 * interpolation_factor
                # Fill synthetic points
                aug_pts[is_synth_pt] = view
            del pts

            # Validate clusters with Linear Approximation (Taylor Expansion)
            # Compute the centroids of augmented clusters
            pt_sums = cp.add.reduceat(aug_pts, aug_starts)
            cluster_centroids = pt_sums / aug_sizes[:, None]

            # Evaluate function and gradient at centroids
            F_at_centroids = f(cluster_centroids)
            J_at_centroids = grad(cluster_centroids)

            F_expanded = F_at_centroids[aug_ids]
            J_expanded = J_at_centroids[aug_ids]
            centroids_expanded = cluster_centroids[aug_ids]

            # Linear approximation: F_taylor = F(c) + J(c) * (x - c)
            dx_centroid = aug_pts - centroids_expanded
            linear_approximation = einsum("ijk,ik->ij", J_expanded, dx_centroid)
            F_taylor = F_expanded + linear_approximation
            F_true = f(aug_pts)

            # Compute radii of clusters
            dist_from_centroid = norm(dx_centroid, axis=1)
            cluster_radii = cp.zeros(len(aug_sizes), dtype=DTYPE)
            scatter_max(cluster_radii, aug_ids, dist_from_centroid)

            # Compute the error threshold for each point
            approximation_error = norm(F_true - F_taylor, axis=1)
            error_threshold = maximum(ATOL, RTOL * norm(F_true, axis=1))

            # Compute the max and mean errors per cluster
            cluster_sum_errors = cp.add.reduceat(approximation_error, aug_starts)
            cluster_max_errors = cp.zeros(len(aug_sizes), dtype=DTYPE)
            scatter_max(cluster_max_errors, aug_ids, approximation_error)
            cluster_mean_errors = cluster_sum_errors / aug_sizes

            # Identify violating clusters by counting points exceeding error threshold per
            # cluster
            is_violating_pt = (approximation_error > error_threshold).astype(int32)
            violating_counts_per_cluster = cp.add.reduceat(is_violating_pt, aug_starts)
            has_violating_pts = violating_counts_per_cluster > 0
            is_converged = ~has_violating_pts

            # Update bad cluster count
            num_violating_clusters = int(cp.sum(has_violating_pts))
            total_bad_clusters_level += num_violating_clusters

            # Store converged clusters
            num_converged_clusters = int(cp.sum(is_converged))
            if num_converged_clusters > 0:
                converged_centroids = cluster_centroids[is_converged]
                converged_F = F_at_centroids[is_converged]
                converged_J = J_at_centroids[is_converged]
                converged_max_err = cluster_max_errors[is_converged]
                converged_mean_err = cluster_mean_errors[is_converged]
                converged_radii = cluster_radii[is_converged]
                converged_sizes = aug_sizes[is_converged]

                # Calculate next storage index
                next_idx = final_idx + num_converged_clusters

                # Store data if within bounds
                if next_idx <= MAX_CLUSTERS_EST:
                    final_cluster_centroids[final_idx:next_idx] = converged_centroids
                    final_cluster_F[final_idx:next_idx] = converged_F
                    final_cluster_J[final_idx:next_idx] = converged_J
                    final_cluster_max_err[final_idx:next_idx] = converged_max_err
                    final_cluster_mean_err[final_idx:next_idx] = converged_mean_err
                    final_cluster_radii[final_idx:next_idx] = converged_radii
                    final_cluster_sizes[final_idx:next_idx] = converged_sizes
                    final_cluster_depths[final_idx:next_idx] = current_depth
                    final_idx = next_idx
                else:
                    msg = f"Storage overflow. Clusters: {next_idx}/{MAX_CLUSTERS_EST}. Increase MAX_CLUSTERS_EST."
                    raise MemoryError(msg)

            # Split violating clusters
            if num_violating_clusters > 0:
                pts_in_violating_clusters = has_violating_pts[aug_ids]
                violating_pts = aug_pts[pts_in_violating_clusters]

                violating_cluster_sizes = aug_sizes[has_violating_pts]
                violating_cluster_starts = get_cluster_starts(violating_cluster_sizes)

                # Compute eigenvectors
                eigenvector_directions = compute_leading_eigenvectors(
                    pts=violating_pts,
                    cluster_starts=violating_cluster_starts,
                    cluster_sizes=violating_cluster_sizes,
                    num_clusters=num_violating_clusters,
                    POWER_ITERS=POWER_ITERS,
                    rng=rng,
                    DTYPE=DTYPE,
                )

                # Split along eigenvectors
                split_pts, left_sizes, right_sizes = split_clusters_along_eigenvectors(
                    pts=violating_pts,
                    cluster_sizes=violating_cluster_sizes,
                    directions=eigenvector_directions,
                )

                # Interleave left and right sizes: [L0, R0, L1, R1, ...]
                next_cluster_sizes = empty(2 * num_violating_clusters, dtype=int32)
                next_cluster_sizes[0::2] = left_sizes
                next_cluster_sizes[1::2] = right_sizes

                # Accumulate for next level
                next_level_pts_list.append(split_pts)
                next_level_sizes_list.append(next_cluster_sizes)

            # Move to next batch
            start_cluster_idx = end_cluster_idx

            # Explicit Cleanup for Batch
            del aug_pts, cluster_centroids, F_at_centroids, J_at_centroids
            del F_expanded, J_expanded, centroids_expanded
            del dx_centroid, linear_approximation, F_taylor, F_true
            mempool.free_all_blocks()

        logger.debug(f"Bad: {total_bad_clusters_level}")

        # Termination condition
        if total_bad_clusters_level == 0:
            break

        # Prepare for next iteration of BFS
        if len(next_level_pts_list) > 0:
            CLUSTER_PTS = cp.concatenate(next_level_pts_list)

            # Filter non-empty clusters
            next_sizes_all = cp.concatenate(next_level_sizes_list)
            mask_nonzero = next_sizes_all > 0
            if not cp.all(mask_nonzero):
                CLUSTER_SIZES = next_sizes_all[mask_nonzero]
            else:
                CLUSTER_SIZES = next_sizes_all

            CLUSTER_STARTS = get_cluster_starts(CLUSTER_SIZES)
        else:
            break

        current_depth += 1

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Finished in {elapsed_time:.4f}s")
    logger.info(f"Final Cluster Count: {final_idx}")

    # Slice metadata to actual size
    _centroids = final_cluster_centroids[:final_idx]
    _F = final_cluster_F[:final_idx]
    _J = final_cluster_J[:final_idx]
    _max_err = final_cluster_max_err[:final_idx]
    _mean_err = final_cluster_mean_err[:final_idx]
    _radii = final_cluster_radii[:final_idx]
    _sizes = final_cluster_sizes[:final_idx]
    _depths = final_cluster_depths[:final_idx]

    _hessian_proxies = cp.where(_radii > 0, 2.0 * _max_err / (_radii**2), 0.0)
    return (
        _centroids,
        _F,
        _J,
        _hessian_proxies,
        _radii,
        _max_err,
        _mean_err,
        _sizes,
        _depths,
    )


if __name__ == "__main__":
    from src.logging_config import setup_logging
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging()

    # Configuration
    cp.random.seed(0)
    rng = cp.random.default_rng(0)

    N_SAMPLES = 100_000
    N_INIT_CLUSTERS = 100
    MIN_PTS_PER_CLUSTER = 100
    ATOL = 1e-2
    RTOL = 1e-2
    DATA_DIM = 3
    F_DIM = 1

    MAX_DEPTH = 50
    MAX_CLUSTERS_EST = 100_000_000
    POWER_ITERS = 4
    DTYPE = float64
    BATCH_SIZE = 10_000_000

    print(f"Generating {N_SAMPLES:,} points on GPU...")
    print(f"{N_INIT_CLUSTERS=}, {MIN_PTS_PER_CLUSTER=}")
    print(f"{ATOL=}, {RTOL=}")
    X_train = cp.random.rand(N_SAMPLES, DATA_DIM, dtype=DTYPE)
    result = adaptive_cluster_algorithm_gpu(
        X_train=X_train,
        f=rosenbrock,
        grad=rosenbrock_grad,
        DATA_DIM=DATA_DIM,
        F_DIM=F_DIM,
        NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        ATOL=ATOL,
        RTOL=RTOL,
        MAX_DEPTH=MAX_DEPTH,
        MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
        POWER_ITERS=POWER_ITERS,
        DTYPE=DTYPE,
        BATCH_SIZE=BATCH_SIZE,
    )

    print(f"Final Cluster Count: {len(result[0])}")
