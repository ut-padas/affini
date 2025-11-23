import time
from typing import Callable

import cupy as cp
import cupyx
from cuml.cluster import KMeans as cuKMeans


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
    total_points = int(cp.sum(cluster_sizes))
    if total_points == 0:
        return cp.array([], dtype=cp.int32)

    # Prefix sum gives the upper boundaries of each segment
    cumulative_offsets = cp.cumsum(cluster_sizes)

    # Search where each index falls in the bins
    cluster_indices = cp.searchsorted(
        cumulative_offsets, cp.arange(total_points), side="right"
    )

    return cluster_indices.astype(cp.int32)


def compute_leading_eigenvectors(
    points: cp.ndarray,
    cluster_starts: cp.ndarray,
    cluster_sizes: cp.ndarray,
    num_clusters: int,
    rng: cp.random.Generator,
    POWER_ITERS: int,
    EPS: float = 1e-9,
    DTYPE=cp.float64,
    CHUNK_SIZE: int = 1_000_000,
) -> cp.ndarray:
    """
    Compute leading eigenvectors for multiple clusters using power iteration.

    Args:
        points (cp.ndarray): Flattened points for all clusters.
        cluster_starts (cp.ndarray): Start indices for each cluster in points.
        cluster_sizes (cp.ndarray): Number of points in each cluster.
        num_clusters (int): Number of clusters.
        rng (cp.random.Generator): Random number generator for initialization.
        POWER_ITERS (int): Number of power iteration steps.
        EPS (float): Small value to avoid division by zero.
        DTYPE (cp.dtype): Data type for computations.
        CHUNK_SIZE (int): Number of points to process in each chunk to reduce memory usage.

    Returns:
        cp.ndarray: Leading eigenvectors, shape (num_clusters, D).
    """
    # Compute cluster means
    point_sums = cp.add.reduceat(points, cluster_starts)
    cluster_sizes_col = cp.maximum(cluster_sizes[:, None], 1)
    centroids = point_sums / cluster_sizes_col

    # Initialize random eigenvectors
    eigenvectors = rng.standard_normal((num_clusters, points.shape[1]), dtype=DTYPE)
    eigenvectors /= cp.linalg.norm(eigenvectors, axis=1, keepdims=True)

    # Broadcast cluster indices
    cluster_indices = repeat_cluster_indices(cluster_sizes)

    n_points = len(points)
    n_dims = points.shape[1]

    # Power iteration loop with chunked processing to reduce memory usage
    for _ in range(POWER_ITERS):
        # Allocate output array for accumulating scaled vectors
        next_eigenvectors = cp.zeros((num_clusters, n_dims), dtype=DTYPE)

        # Determine if we need chunking based on data size
        # For datasets where points * dims * 8 bytes * 5 (temp arrays) > available memory
        # we process in chunks. Otherwise, use the original fast path.
        estimated_memory_gb = (n_points * n_dims * 8 * 5) / 1e9

        if estimated_memory_gb < 10 or n_points <= CHUNK_SIZE:
            # Fast path: process all at once (original algorithm)
            centroids_expanded = centroids[cluster_indices]
            eigenvectors_expanded = eigenvectors[cluster_indices]

            # Center points by centroids
            centered_points = points - centroids_expanded

            # Project onto eigenvectors - einsum for batch dot product
            projections = cp.einsum("ij,ij->i", centered_points, eigenvectors_expanded)

            # Scale vectors - broadcasting projections
            scaled_vectors = centered_points * projections[:, None]

            # Reduce back to clusters using reduceat
            next_eigenvectors = cp.add.reduceat(scaled_vectors, cluster_starts)

            # Clean up
            del centroids_expanded, eigenvectors_expanded, centered_points
            del projections, scaled_vectors

        else:
            # Memory-efficient path: process in chunks
            n_chunks = (n_points + CHUNK_SIZE - 1) // CHUNK_SIZE

            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = min((chunk_idx + 1) * CHUNK_SIZE, n_points)

                # Get chunk of data
                points_chunk = points[start_idx:end_idx]
                cluster_indices_chunk = cluster_indices[start_idx:end_idx]

                # Expand centroids and eigenvectors for this chunk
                centroids_chunk = centroids[cluster_indices_chunk]
                eigenvectors_chunk = eigenvectors[cluster_indices_chunk]

                # Center points by centroids
                centered_chunk = points_chunk - centroids_chunk

                # Project onto eigenvectors
                projections_chunk = cp.einsum(
                    "ij,ij->i", centered_chunk, eigenvectors_chunk
                )

                # Scale vectors
                scaled_chunk = centered_chunk * projections_chunk[:, None]

                # Accumulate into clusters using vectorized scatter-add
                # Flatten and use cp.add.at for efficient accumulation
                flat_indices = (
                    cluster_indices_chunk[:, None] * n_dims
                    + cp.arange(n_dims, dtype=cp.int32)[None, :]
                ).ravel()
                cp.add.at(next_eigenvectors.ravel(), flat_indices, scaled_chunk.ravel())

                # Free chunk memory
                del (
                    points_chunk,
                    cluster_indices_chunk,
                    centroids_chunk,
                    eigenvectors_chunk,
                    centered_chunk,
                    projections_chunk,
                    scaled_chunk,
                    flat_indices,
                )

        # Normalize safely to avoid division by zero
        eigenvector_norms = cp.linalg.norm(next_eigenvectors, axis=1, keepdims=True)
        safe_norms = cp.maximum(eigenvector_norms, EPS)
        eigenvectors = next_eigenvectors / safe_norms

    return eigenvectors


def split_clusters_along_eigenvectors(
    points: cp.ndarray,
    cluster_sizes: cp.ndarray,
    directions: cp.ndarray,
):
    """Split clusters by projecting along eigenvectors and sorting.

    Args:
        points (cp.ndarray): Flattened points for all clusters.
        cluster_sizes (cp.ndarray): Number of points in each cluster.
        directions (cp.ndarray): Eigenvectors for splitting, shape (num_clusters, d).

    Returns:
        tuple: (sorted_points, left_sizes, right_sizes)
    """
    # Project all points onto directions
    cluster_indices = repeat_cluster_indices(cluster_sizes)
    directions_expanded = directions[cluster_indices]
    projection_scores = cp.einsum("ij,ij->i", points, directions_expanded)

    # Perform Lexicographical Sort: (Cluster Id, Projection Score)
    # Goal: Group points by Cluster ID, and order them by Score within the cluster.
    # Strategy: "Radix Sort" logic — sort by the secondary key first, then by the
    # primary key.
    indices_sorted_by_score = cp.argsort(projection_scores)
    cluster_ids_ordered_by_score = cluster_indices[indices_sorted_by_score]
    indices_grouping_clusters = cp.argsort(cluster_ids_ordered_by_score)
    final_lexicographical_order = indices_sorted_by_score[indices_grouping_clusters]
    sorted_points = points[final_lexicographical_order]

    # Calculate split points (Median Split)
    count_left_children = cluster_sizes // 2
    count_right_children = cluster_sizes - count_left_children

    return sorted_points, count_left_children, count_right_children


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
    DEBUG: bool = False,
    SEED: int = 0,
    DTYPE=cp.float64,
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
        DEBUG (bool): Whether to print debug information.
        SEED (int): Random seed for reproducibility.
        DTYPE (cp.dtype): Data type for computations.

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

    print(f"--- Starting GPU Clustering on {X_train.shape} ---")
    start_time = time.time()

    # Initial clustering
    kmeans = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=0)
    init_labels = kmeans.fit_predict(X_train)

    # Sort data by label
    sort_indices = cp.argsort(init_labels)
    current_points = X_train[sort_indices]

    _, init_cluster_sizes = cp.unique(init_labels[sort_indices], return_counts=True)  # type: ignore
    init_cluster_starts = cp.r_[0, cp.cumsum(init_cluster_sizes)[:-1]]

    # Storage buffers
    final_cluster_centroids = cp.empty((MAX_CLUSTERS_EST, DATA_DIM), dtype=DTYPE)
    final_cluster_F = cp.empty((MAX_CLUSTERS_EST, F_DIM), dtype=DTYPE)
    final_cluster_J = cp.empty((MAX_CLUSTERS_EST, F_DIM, DATA_DIM), dtype=DTYPE)
    final_cluster_max_errors = cp.empty(MAX_CLUSTERS_EST, dtype=DTYPE)
    final_cluster_mean_errors = cp.empty(MAX_CLUSTERS_EST, dtype=DTYPE)
    final_cluster_radii = cp.empty(MAX_CLUSTERS_EST, dtype=DTYPE)
    final_cluster_sizes = cp.empty(MAX_CLUSTERS_EST, dtype=cp.int32)
    final_cluster_depths = cp.empty(MAX_CLUSTERS_EST, dtype=cp.int32)

    metadata_index = 0

    # Breadth-First Clustering Loop
    current_depth = 0
    while len(init_cluster_sizes) > 0 and current_depth < MAX_DEPTH:
        num_clusters = len(init_cluster_sizes)
        if DEBUG:
            msg = f"L{current_depth}: {num_clusters} clusters, {int(cp.sum(init_cluster_sizes))} pts. "
            print(msg, end="")

        # Augment clusters with synthetic points
        # Compute expected per-cluster sizes to meet minimum points requirement
        augmented_sizes = cp.maximum(init_cluster_sizes, MIN_PTS_PER_CLUSTER)
        total_augmented_points = int(cp.sum(augmented_sizes))

        augmented_points = cp.empty((total_augmented_points, DATA_DIM), dtype=DTYPE)

        augmented_cluster_ids = repeat_cluster_indices(augmented_sizes)
        augmented_starts = cp.r_[0, cp.cumsum(augmented_sizes)[:-1]]

        # Mask for real vs synthetic points
        point_indices_in_cluster = (
            cp.arange(total_augmented_points, dtype=cp.int32)
            - augmented_starts[augmented_cluster_ids]
        )
        real_sizes_expanded = init_cluster_sizes[augmented_cluster_ids]
        is_real_point = point_indices_in_cluster < real_sizes_expanded

        # Fill real points
        augmented_points[is_real_point] = current_points

        # Identify synthetic points
        is_synthetic_point = ~is_real_point
        num_synthetic_points = int(cp.sum(is_synthetic_point))

        # Fill synthetic points if required
        if num_synthetic_points > 0:
            synthetic_cluster_ids = augmented_cluster_ids[is_synthetic_point]

            current_starts_expanded = init_cluster_starts[synthetic_cluster_ids]
            current_sizes_expanded = init_cluster_sizes[synthetic_cluster_ids]

            random_indices_0 = cp.random.rand(num_synthetic_points, dtype=DTYPE)
            point_index_0 = (random_indices_0 * current_sizes_expanded).astype(cp.int32)

            random_indices_1 = cp.random.rand(num_synthetic_points, dtype=DTYPE)
            offset = (random_indices_1 * (current_sizes_expanded - 1)).astype(
                cp.int32
            ) + 1
            point_index_1 = (point_index_0 + offset) % current_sizes_expanded

            point_0 = current_points[current_starts_expanded + point_index_0]
            point_1 = current_points[current_starts_expanded + point_index_1]

            interpolation_factor = cp.random.rand(num_synthetic_points, 1, dtype=DTYPE)

            # Create view to avoid extra allocation
            synthetic_view = augmented_points[is_synthetic_point]

            # Compute synthetic points
            cp.multiply(point_0, (1.0 - interpolation_factor), out=synthetic_view)
            synthetic_view += point_1 * interpolation_factor

            # Fill synthetic points
            augmented_points[is_synthetic_point] = synthetic_view

        # Validate clusters with Linear Approximation (Taylor Expansion)
        point_sums = cp.add.reduceat(augmented_points, augmented_starts)
        cluster_centroids = point_sums / augmented_sizes[:, None]

        # Evaluate function and gradient at centroids
        F_at_centroids = f(cluster_centroids)
        J_at_centroids = grad(cluster_centroids)

        F_expanded = F_at_centroids[augmented_cluster_ids]
        J_expanded = J_at_centroids[augmented_cluster_ids]
        centroids_expanded = cluster_centroids[augmented_cluster_ids]

        # Linear approximation: F_taylor = F(c) + J(c) * (x - c)
        deviation_from_centroid = augmented_points - centroids_expanded
        linear_approximation = cp.einsum(
            "ijk,ik->ij", J_expanded, deviation_from_centroid
        )
        F_taylor = F_expanded + linear_approximation
        F_true = f(augmented_points)

        # Compute radii of clusters
        dist_from_centroid = cp.linalg.norm(deviation_from_centroid, axis=1)
        cluster_radii = cp.zeros(len(augmented_sizes), dtype=DTYPE)
        cupyx.scatter_max(cluster_radii, augmented_cluster_ids, dist_from_centroid)

        # Compute the error threshold for each point
        approximation_error = cp.linalg.norm(F_true - F_taylor, axis=1)
        error_threshold = cp.maximum(ATOL, RTOL * cp.linalg.norm(F_true, axis=1))

        # Compute the max and mean errors per cluster
        cluster_sum_errors = cp.add.reduceat(approximation_error, augmented_starts)
        cluster_max_errors = cp.zeros(len(augmented_sizes), dtype=DTYPE)
        cupyx.scatter_max(
            cluster_max_errors, augmented_cluster_ids, approximation_error
        )
        cluster_mean_errors = cluster_sum_errors / augmented_sizes

        # Identify violating clusters by counting points exceeding error threshold per
        # cluster
        is_violating_point = (approximation_error > error_threshold).astype(cp.int32)
        violating_counts_per_cluster = cp.add.reduceat(
            is_violating_point, augmented_starts
        )
        has_violating_points = violating_counts_per_cluster > 0
        is_converged = ~has_violating_points

        num_violating_clusters = int(cp.sum(has_violating_points))
        if DEBUG:
            print(f"Bad: {num_violating_clusters}")

        # Store converged clusters
        num_converged_clusters = int(cp.sum(is_converged))
        if num_converged_clusters > 0:
            converged_centroids = cluster_centroids[is_converged]
            converged_F = F_at_centroids[is_converged]
            converged_J = J_at_centroids[is_converged]
            converged_max_errors = cluster_max_errors[is_converged]
            converged_mean_errors = cluster_mean_errors[is_converged]
            converged_radii = cluster_radii[is_converged]
            converged_sizes = augmented_sizes[is_converged]

            # Calculate the potential new indices for both metadata and points
            next_meta_idx = metadata_index + num_converged_clusters

            # Store the converged cluster data if within bounds
            if next_meta_idx <= MAX_CLUSTERS_EST:
                # Store Metadata
                final_cluster_centroids[metadata_index:next_meta_idx] = (
                    converged_centroids
                )
                final_cluster_F[metadata_index:next_meta_idx] = converged_F
                final_cluster_J[metadata_index:next_meta_idx] = converged_J

                final_cluster_max_errors[metadata_index:next_meta_idx] = (
                    converged_max_errors
                )
                final_cluster_mean_errors[metadata_index:next_meta_idx] = (
                    converged_mean_errors
                )
                final_cluster_radii[metadata_index:next_meta_idx] = converged_radii

                final_cluster_sizes[metadata_index:next_meta_idx] = converged_sizes
                final_cluster_depths[metadata_index:next_meta_idx] = current_depth
                metadata_index = next_meta_idx  # Update the index

            else:
                msg = f"Storage overflow. Clusters: {next_meta_idx}/{MAX_CLUSTERS_EST}. Increase MAX_CLUSTERS_EST."
                raise MemoryError(msg)

        # Termination condition
        if num_violating_clusters == 0:
            break

        # Split violating clusters
        points_in_violating_clusters = has_violating_points[augmented_cluster_ids]
        violating_points = augmented_points[points_in_violating_clusters]

        violating_cluster_sizes = augmented_sizes[has_violating_points]
        violating_cluster_starts = cp.r_[0, cp.cumsum(violating_cluster_sizes)[:-1]]

        # Compute eigenvectors
        eigenvector_directions = compute_leading_eigenvectors(
            points=violating_points,
            cluster_starts=violating_cluster_starts,
            cluster_sizes=violating_cluster_sizes,
            num_clusters=num_violating_clusters,
            POWER_ITERS=POWER_ITERS,
            rng=rng,
            DTYPE=DTYPE,
        )

        # Split along eigenvectors
        split_points, left_sizes, right_sizes = split_clusters_along_eigenvectors(
            points=violating_points,
            cluster_sizes=violating_cluster_sizes,
            directions=eigenvector_directions,
        )

        # Interleave left and right sizes: [L0, R0, L1, R1, ...]
        next_cluster_sizes = cp.empty(2 * num_violating_clusters, dtype=cp.int32)
        next_cluster_sizes[0::2] = left_sizes
        next_cluster_sizes[1::2] = right_sizes

        # Filter non-empty clusters
        has_points = next_cluster_sizes > 0
        if not cp.all(has_points):
            # Some clusters are empty, filter them out
            init_cluster_sizes = next_cluster_sizes[has_points]
            current_points = split_points
        else:
            # All clusters have points
            init_cluster_sizes = next_cluster_sizes
            current_points = split_points

        # Prepare for next iteration of BFS
        init_cluster_starts = cp.r_[0, cp.cumsum(init_cluster_sizes)[:-1]]
        current_depth += 1

    if DEBUG:
        elapsed_time = time.time() - start_time
        msg = f"\n--- Finished in {elapsed_time:.4f}s ---\n"
        msg += f"\t> Final Cluster Count: {metadata_index}\n"
        print(f"\n--- Finished in {elapsed_time:.4f}s ---")

    # Slice metadata to actual size
    _centroids = final_cluster_centroids[:metadata_index]
    _F = final_cluster_F[:metadata_index]
    _J = final_cluster_J[:metadata_index]
    _max_errors = final_cluster_max_errors[:metadata_index]
    _mean_errors = final_cluster_mean_errors[:metadata_index]
    _radii = final_cluster_radii[:metadata_index]
    _sizes = final_cluster_sizes[:metadata_index]
    _depths = final_cluster_depths[:metadata_index]

    _hessian_proxies = cp.where(_radii > 0, 2.0 * _max_errors / (_radii**2), 0.0)
    return (
        _centroids,
        _F,
        _J,
        _hessian_proxies,
        _radii,
        _max_errors,
        _mean_errors,
        _sizes,
        _depths,
    )


def sample_function(X: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    A sample function f that returns F and J at given X.
    f(X) = [x1^2 + x2^2, x1 - x2 + x3]
    Args:
        X: (n, 3) data matrix
    Returns:
        F: (n, 2) function values at X
        J: (n, 2, 3) Jacobian at X
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    f1 = x1**2 + x2**2
    f2 = x1 - x2 + x3
    F = cp.stack([f1, f2], axis=1)
    return F


def sample_function_grad(X: cp.ndarray) -> cp.ndarray:
    """
    Gradient of the sample function.
    grad(X) = [[2*x1, 2*x2, 0],
               [1, -1, 1]]
    Args:
        X: (n, 3) data matrix
    Returns:
        J: (n, 2, 3) Jacobian at X
    """
    x1, x2, _ = X[:, 0], X[:, 1], X[:, 2]
    g1 = cp.stack([2 * x1, 2 * x2, cp.zeros_like(x1)], axis=1)
    g2 = cp.stack([cp.ones_like(x1), -cp.ones_like(x1), cp.ones_like(x1)], axis=1)
    J = cp.stack([g1, g2], axis=1)
    return J


if __name__ == "__main__":
    # Configuration
    cp.random.seed(0)
    rng = cp.random.default_rng(0)

    N_SAMPLES = 10_000
    N_INIT_CLUSTERS = 100
    MIN_PTS_PER_CLUSTER = 100
    ATOL = 0.01
    RTOL = 0.01

    MAX_DEPTH = 50
    MAX_CLUSTERS_EST = 100_000_000
    POWER_ITERS = 4
    DTYPE = cp.float64

    print(f"Generating {N_SAMPLES:,} points on GPU...")
    print(f"{N_INIT_CLUSTERS=}, {MIN_PTS_PER_CLUSTER=}")
    print(f"{ATOL=}, {RTOL=}")
    X_train = cp.random.rand(N_SAMPLES, 3, dtype=DTYPE)
    result = adaptive_cluster_algorithm_gpu(
        X_train=X_train,
        f=sample_function,
        grad=sample_function_grad,
        DATA_DIM=3,
        F_DIM=2,
        NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        ATOL=ATOL,
        RTOL=RTOL,
        MAX_DEPTH=MAX_DEPTH,
        MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
        POWER_ITERS=POWER_ITERS,
        DEBUG=True,
        DTYPE=DTYPE,
    )

    print(f"Final Cluster Count: {len(result[0])}")
