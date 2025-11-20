import time
from typing import Callable

import cupy as cp
from cuml.cluster import KMeans as cuKMeans

# Configuration
cp.random.seed(0)
rng = cp.random.default_rng(0)

# Precision
DTYPE = cp.float64


def repeat_cluster_indices(cluster_sizes: cp.ndarray) -> cp.ndarray:
    """Generate an array of cluster indices repeated according to cluster sizes.

    Uses searchsorted to handle zero-size clusters correctly.

    Args:
        cluster_sizes (cp.ndarray): Number of points in each cluster.

    Returns:
        cp.ndarray: Array of cluster indices, shape (total_points,).
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
    power_iters: int,
) -> cp.ndarray:
    """Compute leading eigenvectors for multiple clusters using power iteration.

    Args:
        points (cp.ndarray): Flattened points for all clusters.
        cluster_starts (cp.ndarray): Start indices for each cluster in points.
        cluster_sizes (cp.ndarray): Number of points in each cluster.
        num_clusters (int): Number of clusters.

    Returns:
        cp.ndarray: Leading eigenvectors, shape (num_clusters, d).
    """
    # Compute cluster means
    point_sums = cp.add.reduceat(points, cluster_starts)
    safe_cluster_sizes = cp.maximum(cluster_sizes[:, None], 1)
    cluster_means = point_sums / safe_cluster_sizes

    # Initialize random eigenvectors
    eigenvectors = rng.standard_normal(  # type: ignore
        (num_clusters, points.shape[1]), dtype=DTYPE
    )
    eigenvectors /= cp.linalg.norm(eigenvectors, axis=1, keepdims=True)

    # Broadcast cluster indices
    cluster_indices = repeat_cluster_indices(cluster_sizes)

    # Power iteration loop
    for _ in range(power_iters):
        # Broadcast mean and eigenvector to every point
        means_broadcasted = cluster_means[cluster_indices]
        eigenvectors_broadcasted = eigenvectors[cluster_indices]

        # Center points implicitly
        centered_points = points - means_broadcasted

        # Project onto eigenvectors
        projections = cp.einsum("ij,ij->i", centered_points, eigenvectors_broadcasted)

        # Scale vectors
        scaled_vectors = centered_points * projections[:, None]

        # Reduce back to clusters
        next_eigenvectors = cp.add.reduceat(scaled_vectors, cluster_starts)

        # Normalize safely
        eigenvector_norms = cp.linalg.norm(next_eigenvectors, axis=1, keepdims=True)
        safe_norms = cp.maximum(eigenvector_norms, 1e-9)
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
    directions_broadcasted = directions[cluster_indices]
    projection_scores = cp.einsum("ij,ij->i", points, directions_broadcasted)

    # Global sort by (cluster_id, projection)
    # Two-step argsort
    secondary_sort_permutation = cp.argsort(projection_scores)
    permuted_cluster_ids = cluster_indices[secondary_sort_permutation]
    primary_sort_permutation = cp.argsort(permuted_cluster_ids)
    final_sort_permutation = secondary_sort_permutation[primary_sort_permutation]

    # Reorder points
    sorted_points = points[final_sort_permutation]

    # Split sizes
    left_cluster_sizes = cluster_sizes // 2
    right_cluster_sizes = cluster_sizes - left_cluster_sizes

    return sorted_points, left_cluster_sizes, right_cluster_sizes


def adaptive_cluster_algorithm_gpu(
    X_train: cp.ndarray,
    f: Callable,
    grad: Callable,
    num_initial_clusters: int,
    min_pts_per_cluster: int,
    atol: float,
    rtol: float,
    max_depth: int = 500,
    max_clusters_est: int = 100_000_000,
    power_iters: int = 4,
    debug: bool = False,
) -> tuple:
    """Run fully GPU-based adaptive clustering.

    Args:
        data_points (cp.ndarray): Input data points.

    Returns:
        tuple: (converged_points, metadata) where metadata is (centroids, sizes, depths).
    """
    print(f"--- Starting GPU Clustering on {X_train.shape} ---")
    start_time = time.time()

    # Initial clustering
    kmeans = cuKMeans(n_clusters=num_initial_clusters, random_state=0)
    initial_labels = kmeans.fit_predict(X_train)

    # Sort data by label
    sort_indices = cp.argsort(initial_labels)
    current_points = X_train[sort_indices]

    unique_cluster_labels, initial_cluster_sizes = cp.unique(  # type: ignore
        initial_labels[sort_indices], return_counts=True
    )
    initial_cluster_starts = cp.r_[0, cp.cumsum(initial_cluster_sizes)[:-1]]

    # Storage buffers
    converged_points_store = cp.empty_like(X_train)
    store_index = 0

    # Metadata buffers
    cluster_centroids = cp.empty((max_clusters_est, 3), dtype=DTYPE)
    cluster_sizes = cp.empty(max_clusters_est, dtype=cp.int32)
    cluster_depths = cp.empty(max_clusters_est, dtype=cp.int32)
    metadata_index = 0

    current_depth = 0
    while len(initial_cluster_sizes) > 0 and current_depth < max_depth:
        num_clusters = len(initial_cluster_sizes)
        if debug:
            print(
                f"L{current_depth}: {num_clusters} clusters, {int(cp.sum(initial_cluster_sizes))} pts. ",
                end="",
            )

        # Augment clusters
        augmented_sizes = cp.maximum(initial_cluster_sizes, min_pts_per_cluster)
        total_augmented_points = int(cp.sum(augmented_sizes))

        augmented_points = cp.empty(
            (total_augmented_points, current_points.shape[1]), dtype=DTYPE
        )

        augmented_cluster_ids = repeat_cluster_indices(augmented_sizes)
        augmented_starts = cp.r_[0, cp.cumsum(augmented_sizes)[:-1]]

        # Mask for real vs synthetic points
        point_indices_in_cluster = (
            cp.arange(total_augmented_points, dtype=cp.int32)
            - augmented_starts[augmented_cluster_ids]
        )
        real_sizes_expanded = initial_cluster_sizes[augmented_cluster_ids]
        is_real_point = point_indices_in_cluster < real_sizes_expanded

        # Fill real points
        augmented_points[is_real_point] = current_points

        # Fill synthetic points
        is_synthetic_point = ~is_real_point
        num_synthetic_points = int(cp.sum(is_synthetic_point))

        if num_synthetic_points > 0:
            synthetic_cluster_ids = augmented_cluster_ids[is_synthetic_point]

            current_starts_expanded = initial_cluster_starts[synthetic_cluster_ids]
            current_sizes_expanded = initial_cluster_sizes[synthetic_cluster_ids]

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

            synthetic_view = augmented_points[is_synthetic_point]
            cp.multiply(point_0, (1.0 - interpolation_factor), out=synthetic_view)
            synthetic_view += point_1 * interpolation_factor
            augmented_points[is_synthetic_point] = synthetic_view

        # Validate clusters
        point_sums = cp.add.reduceat(augmented_points, augmented_starts)
        cluster_centroids_local = point_sums / augmented_sizes[:, None]

        F_at_centroids = f(cluster_centroids_local)
        J_at_centroids = grad(cluster_centroids_local)

        F_expanded = F_at_centroids[augmented_cluster_ids]
        J_expanded = J_at_centroids[augmented_cluster_ids]
        centroids_expanded = cluster_centroids_local[augmented_cluster_ids]

        deviation_from_centroid = augmented_points - centroids_expanded
        linear_approximation = cp.einsum(
            "ijk,ik->ij", J_expanded, deviation_from_centroid
        )
        F_taylor = F_expanded + linear_approximation
        F_true = f(augmented_points)

        approximation_error = cp.linalg.norm(F_true - F_taylor, axis=1)
        error_threshold = cp.maximum(atol, rtol * cp.linalg.norm(F_true, axis=1))

        is_violating_point = (approximation_error > error_threshold).astype(cp.int32)
        violating_counts_per_cluster = cp.add.reduceat(
            is_violating_point, augmented_starts
        )

        has_violating_points = violating_counts_per_cluster > 0
        is_converged = ~has_violating_points

        num_violating_clusters = int(cp.sum(has_violating_points))
        if debug:
            print(f"Bad: {num_violating_clusters}")

        # Store converged clusters
        num_converged_clusters = int(cp.sum(is_converged))
        if num_converged_clusters > 0:
            converged_centroids = cluster_centroids_local[is_converged]
            converged_sizes = augmented_sizes[is_converged]

            if metadata_index + num_converged_clusters <= max_clusters_est:
                cluster_centroids[
                    metadata_index : metadata_index + num_converged_clusters
                ] = converged_centroids
                cluster_sizes[
                    metadata_index : metadata_index + num_converged_clusters
                ] = converged_sizes
                cluster_depths[
                    metadata_index : metadata_index + num_converged_clusters
                ] = current_depth
                metadata_index += num_converged_clusters

            points_in_converged_clusters = is_converged[augmented_cluster_ids]
            points_to_store = augmented_points[points_in_converged_clusters]
            num_points_to_store = points_to_store.shape[0]

            if store_index + num_points_to_store <= converged_points_store.shape[0]:
                converged_points_store[
                    store_index : store_index + num_points_to_store
                ] = points_to_store
                store_index += num_points_to_store

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
            power_iters=power_iters,
        )

        # Split along eigenvectors
        split_points, left_sizes, right_sizes = split_clusters_along_eigenvectors(
            points=violating_points,
            cluster_sizes=violating_cluster_sizes,
            directions=eigenvector_directions,
        )

        # Interleave sizes
        next_cluster_sizes = cp.empty(2 * num_violating_clusters, dtype=cp.int32)
        next_cluster_sizes[0::2] = left_sizes
        next_cluster_sizes[1::2] = right_sizes

        # Filter non-empty clusters
        has_points = next_cluster_sizes > 0
        if not cp.all(has_points):
            initial_cluster_sizes = next_cluster_sizes[has_points]
            current_points = split_points
        else:
            initial_cluster_sizes = next_cluster_sizes
            current_points = split_points

        initial_cluster_starts = cp.r_[0, cp.cumsum(initial_cluster_sizes)[:-1]]
        current_depth += 1

    if debug:
        elapsed_time = time.time() - start_time
        print(f"\n--- Finished in {elapsed_time:.4f}s ---")
        print(
            f"Stored {store_index} converged points. Metadata for {metadata_index} clusters."
        )

    return converged_points_store[:store_index], (
        cluster_centroids[:metadata_index],
        cluster_sizes[:metadata_index],
        cluster_depths[:metadata_index],
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
    N_SAMPLES = 10_000
    N_INIT_CLUSTERS = 100
    MIN_PTS_PER_CLUSTER = 100
    ATOL = 0.01
    RTOL = 0.01

    MAX_DEPTH = 50
    MAX_CLUSTERS_EST = 100_000_000
    POWER_ITERS = 4

    print(f"Generating {N_SAMPLES:,} points on GPU...")
    print(f"{N_INIT_CLUSTERS=}, {MIN_PTS_PER_CLUSTER=}")
    print(f"{ATOL=}, {RTOL=}")
    X_train = cp.random.rand(N_SAMPLES, 3, dtype=DTYPE)
    converged_points, metadata = adaptive_cluster_algorithm_gpu(
        X_train=X_train,
        f=sample_function,
        grad=sample_function_grad,
        num_initial_clusters=N_INIT_CLUSTERS,
        min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
        atol=ATOL,
        rtol=RTOL,
        max_depth=MAX_DEPTH,
        max_clusters_est=MAX_CLUSTERS_EST,
        power_iters=POWER_ITERS,
        debug=True,
    )

    print(f"Final Cluster Count: {len(metadata[0])}")
