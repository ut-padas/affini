import time

import cupy as cp
from cuml.cluster import KMeans as cuKMeans

# Configuration
cp.random.seed(0)
rng = cp.random.default_rng(0)

# Precision & Constants
DTYPE = cp.float32
ATOL = 1e-5
RTOL = 1e-5

# User Params
N_SAMPLES = 1_000_000
MIN_PTS_PER_CLUSTER = 100
N_INIT_CLUSTERS = 10
ELL = 8
POWER_ITERS = 4
MAX_DEPTH = 500
MAX_CLUSTERS_EST = 100_000_000  # Estimate for metadata buffer size


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


def objective_function(points: cp.ndarray, output: cp.ndarray = None) -> cp.ndarray:
    """Compute objective function values for 3D points.

    The function is f1 = x^2 + y^2, f2 = x - y + z.

    Args:
        points (cp.ndarray): Input points, shape (n, 3).
        output (cp.ndarray, optional): Output array, shape (n, 2).

    Returns:
        cp.ndarray: Objective values, shape (n, 2).
    """
    n = points.shape[0]
    if output is None:
        output = cp.empty((n, 2), dtype=DTYPE)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    cp.square(x, out=output[:, 0])
    output[:, 0] += cp.square(y)
    cp.subtract(x, y, out=output[:, 1])
    output[:, 1] += z
    return output


def objective_function_gradient(
    points: cp.ndarray, output: cp.ndarray = None
) -> cp.ndarray:
    """Compute gradient of objective function for 3D points.

    Args:
        points (cp.ndarray): Input points, shape (n, 3).
        output (cp.ndarray, optional): Output array, shape (n, 2, 3).

    Returns:
        cp.ndarray: Gradients, shape (n, 2, 3).
    """
    n = points.shape[0]
    if output is None:
        output = cp.empty((n, 2, 3), dtype=DTYPE)
    cp.multiply(points[:, 0], 2, out=output[:, 0, 0])
    cp.multiply(points[:, 1], 2, out=output[:, 0, 1])
    output[:, 0, 2] = 0.0
    output[:, 1, 0] = 1.0
    output[:, 1, 1] = -1.0
    output[:, 1, 2] = 1.0
    return output


def compute_leading_eigenvectors(
    points: cp.ndarray,
    cluster_starts: cp.ndarray,
    cluster_sizes: cp.ndarray,
    num_clusters: int,
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
    eigenvectors = rng.standard_normal((num_clusters, points.shape[1]), dtype=DTYPE)
    eigenvectors /= cp.linalg.norm(eigenvectors, axis=1, keepdims=True)

    # Broadcast cluster indices
    cluster_indices = repeat_cluster_indices(cluster_sizes)

    # Power iteration loop
    for _ in range(POWER_ITERS):
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
    cluster_starts: cp.ndarray,
    cluster_sizes: cp.ndarray,
    directions: cp.ndarray,
    num_clusters: int,
):
    """Split clusters by projecting along eigenvectors and sorting.

    Args:
        points (cp.ndarray): Flattened points for all clusters.
        cluster_starts (cp.ndarray): Start indices for each cluster.
        cluster_sizes (cp.ndarray): Number of points in each cluster.
        directions (cp.ndarray): Eigenvectors for splitting, shape (num_clusters, d).
        num_clusters (int): Number of clusters.

    Returns:
        tuple: (sorted_points, left_sizes, right_sizes)
    """
    # Project all points onto directions
    cluster_indices = repeat_cluster_indices(cluster_sizes)
    directions_broadcasted = directions[cluster_indices]
    projection_scores = cp.einsum("ij,ij->i", points, directions_broadcasted)

    # Global sort by (cluster_id, projection)
    # Two-step argsort to avoid tuple issues
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


def run_adaptive_clustering(data_points: cp.ndarray):
    """Run fully GPU-based adaptive clustering.

    Args:
        data_points (cp.ndarray): Input data points.

    Returns:
        tuple: (converged_points, metadata) where metadata is (centroids, sizes, depths).
    """
    print(f"--- Starting GPU Clustering on {data_points.shape} ---")
    start_time = time.time()

    # Initial clustering
    kmeans = cuKMeans(n_clusters=N_INIT_CLUSTERS, random_state=0)
    initial_labels = kmeans.fit_predict(data_points)

    # Sort data by label
    sort_indices = cp.argsort(initial_labels)
    current_points = data_points[sort_indices]

    unique_cluster_labels, initial_cluster_sizes = cp.unique(
        initial_labels[sort_indices], return_counts=True
    )
    initial_cluster_starts = cp.r_[0, cp.cumsum(initial_cluster_sizes)[:-1]]

    # Storage buffers
    converged_points_store = cp.empty_like(data_points)
    store_index = 0

    # Metadata buffers
    cluster_centroids = cp.empty((MAX_CLUSTERS_EST, 3), dtype=DTYPE)
    cluster_sizes = cp.empty(MAX_CLUSTERS_EST, dtype=cp.int32)
    cluster_depths = cp.empty(MAX_CLUSTERS_EST, dtype=cp.int32)
    metadata_index = 0

    current_depth = 0

    while len(initial_cluster_sizes) > 0 and current_depth < MAX_DEPTH:
        num_clusters = len(initial_cluster_sizes)
        print(
            f"L{current_depth}: {num_clusters} clusters, {int(cp.sum(initial_cluster_sizes))} pts. ",
            end="",
        )

        # Augment clusters
        augmented_sizes = cp.maximum(initial_cluster_sizes, MIN_PTS_PER_CLUSTER)
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

        objective_at_centroids = objective_function(cluster_centroids_local)
        gradient_at_centroids = objective_function_gradient(cluster_centroids_local)

        objective_expanded = objective_at_centroids[augmented_cluster_ids]
        gradient_expanded = gradient_at_centroids[augmented_cluster_ids]
        centroids_expanded = cluster_centroids_local[augmented_cluster_ids]

        deviation_from_centroid = augmented_points - centroids_expanded
        linear_approximation = cp.einsum(
            "ijk,ik->ij", gradient_expanded, deviation_from_centroid
        )
        estimated_objective = objective_expanded + linear_approximation
        true_objective = objective_function(augmented_points)

        approximation_error = cp.linalg.norm(
            true_objective - estimated_objective, axis=1
        )
        error_threshold = cp.maximum(
            ATOL, RTOL * cp.linalg.norm(true_objective, axis=1)
        )

        is_violating_point = (approximation_error > error_threshold).astype(cp.int32)
        violating_counts_per_cluster = cp.add.reduceat(
            is_violating_point, augmented_starts
        )

        has_violating_points = violating_counts_per_cluster > 0
        is_converged = ~has_violating_points

        num_violating_clusters = int(cp.sum(has_violating_points))
        print(f"Bad: {num_violating_clusters}")

        # Store converged clusters
        num_converged_clusters = int(cp.sum(is_converged))
        if num_converged_clusters > 0:
            converged_centroids = cluster_centroids_local[is_converged]
            converged_sizes = augmented_sizes[is_converged]

            if metadata_index + num_converged_clusters <= MAX_CLUSTERS_EST:
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
            violating_points,
            violating_cluster_starts,
            violating_cluster_sizes,
            num_violating_clusters,
        )

        # Split along eigenvectors
        split_points, left_sizes, right_sizes = split_clusters_along_eigenvectors(
            violating_points,
            violating_cluster_starts,
            violating_cluster_sizes,
            eigenvector_directions,
            num_violating_clusters,
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


# Execution
if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} points on GPU...")
    data_points = cp.random.rand(N_SAMPLES, 3, dtype=DTYPE)
    converged_points, metadata = run_adaptive_clustering(data_points)

    print(f"Final Cluster Count: {len(metadata[0])}")
