import gc
import time

import cupy as cp
import numpy as np
from cuml.cluster import KMeans as cuKMeans

# Configuration
cp.random.seed(0)
rng = cp.random.default_rng(0)

# Precision & Constants
DTYPE = cp.float64
ATOL = 1e-5
RTOL = 1e-5

# User Params
N_SAMPLES = 10_000_000
MIN_PTS_PER_CLUSTER = 100
N_INIT_CLUSTERS = 100
ELL = 8
POWER_ITERS = 4
MAX_DEPTH = 500
MAX_CLUSTERS_EST = 100_000_000  # Estimate for metadata buffer size

# Memory Optimization
# Reduced batch size slightly to be safer with 800M points total overhead
MAX_BATCH_POINTS = 10_000_000


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
    eigenvectors = rng.standard_normal(  # type: ignore
        (num_clusters, points.shape[1]), dtype=DTYPE
    )
    eigenvectors /= cp.linalg.norm(eigenvectors, axis=1, keepdims=True)

    # Broadcast cluster indices
    cluster_indices = repeat_cluster_indices(cluster_sizes)
    total_points = points.shape[0]

    # Tiling configuration
    # We process 'points' in chunks to compute 'next_eigenvectors'
    # This avoids allocating (N, D) 'scaled_vectors' array
    TILE_SIZE = 5_000_000  # Process 5M points at a time inside the loop

    for _ in range(POWER_ITERS):
        next_eigenvectors = cp.zeros_like(eigenvectors)

        # Inner Loop: Tiled Reduction
        # We manually iterate over the array, compute partial sums, and aggregate
        # This is slower but 100% memory safe.

        for i in range(0, total_points, TILE_SIZE):
            end = min(i + TILE_SIZE, total_points)

            # Slices for this tile
            pts_tile = points[i:end]
            idx_tile = cluster_indices[i:end]

            # Broadcast inputs
            means_tile = cluster_means[idx_tile]
            eig_tile = eigenvectors[idx_tile]

            # Math
            centered = pts_tile - means_tile
            proj = cp.einsum("ij,ij->i", centered, eig_tile)
            scaled = centered * proj[:, None]

            # Aggregate into global accumulator
            # FIX: Use cp.add.at instead of cp.scatter_add
            # cp.add.at performs unbuffered in-place addition with duplicate handling
            # and is widely supported across CuPy versions.
            cp.add.at(next_eigenvectors, idx_tile, scaled)

            del pts_tile, idx_tile, means_tile, eig_tile, centered, proj, scaled

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
    mempool = cp.get_default_memory_pool()

    # Initial clustering
    kmeans = cuKMeans(n_clusters=N_INIT_CLUSTERS, random_state=0)
    initial_labels = kmeans.fit_predict(data_points)

    # Sort data by label
    sort_indices = cp.argsort(initial_labels)
    current_points = data_points[sort_indices]

    unique_cluster_labels, initial_cluster_sizes = cp.unique(  # type: ignore
        initial_labels[sort_indices], return_counts=True
    )
    initial_cluster_starts = cp.r_[0, cp.cumsum(initial_cluster_sizes)[:-1]]

    # Storage buffers
    converged_blocks = []

    # Metadata buffers
    cluster_centroids = cp.empty((MAX_CLUSTERS_EST, 3), dtype=DTYPE)
    cluster_sizes = cp.empty(MAX_CLUSTERS_EST, dtype=cp.int32)
    cluster_depths = cp.empty(MAX_CLUSTERS_EST, dtype=cp.int32)
    metadata_index = 0
    stored_points_count = 0

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

        aug_cumsum = cp.cumsum(augmented_sizes)
        aug_cumsum_host = cp.asnumpy(aug_cumsum)

        num_batches = (
            total_augmented_points + MAX_BATCH_POINTS - 1
        ) // MAX_BATCH_POINTS

        next_level_points_list = []
        next_level_sizes_list = []

        total_bad = 0
        start_cluster_idx = 0

        # --- OUTER LOOP BATCHING ---
        for b in range(num_batches):
            target_aug_count = (b + 1) * MAX_BATCH_POINTS
            end_cluster_idx = int(np.searchsorted(aug_cumsum_host, target_aug_count))
            end_cluster_idx = min(end_cluster_idx + 1, num_clusters)

            if start_cluster_idx >= end_cluster_idx:
                break

            # --- BATCH SETUP ---
            b_sizes = initial_cluster_sizes[start_cluster_idx:end_cluster_idx]
            b_starts = initial_cluster_starts[start_cluster_idx:end_cluster_idx]

            slice_start = int(b_starts[0])
            slice_end = int(b_starts[-1] + b_sizes[-1])
            b_points = current_points[slice_start:slice_end]
            b_starts_local = b_starts - slice_start

            # --- A. AUGMENT (Batch) ---
            b_aug_sizes = augmented_sizes[start_cluster_idx:end_cluster_idx]
            b_aug_total = int(cp.sum(b_aug_sizes))

            b_aug_pts = cp.empty((b_aug_total, 3), dtype=DTYPE)

            b_aug_ids = repeat_cluster_indices(b_aug_sizes)
            b_aug_starts = cp.r_[0, cp.cumsum(b_aug_sizes)[:-1]]

            # Fill Real
            ramp = cp.arange(b_aug_total, dtype=cp.int32) - b_aug_starts[b_aug_ids]
            mask_real = ramp < b_sizes[b_aug_ids]
            b_aug_pts[mask_real] = b_points

            # Fill Synthetic
            mask_synth = ~mask_real
            n_synth = int(cp.sum(mask_synth))

            if n_synth > 0:
                synth_ids = b_aug_ids[mask_synth]
                curr_starts_exp = b_starts_local[synth_ids]
                curr_sizes_exp = b_sizes[synth_ids]

                rand0 = cp.random.rand(n_synth, dtype=DTYPE)
                idx0 = (rand0 * curr_sizes_exp).astype(cp.int32)
                rand1 = cp.random.rand(n_synth, dtype=DTYPE)
                offset = (rand1 * (curr_sizes_exp - 1)).astype(cp.int32) + 1
                idx1 = (idx0 + offset) % curr_sizes_exp

                p0 = b_points[curr_starts_exp + idx0]
                p1 = b_points[curr_starts_exp + idx1]
                lam = cp.random.rand(n_synth, 1, dtype=DTYPE)

                view = b_aug_pts[mask_synth]
                cp.multiply(p0, (1.0 - lam), out=view)
                view += p1 * lam
                b_aug_pts[mask_synth] = view

            del b_points

            # --- B. VALIDATE (Batch) ---
            b_sums = cp.add.reduceat(b_aug_pts, b_aug_starts)
            b_centroids = b_sums / b_aug_sizes[:, None]

            obj_c = objective_function(b_centroids)
            grad_c = objective_function_gradient(b_centroids)

            obj_exp = obj_c[b_aug_ids]
            grad_exp = grad_c[b_aug_ids]
            cent_exp = b_centroids[b_aug_ids]

            dx = b_aug_pts - cent_exp
            approx = cp.einsum("ijk,ik->ij", grad_exp, dx)
            est_obj = obj_exp + approx
            true_obj = objective_function(b_aug_pts)

            err = cp.linalg.norm(true_obj - est_obj, axis=1)
            norm_true = cp.linalg.norm(true_obj, axis=1)
            thresh = cp.maximum(ATOL, RTOL * norm_true)

            is_viol = (err > thresh).astype(cp.int32)
            viol_counts = cp.add.reduceat(is_viol, b_aug_starts)

            mask_bad = viol_counts > 0
            mask_conv = ~mask_bad

            n_bad_batch = int(cp.sum(mask_bad))
            total_bad += n_bad_batch

            # --- C. STORE CONVERGED (Batch) ---
            n_conv_batch = int(cp.sum(mask_conv))
            if n_conv_batch > 0:
                if metadata_index + n_conv_batch <= MAX_CLUSTERS_EST:
                    cluster_centroids[
                        metadata_index : metadata_index + n_conv_batch
                    ] = b_centroids[mask_conv]
                    cluster_sizes[metadata_index : metadata_index + n_conv_batch] = (
                        b_aug_sizes[mask_conv]
                    )
                    cluster_depths[metadata_index : metadata_index + n_conv_batch] = (
                        current_depth
                    )
                    metadata_index += n_conv_batch

                mask_pts_conv = mask_conv[b_aug_ids]
                pts_conv = b_aug_pts[mask_pts_conv]
                converged_blocks.append(pts_conv)
                stored_points_count += pts_conv.shape[0]

            # --- D. SPLIT VIOLATING (Batch) ---
            if n_bad_batch > 0:
                mask_pts_bad = mask_bad[b_aug_ids]
                pts_bad = b_aug_pts[mask_pts_bad]

                sizes_bad = b_aug_sizes[mask_bad]
                starts_bad = cp.r_[0, cp.cumsum(sizes_bad)[:-1]]

                # Use the Tiled Eigenvector Solver here
                dirs = compute_leading_eigenvectors(
                    pts_bad, starts_bad, sizes_bad, n_bad_batch
                )

                split_pts, l_sizes, r_sizes = split_clusters_along_eigenvectors(
                    pts_bad, sizes_bad, dirs
                )

                next_sizes = cp.empty(2 * n_bad_batch, dtype=cp.int32)
                next_sizes[0::2] = l_sizes
                next_sizes[1::2] = r_sizes

                next_level_points_list.append(split_pts)
                next_level_sizes_list.append(next_sizes)

            start_cluster_idx = end_cluster_idx
            del b_aug_pts, b_centroids, obj_exp, grad_exp, dx, true_obj
            mempool.free_all_blocks()
            gc.collect()  # Force garbage collection

        print(f"Bad: {total_bad}")

        if total_bad == 0:
            break

        if len(next_level_points_list) > 0:
            current_points = cp.concatenate(next_level_points_list)
            next_sizes_all = cp.concatenate(next_level_sizes_list)

            mask_nonzero = next_sizes_all > 0
            if not cp.all(mask_nonzero):
                initial_cluster_sizes = next_sizes_all[mask_nonzero]
            else:
                initial_cluster_sizes = next_sizes_all

            initial_cluster_starts = cp.r_[0, cp.cumsum(initial_cluster_sizes)[:-1]]
        else:
            break

        current_depth += 1

    if len(converged_blocks) > 0:
        final_points = cp.concatenate(converged_blocks)
    else:
        final_points = cp.array([], dtype=DTYPE)

    elapsed_time = time.time() - start_time
    print(f"\n--- Finished in {elapsed_time:.4f}s ---")
    print(
        f"Stored {final_points.shape[0]} converged points. Metadata for {metadata_index} clusters."
    )

    return final_points, (
        cluster_centroids[:metadata_index],
        cluster_sizes[:metadata_index],
        cluster_depths[:metadata_index],
    )


# Execution
if __name__ == "__main__":
    print(f"Generating {N_SAMPLES:,} points on GPU...")
    print(f"{ATOL=}, {RTOL=}")
    data_points = cp.random.rand(N_SAMPLES, 3, dtype=DTYPE)
    converged_points, metadata = run_adaptive_clustering(data_points)

    print(f"Final Cluster Count: {len(metadata[0])}")
