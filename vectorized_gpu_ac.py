import time

import cupy as cp
from cuml.cluster import KMeans as cuKMeans

# --- 1. Configuration ---
cp.random.seed(0)
rng = cp.random.default_rng(0)

# Precision & Constants
DTYPE = cp.float32
ATOL = 1e-3
RTOL = 1e-3

# User Params
N_SAMPLES = 10_000_000
MIN_PTS_PER_CLUSTER = 100
N_INIT_CLUSTERS = 10
ELL = 8
POWER_ITERS = 1
MAX_DEPTH = 500
MAX_CLUSTERS_EST = 100_000_000  # Estimate for metadata buffer size

# --- 2. Low-Level GPU Utilities ---


def gpu_repeat_indices(counts: cp.ndarray) -> cp.ndarray:
    """
    Generates an index array [0,0, 1,1,1, ...] based on counts.
    Uses searchsorted to handle 0-size clusters correctly.
    """
    total_size = int(cp.sum(counts))
    if total_size == 0:
        return cp.array([], dtype=cp.int32)

    # Prefix sum gives the upper boundaries of each segment
    offsets = cp.cumsum(counts)

    # Search where each index falls in the bins
    indices = cp.searchsorted(offsets, cp.arange(total_size), side="right")

    return indices.astype(cp.int32)


def sample_function(X, out=None):
    n = X.shape[0]
    if out is None:
        out = cp.empty((n, 2), dtype=DTYPE)
    x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
    cp.square(x0, out=out[:, 0])
    out[:, 0] += cp.square(x1)
    cp.subtract(x0, x1, out=out[:, 1])
    out[:, 1] += x2
    return out


def sample_function_grad(X, out=None):
    n = X.shape[0]
    if out is None:
        out = cp.empty((n, 2, 3), dtype=DTYPE)
    cp.multiply(X[:, 0], 2, out=out[:, 0, 0])
    cp.multiply(X[:, 1], 2, out=out[:, 0, 1])
    out[:, 0, 2] = 0.0
    out[:, 1, 0] = 1.0
    out[:, 1, 1] = -1.0
    out[:, 1, 2] = 1.0
    return out


# --- 3. Vectorized Core Kernels ---


def vectorized_power_iteration(X_ragged, starts, counts, n_clusters):
    """
    Computes the leading eigenvector for N clusters simultaneously using reduction.
    """
    # 1. Compute Means
    X_sums = cp.add.reduceat(X_ragged, starts)
    # Safety against 0-count clusters to avoid NaN
    safe_counts = cp.maximum(counts[:, None], 1)
    means = X_sums / safe_counts

    # 2. Initialize Random Vectors (K, D)
    V = rng.standard_normal((n_clusters, X_ragged.shape[1]), dtype=DTYPE)
    V /= cp.linalg.norm(V, axis=1, keepdims=True)

    # 3. Broadcast Indices
    cluster_ids = gpu_repeat_indices(counts)

    # 4. Power Loop
    for _ in range(POWER_ITERS):
        # Broadcast Mean and V to every point
        means_expanded = means[cluster_ids]
        V_expanded = V[cluster_ids]

        # Center X implicitly
        X_centered = X_ragged - means_expanded

        # Dot product per point: s = X_c . V
        scores = cp.einsum("ij,ij->i", X_centered, V_expanded)

        # Scale vectors: T = X_c * s
        scaled = X_centered * scores[:, None]

        # Reduce back to clusters: V_new = sum(T)
        V_next = cp.add.reduceat(scaled, starts)

        # Normalize Safe (Avoid division by zero)
        norms = cp.linalg.norm(V_next, axis=1, keepdims=True)
        safe_norms = cp.maximum(norms, 1e-9)
        V = V_next / safe_norms

    return V


def vectorized_split(X_ragged, starts, counts, V_dirs, n_clusters):
    """
    Splits all clusters simultaneously by projecting, sorting, and slicing.
    """
    # 1. Project all points: s = X . V
    cluster_ids = gpu_repeat_indices(counts)
    V_expanded = V_dirs[cluster_ids]
    projections = cp.einsum("ij,ij->i", X_ragged, V_expanded)

    # 2. Global Sort (Robust Replacement for Lexsort)
    # We need to sort by (ClusterID, Projection).
    # Using a 2-step argsort avoids tuple errors and type-casting precision loss.

    # Step A: Sort by Secondary Key (Projection)
    perm_sec = cp.argsort(projections)

    # Step B: Sort by Primary Key (Cluster ID) - MUST BE STABLE
    # We apply the secondary permutation to the primary keys first
    cluster_ids_permuted = cluster_ids[perm_sec]

    # CuPy's argsort (Radix sort) is stable for integers
    perm_pri = cp.argsort(cluster_ids_permuted)

    # Step C: Combine permutations
    sorted_perm = perm_sec[perm_pri]

    # Reorder data based on this sort
    X_sorted = X_ragged[sorted_perm]

    # 3. Counts Logic
    counts_left = counts // 2
    counts_right = counts - counts_left

    return X_sorted, counts_left, counts_right


# --- 4. Main Pipeline (No Host Loops) ---


def run_fully_gpu_clustering(X_raw):
    print(f"--- Starting GPU Clustering on {X_raw.shape} ---")
    t_start = time.time()

    # 1. Initial Setup
    km = cuKMeans(n_clusters=N_INIT_CLUSTERS, random_state=0)
    labels = km.fit_predict(X_raw)

    # Sort data by label
    sort_order = cp.argsort(labels)
    X_curr = X_raw[sort_order]

    unique_labels, counts = cp.unique(labels[sort_order], return_counts=True)
    starts = cp.r_[0, cp.cumsum(counts)[:-1]]

    # 2. Storage (Pre-allocated GPU Buffers)
    X_store = cp.empty_like(X_raw)
    store_offset = 0

    # Metadata Buffers (Pre-allocated)
    meta_centroids = cp.empty((MAX_CLUSTERS_EST, 3), dtype=DTYPE)
    meta_sizes = cp.empty(MAX_CLUSTERS_EST, dtype=cp.int32)
    meta_depths = cp.empty(MAX_CLUSTERS_EST, dtype=cp.int32)
    meta_offset = 0

    level = 0

    while len(counts) > 0 and level < MAX_DEPTH:
        n_clusters = len(counts)
        print(f"L{level}: {n_clusters} clusters, {int(cp.sum(counts))} pts. ", end="")

        # --- A. Augment ---
        new_counts = cp.maximum(counts, MIN_PTS_PER_CLUSTER)
        total_augmented = int(cp.sum(new_counts))

        # Use previous buffer if sizes match, else realloc
        # For simplicity, we realloc here. In prod, use a memory pool or ping-pong buffer.
        X_aug = cp.empty((total_augmented, X_curr.shape[1]), dtype=DTYPE)

        aug_cluster_ids = gpu_repeat_indices(new_counts)
        aug_starts = cp.r_[0, cp.cumsum(new_counts)[:-1]]

        # Mask for Real vs Synthetic
        ramp = cp.arange(total_augmented, dtype=cp.int32) - aug_starts[aug_cluster_ids]
        real_counts_expanded = counts[aug_cluster_ids]
        mask_real = ramp < real_counts_expanded

        # Fill Real
        X_aug[mask_real] = X_curr

        # Fill Synthetic
        mask_synth = ~mask_real
        n_synth = int(cp.sum(mask_synth))

        if n_synth > 0:
            synth_cluster_ids = aug_cluster_ids[mask_synth]

            curr_starts_expanded = starts[synth_cluster_ids]
            curr_counts_expanded = counts[synth_cluster_ids]

            rand0 = cp.random.rand(n_synth, dtype=DTYPE)
            idx0 = (rand0 * curr_counts_expanded).astype(cp.int32)

            rand1 = cp.random.rand(n_synth, dtype=DTYPE)
            offset = (rand1 * (curr_counts_expanded - 1)).astype(cp.int32) + 1
            idx1 = (idx0 + offset) % curr_counts_expanded

            p0 = X_curr[curr_starts_expanded + idx0]
            p1 = X_curr[curr_starts_expanded + idx1]

            lam = cp.random.rand(n_synth, 1, dtype=DTYPE)

            # In-place Write
            view = X_aug[mask_synth]
            cp.multiply(p0, (1.0 - lam), out=view)
            view += p1 * lam
            X_aug[mask_synth] = view

        # --- B. Validate ---
        X_sums = cp.add.reduceat(X_aug, aug_starts)
        centroids = X_sums / new_counts[:, None]

        F_c = sample_function(centroids)
        J_c = sample_function_grad(centroids)

        F_exp = F_c[aug_cluster_ids]
        J_exp = J_c[aug_cluster_ids]
        C_exp = centroids[aug_cluster_ids]

        dx = X_aug - C_exp
        delta = cp.einsum("ijk,ik->ij", J_exp, dx)
        F_est = F_exp + delta
        F_true = sample_function(X_aug)

        err = cp.linalg.norm(F_true - F_est, axis=1)
        thr = cp.maximum(ATOL, RTOL * cp.linalg.norm(F_true, axis=1))

        # Violations Reduction
        is_bad_point = (err > thr).astype(cp.int32)
        bad_counts = cp.add.reduceat(is_bad_point, aug_starts)

        mask_violating_clusters = bad_counts > 0
        mask_converged_clusters = ~mask_violating_clusters

        n_bad = int(cp.sum(mask_violating_clusters))
        print(f"Bad: {n_bad}")

        # --- C. Store Converged ---
        n_conv_clusters = int(cp.sum(mask_converged_clusters))
        if n_conv_clusters > 0:
            # 1. Save Metadata
            conv_centroids = centroids[mask_converged_clusters]
            conv_sizes = new_counts[mask_converged_clusters]

            if meta_offset + n_conv_clusters <= MAX_CLUSTERS_EST:
                meta_centroids[meta_offset : meta_offset + n_conv_clusters] = (
                    conv_centroids
                )
                meta_sizes[meta_offset : meta_offset + n_conv_clusters] = conv_sizes
                meta_depths[meta_offset : meta_offset + n_conv_clusters] = level
                meta_offset += n_conv_clusters

            # 2. Save Points
            point_mask_conv = mask_converged_clusters[aug_cluster_ids]
            pts_to_save = X_aug[point_mask_conv]
            n_save = pts_to_save.shape[0]

            if store_offset + n_save <= X_store.shape[0]:
                X_store[store_offset : store_offset + n_save] = pts_to_save
                store_offset += n_save

        if n_bad == 0:
            break

        # --- D. Split Violating ---
        # Filter to keep only bad clusters for the next round
        point_mask_bad = mask_violating_clusters[aug_cluster_ids]
        X_bad = X_aug[point_mask_bad]

        bad_counts = new_counts[mask_violating_clusters]
        bad_starts = cp.r_[0, cp.cumsum(bad_counts)[:-1]]

        # Power Iteration
        V_dirs = vectorized_power_iteration(X_bad, bad_starts, bad_counts, n_bad)

        # Split
        X_split, counts_L, counts_R = vectorized_split(
            X_bad, bad_starts, bad_counts, V_dirs, n_bad
        )

        # Interleave Counts
        next_counts = cp.empty(2 * n_bad, dtype=cp.int32)
        next_counts[0::2] = counts_L
        next_counts[1::2] = counts_R

        # Setup for next level
        mask_nonzero = next_counts > 0
        if not cp.all(mask_nonzero):
            counts = next_counts[mask_nonzero]
            # X_split is already contiguous/compacted by the sort
            X_curr = X_split
        else:
            counts = next_counts
            X_curr = X_split

        starts = cp.r_[0, cp.cumsum(counts)[:-1]]
        level += 1

    total_time = time.time() - t_start
    print(f"\n--- Finished in {total_time:.4f}s ---")
    print(
        f"Stored {store_offset} converged points. Metadata for {meta_offset} clusters."
    )

    return X_store[:store_offset], (
        meta_centroids[:meta_offset],
        meta_sizes[:meta_offset],
        meta_depths[:meta_offset],
    )


# --- 5. Execution ---

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} points on GPU...")
    X_raw = cp.random.rand(N_SAMPLES, 3, dtype=DTYPE)
    final_pts, meta = run_fully_gpu_clustering(X_raw)

    print(f"Final Cluster Count: {len(meta[0])}")
