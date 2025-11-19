import time

import cupy as cp
from cuml.cluster import KMeans as cuKMeans

# --- 1. Configuration & Constants ---
cp.random.seed(0)
rng = cp.random.default_rng(0)

# Precision settings (Float32 is preferred for GPU)
DTYPE = cp.float32
ATOL = 1e-3
RTOL = 1e-3

N_SAMPLES = 1_000_000
# Algorithm hyperparams
N_INIT_CLUSTERS = 100
MIN_PTS_PER_CLUSTER = 100
ELL = 8  # Sketch size
POWER_ITERS = 1  # Power iterations
MAX_DEPTH = 500  # Max levels of recursion
N_STREAMS = 16  # Number of concurrent CUDA streams


# --- 2. Math Kernels (Vectorized & In-Place) ---
def sample_function(X: cp.ndarray, out: cp.ndarray | None = None) -> cp.ndarray:
    """f(X) = [x1^2 + x2^2, x1 - x2 + x3]"""
    n = X.shape[0]
    if out is None:
        out = cp.empty((n, 2), dtype=DTYPE)

    # Access columns directly
    x0 = X[:, 0]
    x1 = X[:, 1]
    x2 = X[:, 2]

    # f1 = x1^2 + x2^2
    cp.square(x0, out=out[:, 0])
    out[:, 0] += cp.square(x1)

    # f2 = x1 - x2 + x3
    cp.subtract(x0, x1, out=out[:, 1])
    out[:, 1] += x2
    return out


def sample_function_grad(X: cp.ndarray, out: cp.ndarray | None = None) -> cp.ndarray:
    """Returns J: (n, 2, 3)"""
    n = X.shape[0]
    if out is None:
        out = cp.empty((n, 2, 3), dtype=DTYPE)

    # J[:, 0, 0] = 2*x1
    cp.multiply(X[:, 0], 2, out=out[:, 0, 0])
    # J[:, 0, 1] = 2*x2
    cp.multiply(X[:, 1], 2, out=out[:, 0, 1])

    # Constants
    out[:, 0, 2] = 0.0
    out[:, 1, 0] = 1.0
    out[:, 1, 1] = -1.0
    out[:, 1, 2] = 1.0
    return out


def randomized_qr(
    X: cp.ndarray, rng, ell=ELL, power_iterations=POWER_ITERS
) -> cp.ndarray:
    """
    Randomized QR.
    [GPU Note]: This function is per-cluster.
    """
    n, d = X.shape
    X_mean = cp.mean(X, axis=0, keepdims=True)
    X0 = X - X_mean

    # Generate random matrix
    omega = rng.standard_normal((n, ell), dtype=DTYPE)

    Y = X0.T @ omega
    Q, _ = cp.linalg.qr(Y)

    for _ in range(power_iterations):
        temp = X0 @ Q
        Y = X0.T @ temp
        Q, _ = cp.linalg.qr(Y)

    q = Q[:, 0]
    q /= cp.linalg.norm(q)
    return q


def median_cut_projection(X: cp.ndarray, q: cp.ndarray):
    s = X @ q
    med = cp.median(s)
    left_mask = s <= med
    right_mask = s > med
    return left_mask, right_mask


# --- 3. Helper: Vectorized Augmentation & Validation ---
def augment_and_validate_level(X_level, counts_level):
    """
    Processes an entire 'level' (batch of clusters) at once.
    """
    n_clusters = len(counts_level)

    # --- A. Augmentation (Vectorized) ---
    new_counts = cp.maximum(counts_level, MIN_PTS_PER_CLUSTER)

    # FIX: Move new_counts to Host (CPU) for use in cp.repeat
    new_counts_host = cp.asnumpy(new_counts)
    repeats_list = new_counts_host.tolist()

    # Explicit cast to python int for shape argument
    n_total_new = int(new_counts_host.sum())

    # Allocating the main buffer for this level
    X_aug = cp.empty((n_total_new, X_level.shape[1]), dtype=DTYPE)

    # 1. Indices Setup
    orig_starts = cp.r_[0, cp.cumsum(counts_level)[:-1]]
    new_starts = cp.r_[0, cp.cumsum(new_counts)[:-1]]

    # 2. Create "Relative Index" Mask for synthetic slots
    change_points = cp.cumsum(new_counts)[:-1]
    ones = cp.ones(n_total_new, dtype=int)
    ones[change_points] = 1 - new_counts[:-1]
    rel_indices = cp.cumsum(ones) - 1

    # If relative_index >= original_count, it's a synthetic slot
    counts_expanded = cp.repeat(counts_level, repeats_list)
    mask_synthetic = rel_indices >= counts_expanded

    # 3. Copy Real Data (Bulk Copy)
    X_aug[~mask_synthetic] = X_level

    # 4. Generate Synthetic Data
    n_synth = int(cp.sum(mask_synthetic))

    if n_synth > 0:
        # Map synthetic slots back to their cluster properties
        synth_counts = counts_expanded[mask_synthetic]
        orig_starts_expanded = cp.repeat(orig_starts, repeats_list)
        synth_orig_starts = orig_starts_expanded[mask_synthetic]

        # Vectorized Sampling
        rand0 = cp.random.rand(n_synth).astype(DTYPE)
        idx0 = (rand0 * synth_counts).astype(int)

        rand1 = cp.random.rand(n_synth).astype(DTYPE)
        offset = (rand1 * (synth_counts - 1)).astype(int) + 1
        idx1 = (idx0 + offset) % synth_counts

        p0 = X_level[synth_orig_starts + idx0]
        p1 = X_level[synth_orig_starts + idx1]

        lambdas = cp.random.random((n_synth, 1)).astype(DTYPE)

        # In-place write
        target = X_aug[mask_synthetic]
        cp.multiply(1 - lambdas, p0, out=target)
        target += lambdas * p1
        X_aug[mask_synthetic] = target

    # --- B. Validation (Vectorized) ---

    # 1. Centroids
    X_sums = cp.add.reduceat(X_aug, new_starts)
    X_centroids = X_sums / new_counts[:, None]

    # 2. F and J (Computed on K centroids, not N points)
    F_centroids = sample_function(X_centroids)
    J_centroids = sample_function_grad(X_centroids)

    # 3. Broadcast to N points
    cluster_indices_map = cp.repeat(cp.arange(n_clusters), repeats_list)

    F_expanded = F_centroids[cluster_indices_map]
    J_expanded = J_centroids[cluster_indices_map]
    centroids_expanded = X_centroids[cluster_indices_map]

    # 4. Taylor Check
    dx = X_aug - centroids_expanded
    delta = cp.einsum("ijk,ik->ij", J_expanded, dx)
    F_taylor = F_expanded + delta

    F_true = sample_function(X_aug)
    err = cp.linalg.norm(F_true - F_taylor, axis=1)

    # Threshold check
    norm_F = cp.linalg.norm(F_true, axis=1)
    thr = cp.maximum(ATOL, RTOL * norm_F)
    point_violations = err > thr

    # 5. Cluster Violations (Reduce bitwise OR)
    # FIX: Using add.reduceat (Sum). If sum > 0, at least one violation exists.
    # This works reliably on all CuPy versions that support reduceat.
    cluster_violations_count = cp.add.reduceat(
        point_violations.astype(cp.int32), new_starts
    )
    cluster_violations_mask = cluster_violations_count > 0

    return X_aug, new_counts, new_starts, cluster_violations_mask, X_centroids


# --- 4. Main Adaptive Loop (BFS) ---
def run_adaptive_clustering(X_init):
    # Initial Clustering
    print("--- Initializing ---")
    km = cuKMeans(n_clusters=N_INIT_CLUSTERS, random_state=0)
    labels = km.fit_predict(X_init)
    sorted_idxs = cp.argsort(labels)
    X_current = X_init[sorted_idxs].astype(DTYPE)
    unique_labels, counts_current = cp.unique(labels[sorted_idxs], return_counts=True)

    # Stream Pool for Parallel Processing
    streams = [cp.cuda.Stream() for _ in range(N_STREAMS)]

    # Tracking
    final_clusters = []
    level = 0

    # BFS Queue implicitly managed by X_current / counts_current
    while len(counts_current) > 0 and level < MAX_DEPTH:
        print(f"\n--- Level {level} ---")
        print(
            f"Processing {len(counts_current)} clusters ({cp.sum(counts_current)} pts)"
        )

        t0 = time.time()

        # 1. Augment & Validate
        X_aug, counts_aug, starts_aug, violations_mask, centroids = (
            augment_and_validate_level(X_current, counts_current)
        )

        # 2. Process Results
        cluster_idxs = cp.arange(len(counts_current))
        good_idxs = cluster_idxs[~violations_mask]
        bad_idxs = cluster_idxs[violations_mask]

        n_good = int(len(good_idxs))
        n_bad = int(len(bad_idxs))

        print(f"  > Converged: {n_good}")
        print(f"  > Violating: {n_bad}")

        # 3. Store Converged Clusters
        if n_good > 0:
            good_idxs_host = cp.asnumpy(good_idxs)
            starts_aug_host = cp.asnumpy(starts_aug)
            counts_aug_host = cp.asnumpy(counts_aug)

            for i, idx in enumerate(good_idxs_host):
                start = starts_aug_host[idx]
                end = start + counts_aug_host[idx]
                final_clusters.append(
                    {
                        "centroid": centroids[idx],
                        "data": X_aug[start:end],
                        "depth": level,
                        "converged": True,
                    }
                )

        # 4. Prepare Next Level (Splitting)
        if n_bad == 0:
            counts_current = cp.array([])
            break

        next_level_data_parts = []
        next_level_counts = []

        bad_idxs_host = cp.asnumpy(bad_idxs)
        starts_aug_host = cp.asnumpy(starts_aug)
        counts_aug_host = cp.asnumpy(counts_aug)

        # --- STREAM OPTIMIZATION START ---
        # Strategy: Submit all math kernels (QR + Projection) to streams first.
        # This fills the GPU queue. Then iterate and retrieve results (which might block CPU).

        pending_splits = []

        # Phase 1: Submit Async Work
        for i, idx in enumerate(bad_idxs_host):
            stream = streams[i % N_STREAMS]
            start = starts_aug_host[idx]
            end = start + counts_aug_host[idx]

            with stream:
                work_X = X_aug[start:end]
                # Randomized QR (Math heavy)
                q = randomized_qr(work_X, rng)
                # Median Cut (Math heavy)
                left_mask, right_mask = median_cut_projection(work_X, q)

                # Store the futures (GPU arrays) and the stream handle
                pending_splits.append((work_X, left_mask, right_mask, stream))

        # Phase 2: Collect Results (Syncs happen here)
        for work_X, left_mask, right_mask, stream in pending_splits:
            # We generally don't need explicit stream.synchronize() here because
            # the cp.sum call below on the *result array* will implicitly wait
            # for the computation producing that array to finish.

            # Explicitly synchronizing the stream ensures we catch errors cleanly
            # but implicit sync is faster. We rely on implicit sync.

            n_left = int(cp.sum(left_mask))
            if n_left > 0:
                # Slicing [mask] triggers a sync to count elements
                next_level_data_parts.append(work_X[left_mask])
                next_level_counts.append(n_left)

            n_right = int(cp.sum(right_mask))
            if n_right > 0:
                next_level_data_parts.append(work_X[right_mask])
                next_level_counts.append(n_right)

        # --- STREAM OPTIMIZATION END ---

        # 5. Assemble Next Level Data
        if len(next_level_data_parts) > 0:
            X_current = cp.concatenate(next_level_data_parts)
            counts_current = cp.array(next_level_counts)
        else:
            X_current = cp.array([])
            counts_current = cp.array([])

        print(f"  > Time: {time.time() - t0:.4f}s")
        level += 1

    # --- Rescuing Leftovers (Max Depth Reached) ---
    if len(counts_current) > 0:
        print(f"\n[Warning] Max depth ({MAX_DEPTH}) reached.")
        print(f"Storing {len(counts_current)} non-converged clusters as-is.")

        starts_current = cp.r_[0, cp.cumsum(counts_current)[:-1]]
        X_sums = cp.add.reduceat(X_current, starts_current)
        centroids = X_sums / counts_current[:, None]

        counts_current_host = cp.asnumpy(counts_current)
        starts_current_host = cp.asnumpy(starts_current)

        for i in range(len(counts_current)):
            start = starts_current_host[i]
            end = start + counts_current_host[i]
            final_clusters.append(
                {
                    "centroid": centroids[i],
                    "data": X_current[start:end],
                    "depth": level,
                    "converged": False,
                }
            )

    return final_clusters


# --- 5. Execution ---
if __name__ == "__main__":
    # Generate synthetic data
    X_raw = cp.random.rand(N_SAMPLES, 3).astype(DTYPE)
    print(f"Generated raw data: {X_raw.shape}")

    results = run_adaptive_clustering(X_raw)

    print("\n=== Final Results ===")
    print(f"Total Clusters: {len(results)}")

    if len(results) > 0:
        total_points = sum([len(c["data"]) for c in results])
        print(f"Total Points (Augmented): {total_points}")

        # Count converged vs not
        n_converged = sum(1 for c in results if c.get("converged", False))
        print(f"Converged Clusters: {n_converged}")
        print(f"Non-Converged (Max Depth): {len(results) - n_converged}")

        # Check one result
        c0 = results[0]
        print(
            f"Example Cluster 0: Depth {c0['depth']}, Size {len(c0['data'])}, Converged: {c0.get('converged')}"
        )
    else:
        print("No clusters produced.")
