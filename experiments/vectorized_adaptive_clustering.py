import time

import numpy as np
from sklearn.cluster import KMeans

# --- 1. Configuration & Constants ---
np.random.seed(0)
rng = np.random.default_rng(0)

# Precision settings (Float32 is preferred for GPU)
DTYPE = np.float32
ATOL = 1e-3
RTOL = 1e-3

# Algorithm hyperparams
MIN_PTS_PER_CLUSTER = 5
N_INIT_CLUSTERS = 50
ELL = 10  # Sketch size
POWER_ITERS = 2  # Power iterations
MAX_DEPTH = 500  # Max levels of recursion


# --- 2. Math Kernels (Vectorized & In-Place) ---
def sample_function(X: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """f(X) = [x1^2 + x2^2, x1 - x2 + x3]"""
    n = X.shape[0]
    if out is None:
        out = np.empty((n, 2), dtype=DTYPE)
    np.square(X[:, 0], out=out[:, 0])
    out[:, 0] += np.square(X[:, 1])
    np.subtract(X[:, 0], X[:, 1], out=out[:, 1])
    out[:, 1] += X[:, 2]
    return out


def sample_function_grad(X: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Returns J: (n, 2, 3)"""
    n = X.shape[0]
    if out is None:
        out = np.empty((n, 2, 3), dtype=DTYPE)
    np.multiply(X[:, 0], 2, out=out[:, 0, 0])
    np.multiply(X[:, 1], 2, out=out[:, 0, 1])
    out[:, 0, 2] = 0.0
    out[:, 1, 0] = 1.0
    out[:, 1, 1] = -1.0
    out[:, 1, 2] = 1.0
    return out


def randomized_qr(
    X: np.ndarray, rng, ell=ELL, power_iterations=POWER_ITERS
) -> np.ndarray:
    """
    Randomized QR.
    [GPU Note]: This function is per-cluster.
    """
    n, d = X.shape
    X_mean = np.mean(X, axis=0, keepdims=True)
    X0 = X - X_mean

    # [Stream Note]: CuPy random generation here
    omega = rng.standard_normal((n, ell), dtype=DTYPE)

    Y = X0.T @ omega
    Q, _ = np.linalg.qr(Y)

    for _ in range(power_iterations):
        temp = X0 @ Q
        Y = X0.T @ temp
        Q, _ = np.linalg.qr(Y)

    q = Q[:, 0]
    q /= np.linalg.norm(q)
    return q


def median_cut_projection(X: np.ndarray, q: np.ndarray):
    s = X @ q
    med = np.median(s)
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
    new_counts = np.maximum(counts_level, MIN_PTS_PER_CLUSTER)
    n_total_new = new_counts.sum()

    # Allocating the main buffer for this level
    X_aug = np.empty((n_total_new, X_level.shape[1]), dtype=DTYPE)

    # 1. Indices Setup
    orig_starts = np.r_[0, np.cumsum(counts_level)[:-1]]
    new_starts = np.r_[0, np.cumsum(new_counts)[:-1]]

    # 2. Create "Relative Index" Mask for synthetic slots
    change_points = np.cumsum(new_counts)[:-1]
    ones = np.ones(n_total_new, dtype=int)
    ones[change_points] = 1 - new_counts[:-1]
    rel_indices = np.cumsum(ones) - 1

    # If relative_index >= original_count, it's a synthetic slot
    counts_expanded = np.repeat(counts_level, new_counts)
    mask_synthetic = rel_indices >= counts_expanded

    # 3. Copy Real Data (Bulk Copy)
    X_aug[~mask_synthetic] = X_level

    # 4. Generate Synthetic Data
    n_synth = np.sum(mask_synthetic)
    if n_synth > 0:
        # Map synthetic slots back to their cluster properties
        synth_counts = counts_expanded[mask_synthetic]
        orig_starts_expanded = np.repeat(orig_starts, new_counts)
        synth_orig_starts = orig_starts_expanded[mask_synthetic]

        # Vectorized Sampling
        rand0 = np.random.rand(n_synth).astype(DTYPE)  # type: ignore
        idx0 = (rand0 * synth_counts).astype(int)

        rand1 = np.random.rand(n_synth).astype(DTYPE)  # type: ignore
        offset = (rand1 * (synth_counts - 1)).astype(int) + 1
        idx1 = (idx0 + offset) % synth_counts

        p0 = X_level[synth_orig_starts + idx0]
        p1 = X_level[synth_orig_starts + idx1]

        lambdas = np.random.random((n_synth, 1)).astype(DTYPE)

        # In-place write
        target = X_aug[mask_synthetic]
        np.multiply(1 - lambdas, p0, out=target)
        target += lambdas * p1
        X_aug[mask_synthetic] = target

    # --- B. Validation (Vectorized) ---

    # 1. Centroids
    X_sums = np.add.reduceat(X_aug, new_starts)
    X_centroids = X_sums / new_counts[:, None]

    # 2. F and J (Computed on K centroids, not N points)
    F_centroids = sample_function(X_centroids)
    J_centroids = sample_function_grad(X_centroids)

    # 3. Broadcast to N points
    cluster_indices_map = np.repeat(np.arange(n_clusters), new_counts)

    F_expanded = F_centroids[cluster_indices_map]
    J_expanded = J_centroids[cluster_indices_map]
    centroids_expanded = X_centroids[cluster_indices_map]

    # 4. Taylor Check
    dx = X_aug - centroids_expanded
    delta = np.einsum("ijk,ik->ij", J_expanded, dx)
    F_taylor = F_expanded + delta

    F_true = sample_function(X_aug)
    err = np.linalg.norm(F_true - F_taylor, axis=1)

    # Threshold check
    norm_F = np.linalg.norm(F_true, axis=1)
    thr = np.maximum(ATOL, RTOL * norm_F)
    point_violations = err > thr

    # 5. Cluster Violations (Reduce bitwise OR)
    cluster_violations_mask = np.bitwise_or.reduceat(point_violations, new_starts)

    return X_aug, new_counts, new_starts, cluster_violations_mask, X_centroids


# --- 4. Main Adaptive Loop (BFS) ---
def run_adaptive_clustering(X_init):
    # Initial Clustering
    print("--- Initializing ---")
    km = KMeans(n_clusters=N_INIT_CLUSTERS, random_state=0)
    labels = km.fit_predict(X_init)
    sorted_idxs = np.argsort(labels)
    X_current = X_init[sorted_idxs].astype(DTYPE)
    unique_labels, counts_current = np.unique(labels[sorted_idxs], return_counts=True)

    # Tracking
    final_clusters = []  # List of dicts or arrays
    level = 0

    # BFS Queue implicitly managed by X_current / counts_current
    while len(counts_current) > 0 and level < MAX_DEPTH:
        print(f"\n--- Level {level} ---")
        print(
            f"Processing {len(counts_current)} clusters ({np.sum(counts_current)} pts)"
        )

        t0 = time.time()

        # 1. Augment & Validate
        X_aug, counts_aug, starts_aug, violations_mask, centroids = (
            augment_and_validate_level(X_current, counts_current)
        )

        # 2. Process Results
        cluster_idxs = np.arange(len(counts_current))
        good_idxs = cluster_idxs[~violations_mask]
        bad_idxs = cluster_idxs[violations_mask]

        print(f"  > Converged: {len(good_idxs)}")
        print(f"  > Violating: {len(bad_idxs)}")

        # 3. Store Converged Clusters
        for idx in good_idxs:
            start = starts_aug[idx]
            end = start + counts_aug[idx]
            final_clusters.append(
                {
                    "centroid": centroids[idx],
                    "data": X_aug[start:end],
                    "depth": level,
                    "converged": True,
                }
            )

        # 4. Prepare Next Level (Splitting)
        if len(bad_idxs) == 0:
            # If everything converged, we clear X_current so loop terminates cleanly next check
            counts_current = np.array([])
            break

        next_level_data_parts = []
        next_level_counts = []

        for i, idx in enumerate(bad_idxs):
            start = starts_aug[idx]
            end = start + counts_aug[idx]
            work_X = X_aug[start:end]

            # A. Randomized QR
            q = randomized_qr(work_X, rng)

            # B. Median Cut
            left_mask, right_mask = median_cut_projection(work_X, q)

            # C. Collect Children
            n_left = np.sum(left_mask)
            if n_left > 0:
                next_level_data_parts.append(work_X[left_mask])
                next_level_counts.append(n_left)

            n_right = np.sum(right_mask)
            if n_right > 0:
                next_level_data_parts.append(work_X[right_mask])
                next_level_counts.append(n_right)

        # 5. Assemble Next Level Data
        if len(next_level_data_parts) > 0:
            X_current = np.concatenate(next_level_data_parts)
            counts_current = np.array(next_level_counts)
        else:
            X_current = np.array([])
            counts_current = np.array([])

        print(f"  > Time: {time.time() - t0:.4f}s")
        level += 1

    # --- Rescuing Leftovers (Max Depth Reached) ---
    if len(counts_current) > 0:
        print(f"\n[Warning] Max depth ({MAX_DEPTH}) reached.")
        print(f"Storing {len(counts_current)} non-converged clusters as-is.")

        # X_current contains the *children* of the previous level (un-augmented)
        # We compute basic centroids for them and save them.
        starts_current = np.r_[0, np.cumsum(counts_current)[:-1]]

        # Simple centroid calc
        X_sums = np.add.reduceat(X_current, starts_current)
        centroids = X_sums / counts_current[:, None]

        for i in range(len(counts_current)):
            start = starts_current[i]
            end = start + counts_current[i]
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
    print("Generating 400k points...")
    X_raw = np.random.rand(800_000, 3).astype(DTYPE)

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
