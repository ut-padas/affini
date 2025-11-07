"""GPU-based Adaptive Two-Means Clustering for Function Approximation."""

import time
from dataclasses import dataclass
from typing import Callable

import cupy as cp
from cuml.cluster import KMeans as cuKMeans

try:
    _ = cp.is_available()
except Exception as e:
    print("You do not have a working CuPy installation/GPU setup.")
    raise e

# --------------------------- Config ---------------------------------
props = cp.cuda.runtime.getDeviceProperties(0)
num_sms = props["multiProcessorCount"]

SEED = 0
CP_DTYPE = cp.float64
NUM_STREAMS = num_sms
ELL = 8  # randomized-QR sketch size
POWER_ITERS = 1  # randomized-QR power iterations


# --------------------------- Utilities ------------------------------
def randomized_qr_gpu(
    X: cp.ndarray,
    rng: cp.random.Generator,
    ell: int = ELL,
    power_iterations: int = POWER_ITERS,
) -> cp.ndarray:
    """
    Randomized-QR to compute approximate leading left singular vector of X.

    Args:
        X: (n, D) data matrix
        rng: cupy random generator
        ell: sketch size
        power_iterations: number of power iterations

    Returns:
        q: (D,) approximate leading left singular vector
    """
    XT = X.T
    _, N = XT.shape
    omega = rng.standard_normal((N, ell), dtype=X.dtype)  # (n, ell)

    X0 = XT - cp.mean(XT, axis=1, keepdims=True)  # (D,n)

    Y = X0 @ omega  # (D, ell)
    Q, _ = cp.linalg.qr(Y)  # (D, ell)
    for _ in range(power_iterations):
        Y = X0 @ (X0.T @ Q)  # (D, ell)
        Q, _ = cp.linalg.qr(Y)

    q = Q[:, 0]  # (D,)

    # Normalize
    q /= cp.linalg.norm(q)
    return q


def median_cut_projection(
    X: cp.ndarray, q: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Project points onto q and split at median.

    Args:
        X: (n, D) data matrix
        q: (D,) projection direction

    Returns:
        left_mask: (n,) boolean mask for left cluster
        right_mask: (n,) boolean mask for right cluster
    """
    s = X @ q  # (n,)
    med = cp.median(s)
    left = s <= med
    right = s > med

    # Skip edge-case handling for now
    # Ensure both non-empty
    n_left = int(left.sum())
    n_right = int(right.sum())
    if n_left == 0:
        # move the smallest element to left
        k = int(cp.argmin(s))
        left[k] = True
        right[k] = False
    elif n_right == 0:
        # move the largest element to right
        k = int(cp.argmax(s))
        right[k] = True
        left[k] = False
    return left, right


def taylor_errors(
    X: cp.ndarray, F: cp.ndarray, centroid: cp.ndarray, F_c: cp.ndarray, J_c: cp.ndarray
) -> cp.ndarray:
    """
    Compute per-point Taylor approximation errors, given cluster centroid info.

    Args:
        X: (n, D) data matrix
        F: (n, FD) function values at X
        centroid: (1, D) cluster centroid
        F_c: (1, FD) function value at centroid
        J_c: (1, FD, D) Jacobian at centroid

    Returns:
        err: (n,) per-point error norms
    """
    dx = X - centroid  # (n, D)
    f_taylor = F_c + cp.einsum("ijk,nk->nj", J_c, dx)
    err = cp.linalg.norm(F - f_taylor, axis=1)
    return err


def generate_extra_points(
    X: cp.ndarray,
    min_points_per_cluster: int,
    rng: cp.random.Generator,
) -> cp.ndarray:
    """
    Given a cluster with too few points, generate extra points via convex combinations.

    Args:
        X: (Q, D) data matrix for the cluster
        min_points_per_cluster: minimum desired points per cluster
        rng: cupy random generator

    Returns:
        new_X: (n_extra, D) generated extra points
    """
    Q = X.shape[0]
    if Q < 2:
        raise ValueError("At least two points required to generate extra points.")

    n_extra = max(1, min_points_per_cluster - Q + 1)
    lambdas = rng.random(n_extra, dtype=X.dtype)

    idx0 = rng.integers(0, Q, size=n_extra)
    offs = rng.integers(1, Q, size=n_extra)
    idx1 = (idx0 + offs) % Q

    new_X = lambdas[:, None] * X[idx0] + (1.0 - lambdas)[:, None] * X[idx1]
    return new_X


@dataclass
class Cluster:
    """Data class to hold cluster information."""

    centroid: cp.ndarray  # (1, D)
    F_centroid: cp.ndarray  # (1, FD)
    J_centroid: cp.ndarray  # (1, FD, D)
    radius: float
    hessian_proxy: float
    cluster_size: int
    max_error: float
    mean_error: float


def refine_adaptive_clusters(
    X: cp.ndarray,
    labels: cp.ndarray,
    f: Callable[[cp.ndarray], tuple[cp.ndarray, cp.ndarray]],
    abs_tol: float,
    rel_tol: float,
    min_pts_per_cluster: int,
    ell: int,
    power_iters: int,
    num_streams: int,
    seed: int,
) -> list[Cluster]:
    """
    Adaptive two-means clustering with GPU streaming.

    Args:
        X: (N, D) data matrix
        labels: (N,) initial cluster labels from a clustering algorithm
        f: function that returns (F, J) at X.
            Note: f(X) should return F: (N, FD), J: (N, FD, D) given X: (N, D)
        abs_tol: absolute tolerance for Taylor error
        rel_tol: relative tolerance for Taylor error
        min_pts_per_cluster: minimum points per cluster for evaluation
        ell: sketch size for randomized-QR
        power_iters: number of power iterations for randomized-QR
        num_streams: number of CUDA streams to use
        seed: random seed for reproducibility

    Returns:
        final_clusters: list of Cluster objects representing final clusters
    """

    rng = cp.random.default_rng(seed)
    assert rng is not None, "Something went wrong with RNG initialization."

    # Initialize queue with point arrays and cluster IDs from initial KMeans
    cluster_ids = cp.arange(int(labels.max()) + 1)
    queue: list[tuple[cp.ndarray, int]] = []  # (cluster_points, cluster_id)
    for cid in cluster_ids.tolist():
        idx = cp.where(labels == cid)[0]
        if idx.size > 0:
            queue.append((X[idx], cid))  # Store points and cluster ID

    final_clusters: list[Cluster] = []
    n_level = 0
    level_start_time = time.perf_counter()

    # BFS by levels
    while queue:
        if n_level > 0:
            level_time = time.perf_counter() - level_start_time
            print(
                f"Level {n_level}: processing {len(queue)} clusters (prev level took {level_time:.2f}s)"
            )

        level_start_time = time.perf_counter()
        level = queue
        queue = []

        # Batch violation detection for entire level
        if not level:
            break

        # Pre-process clusters: augment small ones with extra points for violation testing
        cluster_points_list = []
        original_cluster_points = []
        cluster_sizes = []

        for cluster_points, cluster_id in level:
            original_cluster_points.append(cluster_points)
            cluster_size = cluster_points.shape[0]

            # Augment small clusters for violation testing (like original version)
            if cluster_size < min_pts_per_cluster:
                new_X = generate_extra_points(cluster_points, min_pts_per_cluster, rng)
                work_X = cp.concatenate([cluster_points, new_X], axis=0)
                cluster_points_list.append(work_X)
            else:
                cluster_points_list.append(cluster_points)

            cluster_sizes.append(cluster_points_list[-1].shape[0])

        cluster_starts = cp.concatenate(
            [cp.array([0]), cp.cumsum(cp.array(cluster_sizes[:-1]))]
        )

        # Combine all clusters for batch processing
        all_X = cp.concatenate(cluster_points_list, axis=0)

        # Create cluster labels using repeat instead of loops
        all_cluster_labels = cp.repeat(
            cp.arange(len(level), dtype=cp.int32), cluster_sizes
        )

        # Compute centroids vectorized (from augmented data)
        centroids = cp.stack([pts.mean(axis=0) for pts in cluster_points_list])

        # Optimized batch violation detection
        # Step 1: Compute function values at centroids
        F_centroids_at_centroids, J_centroids_at_centroids = f(centroids)

        # Step 2: Compute function values at all points
        F_true, _ = f(all_X)

        # Step 3: Vectorized Taylor approximation using broadcasting
        # Expand centroids to match point counts
        centroid_indices = cp.repeat(cp.arange(len(level)), cluster_sizes)
        expanded_centroids = centroids[centroid_indices]  # (total_points, D)
        expanded_F_centroids = F_centroids_at_centroids[centroid_indices]
        expanded_J_centroids = J_centroids_at_centroids[centroid_indices]

        # Vectorized Taylor approximation for all points at once
        dx = all_X - expanded_centroids  # (total_points, D)
        F_taylor = expanded_F_centroids + cp.einsum(
            "nij,nj->ni", expanded_J_centroids, dx
        )
        err = cp.linalg.norm(F_true - F_taylor, axis=1)  # (total_points,)
        thr = cp.maximum(abs_tol, rel_tol * cp.linalg.norm(F_true, axis=1))
        violations_mask = err > thr

        # Vectorized identification of violating clusters
        # Use bincount to find clusters with any violations
        if violations_mask.any():
            violating_cluster_counts = cp.bincount(
                all_cluster_labels[violations_mask], minlength=len(level)
            )
            violating_clusters = set(cp.where(violating_cluster_counts > 0)[0].tolist())
        else:
            violating_clusters = set()

        print(f"  --> {len(violating_clusters)}/{len(level)} clusters need splitting")

        # Pre-compute error statistics for all clusters
        cluster_error_stats = []
        for i, cluster_size in enumerate(cluster_sizes):
            cluster_start = cluster_starts[i] if i < len(cluster_starts) else len(all_X)
            cluster_end = (
                cluster_starts[i + 1] if i + 1 < len(cluster_starts) else len(all_X)
            )
            cluster_mask = slice(cluster_start, cluster_end)
            cluster_errors = err[cluster_mask]
            cluster_error_stats.append(
                (float(cluster_errors.max()), float(cluster_errors.mean()))
            )

        # Batch process non-violating clusters (reuse already computed values)
        non_violating_indices = [
            i for i in range(len(level)) if i not in violating_clusters
        ]

        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]

        # Per-level containers for children to enqueue after sync
        children_left: list[tuple[cp.ndarray, int]] = []
        children_right: list[tuple[cp.ndarray, int]] = []

        for i, (X_cluster, cluster_id) in enumerate(level):
            stream = streams[i % num_streams]
            with stream:
                cluster_size = X_cluster.shape[0]  # Original cluster size

                if i not in violating_clusters:
                    # No violations - create final cluster using precomputed batch values
                    # Use centroid from augmented data but original cluster size
                    centroid = centroids[i : i + 1]  # (1, D)

                    # Reuse already computed function values from centroids
                    F_c = F_centroids_at_centroids[i : i + 1]
                    J_c = J_centroids_at_centroids[i : i + 1]
                    max_err, mean_err = cluster_error_stats[i]

                    # Use original cluster points for radius calculation
                    radius = float(cp.max(cp.linalg.norm(X_cluster - centroid, axis=1)))

                    # hessian proxy (scalar)
                    h_proxy = 2.0 * max_err / (radius**2) if radius > 0 else 0.0

                    final_clusters.append(
                        Cluster(
                            centroid=centroid,
                            F_centroid=F_c,
                            J_centroid=J_c,
                            radius=radius,
                            hessian_proxy=h_proxy,
                            cluster_size=cluster_size,  # Original size, not augmented
                            max_error=max_err,
                            mean_error=mean_err,
                        )
                    )
                else:
                    # Cluster has violations - split it using augmented data
                    # Use the already-computed augmented cluster data
                    work_X = cluster_points_list[i]

                    # Split using randomized-QR direction computed on work_X
                    q = randomized_qr_gpu(work_X, rng, ell, power_iters)
                    left_mask, right_mask = median_cut_projection(work_X, q)

                    # Split work_X into left and right parts (includes synthetic points)
                    left_points = work_X[left_mask]
                    right_points = work_X[right_mask]

                    # Generate new cluster IDs for children
                    left_id = cluster_id * 2
                    right_id = cluster_id * 2 + 1

                    children_left.append((left_points, left_id))
                    children_right.append((right_points, right_id))

        # Synchronize all GPU work for this level before pushing children
        for s in streams:
            s.synchronize()

        # Enqueue next level (point arrays)
        for L, R in zip(children_left, children_right):
            queue.append(L)
            queue.append(R)

        n_level += 1

    return final_clusters


def adaptive_clustering_gpu(
    X: cp.ndarray,
    f: Callable[[cp.ndarray], tuple[cp.ndarray, cp.ndarray]],
    n_init_clusters: int,
    min_pts_per_cluster: int,
    abs_tol: float,
    rel_tol: float,
    *,
    ell: int = ELL,
    power_iters: int = POWER_ITERS,
    num_streams: int = NUM_STREAMS,
    seed: int = SEED,
    return_numpy: bool = True,
) -> list[Cluster]:
    """
    GPU-accelerated adaptive clustering using two-means refinement.

    Args:
        X: (n, d) input data matrix.
        labels: (n,) initial cluster labels.
        f: Function returning (F, J) at X.
        abs_tol, rel_tol: Tolerances for cluster violation detection.
        min_pts_per_cluster: Minimum points per cluster before splitting.
        ell: Sketch size for randomized-QR.
        power_iters: Number of power iterations for randomized-QR.
        num_streams: Number of CUDA streams to use.
        seed: Random seed for reproducibility.
        return_numpy: If True, return NumPy arrays instead of CuPy arrays.

    Returns:
        List of final clusters with centroids, function values, Jacobians, etc.
    """
    X = cp.asarray(X, dtype=CP_DTYPE)

    km = cuKMeans(n_clusters=n_init_clusters, random_state=seed)
    labels = km.fit_predict(X)

    # Run adaptive two-means refinement
    clusters = refine_adaptive_clusters(
        X=X,
        labels=labels,
        f=f,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        min_pts_per_cluster=min_pts_per_cluster,
        ell=ell,
        power_iters=power_iters,
        num_streams=num_streams,
        seed=seed,
    )

    if return_numpy:
        for i, c in enumerate(clusters):
            clusters[i] = Cluster(
                centroid=cp.asnumpy(c.centroid),
                F_centroid=cp.asnumpy(c.F_centroid),
                J_centroid=cp.asnumpy(c.J_centroid),
                radius=c.radius,
                hessian_proxy=c.hessian_proxy,
                cluster_size=c.cluster_size,
                max_error=c.max_error,
                mean_error=c.mean_error,
            )

    return clusters


# --------------------------- Sample function f -----------------------
def sample_function(X: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    A sample function f that returns F and J at given X.
    f(X) = [x1^2 + x2^2, x1 - x2 + x3]
    grad(X) = [[2*x1, 2*x2, 0],
               [1, -1, 1]]

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

    g1 = cp.stack([2 * x1, 2 * x2, cp.zeros_like(x1)], axis=1)
    g2 = cp.stack([cp.ones_like(x1), -cp.ones_like(x1), cp.ones_like(x1)], axis=1)
    J = cp.stack([g1, g2], axis=1)  # (n, FD, D)
    return F, J


# --------------------------- MAIN -----------------------------------
def main(
    N_SAMPLES: int,
    D: int,
    N_INIT_CLUSTERS: int,
    MIN_PTS_PER_CLUSTER: int,
    ABS_TOL: float,
    REL_TOL: float,
) -> list[Cluster]:
    assert cp.is_available(), "GPU not available."
    rng = cp.random.default_rng(SEED)
    assert rng is not None, "Something went wrong with RNG initialization."
    X = rng.random((N_SAMPLES, D), dtype=CP_DTYPE)

    print(f"Running on GPU. N_SAMPLES={N_SAMPLES:,}")
    clusters = adaptive_clustering_gpu(
        X=X,
        f=sample_function,
        n_init_clusters=N_INIT_CLUSTERS,
        min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
        abs_tol=ABS_TOL,
        rel_tol=REL_TOL,
        return_numpy=False,
    )
    print(f"Total final clusters formed: {len(clusters)}")
    return clusters


if __name__ == "__main__":
    N_SAMPLES = 10_000
    D = 3
    N_INIT_CLUSTERS = 100
    MIN_PTS_PER_CLUSTER = 50
    ABS_TOL = 1e-4
    REL_TOL = 1e-4
    print(f"ABS_TOL={ABS_TOL}, REL_TOL={REL_TOL}")

    tic = time.perf_counter()
    clusters = main(
        N_SAMPLES, D, N_INIT_CLUSTERS, MIN_PTS_PER_CLUSTER, ABS_TOL, REL_TOL
    )
    toc = time.perf_counter()
    print(f"Adaptive clustering completed in {toc - tic:0.2f} seconds.")

    rng = cp.random.default_rng(SEED)
    assert rng is not None, "Something went wrong with RNG initialization."

    n_test = 100_000

    counter = 0
    # test the centroids and their errors
    for i, c in enumerate(clusters):
        centroid, F_c, J_c, radius, h_proxy = (
            c.centroid,
            c.F_centroid,
            c.J_centroid,
            c.radius,
            c.hessian_proxy,
        )

        # Convert numpy arrays back to cupy for computation
        centroid_cp = cp.asarray(centroid, dtype=CP_DTYPE)
        F_c_cp = cp.asarray(F_c, dtype=CP_DTYPE)
        J_c_cp = cp.asarray(J_c, dtype=CP_DTYPE)

        # Pick random points in the cluster ball to test error
        u = rng.standard_normal(size=(n_test, D), dtype=CP_DTYPE)
        u = u / cp.linalg.norm(u, axis=1, keepdims=True)
        test_pts = centroid_cp[0] + radius * rng.random((n_test, 1), dtype=CP_DTYPE) * u
        F_test, _ = sample_function(test_pts)
        err = taylor_errors(test_pts, F_test, centroid_cp, F_c_cp, J_c_cp)
        max_err = float(err.max())
        if max_err > max(ABS_TOL, REL_TOL * cp.linalg.norm(F_c_cp)):
            counter += 1

    perct_violate = 100.0 * counter / len(clusters)
    print(
        f"Total clusters violating tolerance on test points: ({counter} / {len(clusters)}) = {perct_violate:0.2f}%"
    )

    avg_cluster_size = sum([c.cluster_size for c in clusters]) / len(clusters)
    print(f"Average cluster size: {avg_cluster_size:0.2f} points per cluster")
