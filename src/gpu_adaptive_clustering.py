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

props = cp.cuda.runtime.getDeviceProperties(0)
num_sms = props["multiProcessorCount"]

SEED = 0
CP_DTYPE = cp.float64
NUM_STREAMS = num_sms
ELL = 8  # randomized-QR sketch size
POWER_ITERS = 1  # randomized-QR power iterations


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
    s = X @ q
    med = cp.median(s)
    left = s <= med
    right = s > med
    return left, right


@dataclass
class Cluster:
    """
    Data class to hold cluster information.

    Attributes:
        centroid: (1, D) cluster centroid
        F_centroid: (1, FD) function value at centroid
        J_centroid: (1, FD, D) Jacobian at centroid
        radius: cluster radius
        hessian_proxy: hessian proxy value
        cluster_size: number of points in cluster
        max_error: maximum Taylor approximation error in cluster
        mean_error: mean Taylor approximation error in cluster
    """

    centroid: cp.ndarray
    F_centroid: cp.ndarray
    J_centroid: cp.ndarray
    radius: cp.ndarray | float
    hessian_proxy: cp.ndarray | float
    cluster_size: int
    max_error: cp.ndarray | float
    mean_error: cp.ndarray | float


def refine_adaptive_clusters(
    X: cp.ndarray,
    labels: cp.ndarray,
    f: Callable,
    grad_f: Callable,
    abs_tol: float,
    rel_tol: float,
    min_pts_per_cluster: int,
    ell: int,
    power_iters: int,
    num_streams: int,
    seed: int,
    should_generate_extra_points: bool = True,
    debug: bool = False,
) -> list[Cluster]:
    """
    Memory-optimized adaptive two-means clustering with GPU streaming.
    Maximizes GPU memory reuse and minimizes array allocations.
    """

    rng = cp.random.default_rng(seed)
    assert rng is not None, "Something went wrong with RNG initialization."

    # Initialize queue with point arrays from initial KMeans
    cluster_ids = cp.arange(int(labels.max()) + 1)
    queue: list[cp.ndarray] = []
    for cid in range(len(cluster_ids)):
        idx = cp.where(labels == cid)[0]
        if idx.size > 0:
            queue.append(X[idx])

    final_clusters: list[Cluster] = []
    n_level = 0

    # Pre-allocate reusable arrays (will be resized as needed)
    max_points_estimate = X.shape[0] * 2  # Conservative estimate
    temp_array_buffer = cp.empty((max_points_estimate, X.shape[1]), dtype=X.dtype)

    start_time = time.perf_counter()
    level_start_time = time.perf_counter()

    # BFS by levels
    while queue:
        if debug and n_level > 0:
            level_time = time.perf_counter() - level_start_time
            print(
                f"Level {n_level}: processing {len(queue)} clusters (prev level took {level_time:.2f}s)"
            )

        level_start_time = time.perf_counter()
        level = queue
        queue = []

        if not level:
            break

        # Phase 1: Process clusters in-place and build consolidated data
        current_clusters = []
        cluster_point_counts = []
        level_mapping = {}
        total_points = 0
        small_clusters_direct = []

        for level_idx, cluster_pts in enumerate(level):
            cluster_size = cluster_pts.shape[0]

            if cluster_size < min_pts_per_cluster:
                if should_generate_extra_points:
                    # Generate extra points directly into cluster_pts
                    n_extra = max(1, min_pts_per_cluster - cluster_size + 1)
                    lambdas = rng.random(n_extra, dtype=cluster_pts.dtype)
                    idx0 = rng.integers(0, cluster_size, size=n_extra)
                    offs = rng.integers(1, cluster_size, size=n_extra)
                    idx1 = (idx0 + offs) % cluster_size

                    # Use view of temp buffer instead of concatenate
                    augmented_size = cluster_size + n_extra
                    work_cluster = temp_array_buffer[:augmented_size]
                    work_cluster[:cluster_size] = cluster_pts
                    work_cluster[cluster_size:] = (
                        lambdas[:, None] * cluster_pts[idx0]
                        + (1.0 - lambdas)[:, None] * cluster_pts[idx1]
                    )

                    level_mapping[level_idx] = len(current_clusters)
                    current_clusters.append(work_cluster)
                    cluster_point_counts.append(augmented_size)
                    total_points += augmented_size
                else:
                    small_clusters_direct.append(cluster_pts)
                    continue
            else:
                level_mapping[level_idx] = len(current_clusters)
                current_clusters.append(cluster_pts)
                cluster_point_counts.append(cluster_size)
                total_points += cluster_size

        # Handle small clusters directly (finalize without further processing)
        for small_cluster in small_clusters_direct:
            centroid = small_cluster.mean(axis=0, keepdims=True)
            F_c = f(centroid)
            J_c = grad_f(centroid)

            F_pts = f(small_cluster)
            dx = small_cluster - centroid
            F_taylor = F_c + cp.einsum("ijk,nk->nj", J_c, dx)
            err = cp.linalg.norm(F_pts - F_taylor, axis=1)

            final_clusters.append(
                Cluster(
                    centroid=centroid,
                    F_centroid=F_c,
                    J_centroid=J_c,
                    radius=cp.max(cp.linalg.norm(small_cluster - centroid, axis=1)),
                    hessian_proxy=2.0
                    * err.max()
                    / (
                        cp.max(cp.linalg.norm(small_cluster - centroid, axis=1)) ** 2
                        + 1e-12
                    ),
                    cluster_size=small_cluster.shape[0],
                    max_error=err.max(),
                    mean_error=err.mean(),
                )
            )

        if not current_clusters:
            continue

        # Phase 2: Consolidated batch processing - reuse arrays
        cluster_count = len(current_clusters)

        # Efficiently stack all points using pre-allocated buffer
        all_X = temp_array_buffer[:total_points]
        start_idx = 0
        for i, cluster in enumerate(current_clusters):
            end_idx = start_idx + cluster_point_counts[i]
            all_X[start_idx:end_idx] = cluster
            start_idx = end_idx

        # Create cluster labels efficiently using in-place operations
        all_cluster_labels = cp.empty(total_points, dtype=cp.int32)
        start_idx = 0
        for i, count in enumerate(cluster_point_counts):
            end_idx = start_idx + count
            all_cluster_labels[start_idx:end_idx] = i
            start_idx = end_idx

        # Compute centroids in-place
        centroids = cp.empty((cluster_count, X.shape[1]), dtype=X.dtype)
        for i, cluster in enumerate(current_clusters):
            centroids[i] = cluster.mean(axis=0)

        # Batch function evaluations (most expensive operations)
        F_centroids = f(centroids)
        J_centroids = grad_f(centroids)
        F_all = f(all_X)

        # Vectorized Taylor approximation using broadcasting (reuse arrays)
        dx_all = cp.empty_like(all_X)
        F_taylor_all = cp.empty_like(F_all)

        start_idx = 0
        for i, count in enumerate(cluster_point_counts):
            end_idx = start_idx + count
            cluster_slice = slice(start_idx, end_idx)

            # Reuse dx_all for displacement computation
            dx_all[cluster_slice] = all_X[cluster_slice] - centroids[i]

            # Reuse F_taylor_all for Taylor approximation
            F_taylor_all[cluster_slice] = F_centroids[i] + cp.einsum(
                "jk,nk->nj", J_centroids[i], dx_all[cluster_slice]
            )

            start_idx = end_idx

        # Error computation and violation detection (reuse arrays)
        err_all = cp.linalg.norm(F_all - F_taylor_all, axis=1)
        thr_all = cp.maximum(abs_tol, rel_tol * cp.linalg.norm(F_all, axis=1))
        violations_mask = err_all > thr_all

        # Identify violating clusters efficiently
        if violations_mask.any():
            violating_cluster_counts = cp.bincount(
                all_cluster_labels[violations_mask], minlength=cluster_count
            )
            violating_clusters = set(
                cp.where(violating_cluster_counts > 0)[0].get().tolist()
            )
        else:
            violating_clusters = set()

        if debug:
            print(
                f"  --> {len(violating_clusters)}/{cluster_count} clusters need splitting"
            )

        # Phase 3: Process clusters with computed statistics (reuse data)
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
        children_left = []
        children_right = []

        # Pre-compute cluster error statistics efficiently
        cluster_errors = []
        start_idx = 0
        for i, count in enumerate(cluster_point_counts):
            end_idx = start_idx + count
            cluster_err = err_all[start_idx:end_idx]
            cluster_errors.append((cluster_err.max(), cluster_err.mean()))
            start_idx = end_idx

        for level_idx, original_cluster in enumerate(level):
            if level_idx not in level_mapping:
                continue

            current_idx = level_mapping[level_idx]
            stream = streams[current_idx % num_streams]

            with stream:
                if current_idx not in violating_clusters:
                    # Create final cluster using pre-computed values
                    centroid = centroids[current_idx : current_idx + 1]
                    cluster_data = current_clusters[current_idx]

                    max_err, mean_err = cluster_errors[current_idx]
                    radius = cp.max(cp.linalg.norm(cluster_data - centroid, axis=1))
                    h_proxy = 2.0 * max_err / (radius**2 + 1e-12)

                    final_clusters.append(
                        Cluster(
                            centroid=centroid,
                            F_centroid=F_centroids[current_idx : current_idx + 1],
                            J_centroid=J_centroids[current_idx : current_idx + 1],
                            radius=radius,
                            hessian_proxy=h_proxy,
                            cluster_size=cluster_data.shape[0],
                            max_error=max_err,
                            mean_error=mean_err,
                        )
                    )
                else:
                    # Split cluster using randomized-QR
                    work_X = current_clusters[current_idx]
                    q = randomized_qr_gpu(work_X, rng, ell, power_iters)
                    s = work_X @ q
                    med = cp.median(s)
                    left_mask = s <= med
                    right_mask = s > med

                    children_left.append(work_X[left_mask])
                    children_right.append(work_X[right_mask])

        # Synchronize streams
        for s in streams:
            s.synchronize()

        # Enqueue children for next level
        queue.extend(children_left)
        queue.extend(children_right)
        n_level += 1

    delta = time.perf_counter() - start_time
    print(f"Clustering Completed. Total levels: {n_level}, time taken: {delta:.2f}s")
    return final_clusters


def adaptive_cluster_algorithm_gpu(
    X: cp.ndarray,
    f: Callable,
    grad_f: Callable,
    num_initial_clusters: int,
    min_pts_per_cluster: int,
    atol: float,
    rtol: float,
    *,
    ell: int = ELL,
    power_iters: int = POWER_ITERS,
    num_streams: int = NUM_STREAMS,
    seed: int = SEED,
    should_generate_extra_points: bool = True,
    return_numpy: bool = True,
    debug: bool = False,
) -> list[Cluster]:
    """
    GPU-accelerated adaptive clustering using two-means refinement.

    Args:
        X: (n, d) input data matrix.
        f: Function returning F for given X.
        grad_f: Function returning J for given X.
        num_initial_clusters: Number of initial clusters for KMeans.
        min_pts_per_cluster: Minimum points per cluster before splitting.
        atol, rtol: Tolerances for cluster violation detection.
        ell: Sketch size for randomized-QR.
        power_iters: Number of power iterations for randomized-QR.
        num_streams: Number of CUDA streams to use.
        seed: Random seed for reproducibility.
        should_generate_extra_points: If True, generate extra points for small clusters;
            if False, directly finalize clusters with fewer than min_pts_per_cluster points.
        return_numpy: If True, return NumPy arrays instead of CuPy arrays.
        debug: If True, print debug information.

    Returns:
        List of final clusters with centroids, function values, Jacobians, etc.
    """
    X = cp.asarray(X, dtype=CP_DTYPE)

    km = cuKMeans(n_clusters=num_initial_clusters, random_state=seed)
    labels = km.fit_predict(X)

    # Run adaptive two-means refinement
    print(f"Should generate extra points: {should_generate_extra_points}")
    clusters = refine_adaptive_clusters(
        X=X,
        labels=labels,
        f=f,
        grad_f=grad_f,
        abs_tol=atol,
        rel_tol=rtol,
        min_pts_per_cluster=min_pts_per_cluster,
        ell=ell,
        power_iters=power_iters,
        num_streams=num_streams,
        seed=seed,
        should_generate_extra_points=should_generate_extra_points,
        debug=debug,
    )

    if return_numpy:
        for i, c in enumerate(clusters):
            clusters[i] = Cluster(
                centroid=cp.asnumpy(c.centroid),
                F_centroid=cp.asnumpy(c.F_centroid),
                J_centroid=cp.asnumpy(c.J_centroid),
                radius=float(c.radius),
                hessian_proxy=float(c.hessian_proxy),
                cluster_size=c.cluster_size,
                max_error=float(c.max_error),
                mean_error=float(c.mean_error),
            )

    return clusters


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
    clusters = adaptive_cluster_algorithm_gpu(
        X=X,
        f=sample_function,
        grad_f=sample_function_grad,
        num_initial_clusters=N_INIT_CLUSTERS,
        min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
        atol=ABS_TOL,
        rtol=REL_TOL,
        return_numpy=False,
        debug=True,
        should_generate_extra_points=False,
    )
    print(f"Total final clusters formed: {len(clusters)}")
    return clusters


if __name__ == "__main__":
    N_SAMPLES = 1_000_000  # Match BFS version
    N_INIT_CLUSTERS = 100
    MIN_PTS_PER_CLUSTER = 5  # Match BFS version
    ABS_TOL = 0.01  # Match BFS version
    REL_TOL = 0.01  # Match BFS version
    print(f"{ABS_TOL=}, {REL_TOL=}")
    print(f"{N_INIT_CLUSTERS=}, {MIN_PTS_PER_CLUSTER=}")
    D = 3

    tic = time.perf_counter()
    clusters = main(
        N_SAMPLES, D, N_INIT_CLUSTERS, MIN_PTS_PER_CLUSTER, ABS_TOL, REL_TOL
    )
    toc = time.perf_counter()
    print(f"Adaptive clustering completed in {toc - tic:0.2f} seconds.")
    raise SystemExit

    def taylor_errors(
        X: cp.ndarray,
        F: cp.ndarray,
        centroid: cp.ndarray,
        F_c: cp.ndarray,
        J_c: cp.ndarray,
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
        max_err_gpu = err.max()
        max_err = float(max_err_gpu)
        if max_err > max(ABS_TOL, REL_TOL * float(cp.linalg.norm(F_c_cp))):
            counter += 1

    perct_violate = 100.0 * counter / len(clusters)
    print(
        f"Total clusters violating tolerance on test points: ({counter} / {len(clusters)}) = {perct_violate:0.2f}%"
    )

    avg_cluster_size = sum([c.cluster_size for c in clusters]) / len(clusters)
    print(f"Average cluster size: {avg_cluster_size:0.2f} points per cluster")
