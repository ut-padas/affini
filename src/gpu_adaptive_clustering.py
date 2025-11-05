import time
from dataclasses import dataclass
from typing import Callable

import cupy as cp
from cuml.cluster import KMeans as cuKMeans
from tqdm import tqdm

try:
    _HAS_GPU = cp.is_available()
except Exception as e:
    print("You do not have a working CuPy installation/GPU setup.")
    raise e

# --------------------------- Config ---------------------------------
props = cp.cuda.runtime.getDeviceProperties(0)
num_sms = props["multiProcessorCount"]

SEED = 0
CP_DTYPE = cp.float64
NUM_STREAMS = 2 * num_sms
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
    # # Ensure both non-empty
    # n_left = int(left.sum())
    # n_right = int(right.sum())
    # if n_left == 0:
    #     # move the smallest element to left
    #     k = int(cp.argmin(s))
    #     left[k] = True
    #     right[k] = False
    # elif n_right == 0:
    #     # move the largest element to right
    #     k = int(cp.argmax(s))
    #     right[k] = True
    #     left[k] = False
    return left, right


def get_centroid(X: cp.ndarray) -> cp.ndarray:
    """
    Given points belonging to a cluster, compute centroid.

    Args:
        X: (n, D) data matrix

    Returns:
        centroid: (D,) centroid of the cluster
    """
    return X.mean(axis=0)


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


def cluster_violation(
    X_cluster: cp.ndarray,
    f: Callable[[cp.ndarray], tuple[cp.ndarray, cp.ndarray]],
    abs_tol: float,
    rel_tol: float,
) -> tuple[bool, float, float, float, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Check if cluster violates Taylor approximation error bounds.

    Args:
        X_cluster: (n, D) data matrix
        f: function that returns (F, J) at X
        abs_tol: absolute tolerance
        rel_tol: relative tolerance

    Returns:
        violated: whether cluster violates tolerances
        max_error: maximum error in cluster
        mean_error: mean error in cluster
        radius: cluster radius
        centroid: (1, D) cluster centroid
        F_c: (1, FD) function value at centroid
        J_c: (1, FD, D) Jacobian at centroid
    """
    cluster_size = X_cluster.shape[0]
    if cluster_size == 0:
        raise ValueError("Empty cluster encountered.")

    centroid = get_centroid(X_cluster)[None, :]  # (1, D)
    F_sub, _ = f(X_cluster)
    F_c, J_c = f(centroid)  # (1, FD), (1, FD, D)

    err = taylor_errors(X_cluster, F_sub, centroid, F_c, J_c)  # (n,)

    # Tolerance per point
    thr = cp.maximum(abs_tol, rel_tol * cp.linalg.norm(F_sub, axis=1))
    violated_mask = err > thr
    violated = bool(violated_mask.any())

    radius = (
        float(cp.max(cp.linalg.norm(X_cluster - centroid, axis=1)))
        if cluster_size > 0
        else 0.0
    )
    max_error = float(err.max()) if cluster_size > 0 else 0.0
    mean_error = float(err.mean()) if cluster_size > 0 else 0.0
    return violated, max_error, mean_error, radius, centroid, F_c, J_c


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


# --------------------------- BFS Adaptive 2-means --------------------
def adaptive_two_means_gpu(
    X: cp.ndarray,
    labels: cp.ndarray,
    f: Callable[[cp.ndarray], tuple[cp.ndarray, cp.ndarray]],
    abs_tol: float,
    rel_tol: float,
    min_pts_per_cluster: int,
    *,
    ell: int = ELL,
    power_iters: int = POWER_ITERS,
    num_streams: int = NUM_STREAMS,
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

    Returns:
        final_clusters: list of Cluster objects representing final clusters
    """

    rng = cp.random.default_rng(SEED)
    assert rng is not None, "Something went wrong with RNG initialization."

    # Initialize queue with point arrays from initial KMeans
    cluster_ids = cp.arange(int(labels.max()) + 1)
    queue: list[cp.ndarray] = []
    for cid in cluster_ids.tolist():
        idx = cp.where(labels == cid)[0]
        if idx.size > 0:
            queue.append(X[idx])  # Store the actual points

    final_clusters: list[Cluster] = []
    n_level = 0

    # BFS by levels
    while queue:
        print(f"Level {n_level}: {len(queue)} clusters to process.")
        level = queue
        queue = []

        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]

        # Per-level containers for children to enqueue after sync
        children_left: list[cp.ndarray] = []
        children_right: list[cp.ndarray] = []

        for i, X_cluster in enumerate(tqdm(level, desc="Processing clusters")):
            stream = streams[i % num_streams]
            with stream:
                cluster_size = X_cluster.shape[0]
                # If too small, augment with convex combos for evaluation/split
                aux_X = None
                if cluster_size < min_pts_per_cluster:
                    new_X = generate_extra_points(X_cluster, min_pts_per_cluster, rng)
                    aux_X = cp.concatenate([X_cluster, new_X], axis=0)
                work_X = X_cluster if aux_X is None else aux_X

                # Test violation amongst the working set
                violated, max_err, mean_err, radius, centroid, F_c, J_c = (
                    cluster_violation(work_X, f, abs_tol, rel_tol)
                )

                if not violated:
                    # hessian proxy (scalar)
                    h_proxy = 2.0 * max_err / (radius**2) if radius > 0 else 0.0
                    # Store final cluster (based on work_X, which may include synthetic points)
                    final_clusters.append(
                        Cluster(
                            centroid=centroid,
                            F_centroid=F_c,
                            J_centroid=J_c,
                            radius=radius,
                            hessian_proxy=h_proxy,
                            cluster_size=cluster_size,
                            max_error=max_err,
                            mean_error=mean_err,
                        )
                    )
                else:
                    # Split using randomized-QR direction computed on work_X
                    q = randomized_qr_gpu(work_X, rng, ell, power_iters)
                    left_mask, right_mask = median_cut_projection(work_X, q)

                    # Split work_X into left and right parts (includes synthetic points)
                    left_points = work_X[left_mask]
                    right_points = work_X[right_mask]

                    children_left.append(left_points)
                    children_right.append(right_points)

        # Synchronize all GPU work for this level before pushing children
        for s in streams:
            s.synchronize()

        # Enqueue next level (point arrays)
        for L, R in zip(children_left, children_right):
            queue.append(L)
            queue.append(R)

        n_level += 1

    return final_clusters


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


# --------------------------- Driver ---------------------------------
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
    print(f"Initial KMeans with {N_INIT_CLUSTERS} clusters...")
    tic = time.perf_counter()
    km = cuKMeans(n_clusters=N_INIT_CLUSTERS, random_state=SEED)
    labels = km.fit_predict(X)
    toc = time.perf_counter()
    print(f"KMeans completed in {toc - tic:0.4f} s.")

    # Run adaptive two-means with streaming
    tic = time.perf_counter()
    clusters = adaptive_two_means_gpu(
        X,
        labels,
        sample_function,
        abs_tol=ABS_TOL,
        rel_tol=REL_TOL,
        min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
        ell=ELL,
        power_iters=POWER_ITERS,
        num_streams=NUM_STREAMS,
    )
    toc = time.perf_counter()
    print(f"Adaptive refinement finished in {toc - tic:0.4f} s.")
    print(f"Total clusters formed: {len(clusters)}")
    return clusters


if __name__ == "__main__":
    N_SAMPLES = 1_000_000
    D = 3
    N_INIT_CLUSTERS = 100
    MIN_PTS_PER_CLUSTER = 50
    ABS_TOL = 1e-2
    REL_TOL = 1e-2

    clusters = main(
        N_SAMPLES, D, N_INIT_CLUSTERS, MIN_PTS_PER_CLUSTER, ABS_TOL, REL_TOL
    )
    rng = cp.random.default_rng(SEED)
    assert rng is not None, "Something went wrong with RNG initialization."

    n_test = 100_000

    counter = 0
    # test the centroids and their errors
    for i, c in tqdm(
        enumerate(clusters), total=len(clusters), desc="Verifying clusters"
    ):
        centroid, F_c, J_c, radius, h_proxy = (
            c.centroid,
            c.F_centroid,
            c.J_centroid,
            c.radius,
            c.hessian_proxy,
        )

        # Pick random points in the cluster ball to test error
        u = rng.standard_normal(size=(n_test, D), dtype=CP_DTYPE)
        u = u / cp.linalg.norm(u, axis=1, keepdims=True)
        test_pts = centroid[0] + radius * rng.random((n_test, 1), dtype=CP_DTYPE) * u
        F_test, _ = sample_function(test_pts)
        err = taylor_errors(test_pts, F_test, centroid, F_c, J_c)
        max_err = float(err.max())
        if max_err > max(ABS_TOL, REL_TOL * cp.linalg.norm(F_c)):
            counter += 1

    perct_violate = 100.0 * counter / len(clusters)
    print(
        f"Total clusters violating tolerance on test points: ({counter} / {len(clusters)}) = {perct_violate:0.2f}%"
    )

    avg_cluster_size = sum([c.cluster_size for c in clusters]) / len(clusters)
    print(f"Average cluster size: {avg_cluster_size:0.2f} points per cluster")
