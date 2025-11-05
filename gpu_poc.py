import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import cupy as cp
from cuml.cluster import KMeans as cuKMeans

# --------------------------- Config ---------------------------------
props = cp.cuda.runtime.getDeviceProperties(0)
num_sms = props["multiProcessorCount"]

SEED = 0
CP_DTYPE = cp.float64
N_SAMPLES = 1_000_000
D = 3  # input dim
FD = 2  # output dim
N_INIT_CLUSTERS = 5
ABS_TOL = 0.05
REL_TOL = 0.01
MIN_PTS_PER_CLUSTER = 5
NUM_STREAMS = num_sms  # or 2 * num_sms
ELL = 8  # randomized-QR sketch size
POWER_ITERS = 1  # randomized-QR power iterations


# --------------------------- Sample function f -----------------------
def sample_function(X: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    X: (n, D)
    Returns:
      F: (n, FD)
      J: (n, FD, D)
    Example: f(x) = [x1^2 + x2^2, x1 - x2 + x3]
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    f1 = x1**2 + x2**2
    f2 = x1 - x2 + x3
    F = cp.stack([f1, f2], axis=1)

    g1 = cp.stack([2 * x1, 2 * x2, cp.zeros_like(x1)], axis=1)
    g2 = cp.stack([cp.ones_like(x1), -cp.ones_like(x1), cp.ones_like(x1)], axis=1)
    J = cp.stack([g1, g2], axis=1)  # (n, FD, D)
    return F, J


# --------------------------- Utilities ------------------------------
def randomized_qr_gpu(
    X: cp.ndarray,
    rng: cp.random.Generator,
    ell: int = 8,
    power_iterations: int = 1,
) -> cp.ndarray:
    """
    Return leading direction q (D,) using randomized range finder on centered data.
    X: (n, D)
    """
    XT = X.T  # (D, n)
    Dm, N = XT.shape
    omega = rng.standard_normal((N, ell), dtype=X.dtype)  # (n, ell)

    e = cp.ones((1, N), dtype=X.dtype)
    mu = (XT @ e.T) / N  # (D,1)
    X0 = XT - mu @ e  # (D,n) centered

    Y = X0 @ omega  # (D, ell)
    Q, _ = cp.linalg.qr(Y, mode="reduced")
    for _ in range(power_iterations):
        Y = X0 @ (X0.T @ Q)  # (D, ell)
        Q, _ = cp.linalg.qr(Y, mode="reduced")

    q = Q[:, 0]  # (D,)
    # normalize for numerical robustness
    q = q / cp.linalg.norm(q)
    return q


def median_cut_projection(
    X: cp.ndarray, q: cp.ndarray
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Project X on q, split by median. Returns boolean masks (left, right).
    Ensures both sides non-empty by moving one point if needed.
    """
    s = X @ q  # (n,)
    med = cp.median(s)
    left = s <= med
    right = s > med

    # Ensure both non-empty
    n_left = int(left.sum())
    n_right = int(right.sum())
    if n_left == 0:
        # move the smallest element to left
        k = int(cp.argmin(s))
        left[k] = True
        right[k] = False
    elif n_right == 0:
        k = int(cp.argmax(s))
        right[k] = True
        left[k] = False
    return left, right


def get_centroid(X: cp.ndarray) -> cp.ndarray:
    return X.mean(axis=0)


def taylor_errors(
    X: cp.ndarray, F: cp.ndarray, centroid: cp.ndarray, F_c: cp.ndarray, J_c: cp.ndarray
) -> cp.ndarray:
    """
    X: (n, D), F: (n, FD),
    centroid: (D,), F_c: (FD,), J_c: (FD, D)
    returns per-point error norms: (n,)
    """
    dx = X - centroid  # (n, D)
    # f_taylor = F_c + J_c @ dx^T per point  -> (n, FD)
    f_taylor = F_c[None, :] + cp.einsum("kd,nd->nk", J_c, dx)
    err = cp.linalg.norm(F - f_taylor, axis=1)
    return err


def cluster_violation(
    X: cp.ndarray,
    f: Callable[[cp.ndarray], Tuple[cp.ndarray, cp.ndarray]],
    abs_tol: float,
    rel_tol: float,
) -> Tuple[bool, float, float, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Returns:
      violated (bool),
      max_error (scalar),
      radius (scalar),
      centroid (D,), F_c (FD,), J_c (FD,D)
    """
    if X.shape[0] == 0:
        raise ValueError("Empty cluster encountered.")

    centroid = get_centroid(X)
    F_sub, _ = f(X)
    F_c, J_c = f(centroid[None, :])
    F_c = F_c.squeeze(0)  # (FD,)
    J_c = J_c.squeeze(0)  # (FD, D)

    err = taylor_errors(X, F_sub, centroid, F_c, J_c)
    # tolerance per point
    thr = cp.maximum(abs_tol, rel_tol * cp.linalg.norm(F_sub, axis=1))
    violated_mask = err > thr
    violated = bool(violated_mask.any())

    radius = (
        float(cp.max(cp.linalg.norm(X - centroid, axis=1))) if X.shape[0] > 0 else 0.0
    )
    max_error = float(err.max()) if X.shape[0] > 0 else 0.0
    return violated, max_error, radius, centroid, F_c, J_c


def generate_extra_points_gpu(
    sub_X: cp.ndarray, f: Callable, min_points: int, rng: cp.random.Generator
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Convex combos to reach min_points (>=2 required in sub_X).
    Returns new_X (m, D) and F(new_X) (m, FD).
    """
    Q = sub_X.shape[0]
    if Q < 2:
        raise ValueError("At least two points required to generate extra points.")

    n_extra = max(0, min_points - Q + 1)
    if n_extra == 0:
        return cp.empty((0, sub_X.shape[1]), dtype=sub_X.dtype), cp.empty(
            (0, FD), dtype=sub_X.dtype
        )

    lambdas = rng.random(n_extra, dtype=sub_X.dtype)  # in [0,1)

    idx0 = rng.integers(0, Q, size=n_extra)
    offs = rng.integers(1, Q, size=n_extra)
    idx1 = (idx0 + offs) % Q

    new_X = lambdas[:, None] * sub_X[idx0] + (1.0 - lambdas)[:, None] * sub_X[idx1]
    F_new, _ = f(new_X)
    return new_X, F_new


@dataclass
class Cluster:
    centroid: cp.ndarray  # (D,)
    F_centroid: cp.ndarray  # (FD,)
    J_centroid: cp.ndarray  # (FD, D)
    radius: float
    hessian_proxy: float


# --------------------------- BFS Adaptive 2-means --------------------
def adaptive_two_means_gpu(
    X: cp.ndarray,
    labels: cp.ndarray,
    f: Callable[[cp.ndarray], Tuple[cp.ndarray, cp.ndarray]],
    abs_tol: float = ABS_TOL,
    rel_tol: float = REL_TOL,
    min_points: int = MIN_PTS_PER_CLUSTER,
    ell: int = ELL,
    power_iters: int = POWER_ITERS,
    num_streams: int = NUM_STREAMS,
) -> List[Cluster]:

    rng = cp.random.default_rng(SEED)
    assert rng is not None, "Something wrong with RNG initialization."

    # Initialize queue with clusters from initial KMeans
    cluster_ids = cp.arange(int(labels.max()) + 1)
    queue: List[cp.ndarray] = []
    for cid in cluster_ids.tolist():
        idx = cp.where(labels == cid)[0]
        if idx.size > 0:
            queue.append(idx)

    good_clusters: List[Cluster] = []

    # BFS by levels
    while queue:
        print(f"Queue size: {len(queue)}")
        level = queue
        queue = []

        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]

        # Per-level containers for children to enqueue after sync
        children_left: List[cp.ndarray] = []
        children_right: List[cp.ndarray] = []

        for i, idx in enumerate(level):
            stream = streams[i % num_streams]
            with stream:
                sub_X = X[idx]

                # If too small, augment with convex combos for evaluation/split
                aux_X = None
                if sub_X.shape[0] < min_points:
                    new_X, _ = generate_extra_points_gpu(sub_X, f, min_points, SEED)
                    aux_X = cp.concatenate([sub_X, new_X], axis=0)
                work_X = sub_X if aux_X is None else aux_X

                # Test violation on real points only (but centroid from real points)
                violated, max_err, radius, centroid, F_c, J_c = cluster_violation(
                    sub_X, f, abs_tol, rel_tol
                )

                if not violated:
                    # hessian proxy (scalar via max across outputs, per spec)
                    h_proxy = 2.0 * max_err / (radius**2) if radius > 0 else 0.0
                    good_clusters.append(
                        Cluster(
                            centroid=centroid,
                            F_centroid=F_c,
                            J_centroid=J_c,
                            radius=radius,
                            hessian_proxy=float(h_proxy),
                        )
                    )
                else:
                    # Split using randomized-QR direction computed on work_X
                    q = randomized_qr_gpu(
                        work_X, ell=ell, power_iterations=power_iters, rng=rng
                    )
                    left_mask, right_mask = median_cut_projection(work_X, q)

                    # Keep only indices for original points (exclude aux indices)
                    n_real = sub_X.shape[0]
                    left_idx = idx[cp.where(left_mask[:n_real])[0]]
                    right_idx = idx[cp.where(right_mask[:n_real])[0]]

                    # Edge-case: if one child empty on real points, fall back to random split
                    if left_idx.size == 0 or right_idx.size == 0:
                        perm = rng.permutation(n_real)
                        half = n_real // 2
                        left_idx = idx[perm[:half]]
                        right_idx = idx[perm[half:]]

                    children_left.append(left_idx)
                    children_right.append(right_idx)

        # Synchronize all GPU work for this level before pushing children
        for s in streams:
            s.synchronize()

        # Enqueue next level
        for L, R in zip(children_left, children_right):
            if L.size == 0 or R.size == 0:
                continue
            queue.append(L)
            queue.append(R)

    return good_clusters


# --------------------------- Driver ---------------------------------
def main():
    assert cp.is_available(), "GPU not available."
    rng = cp.random.default_rng(SEED)
    assert rng is not None, "Something wrong with RNG initialization."
    X = rng.random((N_SAMPLES, D), dtype=CP_DTYPE)

    print(f"Running on GPU. N_SAMPLES={N_SAMPLES:,}")
    tic = time.perf_counter()
    F, J = sample_function(X)
    toc = time.perf_counter()
    print(f"Function & Jacobian computed in {toc - tic:0.4f} s.")

    print(f"Initial KMeans with {N_INIT_CLUSTERS} clusters...")
    tic = time.perf_counter()
    km = cuKMeans(n_clusters=N_INIT_CLUSTERS, random_state=SEED)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_
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
        min_points=MIN_PTS_PER_CLUSTER,
        ell=ELL,
        power_iters=POWER_ITERS,
        num_streams=NUM_STREAMS,
    )
    toc = time.perf_counter()
    print(f"Adaptive refinement finished in {toc - tic:0.4f} s.")
    print(f"Total clusters formed: {len(clusters)}")

    CLUSTERS = [
        [g.centroid, g.F_centroid, g.J_centroid, g.radius, g.hessian_proxy]
        for g in clusters
    ]
    #     for g in good
    # ]
    # for g in GOOD_CLUSTERS:
    #     print(f"Centroid: {g[0]}, Radius: {g[3]:0.4e}, Hessian proxy: {g[4]:0.4e}")
    #     print(f"  F_centroid: {g[1]}")
    #     print(f"  J_centroid: {g[2]}")
    return CLUSTERS


if __name__ == "__main__":
    clusters = main()
    rng = cp.random.default_rng(SEED)
    assert rng is not None, "Something wrong with RNG initialization."

    n_test = 100_000

    counter = 0
    # test the centroids and their errors
    for i, c in enumerate(clusters):
        centroid, F_c, J_c, radius, h_proxy = c

        # Pick random points in the cluster ball to test error
        u = rng.standard_normal(size=(n_test, D), dtype=CP_DTYPE)
        u = u / cp.linalg.norm(u, axis=1, keepdims=True)
        test_pts = (
            centroid[None, :] + radius * rng.random((n_test, 1), dtype=CP_DTYPE) * u
        )
        F_test, _ = sample_function(test_pts)
        err = taylor_errors(test_pts, F_test, centroid, F_c, J_c)
        max_err = float(err.max())
        if max_err > max(ABS_TOL, REL_TOL * cp.linalg.norm(F_c)):
            print(
                f"  WARNING: Max error on test points = {max_err:0.4e} exceeds tolerance!"
            )
            counter += 1

    print(
        f"Total clusters violating tolerance on test points: {counter} out of {len(clusters)}"
    )
