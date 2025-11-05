import time

import cupy as cp
from cuml.cluster import KMeans as cuKMeans

_HAS_GPU = cp.is_available()
if not _HAS_GPU:
    raise ImportError("GPU not available.")

CP_DTYPE = cp.float64
SEED = 0


def sample_function(X: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    X: (n_samples, 3)
    f(x) = [x1^2 + x2^2, x1 - x2 + x3] (n_samples, 2)
    Jacobian/Gradient: (n_samples, 2, 3)
    [[2*x1, 2*x2, 0],
        [1,    -1,  1]]
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    f1 = x1**2 + x2**2
    f2 = x1 - x2 + x3
    F = cp.stack([f1, f2], axis=1)

    g1 = cp.stack([2 * x1, 2 * x2, cp.zeros_like(x1)], axis=1)
    g2 = cp.stack([cp.ones_like(x1), -cp.ones_like(x1), cp.ones_like(x1)], axis=1)
    J = cp.stack([g1, g2], axis=1)
    return F, J


rng = cp.random.default_rng(SEED)
N_SAMPLES = 100_000
X = rng.random((N_SAMPLES, 3), dtype=CP_DTYPE)  # type: ignore

tic = time.perf_counter()
F, J = sample_function(X)
toc = time.perf_counter()
print(f"Function and Jacobian computed in {toc - tic:0.4f} seconds.")
