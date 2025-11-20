import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
from tqdm import tqdm

from src.adaptive_clustering import adaptive_cluster_algorithm
from vectorized_gpu_ac import adaptive_cluster_algorithm_gpu


def rosenbrock(X: cp.ndarray) -> cp.ndarray:
    """
    Computes the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix

    Returns:
        F: (n, 1) function values at X
    """
    x_i = X[:, :-1]
    x_next = X[:, 1:]
    F = cp.sum(100 * (x_next - x_i**2) ** 2 + (1 - x_i) ** 2, axis=1, keepdims=True)
    return F


def rosenbrock_cpu(X: np.ndarray) -> np.ndarray:
    x_i = X[:, :-1]
    x_next = X[:, 1:]
    F = np.sum(100 * (x_next - x_i**2) ** 2 + (1 - x_i) ** 2, axis=1, keepdims=True)
    return F


def rosenbrock_grad(X: cp.ndarray) -> cp.ndarray:
    """
    Computes the gradient of the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix

    Returns:
        J: (n, 1, d) Jacobian at X
    """
    n_samples, n_dims = X.shape
    J = cp.zeros((n_samples, 1, n_dims), dtype=X.dtype)

    # The term (1 - x_i)^2 only exists for i = 0 to d-2
    J[:, 0, :-1] += 2 * (X[:, :-1] - 1)

    # Term from 100*(x_{i+1} - x_i^2)^2 for i < d-1 (derivative w.r.t x_i)
    # This affects 0 to d-2
    J[:, 0, :-1] -= 400 * X[:, :-1] * (X[:, 1:] - X[:, :-1] ** 2)

    # Term from 100*(x_i - x_{i-1}^2)^2 for i > 0 (derivative w.r.t x_{i+1})
    # This affects 1 to d-1
    J[:, 0, 1:] += 200 * (X[:, 1:] - X[:, :-1] ** 2)

    return J


def rosenbrock_grad_cpu(X: np.ndarray) -> np.ndarray:
    n_samples, n_dims = X.shape
    J = np.zeros((n_samples, 1, n_dims), dtype=X.dtype)

    # The term (1 - x_i)^2 only exists for i = 0 to d-2
    J[:, 0, :-1] += 2 * (X[:, :-1] - 1)

    # Term from 100*(x_{i+1} - x_i^2)^2 for i < d-1 (derivative w.r.t x_i)
    # This affects 0 to d-2
    J[:, 0, :-1] -= 400 * X[:, :-1] * (X[:, 1:] - X[:, :-1] ** 2)

    # Term from 100*(x_i - x_{i-1}^2)^2 for i > 0 (derivative w.r.t x_{i+1})
    # This affects 1 to d-1
    J[:, 0, 1:] += 200 * (X[:, 1:] - X[:, :-1] ** 2)

    return J


dims = [3, 20, 100]
n_sizes = [1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7]
tols = [1e-1, 1e-2, 1e-3]

N_INIT_CLUSTERS = 1000
MIN_PTS_PER_CLUSTER = 50

CWD = Path.cwd()
TIMING_DIR = CWD / "timing_results"
TIMING_DIR.mkdir(exist_ok=True, parents=True)

# Measure time
for dim in tqdm(dims, desc="Dimensions"):
    for tol in tqdm(tols, desc="Tolerances", leave=False):
        for n_size in tqdm(n_sizes, desc="Sizes", leave=False):
            n_size = int(n_size)
            # Skip for large sizes and small tolerances to avoid long runtimes / CUDA OOM
            if (n_size > 5e6 and tol < 1e-2) or (dim >= 100 and tol < 1e-2):
                print(
                    f"Skipping dim={dim}, n_size={n_size}, tol={tol} due to expected long runtime/CUDA OOM."
                )
                continue

            # Output json name
            output_json = TIMING_DIR / f"timing_gpu_ac_dim{dim}_n{n_size}_tol{tol}.json"
            if output_json.exists():
                print(f"Skipping existing file: {output_json}")
                continue

            # Generate random data
            X_train = cp.random.rand(n_size, dim, dtype=cp.float64)

            # Time the GPU adaptive clustering
            start_time = time.perf_counter()
            converged_points, metadata = adaptive_cluster_algorithm_gpu(
                X_train=X_train,
                f=rosenbrock,
                grad=rosenbrock_grad,
                num_initial_clusters=N_INIT_CLUSTERS,
                min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
                atol=tol,
                rtol=tol,
                debug=True,
                max_clusters_est=100_000_000 // dim,
            )
            gpu_num_clusters = len(metadata[0])
            gpu_time = time.perf_counter() - start_time

            # Time the CPU adaptive clustering
            X_train_cpu = cp.asnumpy(X_train)
            start_time = time.perf_counter()
            clusters = adaptive_cluster_algorithm(
                X_train=X_train_cpu,
                f=rosenbrock_cpu,
                grad_f=rosenbrock_grad_cpu,
                cupy_f=rosenbrock,
                atol=tol,
                rtol=tol,
                num_initial_clusters=N_INIT_CLUSTERS,
                min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
                seed=0,
                should_generate_extra_points=True,
                debug=True,
                two_means_split_method="median",
            )
            cpu_num_clusters = len(clusters)
            cpu_time = time.perf_counter() - start_time

            # Save results to JSON
            result = {
                "dimension": dim,
                "n_size": n_size,
                "tolerance": tol,
                "gpu_time_seconds": gpu_time,
                "gpu_num_clusters": gpu_num_clusters,
                "cpu_time_seconds": cpu_time,
                "cpu_num_clusters": cpu_num_clusters,
            }
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
