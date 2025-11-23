import json
import time
from pathlib import Path

import cupy as cp
from tqdm import tqdm

from src.adaptive_clustering import adaptive_cluster_algorithm
from src.gpu_adaptive_clustering import adaptive_cluster_algorithm_gpu
from src.problems.rosenbrock import (
    rosenbrock_cupy,
    rosenbrock_grad_cupy,
    rosenbrock_grad_numpy,
    rosenbrock_numpy,
)

if __name__ == "__main__":
    dims = [3, 20]
    n_sizes = [1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7]
    tols = [1e-1, 1e-2, 1e-3]

    N_INIT_CLUSTERS = 100
    MIN_PTS_PER_CLUSTER = 100

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
                output_json = (
                    TIMING_DIR / f"timing_gpu_ac_dim{dim}_n{n_size}_tol{tol}.json"
                )
                if output_json.exists():
                    print(f"Skipping existing file: {output_json}")
                    continue

                # Generate random data
                X_train = cp.random.rand(n_size, dim, dtype=cp.float64)

                # Time the GPU adaptive clustering
                start_time = time.perf_counter()
                result = adaptive_cluster_algorithm_gpu(
                    X_train=X_train,
                    f=rosenbrock_cupy,
                    grad=rosenbrock_grad_cupy,
                    DATA_DIM=dim,
                    F_DIM=1,
                    NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
                    MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                    ATOL=tol,
                    RTOL=tol,
                    DEBUG=True,
                    MAX_CLUSTERS_EST=100_000_000 // dim,
                    POWER_ITERS=3,
                )
                gpu_num_clusters = len(result[0])
                gpu_time = time.perf_counter() - start_time

                # Time the CPU adaptive clustering
                X_train_cpu = cp.asnumpy(X_train)
                start_time = time.perf_counter()
                clusters = adaptive_cluster_algorithm(
                    X_train=X_train_cpu,
                    f=rosenbrock_numpy,
                    grad_f=rosenbrock_grad_numpy,
                    cupy_f=rosenbrock_cupy,
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
