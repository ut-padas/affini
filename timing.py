import json
import time
from pathlib import Path

import cupy as cp
from tqdm import tqdm

from src.adaptive_clustering import adaptive_cluster_algorithm
from src.gpu_adaptive_clustering import adaptive_cluster_algorithm_gpu
from src.logging_config import setup_logging
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

if __name__ == "__main__":
    setup_logging()
    dims = [3, 20, 100]
    n_sizes = [1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8]
    tols = [1e-1, 1e-2, 1e-3]

    N_INIT_CLUSTERS = 100
    MIN_PTS_PER_CLUSTER = 100

    CWD = Path.cwd()
    TIMING_DIR = CWD / "timing_results"
    SUB_TEMP_DIR = TIMING_DIR / "individual_runs"
    SUB_TEMP_DIR.mkdir(exist_ok=True, parents=True)

    # Measure time
    for dim in tqdm(dims, desc="Dimensions"):
        for tol in tqdm(tols, desc="Tolerances", leave=False):
            for n_size in tqdm(n_sizes, desc="Sizes", leave=False):
                n_size = int(n_size)

                # Output json name
                gpu_output_json = SUB_TEMP_DIR / f"gpu_dim{dim}_n{n_size}_tol{tol}.json"
                if not gpu_output_json.exists():
                    # Generate random data
                    cp.random.seed(0)
                    X_train = cp.random.rand(n_size, dim, dtype=cp.float64)

                    # Time the GPU adaptive clustering
                    start_time = time.perf_counter()
                    result = adaptive_cluster_algorithm_gpu(
                        X_train=X_train,
                        f=rosenbrock,
                        grad=rosenbrock_grad,
                        DATA_DIM=dim,
                        F_DIM=1,
                        NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
                        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                        ATOL=tol,
                        RTOL=tol,
                        MAX_CLUSTERS_EST=100_000_000 // dim,
                        POWER_ITERS=2,
                        LOG_MEM_STATS=True,
                    )
                    gpu_num_clusters = len(result[0])
                    gpu_time = time.perf_counter() - start_time

                    with open(gpu_output_json, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "dimension": dim,
                                "n_size": n_size,
                                "tolerance": tol,
                                "gpu_time_seconds": gpu_time,
                                "gpu_num_clusters": gpu_num_clusters,
                            },
                            f,
                            indent=4,
                        )
                else:
                    # Read existing file to avoid recomputation
                    with open(gpu_output_json, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    gpu_time = existing_data["gpu_time_seconds"]
                    gpu_num_clusters = existing_data["gpu_num_clusters"]

                # Skip CPU timing for very large sizes or high dimensions with low tolerance
                dim_l_10 = (dim < 10) and (
                    (tol in [1e-1, 1e-2] and n_size >= 1e8)
                    or (tol == 1e-3 and n_size >= 1e7)
                )
                dim_g_10_l_50 = (10 <= dim < 50) and (n_size >= 5e5)
                dim_g_50 = (dim >= 50) and (n_size >= 1e5)

                if dim_l_10 or dim_g_10_l_50 or dim_g_50:
                    msg = (
                        f"Skip: CPU timing for dim={dim}, n_size={n_size:2e}, tol={tol}"
                    )
                    print(msg)
                    continue

                cpu_output_json = SUB_TEMP_DIR / f"cpu_dim{dim}_n{n_size}_tol{tol}.json"
                if not cpu_output_json.exists():
                    print(f"Timing CPU for dim={dim}, n_size={n_size:2e}, tol={tol}")
                    # Time the CPU adaptive clustering
                    cp.random.seed(0)
                    X_train = cp.random.rand(n_size, dim, dtype=cp.float64)
                    X_train_cpu = cp.asnumpy(X_train)

                    start_time = time.perf_counter()
                    clusters = adaptive_cluster_algorithm(
                        X_train=X_train_cpu,
                        f=rosenbrock,
                        grad_f=rosenbrock_grad,
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
                    with open(cpu_output_json, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "dimension": dim,
                                "n_size": n_size,
                                "tolerance": tol,
                                "cpu_time_seconds": cpu_time,
                                "cpu_num_clusters": cpu_num_clusters,
                            },
                            f,
                            indent=4,
                        )
                else:
                    # Read existing file to avoid recomputation
                    with open(cpu_output_json, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    cpu_time = existing_data["cpu_time_seconds"]
                    cpu_num_clusters = existing_data["cpu_num_clusters"]

                output_fname = TIMING_DIR / f"summary_dim{dim}_n{n_size}_tol{tol}.json"
                with open(output_fname, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "dimension": dim,
                            "n_size": n_size,
                            "tolerance": tol,
                            "gpu_time_seconds": gpu_time,
                            "gpu_num_clusters": gpu_num_clusters,
                            "cpu_time_seconds": cpu_time,
                            "cpu_num_clusters": cpu_num_clusters,
                        },
                        f,
                        indent=4,
                    )
