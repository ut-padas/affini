import argparse
import logging
from functools import partial

import cupy as cp

from ics26_experiments import get_N_INIT_CLUSTERS, get_N_MIN_PER_CLUSTER, get_N_TRAIN

# Import your actual code structure
from src.gpu_adaptive_clustering import adaptive_clustering
from src.logging_utils import setup_logging
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad


# --- 1. Profiling Utils ---
class CudaTimer:
    """Context manager to measure GPU time for specific blocks."""

    def __init__(self, name):
        self.name = name
        self.start = cp.cuda.Event()
        self.end = cp.cuda.Event()

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        self.end.synchronize()
        ms = cp.cuda.get_elapsed_time(self.start, self.end)
        print(f"[Timer] {self.name}: {ms:.3f} ms")


# --- 3. Case Registry ---
CASES = {
    "3D_coarse": {"dim": 3, "tol": 1e-2, "min_pts": 371, "metric": "Linf-local"},
    "3D_medium": {"dim": 3, "tol": 1e-2, "min_pts": 1856, "metric": "Linf-local"},
    "3D_fine": {"dim": 3, "tol": 1e-2, "min_pts": 3712, "metric": "Linf-local"},
    "10D_coarse": {"dim": 10, "tol": 1e-1, "min_pts": 1856, "metric": "Linf-local"},
}


def main():
    # --- OPTIMIZATION START: Enable Memory Pool with Safety Limit ---
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    # NEW ADDITION: Set a hard limit (e.g., 90% of GPU memory)
    # This tells CuPy: "If you hit 90% VRAM usage, stop hoarding memory
    # and actually free some old blocks instead of crashing."
    free_mem, total_mem = cp.cuda.Device(0).mem_info
    pool.set_limit(size=int(total_mem * 0.9))

    parser = argparse.ArgumentParser(description="Adaptive Clustering Profiling Driver")
    parser.add_argument(
        "--case", type=str, choices=CASES.keys(), help="Predefined profiling scenario"
    )

    # Overrides
    parser.add_argument("--dim", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--min-pts", type=int)
    parser.add_argument("--metric", type=str)
    parser.add_argument("--n-train", type=int)

    args = parser.parse_args()

    # Default to medium if no case specified
    params = CASES["3D_medium"].copy()
    if args.case:
        print(f"Loading Case: {args.case}")
        params.update(CASES[args.case])

    # Apply Overrides
    if args.dim:
        params["dim"] = args.dim
    if args.tol:
        params["tol"] = args.tol
    if args.min_pts:
        params["min_pts"] = get_N_MIN_PER_CLUSTER(params["dim"])
    if args.metric:
        params["metric"] = args.metric

    # Heuristics
    N_TRAIN = get_N_TRAIN(params["dim"])
    N_INIT_CLUSTERS = get_N_INIT_CLUSTERS(params["dim"])

    print(f"--- Configuration ---")
    print(f"Dim: {params['dim']}")
    print(f"N_Train: {N_TRAIN:,}")
    print(f"Metric: {params['metric']}")
    print(f"Min Pts (Target Density): {params['min_pts']}")
    print(f"N initial clusters: {N_INIT_CLUSTERS:,}")
    print(f"---------------------")

    cp.random.seed(42)
    print("Generating Data...")
    X = cp.random.uniform(-1.0, 3.0, size=(N_TRAIN, params["dim"]), dtype=cp.float64)

    f_fn = partial(rosenbrock, a=100.0)
    g_fn = partial(rosenbrock_grad, a=100.0)

    # Warmup
    print("Warming up GPU...")
    # Using a dummy call to initialize the new memory pool
    adaptive_clustering(
        X[:1000],
        f_fn=f_fn,
        g_fn=g_fn,
        TOL=1.0,
        MIN_PTS_PER_CLUSTER=100,
        NUM_INITIAL_CLUSTERS=2,
        LOG_MEM_STATS=False,
    )
    cp.cuda.Stream.null.synchronize()

    if params["dim"] >= 5:
        BATCH_SIZE = 500_000
    else:
        BATCH_SIZE = 10_000_000

    print(">>> Starting Run <<<")
    with CudaTimer("Total Adaptive Clustering"):
        results = adaptive_clustering(
            X,
            f_fn=f_fn,
            g_fn=g_fn,
            TOL=params["tol"],
            ERROR_METRIC=params["metric"],
            MIN_PTS_PER_CLUSTER=params["min_pts"],
            NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
            BATCH_SIZE=BATCH_SIZE,
            LOG_MEM_STATS=True,
        )

    print(f"Final Clusters: {len(results['centroids'])}")


if __name__ == "__main__":
    setup_logging(console_level=logging.DEBUG)
    main()
