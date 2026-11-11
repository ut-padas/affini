"""
Accuracy and runtime of AFINNI mode on fixed problems, for comparing two
checkouts (``PYTHONPATH`` selects which one runs).

It only uses the public API that older commits also have, so the same file can run
on a base commit and on a branch::

    # in each checkout (a git worktree for the base), with the same interpreter:
    AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. python scripts/afinni_benchmark.py
    AFINNI_DEVICE=cpu AFINNI_MPI=on PYTHONPATH=. mpiexec -n 4 python scripts/afinni_benchmark.py --stealing

Prints one ``BENCH`` JSON line on rank 0 with, per problem, the leaf and evaluation
counts, the share of test points outside the trust regions (``pout``), the share inside
and within tolerance, test errors, and the median and minimum training time over
``--repeats`` runs. Accuracy must match the base exactly on CPU (to rounding on GPU);
compare times by alternating runs, since single timings are noisy.
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from src import xp  # noqa: E402
from src.adaptive_clustering_utils import ClusteringConfig, Evaluator, SurrogateMode  # noqa: E402
from src.backend import MPI  # noqa: E402
from src.problems.anisotropic_gaussian import aniso_gaussian  # noqa: E402
from src.problems.rosenbrock import rosenbrock  # noqa: E402
from src.problems.vector_quadratic import vector_quadratic  # noqa: E402
from src.surrogate import AFINNISurrogate, compute_error_metrics  # noqa: E402

# name, function, dimension, training points, tolerance, mode
CASES = (
    ("rosenbrock_d4", rosenbrock, 4, 60_000, 1e-2, SurrogateMode.AFFINE_COV_HESS),
    ("vector_quadratic_d6", vector_quadratic, 6, 60_000, 1e-2, SurrogateMode.AFFINE_COV_HESS),
    ("aniso_gaussian_d4", aniso_gaussian, 4, 60_000, 5e-3, SurrogateMode.AFFINE_COV_NO_HESS),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stealing", action="store_true", help="enable work stealing (MPI runs)")
    parser.add_argument("--only", default=None, help="run one case by name")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    rank = MPI.COMM_WORLD.Get_rank()
    out = {}
    for name, f, d, n, tol, mode in CASES:
        if args.only not in (None, name):
            continue
        X = np.random.default_rng(0).uniform(-1, 1, (n, d))
        Xq = np.random.default_rng(1).uniform(-1, 1, (20_000, d))
        times = []
        for _ in range(args.repeats):
            config = ClusteringConfig(
                atol=tol, rtol=tol, mode=mode, num_initial_clusters=50, min_pts_per_cluster=50,
                max_levels=30, use_work_stealing=args.stealing, seed=0,
            )  # fmt: skip
            model = AFINNISurrogate()
            MPI.COMM_WORLD.Barrier()
            t = time.perf_counter()
            model.train(X, config, Evaluator(f_fn=f))
            times.append(time.perf_counter() - t)
        pred, outside = model.predict(Xq)
        if rank == 0:
            m = compute_error_metrics(pred, f(xp.asarray(Xq)), outside, atol=tol, rtol=tol)
            out[name] = {
                "leaves": int(model.N_src), "evals": int(model.profiling_stats["total_f_evals"]),
                "pout": m["perct_out"], "in_good_pct": m["perct_in_good"],
                "all_linf_rel": m["all"]["linf_rel_error"], "all_l2_rel": m["all"]["l2_rel_error"],
                "train_s_median": float(np.median(times)), "train_s_min": float(np.min(times)),
            }  # fmt: skip
    if rank == 0:
        print("BENCH " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
