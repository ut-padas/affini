"""
Time AFINNISurrogate.predict() on a Rosenbrock problem.

Uses only the public train/predict API, so the same script runs on any commit.
To compare two commits (e.g. before and after a refactor), run it from each
checkout's root with --save-predictions on the first and --compare on the second:

    PYTHONPATH=. python scripts/bench_predict.py --save-predictions old.npz   # old checkout
    PYTHONPATH=. python scripts/bench_predict.py --compare old.npz            # new checkout

--compare exits with status 1 if the leaf count, the predictions (beyond --rtol) or
the trust masks differ. Timings use the median of --repeats calls after one warm-up
call; on a GPU the device is synchronized around each call. For a full differential
check of every mode, use scripts/regression_check.py.
"""

import argparse
import logging
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))  # the checkout being measured

from src import IS_CUPY_BACKEND, _to_host, xp  # noqa: E402
from src.config import ClusteringConfig# noqa: E402
from src.core.evaluation import Evaluator# noqa: E402
from src.components.modes import SurrogateMode# noqa: E402
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad, rosenbrock_hess  # noqa: E402
from src.surrogate import AFINNISurrogate  # noqa: E402


def _sync():
    """Wait for queued GPU work, so timings include it."""
    if IS_CUPY_BACKEND:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=200_000)
    parser.add_argument("--n-query", type=int, default=50_000)
    parser.add_argument("--tol", type=float, default=0.05, help="ATOL = RTOL")
    parser.add_argument(
        "--mode",
        default=SurrogateMode.AFFINE_COV_HESS.value,
        choices=[m.value for m in SurrogateMode],
    )
    parser.add_argument("--k", type=int, default=8, help="k_nn neighbours per query")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-predictions", help="write predictions to this .npz")
    parser.add_argument("--compare", help="compare predictions with this .npz")
    parser.add_argument(
        "--rtol", type=float, default=0.0,
        help="allowed relative prediction difference for --compare (0 = identical; use ~1e-12 on GPU)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    mode = SurrogateMode(args.mode)
    f = partial(rosenbrock, a=100.0)
    needs_grad = (
        mode.value.startswith("taylor")
        or mode == SurrogateMode.AFFINE_COV_HESS_EXACT_GRAD
    )
    g = partial(rosenbrock_grad, a=100.0) if needs_grad else None
    hess = partial(rosenbrock_hess, a=100.0) if "exact_hess" in mode.value else None

    rng = np.random.default_rng(args.seed)
    X = xp.asarray(rng.uniform(-2.0, 3.0, (args.n_train, args.dim)))
    Xq = xp.asarray(rng.uniform(-2.0, 3.0, (args.n_query, args.dim)))

    surrogate = AFINNISurrogate()
    t0 = time.perf_counter()
    config = ClusteringConfig(
        mode=mode,
        atol=args.tol,
        rtol=args.tol,
        seed=args.seed,
        min_pts_per_cluster=int(100 * 1.3**args.dim),
        num_initial_clusters=int(25 * 1.3**args.dim),
        use_work_stealing=False,
    )
    surrogate.train(X, config, Evaluator(f_fn=f, g_fn=g, hessian_fn=hess))
    t_train = time.perf_counter() - t0

    surrogate.predict(Xq, k_nn=args.k)  # warm-up (kNN index, kernels)
    times = []
    for _ in range(args.repeats):
        _sync()
        t = time.perf_counter()
        pred, unguaranteed = surrogate.predict(Xq, k_nn=args.k)[:2]
        _sync()
        times.append(time.perf_counter() - t)

    pred, unguaranteed = _to_host(pred), _to_host(unguaranteed)
    print(
        f"backend={'cupy' if IS_CUPY_BACKEND else 'numpy'} mode={mode.value} d={args.dim} "
        f"N_train={args.n_train:,} N_c={surrogate.N_src:,} N_query={args.n_query:,} k={args.k}"
    )
    print(f"train: {t_train:.2f} s")
    print(
        f"predict: median {np.median(times):.4f} s  (min {np.min(times):.4f}, "
        f"max {np.max(times):.4f}, {args.repeats} runs)"
    )

    if args.save_predictions:
        np.savez(args.save_predictions, pred=pred, unguaranteed=unguaranteed, n_leaves=surrogate.N_src)
        print(f"saved predictions to {args.save_predictions}")
    if args.compare and not compare(np.load(args.compare), pred, unguaranteed, surrogate.N_src, args.rtol):
        raise SystemExit(1)


def compare(ref, pred, unguaranteed, n_leaves, rtol) -> bool:
    """Print how the run differs from the saved reference; True if within ``rtol`` and identical masks."""
    problems = []
    if "n_leaves" in ref and int(ref["n_leaves"]) != n_leaves:
        problems.append(f"leaves {int(ref['n_leaves'])} -> {n_leaves}")
    if ref["pred"].shape != pred.shape:
        problems.append(f"prediction shape {ref['pred'].shape} -> {pred.shape}")
    else:
        if not np.array_equal(np.isfinite(pred), np.isfinite(ref["pred"])):
            problems.append("non-finite predictions differ")
        rel = np.abs(ref["pred"] - pred) / np.maximum(np.abs(ref["pred"]), 1e-300)
        worst = float(np.nanmax(rel)) if rel.size else 0.0
        print(f"max relative prediction difference: {worst:.3e}")
        if worst > rtol:
            problems.append(f"predictions differ by {worst:.3e} > rtol={rtol:g}")
        n_mask = int(np.sum(ref["unguaranteed"] != unguaranteed))
        if n_mask:
            problems.append(f"{n_mask} trust-mask entries differ")
    print("compare:", "PASS" if not problems else "FAIL: " + "; ".join(problems))
    return not problems


if __name__ == "__main__":
    main()
