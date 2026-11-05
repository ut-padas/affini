"""
Time AFINNISurrogate.predict() on a Rosenbrock problem.

Uses only the public train/predict API, so the same script runs on any commit.
To compare two commits (e.g. before and after a refactor), run it on each with
--save-predictions on the first and --compare on the second:

    git checkout <old>; uv run python scripts/bench_predict.py --save-predictions old.npz
    git checkout <new>; uv run python scripts/bench_predict.py --compare old.npz

Timings use the median of --repeats calls after one warm-up call; on a GPU the
device is synchronized around each call.
"""

import argparse
import logging
import time
from functools import partial

import numpy as np

from src import IS_CUPY_BACKEND, _to_host, xp
from src.adaptive_clustering_utils import SurrogateMode
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad
from src.surrogate import AFINNISurrogate


def _sync():
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    mode = SurrogateMode(args.mode)
    f = partial(rosenbrock, a=100.0)
    needs_grad = (
        mode.value.startswith("taylor")
        or mode == SurrogateMode.AFFINE_COV_HESS_EXACT_GRAD
    )
    g = partial(rosenbrock_grad, a=100.0) if needs_grad else None

    rng = np.random.default_rng(args.seed)
    X = xp.asarray(rng.uniform(-2.0, 3.0, (args.n_train, args.dim)))
    Xq = xp.asarray(rng.uniform(-2.0, 3.0, (args.n_query, args.dim)))

    surrogate = AFINNISurrogate()
    t0 = time.perf_counter()
    surrogate.train(
        X,
        f_fn=f,
        g_fn=g,
        mode=mode,
        ATOL=args.tol,
        RTOL=args.tol,
        SEED=args.seed,
        MIN_PTS_PER_CLUSTER=int(100 * 1.3**args.dim),
        NUM_INITIAL_CLUSTERS=int(25 * 1.3**args.dim),
        USE_WORK_STEALING=False,
    )
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
        np.savez(args.save_predictions, pred=pred, unguaranteed=unguaranteed)
        print(f"saved predictions to {args.save_predictions}")
    if args.compare:
        ref = np.load(args.compare)
        same = np.array_equal(ref["pred"], pred) and np.array_equal(
            ref["unguaranteed"], unguaranteed
        )
        max_diff = (
            float(np.max(np.abs(ref["pred"] - pred)))
            if ref["pred"].shape == pred.shape
            else float("nan")
        )
        print(
            f"compare with {args.compare}: identical={same}  max |diff|={max_diff:.3e}"
        )


if __name__ == "__main__":
    main()
