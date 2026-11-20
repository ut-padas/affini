"""
Time inference of AFINNI (kNN over the leaf centroids, reranked by trust region) against ART
(routing down the tree) on Rosenbrock, spherical_piecewise and the vector-valued Lorenz-96 flow
map, in several dimensions.

Run from the repository root, with the backend set explicitly::

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off .venv/bin/python scripts/bench_inference.py
    SURROGATE_DEVICE=cpu SURROGATE_MPI=off .venv/bin/python scripts/bench_inference.py \
        --problem rosenbrock --dim 2 3
    SURROGATE_DEVICE=cpu SURROGATE_MPI=off .venv/bin/python scripts/bench_inference.py --list

Both methods start from the same ``ceil(50 (d + 1))`` uniform points in one root (AFINNI gets
all-zero ``initial_labels``, so it skips the initial clustering, and ART a single box root),
train at the same low tolerance and stop at the same level limit (``--max-levels``, 15), so the
limit rather than the tolerance sets the size of both models and their leaf counts are roughly
comparable. A cluster still above tolerance at the limit keeps its fitted model as a leaf in both
(AFINNI with ``fallback_leaves=True``, ART always), so neither model has holes where its
refinement stopped. Each training reports its leaves, oracle evaluations and wall time; each model is
then scored on ``--n-test`` (50,000) uniform test points shared by both, and its inference is
timed over the same points: one untimed warm-up call, then ``--repeats`` timed calls, the
device synchronized around each on a GPU. Inference uses the single-process paths
(``predict_local``, ``predict_route_local``), so no MPI scatter or gather is included, with
each method's default chunk size unless ``--batch-size`` sets both.

Tolerance per problem (``--tol``, 1e-3): Rosenbrock ``rtol = atol = tol`` (f reaches 0, where a
relative test never passes); spherical_piecewise ``rtol = tol``; Lorenz-96 absolute only,
``atol = tol * max|f|`` over the test points. AFINNI trains with its Linf test and ART with the
pointwise one; a test point counts as above tolerance when ``|f - f_hat| > max(atol, rtol |f|)``
in any component.

Outputs, under ``--out`` (default ``scratch/bench_inference/<cpu|gpu>``, so the two backends never
overwrite each other): one ``<problem>_d<d>.json`` per case, reused on a rerun when it was made
with the same settings (``--force`` reruns it anyway), and ``summary.json`` / ``summary.csv`` with
one row per case and method.
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))  # the checkout being measured

from src import IS_CUPY_BACKEND, PROJECT_DIR, to_host, xp  # noqa: E402
from src.components.modes import SurrogateMode  # noqa: E402
from src.config import ARTSettings, TrainingConfig, default_min_pts_per_cluster  # noqa: E402
from src.core.evaluation import Evaluator  # noqa: E402
from src.plotting import pointwise_relative_error  # noqa: E402
from src.problems.lorenz96_flowmap import lorenz96_flowmap  # noqa: E402
from src.problems.rosenbrock import rosenbrock  # noqa: E402
from src.problems.spherical_piecewise import spherical_piecewise  # noqa: E402
from src.surrogate import AdaptiveSurrogate, compute_error_metrics  # noqa: E402

OUT_DIR = PROJECT_DIR / "scratch" / "bench_inference" / ("gpu" if IS_CUPY_BACKEND else "cpu")
# The settings a saved case must match to be reused.
SETTINGS = ("n_test", "tol", "max_levels", "repeats", "batch_size", "seed", "backend")


@dataclass(frozen=True)
class Problem:
    """A benchmark function, its box domain, the dimensions run by default and how ``--tol`` becomes tolerances."""

    f: Callable
    domain: tuple[float, float]
    dims: tuple[int, ...]
    tolerance: str  # "relative_and_absolute", "relative" or "absolute_scaled" (atol = tol * max|f|)
    note: str


PROBLEMS = {
    "rosenbrock": Problem(partial(rosenbrock, a=100.0), (-2.0, 3.0), (2, 5, 10), "relative_and_absolute",
                          "narrow curved valley, scalar output"),
    "spherical_piecewise": Problem(spherical_piecewise, (-3.0, 3.0), (2, 3, 4, 5), "relative",
                                   "affine inside / outside a sphere with a jump across it, scalar output"),
    "lorenz96": Problem(partial(lorenz96_flowmap, forcing=1.0, tf=0.1), (-5.0, 5.0), (5, 8, 10), "absolute_scaled",
                        "Lorenz-96 flow map x(0) -> x(0.1), F = 1, all d outputs"),
}  # fmt: skip


def sync() -> None:
    """Wait for queued GPU work, so a timing includes it."""
    if IS_CUPY_BACKEND:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def tolerances(prob: Problem, tol: float, truth: np.ndarray) -> tuple[float | None, float | None]:
    """``(atol, rtol)`` of ``prob`` at ``tol``; the scaled absolute test uses max|f| over ``truth``."""
    if prob.tolerance == "relative_and_absolute":
        return tol, tol
    if prob.tolerance == "relative":
        return None, tol
    return tol * float(np.max(np.abs(truth))), None


def configs(prob: Problem, d: int, atol, rtol, args) -> dict[str, TrainingConfig]:
    """Both methods' settings: one root each, the same tolerance and level limit, fallback leaves, no work stealing."""
    common = {"atol": atol, "rtol": rtol, "max_levels": args.max_levels, "seed": args.seed,
              "use_work_stealing": False, "profile_timing": True}  # fmt: skip
    lower, upper = np.full(d, prob.domain[0]), np.full(d, prob.domain[1])
    return {
        "afinni": TrainingConfig(mode=SurrogateMode.AFFINE_COV_HESS, error_metric="Linf", num_initial_clusters=1,
                                 min_pts_per_cluster=default_min_pts_per_cluster(d), fallback_leaves=True, **common),
        "art": TrainingConfig(algorithm="art", domain=(lower, upper), error_metric="pointwise",
                              art=ARTSettings(), **common),
    }  # fmt: skip


def predictor(model: AdaptiveSurrogate, method: str, batch_size: int | None) -> Callable:
    """The single-process inference call of ``method``: kNN + trust region (k = 1, 10 candidates) or routing."""
    if method == "afinni":
        return lambda X: model.predict_local(X, 1, 10, False, batch_size=batch_size)
    if batch_size is None:
        return lambda X: model.predict_route_local(X)
    return lambda X: model.predict_route_local(X, batch_size=batch_size)


def time_inference(predict: Callable, X, repeats: int) -> tuple[list[float], tuple]:
    """Seconds of ``repeats`` calls after one untimed warm-up (first-call kernel compilation, caches), and the last result."""
    out = predict(X)
    sync()
    seconds = []
    for _ in range(repeats):
        sync()
        t = time.perf_counter()
        out = predict(X)
        sync()
        seconds.append(time.perf_counter() - t)
    return seconds, out


def scores(truth: np.ndarray, pred, outside, atol, rtol) -> dict:
    """Errors of one model on the test points (host floats)."""
    p = np.asarray(to_host(pred))
    err = pointwise_relative_error(truth, p)
    above = np.any(np.abs(p - truth) > np.maximum(atol or 0.0, (rtol or 0.0) * np.abs(truth)), axis=1)
    summary = compute_error_metrics(xp.asarray(p), xp.asarray(truth), xp.asarray(to_host(outside)), atol=atol, rtol=rtol)
    return {
        "relative_l2": summary["all"]["l2_rel_error"],
        "relative_linf": summary["all"]["linf_rel_error"],
        "median_pointwise": float(np.median(err)),
        "q99_pointwise": float(np.quantile(err, 0.99)),
        "pct_above_tol": 100.0 * float(np.mean(above)),
        "pct_outside": summary["perct_out"],
    }


def run_case(name: str, d: int, args) -> dict:
    """Train both methods on ``name`` in ``d`` dimensions, score them and time their inference."""
    prob = PROBLEMS[name]
    rng = np.random.default_rng(args.seed)
    X0 = xp.asarray(rng.uniform(*prob.domain, (math.ceil(50 * (d + 1)), d)))
    X_test = xp.asarray(np.random.default_rng(args.seed + 1).uniform(*prob.domain, (args.n_test, d)))
    truth = np.asarray(to_host(prob.f(X_test)))
    atol, rtol = tolerances(prob, args.tol, truth)
    record = {"problem": name, "dim": d, "outputs": int(truth.shape[1]), "n_initial": int(X0.shape[0]),
              "n_test": args.n_test, "tol": args.tol, "atol": atol, "rtol": rtol, "max_levels": args.max_levels,
              "repeats": args.repeats, "batch_size": args.batch_size, "seed": args.seed,
              "backend": "cupy" if IS_CUPY_BACKEND else "numpy", "methods": {}}  # fmt: skip
    for method, config in configs(prob, d, atol, rtol, args).items():
        model = AdaptiveSurrogate()
        labels = {"initial_labels": np.zeros(X0.shape[0], dtype=np.int64)} if method == "afinni" else {}
        t = time.perf_counter()
        model.train(X0, config, Evaluator(f_fn=prob.f), **labels)
        train_seconds = time.perf_counter() - t
        seconds, (pred, outside) = time_inference(predictor(model, method, args.batch_size), X_test, args.repeats)
        stats = model.profiling_stats
        entry = {
            "leaves": int(model.N_src),
            "oracle_evals": int(stats["total_f_evals"]),
            "levels": int(stats["num_levels"]),
            "converged": bool(model.converged),
            "train_seconds": train_seconds,
            **scores(truth, pred, outside, atol, rtol),
            "inference_seconds": seconds,
            "inference_median_s": float(np.median(seconds)),
            "inference_min_s": float(np.min(seconds)),
            "inference_us_per_query": 1e6 * float(np.median(seconds)) / args.n_test,
        }
        if method == "art":
            reasons = model.leaf_status_counts()["terminal_reason"]
            entry["terminal_reasons"] = reasons
            entry["fallback_leaves"] = entry["leaves"] - int(reasons["accepted"])
        else:
            entry["fallback_leaves"] = int(model.fallback_src.sum())
        record["methods"][method] = entry
        print(f"[{name} d={d}] {method}: {entry['leaves']:,} leaves, {entry['oracle_evals']:,} evals, "
              f"train {train_seconds:.1f} s, L2 {entry['relative_l2']:.2e}, "
              f"inference median {entry['inference_median_s'] * 1e3:.1f} ms "
              f"({entry['inference_us_per_query']:.2f} us/query)", flush=True)  # fmt: skip
    return record


SUMMARY_FIELDS = ("problem", "dim", "method", "leaves", "fallback_leaves", "oracle_evals", "levels", "converged",
                  "train_seconds", "relative_l2", "relative_linf", "median_pointwise", "q99_pointwise",
                  "pct_above_tol", "pct_outside", "inference_median_s", "inference_min_s",
                  "inference_us_per_query")  # fmt: skip


def summarize(records: list[dict], out: Path) -> None:
    """Write ``summary.json`` / ``summary.csv`` and print one row per case and method."""
    rows = [{"problem": r["problem"], "dim": r["dim"], "method": m, **{k: v for k, v in e.items() if k in SUMMARY_FIELDS}}
            for r in records for m, e in r["methods"].items()]  # fmt: skip
    (out / "summary.json").write_text(json.dumps(records, indent=2))
    with open(out / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    head = f"{'problem':20s}{'d':>3s} {'method':7s}{'leaves':>9s}{'fallback':>9s}{'evals':>12s}{'train s':>9s}" \
           f"{'rel L2':>10s}{'% > tol':>9s}{'infer ms':>10s}{'us/query':>10s}"  # fmt: skip
    print("\n" + head)
    for r in rows:
        print(f"{r['problem']:20s}{r['dim']:3d} {r['method']:7s}{r['leaves']:9,d}{r['fallback_leaves']:9,d}{r['oracle_evals']:12,d}"
              f"{r['train_seconds']:9.1f}{r['relative_l2']:10.2e}{r['pct_above_tol']:9.2f}"
              f"{r['inference_median_s'] * 1e3:10.1f}{r['inference_us_per_query']:10.2f}")  # fmt: skip
    print(f"\nSaved {out / 'summary.csv'} and summary.json")


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    p.add_argument("--problem", nargs="+", choices=list(PROBLEMS), default=list(PROBLEMS))
    p.add_argument("--dim", nargs="+", type=int, help="dimensions to run (default: each problem's own)")
    p.add_argument("--tol", type=float, default=1e-3, help="training tolerance (see the module docstring)")
    p.add_argument("--max-levels", type=int, default=15, help="level / depth limit of both methods")
    p.add_argument("--n-test", type=int, default=50_000, help="test and inference points")
    p.add_argument("--repeats", type=int, default=5, help="timed inference calls per model")
    p.add_argument("--batch-size", type=int, help="inference chunk size for both (default: each method's own)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT_DIR, help="results folder (default: %(default)s)")
    p.add_argument("--force", action="store_true", help="rerun cases whose result file exists")
    p.add_argument("--list", action="store_true", help="list the problems and exit")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.list:
        for name, prob in PROBLEMS.items():
            print(f"{name:20s} d in {prob.dims}, domain {prob.domain}: {prob.note}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.problem:
        for d in args.dim or PROBLEMS[name].dims:
            path = args.out / f"{name}_d{d}.json"
            saved = json.loads(path.read_text()) if path.exists() and not args.force else None
            if saved is not None and all(saved.get(k) == v for k, v in current_settings(args).items()):
                print(f"[{name} d={d}] reusing {path}")
                records.append(saved)
                continue
            record = run_case(name, d, args)
            path.write_text(json.dumps(record, indent=2))
            records.append(record)
    summarize(records, args.out)


def current_settings(args) -> dict:
    """The values of ``SETTINGS`` this invocation would record."""
    values = {"n_test": args.n_test, "tol": args.tol, "max_levels": args.max_levels, "repeats": args.repeats,
              "batch_size": args.batch_size, "seed": args.seed, "backend": "cupy" if IS_CUPY_BACKEND else "numpy"}  # fmt: skip
    return {k: values[k] for k in SETTINGS}


if __name__ == "__main__":
    main()
