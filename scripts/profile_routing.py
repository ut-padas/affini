"""
Where ART prediction spends its time: the stages of ``RoutingTable.route`` (input checks,
normalization, root lookup, each tree level) and the leaf evaluation, for several query counts,
plus ``predict_route_local`` at several chunk sizes. Run it before optimizing routing, on each
backend, to see which stage an optimization has to remove.

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off .venv/bin/python scripts/profile_routing.py
    SURROGATE_DEVICE=gpu SURROGATE_MPI=off .venv/bin/python scripts/profile_routing.py \
        --problem lorenz96 --dim 5 --queries 50000 1000000 10000000

The model is ART exactly as ``scripts/bench_inference.py`` trains it (one box root, the same
tolerance and level limit), saved under ``--out`` and reused while its settings match. The staged
timing runs a copy of ``RoutingTable.route``'s loop with a timer around every stage and stops
unless its leaves equal ``route``'s, so the stages always describe the code being measured. Each
timing is the median of ``--repeats`` calls after a warm-up call; on a GPU the device is
synchronized around every stage, which adds a sync per stage that ``route`` itself does not
have, so the stage times add up to more than the unstaged ``route`` time (both are reported).

Outputs, under ``--out`` (default ``scratch/profile_routing/<cpu|gpu>``): ``<problem>_d<d>.json``
with every timing, and a printed table per query count.
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))  # the checkout being measured
sys.path.insert(0, str(Path(__file__).resolve().parent))  # bench_inference, for the shared case settings

from bench_inference import PROBLEMS, configs, sync, tolerances  # noqa: E402

from src import IS_CUPY_BACKEND, PROJECT_DIR, to_host, xp  # noqa: E402
from src.art.geometry import plane_scores  # noqa: E402
from src.core.evaluation import Evaluator  # noqa: E402
from src.surrogate import AdaptiveSurrogate  # noqa: E402

OUT_DIR = PROJECT_DIR / "scratch" / "profile_routing" / ("gpu" if IS_CUPY_BACKEND else "cpu")


def median_time(fn, repeats: int) -> tuple[float, object]:
    """Median seconds of ``repeats`` calls of ``fn`` after one warm-up, and the last result."""
    out = fn()
    sync()
    seconds = []
    for _ in range(repeats):
        sync()
        t = time.perf_counter()
        out = fn()
        sync()
        seconds.append(time.perf_counter() - t)
    return float(np.median(seconds)), out


def staged_route(table, X) -> tuple[dict, list[dict], object]:
    """
    ``table.route(X)`` stage by stage: seconds per prologue stage, one entry per level (seconds,
    queries still moving), and the leaves. Mirrors ``RoutingTable.route``; ``check_staged`` guards
    the copy.
    """
    stages, levels = {}, []

    def timed(name, fn):
        sync()
        t = time.perf_counter()
        out = fn()
        sync()
        stages[name] = stages.get(name, 0.0) + time.perf_counter() - t
        return out

    X = timed("to_float64", lambda: xp.asarray(X, dtype=xp.float64))
    timed("finite_check", lambda: bool(xp.all(xp.isfinite(X))))
    U = timed("normalize", lambda: table.domain.normalize(X))
    U = timed("outside_and_clip", lambda: (xp.any((U < 0) | (U > 1), axis=1), xp.clip(U, 0.0, 1.0)))[1]
    node = timed("root_lookup", lambda: table.root_node[table.roots.root_of(U)])
    for _ in range(table.max_depth + 1):
        sync()
        t = time.perf_counter()
        act = xp.nonzero(~table.is_leaf[node])[0]
        moving = int(act.shape[0])
        if moving == 0:
            break
        na = node[act]
        side = (plane_scores(U[act], table.normal[na]) - table.offset[na] >= 0).astype(xp.int64)
        node[act] = table.child[na, side]
        sync()
        levels.append({"seconds": time.perf_counter() - t, "moving": moving})
    leaf = timed("leaf_lookup", lambda: table.node_leaf[node])
    return stages, levels, leaf


def check_staged(table, X) -> None:
    """Stop unless the staged copy routes every query to the same leaf as ``RoutingTable.route``."""
    _, _, leaf = staged_route(table, X)
    if not np.array_equal(to_host(leaf), to_host(table.route(X)[0])):
        raise SystemExit("staged_route no longer matches RoutingTable.route: update the copy before profiling.")


def model_for(name: str, d: int, args) -> AdaptiveSurrogate:
    """ART for case (name, d) as bench_inference trains it, loaded from ``--out`` when saved with the same settings."""
    prob = PROBLEMS[name]
    path, meta = args.out / f"{name}_d{d}_art", args.out / f"{name}_d{d}_art.json"
    settings = {"tol": args.tol, "max_levels": args.max_levels, "seed": args.seed}
    if (path.with_suffix(".npz")).exists() and meta.exists() and json.loads(meta.read_text()) == settings:
        print(f"[{name} d={d}] reusing {path.with_suffix('.npz')}")
        return AdaptiveSurrogate.load_model(path)
    rng = np.random.default_rng(args.seed)
    X0 = xp.asarray(rng.uniform(*prob.domain, (int(np.ceil(50 * (d + 1))), d)))
    # The tolerance of a scaled absolute test comes from max|f| over a test-sized sample, as in bench_inference.
    sample = xp.asarray(np.random.default_rng(args.seed + 1).uniform(*prob.domain, (50_000, d)))
    atol, rtol = tolerances(prob, args.tol, np.asarray(to_host(prob.f(sample))))
    config = configs(prob, d, atol, rtol, args)["art"]
    model = AdaptiveSurrogate()
    print(f"[{name} d={d}] training ART (max_levels={args.max_levels}) ...", flush=True)
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the max_levels warning is expected here
        model.train(X0, config, Evaluator(f_fn=prob.f))
    print(f"[{name} d={d}] {model.N_src:,} leaves in {time.perf_counter() - t:.1f} s", flush=True)
    model.save_model(path)
    meta.write_text(json.dumps(settings))
    return model


def profile(model: AdaptiveSurrogate, name: str, d: int, n: int, args) -> dict:
    """Every timing for ``n`` uniform queries in the domain."""
    prob, table = PROBLEMS[name], model.routing
    X = xp.asarray(np.random.default_rng(args.seed + 2).uniform(*prob.domain, (n, d)))
    check_staged(table, X[: min(n, 10_000)])
    params = model.model_params()
    route_s, (leaf, _) = median_time(lambda: table.route(X), args.repeats)
    eval_s, _ = median_time(lambda: model.local_model.evaluate(params, X, leaf), args.repeats)
    # Stages: one warm-up, then the per-stage median over the repeats.
    staged_route(table, X)
    runs = [staged_route(table, X) for _ in range(args.repeats)]
    stages = {k: float(np.median([r[0][k] for r in runs])) for k in runs[0][0]}
    levels = [{"level": i, "moving": runs[0][1][i]["moving"],
               "seconds": float(np.median([r[1][i]["seconds"] for r in runs]))} for i in range(len(runs[0][1]))]  # fmt: skip
    chunks = {}
    for size in sorted({s for s in args.chunks if s < n} | {n}):
        chunks[str(size)] = median_time(lambda s=size: model.predict_route_local(X, batch_size=s), args.repeats)[0]
    return {"queries": n, "route_s": route_s, "evaluate_s": eval_s, "stages_s": stages,
            "levels_s": sum(lv["seconds"] for lv in levels), "levels": levels, "predict_by_chunk_s": chunks}  # fmt: skip


def report(name: str, d: int, model, r: dict) -> None:
    """One query count's table."""
    n = r["queries"]
    print(f"\n[{name} d={d}] {n:,} queries, {model.N_src:,} leaves, depth {model.routing.max_depth}, "
          f"{'cupy' if IS_CUPY_BACKEND else 'numpy'}")  # fmt: skip
    print(f"  route (unstaged)   {r['route_s'] * 1e3:10.2f} ms  {r['route_s'] / n * 1e9:8.1f} ns/query")
    print(f"  evaluate           {r['evaluate_s'] * 1e3:10.2f} ms  {r['evaluate_s'] / n * 1e9:8.1f} ns/query")
    print("  stages (synchronized):")
    for k, v in r["stages_s"].items():
        print(f"    {k:18s} {v * 1e3:10.2f} ms")
    print(f"    {'levels (total)':18s} {r['levels_s'] * 1e3:10.2f} ms over {len(r['levels'])} levels")
    for lv in r["levels"]:
        print(f"      level {lv['level']:2d}  {lv['seconds'] * 1e3:8.2f} ms  {lv['moving']:>12,d} moving")
    print("  predict_route_local by chunk size:")
    for size, s in r["predict_by_chunk_s"].items():
        print(f"    chunk {int(size):>12,d}  {s * 1e3:10.2f} ms")


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    p.add_argument("--problem", choices=list(PROBLEMS), default="rosenbrock")
    p.add_argument("--dim", type=int, default=5)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--max-levels", type=int, default=15)
    p.add_argument("--queries", nargs="+", type=int, default=[50_000, 1_000_000], help="query counts to profile")
    p.add_argument("--chunks", nargs="+", type=int, default=[5_000, 50_000, 500_000],
                   help="predict_route_local chunk sizes (plus one chunk of all queries)")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    args.out.mkdir(parents=True, exist_ok=True)

    model = model_for(args.problem, args.dim, args)
    results = []
    for n in args.queries:
        r = profile(model, args.problem, args.dim, n, args)
        report(args.problem, args.dim, model, r)
        results.append(r)
    record = {"problem": args.problem, "dim": args.dim, "leaves": int(model.N_src),
              "max_depth": int(model.routing.max_depth), "backend": "cupy" if IS_CUPY_BACKEND else "numpy",
              "repeats": args.repeats, "results": results}  # fmt: skip
    path = args.out / f"{args.problem}_d{args.dim}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
