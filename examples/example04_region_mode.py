"""
AFINNI against AFINNI region mode (ART: an oblique regression tree) on several benchmark
problems, each scored on the same independent uniform test points:

=====================  ==========================================================  ==========
problem                what it tests                                               domain
=====================  ==========================================================  ==========
spherical_piecewise    a jump across a circle (ART's TACC preset)                  [-3, 3]^d
anisotropic_gaussian   smooth, rotated and strongly anisotropic                    [-3, 3]^d
wave                   a cosine along an oblique direction                         [-5, 5]^d
rosenbrock             a narrow curved valley whose minimum is 0 (needs atol)      [-2, 3]^d
=====================  ==========================================================  ==========

Each problem's depth limits (``Problem.afinni_max_levels``, ``Problem.region_max_depth``)
are chosen so the two modes reach about the same relative L2 error in 2-D on one CPU each,
so leaves, oracle evaluations and training time compare like for like (``PROBLEMS`` records
the measured match). Each mode trains in its own stage, then ``compare`` loads the saved
models, evaluates them and draws the plots:

    uv run python -m examples.example04_region_mode --problem all            # every problem
    uv run python -m examples.example04_region_mode --problem wave           # one problem
    uv run python -m examples.example04_region_mode afinni --problem wave    # one stage of one
    uv run python -m examples.example04_region_mode compare --problem wave
    uv run python -m examples.example04_region_mode --list                   # the problems
    # live view of a training stage, from another terminal:
    uv run python -m src.monitor scratch/example04_region_mode/wave/stream/region   # or .../afinni

Everything runs in one process on one CPU (``AFINNI_MPI`` defaults to ``off`` here), so
training times compare directly. ``--afinni-max-levels`` and
``--region-max-depth`` override a problem's limits, ``--dim`` and ``--tol`` its setup
(other dimensions and tolerances need their own limits for a matched comparison).

Region mode uses ART's settings (``ceil(50 (d + 1))`` points per node, affine leaves, the
pointwise stopping rule, the consistent soft-oblique splitter); a node at its depth limit
keeps its fitted model. AFINNI uses its default mode (``AFFINE_COV_HESS``, Linf test). A
test point counts as above tolerance when ``|f - f_hat| > max(atol, rtol |f|)``, the rule
both modes train with.

Outputs, under ``scratch/example04_region_mode/<problem>``: both models with their
training times, ``metrics.json``, and in ``plots/`` the error distributions, a metric
comparison, the region tree's statistics and each run's profile dashboard; in 2-D also the
region partition, side-by-side error maps and AFINNI's trust ellipsoids. With
``--problem all``, ``summary.json`` and ``summary.png`` in the parent folder compare every
problem.
"""

import os

# Before importing src: a single-process comparison (training times on one CPU each).
os.environ.setdefault("AFINNI_MPI", "off")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from functools import partial  # noqa: E402
from typing import cast  # noqa: E402

import numpy as np  # noqa: E402

from src import PROJECT_DIR, _to_host, xp  # noqa: E402
from src.config import ClusteringConfig, RegionSettings  # noqa: E402
from src.evaluation import Evaluator  # noqa: E402
from src.heuristic_parameters import (  # noqa: E402
    get_MIN_PTS_PER_CLUSTER,
    get_N_TRAIN,
    get_NUM_INITIAL_CLUSTERS,
)
from src.modes import SurrogateMode  # noqa: E402
from src.problems.anisotropic_gaussian import aniso_gaussian  # noqa: E402
from src.problems.rosenbrock import rosenbrock  # noqa: E402
from src.problems.spherical_piecewise import spherical_piecewise  # noqa: E402
from src.problems.wave import wave  # noqa: E402
from src.surrogate import AFINNISurrogate, compute_error_metrics  # noqa: E402

OUT_DIR = PROJECT_DIR / "scratch" / "example04_region_mode"
SEED = 0
METHODS = ("afinni", "region")


@dataclass(frozen=True)
class Problem:
    """A benchmark and the depth limits at which both modes reach about the same relative L2 error."""

    f: Callable
    domain: tuple[float, float]
    afinni_max_levels: int
    region_max_depth: int
    atol: float | None = None  # needed where f reaches 0, where a relative test never passes
    rtol: float = 1e-2  # the tolerance both modes are scored with (and train with, by default)
    region_rtol: float | None = None  # region mode's training rtol, when it must differ to match L2
    note: str = ""


# Matched in 2-D on 20,000 test points, one CPU each; note = relative L2 and % of points
# above tolerance, AFINNI / region mode. Where a mode converges before its limit, the
# limit is not what stops it ("50" for AFINNI means uncapped in practice).
PROBLEMS = {
    "spherical_piecewise": Problem(spherical_piecewise, (-3.0, 3.0), 7, 12,
                                   note="L2 0.095 / 0.094, 1.7% / 1.7% (neither converges: the jump)"),
    "anisotropic_gaussian": Problem(aniso_gaussian, (-3.0, 3.0), 50, 12,
                                    note="L2 0.0018 / 0.0018, 0.11% / 0.03% (both converge)"),
    "wave": Problem(wave, (-5.0, 5.0), 50, 12, region_rtol=0.02,
                    note="L2 0.0043 / 0.0045, 3.0% / 3.3% (both converge; region trained at rtol 0.02)"),
    "rosenbrock": Problem(partial(rosenbrock, a=100.0), (-2.0, 3.0), 50, 16, atol=1e-2,
                          note="L2 0.0024 / 0.0023, 3.0% / 2.6% (AFINNI converges)"),
}


def configs(name: str, args) -> dict[str, tuple[ClusteringConfig, int]]:
    """Each mode's configuration for problem ``name`` and its number of initial training points."""
    prob, d = PROBLEMS[name], args.dim
    stream = OUT_DIR / name / "stream"
    common = {"seed": SEED, "use_work_stealing": False, "profile_timing": True, "atol": prob.atol}
    rtol = {"afinni": prob.rtol, "region": prob.region_rtol or prob.rtol} if args.tol is None else \
        {"afinni": args.tol, "region": args.tol}  # fmt: skip
    lower, upper = np.full(d, prob.domain[0]), np.full(d, prob.domain[1])
    return {
        "afinni": (ClusteringConfig(
            mode=SurrogateMode.AFFINE_COV_HESS, error_metric="Linf", rtol=rtol["afinni"],
            max_levels=args.afinni_max_levels or prob.afinni_max_levels,
            num_initial_clusters=get_NUM_INITIAL_CLUSTERS(d), min_pts_per_cluster=get_MIN_PTS_PER_CLUSTER(d),
            profile_dir=stream / "afinni", **common,
        ), get_N_TRAIN(d)),  # fmt: skip
        # ART's exact box root starts from one node's worth of uniform points.
        "region": (ClusteringConfig(
            algorithm="region", domain=(lower, upper), error_metric="pointwise", rtol=rtol["region"],
            max_levels=args.region_max_depth or prob.region_max_depth, region=RegionSettings(split_search=args.search),
            profile_dir=stream / "region", **common,
        ), math.ceil(50 * (d + 1))),  # fmt: skip
    }


def train(name: str, method: str, args) -> None:
    """Train one mode and save it with its wall time."""
    config, n = configs(name, args)[method]
    prob = PROBLEMS[name]
    X = xp.asarray(np.random.default_rng(SEED).uniform(*prob.domain, (n, args.dim)))
    model = AFINNISurrogate()
    t0 = time.perf_counter()
    model.train(X, config, Evaluator(f_fn=prob.f))
    seconds = time.perf_counter() - t0
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / f"{method}_model")
    run = {"train_seconds": seconds, "dim": args.dim, "max_levels": config.max_levels,
           "atol": config.atol, "rtol": config.rtol}  # fmt: skip
    (out / f"{method}_run.json").write_text(json.dumps(run))
    print(f"[{name}] {method}: {model.N_src:,} leaves, {model.profiling_stats['total_f_evals']:,} evaluations, "
          f"{seconds:.2f} s, max_levels={config.max_levels}")  # fmt: skip


def metrics(truth, pred, outside, atol, rtol, model: AFINNISurrogate, run: dict) -> dict:
    """The comparison metrics of one model on the shared test set (host floats)."""
    from src.plotting import pointwise_relative_error

    t, p = np.asarray(_to_host(truth)), np.asarray(_to_host(pred))
    err = pointwise_relative_error(t, p)
    above = np.any(np.abs(p - t) > np.maximum(atol or 0.0, rtol * np.abs(t)), axis=1)
    summary = cast(dict, compute_error_metrics(pred, truth, outside, atol=atol, rtol=rtol))
    return {
        "relative_l2": summary["all"]["l2_rel_error"],
        "median_pointwise": float(np.median(err)),
        "q99_pointwise": float(np.quantile(err, 0.99)),
        "pct_above_tol": 100.0 * float(np.mean(above)),
        "pout": summary["perct_out"],
        "leaves": int(model.N_src),
        "evals": int(model.profiling_stats["total_f_evals"]),
        "train_seconds": run["train_seconds"],
        "max_levels": run["max_levels"],
        "train_rtol": run["rtol"],
    }


def compare(name: str, args) -> dict:
    """Load both saved models of ``name``, score them on one test set, print the table and draw the plots."""
    from src import plotting as P  # noqa: N812  (as in example 03)

    prob, d, out = PROBLEMS[name], args.dim, OUT_DIR / name
    models, runs = {}, {}
    for method in METHODS:
        run = json.loads((out / f"{method}_run.json").read_text())
        if run["dim"] != d:
            raise SystemExit(f"[{name}] {method} was trained with dim={run['dim']}; rerun its stage.")
        models[method], runs[method] = AFINNISurrogate.load_model(out / f"{method}_model"), run
    afinni, region = models["afinni"], models["region"]
    # Both modes are scored with the problem's tolerance, whatever they trained with.
    atol, rtol = prob.atol, prob.rtol if args.tol is None else args.tol
    X_test = xp.asarray(np.random.default_rng(SEED + 1).uniform(*prob.domain, (10_000 * d, d)))
    truth = prob.f(X_test)
    # Local (non-collective) prediction: both models are complete on this process.
    pred_a, out_a = afinni._predict_local(X_test, 1, 10, False)
    pred_r, out_r = region._predict_route_local(X_test)
    names = {"afinni": "AFINNI", "region": f"AFINNI region mode (ART, {args.search})"}
    results = {
        names["afinni"]: metrics(truth, pred_a, out_a, atol, rtol, afinni, runs["afinni"]),
        names["region"]: metrics(truth, pred_r, out_r, atol, rtol, region, runs["region"]),
    }
    audit = region.audit(Evaluator(f_fn=prob.f), 3000, seed=SEED + 2)

    tol = f"rtol {rtol:g}" + (f", atol {atol:g}" if atol is not None else "")
    print(f"\n[{name}] {X_test.shape[0]:,} uniform test points on [{prob.domain[0]}, {prob.domain[1]}]^{d}, {tol}")
    w = max(len(n) for n in results) + 2
    print(f"{'metric':18s}" + "".join(f"{n:>{w}s}" for n in results))
    for key in next(iter(results.values())):
        row = [results[n][key] for n in results]
        ints = key in ("leaves", "evals", "max_levels")
        print(f"{key:18s}" + "".join(f"{v:{w},.0f}" if ints else f"{v:{w}.4g}" for v in row))
    print(f"[{name}] AFINNI converged={afinni.converged} ({afinni.num_unresolved_clusters:,} clusters unresolved); "
          f"region leaves by status {region.leaf_status_counts()['terminal_reason']}; audit: "
          f"{audit['n_violations']} of {audit['n_points']} fresh points above tolerance, violation probability "
          f"< {audit['upper_bound']:.3f} at {audit['confidence']:.0%} confidence.")  # fmt: skip
    (out / "metrics.json").write_text(json.dumps({"problem": name, "dim": d, "methods": results, "audit": audit}, indent=2))

    plots = out / "plots"
    errors = {
        names["afinni"]: P.pointwise_relative_error(truth, pred_a),
        names["region"]: P.pointwise_relative_error(truth, pred_r),
    }
    title = f"{name}, d={d}, {tol}"
    P.plot_error_cdf(errors, plots / "error_cdf.png", tol=rtol, title=title)
    P.plot_method_comparison(results, plots / "method_comparison.png", title=title)
    P.plot_region_tree_stats(region, plots / "region_tree_stats.png", title=f"Region tree, {name}")
    for method, model in models.items():
        P.plot_profile_dashboard(model.profiling_stats, plots / f"profile_{method}.png")
    if d == 2:
        lower, upper = np.full(d, prob.domain[0]), np.full(d, prob.domain[1])
        P.plot_region_partition(region, plots / "region_partition.png", f_fn=prob.f, tol=rtol,
                                title=f"Region tree on {name}")  # fmt: skip
        P.plot_error_maps(
            {
                f"{names['afinni']} (kNN + trust region)": lambda X: afinni._predict_local(X, 1, 10, False)[0],
                f"{names['region']} (routed)": lambda X: region._predict_route_local(X)[0],
            },
            prob.f, lower, upper, plots / "error_maps.png", tol=rtol, title=f"Pointwise relative error, {name}",
        )  # fmt: skip
        P.plot_adaptive_clustering_results(
            centroids=afinni.X_src, Q=afinni.Qs_src, domain=prob.domain, f_fn=prob.f,
            img_fpath=plots / "afinni_clusters.png", title=f"AFINNI leaves, {name}",
        )  # fmt: skip
    print(f"[{name}] saved metrics, models and plots under {out.relative_to(PROJECT_DIR)}")
    return results


def summarize(all_results: dict[str, dict]) -> None:
    """``summary.json`` and one grouped bar panel per metric across the problems."""
    import matplotlib.pyplot as plt

    from src.plotting.style import afinni_style, finish

    (OUT_DIR / "summary.json").write_text(json.dumps(all_results, indent=2))
    panels = (
        ("relative_l2", "relative L2 error", True), ("pct_above_tol", "% of points above tol", False),
        ("leaves", "leaves", True), ("evals", "oracle evaluations", True), ("train_seconds", "training time [s]", True),
    )  # fmt: skip
    problems = list(all_results)
    methods = list(next(iter(all_results.values())))
    width = 0.8 / len(methods)
    with afinni_style():
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.8))
        for ax, (key, label, log) in zip(axes, panels, strict=True):
            for j, m in enumerate(methods):
                vals = [all_results[p][m][key] for p in problems]
                ax.bar(np.arange(len(problems)) + (j - (len(methods) - 1) / 2) * width, vals, width, label=m)
            ax.set_xticks(range(len(problems)), [p.replace("_", "\n") for p in problems], fontsize=8)
            ax.set_title(label, fontsize=10)
            if log:
                ax.set_yscale("log")
        axes[0].legend(fontsize=8)
        fig.suptitle("AFINNI vs AFINNI region mode (ART): matched relative L2 error, one CPU each")
    finish(fig, OUT_DIR / "summary.png")
    print(f"Saved {(OUT_DIR / 'summary.json').relative_to(PROJECT_DIR)} and summary.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    p.add_argument("stage", nargs="?", default="all", choices=("all", *METHODS, "compare"))
    p.add_argument("--problem", default="all", choices=("all", *PROBLEMS), help="one problem, or all of them")
    p.add_argument("--list", action="store_true", help="list the problems and exit")
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--tol", type=float, default=None, help="rtol for both modes (default: the problem's)")
    p.add_argument("--search", default="soft", choices=("gap", "bins", "soft"), help="region split search")
    p.add_argument("--region-max-depth", type=int, default=None, help="override the problem's region depth limit")
    p.add_argument("--afinni-max-levels", type=int, default=None, help="override the problem's AFINNI depth limit")
    args = p.parse_args()
    if args.list:
        for name, prob in PROBLEMS.items():
            print(f"{name:22s} domain [{prob.domain[0]}, {prob.domain[1]}]^d   AFINNI levels {prob.afinni_max_levels:2d}   "
                  f"region depth {prob.region_max_depth:2d}   {prob.note}")  # fmt: skip
        raise SystemExit(0)
    names = list(PROBLEMS) if args.problem == "all" else [args.problem]
    if "spherical_piecewise" in names and args.dim not in (2, 3, 4, 5):
        raise SystemExit("spherical_piecewise supports --dim 2..5.")
    all_results = {}
    for name in names:
        for method in METHODS:
            if args.stage in ("all", method):
                train(name, method, args)
        if args.stage in ("all", "compare"):
            all_results[name] = compare(name, args)
    if len(all_results) > 1:
        summarize(all_results)
