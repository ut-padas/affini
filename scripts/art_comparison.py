"""
Side-by-side comparison of ART and AFINNI's region mode on ART's own benchmark problems.

The target function, its domain and the test set all come from an ART checkout
(``examples/tree_high_dim_benchmark.py`` and ``examples/tree_benchmark_helpers.py``), and
both models are scored by ART's ``evaluate_tree_benchmark`` on the same independent
uniform test points (``10000 * d`` by default, seed ``--test-seed``). Region mode uses
ART's settings: affine leaves, the pointwise relative stopping rule with ART's
``--error-tolerance``, ``--max-depth`` levels and ART's ``--sample-multiplier``, starting
from ``ceil(multiplier * (d + 1))`` uniform points (ART's exact box root).

Arguments not listed below are passed to ART's benchmark parser unchanged, so ART runs
in whatever configuration they select. Its defaults are the ``hit_and_run`` sampler and the
``cpu`` soft optimizer with DFS traversal; ART's fastest CPU-only configuration adds
``--sampler parallel_numba`` (multithreaded compiled chains; its other optimizers need a
GPU). Compare against that one when timing matters, e.g.::

    AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. python scripts/art_comparison.py \\
        --art-path ../Adaptive-Regression-Trees-ART- --benchmark quadratic --dimension 2 \\
        --sampler parallel_numba

ART's fastest GPU configuration is ``--traversal bfs --optimizer cupy_batched_cholesky``
(resident batched soft optimizer; sampling, oracle and node fits stay on the CPU).
``--skip-art`` / ``--skip-region`` run one side only. Region mode alone also runs under
``mpiexec`` (``--skip-art``; ``--afinni-work-stealing`` enables stealing); rank 0 prints.

Leaf degree 1 maps to ``affine_lsq`` and degree 2 to ``quadratic_lsq`` (both with an
intercept). Refuses other degrees, ART leaves without an intercept, error metrics other than
``max_pointwise_relative`` and relative-error floors other than 1e-12, which region mode
does not implement. Prints one JSON line per method (prefixed ``COMPARE``; ART's
first), then the resolved configurations (one ``CONFIG`` line); ART's line records the sampler,
optimizer and traversal it ran with. Oracle counts are reported as
each method counts them: AFINNI's include its centroid evaluations when the split
strategy needs ``f(c)`` (the default covariance-Hessian split does).
"""

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from src import _to_host, xp  # noqa: E402
from src.config import ClusteringConfig, RegionSettings  # noqa: E402
from src.core.evaluation import Evaluator  # noqa: E402
from src.infra.backend import MPI  # noqa: E402
from src.surrogate import AFINNISurrogate  # noqa: E402


class _RegionModel:
    """ART's evaluator only needs ``predict``; route on the active backend, return host values."""

    def __init__(self, model: AFINNISurrogate):
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred, _ = self.model._predict_route_local(xp.asarray(X, dtype=xp.float64), batch_size=100_000)
        return np.asarray(_to_host(pred))[:, 0]


def _summary(evaluation, leaves: int, depth: int, queries: int, seconds: float, extra: dict) -> dict:
    m = evaluation.metrics
    q = evaluation.pointwise_relative_quantiles
    return {
        "test_relative_l2": m["test_relative_l2_error"],
        "median_pointwise_rel": m["test_median_pointwise_relative_error"],
        "q95": float(q[0.95]), "q99": float(q[0.99]),
        "max_pointwise_rel": m["test_max_pointwise_relative_error"],
        "frac_pointwise_rel_gt_tol": extra.pop("frac_gt_tol"),
        "leaves": leaves, "max_depth": depth, "oracle_queries": queries, "build_seconds": seconds, **extra,
    }  # fmt: skip


def main() -> None:
    own = argparse.ArgumentParser(description=__doc__, add_help=False)
    own.add_argument("--art-path", type=Path, required=True)
    own.add_argument("--skip-art", action="store_true")
    own.add_argument("--skip-region", action="store_true")
    own.add_argument("--afinni-work-stealing", action="store_true", help="region mode under mpiexec")
    own.add_argument("--afinni-split", default=None, help="AFINNI split strategy (default: the mode's)")
    own.add_argument("--afinni-roots", default="box", choices=("box", "grid", "voronoi"))
    own.add_argument("--afinni-validation", type=int, default=0)
    own.add_argument("--afinni-walk-steps", type=int, default=20)
    own.add_argument("--afinni-search", default="soft", choices=("gap", "bins", "soft"))
    own.add_argument("--afinni-soft-iters", type=int, default=None, help="B4 iterations (default: the mode's)")
    mine, art_argv = own.parse_known_args()
    sys.path.insert(0, str(mine.art_path.resolve()))

    from art.domain import BoxDomain
    from art.sampling import sample_uniform_box
    from examples.tree_benchmark_helpers import (
        build_tree_benchmark,
        evaluate_tree_benchmark,
    )
    from examples.tree_high_dim_benchmark import (
        BENCHMARK_DEFAULT_INTERVALS,
        make_benchmark_target,
        parse_args,
        validate_args,
    )

    warnings.simplefilter("ignore")
    args = parse_args(art_argv)
    validate_args(args)
    # Region mode always fits affine leaves to a pointwise relative rule; any other ART
    # setting would compare two different problems.
    # Degree 1 is affine_lsq, degree 2 quadratic_lsq (both with an intercept: ART always
    # fits one for degree 1, and for degree 2 unless --no-leaf-bias); region mode's
    # relative-error floor is fixed at 1e-12.
    bias = True if args.leaf_degree == 1 else args.leaf_bias
    if args.leaf_degree not in (1, 2) or not bias or args.error_metric != "max_pointwise_relative" \
            or args.relative_error_floor != 1e-12:  # fmt: skip
        raise SystemExit(
            f"art_comparison needs --leaf-degree 1 or 2 with an intercept, --error-metric "
            f"max_pointwise_relative and --relative-error-floor 1e-12 (got {args.leaf_degree}, "
            f"bias={bias}, {args.error_metric!r}, {args.relative_error_floor}): AFINNI's region "
            "mode fits affine or quadratic leaves against the pointwise relative tolerance with that floor"
        )
    leaf_model = {1: "affine_lsq", 2: "quadratic_lsq"}[args.leaf_degree]
    low, high = BENCHMARK_DEFAULT_INTERVALS[args.benchmark]
    low = low if args.domain_low is None else args.domain_low
    high = high if args.domain_high is None else args.domain_high
    domain = BoxDomain.hypercube(args.dimension, low, high)
    n_test = args.dimension * 10_000 if args.n_test is None else args.n_test
    target, _ = make_benchmark_target(args, domain)
    tol = args.error_tolerance

    def uniform(n, rng):
        return sample_uniform_box(domain.bounds, n, random_state=rng)

    def evaluate(model):
        evaluation = evaluate_tree_benchmark(
            model, target, domain, n_test=n_test, error_metric=args.error_metric,
            relative_error_floor=args.relative_error_floor, random_state=args.test_seed,
            batch_size=args.test_batch_size, point_sampler=uniform,
        )  # fmt: skip
        rng = np.random.default_rng(args.test_seed)
        X = uniform(n_test, rng)
        y = np.asarray(target(X), dtype=float).reshape(-1)
        rel = np.abs(model.predict(X) - y) / np.maximum(np.abs(y), args.relative_error_floor)
        return evaluation, float(np.mean(rel > tol))

    common = {"benchmark": args.benchmark, "dimension": args.dimension, "tolerance": tol, "n_test": n_test}
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1 and not mine.skip_art:
        raise SystemExit("ART runs in one process: pass --skip-art under mpiexec")
    if not mine.skip_art:
        t = time.perf_counter()
        run = build_tree_benchmark(args, domain, target)
        seconds = time.perf_counter() - t
        evaluation, frac = evaluate(run.tree)
        from art.tree import LeafNode

        statuses = {}
        for node in run.tree.iter_nodes():
            if isinstance(node, LeafNode):
                statuses[str(node.status)] = statuses.get(str(node.status), 0) + 1
        print("COMPARE " + json.dumps({"method": "ART", **common, **_summary(
            evaluation, run.tree.num_leaves(), run.tree.max_depth(), int(run.build_result.oracle_queries), seconds,
            {"frac_gt_tol": frac, "terminal_reasons": statuses,
             "art_config": {"sampler": args.sampler, "optimizer": args.optimizer, "traversal": args.traversal}},
        )}), flush=True)  # fmt: skip
    if mine.skip_region:
        return

    def f(x):
        return xp.asarray(np.asarray(target(np.asarray(_to_host(x))), dtype=np.float64).reshape(-1, 1))

    n_root = math.ceil(args.sample_multiplier * (args.dimension + 1))
    X0 = uniform(n_root, np.random.default_rng(args.seed))
    config = ClusteringConfig(
        algorithm="region", domain=(domain.bounds[:, 0], domain.bounds[:, 1]), rtol=tol, error_metric="pointwise",
        max_levels=args.max_depth, seed=args.seed, use_work_stealing=mine.afinni_work_stealing,
        split_strategy=mine.afinni_split,
        num_initial_clusters=max(1, 2 ** args.dimension), local_model=leaf_model,
        region=RegionSettings(roots=mine.afinni_roots, sample_multiplier=args.sample_multiplier,
                              validation_samples=mine.afinni_validation, walk_steps=mine.afinni_walk_steps,
                              split_search=mine.afinni_search,
                              **({} if mine.afinni_soft_iters is None else {"soft_iters": mine.afinni_soft_iters})),
    )  # fmt: skip
    model = AFINNISurrogate()
    comm.Barrier()
    t = time.perf_counter()
    model.train(X0, config, Evaluator(f_fn=f))
    seconds = time.perf_counter() - t
    if comm.Get_rank() != 0:
        return
    # The resolved components (local model, acceptance, split strategy and their settings), not the request.
    # "effective" states what both methods actually ran where a requested flag or a
    # registered component name says otherwise (ART forces the intercept for degree 1;
    # region mode samples by hit-and-run unless region.sampling says convex_combination,
    # whatever the generic AFINNI sampler component is).
    effective = {
        "art": {"leaf_degree": args.leaf_degree, "leaf_include_bias": bias, "error_metric": args.error_metric,
                "relative_error_floor": args.relative_error_floor, "sampler": args.sampler},
        "afinni_region": {"leaf_model": leaf_model, "sampling": config.region.sampling,
                          "acceptance": "pointwise", "relative_error_floor": 1e-12,
                          "split_search": config.region.split_search, "roots": config.region.roots},
    }  # fmt: skip
    print("CONFIG " + json.dumps({"effective": effective,
                                  "art_requested": {k: v for k, v in sorted(vars(args).items()) if not callable(v)},
                                  "afinni_region": {"region": config.region.__dict__, **model.config["components"]}},
                                 default=str), flush=True)  # fmt: skip
    evaluation, frac = evaluate(_RegionModel(model))
    depth = int(max(int(p).bit_length() - 1 for p in model.region_leaves["path_code"]))
    print("COMPARE " + json.dumps({"method": "AFINNI-region", **common, **_summary(
        evaluation, int(model.N_src), depth, int(model.profiling_stats["total_f_evals"]), seconds,
        {"frac_gt_tol": frac, "terminal_reasons": model.leaf_status_counts()["terminal_reason"],
         "split_strategy": model.config["components"]["split_strategy"]["name"],
         "split_search": mine.afinni_search, "region_counters": model.profiling_stats.get("region", {})},
    )}), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
