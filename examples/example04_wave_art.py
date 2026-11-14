"""
ART, adaptive regression trees, on one problem: train an oblique tree over a box domain,
predict by routing, inspect its leaves, audit it on fresh points, and save and reload it.

    uv run python -m examples.example04_wave_art                    # 2-D wave, soft split search
    uv run python -m examples.example04_wave_art --search bins --roots grid
    uv run python -m examples.example04_wave_art --dim 3 --max-depth 16

The function is a cosine along an oblique direction on ``[-5, 5]^d`` (``src.problems.wave``).
ART starts from ``ceil(50 (d + 1))`` uniform points, one node's worth: every node is topped up
with fresh points inside its polytope before it is fitted, and a node whose affine model misses
the tolerance is cut by a plane. ``example05_afinni_vs_art.py`` compares ART with AFINNI
on four problems.

Outputs, under ``scratch/example04_wave_art``: the model file and, in ``plots/``, the tree's
statistics, the profile dashboard and, in 2-D, the partition with its pointwise errors.
"""

import argparse
import math

import numpy as np

from src import PROJECT_DIR, to_host, xp
from src.config import ARTSettings, TrainingConfig
from src.core.evaluation import Evaluator
from src.problems.wave import wave
from src.surrogate import AFINNISurrogate, compute_error_metrics

OUT_DIR = PROJECT_DIR / "scratch" / "example04_wave_art"
LOWER, UPPER = -5.0, 5.0
SEED = 0


def main(args) -> None:
    """Train, predict, inspect, audit and save one ART model."""
    from src import plotting

    d = args.dim
    rng = np.random.default_rng(SEED)
    # 1. Configure: ART needs the box it partitions (what routing covers and audit samples).
    #    atol matters where f crosses 0, where a purely relative test can never pass.
    config = TrainingConfig(
        algorithm="art", domain=([LOWER] * d, [UPPER] * d), atol=1e-2, rtol=1e-2,
        max_levels=args.max_depth, seed=SEED, profile_timing=True,
        art=ARTSettings(split_search=args.search, roots=args.roots, validation_samples=10),
    )  # fmt: skip

    # 2. Train from one node's worth of uniform points.
    X = xp.asarray(rng.uniform(LOWER, UPPER, (math.ceil(50 * (d + 1)), d)))
    model = AFINNISurrogate()
    model.train(X, config, Evaluator(f_fn=wave))
    stats = model.profiling_stats
    print(f"Trained: {model.N_src:,} leaves, {stats['total_f_evals']:,} evaluations, depth "
          f"{stats['num_levels']}, converged={model.converged}, {stats['time_total']:.2f} s")  # fmt: skip

    # 3. Predict by routing: each query falls in exactly one leaf; points outside the box are flagged.
    X_test = xp.asarray(rng.uniform(LOWER, UPPER, (10_000 * d, d)))
    pred, outside = model.predict(X_test)
    errors = compute_error_metrics(pred, wave(X_test), outside, atol=config.atol, rtol=config.rtol)
    print(f"Test: relative L2 {errors['all']['l2_rel_error']:.2e} on {X_test.shape[0]:,} uniform points, "
          f"{int(to_host(outside).sum())} outside the domain")  # fmt: skip
    _, flagged = model.predict(xp.asarray([[UPPER + 1.0] * d]))
    print(f"A point beyond the box is flagged outside: {bool(to_host(flagged)[0])}")

    # 4. Leaves: why each one stopped, and whether its validation points passed.
    counts = model.leaf_status_counts()
    print(f"Leaves by terminal reason {counts['terminal_reason']}, by validation {counts['validation_status']}")
    details = model.route_details(X_test[:3])
    print(f"Leaf rows of the first three test points: {to_host(details['leaf']).tolist()}")

    # 5. Audit: violations on fresh uniform points, with a binomial upper bound on their rate.
    audit = model.audit(Evaluator(f_fn=wave), 3_000, seed=SEED + 1)
    print(f"Audit: {audit['n_violations']} of {audit['n_points']:,} fresh points above tolerance, violation "
          f"probability < {audit['upper_bound']:.3f} at {audit['confidence']:.0%} confidence")  # fmt: skip

    # 6. Save and reload: the routing tree and leaf statuses are part of the model file.
    path = model.save_model(OUT_DIR / "art_model")
    reloaded = AFINNISurrogate.load_model(path)
    same = bool(np.array_equal(to_host(reloaded.predict(X_test)[0]), to_host(pred)))
    print(f"Saved {path.relative_to(PROJECT_DIR)}; reloaded predictions identical: {same}")

    # 7. Plots.
    plots = OUT_DIR / "plots"
    plotting.plot_art_tree_stats(model, plots / "art_tree_stats.png", title=f"ART tree, wave, d={d}")
    plotting.plot_profile_dashboard(stats, plots / "profile.png")
    if d == 2:
        plotting.plot_art_partition(model, plots / "art_partition.png", f_fn=wave, tol=config.rtol, title="ART tree on wave")
    print(f"Plots in {plots.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--search", default="soft", choices=("gap", "bins", "soft"), help="how a node's plane is chosen")
    p.add_argument("--roots", default="box", choices=("box", "grid", "voronoi"), help="the tree's root cells")
    p.add_argument("--max-depth", type=int, default=14, help="depth limit; deeper nodes keep their model as a leaf")
    main(p.parse_args())
