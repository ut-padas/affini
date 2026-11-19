"""The appendix section on how fast AFINNI's queue grows and how the host-memory guard works.

Usage, from the repository root:

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/reports/adaptive-surrogates/figures/appendix/memory.py

Trains AFINNI twice on the running example: with the companion's settings, and with a tolerance
a thousand times tighter, whose queue grows for many levels. Each run is recorded by wrapping
the functions the growth law is about (they change no result): ``augmentation_targets`` as the
plan calls it, for the points of every rejected cluster; ``check_host_memory`` and
``ClusterQueueBuilder._reserve``, for how often the guard runs and whether the next level's
buffer ever had to grow. The script stops unless the next level's points are exactly the
augmentation targets of this level's rejected clusters, and unless both bounds of the growth
law hold at every level. Writes apx-memory-growth.svg and apx-memory.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_figures as mf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import src.afinni.plan as afinni_plan  # noqa: E402
import src.core.clusters as clusters  # noqa: E402
from src.config import TrainingConfig  # noqa: E402
from src.core.evaluation import Evaluator  # noqa: E402
from src.surrogate import AdaptiveSurrogate  # noqa: E402

TIGHT = 1e-3  # the larger run's tolerance, in units of the example's
TIGHT_LEVELS = 40  # its depth limit, deep enough that it converges


def record(atol: float, max_levels: int) -> dict:
    """Train AFINNI on the running example; per level, the rejected clusters' points and their augmentation targets."""
    rejected: dict[int, list[np.ndarray]] = {}
    state = {"level": None, "checks": 0, "grows": 0, "appends": 0}
    targets_fn, pb = afinni_plan.augmentation_targets, afinni_plan.AFINNIPlan.process_batch
    check_fn, reserve_fn, append_fn = clusters.check_host_memory, clusters.ClusterQueueBuilder._reserve, clusters.ClusterQueueBuilder.append

    def process_batch(self, batch, **kw):
        state["level"] = batch.level
        return pb(self, batch, **kw)

    def targets(counts, min_pts):
        rejected.setdefault(state["level"], []).append(mf.to_host(counts).copy())
        return targets_fn(counts, min_pts)

    def check(n_bytes, what):
        state["checks"] += 1
        return check_fn(n_bytes, what)

    def reserve(self, n_pts, n_clusters):
        state["grows"] += int(n_pts > self._buf["X"].shape[0])
        return reserve_fn(self, n_pts, n_clusters)

    def append(self, *a, **kw):
        state["appends"] += 1
        return append_fn(self, *a, **kw)

    afinni_plan.augmentation_targets, afinni_plan.AFINNIPlan.process_batch = targets, process_batch
    clusters.check_host_memory, clusters.ClusterQueueBuilder._reserve, clusters.ClusterQueueBuilder.append = check, reserve, append
    try:
        X0 = np.random.default_rng(mf.SEED).uniform(mf.LOWER, mf.UPPER, (mf.N0, 2))
        settings = {**mf.AFINNI_SETTINGS, "atol": atol, "max_levels": max_levels}
        s = AdaptiveSurrogate()
        s.train(X0, TrainingConfig(**settings, use_work_stealing=False, seed=mf.SEED), Evaluator(f_fn=mf.f_xp))
    finally:
        afinni_plan.augmentation_targets, afinni_plan.AFINNIPlan.process_batch = targets_fn, pb
        clusters.check_host_memory, clusters.ClusterQueueBuilder._reserve, clusters.ClusterQueueBuilder.append = check_fn, reserve_fn, append_fn
    if not s.profiling_stats["converged"] or s.profiling_stats["num_unresolved_clusters"]:
        raise SystemExit("A recorded run stopped at max_levels: the growth law below assumes no child is dropped.")

    n_min = settings["min_pts_per_cluster"]
    stats = s.profiling_stats["level_stats"]
    P = [int(lv["points_start"]) for lv in stats]
    levels = []
    for lv in stats[:-1]:
        counts = np.concatenate(rejected[lv["level"]])
        rule = np.where(counts < 2 * n_min, np.maximum(2 * counts, 2 * n_min), counts)  # the growth law's t(n)
        if not np.array_equal(rule, mf.to_host(targets_fn(counts, n_min))):
            raise SystemExit("The restated augmentation target differs from the library's.")
        nxt = P[lv["level"] + 1]
        if int(rule.sum()) != nxt:
            raise SystemExit("The next level's points are not the augmentation targets of this level's rejected clusters.")
        R, P_rej = len(counts), int(counts.sum())
        lower, upper = 2 * n_min * R, 2 * P_rej
        if not lower <= nxt or (counts.min() >= n_min and not nxt <= upper) or not nxt <= 2 * P[lv["level"]]:
            raise SystemExit(f"The growth law fails at level {lv['level']}.")
        levels.append({"level": lv["level"], "points": P[lv["level"]], "clusters": int(lv["clusters_start"]),
                       "rejected": R, "rejected_points": P_rej, "smallest_rejected": int(counts.min()),
                       "next": nxt, "lower": lower, "upper": upper})  # fmt: skip
    two = [P[k] + P[k + 1] for k in range(len(P) - 1)]
    return {"atol": atol, "n_min": n_min, "levels": levels, "points": P, "initial_smallest": int(stats[0]["size_min"]),
            "leaves": int(s.profiling_stats["num_final_clusters"]), "checks": state["checks"],
            "appends": state["appends"], "grows": state["grows"],
            "peak_two_levels": max(two), "peak_level": max(P),
            "tight_levels": sum(lv["next"] == lv["upper"] for lv in levels),
            # The first level l + 1 from which every next level sits on the upper bound.
            "tight_from": next(lv["level"] + 1 for k, lv in enumerate(levels)
                               if all(v["next"] == v["upper"] for v in levels[k:]))}  # fmt: skip


def draw(runs: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.2), gridspec_kw={"wspace": 0.28})
    for ax, run, title in zip(axes, runs, ("the running example", "a thousand times tighter"), strict=True):
        lv = run["levels"]
        x = np.array([v["level"] + 1 for v in lv])
        lower, upper, nxt = (np.array([v[k] for v in lv]) for k in ("lower", "upper", "next"))
        ax.fill_between(x, lower, upper, color="#ececec", lw=0, zorder=1)
        ax.plot(x, upper, color=mf.GREY, lw=0.9, zorder=2, label="$2 P_\\ell^{rej}$, upper bound")
        ax.plot(x, lower, color=mf.GREY, lw=0.9, ls="--", zorder=2, label="$2 N_{min} R_\\ell$, lower bound")
        ax.plot(x, nxt, color=mf.DARK, lw=0.9, marker="o", ms=2.6, zorder=3, label="$P_{\\ell+1}$, the next level")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel("level $\\ell + 1$", fontsize=6, labelpad=1)
        ax.set_title(f"{title} ($\\alpha_{{tol}}$ = {run['atol']:g})", fontsize=6.5, color=mf.DARK, pad=3, loc="left")
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("points", fontsize=6, labelpad=1)
    axes[0].legend(fontsize=5.5, frameon=False, loc="upper right", handlelength=2.2)
    mf.save(fig, "apx-memory-growth")


def main() -> None:
    example = record(mf.ALPHA, mf.AFINNI_SETTINGS["max_levels"])
    tight = record(mf.ALPHA * TIGHT, TIGHT_LEVELS)
    draw([example, tight])

    # The queue's bytes per point (d = 2, f scalar) and the leaf buffer's initial reservation.
    config = TrainingConfig(atol=1.0)
    builder = clusters.ClusterQueueBuilder(2, 1, has_g=False, capacity_pts=1, capacity_clusters=1)
    model = config.components().local_model

    def leaf_row_bytes(d: int) -> int:
        return clusters.ColumnBuffer(afinni_plan.AFINNIPlan(config, d, 1, 0).leaf_columns(model), capacity=1).row_bytes

    per_row = clusters.ColumnBuffer.INITIAL_RESERVE_BYTES // config.max_clusters_est  # where the hint meets the cap
    d_cap = next(d for d in range(1, 200) if leaf_row_bytes(d) > per_row)
    numbers = {
        "example": example, "tight": tight, "tight_factor": TIGHT,
        "queue_row_bytes": builder._row_bytes(),
        "reserve_share_percent": round(100 * clusters.HOST_MEMORY_RESERVE_SHARE),
        "reserve_min_gib": clusters.HOST_MEMORY_RESERVE_MIN / 2**30,
        "initial_reserve_gib": clusters.ColumnBuffer.INITIAL_RESERVE_BYTES / 2**30,
        "max_clusters_est": f"{config.max_clusters_est:,}", "cap_row_bytes": f"{per_row:,}",
        "leaf_row_bytes_d2": leaf_row_bytes(2), "leaf_cap_d": d_cap,
        "peak_two_levels_kib": f"{round(tight['peak_two_levels'] * builder._row_bytes() / 2**10):,}",
        # The large numbers as the text prints them, with thousands separators.
        "text": {"peak_level": f"{tight['peak_level']:,}", "peak_two_levels": f"{tight['peak_two_levels']:,}"},
    }  # fmt: skip
    (mf.OUT / "apx-memory.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps({k: v for k, v in numbers.items() if k not in ("example", "tight")}),
          {k: v for k, v in tight.items() if k != "levels"})


if __name__ == "__main__":
    main()
