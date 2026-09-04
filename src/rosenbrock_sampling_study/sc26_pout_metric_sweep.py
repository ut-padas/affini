"""Run and aggregate SC26-style P_out races for several error criteria."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .sc26_pout_race import METHOD_LABELS, RACE_METHODS, SC26PoutRaceConfig, run
from .sc26_pout_race_analyze import analyze


METRICS = ("L2", "Linf-local")
COLORS = {
    "uniform": "#4C78A8",
    "halo": "#1F9D8A",
    "cloud": "#8B6BBE",
    "random_chord": "#F58518",
}


def _slug(metric: str) -> str:
    return metric.lower().replace("-", "_")


def _tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _aggregate(root: Path, error_metrics: tuple[str, ...]) -> None:
    frames: list[pd.DataFrame] = []
    finals: list[pd.DataFrame] = []
    for metric in error_metrics:
        metric_root = root / _slug(metric)
        metrics_path = metric_root / "analysis" / "all_metrics.csv"
        if not metrics_path.exists():
            continue
        frame = pd.read_csv(metrics_path)
        frame["error_metric"] = metric
        frames.append(frame)
        final = frame.sort_values("round").groupby("method", as_index=False).tail(1)
        status_path = metric_root / "RACE_COMPLETE.json"
        if status_path.exists():
            statuses = json.loads(status_path.read_text())
            final["run_status"] = final["method"].map(statuses)
        finals.append(final)
    if not frames:
        raise RuntimeError(f"No metric-sweep data found under {root}")

    analysis = root / "analysis"
    figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    all_metrics = pd.concat(frames, ignore_index=True)
    final_metrics = pd.concat(finals, ignore_index=True).sort_values(
        ["error_metric", "p_out_percent"]
    )
    all_metrics.to_csv(analysis / "metric_sweep_all_metrics.csv", index=False)
    final_metrics.to_csv(analysis / "metric_sweep_final_table.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(1, len(frames), figsize=(10 * len(frames), 6), squeeze=False)
    for axis, (metric, frame) in zip(axes.flat, all_metrics.groupby("error_metric", sort=False)):
        for method in RACE_METHODS:
            subset = frame[frame["method"] == method].sort_values("round")
            if not subset.empty:
                axis.plot(
                    subset["n_train_before"],
                    subset["p_out_percent"],
                    marker="o",
                    linewidth=2.2,
                    color=COLORS[method],
                    label=METHOD_LABELS[method],
                )
        axis.axhline(2.0, color="#333333", linestyle="--", linewidth=1.2)
        axis.set_xscale("log")
        axis.set_xlabel("Persistent training points before acquisition")
        axis.set_ylabel("P_out (%)")
        axis.set_title(f"Acceptance metric: {metric}")
        axis.set_ylim(bottom=0)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "pout_by_error_metric.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class SweepConfig:
    output_root: str
    methods: tuple[str, ...]
    error_metrics: tuple[str, ...]
    target_pout: float
    max_rounds: int
    max_training_points: int
    min_free_gpu_gib: float
    points_per_cluster: int
    cluster_reference: int


def run_sweep(config: SweepConfig) -> int:
    unknown_metrics = set(config.error_metrics) - set(METRICS)
    if unknown_metrics:
        raise ValueError(f"Unsupported error metrics: {sorted(unknown_metrics)}")
    root = Path(config.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "metric_sweep_config.json").write_text(
        json.dumps({**asdict(config), "methods": list(config.methods), "error_metrics": list(config.error_metrics)}, indent=2)
        + "\n"
    )
    for metric in config.error_metrics:
        output = root / _slug(metric)
        print(f"\n=== Error-metric arm: {metric} ===", flush=True)
        race = SC26PoutRaceConfig(
            output_dir=str(output),
            methods=config.methods,
            error_metric=metric,
            target_pout=config.target_pout,
            max_rounds=config.max_rounds,
            max_training_points=config.max_training_points,
            min_free_gpu_gib=config.min_free_gpu_gib,
            points_per_cluster=config.points_per_cluster,
            cluster_reference=config.cluster_reference,
        )
        run(race)
        analyze(output)
        _release_gpu_memory()
    _aggregate(root, config.error_metrics)
    print(f"Metric-sweep CSVs and figure written to {root / 'analysis'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/sc26_rosenbrock_pout_race/metric_sweep")
    parser.add_argument("--methods", type=_tuple, default=RACE_METHODS)
    parser.add_argument("--error-metrics", type=_tuple, default=METRICS)
    parser.add_argument("--target-pout", type=float, default=0.02)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-training-points", type=int, default=5_000_000)
    parser.add_argument("--min-free-gpu-gib", type=float, default=8.0)
    parser.add_argument("--points-per-cluster", type=int, default=500)
    parser.add_argument("--cluster-reference", type=int, default=10_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_sweep(
        SweepConfig(
            output_root=args.output_root,
            methods=args.methods,
            error_metrics=args.error_metrics,
            target_pout=args.target_pout,
            max_rounds=args.max_rounds,
            max_training_points=args.max_training_points,
            min_free_gpu_gib=args.min_free_gpu_gib,
            points_per_cluster=args.points_per_cluster,
            cluster_reference=args.cluster_reference,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
