"""Aggregate an aggressive P_out race and plot its progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .pout_race import METHOD_LABELS, RACE_METHODS


COLORS = {
    "uniform_gap": "#F58518",
    "shell": "#2A9D8F",
    "gaussian": "#8F63C8",
    "legacy_halo": "#C44E52",
}


def _read(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(f"*/{filename}")):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path.relative_to(root))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _plot(
    metrics: pd.DataFrame,
    x: str,
    xlabel: str,
    output: Path,
    target: float,
    log_x: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for method in RACE_METHODS:
        frame = metrics[metrics["method"] == method].sort_values("checkpoint")
        if frame.empty:
            continue
        ax.plot(
            frame[x],
            100 * frame["p_out"],
            marker="o",
            markersize=6,
            linewidth=2.4,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.axhline(
        100 * target,
        color="#333333",
        linestyle="--",
        linewidth=1.3,
        label=f"Target ({100*target:g}%)",
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Uncovered test volume (%)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.35, which="both")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _summary(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    target = float(config["target_pout"])
    initial_n = int(config["initial_n"])
    for method, frame in metrics.groupby("method"):
        frame = frame.sort_values("checkpoint")
        reached = frame[frame["p_out"] <= target]
        final = frame.iloc[-1]
        hit = not reached.empty
        finish = reached.iloc[0] if hit else final
        rows.append(
            {
                "method": method,
                "target_reached": hit,
                "checkpoint_to_target": int(finish["checkpoint"]) if hit else np.nan,
                "compute_seconds_to_target": (
                    float(finish["cumulative_compute_seconds"]) if hit else np.nan
                ),
                "n_train_to_target": int(finish["n_train"]) if hit else np.nan,
                "new_points_to_target": (
                    int(finish["n_train"] - initial_n) if hit else np.nan
                ),
                "fit_points_processed_to_target": (
                    int(finish["cumulative_fit_points"]) if hit else np.nan
                ),
                "final_checkpoint": int(final["checkpoint"]),
                "final_p_out": float(final["p_out"]),
                "final_n_train": int(final["n_train"]),
                "final_compute_seconds": float(final["cumulative_compute_seconds"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    return summary.sort_values(
        ["target_reached", "compute_seconds_to_target", "final_p_out"],
        ascending=[False, True, True],
        na_position="last",
    )


def _markdown(summary: pd.DataFrame, config: dict) -> str:
    target = 100 * float(config["target_pout"])
    lines = [
        "# Aggressive Rosenbrock P_out race",
        "",
        f"Target uncovered volume: {target:g}%",
        "",
        "Each method receives the same initial design and raw candidate count per "
        "cluster. Every proposal classified as uncovered is appended. Batch sizes "
        "therefore differ by method and round.",
        "",
        "The primary ranking is measured compute time to target. Rounds and added "
        "training points are reported separately.",
        "",
        "| Rank | Method | Hit target | Compute time to target (s) | Rounds | "
        "Training points at target | Added points | Final P_out (%) |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary.itertuples(index=False), start=1):
        def value(item: object, spec: str = "") -> str:
            if pd.isna(item):
                return "not reached"
            return format(item, spec)

        lines.append(
            f"| {rank} | {row.method_label} | {'yes' if row.target_reached else 'no'} | "
            f"{value(row.compute_seconds_to_target, '.2f')} | "
            f"{value(row.checkpoint_to_target, '.0f')} | "
            f"{value(row.n_train_to_target, ',.0f')} | "
            f"{value(row.new_points_to_target, ',.0f')} | "
            f"{100*row.final_p_out:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reading the result",
            "",
            "A fast wall-clock result on Rosenbrock does not imply sample efficiency. "
            "The function and gradient are cheap, so model fitting and membership "
            "filtering dominate this benchmark. Use the training-point and fit-point "
            "columns when considering a more expensive scientific model.",
            "",
            "The legacy halo clips proposals to the boundary. Gaussian rejects the same "
            "out-of-domain proposals. Their difference isolates the effect of clipping.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(root: Path) -> int:
    root = root.expanduser().resolve()
    metrics = _read(root, "metrics.csv")
    acquisitions = _read(root, "acquisitions.csv")
    if metrics.empty:
        raise RuntimeError(f"No race metrics found below {root}")
    config = json.loads((root / "race_config.json").read_text())
    analysis = root / "analysis"
    figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(analysis / "all_metrics.csv", index=False)
    if not acquisitions.empty:
        acquisitions.to_csv(analysis / "all_acquisitions.csv", index=False)
    summary = _summary(metrics, config)
    summary.to_csv(analysis / "race_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    target = float(config["target_pout"])
    _plot(
        metrics,
        "checkpoint",
        "Aggressive acquisition round",
        figures / "pout_vs_round.png",
        target,
    )
    _plot(
        metrics,
        "cumulative_compute_seconds",
        "Cumulative measured compute time (s)",
        figures / "pout_vs_compute_time.png",
        target,
    )
    _plot(
        metrics,
        "n_train",
        "Cumulative training points",
        figures / "pout_vs_training_size.png",
        target,
        log_x=True,
    )
    (analysis / "results_summary.md").write_text(_markdown(summary, config))
    print(f"Race analysis written to {analysis}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args(argv)
    return analyze(args.input)


if __name__ == "__main__":
    raise SystemExit(main())
