"""Aggregate completed arms and generate research-ready plots and tables."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.switch_backend("Agg")


METHOD_LABELS = {
    "uniform": "Uniform",
    "uniform_gap": "Uniform gap",
    "shell": "Normalized shell",
    "gaussian": "Gaussian cloud",
}
METHOD_COLORS = {
    "uniform": "#4C78A8",
    "uniform_gap": "#F58518",
    "shell": "#2A9D8F",
    "gaussian": "#9C6ADE",
}


def _read_tables(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(f"dimension_*/seed_*/*/{filename}")):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        frame["source_file"] = str(path.relative_to(root))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _confidence(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    grouped = frame.groupby(["dimension", "method", "iteration"], as_index=False)
    result = grouped.agg(
        mean=(value, "mean"),
        std=(value, "std"),
        count=(value, "count"),
        normalized_added_budget=("normalized_added_budget", "mean"),
    )
    result["sem"] = result["std"] / np.sqrt(result["count"])
    result["low"] = result["mean"] - 1.96 * result["sem"]
    result["high"] = result["mean"] + 1.96 * result["sem"]
    single = result["count"] == 1
    result.loc[single, "low"] = result.loc[single, "mean"]
    result.loc[single, "high"] = result.loc[single, "mean"]
    return result


def _plot_curves(
    metrics: pd.DataFrame,
    value: str,
    ylabel: str,
    output: Path,
    percent: bool = False,
    reference: float | None = None,
) -> None:
    dimensions = sorted(metrics["dimension"].unique())
    columns = 2
    rows = math.ceil(len(dimensions) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.8 * rows), squeeze=False)
    summary = _confidence(metrics, value)

    for axis, dimension in zip(axes.flat, dimensions, strict=False):
        dim_raw = metrics[metrics["dimension"] == dimension]
        dim_summary = summary[summary["dimension"] == dimension]
        for method in sorted(dim_raw["method"].unique()):
            raw = dim_raw[dim_raw["method"] == method]
            aggregate = dim_summary[dim_summary["method"] == method].sort_values(
                "iteration"
            )
            multiplier = 100.0 if percent else 1.0
            for _, seed_frame in raw.groupby("seed"):
                seed_frame = seed_frame.sort_values("normalized_added_budget")
                axis.plot(
                    seed_frame["normalized_added_budget"],
                    multiplier * seed_frame[value],
                    color=METHOD_COLORS.get(method),
                    alpha=0.18,
                    linewidth=0.9,
                )
            axis.plot(
                aggregate["normalized_added_budget"],
                multiplier * aggregate["mean"],
                color=METHOD_COLORS.get(method),
                label=METHOD_LABELS.get(method, method),
                linewidth=2.2,
                marker="o",
                markersize=4,
            )
            if len(raw["seed"].unique()) > 1:
                x_values = aggregate["normalized_added_budget"]
                axis.fill_between(
                    x_values,
                    multiplier * aggregate["low"],
                    multiplier * aggregate["high"],
                    color=METHOD_COLORS.get(method),
                    alpha=0.12,
                )
        if reference is not None:
            axis.axhline(
                multiplier * reference,
                color="#B44",
                linestyle="--",
                linewidth=1.2,
            )
        axis.set_title(f"Dimension {dimension}")
        axis.set_xlabel("Cumulative new evaluations / initial design size")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    for axis in axes.flat[len(dimensions) :]:
        axis.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _run_level_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dimension, seed, method), frame in metrics.groupby(
        ["dimension", "seed", "method"]
    ):
        frame = frame.sort_values("normalized_added_budget")
        budget = frame["normalized_added_budget"].to_numpy()
        coverage = frame["p_out"].to_numpy()
        cauc = float(np.trapezoid(coverage, budget))
        max_budget = float(budget[-1])
        normalized_cauc = cauc / max_budget if max_budget > 0 else float("nan")
        final = frame.iloc[-1]
        row = {
            "dimension": dimension,
            "seed": seed,
            "method": method,
            "checkpoints": len(frame),
            "max_normalized_budget": max_budget,
            "coverage_auc": cauc,
            "normalized_coverage_auc": normalized_cauc,
            "final_iteration": int(final["iteration"]),
            "final_p_out": float(final["p_out"]),
            "final_p_bad_given_in": float(final["p_bad_given_in"]),
            "final_l2_relative_all": float(final["l2_relative_all"]),
            "final_n_clusters": int(final["n_clusters"]),
            "total_train_seconds": float(frame["train_seconds"].sum()),
            "peak_host_memory_gib": float(frame["peak_host_memory_gib"].max()),
            "peak_gpu_pool_memory_gib": float(frame["gpu_pool_memory_gib"].max()),
        }
        for target in (0.10, 0.05, 0.01):
            reached = frame[frame["p_out"] <= target]
            key = f"budget_to_pout_{int(100 * target):02d}pct"
            row[key] = (
                float(reached.iloc[0]["normalized_added_budget"])
                if len(reached)
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_cauc(run_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension, frame in run_metrics.groupby("dimension"):
        methods = sorted(frame["method"].unique())
        pivot = frame.pivot(index="seed", columns="method", values="normalized_coverage_auc")
        for method_a, method_b in itertools.combinations(methods, 2):
            differences = (pivot[method_a] - pivot[method_b]).dropna()
            n = len(differences)
            mean = float(differences.mean()) if n else float("nan")
            std = float(differences.std(ddof=1)) if n > 1 else float("nan")
            half_width = 1.96 * std / math.sqrt(n) if n > 1 else float("nan")
            rows.append(
                {
                    "dimension": dimension,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_paired_seeds": n,
                    "mean_cauc_a_minus_b": mean,
                    "ci95_low": mean - half_width if n > 1 else float("nan"),
                    "ci95_high": mean + half_width if n > 1 else float("nan"),
                    "seeds_favoring_a": int((differences < 0).sum()),
                    "seeds_favoring_b": int((differences > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 30) -> str:
    if frame.empty:
        return "No completed rows."
    subset = frame[columns].head(limit).copy()
    for column in subset.select_dtypes(include=["float"]).columns:
        subset[column] = subset[column].map(lambda value: f"{value:.4g}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in subset.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def analyze(root: Path) -> int:
    root = root.expanduser().resolve()
    analysis_dir = root / "analysis"
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = _read_tables(root, "metrics.csv")
    acquisitions = _read_tables(root, "acquisitions.csv")
    if metrics.empty:
        raise RuntimeError(f"No checkpoint metrics found below {root}")

    metrics.to_csv(analysis_dir / "all_metrics.csv", index=False)
    if not acquisitions.empty:
        acquisitions["selection_efficiency"] = (
            acquisitions["selected"] / acquisitions["proposed"]
        )
        acquisitions.to_csv(analysis_dir / "all_acquisitions.csv", index=False)

    run_metrics = _run_level_metrics(metrics)
    run_metrics.to_csv(analysis_dir / "run_metrics.csv", index=False)
    paired = _paired_cauc(run_metrics)
    paired.to_csv(analysis_dir / "paired_cauc.csv", index=False)

    final_summary = (
        run_metrics.groupby(["dimension", "method"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            mean_final_p_out=("final_p_out", "mean"),
            std_final_p_out=("final_p_out", "std"),
            mean_normalized_cauc=("normalized_coverage_auc", "mean"),
            mean_final_p_bad_given_in=("final_p_bad_given_in", "mean"),
            mean_total_train_seconds=("total_train_seconds", "mean"),
            mean_final_clusters=("final_n_clusters", "mean"),
        )
    )
    final_summary.to_csv(analysis_dir / "final_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper")
    _plot_curves(
        metrics,
        "p_out",
        "Uncovered test volume (%)",
        figures_dir / "coverage_vs_budget.png",
        percent=True,
    )
    _plot_curves(
        metrics,
        "p_bad_given_in",
        "Failure rate inside trust regions (%)",
        figures_dir / "calibration_vs_budget.png",
        percent=True,
        reference=0.05,
    )

    heatmap = final_summary.pivot(
        index="method", columns="dimension", values="mean_final_p_out"
    )
    heatmap = heatmap.reindex([m for m in METHOD_LABELS if m in heatmap.index])
    fig, axis = plt.subplots(figsize=(1.25 * len(heatmap.columns) + 3, 4.2))
    sns.heatmap(
        100 * heatmap,
        annot=True,
        fmt=".2f",
        cmap="mako_r",
        cbar_kws={"label": "Final uncovered volume (%)"},
        ax=axis,
    )
    axis.set_yticklabels([METHOD_LABELS.get(item.get_text(), item.get_text()) for item in axis.get_yticklabels()], rotation=0)
    axis.set_xlabel("Dimension")
    axis.set_ylabel("Sampler")
    fig.tight_layout()
    fig.savefig(figures_dir / "final_coverage_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if not acquisitions.empty:
        fig, axis = plt.subplots(figsize=(9, 5))
        sns.lineplot(
            data=acquisitions,
            x="iteration_from",
            y="selection_efficiency",
            hue="method",
            style="dimension",
            markers=True,
            estimator="mean",
            errorbar=("ci", 95),
            palette=METHOD_COLORS,
            ax=axis,
        )
        axis.set_yscale("log")
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Selected points / raw proposals")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "proposal_efficiency.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    config = json.loads((root / "config.json").read_text())
    report = f"""# Rosenbrock sampling study results

Study directory: `{root}`

- Preset: `{config['preset']}`
- Dimensions: `{config['dimensions']}`
- Seeds: `{config['seeds']}`
- Samplers: `{config['samplers']}`
- Acquisition iterations: `{config['iterations']}`
- Error tolerance: `{config['tolerance']}`

## Final checkpoint summary

Lower final uncovered volume and lower normalized coverage AUC are better. The inside failure rate must also remain within the calibration target.

{_markdown_table(final_summary, ['dimension', 'method', 'seeds', 'mean_final_p_out', 'mean_normalized_cauc', 'mean_final_p_bad_given_in', 'mean_total_train_seconds'])}

## Paired normalized coverage-AUC differences

Negative `mean_cauc_a_minus_b` favors method A. With one seed, confidence limits remain undefined. Use the pilot or full preset for statistical inference.

{_markdown_table(paired, ['dimension', 'method_a', 'method_b', 'n_paired_seeds', 'mean_cauc_a_minus_b', 'ci95_low', 'ci95_high'])}

## Generated figures

- `figures/coverage_vs_budget.png`
- `figures/calibration_vs_budget.png`
- `figures/final_coverage_heatmap.png`
- `figures/proposal_efficiency.png`
"""
    (analysis_dir / "results_summary.md").write_text(report)
    print(f"Analysis written to {analysis_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args(argv)
    return analyze(args.input)


if __name__ == "__main__":
    raise SystemExit(main())
