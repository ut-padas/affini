"""Create paper-style tables and plots for the SC26-matched P_out race."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .sc26_pout_race import METHOD_LABELS, RACE_METHODS


COLORS = {"uniform": "#4C78A8", "halo": "#1F9D8A", "cloud": "#8B6BBE", "random_chord": "#F58518"}


def _read(root: Path, name: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(root.glob(f"*/{name}"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _format(value: float, style: str) -> str:
    return "--" if pd.isna(value) else format(value, style)


def _markdown_table(metrics: pd.DataFrame) -> str:
    lines = [
        "# SC26-matched 5D Rosenbrock P_out race",
        "",
        "The model uses the SC26 configuration: `Linf-local`, `AFFINE_COV_HESS`, `rtol = 0.05`, domain `[-2, 3]^5`, 185,646 initial uniform points, 92,823 fixed uniform test points, 92 initial clusters, and 371 minimum points per cluster. Work stealing is disabled because this run uses one idev GPU rank.",
        "",
        "`N` is the fit's total oracle function-evaluation count, matching the paper's notation. `N_train_before` is the persistent input set used for the row's fit and metrics. `N_train_after` includes retained proposals that seed the next round. `e_in`, `e_out`, and `e` are relative L2 errors, even though the surrogate leaf criterion is local Linf.",
        "",
        "| Method | iter | d | rtol | N | N_train before | N_train after | N_c | p_out (%) | e_in | e_out | e | added | stop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics.sort_values(["method", "round"]).itertuples(index=False):
        lines.append(
            f"| {row.method_label} | {row.round} | {row.dimension} | {row.rtol:.2g} | "
            f"{_format(row.N, ',.0f')} | {row.n_train_before:,} | {row.n_train_after:,} | {row.n_clusters:,} | "
            f"{row.p_out_percent:.3g} | {_format(row.e_in, '.3E')} | "
            f"{_format(row.e_out, '.3E')} | {_format(row.e, '.3E')} | {row.n_added:,} | {row.termination or ''} |"
        )
    return "\n".join(lines) + "\n"


def _final_table(metrics: pd.DataFrame) -> pd.DataFrame:
    final = metrics.sort_values("round").groupby("method", as_index=False).tail(1).copy()
    final["target_reached"] = final["p_out_percent"] < 2.0
    return final[["method", "method_label", "round", "N", "n_train_before", "n_train_after", "n_clusters", "p_out_percent", "e_in", "e_out", "e", "target_reached", "termination"]].sort_values("p_out_percent")


def _plot(metrics: pd.DataFrame, x: str, xlabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in RACE_METHODS:
        frame = metrics[metrics["method"] == method].sort_values("round")
        if not frame.empty:
            ax.plot(frame[x], frame["p_out_percent"], marker="o", linewidth=2.4, label=METHOD_LABELS[method], color=COLORS[method])
    ax.axhline(2.0, color="#333333", linestyle="--", linewidth=1.2, label="2% target")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P_out (%)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def analyze(root: Path) -> int:
    root = root.expanduser().resolve()
    metrics = _read(root, "metrics.csv")
    if metrics.empty:
        raise RuntimeError(f"No metrics.csv files found under {root}")
    analysis = root / "analysis"
    figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "race_config.json").read_text())
    metrics.to_csv(analysis / "all_metrics.csv", index=False)
    _final_table(metrics).to_csv(analysis / "final_paper_style_table.csv", index=False)
    (analysis / "paper_style_iteration_table.md").write_text(_markdown_table(metrics))
    (analysis / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    _plot(metrics, "round", "Acquisition iteration", figures / "pout_vs_iteration.png")
    _plot(metrics, "n_train_before", "Training points before acquisition", figures / "pout_vs_training_points.png")
    print(f"Wrote paper-style tables and figures to {analysis}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    return analyze(parser.parse_args(argv).input)


if __name__ == "__main__":
    raise SystemExit(main())
