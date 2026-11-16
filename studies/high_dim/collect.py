"""
Collect the study's run records into tables and plots.

    .venv/bin/python -m studies.high_dim.collect --results studies/high_dim/results

Writes into ``<results>``:

- ``runs.csv``: one row per run (settings, outcome, leaves, evaluations, phase times, memory,
  test errors);
- ``finest.csv``: per (problem, variant, d) the finest reached tolerance, its cost, and why the
  next tolerance failed;
- ``memory.csv``: measured peak GPU memory per run next to the prediction of
  ``predict_limits.py`` (``predicted.csv`` in the same folder, written by ``run_study.sh``);
- ``finest_tol_vs_d.png``, ``cost_vs_d.png``, ``error_vs_tol.png`` (see ``HIGH_DIM_STUDY.md``).

It needs no GPU and reads only the JSON records, so it can run while the sweep is going.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Fixed categorical order (one colour per variant, never cycled).
COLOURS = {
    "afinni": "#2a78d6", "art_affine": "#eb6834", "art_quad": "#1baf7a",
    "art_affine_m10": "#eda100", "art_quad_m4": "#6250d6",
}  # fmt: skip
MARKERS = {"afinni": "o", "art_affine": "s", "art_quad": "^", "art_affine_m10": "D", "art_quad_m4": "v"}
PHASES = ("time_augment", "time_error_calc", "time_good_trust_matrix", "time_bad_split", "time_oracle",
          "time_slicing", "time_transition")  # fmt: skip


def flatten(r: dict) -> dict:
    """One CSV row from a run record."""
    row = {k: r.get(k) for k in (
        "problem", "variant", "algorithm", "leaf", "exact", "d", "tol", "atol", "status", "reached", "failure",
        "failure_stage", "failure_where", "leaves", "f_evals", "levels", "max_depth", "converged",
        "unresolved_clusters", "time_train", "time_predict", "wall_process", "effective_batch_size",
        "gpu_peak_bytes", "gpu_pool_peak_bytes", "nvml_process_peak_bytes", "host_rss_peak_bytes",
        "err_max_rel", "err_rms_rel", "err_p99_rel", "frac_within_tol", "frac_outside", "err_max_rel_inside",
        "frac_within_tol_inside", "level_reached_log", "n0", "gpu_name", "device",
    )}  # fmt: skip
    for k in PHASES:
        row[k] = (r.get("phases") or {}).get(k)
    status = r.get("leaf_status") or {}
    for group in ("terminal_reason", "validation_status"):
        for name, count in (status.get(group) or {}).items():
            row[f"{group}:{name}"] = count
    for name in ("insufficient_sampling", "rank_deficient", "split_failed"):
        row[f"art:{name}"] = (r.get("art_counters") or {}).get(name)
    row["failure_message"] = (r.get("failure_message") or "")[:200]
    return row


def finest(df):
    """Per (problem, variant, d): the finest reached tolerance and the first failure below it."""
    import pandas as pd

    rows = []
    for (problem, variant, d), g in df.groupby(["problem", "variant", "d"]):
        g = g.sort_values("tol", ascending=False)
        ok = g[g["reached"] == True]  # noqa: E712 (pandas object column)
        best = ok.iloc[-1] if len(ok) else None
        below = g[g["tol"] < best["tol"]] if best is not None else g
        fail = below.iloc[0] if len(below) else None
        rows.append({
            "problem": problem, "variant": variant, "d": d,
            "finest_tol": None if best is None else best["tol"],
            "leaves": None if best is None else best["leaves"], "f_evals": None if best is None else best["f_evals"],
            "time_train": None if best is None else best["time_train"],
            "gpu_peak_bytes": None if best is None else best["gpu_peak_bytes"],
            "err_max_rel": None if best is None else best["err_max_rel"],
            "frac_within_tol": None if best is None else best["frac_within_tol"],
            "next_tol": None if fail is None else fail["tol"], "next_failure": None if fail is None else fail["failure"],
            "next_failure_where": None if fail is None else fail["failure_where"],
        })  # fmt: skip
    return pd.DataFrame(rows)


def plots(df, best, predicted, out: Path, ref_tol: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2,
                         "axes.edgecolor": "#8a8981", "font.size": 9})  # fmt: skip
    problems = sorted(df["problem"].unique())
    variants = [v for v in COLOURS if v in set(df["variant"])]

    # 1. Finest tolerance reached against d; a cross where no tolerance of the ladder was reached.
    fig, axes = plt.subplots(1, len(problems), figsize=(3.6 * len(problems), 3.2), squeeze=False, sharey=True)
    for ax, problem in zip(axes[0], problems, strict=True):
        for v in variants:
            g = best[(best["problem"] == problem) & (best["variant"] == v)].sort_values("d")
            if g.empty:
                continue
            hit = g[g["finest_tol"].notna()]
            ax.plot(hit["d"], hit["finest_tol"], marker=MARKERS[v], markersize=6, color=COLOURS[v], label=v)
            miss = g[g["finest_tol"].isna()]
            if len(miss):
                ax.scatter(miss["d"], [max(df["tol"])] * len(miss), marker="x", s=40, color=COLOURS[v])
        ax.set_yscale("log")
        ax.set_title(problem)
        ax.set_xlabel("d")
    axes[0][0].set_ylabel("finest tolerance reached (x std f)")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "finest_tol_vs_d.png", dpi=150)
    plt.close(fig)

    # 2. Cost against d at a reference tolerance: evaluations, training time, peak GPU memory.
    ref = df[np.isclose(df["tol"], ref_tol)]
    metrics = (("f_evals", "f evaluations"), ("time_train", "training time (s)"), ("gpu_peak_bytes", "peak GPU memory (GB)"))
    fig, axes = plt.subplots(len(metrics), len(problems), figsize=(3.6 * len(problems), 2.8 * len(metrics)),
                             squeeze=False, sharex=True)  # fmt: skip
    for j, problem in enumerate(problems):
        for i, (key, label) in enumerate(metrics):
            ax = axes[i][j]
            for v in variants:
                g = ref[(ref["problem"] == problem) & (ref["variant"] == v) & ref[key].notna()].sort_values("d")
                y = g[key] / 1e9 if key == "gpu_peak_bytes" else g[key]
                if len(g):
                    ax.plot(g["d"], y, marker=MARKERS[v], markersize=6, color=COLOURS[v], label=v)
                if key == "gpu_peak_bytes" and predicted is not None:
                    pv = predicted[(predicted["variant"] == v) & (predicted["d"] <= df["d"].max())].sort_values("d")
                    ax.plot(pv["d"], pv["est_per_node"] / 1e9, color=COLOURS[v], linewidth=1, linestyle=":")
            ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(f"{problem}, tol = {ref_tol:g}")
            if i == len(metrics) - 1:
                ax.set_xlabel("d")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.005, "dotted: predicted peak of one node (predict_limits.py est_per_node)", fontsize=7, color="#6b6a63")
    fig.tight_layout()
    fig.savefig(out / "cost_vs_d.png", dpi=150)
    plt.close(fig)

    # 3. Test error against the requested tolerance (both relative to std f), per d.
    ok = df[df["status"] == "ok"]
    fig, axes = plt.subplots(2, len(problems), figsize=(3.6 * len(problems), 5.6), squeeze=False, sharex=True)
    for j, problem in enumerate(problems):
        g = ok[ok["problem"] == problem]
        for v in variants:
            gv = g[g["variant"] == v]
            if gv.empty:
                continue
            axes[0][j].scatter(gv["tol"], gv["err_p99_rel"], c=COLOURS[v], marker=MARKERS[v], s=28, label=v,
                               edgecolors="white", linewidths=0.8)  # fmt: skip
            axes[1][j].scatter(gv["tol"], gv["frac_within_tol"], c=COLOURS[v], marker=MARKERS[v], s=28,
                               edgecolors="white", linewidths=0.8)  # fmt: skip
            for _, r in gv.iterrows():
                axes[0][j].annotate(str(int(r["d"])), (r["tol"], r["err_p99_rel"]), fontsize=6, color="#6b6a63",
                                    xytext=(3, 2), textcoords="offset points")  # fmt: skip
        t = np.array(sorted(g["tol"].unique())) if len(g) else np.array([1e-3, 1])
        axes[0][j].plot(t, t, color="#8a8981", linewidth=1, linestyle="--")
        axes[0][j].set(xscale="log", yscale="log", title=problem)
        axes[1][j].set(xscale="log", ylim=(0, 1.02), xlabel="tolerance (x std f)")
    axes[0][0].set_ylabel("test error, 99th percentile (x std f)")
    axes[1][0].set_ylabel("test share within tolerance")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.005, "dashed: error = tolerance; labels: d", fontsize=7, color="#6b6a63")
    fig.tight_layout()
    fig.savefig(out / "error_vs_tol.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--ref-tol", type=float, default=0.3, help="tolerance of the cost-vs-d plot")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.results)
    records = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))]
    if not records:
        print(f"No run records in {out / 'runs'}.", file=sys.stderr)
        return 1
    df = pd.DataFrame([flatten(r) for r in records])
    df.to_csv(out / "runs.csv", index=False)
    best = finest(df)
    best.to_csv(out / "finest.csv", index=False)
    predicted = pd.read_csv(out / "predicted.csv") if (out / "predicted.csv").exists() else None
    if predicted is not None:
        cols = ["variant", "d", "charged_per_node", "est_per_node", "nodes_per_batch_24GB", "fits_24GB", "fits_48GB",
                "fits_80GB"]  # fmt: skip
        mem = df[["problem", "variant", "d", "tol", "status", "failure", "gpu_peak_bytes", "gpu_pool_peak_bytes",
                  "host_rss_peak_bytes", "effective_batch_size"]].merge(predicted[cols], on=["variant", "d"], how="left")  # fmt: skip
        mem.to_csv(out / "memory.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 400, "display.max_columns", 20):
        print(best.to_string(index=False))
        print()
        print(df.groupby(["status", "failure"], dropna=False).size().to_string())
    if not args.no_plots:
        plots(df, best, predicted, out, args.ref_tol)
        print(f"Plots in {out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
