"""Draw the results panels of the global-approximants companion from committed records, and write results.json.

Usage, from the repository root (``make report-global-figures``):

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/reports/global-approximants/figures/results.py

Nothing is trained here. The function suite and the benchmark's step maps are read from the benchmark's store
(``results/benchmark/runs/``, through ``src.benchmark.store``), at the two tolerances the companion compares, 0.01
and 0.05; the chart recipe's studies and the comparison of every method under one protocol are read from the
theory's records (``theory/panel/data/``). A case missing from a store is reported as missing, never filled in.
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.benchmark.store import Store  # noqa: E402
from src.benchmark.table import (  # noqa: E402
    FUNCTION_CLASSES,
    RESULTS_CHANGED,
    cases,
    stale_share,
    summarise,
)
from src.problems.ode import STEP_MAPS  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
THEORY = ROOT / "theory" / "panel" / "data"
TOLS = (0.05, 0.01)
FUNCTION_METHODS = ("chebtt", "chebtt_metric", "chebtt_adaptive", "sparsecheb", "chebtree", "hub", "afinni", "art", "art_quad")
SHOWN = ("chebtt", "chebtt_metric", "chebtt_adaptive", "afinni", "art", "art_quad")
STEP_METHODS = ("chebtt_box", "chebtt_roll", "chebtt_adaptive", "slowtable", "afinni", "art", "art_quad")
COMPARED = ("recipe_100", "recipe_340", "chebtt_roll", "afinni_roll", "art_roll", "art_quad_roll")
COMPARED_PROBLEMS = ("bruss_d2", "robertson", "fisher_kpp_d3", "lorenz96_d4", "chemakzo", "allen_cahn_d5", "fisher_kpp_d5", "fisher_kpp_d10",
                     "burgers_d20", "pollu")  # fmt: skip
RECIPE_PROBLEMS = ("bruss_d2", "fisher_kpp_d2", "fisher_kpp_d3", "lorenz96_d4", "robertson", "chemakzo", "allen_cahn_d5", "fisher_kpp_d5",
                   "fisher_kpp_d10", "bruss_d20", "burgers_d20", "pollu", "hires")  # fmt: skip
COLOURS = {"chebtt": "#ef7d1a", "chebtt_metric": "#e1a800", "chebtt_adaptive": "#6d3b0f", "sparsecheb": "#7cb342", "chebtree": "#a1887f",
           "hub": "#5c6bc0", "afinni": "#1f5fa8", "art": "#2e7d32", "art_quad": "#00a6c7", "chebtt_box": "#ef7d1a", "chebtt_roll": "#c0392b",
           "slowtable": "#8e24aa", "recipe_100": "#6a1b9a", "recipe_340": "#d81b8a", "chebtt_roll_c": "#ef7d1a", "afinni_roll": "#1f5fa8",
           "art_roll": "#2e7d32", "art_quad_roll": "#00a6c7"}  # fmt: skip
LABELS = {"chebtt": "ChebTT", "chebtt_metric": "ChebTT-M", "chebtt_adaptive": "ChebTT-A", "sparsecheb": "SparseCheb", "chebtree": "ChebTree",
          "hub": "hub", "afinni": "AFINNI", "art": "ART (affine)", "art_quad": "ART (quadratic)", "chebtt_box": "ChebTT", "chebtt_roll": "ChebTT, rollout margin",
          "slowtable": "SlowTable", "recipe_100": "recipe, 100", "recipe_340": "recipe, 340", "afinni_roll": "AFINNI", "art_roll": "ART (affine)",
          "art_quad_roll": "ART (quadratic)"}  # fmt: skip

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "svg.fonttype": "path", "svg.hashsalt": "global-approximants",
    "axes.linewidth": 0.6, "axes.edgecolor": "#777777",
})  # fmt: skip


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.02, metadata={"Date": None})
    plt.close(fig)


def load(name: str) -> dict:
    return json.loads((THEORY / f"{name}.json").read_text())


def worst(rs: list) -> float:
    """The case's worst error over the tolerance over its seeds; a failed seed counts as infinite."""
    return max(float("inf") if "failed" in r else r.get("worst_ratio", float("inf")) for r in rs)


def evals(rs: list) -> float:
    """The case's mean evaluations over the seeds that recorded them (a failed run may not)."""
    v = [r["evals"] for r in rs if "evals" in r]
    return float(np.mean(v)) if v else float("nan")


def function_suite() -> dict:
    records = Store("functions", "cpu-8rank").runs()
    runs = cases(records)
    out = {"stale": {m: s for m, s in stale_share(records).items() if m in FUNCTION_METHODS}, "changed": {m: RESULTS_CHANGED[m][0] for m in FUNCTION_METHODS},
           "classes": {}, "matched": {}, "dims": {}}  # fmt: skip
    for tol in TOLS:
        at = {k: v for k, v in runs.items() if k[2] == tol}
        for cls, problems in FUNCTION_CLASSES.items():
            s = summarise(at, set(problems))
            out["classes"][f"{cls}/{tol}"] = {m: {"ran": v["ran"], "held": v["held"], "won": v["won"], "held_high": v["held_high"],
                                                  "evals": median(v["evals"]) if v["evals"] else None,
                                                  "predict_us": median(v["predict_us"]) if v["predict_us"] else None}
                                              for m, v in s.items() if m in FUNCTION_METHODS}  # fmt: skip
            # the cases every shown method ran: the worst error over the tolerance and the evaluations, side by side
            common = [k for k in at if k[0] in problems and all(m in at[k] for m in SHOWN)]
            out["matched"][f"{cls}/{tol}"] = {"cases": len(common), "methods": {m: {
                "held": sum(worst(at[k][m]) <= 1 for k in common),
                "median_ratio": median(worst(at[k][m]) for k in common) if common else None,
                "median_evals": float(np.nanmedian([evals(at[k][m]) for k in common])) if common else None} for m in SHOWN},
                "points": {m: [[evals(at[k][m]), worst(at[k][m]), k[1]] for k in common] for m in SHOWN}}  # fmt: skip
        dims = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for (_, dd, _), by in at.items():
            for m, rs in by.items():
                if m in SHOWN:
                    dims[m][dd][0] += 1
                    dims[m][dd][1] += worst(rs) <= 1
        out["dims"][str(tol)] = {m: {str(k): v for k, v in sorted(dd.items())} for m, dd in dims.items()}
    out["total_cases"] = {str(tol): sum(1 for k in runs if k[2] == tol) for tol in TOLS}
    return out


def step_suite() -> dict:
    records = Store("step_maps", "cpu-1rank").runs()
    runs = {k: v for k, v in cases(records).items() if k[2] == 0.01}
    regimes = {regime: {n for n, p in STEP_MAPS.items() if p.regime == regime} for regime in ("stiff", "nonstiff")}
    out = {}
    for regime, problems in regimes.items():
        s = summarise(runs, problems)
        out[regime] = {m: {"ran": v["ran"], "held": v["held"], "won": v["won"], "evals": median(v["evals"]) if v["evals"] else None}
                       for m, v in s.items() if m in STEP_METHODS}  # fmt: skip
    return out


def recipe() -> dict:
    r4g, p15, p17, p21, p13 = load("r4g_capped"), load("p15_denominators"), load("p17_audit"), load("p21_long"), load("p13_learning")
    p18, p20, p22 = load("p18_active"), load("p20_closed_active"), load("p22_long_training")
    out = {"problems": {}, "learning": {}, "active": {}, "closed_active": {}, "horizon": {}}
    for n in RECIPE_PROBLEMS:
        g = r4g.get(f"{n}/340", {})
        lg = p21.get(n, {}).get("arms", {}).get("decay", {})
        au = p17.get(n, {})
        cols = au.get("columns", {})
        out["problems"][n] = {"d": len(STEP_MAPS[n].system.components),
                              "served": g.get("served"), "rollout_worst": g.get("rollout_worst"), "mixing_worst": max((v["worst"] for v in g.get("mixing_chained", {}).values()), default=None),
                              "rollout_refused": g.get("rollout_solver_share"), "denominator": p15.get(n, {}).get("denominator"),
                              "denominator_us": p15.get(n, {}).get("denominator_us"), "speedup_sketched": p15.get(n, {}).get("speedup_sketched_against_denominator"),
                              "columns_worst": au.get("worst"), "columns_refused": float(np.mean([c["solver_share"] for c in cols.values()])) if cols else None,
                              "columns_speedup": au.get("median_speedup"), "long_steady": lg.get("steady_rate"), "long_fresh": lg.get("fresh_rate"),
                              "long_worst": lg.get("worst"), "long_speedup": lg.get("speedup_steady"),
                              "account_steady": p21.get(n, {}).get("arms", {}).get("account", {}).get("steady_rate"),
                              "gate_steady": p21.get(n, {}).get("arms", {}).get("gate", {}).get("steady_rate")}  # fmt: skip
    for k, v in p13.items():
        n, t = k.split("/")
        out["learning"].setdefault(n, {})[t] = {"served": v.get("served"), "rollout_worst": v.get("rollout_worst"), "refused": v.get("rollout_solver_share", v.get("refused_percent", 0) / 100)}
    for k, v in p18.items():
        n, arm = k.split("/")
        out["active"].setdefault(n, {})[arm] = {"served": v.get("served"), "rollout_worst": v.get("rollout_worst"), "mixing_refused": max(m["solver_share"] for m in v.get("mixing_chained", {}).values()) if v.get("mixing_chained") else None,
                                                "speedup": v.get("speedup_effective")}  # fmt: skip
    for k, v in p20.items():
        n, arm = k.split("/")
        rs = v.get("rounds", [])
        out["closed_active"].setdefault(n, {})[arm] = {"audit_refused": [r["audit_solver_share"] for r in rs], "audit_worst": [r["audit_worst"] for r in rs]}
    for n in p20:
        n = n.split("/")[0]
        out["closed_active"][n]["start_refused"] = out["problems"].get(n, {}).get("columns_refused")
    p23 = load("p23_full_horizon")
    for n, v in p22.items():
        full = p23.get(n, {}).get("arms", {}).get("gate", {})
        out["horizon"][n] = {"steady_short": p21.get(n, {}).get("arms", {}).get("gate", {}).get("steady_rate"),
                             "steady_400": v["arms"]["gate"]["steady_rate"], "steady_full": full.get("steady_rate"), "worst_full": full.get("worst"),
                             "speedup_full": full.get("speedup_steady")}  # fmt: skip
    return out


def problems() -> dict:
    """The two suites' problems: each function's box and the dimensions it ran at, each step map's size and setting."""
    from src.benchmark.step_maps import dim
    from src.problems.registry import FUNCTIONS, GRID, N_TEST

    dims = defaultdict(set)
    for r in Store("functions", "cpu-8rank").runs():
        dims[r["problem"]].add(int(r["d"]))
    fn = {cls: {n: {"lo": FUNCTIONS[n].lo, "hi": FUNCTIONS[n].hi, "vector": FUNCTIONS[n].vector, "dims": sorted(dims[n])} for n in names}
          for cls, names in FUNCTION_CLASSES.items()}  # fmt: skip
    st = {n: {"regime": p.regime, "d": dim(n), "held": p.system.held, "dt": p.dt, "steps": p.steps, "train": p.n_train, "test": p.n_test,
              "log": bool(p.system.log)} for n, p in STEP_MAPS.items()}  # fmt: skip
    seeds = len({r["seed"] for r in Store("functions", "cpu-8rank").runs()})
    return {"functions": fn, "step_maps": st, "function_test": {"grid": GRID, "n_test": N_TEST, "seeds": seeds}}


def compared() -> dict:
    """Every comparison record at both tolerances; a case whose training took longer than the study's compute budget is marked."""
    import re

    budget = int(re.search(r"^BUDGET = (\d+)", (ROOT / "theory" / "panel" / "p25_compare.py").read_text(), re.M).group(1))
    out = {"budget_s": budget}
    for tol in TOLS:
        path = THEORY / f"p25_compare_{tol}.json"
        recs = {}
        for f in sorted((THEORY / "records" / "p25").glob(f"*_{tol}.json")):
            r = json.loads(f.read_text())
            k = r["key"]
            if k["tol"] != tol:
                continue
            recs[f"{k['problem']}/{k['method']}"] = {"status": r.get("status"), "failed": (r.get("failure") or {}).get("message"),
                **{x: r.get(x) for x in ("rollout_worst", "rollout_fresh_rate", "rollout_steady_rate", "solver_calls", "train_s", "evaluate_us_1000", "denominator_us",
                                          "one_step_tol", "margin_rounds", "served")},
                "columns_worst": max((c["worst"] for c in r.get("columns", {}).values()), default=None),
                "columns_refused": float(np.mean([c["solver_share"] for c in r["columns"].values()])) if r.get("columns") else None,
                "mixing_worst": max((m["worst"] for m in r.get("mixing", {}).values()), default=None),
                "long": {x: (r.get("long") or {}).get(x) for x in ("steady_rate", "fresh_rate", "worst", "speedup_steady", "certified_horizon")},
                "over_budget": bool(r.get("train_s") and r["train_s"] > budget)}  # fmt: skip
        out[str(tol)] = {"records": recs, "summary_file": path.exists()}
    return out


def panels(fs: dict, cmp: dict, rc: dict) -> None:
    classes = list(FUNCTION_CLASSES)
    # held share by class, method and tolerance
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    for ax, tol in zip(axes, TOLS, strict=True):
        w = 0.8 / len(SHOWN)
        for i, m in enumerate(SHOWN):
            share = []
            for c in classes:
                v = fs["classes"][f"{c}/{tol}"].get(m)
                share.append(v["held"] / v["ran"] if v and v["ran"] else 0.0)
            ax.barh(np.arange(len(classes)) + (i - len(SHOWN) / 2 + 0.5) * w, share, height=w, color=COLOURS[m], label=LABELS[m])
        ax.set_yticks(np.arange(len(classes)), classes)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"share of cases held (tolerance {tol})")
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=6, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.07))
    save(fig, "functions-held")
    # the matched cases: evaluations against the worst error over the tolerance
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, tol in zip(axes, TOLS, strict=True):
        for m in SHOWN:
            pts = [p for c in classes for p in fs["matched"][f"{c}/{tol}"]["points"][m]]
            if pts:
                P = np.array(pts, dtype=float)
                ax.scatter(P[:, 0], np.minimum(P[:, 1], 1e3), s=5, color=COLOURS[m], alpha=0.7, label=LABELS[m], lw=0)
        ax.axhline(1, color="#c0392b", lw=0.6, ls="--")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlabel(f"evaluations (tolerance {tol})")
    axes[0].set_ylabel("worst error / tolerance")
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=6, fontsize=6, frameon=False, markerscale=2, bbox_to_anchor=(0.5, 1.07))
    save(fig, "functions-matched")
    # the comparison under one protocol: rollout error and steady speed-up per problem
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.2), sharex=True)
    x = np.arange(len(COMPARED_PROBLEMS))
    for j, tol in enumerate(TOLS):
        recs = cmp[str(tol)]["records"]
        for i, m in enumerate(COMPARED):
            dx = (i - len(COMPARED) / 2 + 0.5) * 0.12
            for p_i, p in enumerate(COMPARED_PROBLEMS):
                r = recs.get(f"{p}/{m}")
                if r is None:
                    continue
                if r["status"] != "done":
                    axes[0, j].plot(p_i + dx, 30, "x", color=COLOURS[m], ms=3.5)
                    axes[1, j].plot(p_i + dx, 0.3, "x", color=COLOURS[m], ms=3.5)
                    continue
                # a case that finished beyond the compute budget keeps its values, drawn hollow
                face = "none" if r["over_budget"] else COLOURS[m]
                axes[0, j].plot(p_i + dx, max(r["rollout_worst"], 1e-7) / tol, "o", color=COLOURS[m], mfc=face, ms=3)
                sp = r["long"].get("speedup_steady")
                if sp:
                    axes[1, j].plot(p_i + dx, sp, "o", color=COLOURS[m], mfc=face, ms=3)
        axes[0, j].axhline(0.5, color="#c0392b", lw=0.6, ls="--")
        axes[1, j].axhline(1, color="#424242", lw=0.6, ls="--")
        axes[0, j].set_yscale("log"), axes[1, j].set_yscale("log")
        axes[0, j].set_title(f"tolerance {tol}", fontsize=7)
        axes[1, j].set_xticks(x, COMPARED_PROBLEMS, rotation=60, ha="right")
    axes[0, 0].set_ylabel("rollout worst / tolerance")
    axes[1, 0].set_ylabel("steady speed-up (long chains)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOURS[m], label=LABELS[m]) for m in COMPARED]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save(fig, "compare-steps")
    # the recipe: long-chain steady refusal and speed-up per problem
    probs = [n for n in RECIPE_PROBLEMS if rc["problems"][n]["long_speedup"]]
    fig, ax = plt.subplots(figsize=(5.0, 1.9))
    ax.bar(np.arange(len(probs)), [rc["problems"][n]["long_speedup"] for n in probs], color=COLOURS["recipe_340"])
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(probs)), probs, rotation=50, ha="right")
    ax.set_ylabel("steady speed-up over LSODA/VODE")
    ax.axhline(1, color="#424242", lw=0.6, ls="--")
    save(fig, "recipe-long")
    # refusal-driven data in closed columns
    probs = sorted(rc["closed_active"])
    fig, ax = plt.subplots(figsize=(5.0, 1.9))
    for i, n in enumerate(probs):
        v = rc["closed_active"][n]
        if v.get("start_refused") is not None:
            ax.plot(i - 0.2, 100 * v["start_refused"], "s", color="#8a8a8a", ms=3.5)
        for arm, dx, c in (("uniform", 0.0, "#1f5fa8"), ("refused", 0.2, "#d81b8a")):
            if v.get(arm, {}).get("audit_refused"):
                ax.plot(i + dx, 100 * v[arm]["audit_refused"][-1], "o", color=c, ms=3.5)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(probs)), probs, rotation=50, ha="right")
    ax.set_ylabel("closed columns refused (%)")
    save(fig, "recipe-active")


def clean(v):
    """JSON without NaN or infinity (Typst's reader rejects them): a non-finite number becomes null."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def main() -> None:
    fs = function_suite()
    st = step_suite()
    rc = recipe()
    cmp = compared()
    panels(fs, cmp, rc)
    for m in ("points",):
        for v in fs["matched"].values():
            v.pop(m)
    (OUT / "results.json").write_text(json.dumps(clean({"problems": problems(), "functions": fs, "steps": st, "recipe": rc, "compare": cmp, "labels": LABELS, "colours": COLOURS}),
                                                 indent=1, sort_keys=True, default=str) + "\n")  # fmt: skip


if __name__ == "__main__":
    main()
