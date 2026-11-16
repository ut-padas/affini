"""
Collect Part B: tables, convergence slopes, pass/fail checks and plots comparing the reference
ART with AFINNI's ART.

Usage: ``collect_art.py --out OUT`` (the ``--out`` of run_art.sh). Writes to ``OUT/report/``:

- ``runs.csv``: every run, flattened.
- ``paired.csv`` / ``.md``: per (problem, d, degree, tol) the median over seeds of both codes'
  leaves, evaluations, errors, share of test points above the tolerance and depth-limited leaves,
  with AFINNI / reference ratios.
- ``slopes.csv``: per code the log-log slope of relative L2 error and of the 99th-percentile
  pointwise relative error against leaves, with standard errors and the theoretical slope.
- ``checks.csv`` / ``.md``: checks B1-B8 (CONVERGENCE_STUDY.md, Part B) as pass / WARN / FAIL.
- ``err_vs_leaves_<problem>_deg<k>.png``, ``frac_vs_tol_<problem>_deg<k>.png``, ``depth_sweep.png``.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
SIDES = {"art": "reference ART", "afinni": "AFINNI's ART"}
COMPARED = ("leaves", "oracle_queries", "max_depth", "test_relative_l2", "test_rmse", "test_max_absolute_error",
            "max_pointwise_rel", "q99", "frac_pointwise_rel_gt_tol")  # fmt: skip


def flag(argv: list[str], name: str, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def load(results: Path) -> list[dict]:
    runs = []
    for path in sorted(results.glob("*/*.json")):
        rec = json.loads(path.read_text())
        argv, r = rec["argv"], rec["result"]
        reasons = r.get("terminal_reasons", {})
        leaves = r["leaves"]
        runs.append({
            "group": path.parent.name, "file": path.name, "case": rec["case"], "side": rec["side"],
            "tag": ".".join(path.stem.split(".")[2:]), "argv": " ".join(argv), "problem": flag(argv, "--benchmark"),
            "dim": int(flag(argv, "--dimension")), "degree": int(flag(argv, "--leaf-degree", 1)),
            "tol": float(flag(argv, "--error-tolerance")), "depth_limit": int(flag(argv, "--max-depth", 10)),
            "seed": int(flag(argv, "--seed", 1)), "mpi_size": rec["mpi_size"], "afinni_gpu": rec["afinni_gpu"],
            **{k: r.get(k) for k in COMPARED}, "median_pointwise_rel": r.get("median_pointwise_rel"),
            "depth_limited_leaves": reasons.get("max_depth", 0),
            "depth_limited_share": reasons.get("max_depth", 0) / leaves if leaves else float("nan"),
            "build_s": r["build_seconds"], "wall_s": rec["wall_s"], "peak_rss_gib": rec["peak_rss_gib_rank0"],
        })  # fmt: skip
    return runs


def theory_slope(problem: str, d: int, degree: int) -> float:
    """Error ~ h^(k+1) and leaves ~ h^-d_eff, so error ~ leaves^(-(k+1)/d_eff); a plane wave varies along one direction."""
    return -(degree + 1) / (1 if problem == "plane_wave" else d)


def fit(x, y):
    pts = [(math.log(a), math.log(b)) for a, b in zip(x, y) if a and b and a > 0 and b > 0]
    X, Y = (np.array(v) for v in zip(*pts)) if len(pts) >= 2 else (np.array([]), np.array([]))
    if len(pts) < 3 or np.ptp(X) == 0:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([X, np.ones_like(X)]).T
    coef, res, *_ = np.linalg.lstsq(A, Y, rcond=None)
    s2 = float(res[0]) / (len(pts) - 2) if res.size else 0.0
    return float(coef[0]), math.sqrt(s2 / float(np.sum((X - X.mean()) ** 2))), float(coef[1])


def median(values):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if v else float("nan")


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        path.with_suffix(".md").write_text("(no rows)\n")
        return
    keys = list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.3g}"
        return str(v)

    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def paired(runs):
    cells = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r["group"] == "conv" and not r["tag"]:
            cells[(r["problem"], r["dim"], r["degree"], r["tol"])][r["side"]].append(r)
    rows = []
    for (problem, d, degree, tol), by in sorted(cells.items()):
        row = {"problem": problem, "dim": d, "degree": degree, "tol": tol,
               "seeds": min(len(by["art"]), len(by["afinni"]))}  # fmt: skip
        for side in ("art", "afinni"):
            for k in ("leaves", "oracle_queries", "test_relative_l2", "q99", "max_pointwise_rel",
                      "frac_pointwise_rel_gt_tol", "depth_limited_share", "build_s"):  # fmt: skip
                row[f"{side}_{k}"] = median(r[k] for r in by[side])
        for k in ("leaves", "oracle_queries", "test_relative_l2", "q99"):
            a, b = row[f"afinni_{k}"], row[f"art_{k}"]
            row[f"ratio_{k}"] = a / b if b and not math.isnan(a) else float("nan")
        rows.append(row)
    return rows


def slopes(runs):
    groups = defaultdict(list)
    for r in runs:
        if r["group"] == "conv" and not r["tag"]:
            groups[(r["problem"], r["dim"], r["degree"], r["side"])].append(r)
    out = []
    for (problem, d, degree, side), rs in sorted(groups.items()):
        s, se, c = fit([r["leaves"] for r in rs], [r["test_relative_l2"] for r in rs])
        s99, se99, c99 = fit([r["leaves"] for r in rs], [r["q99"] for r in rs])
        out.append({"problem": problem, "dim": d, "degree": degree, "side": side, "points": len(rs),
                    "slope_relL2": s, "se_relL2": se, "intercept_relL2": c, "slope_q99": s99, "se_q99": se99,
                    "intercept_q99": c99, "theory": theory_slope(problem, d, degree),
                    "leaf_range": f"{min(r['leaves'] for r in rs)}-{max(r['leaves'] for r in rs)}"})  # fmt: skip
    return out


def same_result(a: dict, b: dict, rtol: float = 0.0) -> tuple[bool, str]:
    diffs = []
    for k in COMPARED:
        x, y = a[k], b[k]
        if x == y:
            continue
        if rtol and isinstance(x, float) and isinstance(y, float) and math.isclose(x, y, rel_tol=rtol, abs_tol=1e-300):
            continue
        diffs.append(f"{k}: {x} vs {y}")
    return not diffs, "; ".join(diffs[:4])


def checks(runs, pairs, slope_rows, exact_path: Path):
    out = []

    def add(ref, check, value, limit, result, note=""):
        out.append({"ref": ref, "check": check, "value": value, "limit": limit, "result": result, "note": note})

    for p in pairs:
        ref = f"{p['problem']} d={p['dim']} deg={p['degree']} tol={p['tol']}"
        if p["seeds"] == 0:
            add(ref, "B1/B2/B5 paired checks", float("nan"), float("nan"), "missing", "one code has no run")
            continue
        fa, fr = p["afinni_frac_pointwise_rel_gt_tol"], p["art_frac_pointwise_rel_gt_tol"]
        limit = max(2 * fr, fr + 0.005)
        add(ref, "B1 share above tol (AFINNI) <= max(2 ref, ref + 0.5%)", fa, limit, "pass" if fa <= limit else "FAIL")
        q_a, q_r = p["afinni_q99"], p["art_q99"]
        if q_r <= p["tol"] and q_a > 1.2 * p["tol"]:
            add(ref, "B1b q99 <= 1.2 tol where the reference has q99 <= tol", q_a, 1.2 * p["tol"], "FAIL")
        ratio = p["ratio_leaves"]
        res = "pass" if 0.5 <= ratio <= 2 else "WARN" if 1 / 3 <= ratio <= 3 else "FAIL"
        add(ref, "B2 leaves AFINNI / reference in [0.5, 2] (FAIL outside [1/3, 3])", ratio, 2.0, res)
        sa, sr = p["afinni_depth_limited_share"], p["art_depth_limited_share"]
        add(ref, "B5 depth-limited leaf share AFINNI <= reference + 5%", sa, sr + 0.05, "pass" if sa <= sr + 0.05 else "FAIL")
    by = {(s["problem"], s["dim"], s["degree"], s["side"]): s for s in slope_rows}
    for (problem, d, degree, side), s in sorted(by.items()):
        if side != "afinni" or (problem, d, degree, "art") not in by:
            continue
        r = by[(problem, d, degree, "art")]
        ref = f"{problem} d={d} deg={degree}"
        for metric in ("relL2", "q99"):
            sa, sr = s[f"slope_{metric}"], r[f"slope_{metric}"]
            if math.isnan(sa) or math.isnan(sr):
                add(ref, f"B3 slope({metric}) agreement", float("nan"), float("nan"), "missing", "fewer than 3 points")
                continue
            band = 2 * math.hypot(s[f"se_{metric}"], r[f"se_{metric}"]) + 0.15
            add(ref, f"B3 |slope({metric}) AFINNI - reference| <= 2 se + 0.15", sa - sr, band,
                "pass" if abs(sa - sr) <= band else "FAIL", f"AFINNI {sa:.2f}, reference {sr:.2f}, theory {s['theory']:.2f}")  # fmt: skip
        # B4: error at the same leaf count, from each code's own fit at the reference's median leaf count.
        rs = [x for x in runs if x["group"] == "conv" and not x["tag"] and x["problem"] == problem and x["dim"] == d
              and x["degree"] == degree and x["side"] == "art"]  # fmt: skip
        if rs and not math.isnan(s["slope_relL2"]) and not math.isnan(r["slope_relL2"]):
            L = math.log(median(x["leaves"] for x in rs))
            ratio = math.exp((s["slope_relL2"] * L + s["intercept_relL2"]) - (r["slope_relL2"] * L + r["intercept_relL2"]))
            res = "pass" if 0.5 <= ratio <= 2 else "FAIL" if ratio > 2 else "WARN"
            add(ref, "B4 relL2 at equal leaves, AFINNI / reference in [0.5, 2]", ratio, 2.0, res,
                "WARN: AFINNI better by > 2x, worth a look" if res == "WARN" else "")  # fmt: skip
    for r in runs:
        if r["group"] == "exact_target":
            ok = r["leaves"] == 1 and (r["max_pointwise_rel"] or 1) <= 1e-6
            add(f"{r['case']} {SIDES[r['side']]}", "B6 exactly representable target: 1 leaf, max rel err <= 1e-6",
                r["max_pointwise_rel"], 1e-6, "pass" if ok else "FAIL", f"leaves {r['leaves']}")  # fmt: skip
    if exact_path.exists():
        for e in json.loads(exact_path.read_text()):
            if e.get("check") == "slab":
                for side in ("ref", "afinni"):
                    add(f"slab aspect {e['aspect']:g} deg={e['degree']} {SIDES['art' if side == 'ref' else 'afinni']}",
                        "B6 exactly affine target on a thin box: max rel err <= 1e-6", e[f"max_rel_err_{side}"], 1e-6,
                        "pass" if e[f"result_{side}"] == "pass" else "FAIL")  # fmt: skip
            else:
                add(f"root fit {e['problem']} d={e['dim']} deg={e['degree']} tol={e['tol']}",
                    "B6 root fit: same model, error and decision", e.get("pred_diff_rel"), 1e-6,
                    "pass" if e["result"] == "pass" else "FAIL", e.get("error", ""))  # fmt: skip
    mpi = defaultdict(dict)
    for r in runs:
        if r["group"] == "mpi":
            mpi[r["case"]][r["mpi_size"]] = r
    for case, by_p in mpi.items():
        base = by_p.get(min(by_p))
        for p, r in sorted(by_p.items()):
            if r is base:
                continue
            ok, why = same_result(base, r)
            add(f"{case} np={p}", "B7 AFINNI's ART identical on any rank count (with stealing)", r["leaves"], base["leaves"],
                "pass" if ok else "FAIL", why)  # fmt: skip
    index = {(r["side"], r["argv"]): r for r in runs if r["group"] == "conv" and not r["tag"]}
    for r in runs:
        twin = index.get((r["side"], r["argv"]))
        if r["group"] == "repeat" and twin:
            ok, why = same_result(twin, r)
            add(f"{r['case']} {SIDES[r['side']]}", "B8 same seed, same machine: identical result", r["leaves"], twin["leaves"],
                "pass" if ok else "FAIL", why)  # fmt: skip
        if r["group"] == "gpu_vs_cpu" and twin:
            exact = twin["leaves"] == r["leaves"] and twin["oracle_queries"] == r["oracle_queries"]
            ok, why = same_result(twin, r, rtol=1e-8)
            add(f"{r['case']}", "B8b AFINNI's ART GPU vs CPU: same leaves and evaluations, errors to 1e-8",
                twin["leaves"], r["leaves"], "pass" if exact and ok else "WARN" if exact else "FAIL", why)  # fmt: skip
    return out


def plots(runs, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "font.size": 9})  # fmt: skip
    colors = {"art": BLUE, "afinni": ORANGE}
    conv = [r for r in runs if r["group"] == "conv" and not r["tag"]]
    for problem, degree in sorted({(r["problem"], r["degree"]) for r in conv}):
        rs = [r for r in conv if r["problem"] == problem and r["degree"] == degree]
        dims = sorted({r["dim"] for r in rs})
        for kind in ("err_vs_leaves", "frac_vs_tol"):
            fig, axes = plt.subplots(1, len(dims), figsize=(3.3 * len(dims), 2.9), squeeze=False)
            for ax, d in zip(axes[0], dims):
                for side in ("art", "afinni"):
                    pts = [r for r in rs if r["dim"] == d and r["side"] == side]
                    if kind == "err_vs_leaves":
                        ax.plot([r["leaves"] for r in pts], [r["test_relative_l2"] for r in pts], "o", color=colors[side],
                                ms=4, alpha=0.6, label=SIDES[side])  # fmt: skip
                        meds = sorted((median(x["leaves"] for x in pts if x["tol"] == t),
                                       median(x["test_relative_l2"] for x in pts if x["tol"] == t)) for t in {x["tol"] for x in pts})  # fmt: skip
                        if meds:
                            ax.plot(*zip(*meds), "-", color=colors[side], lw=1.5)
                    else:
                        meds = sorted((t, median(x["frac_pointwise_rel_gt_tol"] for x in pts if x["tol"] == t))
                                      for t in {x["tol"] for x in pts})  # fmt: skip
                        if meds:
                            ax.plot(*zip(*meds), "o-", color=colors[side], ms=4, lw=1.5, label=SIDES[side])
                if kind == "err_vs_leaves":
                    xs = [r["leaves"] for r in rs if r["dim"] == d]
                    ys = [r["test_relative_l2"] for r in rs if r["dim"] == d and r["side"] == "art"]
                    if xs and ys:
                        x0, x1 = min(xs), max(xs)
                        y0 = max(ys)
                        k = theory_slope(problem, d, degree)
                        ax.plot([x0, x1], [y0, y0 * (x1 / x0) ** k], color=MUTED, lw=1, ls="--", label=f"slope {k:.2f}")
                    ax.set_xlabel("leaves")
                    ax.set_ylabel("test relative L2 error")
                    ax.set_yscale("log")
                else:
                    ax.set_xlabel("tolerance")
                    ax.set_ylabel("share of test points above tol")
                    ax.set_yscale("symlog", linthresh=1e-4)
                ax.set_xscale("log")
                ax.set_title(f"{problem} d={d}, degree {degree}", color=INK)
                ax.legend(fontsize=7, frameon=False)
            fig.tight_layout()
            fig.savefig(out / f"{kind}_{problem}_deg{degree}.png", dpi=130)
            plt.close(fig)
    depth = [r for r in runs if r["group"] == "depth"]
    if depth:
        dims = sorted({r["dim"] for r in depth})
        fig, axes = plt.subplots(1, len(dims), figsize=(3.3 * len(dims), 2.9), squeeze=False)
        for ax, d in zip(axes[0], dims):
            for side in ("art", "afinni"):
                pts = [r for r in depth if r["dim"] == d and r["side"] == side]
                ax.plot([r["leaves"] for r in pts], [r["frac_pointwise_rel_gt_tol"] for r in pts], "o", color=colors[side],
                        ms=4, label=SIDES[side])  # fmt: skip
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("leaves")
            ax.set_ylabel("share above tol 0.01")
            ax.set_title(f"spherical_piecewise d={d} (depth sweep)", color=INK)
            ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "depth_sweep.png", dpi=130)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    report = a.out / "report"
    report.mkdir(parents=True, exist_ok=True)
    runs = load(a.out / "results")
    write_table(report / "runs.csv", runs)
    pairs = paired(runs)
    write_table(report / "paired.csv", pairs)
    slope_rows = slopes(runs)
    write_table(report / "slopes.csv", slope_rows)
    chk = checks(runs, pairs, slope_rows, a.out / "results" / "exact_checks.json")
    write_table(report / "checks.csv", chk)
    plots(runs, report)
    counts = defaultdict(int)
    for c in chk:
        counts[c["result"]] += 1
    print(f"{len(runs)} runs, {len(pairs)} paired cells; checks: {dict(counts)}; report in {report}")
    for c in chk:
        if c["result"] in ("FAIL", "WARN"):
            v = c["value"]
            print(f"{c['result']} {c['ref']}: {c['check']} ({v if not isinstance(v, float) else f'{v:.4g}'}) {c['note']}")


if __name__ == "__main__":
    main()
