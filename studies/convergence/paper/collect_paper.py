"""
Collect Part A: compare every number the manuscript reports with the re-runs, check its accuracy
claims, and plot the convergence.

Usage: ``collect_paper.py --out OUT`` (the ``--out`` of run_paper.sh). Writes to ``OUT/report/``:

- ``comparison.csv`` / ``comparison.md``: one row per reported number: paper value, re-run value
  (seed 0), the seed replicates' spread, relative difference and a verdict (``same`` within the
  noise band, ``differs`` outside it, ``missing``).
- ``checks.csv`` / ``checks.md``: the pass/fail accuracy checks (CONVERGENCE_STUDY.md, Part A).
- ``slopes.csv``: fitted log-log slopes of e_in against N_c, paper and re-run, beside -2/d.
- ``runs.csv``: every run found, flattened; ``cost.csv``: wall time and rank 0's peak memory per case;
  ``*.png``: the plots.

The noise band of a quantity is max(3 sigma, floor): sigma is the standard deviation of log(value)
over the seed replicates (when there are at least two), the floor 25% for N, N_c, e_in, e and e_nh
(the manuscript's own Lorenz d=5 rtol 0.1 runs differ by 17% in N_c) and max(1 point, 30%) for p_o.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import COARSE  # noqa: E402
from paper_values import PAPER_NOTES, records  # noqa: E402

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
COUNT_FLOOR = math.log(1.25)
QUANTITIES = ("N", "N_c", "p_o", "e_in", "e", "e_nh", "p_in_good")


def load_runs(results: Path) -> list[dict]:
    """Every launcher result JSON under results/seed_*/<problem>/, flattened."""
    runs = []
    for path in sorted(results.glob("seed_*/*/*.json")):
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping unreadable {path}: {exc}")
            continue
        m = data["error_metrics"]
        n_in = m["n_test"] - m["n_out"]
        runs.append({
            "file": str(path.relative_to(results)), "problem": path.parent.name,
            "seed": int(path.parent.parent.name.split("_")[1]), "dim": data["dim"],
            "atol": data["atol"], "rtol": data["rtol"], "mode": data["error_eval_mode"].removesuffix("_v2"),
            "N_0": data["N_0"], "mpi_size": data["mpi_size"], "coarse_factor": data["coarse_factor"],
            "num_initial_clusters": data["num_initial_clusters"], "min_pts_per_cluster": data["min_pts_per_cluster"],
            "gpu": data["IS_CUPY_BACKEND"], "N": data["N"], "N_c": data["N_c"], "p_o": m["perct_out"],
            "e_in": m["in"]["l2_rel_error"], "e": m["all"]["l2_rel_error"],
            "p_in_good": 100.0 * m["n_in_good"] / n_in if n_in else float("nan"),
            "converged": data["converged"], "unresolved": data["num_unresolved_pts"],
            "train_s": data["Train_time_s"], "levels": data["N_levels_max"],
        })  # fmt: skip
    return runs


def same_tol(run: dict, rec: dict) -> bool:
    if "atol" in rec:  # Table 3: atol only
        return run["rtol"] is None and math.isclose(run["atol"] or -1, rec["atol"])
    if rec["problem"].startswith("lorenz96"):  # atol only, scaled by max|f|
        return run["rtol"] is None and math.isclose(run["atol"] or -1, rec["tol"])
    return math.isclose(run["atol"] or -1, rec["tol"]) and math.isclose(run["rtol"] or -1, rec["tol"])


def matching(runs: list[dict], rec: dict, mode: str | None = None) -> list[dict]:
    """The single-rank runs with the record's settings (every seed)."""
    mode = mode or rec["mode"]
    if rec["group"] == "lorenz_dims" or (rec["group"] == "wave_exact" and mode.endswith("exact_grad")):
        coarse = 10
    else:
        coarse = COARSE[rec["dim"]]
    out = []
    for r in runs:
        if r["problem"] != rec["problem"] or r["dim"] != rec["dim"] or r["mode"] != mode or r["mpi_size"] != 1:
            continue
        if r["coarse_factor"] != coarse or not same_tol(r, rec):
            continue
        if "N_0" in rec and r["N_0"] != rec["N_0"]:
            continue
        if rec["group"] == "conv" and r["N_0"] != int(50_000 * 1.3 ** rec["dim"]):
            continue
        if rec["group"] == "lorenz_dims" and r["N_0"] != min(int(5e4 * 1.3 ** rec["dim"]), 10**6):
            continue
        out.append(r)
    return out


def verdict(quantity: str, paper: float, values: list[float], seed0: float | None) -> tuple[str, float, float]:
    """(verdict, relative difference, noise band as a log ratio or in points for p_o)."""
    if seed0 is None or not math.isfinite(seed0):
        return "missing", float("nan"), float("nan")
    rel = (seed0 - paper) / paper if paper else float("nan")
    logs = [math.log(v) for v in values if v and v > 0 and math.isfinite(v)]
    sigma = float(np.std(logs, ddof=1)) if len(logs) >= 2 else 0.0
    if quantity in ("p_o", "p_in_good"):
        band = max(1.0, 0.3 * abs(paper), 3 * sigma * abs(paper))
        return ("same" if abs(seed0 - paper) <= band else "differs"), rel, band
    band = max(COUNT_FLOOR, 3 * sigma)
    if paper <= 0 or seed0 <= 0:
        return "differs", rel, band
    return ("same" if abs(math.log(seed0 / paper)) <= band else "differs"), rel, band


def slope(x: list[float], y: list[float]) -> tuple[float, float]:
    """Least-squares slope of log y against log x and its standard error."""
    pts = [(math.log(a), math.log(b)) for a, b in zip(x, y) if a and b and a > 0 and b > 0]
    if len(pts) < 2:
        return float("nan"), float("nan")
    X, Y = np.array(pts).T
    if len(pts) == 2:
        return float((Y[1] - Y[0]) / (X[1] - X[0])), float("nan")
    A = np.vstack([X, np.ones_like(X)]).T
    coef, res, *_ = np.linalg.lstsq(A, Y, rcond=None)
    dof = len(pts) - 2
    s2 = float(res[0]) / dof if res.size else 0.0
    se = math.sqrt(s2 / float(np.sum((X - X.mean()) ** 2)))
    return float(coef[0]), se


def write_table(path_csv: Path, rows: list[dict]) -> None:
    if not rows:
        path_csv.write_text("")
        path_csv.with_suffix(".md").write_text("(no rows)\n")
        return
    keys = list(rows[0])
    with path_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.3g}"
        return "" if v is None else str(v)

    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    path_csv.with_suffix(".md").write_text("\n".join(lines) + "\n")


def compare(runs: list[dict]) -> tuple[list[dict], dict]:
    """The comparison table, and per record the matched runs (for the checks and plots)."""
    rows, matched = [], {}
    for rec in records():
        hits = matching(runs, rec)
        nh = matching(runs, rec, "affine_cov_no_hess") if "e_nh" in rec else []
        matched[rec["ref"]] = (rec, hits, nh)
        for q in QUANTITIES:
            if q not in rec:
                continue
            source = nh if q == "e_nh" else hits
            key = "e" if q == "e_nh" else q
            seed0 = next((r[key] for r in source if r["seed"] == 0), None)
            values = [r[key] for r in source]
            v, rel, band = verdict(q, rec[q], values, seed0)
            rows.append({
                "ref": rec["ref"], "loc": rec["loc"], "problem": rec["problem"], "dim": rec["dim"],
                "tol": rec.get("tol", rec.get("atol")), "quantity": q, "paper": rec[q], "rerun_seed0": seed0,
                "n_seeds": len(values), "seed_min": min(values) if values else None,
                "seed_max": max(values) if values else None, "rel_diff": rel, "band": band, "verdict": v,
            })  # fmt: skip
    return rows, matched


def checks(matched: dict, runs: list[dict]) -> list[dict]:
    """Pass/fail accuracy checks; a check with no run is 'missing', never 'pass'."""
    out = []

    def add(ref, check, value, limit, ok, note=""):
        out.append({"ref": ref, "check": check, "value": value, "limit": limit,
                    "result": "missing" if ok is None else ("pass" if ok else "FAIL"), "note": note})  # fmt: skip

    for ref, (rec, hits, nh) in matched.items():
        seed0 = next((r for r in hits if r["seed"] == 0), None)
        if rec["group"] in ("conv", "lorenz_dims", "wave_exact") and "e_in" in rec:
            tol = rec["tol"]
            e_in = seed0["e_in"] if seed0 else float("nan")
            note = "paper also exceeds" if rec["e_in"] > tol else ""
            add(ref, "A1 e_in <= 1.1 tol", e_in, 1.1 * tol, None if seed0 is None else e_in <= 1.1 * tol, note)
            worst = max((r["e_in"] for r in hits), default=float("nan"))
            if len(hits) > 1:
                add(ref, "A1s e_in <= 1.1 tol, every seed", worst, 1.1 * tol, worst <= 1.1 * tol)
        if seed0 is not None:
            add(ref, "A5 converged, no unresolved points", seed0["unresolved"], 0,
                bool(seed0["converged"]) and seed0["unresolved"] == 0)  # fmt: skip
        if "e_nh" in rec:
            nh0 = next((r for r in nh if r["seed"] == 0), None)
            claim_holds_in_paper = rec["e"] <= rec["e_nh"]
            if seed0 is None or nh0 is None:
                add(ref, "A2 e(Hessian split) <= 1.25 e(no Hessian)", float("nan"), float("nan"), None)
            elif claim_holds_in_paper:
                add(ref, "A2 e(Hessian split) <= 1.25 e(no Hessian)", seed0["e"], 1.25 * nh0["e"], seed0["e"] <= 1.25 * nh0["e"])
            else:
                add(ref, "A2 e(Hessian split) <= 1.25 e(no Hessian)", seed0["e"], 1.25 * nh0["e"], True,
                    "not claimed for this row (paper has e > e_nh)")  # fmt: skip
        if rec["group"] == "rosen_weak":
            good = seed0["p_in_good"] if seed0 else float("nan")
            add(ref, "A3 p_in_good >= 95%", good, 95.0, None if seed0 is None else good >= 95.0)
    # A4: the slope claim (Figure 7): e_in falls at least as fast as N_c^(-2/d), quadratic d=5 excepted.
    groups = defaultdict(list)
    for rec, hits, _ in matched.values():
        if rec["group"] == "conv" and "N" in rec:
            groups[(rec["problem"], rec["dim"])].append((rec, next((r for r in hits if r["seed"] == 0), None)))
    for (problem, d), items in sorted(groups.items()):
        items = [(r, h) for r, h in items if h is not None]  # the paper's slope over the same rows
        sp, _ = slope([r["N_c"] for r, _ in items], [r["e_in"] for r, _ in items])
        got = [(h["N_c"], h["e_in"]) for _, h in items]
        sr, se = slope([a for a, _ in got], [b for _, b in got])
        theory = -2.0 / d
        note = f"{len(items)} tolerances; " if items else "no runs; "
        ok = None if math.isnan(sr) else sr <= theory + 0.1 + (2 * se if math.isfinite(se) else 0.0)
        note += f"paper slope {sp:.2f}" + ("; paper exempts this case" if (problem.startswith("anisotropic_quadratic") and d == 5) else "")
        add(f"{problem} d={d}", "A4 slope(e_in vs N_c) <= -2/d + 0.1 (+2 se)", sr, theory + 0.1, ok, note)
        ok = None if math.isnan(sr) else abs(sr - sp) <= max(0.2, 2 * se if math.isfinite(se) else 0.0)
        add(f"{problem} d={d}", "A4b |slope - paper slope| <= max(0.2, 2 se)", sr - sp if not math.isnan(sr) else sr, 0.2, ok)
    # A6: MPI. AFINNI's trees depend on the rank count, so only accuracy and counts within noise are required.
    mpi = defaultdict(dict)
    for r in runs:
        if r["problem"] == "rosenbrock" and r["N_0"] == 100_000 and r["seed"] == 0 and r["mode"] == "affine_cov_hess":
            mpi[r["rtol"]][r["mpi_size"]] = r
    for tol, by_p in sorted(mpi.items(), key=lambda kv: -(kv[0] or 0)):
        base = by_p.get(1)
        for p, r in sorted(by_p.items()):
            add(f"MPI rosenbrock d=5 tol {tol} P={p}", "A6 e_in <= 1.1 tol", r["e_in"], 1.1 * tol, r["e_in"] <= 1.1 * tol)
            if base is not None and p != 1:
                ratio = r["N_c"] / base["N_c"]
                add(f"MPI rosenbrock d=5 tol {tol} P={p}", "A6b N_c(P) / N_c(1) in [0.8, 1.25]", ratio, 1.25, 0.8 <= ratio <= 1.25)
    return out


def plots(matched: dict, runs: list[dict], iterative: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "font.size": 9})  # fmt: skip
    groups = defaultdict(list)
    for rec, hits, _ in matched.values():
        if rec["group"] == "conv" and "N" in rec:
            groups[(rec["problem"], rec["dim"])].append((rec, hits))
    keys = sorted(groups)
    if keys:
        for name, xq, yq, xlabel, ylabel, ref_line in (
            ("convergence_e_in_vs_Nc.png", "N_c", "e_in", "leaf clusters N_c", "e_in (relative L2, inside)", "slope"),
            ("e_in_vs_tol.png", "tol", "e_in", "tolerance", "e_in (relative L2, inside)", "identity"),
            ("evaluations_vs_tol.png", "tol", "N", "tolerance", "oracle evaluations N", None),
        ):
            cols = min(4, len(keys))
            rows_ = math.ceil(len(keys) / cols)
            fig, axes = plt.subplots(rows_, cols, figsize=(3.2 * cols, 2.8 * rows_), squeeze=False)
            for ax, key in zip(axes.flat, keys):
                items = groups[key]
                px = [r["tol"] if xq == "tol" else r[xq] for r, _ in items]
                py = [r[yq] for r, _ in items]
                ax.plot(px, py, "o--", color=BLUE, mfc="none", ms=6, lw=1.5, label="paper")
                for s_ in sorted({h["seed"] for _, hs in items for h in hs}):
                    pts = [((r["tol"] if xq == "tol" else h[xq]), h[yq]) for r, hs in items for h in hs if h["seed"] == s_]
                    if pts:
                        pts.sort()
                        ax.plot(*zip(*pts), "o-" if s_ == 0 else "o:", color=ORANGE, ms=5 if s_ == 0 else 3,
                                lw=1.5 if s_ == 0 else 0.8, label="re-run" if s_ == 0 else None)  # fmt: skip
                if ref_line == "slope" and px:
                    x0, y0 = px[0], py[0]
                    xs = np.array([min(px), max(px)])
                    ax.plot(xs, y0 * (xs / x0) ** (-2.0 / key[1]), color=MUTED, lw=1, label=f"N_c^(-2/{key[1]})")
                if ref_line == "identity":
                    xs = np.array([min(px), max(px)])
                    ax.plot(xs, xs, color=MUTED, lw=1, label="e_in = tol")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"{key[0]} d={key[1]}", color=INK)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=7, frameon=False)
            for ax in list(axes.flat)[len(keys):]:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out / name, dpi=130)
            plt.close(fig)
    csvs = sorted(iterative.glob("active_learning_halo/halo_metrics_*_sampling_uniform.csv"))
    if csvs:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        colors = [BLUE, ORANGE, "#1baf7a", "#eda100"]
        for c, path in zip(colors, csvs):
            with path.open() as f:
                rows_ = list(csv.DictReader(f))
            ax.plot([int(r["Iteration"]) for r in rows_], [float(r["P_Out (%)"]) for r in rows_], "o-", color=c,
                    label=path.name.split("_f_")[1].split("_tol")[0])  # fmt: skip
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("p_out (%)")
        ax.set_title("Iterative refinement, uniform resampling (Figure 9)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "iterative_p_out.png", dpi=130)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    report = a.out / "report"
    report.mkdir(parents=True, exist_ok=True)
    runs = load_runs(a.out / "results")
    write_table(report / "runs.csv", runs)
    rows, matched = compare(runs)
    write_table(report / "comparison.csv", rows)
    chk = checks(matched, runs)
    write_table(report / "checks.csv", chk)
    slope_rows = [c for c in chk if c["check"].startswith("A4 ")]
    write_table(report / "slopes.csv", slope_rows)
    cost = [json.loads(p.read_text()) for p in sorted((a.out / "results").glob("seed_*/*.peak.json"))]
    write_table(report / "cost.csv", [{k: c[k] for k in ("case", "seed", "wall_s", "peak_rss_gib_rank0")} for c in cost])
    plots(matched, runs, a.out / "results" / "iterative", report)
    (report / "paper_notes.txt").write_text("\n".join(f"- {n}" for n in PAPER_NOTES) + "\n")
    n = defaultdict(int)
    for c in chk:
        n[c["result"]] += 1
    m = defaultdict(int)
    for r in rows:
        m[r["verdict"]] += 1
    print(f"{len(runs)} runs; numbers: {dict(m)}; checks: {dict(n)}; report in {report}")
    for c in chk:
        if c["result"] == "FAIL":
            print(f"FAIL {c['ref']}: {c['check']} (value {c['value']:.4g}, limit {c['limit']:.4g}) {c['note']}")


if __name__ == "__main__":
    main()
