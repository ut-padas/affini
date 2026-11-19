"""
The companion's appendix tables: every table of the AFINNI manuscript as it prints it, and the same
tables with the numbers of recorded re-runs (``studies/sanity/paper/record_results.py``).

    PYTHONPATH=. python docs/reports/adaptive-surrogates/figures/manuscript_tables.py [--record <record> ...]

Writes ``figures/manuscript.json``, which the appendix reads. The manuscript's numbers come from
``studies/sanity/paper/paper_values.py``; a re-run number is a record's seed-0 value, a later
``--record`` replacing an earlier one's values where it has them (a supplementary record of a few
groups after the main one), and a cell no record has is "–". Each table names the records its
re-run values come from. Cells are written as formatted text, one format per column for both
sides, so the two versions of a table can be compared digit by digit.

A re-run number that differs from the manuscript's beyond the record's own noise band
(``collect_paper.verdict``: 25% for counts, errors and times, max(1 point, 30%) for percentages) is
marked better or worse, and each table lists those differences. Lower is better for every quantity
but ``p_in_good``.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "studies" / "sanity" / "paper"))
sys.path.insert(0, str(ROOT))
import paper_values as pv  # noqa: E402
from cases import group_coarse_factor, lorenz_dims_sizes  # noqa: E402
from collect_paper import verdict  # noqa: E402

OUT = Path(__file__).resolve().parent / "manuscript.json"
MISSING = "–"
NAMES = {pv.QUAD: "Quadratic", pv.GAUSS: "Gaussian", pv.ROSEN: "Rosenbrock", pv.LORENZ: "Lorenz96", pv.WAVE: "Wave"}
# Table 1's group headers give each problem's N_0, as the manuscript prints it.
TABLE1_N0 = {(pv.QUAD, 5): 1.86e5, (pv.QUAD, 10): 6.89e5, (pv.GAUSS, 2): 8.45e4, (pv.GAUSS, 5): 1.86e5,
             (pv.ROSEN, 2): 8.45e4, (pv.ROSEN, 5): 1.86e5, (pv.LORENZ, 5): 1.86e5, (pv.WAVE, 2): 8.45e4,
             (pv.WAVE, 5): 1.86e5}  # fmt: skip
# How the difference lists name a quantity (Typst markup).
SYMBOL = {"N": "$N$", "N_c": "$N_c$", "p_o": "$p_o$", "e_in": '$e_"in"$', "e": "$e$", "e_nh": '$e_"nh"$',
          "p_in_good": '$p_"in,good"$', "time": "the time"}  # fmt: skip


def sci(x: float | None, digits: int = 3) -> str:
    """The manuscript's E notation: 2.32E5, 7.43E-2, 3E-2."""
    if x is None:
        return MISSING
    mantissa, exponent = f"{x:.{digits - 1}E}".split("E")
    return f"{mantissa}E{int(exponent)}"


def count(x: float | None) -> str:
    """Counts as Table 1 prints them: with thousands separators below 1E5, in E notation above."""
    if x is None:
        return MISSING
    return f"{round(x):,}" if x < 1e5 else sci(x)


def grouped(x: float | None) -> str:
    return MISSING if x is None else f"{round(x):,}"


def pct(x: float | None, unit: str = "") -> str:
    """Percentages with two decimals below 10 and three significant digits above: 0.03, 6.15, 16.9, 20.0."""
    if x is None:
        return MISSING
    return (f"{x:.2f}" if x < 10 else f"{x:#.3g}") + unit


def percent(x: float | None) -> str:
    return pct(x, "%")


def tol(x: float) -> str:
    return f"{x:g}" if x >= 0.01 else sci(x, 1)


def fixed1(x: float | None) -> str:
    return MISSING if x is None else f"{x:,.1f}"


def compare(q: str, paper: float | None, rerun: float | None) -> str | None:
    """'better' or 'worse' when the re-run differs from the manuscript beyond the noise band, else None."""
    if paper is None or rerun is None:
        return None
    if verdict("p_o" if q.startswith("p_") else q, paper, [rerun], rerun)[0] != "differs":
        return None
    higher_is_better = q == "p_in_good"
    return "better" if (rerun > paper) == higher_is_better else "worse"


def read_records(records: list[Path]) -> tuple[dict, dict, list[dict]]:
    """
    Seed-0 values by (table, row, quantity) and case wall times, each as (value, index of its record),
    a later record replacing an earlier one's; and each record's provenance.
    """
    values, walls, abouts = {}, {}, []
    for i, record in enumerate(records):
        with (record / "manuscript_results.csv").open(newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    v = float(r["value"])
                except ValueError:
                    continue
                if v == v:  # NaN: the quantity was not measured (e.g. a skipped exact-outside pass)
                    values[(r["table"], r["row"], r["quantity"])] = (v, i)
        if (record / "cost.csv").exists():
            with (record / "cost.csv").open(newline="") as fh:
                walls.update({r["case"]: (float(r["wall_s"]), i) for r in csv.DictReader(fh) if r["seed"] == "0"})
        prov = json.loads((record / "provenance.json").read_text())
        abouts.append({"name": record.name, "commit": prov["commit"][:7], "recorded": prov["recorded"][:10],
                       "machine": prov["machine"]["label"], "gpu_runs": prov.get("gpu_runs", "")})  # fmt: skip
    return values, walls, abouts


# Why a table's re-run differs from the manuscript, by (table, quantity, better or worse): the lab's
# analysis of each difference the colouring finds, with the evidence's location. A difference with
# no entry here reads "not analysed", so a new record cannot silently inherit a stale explanation.
CAUSES = {
    ("table-1a", "N", "worse"): "\\#6, \\#7 (Quadratic $d = 10$): more evaluations at the two loosest "
        "tolerances for an equal or smaller $e$, and fewer at \\#8, \\#9: the series converges along a "
        "different path. Not traced to one change. \\#17, \\#19: the Gaussian note under $N_c$.",
    ("table-1a", "N_c", "worse"): "\\#6, \\#7: as for $N$. \\#15, \\#17, \\#19 (Gaussian $d = 5$): the "
        "function is not the manuscript's. Its rotation is drawn with NumPy from a fixed seed, while the "
        "draft's GPU runs drew it with CuPy, so the two tables describe two rotated Gaussians "
        "(`studies/sanity/README.md`, D4); its trees also vary strongly between seeds.",
    ("table-1a", "e", "worse"): "\\#15 (Gaussian $d = 5$): within the seed spread; over ten seeds the "
        "re-run's $e$ at this row brackets the manuscript's (GH200 seed study). $e$ is set by the few "
        "points flagged outside, so it varies more than $e_\"in\"$.",
    ("table-1a", "e_in", "worse"): "\\#16, \\#19 (Gaussian $d = 5$): $e_\"in\"$ lies orders of magnitude "
        "below the tolerance on both sides; the different instance and the seed noise move it. No loss of "
        "accuracy against the target.",
    ("table-1a", "N_c", "better"): "Smaller trees for an equal or smaller error (Quadratic $d = 5$ and "
        "\\#9). Not traced to one change: the training code changed in several places since the draft, "
        "among them the initial clustering (README, D5).",
    ("table-1a", "e_nh", "better"): "The runs without Hessian splits are more accurate than the draft's in "
        "every row (README, D3). No code state or setting tried reproduces the draft's $e_\"nh\"$, and its "
        "runs are not in the repository. It is why the check that Hessian splits are not worse (A2) fails "
        "at \\#15 and \\#17.",
    ("table-1a", "N", "better"): "As for $N_c$: smaller trees cost fewer evaluations.",
    ("table-1a", "p_o", "better"): "Fewer points flagged outside, with the draft's own candidate counts "
        "(50 at $d = 5$, 200 at $d = 10$). Not traced: the trees and the trust regions' regularization "
        "changed since the draft (README, D4, D5).",
    ("table-1a", "e_in", "better"): "Not traced; it comes with the smaller trees above, and at \\#15, \\#17 "
        "with the Gaussian's different instance.",
    ("table-1a", "e", "better"): "Follows the smaller $p_o$: fewer points are predicted by a far leaf "
        "(README, D2). \\#18, \\#19: the Gaussian's instance and seed noise, as above.",
    ("table-1b", "e", "worse"): "\\#42 (Wave $d = 5$): $e$ is set by the points the candidate search flags "
        "outside, most of which lie inside some trust region (the exact search flags far fewer), while "
        "$e_\"in\"$ is better than the manuscript's. The search misses the leaf; the surrogate is not less "
        "accurate. Figure 8 shows the same.",
    ("table-1b", "p_o", "better"): "\\#20, \\#21 (Rosenbrock $d = 2$): the draft scored its $d = 2$ rows with "
        "10 candidates, the re-run with 50 (README, D1, D4). \\#25, \\#26 (Rosenbrock $d = 5$) and \\#40, "
        "\\#41 (Wave $d = 5$): not traced.",
    ("table-1b", "e_in", "better"): "Not traced; at \\#40-42 it comes with the smaller Wave trees (below).",
    ("table-1b", "e", "better"): "\\#21: the candidate count (README, D1, D2). \\#36: an artefact: the "
        "manuscript's $e$ there is almost certainly a typo (see the inconsistencies at the end). \\#40: "
        "follows the smaller $p_o$.",
    ("table-1b", "e_nh", "better"): "As in Table 1a: the runs without Hessian splits are more accurate than "
        "the draft's (README, D3).",
    ("table-1b", "N", "better"): "Wave $d = 5$: the re-run's trees are much smaller for an equal or smaller "
        "$e_\"in\"$, and Figure 8's exact-gradient runs show the same. Not traced: the draft-era code could not "
        "be installed to check it.",
    ("table-1b", "N_c", "better"): "As for $N$.",
    ("table-3", "p_o", "better"): "The draft's settings for this table are not in its script. The re-run "
        "predicts with 50 candidates, and $p_o$ falls as the candidate count grows (README, D1); most "
        "likely the draft used fewer.",
    ("table-3", "e", "better"): "Follows the smaller $p_o$ (README, D2).",
    ("figure-8", "e", "worse"): "LSQ 0.02 is Table 1b's \\#42 (see there). Exact 0.005: at the smallest "
        "tolerance 50 candidates still miss some containing leaves; with 200 the GH200 study measured an "
        "$e$ below the manuscript's.",
    ("figure-8", "N_c", "better"): "Both gradients give much smaller trees than the draft, the exact one "
        "most. Not traced: the draft-era code could not be installed to check it.",
    ("figure-8", "p_o", "better"): "LSQ: as Table 1b \\#40, \\#41. Exact gradient: the re-run predicts "
        "with 50 candidates, the draft's script with the launcher's 10 (the note above the table).",
    ("figure-8", "e_in", "better"): "Exact gradient: more candidates also give inside points a closer "
        "leaf (the GH200 study measured $e_\"in\"$ falling from 10 to 50 candidates on the same trees). LSQ: "
        "not traced.",
    ("figure-8", "e", "better"): "Follows the smaller $p_o$ (README, D2).",
    ("table-4", "e", "worse"): "\\#18 (Lorenz-96): Table 4's row is not Table 1b \\#31's run; it matches "
        "older rows the draft commented out (see the inconsistencies at the end). Against Table 1b \\#31 "
        "the re-run is within the noise band. Its values also come from a float64 integrator in place of "
        "the draft's float32 flow map.",
    ("table-4", "e", "better"): "\\#1 is the run of Table 1b \\#21 (see there).",
    ("table-4", "N_c", "better"): "\\#18: as for its $e$.",
}


def differences(found: list[tuple[str, str, float, float, str]], table: str = "") -> list[dict]:
    """
    The marked cells of one table, one line per (quantity, direction) in order of first appearance:
    the rows and the range of re-run / manuscript ratios, e.g. "$e$ 2.4-4.8x the manuscript's (#15, #17)".
    """
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for label, q, paper, rerun, mark in found:
        groups[(q, mark)].append((label, rerun / paper if paper else float("inf")))
    def ratio(r: float) -> str:
        return f"{r:.0f}" if r >= 10 else f"{r:.2g}"

    lines = []
    for (q, mark), items in sorted(groups.items(), key=lambda kv: kv[0][1] != "worse"):  # regressions first
        lo, hi = ratio(min(r for _, r in items)), ratio(max(r for _, r in items))
        span = f"{lo}×" if lo == hi else f"{lo}–{hi}×"
        rows = ", ".join(label for label, _ in items)
        lines.append({"mark": mark, "text": f"{SYMBOL[q]} {span} the manuscript's ({rows})",
                      "cause": CAUSES.get((table, q, mark), "Not analysed.")})  # fmt: skip
    return lines


def tables(values: dict, walls: dict) -> list[dict]:
    """
    Each table with its manuscript rows, its re-run rows (the same layout, each re-run cell marked
    better, worse or neither), its list of differences and the records behind its re-run side.
    """
    used, found = set(), []

    def got(table: str, row: str, q: str):
        value, source = values.get((table, row, q), (None, None))
        if source is not None:
            used.add(source)
        return value

    def wall(case: str):
        value, source = walls.get(case, (None, None))
        if source is not None:
            used.add(source)
        return value

    def row_pair(label: str, keys: list[str], cols: list[tuple], strong: bool = False) -> tuple[dict, dict]:
        """
        One row on both sides. ``keys`` are the leading cells shared by both (row number, setting);
        each column is (quantity or None, format, manuscript value, re-run value).
        """
        paper_cells, rerun_cells, marks = list(keys), list(keys), [None] * len(keys)
        for q, f, p, r in cols:
            paper_cells.append(f(p))
            rerun_cells.append(f(r))
            mark = compare(q, p, r) if q else None
            marks.append(mark)
            if mark:
                found.append((label, q, p, r, mark))
        paper_row = {"cells": paper_cells, **({"strong": True} if strong else {})}
        return paper_row, {"cells": rerun_cells, "marks": marks}

    out = []

    def add(entry: dict) -> None:
        out.append({**entry, "sources": sorted(used), "differences": differences(found, entry["key"])})
        used.clear()
        found.clear()

    def group(head: str, paper: list, rerun: list) -> None:
        paper.append({"group": head})
        rerun.append({"group": head})

    for part, rows in (("a", [r for r in pv._CONVERGENCE if r[0] <= 19]), ("b", [r for r in pv._CONVERGENCE if r[0] > 19])):
        name = f"Table 1{part}"
        paper, rerun, last = [], [], None
        for row, problem, d, rtol, N, Nc, po, e_in, e, e_nh in rows:
            if (problem, d) != last:
                group(f"*{NAMES[problem]}* ($d = {d}$, $N_0 = $ {sci(TABLE1_N0[(problem, d)])})", paper, rerun)
                last = (problem, d)
            g = lambda q, row=row, name=name: got(name, str(row), q)  # noqa: E731
            p, r = row_pair(f"\\#{row}", [str(row), tol(rtol)], [
                ("N", sci, N, g("N")), ("N_c", count, Nc, g("N_c")), ("p_o", pct, po, g("p_o")),
                ("e_in", sci, e_in, g("e_in")), ("e", sci, e, g("e")), ("e_nh", sci, e_nh, g("e_nh"))])  # fmt: skip
            paper.append(p)
            rerun.append(r)
        caption = ("Convergence results for the Quadratic and Gaussian functions." if part == "a"
                   else "Convergence results for the Rosenbrock, Lorenz96, and Wave functions.")  # fmt: skip
        add({"key": f"table-1{part}", "title": name, "label": "tab:combined_convergence_" + part.upper(),
             "caption": caption, "header": ["\\#", '$r_"tol"$', "$N$", "$N_c$", "$p_o$", '$e_"in"$', "$e$", '$e_"nh"$'],
             "paper": paper, "rerun": rerun})  # fmt: skip

    paper, rerun = [], []
    for row, d, N, Nc, po, e_in, e in pv._LORENZ_DIMS:
        g = lambda q, row=row: got("Table 2", str(row), q)  # noqa: E731
        p, r = row_pair(f"$d = {d}$", [str(row), str(d)], [
            ("N", sci, N, g("N")), ("N_c", count, Nc, g("N_c")), ("p_o", percent, po, g("p_o")),
            ("e_in", sci, e_in, g("e_in")), ("e", sci, e, g("e"))])  # fmt: skip
        paper.append(p)
        rerun.append(r)
    add({"key": "table-2", "title": "Table 2", "label": "tab:lorenz_dim_scaling",
         "note": "The re-run sizes each run as the code that produced this table did "
                 "(`cap_initial_augmentation_budget` at `91ddd97`), not as the caption states: "
                 + "; ".join(f"$d = {d}$: $K_0 = $ {k0:,}, $M = $ {m:,}" for d, _, k0, m in
                             ((d, *lorenz_dims_sizes(d)) for _, d, *_ in pv._LORENZ_DIMS))
                 + ". The caption's $K_0 <= 1000$, $M <= 500$ do not reproduce the table "
                   "(`studies/sanity/README.md`, Table 2).",
         "caption": "Lorenz96 dimension scaling at fixed $r_\"tol\" = 0.1$. Parameters are scaled as "
                    "$N_\"init\" = min(5 dot 10^4 dot 1.3^d, 10^6)$, $K_\"init\" = min(25 dot 1.3^d, 1000)$, and "
                    "$M = min(100 dot 1.3^d, 500)$ to maintain domain coverage while respecting memory limits.",
         "header": ["\\#", "$d$", "$N$", "$N_c$", "$p_o$", '$e_"in"$', "$e$"], "paper": paper, "rerun": rerun})  # fmt: skip

    paper, rerun = [], []
    for i, (a, n0, Nc, po, e_in, e, good) in enumerate(pv._ROSEN_WEAK, 1):
        g = lambda q, i=i: got("Table 3", str(i), q)  # noqa: E731
        p, r = row_pair(f"atol {a:.1f}", [f"{a:.1f}", grouped(n0)], [
            ("N_c", grouped, Nc, g("N_c")), ("p_o", percent, po, g("p_o")), ("e_in", sci, e_in, g("e_in")),
            ("e", sci, e, g("e")), ("p_in_good", percent, good, g("p_in_good"))])  # fmt: skip
        paper.append(p)
        rerun.append(r)
    add({"key": "table-3", "title": "Table 3", "label": "tab:rosenbrock_weak_scaling",
         "caption": "Rosenbrock scaling in five dimensions, scaling both $r_\"tol\"$ and $N_0$ at the rate "
                    "$N_0 prop epsilon^(-5/2)$. The errors are similar to rows 25-29 in Table 1b, but $N_c$ "
                    "increases at a faster rate, while $p_o$ scales at a much slower rate.",
         "header": ["atol", "$N_0$", "$N_c$", '$p_"out"$', '$e_"in"$', "$e$", '$p_"in,good"$'],
         "paper": paper, "rerun": rerun})  # fmt: skip

    paper, rerun = [], []
    for rtol, lsq, exact in pv._WAVE_EXACT:
        cols = []
        for kind, (Nc, po, e_in, e) in (("LSQ", lsq), ("exact", exact)):
            t = f"Figure 8 table, {kind} gradient, rtol {rtol}"
            cols += [("N_c", grouped, Nc, got(t, "", "N_c")), ("p_o", pct, po, got(t, "", "p_o")),
                     ("e_in", sci, e_in, got(t, "", "e_in")), ("e", sci, e, got(t, "", "e"))]  # fmt: skip
        p, r = row_pair("", [f"{rtol:g}"], cols)
        paper.append(p)
        rerun.append(r)
    # Name each marked cell by its gradient and tolerance: after r_tol, four LSQ columns, then four exact.
    marked = [(i, k) for i, r in enumerate(rerun) for k, m in enumerate(r["marks"]) if m]
    found[:] = [(f"{'LSQ' if k <= 4 else 'exact'} {pv._WAVE_EXACT[i][0]:g}", *f[1:])
                for (i, k), f in zip(marked, found, strict=True)]  # fmt: skip
    add({"key": "figure-8", "title": "Figure 8 (table)", "label": "fig:exact_grad_vs_approx_grad_comparison",
         "note": f"The re-run predicts the exact-gradient runs with `coarse_factor` = "
                 f"{group_coarse_factor('wave_exact', 5)} candidates, as its LSQ column (Table 1b's Wave $d = 5$ "
                 f"runs) is predicted; the manuscript's script used the launcher's 10. Training does not "
                 f"depend on it.",
         "caption": "Wave in five dimensions, with the gradient estimated by least squares (LSQ) and with the "
                    "exact gradient.",
         "groups": [["", 1], ["LSQ gradient", 4], ["exact gradient", 4]],
         "header": ['$r_"tol"$', "$N_c$", "$p_o$", '$e_"in"$', "$e$", "$N_c$", "$p_o$", '$e_"in"$', "$e$"],
         "paper": paper, "rerun": rerun})  # fmt: skip

    paper, rerun, last = [], [], None
    for row, problem, d, method, Nc, e, inf in pv._OTHER_SURROGATES:
        if (problem, d) != last:
            group(f"*{NAMES[problem]}* ($d = {d}$)", paper, rerun)
            last = (problem, d)
        mine = method == "AFINNI"
        t, r_ = "Table 4", f"{row} (AFINNI)"
        p, r = row_pair(f"\\#{row}", [str(row), method], [
            ("N_c", lambda x: sci(x, 2), Nc, got(t, r_, "N_c") if mine else None),
            ("e", lambda x: sci(x, 2), e, got(t, r_, "e") if mine else None),
            (None, lambda x: sci(x, 1), inf, None)])  # fmt: skip
        paper.append(p)
        rerun.append(r)
    add({"key": "table-4", "title": "Table 4", "label": "tab:other_surrogates",
         "caption": "Comparison of AFINNI with alternative methods on an inference task using 50,000 uniformly "
                    "sampled points (inference time in seconds). Only the AFINNI rows are re-run: the other "
                    "methods need libraries this repository does not ship.",
         "header": ["\\#", "Method", "$N_c$", "$e$", "Inf."], "paper": paper, "rerun": rerun})  # fmt: skip

    paper, rerun = [], []
    for lvl, nc, pts, t, dev, host in pv._RUN_PROGRESSION:
        p, r = row_pair("", [f"{lvl:02d}"], [
            (None, grouped, nc, None), (None, lambda x: MISSING if x is None else f"{x:.2f}", pts, None),
            (None, lambda x: MISSING if x is None else f"{x:.2f}", t, None),
            (None, lambda x: MISSING if x is None else f"{x[0]} / {x[1]}", (dev, host), None)],
            strong=lvl == pv.RUN_PROGRESSION_PEAK_LEVEL)  # fmt: skip
        paper.append(p)
        rerun.append(r)
    add({"key": "table-5", "title": "Table 5", "label": "tab:run_progression",
         "caption": "Run progression for the Rosenbrock function ($d = 10$): clusters, points ($10^6$), time "
                    "(s) and device / host memory (GB) per level; level 11 is the peak. This run is not in "
                    "the re-run matrix.",
         "header": ["Lvl", "Clusters", "Points", "Time", "Memory"], "paper": paper, "rerun": rerun})  # fmt: skip

    paper, rerun = [], []
    for P, n0, st, dy, sp in pv._MPI_SCALING:
        p, r = row_pair(f"$P = {P}$", [str(P), sci(n0)], [
            (None, fixed1, st, None), ("time", fixed1, dy, wall(f"mpi_np{P}_tol0.005")),
            (None, lambda x: MISSING if x is None else f"{x:.2f}×", sp, None)])  # fmt: skip
        paper.append(p)
        rerun.append(r)
    add({"key": "table-6", "title": "Table 6", "label": "tab:mpi_scaling_results",
         "caption": "Distributed performance scaling on the 5D Rosenbrock function at $r_\"tol\" = 5 dot 10^(-3)$: "
                    "time without (static) and with (dynamic) work stealing. Rows from $P = 256$ are the "
                    "manuscript's scale-out (weak scaling) part. In the re-run, the dynamic column is the "
                    "whole case's wall time (training and prediction, on CPU ranks); static runs are not in "
                    "the re-run matrix.",
         "header": ["Ranks $P$", "$N_0$", "Static (s)", "Dynamic (s)", "Speedup"], "paper": paper, "rerun": rerun})  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, action="append", default=[],
                        help="a record directory written by record_results.py; repeat it, the main record first")  # fmt: skip
    a = parser.parse_args()
    values, walls, abouts = read_records(a.record)
    # Appendix A's size table, from the manuscript study's own rule.
    from studies.manuscript.spec import sizes as study_sizes

    sizes = [dict(zip(("d", "N_0", "K_0", "M"), (d, *study_sizes(d)), strict=True)) for d in (2, 5, 10, 20, 40, 80)]
    data = {"records": abouts, "tables": tables(values, walls), "notes": pv.PAPER_NOTES, "sizes": sizes}
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(abouts)} records, {len(values)} values)")


if __name__ == "__main__":
    main()
