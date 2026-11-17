"""
The companion's appendix tables: every table of the AFINNI manuscript as it prints it, and the same
tables with the numbers of a recorded re-run (``studies/convergence/paper/record_results.py``).

    PYTHONPATH=. python docs/report/figures/manuscript_tables.py [--record <record> ...]

Writes ``figures/manuscript.json``, which the appendix reads. The manuscript's numbers come from
``studies/convergence/paper/paper_values.py``; a re-run number is a record's seed-0 value, a later
``--record`` replacing an earlier one's values where it has them (a supplementary record of a few
groups after the main one), and a cell no record has is "–". Each table names the records its
re-run values come from. Cells are written as formatted text, one format per column for both
sides, so the two versions of a table can be compared digit by digit.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "studies" / "convergence" / "paper"))
import paper_values as pv  # noqa: E402
from cases import lorenz_dims_sizes  # noqa: E402

OUT = Path(__file__).resolve().parent / "manuscript.json"
MISSING = "–"
NAMES = {pv.QUAD: "Quadratic", pv.GAUSS: "Gaussian", pv.ROSEN: "Rosenbrock", pv.LORENZ: "Lorenz96", pv.WAVE: "Wave"}
# Table 1's group headers give each problem's N_0, as the manuscript prints it.
TABLE1_N0 = {(pv.QUAD, 5): 1.86e5, (pv.QUAD, 10): 6.89e5, (pv.GAUSS, 2): 8.45e4, (pv.GAUSS, 5): 1.86e5,
             (pv.ROSEN, 2): 8.45e4, (pv.ROSEN, 5): 1.86e5, (pv.LORENZ, 5): 1.86e5, (pv.WAVE, 2): 8.45e4,
             (pv.WAVE, 5): 1.86e5}  # fmt: skip


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


def tol(x: float) -> str:
    return f"{x:g}" if x >= 0.01 else sci(x, 1)


def fixed(x: float | None, places: int) -> str:
    return MISSING if x is None else f"{x:,.{places}f}"


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


def tables(values: dict, walls: dict) -> list[dict]:
    """Each table with its manuscript rows, its re-run rows (the same layout) and the records behind them."""
    used = set()

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

    out = []

    def add(entry: dict) -> None:
        out.append({**entry, "sources": sorted(used)})
        used.clear()
    for part, rows in (("a", [r for r in pv._CONVERGENCE if r[0] <= 19]), ("b", [r for r in pv._CONVERGENCE if r[0] > 19])):
        name = f"Table 1{part}"
        paper, rerun, last = [], [], None
        for row, problem, d, rtol, N, Nc, po, e_in, e, e_nh in rows:
            if (problem, d) != last:
                head = f"*{NAMES[problem]}* ($d = {d}$, $N_0 = $ {sci(TABLE1_N0[(problem, d)])})"
                paper.append({"group": head})
                rerun.append({"group": head})
                last = (problem, d)
            paper.append({"cells": [str(row), tol(rtol), sci(N), count(Nc), pct(po), sci(e_in), sci(e), sci(e_nh)]})
            g = lambda q, row=row, name=name: got(name, str(row), q)  # noqa: E731
            rerun.append({"cells": [str(row), tol(rtol), sci(g("N")), count(g("N_c")), pct(g("p_o")), sci(g("e_in")),
                                    sci(g("e")), sci(g("e_nh"))]})  # fmt: skip
        caption = ("Convergence results for the Quadratic and Gaussian functions." if part == "a"
                   else "Convergence results for the Rosenbrock, Lorenz96, and Wave functions.")  # fmt: skip
        add({"key": f"table-1{part}", "title": name, "label": "tab:combined_convergence_" + part.upper(),
                    "caption": caption, "header": ["\\#", '$r_"tol"$', "$N$", "$N_c$", "$p_o$", '$e_"in"$', "$e$", '$e_"nh"$'],
                    "paper": paper, "rerun": rerun})  # fmt: skip

    paper = [{"cells": [str(row), str(d), sci(N), count(Nc), pct(po, "%"), sci(e_in), sci(e)]} for row, d, N, Nc, po, e_in, e in pv._LORENZ_DIMS]
    rerun = [{"cells": [str(row), str(d)] + [f(got("Table 2", str(row), q)) for q, f in
              (("N", sci), ("N_c", count), ("p_o", lambda x: pct(x, "%")), ("e_in", sci), ("e", sci))]}
             for row, d, *_ in pv._LORENZ_DIMS]  # fmt: skip
    add({"key": "table-2", "title": "Table 2", "label": "tab:lorenz_dim_scaling",
                "caption": "Lorenz96 dimension scaling at fixed $r_\"tol\" = 0.1$. Parameters are scaled as "
                           "$N_\"init\" = min(5 dot 10^4 dot 1.3^d, 10^6)$, $K_\"init\" = min(25 dot 1.3^d, 1000)$, and "
                           "$M = min(100 dot 1.3^d, 500)$ to maintain domain coverage while respecting memory limits.",
                "note": "The re-run sizes each run as the code that produced this table did "
                        "(`cap_initial_augmentation_budget` at `91ddd97`), not as the caption states: "
                        + "; ".join(f"$d = {d}$: $K_0 = $ {k0:,}, $M = $ {m:,}" for d, _, k0, m in
                                    ((d, *lorenz_dims_sizes(d)) for _, d, *_ in pv._LORENZ_DIMS))
                        + ". The caption's $K_0 <= 1000$, $M <= 500$ do not reproduce the table "
                          "(`CONVERGENCE_STUDY.md`, Table 2).",
                "header": ["\\#", "$d$", "$N$", "$N_c$", "$p_o$", '$e_"in"$', "$e$"], "paper": paper, "rerun": rerun})  # fmt: skip

    paper = [{"cells": [f"{a:.1f}", grouped(n0), grouped(Nc), pct(po, "%"), sci(e_in), sci(e), pct(good, "%")]}
             for a, n0, Nc, po, e_in, e, good in pv._ROSEN_WEAK]  # fmt: skip
    rerun = [{"cells": [f"{a:.1f}", grouped(n0)] + [f(got("Table 3", str(i), q)) for q, f in
              (("N_c", grouped), ("p_o", lambda x: pct(x, "%")), ("e_in", sci), ("e", sci), ("p_in_good", lambda x: pct(x, "%")))]}
             for i, (a, n0, *_) in enumerate(pv._ROSEN_WEAK, 1)]  # fmt: skip
    add({"key": "table-3", "title": "Table 3", "label": "tab:rosenbrock_weak_scaling",
                "caption": "Rosenbrock scaling in five dimensions, scaling both $r_\"tol\"$ and $N_0$ at the rate "
                           "$N_0 prop epsilon^(-5/2)$. The errors are similar to rows 25-29 in Table 1b, but $N_c$ "
                           "increases at a faster rate, while $p_o$ scales at a much slower rate.",
                "header": ["atol", "$N_0$", "$N_c$", '$p_"out"$', '$e_"in"$', "$e$", '$p_"in,good"$'],
                "paper": paper, "rerun": rerun})  # fmt: skip

    paper, rerun = [], []
    for rtol, lsq, exact in pv._WAVE_EXACT:
        paper.append({"cells": [f"{rtol:g}", grouped(lsq[0]), pct(lsq[1]), sci(lsq[2]), sci(lsq[3]),
                                grouped(exact[0]), pct(exact[1]), sci(exact[2]), sci(exact[3])]})  # fmt: skip
        cells = [f"{rtol:g}"]
        for kind in ("LSQ", "exact"):
            t = f"Figure 8 table, {kind} gradient, rtol {rtol}"
            cells += [grouped(got(t, "", "N_c")), pct(got(t, "", "p_o")), sci(got(t, "", "e_in")), sci(got(t, "", "e"))]
        rerun.append({"cells": cells})
    add({"key": "figure-8", "title": "Figure 8 (table)", "label": "fig:exact_grad_vs_approx_grad_comparison",
                "caption": "Wave in five dimensions, with the gradient estimated by least squares (LSQ) and with the "
                           "exact gradient.",
                "groups": [["", 1], ["LSQ gradient", 4], ["exact gradient", 4]],
                "header": ['$r_"tol"$', "$N_c$", "$p_o$", '$e_"in"$', "$e$", "$N_c$", "$p_o$", '$e_"in"$', "$e$"],
                "paper": paper, "rerun": rerun})  # fmt: skip

    paper, rerun, last = [], [], None
    for row, problem, d, method, Nc, e, inf in pv._OTHER_SURROGATES:
        if (problem, d) != last:
            head = f"*{NAMES[problem]}* ($d = {d}$)"
            paper.append({"group": head})
            rerun.append({"group": head})
            last = (problem, d)
        paper.append({"cells": [str(row), method, sci(Nc, 2), sci(e, 2), sci(inf, 1)]})
        t, r = "Table 4", f"{row} (AFINNI)"
        rerun.append({"cells": [str(row), method, sci(got(t, r, "N_c"), 2), sci(got(t, r, "e"), 2), MISSING]})
    add({"key": "table-4", "title": "Table 4", "label": "tab:other_surrogates",
                "caption": "Comparison of AFINNI with alternative methods on an inference task using 50,000 uniformly "
                           "sampled points (inference time in seconds). Only the AFINNI rows are re-run: the other "
                           "methods need libraries this repository does not ship.",
                "header": ["\\#", "Method", "$N_c$", "$e$", "Inf."], "paper": paper, "rerun": rerun})  # fmt: skip

    paper = [{"cells": [f"{lvl:02d}", grouped(nc), f"{pts:.2f}", f"{t:.2f}", f"{dev} / {host}"], "strong": lvl == pv.RUN_PROGRESSION_PEAK_LEVEL}
             for lvl, nc, pts, t, dev, host in pv._RUN_PROGRESSION]  # fmt: skip
    rerun = [{"cells": [f"{lvl:02d}"] + [MISSING] * 4} for lvl, *_ in pv._RUN_PROGRESSION]
    add({"key": "table-5", "title": "Table 5", "label": "tab:run_progression",
                "caption": "Run progression for the Rosenbrock function ($d = 10$): clusters, points ($10^6$), time "
                           "(s) and device / host memory (GB) per level; level 11 is the peak. This run is not in "
                           "the re-run matrix.",
                "header": ["Lvl", "Clusters", "Points", "Time", "Memory"], "paper": paper, "rerun": rerun})  # fmt: skip

    paper = [{"cells": [str(p), sci(n0), fixed(st, 1), fixed(dy, 1), f"{sp:.2f}×"]} for p, n0, st, dy, sp in pv._MPI_SCALING]
    rerun = [{"cells": [str(p), sci(n0), MISSING, fixed(wall(f"mpi_np{p}_tol0.005"), 1), MISSING]}
             for p, n0, *_ in pv._MPI_SCALING]  # fmt: skip
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
    data = {"records": abouts, "tables": tables(values, walls), "notes": pv.PAPER_NOTES}
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(abouts)} records, {len(values)} values)")


if __name__ == "__main__":
    main()
