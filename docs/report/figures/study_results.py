"""
Appendix A's results: the manuscript study's convergence, dimension and absolute-tolerance tables,
one row per method (AFINNI, ART with affine leaves, ART with quadratic leaves) for each setting,
the same experiments for AFINNI alone with its candidate search beside its exact search, and three
figures, all from the study's collected data (``studies/manuscript/collect.py``).

    PYTHONPATH=. python docs/report/figures/study_results.py [--data studies/manuscript/output/data]

Writes ``figures/study.json``, which the appendix reads, and the panels ``figures/study-*.svg``.
Every row the study specifies (``studies/manuscript/spec.py``) appears, so a table read while the
study runs shows what is missing: a row without a finished seed gives the reason instead of numbers
(not finished yet, out of memory or time, or skipped by the runner's projection). A cell is the
median over the finished seeds, and the last column gives their number. The formats are the sanity
check's (``manuscript_tables.py``), so the two appendices print a number the same way. The figures
are drawn at the text width of the companion, 16.2 cm, so their type prints at its own size.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from manuscript_tables import MISSING, count, grouped, percent, sci, tol  # noqa: E402

from studies.manuscript import spec  # noqa: E402
from studies.manuscript.tables import TABLES, matches, number  # noqa: E402

N_TEST = spec.N_TEST

OUT = Path(__file__).resolve().parent / "study.json"
METHODS = {"afinni": "AFINNI", "art_affine": "ART affine", "art_quadratic": "ART quadratic"}
EXHAUSTED_MEMORY = {"host_memory_guard", "device_out_of_memory", "killed"}


def duration(x: float | None) -> str:
    """Seconds or microseconds: three significant digits below 100, whole units with separators above."""
    if x is None:
        return MISSING
    return f"{x:.3g}" if x < 100 else f"{round(x):,}"


def share(x: float | None) -> str:
    """p_in,good with two decimals, so ART's shares near 100% stay apart."""
    return MISSING if x is None else f"{x:.2f}%"


# The leading cells of a row (its setting) and the method column, by experiment.
TOL = ('$"tol"$', lambda r: tol(r["tol"]))
METHOD = ("Method", lambda r: METHODS[r["method"]])
DIM = ("$d$", lambda r: str(r["d"]))
ATOL = [('$a_"tol"$', lambda r: f'{r["atol"]:.1f}'), ("$N_0$", lambda r: grouped(r["n0"]))]
GROUP = lambda r: f'*{spec.FUNCTIONS[r["function"]].name}* ($d = {r["d"]}$)'  # noqa: E731
# the spec's order of functions and dimensions, tolerances loosest first
CONVERGENCE_ORDER = lambda k: (list(spec.FUNCTIONS).index(k[0]), k[1], -k[2], list(METHODS).index(k[3]))  # noqa: E731
T_TRAIN = ('$t_"train"$ (s)', "train_s", duration)
T_INF = ('$t_"inf"$ (µs)', "infer_us", duration)
TIMES = ('$t_"train"$ is the wall time of training, and $t_"inf"$ is the inference time per test point, '
         "measured on one call with $10^5$ test points.")
# Each table: its short title in the list of tables, the TABLES entry it reads, its methods, its setting cells, its columns (header,
# tables.json column, format). A column with "pair" shows the candidate search's value and the
# exact search's (its tables.json name plus "_exact_search") under one header.
LAYOUT = {
    "convergence": {
        "short": "Convergence: AFINNI and ART",
        "table": "convergence", "methods": tuple(METHODS), "keys": [TOL, METHOD], "group": GROUP, "order": CONVERGENCE_ORDER,
        "columns": [("$N$", "N", sci), ("$N_c$", "N_c", count), ("$p_o$", "p_o", percent), ('$e_"in"$', "e_in", sci),
                    ("$e$", "e", sci), ('$e_"nh"$', "e_nh", sci), T_TRAIN, T_INF],
        "caption": "Convergence of AFINNI and ART for each function, dimension and tolerance. AFINNI's "
                   '$p_o$, $e_"in"$, $e$ and $e_"nh"$ come from its candidate search, and $e_"nh"$ is AFINNI\'s $e$ '
                   "without Hessian-informed splits. " + TIMES,
    },
    "dimension": {
        "short": "Dimension: AFINNI and ART on Lorenz-96",
        "table": "dimension", "methods": tuple(METHODS), "keys": [DIM, METHOD], "group": None,
        "order": lambda k: (k[0], list(METHODS).index(k[1])),
        "columns": [("$N$", "N", sci), ("$N_c$", "N_c", count), ("$p_o$", "p_o", percent), ('$e_"in"$', "e_in", sci),
                    ("$e$", "e", sci), T_TRAIN, T_INF],
        "caption": 'Lorenz-96 at $"tol" = 0.1$ in each dimension. AFINNI\'s values come from its candidate search. ' + TIMES,
    },
    "absolute": {
        "short": "Absolute tolerance: AFINNI and ART",
        "table": "absolute", "methods": tuple(METHODS), "keys": [*ATOL, METHOD], "group": None,
        "order": lambda k: (-k[0], list(METHODS).index(k[2])),
        "columns": [("$N_c$", "N_c", count), ("$p_o$", "p_o", percent), ('$e_"in"$', "e_in", sci), ("$e$", "e", sci),
                    ('$p_"in,good"$', "p_in_good", share), T_TRAIN, T_INF],
        "caption": 'Rosenbrock at $d = 5$ with an absolute tolerance, and $N_0 prop a_"tol"^(-5\\/2)$. AFINNI\'s values '
                   "come from its candidate search. " + TIMES,
    },
    "convergence-search": {
        "short": "Convergence: AFINNI's two searches",
        "table": "convergence", "methods": ("afinni",), "keys": [TOL], "group": GROUP, "order": CONVERGENCE_ORDER,
        "columns": [("$p_o$", "p_o", percent, "pair"), ('$e_"in"$', "e_in", sci, "pair"), ("$e$", "e", sci, "pair"),
                    ('$t_"inf"$ (µs)', "infer_us", duration, "pair")],
        "caption": "AFINNI's convergence runs, predicted by the candidate search (cand.) and by the same search "
                   "completed by the exact search (exact). Both use the same trained models.",
    },
    "dimension-search": {
        "short": "Dimension: AFINNI's two searches",
        "table": "dimension", "methods": ("afinni",), "keys": [DIM], "group": None, "order": lambda k: (k[0],),
        "columns": [("$p_o$", "p_o", percent, "pair"), ('$e_"in"$', "e_in", sci, "pair"), ("$e$", "e", sci, "pair"),
                    ('$t_"inf"$ (µs)', "infer_us", duration, "pair")],
        "caption": 'AFINNI on Lorenz-96 at $"tol" = 0.1$, with the candidate search and with the exact search.',
    },
    "absolute-search": {
        "short": "Absolute tolerance: AFINNI's two searches",
        "table": "absolute", "methods": ("afinni",), "keys": ATOL, "group": None, "order": lambda k: (-k[0],),
        "columns": [("$p_o$", "p_o", percent, "pair"), ('$e_"in"$', "e_in", sci, "pair"), ("$e$", "e", sci, "pair"),
                    ('$p_"in,good"$', "p_in_good", share, "pair"), ('$t_"inf"$ (µs)', "infer_us", duration, "pair")],
        "caption": "AFINNI on Rosenbrock at $d = 5$ with an absolute tolerance, with the candidate search and with "
                   "the exact search.",
    },
}


def why_missing(records: list[dict | None]) -> str:
    """
    Why a row has no finished seed, from its seeds' records (None: no record yet). A seed that ran out
    of memory or time, or was skipped, decides it even while other seeds have no record: the runner
    skips those for the same reason when it reaches them (``run.blocked``, ``run.project``).
    """
    found = [r for r in records if r is not None]
    failures = {r.get("failure") for r in found if r["status"] == "failed"}
    if failures & EXHAUSTED_MEMORY:
        return "out of memory"
    if "timeout" in failures:
        return "over the time limit"
    reasons = " ".join(str(r.get("reason") or "") for r in found if r["status"] == "skipped")
    if "ended with" in reasons:
        return "not run: a looser tolerance ran out of memory or time"
    if "GiB" in reasons:
        return "not run: projected over the memory limit"
    if "projected" in reasons:
        return "not run: projected over the time limit"
    return "not finished yet" if len(found) < len(records) else "failed"


def build(data: Path) -> dict:
    with (data / "cases.csv").open(newline="") as fh:
        by_key = {r["key"]: r for r in csv.DictReader(fh)}
    built = json.loads((data / "tables" / "tables.json").read_text())
    study = [c for c in spec.cases() if set(c.experiments) & {lay["table"] for lay in LAYOUT.values()}]
    out = []
    for name, layout in LAYOUT.items():
        by = TABLES[layout["table"]]["by"]
        where = TABLES[layout["table"]]["where"]
        seeds: dict[tuple, list] = defaultdict(list)  # a row's seeds' records, in the study's order
        for c in study:
            row = {**c.as_dict(), "experiments": " ".join(c.experiments)}
            if matches(row, where) and row["method"] in layout["methods"]:
                seeds[tuple(row[k] for k in by)].append(by_key.get(c.key))
        values = {tuple(number(r[k]) if number(r[k]) is not None else r[k] for k in by): r for r in built[layout["table"]]}
        columns = [(col if len(col) == 4 else (*col, None)) for col in layout["columns"]]
        # (tables.json column, format) per value cell: a pair gives the candidate search's, then the exact search's
        cells = [(c, f) for _, c, f, pair in columns for c in ((c, c + "_exact_search") if pair else (c,))]
        with_method = layout["keys"][-1] is METHOD
        rows, last_group, last_setting = [], None, None
        for k in sorted(seeds, key=layout["order"]):
            r = dict(zip(by, k, strict=True))
            if layout["group"] and layout["group"](r) != last_group:
                last_group, last_setting = layout["group"](r), None
                rows.append({"group": last_group})
            keys = [f(r) for _, f in layout["keys"]]
            setting = keys[:-1] if with_method else keys  # every leading cell but the method
            shown = [""] * len(setting) if setting == last_setting else setting
            last_setting = setting
            got = values.get(tuple(number(x) if number(x) is not None else x for x in k))
            done = int(got[cells[0][0] + ".seeds"]) if got else 0
            total = len(seeds[k])
            # a rule between settings; a table with one row per setting needs none
            entry = {"cells": shown + keys[len(setting):], "first": with_method and shown != [""] * len(setting)}
            if done == 0:
                entry["status"] = why_missing(seeds[k])
            else:
                entry["cells"] += [f(got[c]) if got.get(c) is not None else MISSING for c, f in cells]
                entry["cells"].append(str(done) if done == total else f"{done} of {total}")
            rows.append(entry)
        lead = [h for h, _ in layout["keys"]]
        if any(pair for *_, pair in columns):  # two header rows: the quantities, then cand. and exact under each pair
            top = [{"text": h, "rowspan": 2} for h in lead]
            top += [{"text": h, "colspan": 2} if pair else {"text": h, "rowspan": 2} for h, _, _, pair in columns]
            top.append({"text": "Seeds", "rowspan": 2})
            header = [h for *_, pair in columns if pair for h in ("cand.", "exact")]
        else:
            top, header = [], lead + [h for h, *_ in columns] + ["Seeds"]
        out.append({"key": name, "short": layout["short"], "caption": layout["caption"], "top": top, "header": header,
                    "columns": len(lead) + len(cells) + 1, "keys": len(lead), "rows": rows})  # fmt: skip
    states = Counter((by_key[c.key]["status"] if c.key in by_key else "pending") for c in study)
    provenance = json.loads((data / "provenance.json").read_text())
    return {"progress": {"cases": len(study), **{s: states.get(s, 0) for s in ("done", "failed", "skipped", "pending")}},
            "commits": sorted({(p.get("commit") or "")[:7] for p in provenance}),
            "gpus": sorted({(p.get("gpu") or {}).get("name") or "CPU" for p in provenance}), "tables": out}  # fmt: skip


# ----------------------------------------------------------------------------- figures
FIGURES = Path(__file__).resolve().parent
TEXT_WIDTH = 16.2 / 2.54  # inches: the companion's A4 page less its 2.4 cm margins
COLOUR = {"afinni": "#1f5fa8", "art_affine": "#ef7d1a", "art_quadratic": "#2e7d32"}  # one colour per method
MARKER = {"afinni": "o", "art_affine": "s", "art_quadratic": "^"}
INK, MUTED, GRID, HIGHLIGHT = "#222222", "#6b6b6b", "#e6e6e6", "#fbf3e2"
EXACT = 1e-10  # an error below this is exact to rounding; a log axis cannot draw it beside the others
FOCUS = 4  # problems highlighted in the convergence figure: the largest gaps between AFINNI and ART
SEARCH_GAP = 2.0  # the search figure shows the settings where the candidate search flags this many times more
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": "#777777",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "svg.fonttype": "path", "svg.hashsalt": "afinni",  # a fixed salt and no date: identical SVGs on every run
}


def value(row: dict, key: str) -> float | None:
    v = number(row.get(key))
    return None if v is None or v != v else v


def curve(table: list[dict], x: str, y: str, **where) -> tuple[np.ndarray, ...]:
    """(x, median, min, max) of the rows matching ``where`` with a finished ``y``, sorted by x."""
    pts = sorted((value(r, x), value(r, y), value(r, y + ".min"), value(r, y + ".max")) for r in table
                 if all(str(r.get(k)) == str(v) or value(r, k) == v for k, v in where.items())
                 and value(r, y) is not None and value(r, x) is not None)  # fmt: skip
    return tuple(np.array(v) for v in zip(*pts, strict=True)) if pts else (np.array([]),) * 4


def method_lines(ax, table: list[dict], x: str, y: str, where: dict) -> list[str]:
    """Each method's median with a min-to-max band; returns the methods whose errors are all exact to rounding."""
    exact = []
    for m in METHODS:
        xs, med, lo, hi = curve(table, x, y, method=m, **where)
        if not len(xs):
            continue
        if y == "e":
            keep = med > EXACT
            if not keep.any():
                exact.append(METHODS[m])
            xs, med, lo, hi = xs[keep], med[keep], np.maximum(lo[keep], EXACT), hi[keep]
        if len(xs):
            ax.plot(xs, med, color=COLOUR[m], marker=MARKER[m], ms=3.4, lw=1.3, mec="white", mew=0.4, zorder=3)
            if len(xs) > 1:
                ax.fill_between(xs, lo, hi, color=COLOUR[m], alpha=0.15, lw=0, zorder=2)
    return exact


def log_ticks(ax) -> None:
    """Few readable ticks on each log axis, also on one spanning less than a decade."""
    for a, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        lo, hi = sorted(lim)
        if hi / lo < 12:
            a.set_major_locator(ticker.FixedLocator(np.geomspace(lo * 1.06, hi / 1.06, 3)))
            a.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2g}".replace("e+0", "e").replace("e-0", "e-")))
        else:
            a.set_major_locator(ticker.LogLocator(base=10, numticks=4))
        a.set_minor_formatter(ticker.NullFormatter())


def method_handles(methods=tuple(METHODS)) -> list:
    return [Line2D([], [], color=COLOUR[m], marker=MARKER[m], lw=1.4, ms=4.5, label=METHODS[m]) for m in methods]


def largest_gaps(conv: list[dict]) -> list[tuple[str, int]]:
    """The FOCUS (function, d) pairs with the largest ratio of evaluations or of error between AFINNI and either ART variant."""
    idx = {(r["function"], int(value(r, "d")), value(r, "tol"), r["method"]): r for r in conv}
    gap: dict[tuple, float] = defaultdict(float)
    for (f, d, t, m), r in idx.items():
        for art in ("art_affine", "art_quadratic"):
            o = idx.get((f, d, t, art)) if m == "afinni" else None
            for q in ("N", "e"):
                a, b = (value(r, q), value(o, q)) if o else (None, None)
                if a and b and a > EXACT and b > EXACT:
                    gap[(f, d)] = max(gap[(f, d)], abs(math.log10(b / a)))
    return sorted(gap, key=gap.get, reverse=True)[:FOCUS]


def draw_convergence(conv: list[dict], focus: list) -> None:
    """
    Error against evaluations, one panel per (function, d): rows are functions, columns dimensions. A
    method whose errors are all exact to rounding has no line, and a panel where every method is exact
    says so.
    """
    funcs = list(spec.FUNCTIONS)
    dims = sorted({d for f in funcs for d in spec.FUNCTIONS[f].dims if f != "lorenz96"})
    fig, axes = plt.subplots(len(funcs), len(dims), figsize=(TEXT_WIDTH, 1.62 * len(funcs) + 0.4), squeeze=False)
    for i, f in enumerate(funcs):
        for j, d in enumerate(dims):
            ax = axes[i, j]
            if not any(r["function"] == f and value(r, "d") == d for r in conv):
                ax.set_visible(False)
                continue
            exact = method_lines(ax, conv, "N", "e", {"function": f, "d": d})
            if (f, d) in focus:
                ax.set_facecolor(HIGHLIGHT)
            ax.set_title(f"{spec.FUNCTIONS[f].name}, $d = {d}$", pad=3)
            if len(exact) == len(METHODS):
                ax.text(0.5, 0.5, "every method exact", transform=ax.transAxes, ha="center", va="center", color=MUTED)
                ax.set_xticks([]), ax.set_yticks([])
                continue
            ax.set_xscale("log"), ax.set_yscale("log")
            log_ticks(ax)
            if j == 0 or (f == "lorenz96"):
                ax.set_ylabel("error $e$")
            ax.set_xlabel("evaluations $N$" if i == len(funcs) - 1 or f == "lorenz96" or i == len(funcs) - 2 else "")
    # Lorenz-96 runs at d = 5 only: centre its panel under the d = 5 column
    fig.legend(handles=method_handles(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.32 / fig.get_figheight()), h_pad=0.9, w_pad=1.0)
    fig.savefig(FIGURES / "study-convergence.svg", bbox_inches="tight", pad_inches=0.02, metadata={"Date": None})
    plt.close(fig)


def draw_dimension(dim: list[dict]) -> None:
    """Lorenz-96 at tol 0.1: evaluations and leaves against the dimension."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.35))
    for ax, (q, label) in zip(axes, [("N", "evaluations $N$"), ("N_c", "leaves $N_c$")], strict=True):
        method_lines(ax, dim, "d", q, {})
        ax.set_xscale("log", base=2), ax.set_yscale("log")
        ds = spec.FUNCTIONS["lorenz96"].dims
        ax.set_xticks(ds, [str(d) for d in ds])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlabel("dimension $d$"), ax.set_ylabel(label)
    fig.legend(handles=method_handles(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()), w_pad=2.0)
    fig.savefig(FIGURES / "study-dimension.svg", bbox_inches="tight", pad_inches=0.02, metadata={"Date": None})
    plt.close(fig)


def search_gaps(conv: list[dict]) -> list[dict]:
    """
    AFINNI's convergence rows where the candidate search flags at least SEARCH_GAP times the exact
    search's share, grouped by problem in the spec's order, each problem's tolerances loosest first.
    """
    rows = [r for r in conv if r["method"] == "afinni" and value(r, "p_o") and value(r, "p_o_exact_search")
            and value(r, "p_o") >= SEARCH_GAP * value(r, "p_o_exact_search")]  # fmt: skip
    order = list(spec.FUNCTIONS)
    return sorted(rows, key=lambda r: (order.index(r["function"]), value(r, "d"), -value(r, "tol")))


def draw_search(rows: list[dict]) -> None:
    """For each setting of ``rows``: p_o and e under the exact search (filled) and the candidate search (open)."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.2 * len(rows) + 0.95), sharey=True)
    for i, r in enumerate(rows):
        for ax, q in zip(axes, ("p_o", "e"), strict=True):
            c, x = value(r, q), value(r, q + "_exact_search")
            ax.plot([x, c], [i, i], color="#b8c7da", lw=2.0, zorder=1, solid_capstyle="round")
            ax.scatter(x, i, color=COLOUR["afinni"], s=16, zorder=3)
            ax.scatter(c, i, facecolor="white", edgecolor=COLOUR["afinni"], s=16, lw=1.1, zorder=3)
    # a problem's name labels its first row only, and a rule separates the problems
    labels, last = [], None
    for i, r in enumerate(rows):
        problem = f"{spec.FUNCTIONS[r['function']].name}, $d = {int(value(r, 'd'))}$"
        labels.append(f"{problem},  tol {value(r, 'tol'):g}" if problem != last else f"tol {value(r, 'tol'):g}")
        if last is not None and problem != last:
            for ax in axes:
                ax.axhline(i - 0.5, color="#9a9a9a", lw=0.6, zorder=0)
        last = problem
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    for ax, label in zip(axes, ("$p_o$ (%)", "error $e$"), strict=True):
        ax.set_xscale("log"), ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
    fig.legend(handles=[Line2D([], [], ls="", marker="o", color=COLOUR["afinni"], label="exact search"),
                        Line2D([], [], ls="", marker="o", mfc="white", mec=COLOUR["afinni"], label="candidate search")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()), w_pad=1.5)
    fig.savefig(FIGURES / "study-search.svg", bbox_inches="tight", pad_inches=0.02, metadata={"Date": None})
    plt.close(fig)


def gaussian_bump(samples: int = 2_000_000, seed: int = 0) -> dict[int, np.ndarray]:
    """
    f - 1 of the study's Gaussian at uniform points of its domain, for each dimension it runs at:
    the height of its bump, exactly 0 wherever exp(...) falls below half an ulp of 1.
    """
    import os

    os.environ.setdefault("SURROGATE_DEVICE", "cpu")
    os.environ.setdefault("SURROGATE_MPI", "off")
    from studies.manuscript.case import functions

    f, _ = functions("gaussian")
    lo, hi = spec.FUNCTIONS["gaussian"].domain
    rng = np.random.default_rng(seed)
    return {d: np.asarray(f(rng.uniform(lo, hi, size=(samples, d)))).reshape(-1) - 1.0
            for d in spec.FUNCTIONS["gaussian"].dims}  # fmt: skip


def draw_gaussian(bump: dict[int, np.ndarray], tol: float) -> None:
    """
    Left: the share of the domain where f - 1 exceeds a threshold, against the threshold, one line
    per dimension. Right: the expected number of test points where f - 1 exceeds ``tol``.
    """
    shades = dict(zip(sorted(bump), ("#9ec1e6", "#4f8fd0", "#163f73"), strict=False))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.4), gridspec_kw={"width_ratios": [1.6, 1]})
    thresholds = np.logspace(-16, 0, 161)
    for d, g in sorted(bump.items()):
        share = np.array([np.mean(g > t) for t in thresholds]) * 100
        if share.any():
            a1.plot(thresholds[share > 0], share[share > 0], color=shades[d], lw=1.6, label=f"$d = {d}$")
        else:  # f = 1 at every sampled point: no line to draw on a log axis
            a1.text(0.03, 0.06, f"$d = {d}$: $f = 1$ at every sampled point", transform=a1.transAxes,
                    color=shades[d], fontsize=7.5)  # fmt: skip
    a1.axvline(tol, color=MUTED, ls="--", lw=0.8)
    a1.text(tol * 0.6, 2e-4, f"tol {tol:g}", ha="right", va="bottom", color=MUTED, fontsize=7)
    a1.set_xscale("log"), a1.set_yscale("log")
    a1.set_xlim(1e-16, 1), a1.set_ylim(1e-4, 150)
    a1.set_xlabel("threshold $t$"), a1.set_ylabel("share of the domain with $f - 1 > t$ (%)")
    a1.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
    a1.xaxis.set_minor_formatter(ticker.NullFormatter())
    a1.legend(loc="center", bbox_to_anchor=(0.5, 0.66))
    ds = sorted(bump)
    counts = [np.mean(bump[d] > tol) * N_TEST for d in ds]
    a2.bar(range(len(ds)), [max(c, 0.3) for c in counts], color=[shades[d] for d in ds], width=0.6)
    for k, c in enumerate(counts):
        a2.text(k, max(c, 0.3) * 1.3, f"{c:,.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    a2.set_yscale("log"), a2.set_ylim(0.2, N_TEST * 3)
    a2.set_xticks(range(len(ds)), [f"$d = {d}$" for d in ds])
    a2.set_ylabel(f"test points with $f - 1 >$ {tol:g}")
    a2.yaxis.set_minor_formatter(ticker.NullFormatter())
    a2.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGURES / "study-gaussian.svg", bbox_inches="tight", pad_inches=0.02, metadata={"Date": None})
    plt.close(fig)

def figures(built: dict) -> dict:
    """Draw the four panels and return the numbers their captions and text quote."""
    plt.rcParams.update(STYLE)
    conv, dim = built["convergence"], built["dimension"]
    focus = largest_gaps(conv)
    draw_convergence(conv, focus)
    draw_dimension(dim)
    gaps = search_gaps(conv)
    draw_search(gaps)
    tol = min(spec.TOLERANCES)
    bump = gaussian_bump()
    draw_gaussian(bump, tol)
    gaussian = {"tol": f"{tol:g}", "dims": [
        {"d": d, "share": f"{np.mean(g > tol) * 100:.2g}%", "exact": f"{np.mean(g == 0) * 100:.3g}%",
         "test": f"{np.mean(g > tol) * N_TEST:,.0f}", "train": f"{np.mean(g > tol) * spec.sizes(d)[0]:,.0f}"}
        for d, g in sorted(bump.items())]}  # fmt: skip
    afinni = {int(value(r, "d")): value(r, "N") for r in dim if r["method"] == "afinni" and value(r, "N")}
    ds = sorted(afinni)
    return {"focus": [f"{spec.FUNCTIONS[f].name} at $d = {d}$" for f, d in focus],
            "search_gap": f"{SEARCH_GAP:g}", "search_settings": len(gaps),
            "search_total": sum(1 for r in conv if r["method"] == "afinni" and value(r, "p_o") is not None),
            "afinni_growth": f"{afinni[ds[-1]] / afinni[ds[0]]:,.0f}", "dims": [ds[0], ds[-1]], "gaussian": gaussian}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "studies" / "manuscript" / "output" / "data",
                        help="collect.py's output folder")  # fmt: skip
    a = parser.parse_args()
    data = build(a.data)
    data["figures"] = figures(json.loads((a.data / "tables" / "tables.json").read_text()))
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {data['progress']}")


if __name__ == "__main__":
    main()
