"""
The benchmark's summary table, from the store alone: per method and problem class, how many cases it ran,
held and won, at what cost, and whether its records describe the current code.

    PYTHONPATH=. .venv/bin/python -m src.benchmark table [--write README.md]

A case is (problem, d, tolerance). A method *holds* a case when its worst error over the tolerance is at most
1 on every recorded seed (failed runs do not hold), and *wins* it when it holds with the fewest mean
evaluations of every method that holds it. Counts depend on which cases each method ran (the sweeps stop a
method at its first failure on a family), so ``ran`` is shown beside them. Evaluations and inference time are
medians over the held cases.

``RESULTS_CHANGED`` dates the last change to each method's results (most commits change none, so this
cannot be read from the history); a method with records written before it is marked stale, with the share
of its records affected. Add an entry with every deliberate numerical change.

With ``--write FILE``, the table replaces the text between ``<!-- benchmark-table -->`` and
``<!-- /benchmark-table -->`` in FILE; without it, it is printed.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from src.benchmark.store import Store

# method -> (time of the last change to its results, what changed)
RESULTS_CHANGED = {
    "chebtt": ("2026-11-18T22:22", "piecewise axes at jumps and kinks, and the degree-cut compression"),
    "chebtt_metric": ("2026-11-18T22:22", "as ChebTT"),
    "chebtt_adaptive": ("2026-11-18T22:22", "as ChebTT"),
    "chebtt_box": ("2026-11-18T22:22", "as ChebTT"),
    "chebtt_smooth": ("2026-11-18T22:22", "as ChebTT"),
    "chebtt_roll": ("2026-11-18T22:22", "as ChebTT"),
    "chebtree": ("2026-11-17T23:54", "verification along centre fibres before a box is accepted"),
    "sparsecheb": ("2026-11-17T23:05", "the look-ahead after a failed check"),
    "kernel": ("2026-11-18T00:30", "first version"),
    "hub": ("2026-11-18T23:13", "contiguous rotations (predictions within 1e-9)"),
    "afinni": ("2026-11-17T18:06", "per-batch random streams"),
    "art": ("2026-11-16T17:55", "leaf fits as the reference ART's"),
    "art_quad": ("2026-11-16T17:55", "leaf fits as the reference ART's"),
}
FUNCTION_CLASSES = {
    "smooth": ("rosenbrock", "gaussian", "wave", "genz_oscillatory", "genz_product_peak", "genz_corner_peak", "genz_gaussian"),
    "many local features": ("ackley", "rastrigin", "griewank", "styblinski_tang", "michalewicz"),
    "kinks, jumps, fronts": ("genz_continuous", "genz_discontinuous", "oblique_jump", "ball_jump", "front"),
    "ODE flow maps": ("lorenz96", "lorenz96_chaotic", "lotka_volterra"),
    "vector-valued": ("rosenbrock_gradient", "genz_triple", "lotka_volterra_state", "swirl", "wave_pair", "shared_latent", "lorenz96_state"),
}
START, END = "<!-- benchmark-table -->", "<!-- /benchmark-table -->"


def cases(records: list) -> dict:
    """(problem, d, tol) -> method -> its records (every seed)."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        out[(r["problem"], int(r["d"]), float(r["tol"]))][r["method"]].append(r)
    return out


def summarise(runs: dict, problems: set) -> dict:
    """method -> ran, held, won, held at d >= 10, median evaluations and inference us over the held cases."""
    stats: dict = defaultdict(lambda: {"ran": 0, "held": 0, "won": 0, "held_high": 0, "evals": [], "predict_us": []})
    for (problem, d, _), by_method in runs.items():
        if problem not in problems:
            continue
        holders = {}
        for method, rs in by_method.items():
            s = stats[method]
            s["ran"] += 1
            if all("failed" not in r and r.get("worst_ratio", float("inf")) <= 1.0 for r in rs):
                s["held"] += 1
                s["held_high"] += d >= 10
                evals = sum(r["evals"] for r in rs) / len(rs)
                s["evals"].append(evals)
                s["predict_us"].append(median(r["predict_us"] for r in rs))
                holders[method] = evals
        if holders:  # a tie is a win for every tied method (ChebTT-A's first attempt is a ChebTT build, for one)
            least = min(holders.values())
            for method, evals in holders.items():
                stats[method]["won"] += evals == least
    return stats


def stale_share(records: list) -> dict:
    """method -> share of its records written before the last change to its results."""
    n, old = defaultdict(int), defaultdict(int)
    for r in records:
        n[r["method"]] += 1
        changed = RESULTS_CHANGED.get(r["method"], ("", ""))[0]
        old[r["method"]] += r.get("recorded", "") < changed
    return {m: old[m] / n[m] for m in n}


def _num(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}k" if v < 1e4 else f"{v / 1e3:.0f}k"
    return f"{v:.0f}" if v >= 10 else f"{v:.2g}"


def markdown(labels: dict) -> str:
    """The table for both suites, with a key."""
    functions = Store("functions", "cpu-8rank").runs()
    steps = Store("step_maps", "cpu-1rank").runs()
    stale = stale_share(functions + steps)
    lines = []
    for suite, records, classes in (("function suite (8 CPU ranks, 3 seeds)", functions, FUNCTION_CLASSES),
                                    ("step-map suite (1 CPU core, rollouts of held-out trajectories)", steps, None)):  # fmt: skip
        runs = cases(records)
        if classes is None:
            from src.problems.ode import STEP_MAPS

            classes = {regime: tuple(n for n, p in STEP_MAPS.items() if p.regime == regime) for regime in ("stiff", "nonstiff")}
        lines += [f"**{suite[0].upper() + suite[1:]}**", "",
                  "| class | method | ran | held | won | held at d >= 10 | median evaluations (held) | median inference us (held) |",
                  "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for cls, problems in classes.items():
            stats = summarise(runs, set(problems))
            order = sorted(stats, key=lambda m: (-stats[m]["won"], -stats[m]["held"], m))
            for i, m in enumerate(order):
                s = stats[m]
                mark = " *" if stale.get(m, 0) > 0 else ""
                ev = _num(median(s["evals"])) if s["evals"] else "-"
                us = f"{median(s['predict_us']):.2g}" if s["predict_us"] else "-"
                lines.append(f"| {cls if i == 0 else ''} | {labels.get(m, m)}{mark} | {s['ran']} | {s['held']} | {s['won']} | {s['held_high']} | {ev} | {us} |")
        lines.append("")
    unrun = [m for m in labels if m not in stale]
    lines.append("A case is a problem, dimension and tolerance; *held*: worst error within the tolerance on every seed; "
                 "*won*: held with the fewest evaluations (ties count for each). Medians are over each method's own held cases, "
                 "which differ in difficulty between methods. *Ran* differs between methods because the sweeps stop a method at its "
                 "first failure on a family. " + stale_note(stale, labels)
                 + (f" Registered but not yet run: {', '.join(labels[m] for m in unrun)}." if unrun else ""))
    return "\n".join(lines)


def stale_note(stale: dict, labels: dict) -> str:
    """The key's sentence on records older than their method's last change to its results, or that there are none."""
    methods = sorted((m for m, v in stale.items() if v > 0), key=lambda m: -stale[m])
    where = "`RESULTS_CHANGED` in `src/benchmark/table.py`"
    if not methods:
        return f"No record is older than its method's last change to its results ({where})."
    return (f"* marks records older than the method's last change to its results (share of its records; see {where}): "
            + "; ".join(f"{labels.get(m, m)} {100 * stale[m]:.0f}% ({RESULTS_CHANGED[m][1]})" for m in methods) + ".")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--write", type=Path, default=None, help="replace the marked block in this file")
    a = parser.parse_args(argv)
    from src.benchmark.methods import METHODS
    from src.benchmark.step_maps import METHODS as STEP_METHODS

    labels = {m: v[0].split(":")[0] for m, v in {**STEP_METHODS, **METHODS}.items()}
    text = markdown(labels)
    if a.write is None:
        print(text)
        return
    doc = a.write.read_text()
    if START not in doc or END not in doc:
        raise SystemExit(f"{a.write} has no {START} ... {END} block")
    a.write.write_text(re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: f"{START}\n{text}\n{END}", doc, flags=re.S))


if __name__ == "__main__":
    main()
