"""
Markdown tables of ``scripts/bench_inference.py`` results, with the CPU and GPU runs side by side:
one table of the models (leaves, oracle evaluations, training time, test errors) and one of
inference times, with ART's speedup over AFINNI on each backend and the GPU's over the CPU.

    python scripts/bench_inference_report.py                                  # scratch/bench_inference
    python scripts/bench_inference_report.py --root path/to/bench_inference --out report.md

Reads every ``<root>/<cpu|gpu>/<problem>_d<d>.json`` (the per-case records, not ``summary.json``,
which holds only the cases of the last invocation), so results copied from a cluster can be
tabled anywhere: only the standard library is used. A case missing on one backend shows "–" there.
The model table reports the CPU run where both exist; the two backends train at rounding-level
differences, so a leaf or evaluation count that differs between them is marked with "*".
"""

# The annotations below also run on Pythons older than 3.10, so the script runs on any machine.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BACKENDS = ("cpu", "gpu")
METHODS = ("afinni", "art")
NAMES = {"afinni": "AFINNI", "art": "ART"}
PROBLEM_ORDER = ("rosenbrock", "spherical_piecewise", "lorenz96")


def load(root: Path) -> dict:
    """``{(problem, dim): {backend: record}}`` from every per-case record under ``root``."""
    cases: dict = {}
    for backend in BACKENDS:
        for path in sorted((root / backend).glob("*_d*.json")):
            record = json.loads(path.read_text())
            if "methods" in record:  # a per-case record, not a summary or another file
                cases.setdefault((record["problem"], record["dim"]), {})[backend] = record
    order = {p: i for i, p in enumerate(PROBLEM_ORDER)}
    return dict(sorted(cases.items(), key=lambda kv: (order.get(kv[0][0], len(order)), kv[0][0], kv[0][1])))


def fmt(value, kind: str) -> str:
    """One cell: "–" for a missing or non-finite value."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "sci":
        return f"{value:.2e}"
    if kind == "pct":
        return f"{value:.1f}"
    if kind == "x":
        return f"{value:.1f}×"
    return f"{value:,.2f}"


def entry(runs: dict, backend: str, method: str) -> dict | None:
    """The method's entry of one backend's record, or None."""
    return runs[backend]["methods"].get(method) if backend in runs else None


def table(header: list[str], rows: list[list[str]], right_from: int = 2) -> str:
    """A Markdown table, columns from ``right_from`` on right-aligned."""
    align = ["---" if i < right_from else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def model_table(cases: dict) -> str:
    """Leaves, evaluations, training time and errors per case and method."""
    rows = []
    for (problem, d), runs in cases.items():
        for method in METHODS:
            ref = entry(runs, "cpu", method) or entry(runs, "gpu", method)
            if ref is None:
                continue
            cpu, gpu = entry(runs, "cpu", method), entry(runs, "gpu", method)
            differs = cpu is not None and gpu is not None and (
                cpu["leaves"] != gpu["leaves"] or cpu["oracle_evals"] != gpu["oracle_evals"]
            )
            mark = "*" if differs else ""
            rows.append([
                problem, str(d), NAMES[method],
                fmt(ref["leaves"], "int") + mark, fmt(ref.get("fallback_leaves"), "int"),
                fmt(ref["oracle_evals"], "int") + mark,
                fmt(cpu["train_seconds"] if cpu else None, "f"), fmt(gpu["train_seconds"] if gpu else None, "f"),
                fmt(ref["relative_l2"], "sci"), fmt(ref["q99_pointwise"], "sci"),
                fmt(ref["pct_above_tol"], "pct"), fmt(ref["pct_outside"], "pct"),
            ])  # fmt: skip
    header = ["problem", "d", "method", "leaves", "fallback", "evaluations", "train CPU [s]", "train GPU [s]",
              "rel. L2", "q99 pointwise", "% > tol", "% outside"]  # fmt: skip
    return table(header, rows, right_from=3)


def inference_table(cases: dict) -> str:
    """Median inference time per query on each backend, ART's speedup over AFINNI, and the GPU's over the CPU."""
    rows = []
    for (problem, d), runs in cases.items():
        n_test = next(iter(runs.values()))["n_test"]
        us = {(b, m): (e["inference_us_per_query"] if (e := entry(runs, b, m)) else None)
              for b in BACKENDS for m in METHODS}  # fmt: skip

        def ratio(a, b):
            return a / b if a is not None and b else None

        rows.append([
            problem, str(d), fmt(n_test, "int"),
            fmt(us["cpu", "afinni"], "f"), fmt(us["cpu", "art"], "f"), fmt(ratio(us["cpu", "afinni"], us["cpu", "art"]), "x"),
            fmt(us["gpu", "afinni"], "f"), fmt(us["gpu", "art"], "f"), fmt(ratio(us["gpu", "afinni"], us["gpu", "art"]), "x"),
            fmt(ratio(us["cpu", "afinni"], us["gpu", "afinni"]), "x"), fmt(ratio(us["cpu", "art"], us["gpu", "art"]), "x"),
        ])  # fmt: skip
    header = ["problem", "d", "queries", "CPU AFINNI [µs/q]", "CPU ART [µs/q]", "CPU ART speedup",
              "GPU AFINNI [µs/q]", "GPU ART [µs/q]", "GPU ART speedup", "GPU/CPU AFINNI", "GPU/CPU ART"]  # fmt: skip
    return table(header, rows, right_from=1)


def settings_line(cases: dict) -> str:
    """The settings shared by the records, or a note that they differ."""
    keys = ("tol", "max_levels", "n_test", "repeats", "seed")
    seen = {tuple(r.get(k) for k in keys) for runs in cases.values() for r in runs.values()}
    if len(seen) != 1:
        return "Settings differ between records; see the per-case JSON files."
    values = dict(zip(keys, next(iter(seen))))  # noqa: B905 (zip(strict=) needs Python 3.10)
    return ", ".join(f"{k} = {v}" for k, v in values.items())


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("scratch/bench_inference"),
                   help="folder holding cpu/ and gpu/ (default: %(default)s)")
    p.add_argument("--out", type=Path, help="also write the tables to this Markdown file")
    args = p.parse_args()
    cases = load(args.root)
    if not cases:
        raise SystemExit(f"No per-case records under {args.root}/cpu or {args.root}/gpu.")
    found = ", ".join(b for b in BACKENDS if any(b in runs for runs in cases.values()))
    text = "\n\n".join([
        f"# Inference benchmark: AFINNI against ART\n\nBackends: {found}. Settings: {settings_line(cases)}.",
        "## Models\n\n" + model_table(cases)
        + "\n\nModel columns are from the CPU run where both exist; * marks a count that differs between backends.",
        "## Inference (median over the timed calls)\n\n" + inference_table(cases)
        + "\n\nSpeedup columns: AFINNI time / ART time on one backend; CPU time / GPU time for one method.",
    ])  # fmt: skip
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
