"""
AFINNI and ART with and without kernel ridge regression on their leaves, case by case, on the same
trees: the KRR models are fitted after the tree is built (``KRREnrichedModel``), so the comparison
is the accuracy each leaf model gives and what it costs in training and inference time.

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off .venv/bin/python scripts/compare_enriched_krr.py
    SURROGATE_DEVICE=cpu SURROGATE_MPI=off .venv/bin/python scripts/compare_enriched_krr.py \
        --case rosenbrock:2 spherical_piecewise:3 --variants base krr_values --krr-settings '{"kernel": "matern52"}'
    SURROGATE_DEVICE=cpu SURROGATE_MPI=off .venv/bin/python scripts/compare_enriched_krr.py --report-only

Variants (``--variants``, all by default):

- ``base``: each method's default local model (affine least squares);
- ``krr_enriched``: ``local_model="krr_enriched"``, the affine fit plus a restricted KRR correction
  of its residuals on every leaf;
- ``krr_values``: ``local_model=("krr_enriched", {"target": "values"})``, a restricted KRR model of
  the function values on every leaf, replacing the affine fit (which serves only the tree).

Both KRR variants keep the affine fit on an output where KRR would raise the leaf's largest error
at its own points (the fallback), and take ``--krr-settings`` (a JSON object) on top.

Each variant of each case runs ``scripts/bench_inference.py`` in its own process with the
benchmark's settings (one root, tolerance 1e-3, 15 levels, 50,000 test points); a case already
saved with the same settings is reused, so an interrupted comparison resumes. The default cases are
those that train in minutes on one CPU; Rosenbrock d = 10 and Lorenz-96 d = 8 and 10 take hours there.

Outputs, under ``--out`` (default ``scratch/compare_enriched_krr``): one folder per variant with the
benchmark's per-case records and logs, and ``comparison.md`` with the tables.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ("rosenbrock:2", "spherical_piecewise:2", "spherical_piecewise:3", "spherical_piecewise:4",
                 "rosenbrock:5", "spherical_piecewise:5", "lorenz96:5")  # fmt: skip
# Variant -> KRREnrichedModel settings (None: each method's default local model).
VARIANTS = {"base": None, "krr_enriched": {}, "krr_values": {"target": "values"}}
LABELS = {"base": "base", "krr_enriched": "KRR residual", "krr_values": "KRR values"}
METHODS = {"afinni": "AFINNI", "art": "ART"}


def run(case: str, variant: str, args) -> None:
    """One variant of one case through bench_inference.py, in its own process."""
    problem, d = case.split(":")
    cmd = [sys.executable, str(ROOT / "scripts" / "bench_inference.py"), "--problem", problem, "--dim", d,
           "--out", str(args.out / variant), "--tol", str(args.tol), "--max-levels", str(args.max_levels),
           "--n-test", str(args.n_test), "--repeats", str(args.repeats)]  # fmt: skip
    if VARIANTS[variant] is not None:
        settings = {**VARIANTS[variant], **json.loads(args.krr_settings)}
        cmd += ["--local-model", "krr_enriched", "--local-model-settings", json.dumps(settings, sort_keys=True)]
    t = time.perf_counter()
    print(f"[{case}] {variant} ...", flush=True)
    log = args.out / variant / f"{problem}_d{d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        code = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=os.environ.copy())
    print(f"[{case}] {variant}: exit {code} after {time.perf_counter() - t:.0f} s (log {log})", flush=True)


def load(args) -> dict:
    """``{(problem, d): {variant: record}}`` of every saved case of the requested ones."""
    out = {}
    for case in args.case:
        problem, d = case.split(":")
        for variant in args.variants:
            path = args.out / variant / f"{problem}_d{d}.json"
            if path.exists():
                out.setdefault((problem, int(d)), {})[variant] = json.loads(path.read_text())
    return out


def fmt(v, kind="sci") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return {"sci": f"{v:.2e}", "x": f"{v:.1f}×", "pct": f"{v:.1f}", "s": f"{v:,.1f}", "us": f"{v:.2f}", "int": f"{int(v):,}"}[kind]


def ratio(a, b):
    return a / b if a is not None and b else None


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * 2 + ["---:"] * (len(header) - 2)) + "|"] + \
        ["| " + " | ".join(r) + " |" for r in rows]  # fmt: skip


def report(args) -> str:
    """The comparison as Markdown: per method, an accuracy table and a cost table."""
    cases = load(args)
    krr = [v for v in args.variants if VARIANTS[v] is not None]
    lines = [
        "# Kernel ridge regression on the leaves: AFINNI and ART",
        "",
        f"`scripts/compare_enriched_krr.py`, CPU, one process. Settings of `scripts/bench_inference.py`: one root, "
        f"tolerance {args.tol:g}, {args.max_levels} levels, {args.n_test:,} uniform test points, fallback leaves. "
        f"Extra KRR settings: `{args.krr_settings}` (defaults: Gaussian kernel in the leaf's trust metric, "
        f"`center_fraction` 0.1, ridge 1e-6, fallback on).",
        "",
        "- **base**: each method's default local model (affine least squares).",
        "- **KRR residual** (`krr_enriched`): the affine fit plus a restricted KRR correction of its residuals.",
        "- **KRR values** (`krr_values`, `target=\"values\"`): a restricted KRR model of the function values alone.",
        "- Every variant has the base model's tree (*same tree*: equal leaves and oracle evaluations). *gain*: base "
        "error / KRR error (above 1: KRR is more accurate). *Inside*: AFINNI's test points inside a trust region (ART: "
        "all of them). *On base / worse*: leaves that keep the affine fit for some output because KRR would raise "
        "their largest error at their own points (the fallback), and leaves still worse than the affine fit (0).",
        "",
    ]
    for method, label in METHODS.items():
        acc, cost = [], []
        for (problem, d), runs in cases.items():
            e = {v: runs.get(v, {}).get("methods", {}).get(method) for v in args.variants}
            g = lambda v, key, e=e: None if e.get(v) is None else e[v].get(key)  # noqa: E731
            present = [v for v in args.variants if e.get(v)]
            if not present:
                continue
            ref = e[present[0]]
            same = "yes" if all((e[v]["leaves"], e[v]["oracle_evals"]) == (ref["leaves"], ref["oracle_evals"])
                                for v in present) else "**no**"  # fmt: skip
            p = next((g(v, "krr_centers") for v in krr if g(v, "krr_centers") is not None), None)
            row = [problem, str(d), fmt(ref["leaves"], "int"), same if len(present) > 1 else "–", fmt(p, "int")]
            row += [fmt(g(v, "relative_l2")) for v in args.variants]
            row += [fmt(ratio(g("base", "relative_l2"), g(v, "relative_l2")), "x") for v in krr]
            row += [fmt(g(v, "relative_l2_inside")) for v in args.variants]
            row += [fmt(ratio(g("base", "relative_l2_inside"), g(v, "relative_l2_inside")), "x") for v in krr]
            row += [fmt(g(v, "pct_above_tol"), "pct") for v in args.variants]
            row += [f"{fmt(g(v, 'krr_leaves_fallback'), 'int')} / {fmt(g(v, 'krr_leaves_worse'), 'int')}" for v in krr]
            acc.append(row)
            cost.append([problem, str(d)] + [fmt(g(v, "train_seconds"), "s") for v in args.variants]
                        + [fmt(g(v, "inference_us_per_query"), "us") for v in args.variants])  # fmt: skip
        if not acc:
            continue
        names = [LABELS[v] for v in args.variants]
        knames = [LABELS[v] for v in krr]
        header = (["problem", "d", "leaves", "same tree", "p"] + [f"rel. L2 {n}" for n in names]
                  + [f"gain {n}" for n in knames] + [f"inside {n}" for n in names] + [f"inside gain {n}" for n in knames]
                  + [f"% > tol {n}" for n in names] + [f"on base / worse {n}" for n in knames])  # fmt: skip
        lines += [f"## {label}", "", "### Accuracy", ""] + table(header, acc) + ["", "### Cost", ""]
        lines += table(["problem", "d"] + [f"train {n} [s]" for n in names] + [f"µs/query {n}" for n in names], cost)
        lines.append("")
    missing = [c for c in args.case if len(cases.get((c.split(":")[0], int(c.split(":")[1])), {})) < len(args.variants)]
    if missing:
        lines += [f"Not every variant has run yet (or one failed; see the logs): {', '.join(missing)}.", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    p.add_argument("--case", nargs="+", default=list(DEFAULT_CASES), help="problem:dimension pairs")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--krr-settings", default="{}", help="extra KRREnrichedModel settings (JSON) for both KRR variants")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--max-levels", type=int, default=15)
    p.add_argument("--n-test", type=int, default=50_000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", type=Path, default=ROOT / "scratch" / "compare_enriched_krr")
    p.add_argument("--report-only", action="store_true", help="only rewrite comparison.md from the saved cases")
    args = p.parse_args()
    if not args.report_only:
        for case in args.case:
            for variant in args.variants:
                run(case, variant, args)
                (args.out / "comparison.md").write_text(report(args) + "\n")  # kept current while it runs
    text = report(args)
    (args.out / "comparison.md").write_text(text + "\n")
    print(text)
    print(f"\nWrote {args.out / 'comparison.md'}")


if __name__ == "__main__":
    main()
