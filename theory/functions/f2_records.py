"""
Function-suite pass, kinks and jumps and vector outputs (memo Part I), against the benchmark's records (no new runs).

Predictions:
  axis-aligned kinks and jumps (genz_continuous, genz_discontinuous): ChebTT with breakpoints holds them (each piece analytic);
  an oblique jump (oblique_jump) or a curved one (ball_jump): no breakpoint of an axis represents it, its TT rank is of order n,
      so ChebTT should fail where ChebTT-A's interface term (a fitted plane or quadric) holds;
  a sharp smooth front on a sphere (front): analytic but with a nearby singularity across an oblique surface: costly for ChebTT,
      cheaper with the interface term.
Reported per problem, method and dimension: cases run, cases held on every seed (worst error within the tolerance), and the
median evaluations of the held ones (seed 1).
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results/benchmark/runs/functions/cpu-8rank"
OUT = Path(__file__).resolve().parent / "data"
PROBLEMS = ("genz_continuous", "genz_discontinuous", "oblique_jump", "ball_jump", "front")
METHODS = ("chebtt", "chebtt_metric", "chebtt_adaptive", "art", "art_quad", "afinni", "hub")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cases = defaultdict(lambda: defaultdict(list))  # (problem, method, d) -> tol -> [records]
    for prob in PROBLEMS:
        for p in ROOT.glob(f"{prob}/d*/*/*.json"):
            r = json.loads(p.read_text())
            if r.get("method") in METHODS:
                cases[(prob, r["method"], int(r["d"]))][float(r["tol"])].append(r)
    table = {}
    for (prob, meth, d), by_tol in sorted(cases.items()):
        ran = held = 0
        evals = []
        for _tol, rs in by_tol.items():
            ran += 1
            ok = all(r.get("status") == "done" and r.get("worst_ratio", 9) <= 1 for r in rs)
            held += ok
            if ok:
                evals += [r["evals"] for r in rs if r.get("seed") == 1]
        evals.sort()
        table.setdefault(prob, {}).setdefault(meth, {})[d] = {"ran": ran, "held": held, "median_evals_held": evals[len(evals) // 2] if evals else None}
    (OUT / "f2_records.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    for prob, by_m in table.items():
        print(prob)
        for meth in METHODS:
            if meth in by_m:
                cells = "  ".join(f"d{d}: {c['held']}/{c['ran']}" + (f" ({c['median_evals_held']})" if c["median_evals_held"] else "") for d, c in sorted(by_m[meth].items()))
                print(f"   {meth:16s} {cells}")


if __name__ == "__main__":
    main()
