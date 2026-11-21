"""
Function-suite pass, vector-valued functions (memo Part I): the Jacobian's local rank (Proposition 6.5's lower count) and the
structure each problem's definition gives, against which methods held in the benchmark's records.

For each vector problem and d = 5, 10: the singular values of the Jacobian (central differences) at 200 uniform points of its
box, normalised by the largest, and the local rank at thresholds 1e-2 and 1e-6 (the median over the points). Predictions:
  wave_pair (cos, sin of 1.5 q.x): rank 1 at every point, a global coordinate count of 1: a rotated model (ChebTT-M, the hub)
      should cost about as much at d = 10 as at d = 2; axis-aligned ChebTT pays for the obliqueness;
  shared_latent (six outputs of two oblique coordinates): rank 2;
  rosenbrock_gradient: full rank, but each output depends on at most three inputs (a window): per-output ChebTT is cheap and a
      rotation gains nothing;
  lotka_volterra_state, genz_triple, swirl: full rank (no reduction available from the Jacobian).
The records' table: cases held on every seed over cases run, and the median evaluations of the held ones (seed 1).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.problems.registry import VECTOR_FUNCTIONS

ROOT = Path(__file__).resolve().parents[2] / "results/benchmark/runs/functions/cpu-8rank"
OUT = Path(__file__).resolve().parent / "data"
PROBLEMS = ("wave_pair", "shared_latent", "rosenbrock_gradient", "lotka_volterra_state", "genz_triple", "swirl")
METHODS = ("chebtt", "chebtt_metric", "chebtt_adaptive", "hub", "art_quad", "art")


def jacobian_spectrum(name: str, d: int, n: int = 200) -> dict:
    prob = VECTOR_FUNCTIONS[name]
    rng = np.random.default_rng(0)
    X = rng.uniform(prob.lo, prob.hi, (n, d))
    h = 1e-5 * (prob.hi - prob.lo)
    f = lambda Z: np.asarray(prob.f(Z)).reshape(len(Z), -1)  # noqa: E731
    cols = [(f(X + h * e) - f(X - h * e)) / (2 * h) for e in np.eye(d)]
    J = np.stack(cols, axis=2)  # (n, m, d)
    s = np.linalg.svd(J, compute_uv=False)
    s = s / np.maximum(s[:, :1], 1e-300)
    return {"m": int(J.shape[1]), "median_normalised_sigma": np.median(s, axis=0).tolist(),
            "rank_1e-2": int(np.median(np.sum(s > 1e-2, axis=1))), "rank_1e-6": int(np.median(np.sum(s > 1e-6, axis=1)))}  # fmt: skip


def records() -> dict:
    cases = defaultdict(lambda: defaultdict(list))
    for prob in PROBLEMS:
        for p in ROOT.glob(f"{prob}/d*/*/*.json"):
            r = json.loads(p.read_text())
            if r.get("method") in METHODS:
                cases[(prob, r["method"], int(r["d"]))][float(r["tol"])].append(r)
    table = {}
    for (prob, meth, d), by_tol in sorted(cases.items()):
        held, evals = 0, []
        for rs in by_tol.values():
            ok = all(r.get("status") == "done" and r.get("worst_ratio", 9) <= 1 for r in rs)
            held += ok
            evals += [r["evals"] for r in rs if ok and r.get("seed") == 1]
        evals.sort()
        table.setdefault(prob, {}).setdefault(meth, {})[d] = {"ran": len(by_tol), "held": held, "median_evals_held": evals[len(evals) // 2] if evals else None}
    return table


def main() -> None:
    OUT.mkdir(exist_ok=True)
    spec = {name: {d: jacobian_spectrum(name, d) for d in (5, 10)} for name in PROBLEMS}
    table = records()
    (OUT / "f3_vector.json").write_text(json.dumps({"jacobian": spec, "records": table}, indent=1, sort_keys=True) + "\n")
    for name in PROBLEMS:
        s5, s10 = spec[name][5], spec[name][10]
        print(f"{name} (m = {s5['m']} at d=5): local rank at 1e-2 / 1e-6: d=5 {s5['rank_1e-2']}/{s5['rank_1e-6']}, d=10 {s10['rank_1e-2']}/{s10['rank_1e-6']}; "
              f"median sigma (d=10) {np.round(s10['median_normalised_sigma'][:6], 3).tolist()}")  # fmt: skip
        for meth in METHODS:
            if meth in table.get(name, {}):
                cells = "  ".join(f"d{d}: {c['held']}/{c['ran']}" + (f" ({c['median_evals_held']})" if c["median_evals_held"] else "") for d, c in sorted(table[name][meth].items()))
                print(f"   {meth:16s} {cells}")


if __name__ == "__main__":
    main()
