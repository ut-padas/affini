"""
A lift onto the curved slow manifold for a reduced-coordinate surrogate (memo section 8, Proposition 8.2's middle term; Demonstration
8.4 found that an affine lift needs 7 of 8 coordinates on HIRES, whose visited set is three-dimensional). Usage:
curved_lift.py <problem> [k] [tol].

Construction. Chart: the top-k principal directions V of the training trajectories' visited states (mean mu), pi(z) = V^T (z - mu).
Lift of a chart value u, in three variants:
  affine     mu + V u;
  decoder    D(u), a least-squares polynomial (total degree 4) from chart values to states, fitted on the visited states (no solve);
  polished   D(u), then n constrained-run iterations (Gear, Kaper, Kevrekidis, Zagaris 2005): relax for a time s with the solver,
             then reset the chart coordinates to u along V, which pulls the state onto the slow manifold while keeping its chart value.
The lift error of each variant: max |Phi(lift(pi(z))) - Phi(z)| over 400 held-out visited states (the problem's scored entries).
Then a reduced ChebTT with the best variant: build_vector of the next state (all d outputs) on the chart box of the training states
padded by 2%, with the oracle u -> Phi(lift(u)), projected onto the linear invariants (the benchmark's Projected); scored like the
benchmark (one-step worst, share refused, 20 rollouts, survived error) and compared with the archived records. Solver calls are counted
per query: one per relaxation plus one for the step. Nothing is written to the benchmark's store.

Predictions, committed before the run (HIRES, k = 3, tolerance 0.01 dex):
  P1  the decoder lift's error is several times below the affine lift's (3.9e-2 at k = 3), and the polished lift's is within a quarter
      of the tolerance (2.5e-3);
  P2  with the polished lift the reduced ChebTT finishes within the benchmark's 240 s (box ChebTT did not), holds the tolerance in one
      step on the test states, and its rollouts stay bounded over the trajectories' 170 steps;
  P3  the linear invariant (y7 + y8) is conserved by the projected surrogate to rounding.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.benchmark.step_maps import (
    ROLLOUTS,
    STEP_MAPS,
    Projected,
    data,
    invariants_of,
    rollout,
    survived,
    visited,
)
from src.schemes import chebtt

OUT = Path(__file__).resolve().parent / "data"
RECORDS = Path(__file__).resolve().parents[2] / "results/benchmark/runs/step_maps/cpu-1rank"


def poly(U, q):
    cols = [np.ones(len(U))]
    for deg in range(1, q + 1):
        for idx in itertools.combinations_with_replacement(range(U.shape[1]), deg):
            cols.append(np.prod(U[:, list(idx)], axis=1))
    return np.stack(cols, axis=1)


class Lift:
    """Chart values to states: the decoder, then n constrained-run iterations of length s."""

    def __init__(self, p, mu, V, coef, q, n, s):
        self.p, self.mu, self.V, self.coef, self.q, self.n, self.s = p, mu, V, coef, q, n, s

    def __call__(self, U):
        Z = poly(U, self.q) @ self.coef if self.coef is not None else self.mu + U @ self.V.T
        for _ in range(self.n):
            Z = self.p.system.coords_step(Z, self.s)
            Z = Z + (U - (Z - self.mu) @ self.V) @ self.V.T
        return Z


class Reduced:
    """Psi(z) = G(pi(z)) as an increment model (the benchmark's interface), refusing chart values outside the box."""

    def __init__(self, model, mu, V, lo, hi, evals):
        self.model, self.mu, self.V, self.lo, self.hi, self.evals = model, mu, V, lo, hi, evals

    def refuses(self, Z):
        U = (Z - self.mu) @ self.V
        return np.any((self.lo > U) | (self.hi < U), axis=1)

    def predict(self, Z):
        U = np.clip((Z - self.mu) @ self.V, self.lo, self.hi)
        return np.asarray(self.model.predict(U)) - Z


def record(name, method, tol):
    for p in (RECORDS / name).rglob(f"{method}/tol{tol:g}-s1.json"):
        r = json.loads(p.read_text())
        if r.get("status") == "done":
            return {"evals": r["evals"], "worst": r["worst"], "rollout_worst": max(r["rollout_error_by_step"]), "survived": r.get("survived_worst")}
        return {"failed": r.get("failed")}
    return None


def main() -> None:
    name = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    p = STEP_MAPS[name]
    train, test = data(name)
    X = visited(train)
    Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    mu = X.mean(axis=0)
    V = np.linalg.svd(X - mu, full_matrices=False)[2][:k].T
    q = 4
    coef = np.linalg.lstsq(poly((X - mu) @ V, q), X, rcond=None)[0]
    idx = np.random.default_rng(0).choice(len(Xe), min(400, len(Xe)), replace=False)
    mask = p.scored(Ye[idx])
    Ue = (Xe[idx] - mu) @ V
    decoder_residual = float(np.max(np.abs(poly(Ue, q) @ coef - Xe[idx])))
    variants = {"affine": Lift(p, mu, V, None, q, 0, 0.0), "decoder": Lift(p, mu, V, coef, q, 0, 0.0)}
    for n, s in ((1, 0.5), (3, 0.5), (3, 1.0), (6, 0.5)):
        variants[f"polished n={n} s={s}"] = Lift(p, mu, V, coef, q, n, s)
    lift_err = {}
    for key, lift in variants.items():
        t0 = time.perf_counter()
        e = float(np.max(np.where(mask, np.abs(p.step(lift(Ue)) - Ye[idx]), 0.0)))
        lift_err[key] = {"lift_error": e, "solver_calls_per_query": lift.n + 1, "seconds": time.perf_counter() - t0}
        print(f"{name} k={k}: lift {key:22s} error {e:.2e} ({lift.n + 1} solver calls per query)", flush=True)
    out = {"problem": name, "k": k, "tol": tol, "decoder_degree": q, "decoder_residual": decoder_residual, "lifts": lift_err,
           "records": {m: record(name, m, tol) for m in ("chebtt_box", "chebtt_roll", "slowtable")}}  # fmt: skip
    best = min((v["lift_error"], key) for key, v in lift_err.items())[1]
    lift = variants[best]
    U = (X - mu) @ V
    lo, hi = U.min(0), U.max(0)
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    calls = {"n": 0}

    def oracle(W):
        calls["n"] += len(W)
        return p.step(lift(W))

    t0 = time.perf_counter()
    try:
        model = chebtt.build_vector(oracle, lo, hi, tol / 2, X.shape[1], seed=1, max_evals=200_000)
        build_s = time.perf_counter() - t0
        red = Reduced(model, mu, V, lo, hi, int(model.evals))
        W = invariants_of(p, train)
        sur = Projected(red, p, W) if len(W) else red
        refused = np.asarray(sur.refuses(Xe), dtype=bool)
        full = p.scored(Ye)
        e = np.max(np.where(full, np.abs(Xe + sur.predict(Xe) - Ye), 0.0), axis=1)
        roll = rollout(p, sur, test[:ROLLOUTS])
        inv = None
        if len(W):
            Yn = 10.0 ** (Xe + sur.predict(Xe)) if p.system.log else Xe + sur.predict(Xe)
            Y0 = 10.0**Xe if p.system.log else Xe
            inv = float(np.max(np.abs((Yn - Y0) @ W.T) / np.maximum(np.abs(Y0 @ W.T), 1e-300)))
        out["reduced"] = {"lift": best, "distinct_queries": int(model.evals), "solver_calls": calls["n"] * (lift.n + 1), "build_s": build_s,
                          "worst": float(e[~refused].max()) if (~refused).any() else None, "refused_percent": float(100 * refused.mean()),
                          "survived": survived(p, sur, test), "rollout_worst": roll["rollout_worst"],
                          "rollout_solved_share": float(np.mean(roll["rollout_solved_by_step"])), "invariant_relative_change_max": inv}  # fmt: skip
    except RuntimeError as err:
        out["reduced"] = {"lift": best, "failed": str(err)[:300], "build_s": time.perf_counter() - t0}
    OUT.mkdir(exist_ok=True)
    (OUT / f"curved_lift_{name}_k{k}.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    print("   decoder residual", f"{decoder_residual:.2e}", "; reduced ChebTT:", out["reduced"])
    for m, r in out["records"].items():
        print(f"   record {m}: {r}")


if __name__ == "__main__":
    main()
