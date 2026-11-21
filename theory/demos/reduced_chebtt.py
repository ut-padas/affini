"""
Theorem 8's design with the library's ChebTT (memo section 8, open item 1): a reduced-coordinate ChebTT of a stiff step map against
the benchmark's box ChebTT, scored with the benchmark's own metrics. Usage: reduced_chebtt.py <problem> [tol].

Design (each step is the memo's, nothing tuned to the problem):
  chart   the principal directions of the training trajectories' visited states (mean mu, directions V);
  lift    u -> mu + V_k u; ChebTT queries the solver there. Its error on the visited states, |Phi(lift(pi(z))) - Phi(z)|, is
          Proposition 8.2's middle term, measured on 400 held-out states for k = 1, 2, ...;
  k       the smallest k whose lift error is within a quarter of the tolerance (the rest of the budget goes to ChebTT). A data-only
          test (a least-squares polynomial of degree 3, or 2 above k = 5, in the chart coordinates) is reported beside it;
  model   build_vector of the next state (all d outputs) on the chart box of the training states padded by 2%, refusing states
          whose chart coordinates fall outside it (the solver steps them in a rollout), projected onto the system's linear
          invariants (the benchmark's Projected) so they are conserved exactly.
Scored like the benchmark (`src/benchmark/step_maps.py`): the one-step worst over the test states (the problem's scored entries),
the share refused, the rollout of 20 test trajectories, the error surviving one exact step; compared with the archived records of
box ChebTT and SlowTable at the same tolerance, seed 1. Nothing is written to the benchmark's store.
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


class Reduced:
    """Psi(z) = G(V^T (z - mu)) as an increment model (the benchmark's interface), refusing chart coordinates outside the box."""

    def __init__(self, model, mu, V, lo, hi, evals):
        self.model, self.mu, self.V, self.lo, self.hi, self.evals = model, mu, V, lo, hi, evals

    def chart(self, Z):
        return (Z - self.mu) @ self.V

    def refuses(self, Z):
        U = self.chart(Z)
        return np.any((self.lo > U) | (self.hi < U), axis=1)

    def predict(self, Z):
        U = np.clip(self.chart(Z), self.lo, self.hi)
        return np.asarray(self.model.predict(U)) - Z


def record(name, method, tol):
    for p in (RECORDS / name).rglob(f"{method}/tol{tol:g}-s1.json"):
        r = json.loads(p.read_text())
        if r.get("status") == "done":
            return {"evals": r["evals"], "worst": r["worst"], "rollout_worst": max(r["rollout_error_by_step"]), "survived": r.get("survived_worst"),
                    "refused_percent": r.get("refused_percent"), "train_s": r.get("train_s")}  # fmt: skip
        return {"failed": r.get("failed")}
    return None


def main() -> None:
    name = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    p = STEP_MAPS[name]
    train, test = data(name)
    X = visited(train)
    Y = train[:, 1:].transpose(1, 0, 2).reshape(-1, train.shape[2])
    Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    mask = p.scored(Ye)
    mu = X.mean(axis=0)
    _, sv, Vt = np.linalg.svd(X - mu, full_matrices=False)
    d = X.shape[1]
    # k: the smallest chart dimension at which a function of the chart predicts the next state
    fit = []
    for k in range(1, min(d, 8) + 1):
        V = Vt[:k].T
        q = 3 if k <= 5 else 2
        A = poly((X - mu) @ V, q)
        coef = np.linalg.lstsq(A, Y, rcond=None)[0]
        err = float(np.max(np.where(mask, np.abs(poly((Xe - mu) @ V, q) @ coef - Ye), 0.0)))
        fit.append({"k": k, "degree": q, "heldout_one_step_worst": err})
    print(f"{name}: data-only fit test", [(f["k"], f"{f['heldout_one_step_worst']:.2e}") for f in fit], flush=True)
    # the lift's error on held-out visited states (Proposition 8.2's middle term), for growing k
    idx = np.random.default_rng(0).choice(len(Xe), min(400, len(Xe)), replace=False)
    lift = []
    for k in range(1, d + 1):
        V = Vt[:k].T
        e = float(np.max(np.where(mask[idx], np.abs(p.step(mu + ((Xe[idx] - mu) @ V) @ V.T) - Ye[idx]), 0.0)))
        lift.append({"k": k, "lift_error": e})
        print(f"   k = {k}: lift error {e:.2e}", flush=True)
        if e <= tol / 4:
            break
    k = lift[-1]["k"]
    V = Vt[:k].T
    lift_err = lift[-1]["lift_error"]
    out = {"problem": name, "tol": tol, "d": d, "singular_values_of_visited": (sv / sv[0]).tolist()[:10], "fit_test": fit, "lift_by_k": lift, "k": k,
           "lift_error": lift_err, "records": {m: record(name, m, tol) for m in ("chebtt_box", "chebtt_roll", "slowtable")}}  # fmt: skip
    U = (X - mu) @ V
    lo, hi = U.min(0), U.max(0)
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    t0 = time.perf_counter()
    try:
        model = chebtt.build_vector(lambda W: p.step(mu + W @ V.T), lo, hi, tol / 2, d, seed=1, max_evals=200_000)  # lift <= tol/4, model <= tol/2
        out["build_s"] = time.perf_counter() - t0
        red = Reduced(model, mu, V, lo, hi, int(model.evals))
        W = invariants_of(p, train)
        surrogate = Projected(red, p, W) if len(W) else red
        refused = np.asarray(surrogate.refuses(Xe), dtype=bool)
        e = np.max(np.where(mask, np.abs(Xe + surrogate.predict(Xe) - Ye), 0.0), axis=1)
        out["reduced"] = {"evals": int(model.evals), "invariants": int(len(W)), "worst": float(e[~refused].max()) if (~refused).any() else None,
                          "refused_percent": float(100 * refused.mean()), "survived": survived(p, surrogate, test),
                          "rollout_worst": rollout(p, surrogate, test[:ROLLOUTS])["rollout_worst"], "build_s": out["build_s"]}  # fmt: skip
    except RuntimeError as err:
        out["reduced"] = {"failed": str(err)[:300], "build_s": time.perf_counter() - t0}
    OUT.mkdir(exist_ok=True)
    (OUT / f"reduced_chebtt_{name}.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    print(f"{name} (d = {d}, tol {tol}): visited singular values / first {np.round(out['singular_values_of_visited'][:8], 4).tolist()}")
    print(f"   k = {k}, lift error {lift_err:.2e}")
    print("   reduced ChebTT:", out["reduced"])
    for m, r in out["records"].items():
        print(f"   record {m}: {r}")


if __name__ == "__main__":
    main()
