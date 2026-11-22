"""
The split replay (council: Sol's first experiment, Astra's closed-chain audit): a scripted one-dimensional column of cells under operator
splitting, where every cell's neighbours carry the surrogate's own errors. The audit of every surrogate that holds chained mixing: the
generic recipe where R4c serves it (generic3.rebuild, the same surrogate) and rollout-tuned ChebTT on the problems of p4 (rebuilt as the
benchmark builds it). Nothing is written to the benchmark's store.

The column: the problem's test trajectories are its cells, ordered along the first principal coordinate of their initial states (so
neighbours start alike and the column has a gradient). Each step is transport, then chemistry:
  transport   x_i <- (1 - 2D) x_i + D x_{i-1} + D x_{i+1} in the system's own variables (abundances, not log10), reflecting ends, on the
              free components: a convex combination, so it is conservative (column totals are kept) and keeps positivity;
  environment each cell's held inputs follow its own test trajectory's path (cells differ, and change in time);
  chemistry   the surrogate on accepted inputs, the solver on refused ones.
D = 0.005 and 0.05, the same total exchange per step (2D) as the chained mixing weights 0.01 and 0.1. The reference column runs the same
splitting with the solver alone. Scores per (surrogate, D): the worst scored error of the hybrid column against the reference over every
cell and step, by step; the solver share; the worst one-step defect on the hybrid's *own accepted inputs* (the surrogate's answer against
the solver's from the same input, which isolates the step from the inherited error); the largest relative drift of the linear invariants
(element totals, `invariants_of`) between the hybrid and reference cells; and the column's wall time with the solver's share of it.
One record per (problem, surrogate) under data/records/p6/; a case is limited to 900 s. Usage: p6_column.py [problem ...].

Predictions, committed before the run (from R4c and p4: every audited surrogate held chained mixing at both weights):
  S1  every audited surrogate holds the tolerance in the column at both D, with the worst within a factor 2 of its chained-mixing worst at
      the same total exchange (neighbours carrying surrogate error do not compound: transport averages errors of both signs);
  S2  the column's solver share is within 5 percentage points of the chained-mixing share at the same exchange;
  S3  the defect on the recipe's own accepted inputs stays within half the tolerance (its gate is calibrated on mixed inputs for the
      maximum); ChebTT's own-input defect stays within the tolerance;
  S4  the recipe's invariant drift is below 1e-6 relative wherever the system has invariants (it projects onto the input's invariants and
      transport keeps them), and ChebTT's (no projection) is below 5e-2.
"""

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from generic3 import rebuild

from src.benchmark.step_maps import METHODS, STEP_MAPS, data, invariants_of
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
TOL = 0.01
EXCHANGE = (0.005, 0.05)


def surrogates(name: str) -> list[str]:
    """The audited surrogates of ``name``: the recipe where R4c serves it, rollout-tuned ChebTT where p4 holds it."""
    out = []
    r4c = OUT / "records" / "r4c" / f"{name}.json"
    if r4c.exists() and json.loads(r4c.read_text()).get("served"):
        out.append("recipe")
    p4 = OUT / "records" / "p4" / f"{name}.json"
    if p4.exists() and json.loads(p4.read_text()).get("status") == "done":
        out.append("chebtt_roll")
    return out


def candidates() -> list[str]:
    names = {Path(f).stem for f in glob.glob(str(OUT / "records" / "r4c" / "*.json")) + glob.glob(str(OUT / "records" / "p4" / "*.json"))}
    return sorted((n for n in names if surrogates(n)), key=lambda n: (data(n)[0].shape[2], n))


def column(P, sur, test, D, W) -> dict:
    """The hybrid and the reference column side by side (module docstring)."""
    h = P.system.held
    free = test.shape[2] - h
    order = np.argsort((test[:, 0, :free] - test[:, 0, :free].mean(0)) @ np.linalg.svd(test[:, 0, :free] - test[:, 0, :free].mean(0))[2][0])
    true = test[order]
    lin = (lambda Z: 10.0**Z) if P.system.log else (lambda Z: Z)
    back = (lambda Y: np.log10(np.maximum(Y, 1e-300))) if P.system.log else (lambda Y: Y)

    def transport(Z, k):
        X = lin(Z[:, :free])
        Xp = np.concatenate([X[:1], X, X[-1:]])  # reflecting ends: an end cell exchanges with itself
        out = Z.copy()
        out[:, :free] = back((1 - 2 * D) * X + D * Xp[:-2] + D * Xp[2:])
        if h:
            out[:, free:] = true[:, k, free:]
        return out

    Zt, Zs = true[:, 0].copy(), true[:, 0].copy()
    worst_by_step, solved, own, drift = [], [], 0.0, 0.0
    solve_s, t0 = 0.0, time.perf_counter()
    for k in range(true.shape[1] - 1):
        t1 = time.perf_counter()
        Zt = P.step(transport(Zt, k))
        Zin = transport(Zs, k)
        r = np.asarray(sur.refuses(Zin), dtype=bool)
        Zs = Zin + sur.predict(Zin)
        exact = P.step(Zin)  # the solver from the hybrid's own inputs: the fallback for refused cells, the audit for accepted ones
        solve_s += time.perf_counter() - t1
        if (~r).any():
            a = ~r
            e = np.where(P.scored(exact[a]), np.abs(Zs[a] - exact[a]), 0.0)
            own = max(own, float(np.max(e)) if np.all(np.isfinite(e)) else np.inf)
        Zs[r] = exact[r]
        e = np.where(P.scored(Zt), np.abs(Zs - Zt), 0.0)
        worst_by_step.append(float(np.max(e)) if np.all(np.isfinite(e)) else np.inf)
        solved.append(float(r.mean()))
        if len(W):
            mt, ms = lin(Zt[:, :free]) @ W[:, :free].T, lin(Zs[:, :free]) @ W[:, :free].T
            scale = np.maximum(np.abs(mt), 1e-300)
            big = np.abs(mt) > 1e-12 * np.abs(mt).max()  # a law whose total is at rounding level carries no relative drift
            drift = max(drift, float(np.max(np.where(big, np.abs(ms - mt) / scale, 0.0))))
    return {"worst": max(worst_by_step), "worst_by_step": worst_by_step, "solver_share": float(np.mean(solved)),
            "solver_share_by_step": solved, "own_input_defect_worst": own, "invariant_drift_relative": drift if len(W) else None,
            "cells": len(true), "steps": true.shape[1] - 1, "wall_s": time.perf_counter() - t0, "step_and_solve_s": solve_s}  # fmt: skip


def case(name: str, method: str) -> dict:
    P = STEP_MAPS[name]
    train, test = data(name)
    t0 = time.perf_counter()
    if method == "recipe":
        rec = json.loads((OUT / "records" / "r4c" / f"{name}.json").read_text())
        sur = rebuild(P, train, rec)
        from src.benchmark.step_maps import rollout

        # the rebuilt surrogate is R4c's: its 20-trajectory rollout matches to rounding (not bits: BLAS's thread count, which this
        # run need not share with R4c's, changes the least-squares fit's rounding)
        check = float(rollout(P, sur, test[:20])["rollout_worst"])
        if not np.isclose(check, rec["rollout_worst_20"], rtol=1e-3, atol=1e-12):
            raise RuntimeError(f"rebuilt recipe differs from R4c's: rollout {check} against {rec['rollout_worst_20']}")
    else:
        sur = METHODS["chebtt_roll"][2](P, train, TOL, 1)
    build_s = time.perf_counter() - t0
    W = invariants_of(P, train)
    out = {"build_s": build_s, "invariants": int(len(W)), "columns": {}}
    for D in EXCHANGE:
        c = column(P, sur, test, D, W)
        out["columns"][str(D)] = c
        log(f"  {name} {method} D={D}: worst {c['worst']:.2e} (solver {100 * c['solver_share']:.1f}%), own-input defect "
            f"{c['own_input_defect_worst']:.2e}, invariant drift {c['invariant_drift_relative']}, {c['wall_s']:.0f} s")  # fmt: skip
    return out


def main() -> None:
    names = sys.argv[1:] or candidates()
    log(f"column candidates: {[(n, surrogates(n)) for n in names]}")
    store = RecordStore(OUT / "records" / "p6", path_of=lambda key: OUT / "records" / "p6" / f"{key['problem']}_{key['method']}.json")
    for name in names:
        for method in surrogates(name):
            recorded(store, {"problem": name, "method": method, "tol": TOL, "exchange": list(EXCHANGE)}, case, name, method, limit=900)
    summary = {f"{r['key']['problem']}/{r['key']['method']}": {"status": r.get("status"), "build_s": r.get("build_s"),
               "columns": {D: {k: v for k, v in c.items() if not k.endswith("_by_step")} for D, c in (r.get("columns") or {}).items()}}
               for r in store.records()}  # fmt: skip
    (OUT / "p6_column.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
