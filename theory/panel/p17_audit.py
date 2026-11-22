"""
The end-to-end audit (council rounds 3 and 4: Sol's and Astra's qualification on fresh closed blocks with measured wall time). Every problem
R4g serves at 340 training trajectories, its surrogate and account rebuilt exactly as R4g deployed them (generic4, gate="capped",
sketch, account; the same 340 training and 100 calibration trajectories, lambda chosen again on the same chains), is run in closed columns
(p11: neighbours from the hybrid column itself; transport, then chemistry) whose cells are 100 fresh trajectories no fit, calibration or
test has seen (p13's seed 14), sorted along their first principal coordinate or in random order, at D = 0.005, 0.05 and 0.25. Scores:
the worst scored error over every cell and step against the solver-only column, the solver share, and the wall time: the hybrid column's
measured evaluation and gate time plus its refused cells at the problem's fastest qualified exact solver (p15's denominator, per state
and step), against the exact column at the same solver. One record per problem under data/records/p17/; a case is limited to 1,800 s.
Usage: p17_audit.py [problem ...].

Predictions, committed before the run:
  A1  every audited problem holds the tolerance in all six columns (its worst at most 0.01), and at most two problems exceed tol / 2 in
      any column;
  A2  the solver share in the columns is within 10 points of the problem's plain-rollout share in R4g, except hires (above 80%);
  A3  the end-to-end speed-up against the fastest exact solver (the median over the six columns) is at least 2 on at least 9 of the 13,
      and at least 10 on at least 4.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from generic4 import Account, calibrate_fresh, rebuild
from p13_learning import extra_trajectories

from src.benchmark.step_maps import STEP_MAPS, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
TOL = 0.01
VARIANT = {"gate": "capped", "sketch": True, "account": True}


def column(P, pol_factory, cells, D, order, ev_cost: list):
    """The hybrid closed column and the solver-only column on ``cells``; returns (worst, solver share, refused cell-steps)."""
    h = P.system.held
    free = cells.shape[2] - h
    X0 = cells[:, 0, :free] - cells[:, 0, :free].mean(0)
    perm = np.argsort(X0 @ np.linalg.svd(X0, full_matrices=False)[2][0]) if order == "sorted" else np.random.default_rng(4).permutation(len(cells))
    true = cells[perm]
    lin = (lambda Z: 10.0**Z) if P.system.log else (lambda Z: Z)
    back = (lambda Y: np.log10(np.maximum(Y, 1e-300))) if P.system.log else (lambda Y: Y)

    def transport(Z, k):
        X = lin(Z[:, :free])
        Xp = np.concatenate([X[:1], X, X[-1:]])
        out = Z.copy()
        out[:, :free] = back((1 - 2 * D) * X + D * Xp[:-2] + D * Xp[2:])
        if h:
            out[:, free:] = true[:, k, free:]
        return out

    pol = pol_factory()
    Zt, Zs = true[:, 0].copy(), true[:, 0].copy()
    worst, refused = 0.0, 0
    for k in range(true.shape[1] - 1):
        Zt = P.step(transport(Zt, k))
        Zin = transport(Zs, k)
        t0 = time.perf_counter()
        r = np.asarray(pol.refuses(Zin), dtype=bool)
        nxt = Zin + pol.predict(Zin)
        ev_cost.append(time.perf_counter() - t0)
        if r.any():
            nxt[r] = P.step(Zin[r])
        refused += int(r.sum())
        Zs = nxt
        e = np.where(P.scored(Zt), np.abs(Zs - Zt), 0.0)
        worst = max(worst, float(np.max(e)) if np.all(np.isfinite(e)) else float("inf"))
    return worst, refused / (len(true) * (true.shape[1] - 1)), refused


def case(name: str, dstore: RecordStore) -> dict:
    P = STEP_MAPS[name]
    train, _ = data(name)
    pool, _ = extra_trajectories(name, 360, dstore)
    cal, more = pool[:100], pool[120:360]
    train = np.concatenate([train, more[:240]])
    rec = json.loads((OUT / "records" / "r4g" / f"{name}_340.json").read_text())
    sur = rebuild(P, train, rec, None, VARIANT)
    calibrate_fresh(P, sur, cal)
    fresh, _ = extra_trajectories(name, 480, dstore)  # seeds 11-14; the audit's cells are seed 14's, never fitted or calibrated on
    cells = fresh[360:460]
    den = json.loads((OUT / "p15_denominators.json").read_text()).get(name, {})
    den_us = den.get("denominator_us")
    out = {"lambda": sur.lam, "cells": int(len(cells)), "steps": int(cells.shape[1] - 1), "denominator": den.get("denominator"), "denominator_us": den_us, "columns": {}}
    for order in ("sorted", "unsorted"):
        for D in (0.005, 0.05, 0.25):
            ev = []
            worst, share, refused = column(P, lambda: Account(sur), cells, D, order, ev)
            cs = len(cells) * (cells.shape[1] - 1)
            hybrid_s = sum(ev) + refused * den_us * 1e-6 if den_us else None
            exact_s = cs * den_us * 1e-6 if den_us else None
            c = {"worst": worst, "solver_share": share, "evaluate_s": sum(ev), "hybrid_s": hybrid_s, "exact_s": exact_s,
                 "speedup": exact_s / hybrid_s if hybrid_s else None}  # fmt: skip
            out["columns"][f"{order}/{D}"] = c
            log(f"  {name} {order} D={D}: worst {worst:.2e}, solver {100 * share:.1f}%, speed-up {c['speedup'] or float('nan'):.1f}")
    sp = [c["speedup"] for c in out["columns"].values() if c["speedup"]]
    out["median_speedup"] = float(np.median(sp)) if sp else None
    out["worst"] = max(c["worst"] for c in out["columns"].values())
    return out


def main() -> None:
    import glob

    served = sorted(json.loads(Path(f).read_text())["key"]["problem"] for f in glob.glob(str(OUT / "records" / "r4g" / "*_340.json"))
                    if json.loads(Path(f).read_text()).get("served"))  # fmt: skip
    names = sys.argv[1:] or served
    dstore = RecordStore(OUT / "records" / "p13_data", path_of=lambda k: OUT / "records" / "p13_data" / f"{k['problem']}_{k['chunk']}.json")
    store = RecordStore(OUT / "records" / "p17", path_of=lambda k: OUT / "records" / "p17" / f"{k['problem']}.json")
    for name in names:
        rec = recorded(store, {"problem": name, "method": "generic4-capped-audit", "tol": TOL}, case, name, dstore, limit=1800)
        log(f"{name}: worst {rec.get('worst')}, median speed-up {rec.get('median_speedup')}")
    summary = {r["key"]["problem"]: {k: r.get(k) for k in ("status", "lambda", "denominator", "denominator_us", "worst", "median_speedup", "columns")}
               for r in store.records()}  # fmt: skip
    (OUT / "p17_audit.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
