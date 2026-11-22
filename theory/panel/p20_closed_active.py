"""
Refusal-driven data in closed columns, iterated (council round 5: Fable's second action, Sol's matched-budget acquisition, Astra's
freeze-and-audit). On the full-chart problems R4g serves at 340 trajectories, does training on the coupled inputs the deployed surrogate
refuses cut the closed-column refusal rate several-fold, more than the same number of labels on coupled inputs drawn uniformly?

Start: R4g's surrogate at 340 training and 100 calibration trajectories. Each round deploys the current surrogate (with its account) in
closed columns on a pool of 100 trajectories (seed 15), sorted and unsorted, D = 0.005, 0.05 and 0.25, and collects the inputs:
  refused  the inputs the surrogate refused, with the solver's step that deployment computed anyway;
  uniform  the same number of inputs drawn uniformly from all the columns' inputs, labelled by the solver (the control at equal cost).
Each arm adds its pairs to its own training set and refits the whole recipe (selection, fit, fresh calibration); three rounds. After each
round the surrogate is audited on p17's independent columns (seed 14, never fitted or calibrated on): the worst error, the solver share,
and the speed-up against the fastest exact solver (p15). One record per (problem, arm, round) under data/records/p20/; a case is limited to
1,800 s. Usage: p20_closed_active.py [problem ...].

Predictions, committed before the run:
  C1  chemakzo's audit solver share falls from 96-100% (p17) to below 22% after one refused round and below 5% after three, every
      audit column within tol / 2 (Fable's prediction);
  C2  after three rounds the refused arm's audit solver share is below the uniform arm's on at least 6 of the 8 problems;
  C3  every served round holds tol / 2 in all six audit columns;
  C4  the median audit speed-up over the problems after three refused rounds is at least 1.5x their p17 median.
"""

import json
import sys
from pathlib import Path

import numpy as np
from generic4 import Account, calibrate_fresh, rebuild, run
from p13_learning import extra_trajectories
from p17_audit import column

from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
TOL = 0.01
VARIANT = {"gate": "capped", "sketch": True, "account": True}
PROBLEMS = ("chemakzo", "fisher_kpp_d2", "fisher_kpp_d3", "fisher_kpp_d5", "robertson", "lorenz96_d4", "bruss_d2", "allen_cahn_d5")
ROUNDS = 3


def collect(P, sur, pool, rng):
    """Closed columns on ``pool`` (both orders, three exchanges): the refused (input, solver step) pairs, and all inputs seen."""
    h = P.system.held
    free = pool.shape[2] - h
    lin = (lambda Z: 10.0**Z) if P.system.log else (lambda Z: Z)
    back = (lambda Y: np.log10(np.maximum(Y, 1e-300))) if P.system.log else (lambda Y: Y)
    RX, RY, ALL = [], [], []
    for order in ("sorted", "unsorted"):
        X0 = pool[:, 0, :free] - pool[:, 0, :free].mean(0)
        perm = np.argsort(X0 @ np.linalg.svd(X0, full_matrices=False)[2][0]) if order == "sorted" else rng.permutation(len(pool))
        true = pool[perm]
        for D in (0.005, 0.05, 0.25):
            pol = Account(sur)
            Zs = true[:, 0].copy()
            for k in range(true.shape[1] - 1):
                X = lin(Zs[:, :free])
                Xp = np.concatenate([X[:1], X, X[-1:]])
                Zin = Zs.copy()
                Zin[:, :free] = back((1 - 2 * D) * X + D * Xp[:-2] + D * Xp[2:])
                if h:
                    Zin[:, free:] = true[:, k, free:]
                r = np.asarray(pol.refuses(Zin), dtype=bool)
                nxt = Zin + pol.predict(Zin)
                if r.any():
                    nxt[r] = P.step(Zin[r])
                    RX.append(Zin[r]), RY.append(nxt[r])
                ALL.append(Zin)
                Zs = nxt
    d = pool.shape[2]
    RX = np.concatenate(RX) if RX else np.empty((0, d))
    RY = np.concatenate(RY) if RY else np.empty((0, d))
    ok = np.all(np.isfinite(RX), axis=1) & np.all(np.isfinite(RY), axis=1)
    return RX[ok], RY[ok], np.concatenate(ALL)


def audit(P, sur, cells, den_us) -> dict:
    cols = {}
    for order in ("sorted", "unsorted"):
        for D in (0.005, 0.05, 0.25):
            ev = []
            worst, share, refused = column(P, lambda: Account(sur), cells, D, order, ev)
            cs = len(cells) * (cells.shape[1] - 1)
            sp = (cs * den_us) / (1e6 * sum(ev) + refused * den_us) if den_us else None
            cols[f"{order}/{D}"] = {"worst": worst, "solver_share": share, "speedup": sp}
    return {"columns": cols, "worst": max(c["worst"] for c in cols.values()), "solver_share": float(np.mean([c["solver_share"] for c in cols.values()])),
            "median_speedup": float(np.median([c["speedup"] for c in cols.values() if c["speedup"]])) if den_us else None}  # fmt: skip


def case(name: str, arm: str, dstore: RecordStore) -> dict:
    P = STEP_MAPS[name]
    train, test = data(name)
    trajs, _ = extra_trajectories(name, 600, dstore)  # seeds 11-15: 11 calibrates, 12-13 train, 14 audits, 15 is the pool
    cal, more, cells, pool = trajs[:100], trajs[120:360], trajs[360:460], trajs[480:580]
    train = np.concatenate([train, more[:240]])
    ref = json.loads((OUT / "records" / "r4e" / f"{name}.json").read_text())
    rec = json.loads((OUT / "records" / "r4g" / f"{name}_340.json").read_text())
    den_us = json.loads((OUT / "p15_denominators.json").read_text()).get(name, {}).get("denominator_us")
    sur = rebuild(P, train, rec, None, VARIANT)
    calibrate_fresh(P, sur, cal)
    extra_X, extra_Y = np.empty((0, train.shape[2])), np.empty((0, train.shape[2]))
    rounds = []
    rng = np.random.default_rng(3)
    for r in range(ROUNDS):
        RX, RY, ALL = collect(P, sur, pool, np.random.default_rng(11 + r))
        m = len(RX)
        if arm == "uniform" and m:
            pick = rng.choice(len(ALL), min(m, len(ALL)), replace=False)
            RX = ALL[pick]
            RY = P.step(RX)
            ok = np.all(np.isfinite(RY), axis=1)
            RX, RY = RX[ok], RY[ok]
        extra_X, extra_Y = np.concatenate([extra_X, RX]), np.concatenate([extra_Y, RY])
        out = run(P, train, test, name, extra=(extra_X, extra_Y) if len(extra_X) else None, solve_us=baseline_cost(name)["solve_us"], variant=VARIANT,
                  k=(ref["k"], ref["decoder_degree"]), cal_trajs=cal)  # fmt: skip
        sur = rebuild(P, train, out, (extra_X, extra_Y) if len(extra_X) else None, VARIANT)
        calibrate_fresh(P, sur, cal)
        a = audit(P, sur, cells, den_us)
        rounds.append({"round": r + 1, "new_pairs": int(len(RX)), "extra_pairs": int(len(extra_X)), "served": out["served"], "degree": out["degree"],
                       "rollout_worst": out["rollout_worst"], "rollout_solver_share": out["rollout_solver_share"], "audit": a})  # fmt: skip
        log(f"  {name} {arm} round {r + 1}: +{len(RX)} pairs, served {out['served']}, audit worst {a['worst']:.2e}, solver {100 * a['solver_share']:.1f}%, "
            f"speed-up {a['median_speedup'] or float('nan'):.1f}")  # fmt: skip
    return {"rounds": rounds}


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    dstore = RecordStore(OUT / "records" / "p13_data", path_of=lambda k: OUT / "records" / "p13_data" / f"{k['problem']}_{k['chunk']}.json")
    store = RecordStore(OUT / "records" / "p20", path_of=lambda k: OUT / "records" / "p20" / f"{k['problem']}_{k['arm']}.json")
    for name in names:
        for arm in ("refused", "uniform"):
            recorded(store, {"problem": name, "arm": arm, "rounds": ROUNDS, "tol": TOL}, case, name, arm, dstore, limit=3600)
    summary = {f"{r['key']['problem']}/{r['key']['arm']}": {"status": r.get("status"), "rounds": [{k: v for k, v in x.items() if k != "audit"} | {
        "audit_worst": x["audit"]["worst"], "audit_solver_share": x["audit"]["solver_share"], "audit_speedup": x["audit"]["median_speedup"]} for x in r.get("rounds", [])]}
        for r in store.records()}  # fmt: skip
    (OUT / "p20_closed_active.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
