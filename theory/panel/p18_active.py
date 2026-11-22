"""
Refusal-driven data (the refusal rate caps the speed-up at 1/rho, Corollary 5.3): does training on the states the deployed surrogate
refuses lower its refusal rate more than the same number of solves spent on fresh trajectories? Generic: no problem-specific choice.

For every problem of R4g at 100 training trajectories (generic4, the capped gate with the chain-calibrated account; the same 100
calibration trajectories), the deployed surrogate is run, plain and under chained mixing (w = 0.01), on 100 pool trajectories (p13's
seed 15, never fitted, calibrated on or tested); every refused input is stepped by the solver, as deployment does, and those pairs (input,
solver's next state) are the refusal-driven data, at no solve beyond deployment's own. Two arms then refit the whole recipe (selection,
fit, fresh calibration) with the same number of extra one-step pairs:
  refused  the refused pairs;
  random   pairs drawn uniformly from fresh trajectories (seed 16), one solve each, the same count (the control at equal solver cost).
Scored on the 40 test trajectories as R4g (rollouts, chained mixing, the account), with the refusal rate (the rollout's solver share) the
quantity of interest. One record per (problem, arm) under data/records/p18/; a case is limited to 1,800 s. Usage: p18_active.py [problem ...].

Predictions, committed before the run:
  R1  on the problems R4g serves at 100 with a rollout solver share of at least 3%, the refused arm's share is at least 2x lower than
      R4g's on at least half of them, and lower than the random arm's on at least two thirds;
  R2  the refused arm serves at least one problem R4g does not serve at 100 (fisher_kpp_d10, allen_cahn_d5, pollu, lorenz96_d20 or
      hires at under 50% solver);
  R3  no served arm's 40 test rollouts or chained mixing exceed tol / 2;
  R4  nelson, primordial and the two oscillators stay unserved in both arms (their refusals are a bias or jump limit, not a data limit).
"""

import json
import sys
from pathlib import Path

import numpy as np
from generic4 import Account, calibrate_fresh, rebuild, run
from p13_learning import extra_trajectories

from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
VARIANT = {"gate": "capped", "sketch": True, "account": True}
PROBLEMS = ("bruss_d2", "fisher_kpp_d2", "robertson", "fisher_kpp_d3", "lorenz96_d4", "fisher_kpp_d5", "chemakzo", "bruss_d20", "burgers_d20",
            "fisher_kpp_d10", "allen_cahn_d5", "pollu", "lorenz96_d20", "hires", "nelson", "primordial", "oregonator", "van_der_pol")  # fmt: skip


def refused_pairs(P, sur, trajs, w, rng) -> tuple[np.ndarray, np.ndarray]:
    """The (input, solver step) pairs a deployment of ``sur`` (with its account) refuses along ``trajs``, plain (w = 0) or mixed."""
    h = P.system.held
    free = trajs.shape[2] - h
    n = len(trajs)
    pol = Account(sur)
    lin = (lambda Z: 10.0**Z) if P.system.log else (lambda Z: Z)
    back = (lambda Y: np.log10(np.maximum(Y, 1e-300))) if P.system.log else (lambda Y: Y)
    partners = np.array([[rng.choice([j for j in range(n) if j != i]) for i in range(n)] for _ in range(trajs.shape[1] - 1)]) if w else None
    Z = trajs[:, 0].copy()
    Xs, Ys = [], []
    for k in range(trajs.shape[1] - 1):
        Zin = Z.copy()
        if w:
            Zin[:, :free] = back((1 - w) * lin(Z[:, :free]) + w * lin(trajs[partners[k], k, :free]))
        if h:
            Zin[:, free:] = trajs[:, k, free:]
        r = np.asarray(pol.refuses(Zin), dtype=bool)
        nxt = Zin + pol.predict(Zin)
        if r.any():
            nxt[r] = P.step(Zin[r])
            Xs.append(Zin[r]), Ys.append(nxt[r])
        Z = nxt
    if not Xs:
        return np.empty((0, trajs.shape[2])), np.empty((0, trajs.shape[2]))
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    ok = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    return X[ok], Y[ok]


def pool_data(name: str, dstore: RecordStore) -> dict:
    """The refused pairs of R4g's deployed surrogate at 100 trajectories on the pool (seed 15), kept as arrays with the record."""
    P = STEP_MAPS[name]
    train, _ = data(name)
    trajs, _ = extra_trajectories(name, 600, dstore)  # seeds 11-15: 11 calibrates, 15 is the pool
    cal, pool = trajs[:100], trajs[480:580]
    rec = json.loads((OUT / "records" / "r4g" / f"{name}_100.json").read_text())
    sur = rebuild(P, train, rec, None, VARIANT)
    calibrate_fresh(P, sur, cal)
    Xa, Ya = refused_pairs(P, sur, pool, 0.0, np.random.default_rng(5))
    Xb, Yb = refused_pairs(P, sur, pool, 0.01, np.random.default_rng(6))
    X, Y = np.concatenate([Xa, Xb]), np.concatenate([Ya, Yb])
    return {"refused_plain": int(len(Xa)), "refused_mixed": int(len(Xb)), "pool_steps": int(len(pool) * (pool.shape[1] - 1)), "_arrays": {"X": X, "Y": Y}}


def case(name: str, arm: str, dstore: RecordStore, pstore: RecordStore) -> dict:
    P = STEP_MAPS[name]
    train, test = data(name)
    trajs, _ = extra_trajectories(name, 720, dstore)  # seed 16 (the sixth chunk) gives the random arm's pairs
    cal = trajs[:100]
    prec = recorded(pstore, {"problem": name, "pool": "seed15", "method": "r4g-100"}, pool_data, name, dstore, limit=1800)
    if prec.get("status") != "done":
        raise RuntimeError(f"pool data failed: {prec.get('failure', {}).get('message')}")
    z = np.load(pstore.path(prec["key"]).with_suffix(".npz"))
    X, Y = z["X"], z["Y"]
    m = len(X)
    if arm == "random":
        rand = trajs[600:720]
        Xr, Yr = rand[:, :-1].reshape(-1, rand.shape[2]), rand[:, 1:].reshape(-1, rand.shape[2])
        pick = np.random.default_rng(8).choice(len(Xr), min(m, len(Xr)), replace=False)
        X, Y = Xr[pick], Yr[pick]
    ref = json.loads((OUT / "records" / "r4e" / f"{name}.json").read_text())
    extra = (X, Y) if len(X) else None
    out = run(P, train, test, name, extra=extra, solve_us=baseline_cost(name)["solve_us"], variant=VARIANT, k=(ref["k"], ref["decoder_degree"]), cal_trajs=cal)
    out["extra_pairs"] = int(len(X))
    out["pool"] = {k: prec[k] for k in ("refused_plain", "refused_mixed", "pool_steps")}
    log(f"  {name} {arm}: {len(X)} extra pairs, served {out['served']}, rollout solver share {100 * out['rollout_solver_share']:.1f}%, rollout {out['rollout_worst']:.2e}")
    return out


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    dstore = RecordStore(OUT / "records" / "p13_data", path_of=lambda k: OUT / "records" / "p13_data" / f"{k['problem']}_{k['chunk']}.json")
    pstore = RecordStore(OUT / "records" / "p18_pool", path_of=lambda k: OUT / "records" / "p18_pool" / f"{k['problem']}.json")
    store = RecordStore(OUT / "records" / "p18", path_of=lambda k: OUT / "records" / "p18" / f"{k['problem']}_{k['arm']}.json")
    fields = ("status", "extra_pairs", "pool", "served", "degree", "rollout_worst", "rollout_solver_share", "mixing_chained", "fresh_calibration", "speedup_effective")
    for name in names:
        for arm in ("refused", "random"):
            recorded(store, {"problem": name, "arm": arm, "tol": 0.01}, case, name, arm, dstore, pstore, limit=1800)
    summary = {f"{r['key']['problem']}/{r['key']['arm']}": {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "p18_active.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
