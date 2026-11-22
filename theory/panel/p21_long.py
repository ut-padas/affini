"""
Long chains (council round 5: Fable's first action). Is the refusal rate a start-up transient, and does the account, which never forgets,
refuse long chains for good? The 11 problems R4g serves at 340 trajectories other than hires and pollu (which fail p17's closed
columns), each surrogate rebuilt exactly as R4g deployed it (340
training, 100 calibration trajectories, the capped gate, lambda), on 64 new trajectories of 2,000 steps (seed 40; the benchmark's own
initial law, burn-in and driving) in three arms:
  gate     the capped gate alone, no account;
  account  the present account (charges summed forever);
  decay    the account with geometric decay gamma, the smallest of 0.5, 0.8, 0.9, 0.95, 0.99 for which at most 5% of 40 calibration
           chains of 400 steps (seed 41) exceed tol / 2 anywhere (1 if none does).
Plain chains over the 2,000 steps (the solver steps refused inputs), and a closed column (p17's, sorted, D = 0.05) over the first 400.
Scores per arm: the refusal rate by step and by cause (gate, account); the fresh-start rate (steps 1-10) and the steady rate (steps 11
onward) with its 95% upper bound (3 / n at zero refusals in n cell-steps); the certified horizon (the first step whose worst error exceeds
tol / 2, or none); the worst error; the effective speed-up against the problem's fastest exact solver (p15) at the steady rate, with the
evaluation timed at 1,000 cells per call. One record per problem under data/records/p21/; a case is limited to 1,800 s. Usage:
p21_long.py [problem ...].

Predictions, committed before the run:
  L1  with the present account the solver share over steps 200-2,000 of the plain chains exceeds 50% on at least 5 of the problems
      (budget exhaustion), while the gate alone refuses at most 1% of steps from step 11 on at least 8;
  L2  the gate alone holds the tolerance over the 2,000 steps on at least 7 problems (lorenz96_d4, chaotic, excepted from the count);
  L3  the decaying account holds tol / 2 over the 2,000 steps wherever the gate alone does, with a steady solver share below 2%;
  L4  the gate alone's steady refusal rate is bounded below 1e-4 on at least 6 problems.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from generic4 import Account, calibrate_fresh, rebuild
from p13_learning import extra_trajectories
from p17_audit import column

import src.problems.ode as ode
from src.benchmark.step_maps import STEP_MAPS, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
TOL = 0.01
VARIANT = {"gate": "capped", "sketch": True, "account": True}
DECAYS = (0.5, 0.8, 0.9, 0.95, 0.99)
PROBLEMS = ("bruss_d2", "fisher_kpp_d2", "fisher_kpp_d3", "fisher_kpp_d5", "robertson", "lorenz96_d4", "chemakzo", "allen_cahn_d5", "bruss_d20",
            "burgers_d20", "fisher_kpp_d10")  # fmt: skip: R4g's served problems at 340 but hires and pollu (which fail closed columns)


def long_trajectories(name: str, n: int, steps: int, seed: int) -> dict:
    p = STEP_MAPS[name]
    t0 = time.perf_counter()
    Y = p.initial(n, seed)
    if p.burn > 0:
        Y = p.system.step(Y, p.burn)
    path = p.held_path(n, seed, steps) if p.held_path else None
    Z = ode.trajectories(p.system, Y, p.dt, steps, path)
    ok = np.all(np.isfinite(Z), axis=(1, 2))
    return {"solver_s": time.perf_counter() - t0, "kept": int(ok.sum()), "_arrays": {"Z": Z[ok]}}


def chain(P, pol, true) -> dict:
    """A plain chain with fallback under ``pol``; per step the refusal shares by cause and the worst error."""
    h = P.system.held
    Z = true[:, 0].copy()
    gate, acct, worst = [], [], []
    cell_worst = np.zeros(len(true))
    for k in range(true.shape[1] - 1):
        if h:
            Z[:, -h:] = true[:, k, -h:]
        r = np.asarray(pol.refuses(Z), dtype=bool)
        g = getattr(pol, "by_gate", None)
        gate.append(float((g if g is not None else r).mean()))
        a = getattr(pol, "by_account", None)
        acct.append(float(a.mean()) if a is not None else 0.0)
        nxt = Z + pol.predict(Z)
        if r.any():
            nxt[r] = P.step(Z[r])
        Z = nxt
        e = np.where(P.scored(true[:, k + 1]), np.abs(Z - true[:, k + 1]), 0.0)
        em = np.where(np.isfinite(e), e, np.inf).max(axis=1)
        cell_worst = np.maximum(cell_worst, em)
        worst.append(float(em.max()))
    return {"gate_by_step": gate, "account_by_step": acct, "worst_by_step": worst, "cell_worst": cell_worst}


def summarise(c: dict, n_cells: int, den_us, ev_us) -> dict:
    tot = np.array(c["gate_by_step"]) + np.array(c["account_by_step"])
    steady = tot[10:]
    n = n_cells * len(steady)
    rate = float(steady.mean()) if len(steady) else 0.0
    over = [i + 1 for i, w in enumerate(c["worst_by_step"]) if w > TOL / 2]
    out = {"fresh_rate": float(tot[:10].mean()), "steady_rate": rate, "steady_upper_95": max(rate, 3.0 / n) if rate == 0 else rate + 2 * np.sqrt(rate / n),
           "share_200_on": float(tot[200:].mean()) if len(tot) > 200 else None, "worst": max(c["worst_by_step"]),
           "certified_horizon": over[0] if over else None, "account_share": float(np.mean(c["account_by_step"]))}  # fmt: skip
    if den_us and ev_us:
        out["speedup_steady"] = den_us / (ev_us + rate * den_us)
    return out


def case(name: str, dstore: RecordStore, lstore: RecordStore) -> dict:
    P = STEP_MAPS[name]
    train, _ = data(name)
    pool, _ = extra_trajectories(name, 360, dstore)
    cal, more = pool[:100], pool[120:360]
    train = np.concatenate([train, more[:240]])
    rec = json.loads((OUT / "records" / "r4g" / f"{name}_340.json").read_text())
    sur = rebuild(P, train, rec, None, VARIANT)
    calibrate_fresh(P, sur, cal)
    trajs = {}
    for label, n, steps, seed in (("test", 64, 2000, 40), ("cal", 40, 400, 41)):
        lr = recorded(lstore, {"problem": name, "set": label, "n": n, "steps": steps, "seed": seed}, long_trajectories, name, n, steps, seed, limit=3600)
        if lr.get("status") != "done":
            raise RuntimeError(f"long trajectories {label}: {lr.get('failure', {}).get('message')}")
        trajs[label] = np.load(lstore.path(lr["key"]).with_suffix(".npz"))["Z"]
    test, lcal = trajs["test"], trajs["cal"]
    gamma = 1.0
    for g in DECAYS:
        c = chain(P, Account(sur, decay=g), lcal)
        if np.mean(c["cell_worst"] > TOL / 2) <= 0.05:
            gamma = g
            break
    den = json.loads((OUT / "p15_denominators.json").read_text()).get(name, {})
    den_us = den.get("denominator_us")
    X = test[:, :-1].reshape(-1, test.shape[2])[:1000]
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        sur.evaluate(X, cache=False)
        ts.append(time.perf_counter() - t0)
    ev_us = 1e6 * float(np.median(ts)) / len(X)
    out = {"lambda": sur.lam, "gamma": gamma, "cells": int(len(test)), "steps": int(test.shape[1] - 1), "evaluate_us_1000": ev_us,
           "denominator": den.get("denominator"), "denominator_us": den_us, "arms": {}}  # fmt: skip
    for arm, pol in (("gate", lambda: sur), ("account", lambda: Account(sur)), ("decay", lambda: Account(sur, decay=gamma))):
        c = chain(P, pol(), test)
        out["arms"][arm] = {**summarise(c, len(test), den_us, ev_us), **{k: c[k] for k in ("gate_by_step", "account_by_step")},
                            "worst_by_step_every_10": c["worst_by_step"][::10]}  # fmt: skip
        a = out["arms"][arm]
        cw, cs, _ = column(P, pol, test[:, :401], 0.05, "sorted", [])
        a["closed_column_400"] = {"worst": cw, "solver_share": cs}
        log(f"  {name} {arm}: fresh {100 * a['fresh_rate']:.1f}%, steady {100 * a['steady_rate']:.3f}% (95% <= {a['steady_upper_95']:.1e}), "
            f"from step 200 {100 * (a['share_200_on'] or 0):.1f}%, worst {a['worst']:.2e}, horizon {a['certified_horizon']}, speed-up {a.get('speedup_steady', float('nan')):.1f}; closed column {cw:.2e} at {100 * cs:.1f}%")  # fmt: skip
    return out


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    dstore = RecordStore(OUT / "records" / "p13_data", path_of=lambda k: OUT / "records" / "p13_data" / f"{k['problem']}_{k['chunk']}.json")
    lstore = RecordStore(OUT / "records" / "p21_data", path_of=lambda k: OUT / "records" / "p21_data" / f"{k['problem']}_{k['set']}.json")
    store = RecordStore(OUT / "records" / "p21", path_of=lambda k: OUT / "records" / "p21" / f"{k['problem']}.json")
    log(f"long chains on {names}")
    for name in names:
        recorded(store, {"problem": name, "method": "generic4-capped-long", "tol": TOL}, case, name, dstore, lstore, limit=1800)
    summary = {r["key"]["problem"]: {"status": r.get("status"), "gamma": r.get("gamma"), "lambda": r.get("lambda"), "evaluate_us_1000": r.get("evaluate_us_1000"),
               "denominator_us": r.get("denominator_us"), "arms": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_step") and not kk.endswith("every_10")}
                                                                    for k, v in (r.get("arms") or {}).items()}} for r in store.records()}  # fmt: skip
    (OUT / "p21_long.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
