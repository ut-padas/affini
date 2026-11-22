"""
A fair comparison of the chart recipe with AFINNI, ART and rollout-tuned ChebTT (theory/COMPARISON_PLAN.md), at one tolerance per run
(THEORY_TOL: 0.05 or 0.01). Every method trains on the same data and is scored by the same protocol with the same refusal definition.

Data: the benchmark's 100 training and 40 test trajectories (src.benchmark.step_maps.data). The box methods take their box from those
training states and query the solver inside it. The recipe also uses 100 calibration trajectories (p13's seed 11), and its best version
340 training trajectories (seeds 12-13 added). All solver calls are counted.

Methods, each in its best version, value-only (no method uses gradients):
  recipe_100, recipe_340  generic4 with the capped gate and a decaying account (gamma 0.5), at R4e's chart dimension;
  chebtt_roll             as the benchmark builds it (ChebTT retrained until its training chains hold tol / 2);
  afinni_roll, art_roll, art_quad_roll   the benchmark's AFINNI, ART (affine leaves), ART (quadratic leaves), wrapped in the same
                          propagation-margin loop as chebtt_roll (retrained at a tighter one-step tolerance until the chains along the
                          training trajectories hold tol / 2, at most three rounds).
Refusal, the same for every method: a cell-step handed to the solver, whatever the reason. Outside the box for every box method; outside
every trust region for AFINNI (its "pout"); the gate or the account for the recipe.

Protocol, the same for every method:
  - the 40 test rollouts: the worst error, and the refusal rate over steps 1-10 and from step 11;
  - chained mixing at w = 0.01 and 0.1;
  - p17's fresh closed columns (100 trajectories of seed 14, sorted and unsorted, D = 0.005, 0.05, 0.25);
  - p21's long chains (64 trajectories of 2,000 steps, seed 40): the start-up and steady refusal, the worst error, the certified horizon;
  - the evaluation (prediction and gate) timed at 1,000 cells per call; the effective speed-up against the problem's fastest exact solver
    (p15) at the long chains' steady refusal rate;
  - the training time and solver calls.
One record per (problem, method, tolerance) under data/records/p25/; a case is limited to BUDGET = 900 s of compute on this laptop (a
timeout is recorded as a result). The first cases ran under a 2,400 s limit; a record that took longer than BUDGET is reported as over the
budget, so every method is judged by the same compute budget.
Usage: THEORY_TOL=0.05 p25_compare.py [problem ...].

Prediction, committed before the run (the user's): at both tolerances AFINNI and both ARTs fare worse than chebtt_roll and the recipe:
on the problems all methods finish, their worst rollout error is higher or they need more solver calls, and they time out on more problems.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from generic import TOL, mixing
from generic4 import Account, calibrate_fresh, rebuild, run
from p13_learning import extra_trajectories
from p17_audit import column
from p21_long import chain, long_trajectories, summarise

from src import to_host, xp
from src.benchmark.step_maps import METHODS, STEP_MAPS, baseline_cost, data, rollout
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
VARIANT = {"gate": "capped", "sketch": True, "account": True}
PROBLEMS = ("bruss_d2", "robertson", "fisher_kpp_d3", "lorenz96_d4", "chemakzo", "allen_cahn_d5", "fisher_kpp_d5", "fisher_kpp_d10", "burgers_d20", "pollu")
METHOD_NAMES = ("recipe_100", "recipe_340", "chebtt_roll", "afinni_roll", "art_roll", "art_quad_roll")
GAMMA = 0.5
BUDGET = 900


class Native:
    """A box method as deployed: the benchmark's model, refusing outside its box and, for AFINNI, outside every trust region."""

    def __init__(self, model, afinni: bool):
        self.model, self.afinni, self.evals = model, afinni, model.evals

    def predict(self, Z):
        return self.model.predict(Z)

    def refuses(self, Z):
        r = np.asarray(self.model.refuses(Z), dtype=bool)
        if self.afinni:
            Q = np.clip(Z, self.model.lo, self.model.hi)  # AFINNI's inputs are the whole state, held inputs included
            out = self.model.models.inner.predict(xp.asarray(Q))
            r |= np.asarray(to_host(out[1]), dtype=bool).reshape(len(Z))
        return r


def rolled(P, train, name: str):
    """The benchmark's box method ``name`` retrained at a tighter one-step tolerance until its training chains hold tol / 2."""
    one, evals, rounds = TOL, 0, 0
    for _ in range(3):
        rounds += 1
        model = Native(METHODS[name][2](P, train, one, 1), name == "afinni")
        evals += model.evals
        worst = rollout(P, model, train)["rollout_worst"]
        if worst <= 0.5 * TOL:
            break
        one *= 0.8 * max(0.5 * TOL / worst, 0.125)
    model.evals, model.one_step_tol, model.rounds = evals, one, rounds
    return model


def build(P, name: str, method: str, dstore):
    train, test = data(name)
    if method.startswith("recipe"):
        pool, _ = extra_trajectories(name, 360, dstore)
        cal = pool[:100]
        if method == "recipe_340":
            train = np.concatenate([train, pool[120:360]])
        ref = json.loads((OUT / "records" / "r4e" / f"{name}.json").read_text())
        out = run(P, train, test, name, solve_us=baseline_cost(name)["solve_us"], variant=VARIANT, k=(ref["k"], ref["decoder_degree"]), cal_trajs=cal)
        sur = rebuild(P, train, out, None, VARIANT)
        calibrate_fresh(P, sur, cal)
        # every solve the recipe paid: its trajectories' steps (training and calibration) and the labels of the mixed calibration states
        steps = (len(train) + len(cal)) * (train.shape[1] - 1)
        calls = steps + 2 * len(cal) * (train.shape[1] - 1)
        return sur, lambda: Account(sur, decay=GAMMA), {"served": out["served"], "degree": out["degree"], "solver_calls": int(calls)}
    if method == "chebtt_roll":
        model = METHODS["chebtt_roll"][2](P, train, TOL, 1)
        return model, lambda: model, {"solver_calls": int(model.evals), "one_step_tol": getattr(model, "one_step_tol", None)}
    model = rolled(P, train, method.removesuffix("_roll"))
    return model, lambda: model, {"solver_calls": int(model.evals), "one_step_tol": model.one_step_tol, "margin_rounds": model.rounds}


def case(name: str, method: str, dstore, lstore) -> dict:
    P = STEP_MAPS[name]
    _, test = data(name)
    t0 = time.perf_counter()
    model, pol, info = build(P, name, method, dstore)
    out = {"tol": TOL, "train_s": time.perf_counter() - t0, **info}
    roll = rollout(P, pol(), test)
    by = np.asarray(roll["rollout_solved_by_step"])
    out.update({"rollout_worst": float(roll["rollout_worst"]), "rollout_fresh_rate": float(by[:10].mean()), "rollout_steady_rate": float(by[10:].mean())})
    out["mixing"] = {str(w): mixing(P, pol(), test, w, np.random.default_rng(1)) for w in (0.01, 0.1)}
    fresh, _ = extra_trajectories(name, 480, dstore)
    cells = fresh[360:460]
    den_us = json.loads((OUT / "p15_denominators.json").read_text()).get(name, {}).get("denominator_us")
    cols = {}
    for order in ("sorted", "unsorted"):
        for D in (0.005, 0.05, 0.25):
            w, sh, _ = column(P, pol, cells, D, order, [])
            cols[f"{order}/{D}"] = {"worst": w, "solver_share": sh}
    out["columns"] = cols
    lr = recorded(lstore, {"problem": name, "set": "test", "n": 64, "steps": 2000, "seed": 40}, long_trajectories, name, 64, 2000, 40, limit=3600)
    long_test = np.load(lstore.path(lr["key"]).with_suffix(".npz"))["Z"]
    X = long_test[:, :-1].reshape(-1, long_test.shape[2])[:1000]
    p0 = pol()
    ts = []
    for _ in range(3):
        Xc = X.copy()
        t1 = time.perf_counter()
        p0.refuses(Xc)
        p0.predict(Xc)
        ts.append(time.perf_counter() - t1)
    ev_us = 1e6 * float(np.median(ts)) / len(X)
    c = chain(P, pol(), long_test)
    out["long"] = summarise(c, len(long_test), den_us, ev_us)
    out["evaluate_us_1000"], out["denominator_us"] = ev_us, den_us
    log(f"  {name} {method} tol {TOL}: rollout {out['rollout_worst']:.2e}, columns worst {max(v['worst'] for v in cols.values()):.2e} "
        f"(solver {100 * np.mean([v['solver_share'] for v in cols.values()]):.1f}%), long steady {100 * out['long']['steady_rate']:.3f}%, "
        f"speed-up {out['long'].get('speedup_steady', float('nan')):.1f}, calls {out['solver_calls']}, {out['train_s']:.0f} s")  # fmt: skip
    return out


def main() -> None:
    names = [a for a in sys.argv[1:] if a in STEP_MAPS] or list(PROBLEMS)
    methods = [a for a in sys.argv[1:] if a in METHOD_NAMES] or list(METHOD_NAMES)
    dstore = RecordStore(OUT / "records" / "p13_data", path_of=lambda k: OUT / "records" / "p13_data" / f"{k['problem']}_{k['chunk']}.json")
    lstore = RecordStore(OUT / "records" / "p21_data", path_of=lambda k: OUT / "records" / "p21_data" / f"{k['problem']}_{k['set']}.json")
    store = RecordStore(OUT / "records" / "p25", path_of=lambda k: OUT / "records" / "p25" / f"{k['problem']}_{k['method']}_{k['tol']}.json")
    for name in names:
        for method in methods:
            recorded(store, {"problem": name, "method": method, "tol": TOL}, case, name, method, dstore, lstore, limit=BUDGET)
    summary = {f"{r['key']['problem']}/{r['key']['method']}/{r['key']['tol']}": {k: v for k, v in r.items() if k not in ("key", "provenance", "schema", "recorded")}
               for r in store.records()}  # fmt: skip
    (OUT / f"p25_compare_{TOL}.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
