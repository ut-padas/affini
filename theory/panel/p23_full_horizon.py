"""
Full-horizon training (p22's finding that 400-step training trajectories leave robertson and chemakzo refusing 73-81% of a 2,000-step
horizon). As p22, with the 40 long training trajectories spanning the whole 2,000-step horizon (seed 43) instead of 400 steps.

Coverage in time (p21's finding: robertson and chemakzo refuse 88-96% of their steady steps because their long-time states lie beyond the
40-step training trajectories). R4g's recipe (generic4, the capped gate, decaying account) at 340 training and 100 calibration
trajectories, plus the one-step pairs of 40 long training trajectories (400 steps, seed 42; the benchmark's initial law and driving), on
robertson and chemakzo, with bruss_d20 as a control (served in steady state already). Scored on p21's 64 test chains of 2,000 steps
(seed 40) with the gate alone and with the decaying account (p21's gamma), as p21. One record per problem under data/records/p22/; a case
is limited to 3,600 s. Usage: p23_full_horizon.py [problem ...].

Prediction, committed before the run: with training spanning the horizon, robertson's and chemakzo's steady refusal rate (steps
11-2,000, gate alone) falls below 5% (from 73-81% with 400-step training), every chain within tol / 2; bruss_d20's stays below 1e-3.
"""

import json
import sys
from pathlib import Path

import numpy as np
from generic4 import Account, calibrate_fresh, rebuild, run
from p13_learning import extra_trajectories
from p21_long import chain, long_trajectories, summarise

from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
VARIANT = {"gate": "capped", "sketch": True, "account": True}
PROBLEMS = ("robertson", "chemakzo", "bruss_d20")


def case(name: str, dstore: RecordStore, lstore: RecordStore) -> dict:
    P = STEP_MAPS[name]
    train, test = data(name)
    pool, _ = extra_trajectories(name, 360, dstore)
    cal, more = pool[:100], pool[120:360]
    train = np.concatenate([train, more[:240]])
    sets = {}
    for label, n, steps, seed in (("test", 64, 2000, 40), ("train_full", 40, 2000, 43)):
        lr = recorded(lstore, {"problem": name, "set": label, "n": n, "steps": steps, "seed": seed}, long_trajectories, name, n, steps, seed, limit=3600)
        if lr.get("status") != "done":
            raise RuntimeError(f"long trajectories {label}: {lr.get('failure', {}).get('message')}")
        sets[label] = np.load(lstore.path(lr["key"]).with_suffix(".npz"))["Z"]
    L = sets["train_full"]
    extra = (L[:, :-1].reshape(-1, L.shape[2]), L[:, 1:].reshape(-1, L.shape[2]))
    ref = json.loads((OUT / "records" / "r4e" / f"{name}.json").read_text())
    out = run(P, train, test, name, extra=extra, solve_us=baseline_cost(name)["solve_us"], variant=VARIANT, k=(ref["k"], ref["decoder_degree"]), cal_trajs=cal)
    sur = rebuild(P, train, out, extra, VARIANT)
    calibrate_fresh(P, sur, cal)
    gamma = json.loads((OUT / "records" / "p21" / f"{name}.json").read_text()).get("gamma", 0.5)
    den = json.loads((OUT / "p15_denominators.json").read_text()).get(name, {})
    p21 = json.loads((OUT / "records" / "p21" / f"{name}.json").read_text())
    res = {"extra_pairs": int(len(extra[0])), "served": out["served"], "degree": out["degree"], "gamma": gamma, "arms": {}}
    for arm, pol in (("gate", lambda: sur), ("decay", lambda: Account(sur, decay=gamma))):
        c = chain(P, pol(), sets["test"])
        a = summarise(c, len(sets["test"]), den.get("denominator_us"), p21.get("evaluate_us_1000"))
        res["arms"][arm] = a
        log(f"  {name} {arm}: steady {100 * a['steady_rate']:.3f}% (p21 {100 * p21['arms'][arm]['steady_rate']:.1f}%), worst {a['worst']:.2e}, horizon {a['certified_horizon']}")
    return res


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    dstore = RecordStore(OUT / "records" / "p13_data", path_of=lambda k: OUT / "records" / "p13_data" / f"{k['problem']}_{k['chunk']}.json")
    lstore = RecordStore(OUT / "records" / "p21_data", path_of=lambda k: OUT / "records" / "p21_data" / f"{k['problem']}_{k['set']}.json")
    store = RecordStore(OUT / "records" / "p23", path_of=lambda k: OUT / "records" / "p23" / f"{k['problem']}.json")
    for name in names:
        recorded(store, {"problem": name, "method": "generic4-capped-full-horizon", "tol": 0.01}, case, name, dstore, lstore, limit=3600)
    summary = {r["key"]["problem"]: {k: r.get(k) for k in ("status", "extra_pairs", "served", "degree", "gamma", "arms")} for r in store.records()}
    (OUT / "p23_full_horizon.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
