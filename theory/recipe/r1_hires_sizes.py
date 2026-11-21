"""
The recipe (common.py) on HIRES with fresh trajectories: three training seeds (11, 12, 13), each with n = 25, 50, 100, 200 and 400
training trajectories (the first n of 400), against one fresh test set of 40 (seed 21), all generated like the benchmark's (its initial
law, burn-in, step and horizon) and cached in data/hires_<seed>_<n>.npz. One record per (seed, n) under data/records/r1/, written when
the case ends; a case is limited to 120 s. g5 used the benchmark's single split (100 training, 40 test trajectories).

Predictions, committed before the run (tolerance 0.01 dex):
  H1  at n = 100 the rule passes and its fit holds the tolerance on all 40 fresh test rollouts in at least 2 of the 3 seeds;
  H2  the median over seeds of the test rollout error falls at least 10-fold from n = 25 to n = 400, and the selected degree does not fall;
  H3  the number of degrees whose training chains diverge (above 1 dex) falls with n: at n = 400 at most one, in every seed;
  H4  at n = 25 the rule fails (no degree within the tolerance on the training chains) in at least 2 of the 3 seeds.
"""

import json

import numpy as np
from common import OUT, log, recorded, run_recipe

from src.benchmark.step_maps import STEP_MAPS
from src.infra.records import RecordStore
from src.problems.ode.system import trajectories

P = STEP_MAPS["hires"]
SIZES = (25, 50, 100, 200, 400)


def generated(n: int, seed: int) -> np.ndarray:
    path = OUT / f"hires_{seed}_{n}.npz"
    if path.exists():
        return np.load(path)["Z"]
    log(f"generating {n} HIRES trajectories, seed {seed}")
    Y = P.initial(n, seed)
    if P.burn > 0:
        Y = P.system.step(Y, P.burn)
    Z = trajectories(P.system, Y, P.dt, P.steps)
    np.savez_compressed(path, Z=Z)
    return Z


def main() -> None:
    OUT.mkdir(exist_ok=True)
    test = generated(40, 21)
    store = RecordStore(OUT / "records" / "r1", path_of=lambda key: OUT / "records" / "r1" / f"hires_s{key['seed']}_n{key['n']}.json")
    for seed in (11, 12, 13):
        pool = generated(max(SIZES), seed)
        for n in SIZES:
            rec = recorded(store, {"problem": "hires", "seed": seed, "n": n, "method": "recipe", "tol": 0.01}, run_recipe, P, pool[:n], test)
            if rec.get("status") == "done":
                diverged = sum(r["train_chains_worst"] > 1 for r in rec["degree_curve"])
                log(f"seed {seed} n={n}: k={rec['k']} q={rec['degree']} passes={rec['passes']} rollout {rec['rollout_worst']:.3e} diverged degrees {diverged}")
    fields = ("status", "k", "degree", "passes", "one_step_worst", "rollout_worst", "mixing_rollout_worst", "solves", "seconds")
    summary = {f"s{r['key']['seed']}_n{r['key']['n']}": {**{k: r.get(k) for k in fields},
               "diverged_degrees": sum(c["train_chains_worst"] > 1 for c in r.get("degree_curve", []))} for r in store.records()}  # fmt: skip
    (OUT / "r1_hires_sizes.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
