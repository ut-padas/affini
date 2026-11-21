"""
The recipe (common.py) on every step-map problem of the benchmark, at tolerance 0.01, on the benchmark's own cached trajectories,
beside the best finished benchmark record of each problem at that tolerance (read only). One record per problem under
data/records/r2/, written when the case ends; a case is limited to 120 s (180 s above 15 dimensions) and recorded as failed past it.
Usage: r2_step_maps.py [problem ...] (default: all, smallest input dimension first).

Predictions, committed before the run (HIRES is informed by g5 and the smoke test of common.py, which reproduced its 1.5e-4):
  S1  where the visited set is low-dimensional against d (HIRES, POLLU, CHEMAKZO, and the stiff ladders bruss, allen_cahn and
      fisher_kpp at d >= 20), the chosen k is at most d / 2 and the rule's fit holds the tolerance on all test rollouts, with no solve
      beyond the k check;
  S2  on Lorenz-96 at d >= 10 (non-stiff, chaotic) and on the Burgers ladder the rule fails: no degree holds the tolerance on the
      training chains;
  S3  where the rule passes and a finished benchmark record exists, the recipe's test rollout is at most the record's, with fewer
      solves than the record's evaluations;
  S4  the mixing rollout at w = 0.1 fails the tolerance on every problem where the rule passes, and at w = 0.01 on most of them (a
      value-only fit has no control off the visited set);
  S5  Nelson (four held inputs as chart coordinates, 1,440 training pairs) is data-limited: its degree is capped at 5 or less by the
      1.5-pairs rule and its rule fails.
"""

import json
import sys

from common import OUT, best_record, log, recorded, run_recipe

from src.benchmark.step_maps import STEP_MAPS, data
from src.infra.records import RecordStore


def case(name: str) -> dict:
    P = STEP_MAPS[name]
    train, test = data(name)
    return {"benchmark_best": best_record(name), **run_recipe(P, train, test)}


def main() -> None:
    names = sys.argv[1:] or sorted(STEP_MAPS, key=lambda n: (data(n)[0].shape[2], n))
    store = RecordStore(OUT / "records" / "r2", path_of=lambda key: OUT / "records" / "r2" / f"{key['problem']}.json")
    fields = ("status", "d", "k", "degree", "passes", "one_step_worst", "rollout_worst", "mixing_rollout_worst", "solves", "seconds", "benchmark_best")
    for name in names:
        d = data(name)[0].shape[2]
        rec = recorded(store, {"problem": name, "method": "recipe", "tol": 0.01}, case, name, limit=180 if d > 15 else 120)
        log(f"{name}: {json.dumps({k: rec.get(k) for k in fields}, default=str)}")
    summary = {r["key"]["problem"]: {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "r2_step_maps.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
