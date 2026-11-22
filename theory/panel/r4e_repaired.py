"""
The fourth recipe (generic4.py: the truncated-SVD fit, the bound and out-of-span term from one precomputed factor, one evaluation per batch,
the batched projection, selection by estimated cost, chains stopped once they cannot be served, laws verified off the data, consistent
calibration, the three budgets recorded) on R4c's 17 problems and fisher_kpp_d10, at tolerance 0.01, M = 0. One record per problem under
data/records/r4e/; a case is limited to 1,800 s (3,600 s for chemakzo). Usage: r4e_repaired.py [problem ...].

Smoke tests before the predictions (bruss_d2, robertson, fisher_kpp_d5): bruss_d2 served at degree 10 (rollout 3.3e-7); robertson at
degree 5 (rank 50 of 56, rollout 5.5e-5, evaluation 2.8 us, effective speed-up 388 against R4c's 32); fisher_kpp_d5 at degree 6 (rank 461
of 462, rollout 1.9e-3, 4.9% refused, evaluation 13.5 us, effective speed-up 15.7, 70 from step 10, against R4c's 1.3; 6 s against 651 s).

Predictions, committed before the run:
  F1  every problem R4c or r4d served is served, and no served problem's 40 test rollouts exceed the tolerance;
  F2  on R4c's three singular fits (chemakzo, fisher_kpp_d5, fisher_kpp_d10) the evaluation (prediction and gate) costs at most twice
      R4c's prediction alone, and the effective speed-up is at least 3x R4c's (r4d's for fisher_kpp_d10);
  F3  the median effective speed-up over the served problems is at least 20 (R4c's V5, failed at 14);
  F4  hires is served and its chained mixing at w = 0.01 still exceeds the tolerance (the repair does not touch accumulation); under the
      budget tol / (2 A_n) it would refuse more than 95% of its test inputs;
  F5  the verified laws: bruss_d20 none, burgers_d10 and burgers_d20 one each, pollu one fewer than R4c's six; robertson, hires,
      primordial and nelson unchanged;
  F6  under the budget tol / (2 A_n) the refused share exceeds 50% on every served problem with A_n above 100 and stays below 20% on the
      served lattices with A_n at most 40.
"""

import json
import sys
from pathlib import Path

from generic4 import run

from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
PROBLEMS = ("bruss_d2", "fisher_kpp_d2", "robertson", "fisher_kpp_d3", "oregonator", "van_der_pol", "lorenz96_d4", "allen_cahn_d5",
            "fisher_kpp_d5", "hires", "primordial", "nelson", "bruss_d20", "burgers_d20", "lorenz96_d20", "pollu", "fisher_kpp_d10", "chemakzo")  # fmt: skip
LIMITS = {"chemakzo": 3600}


def case(name: str) -> dict:
    train, test = data(name)
    return run(STEP_MAPS[name], train, test, name, solve_us=baseline_cost(name)["solve_us"])


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    store = RecordStore(OUT / "records" / "r4e", path_of=lambda key: OUT / "records" / "r4e" / f"{key['problem']}.json")
    fields = ("status", "d", "k", "degree", "rank", "terms", "laws", "served", "e_hat", "refused_percent", "rollout_worst", "rollout_solver_share",
              "rollout_solver_share_from_step_10", "mixing_chained", "evaluate_us", "solve_us", "speedup_effective", "speedup_effective_from_step_10",
              "budgets", "train_s")  # fmt: skip
    for name in names:
        rec = recorded(store, {"problem": name, "method": "generic4", "tol": 0.01}, case, name, limit=LIMITS.get(name, 1800))
        log(f"{name}: served={rec.get('served')} rollout={rec.get('rollout_worst')} refused={rec.get('refused_percent')} speedup={rec.get('speedup_effective')}")
    summary = {r["key"]["problem"]: {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "r4e_repaired.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
