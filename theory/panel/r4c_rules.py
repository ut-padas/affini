"""
The generic recipe's third version (generic3.py: the gate calibrated for the maximum on states set apart for it, mixed inputs in the
calibration, a propagation margin in the selection, k up to the free dimension) on the chemistry problems and on the cases that discriminate
R4b's failures, at tolerance 0.01, M = 0. One record per problem under data/records/r4c/; a case is limited to 1,800 s (900 s for the
first eight, run before generic3's chain shortcut).
Usage: r4c_rules.py [problem ...].

The problems: the chemistry and oscillator class (robertson, chemakzo, hires, pollu, primordial, nelson, oregonator, van_der_pol); R4b's
four selection-test misses (fisher_kpp_d3, chemakzo, fisher_kpp_d5, lorenz96_d20) and full-chart mixing failures (fisher_kpp_d2, the same);
a propagation failure (allen_cahn_d5); controls served cleanly in R4b (bruss_d2, lorenz96_d4, burgers_d20, bruss_d20).

Predictions, committed before the run (a correctness smoke test on bruss_d2 preceded them: served at degree 8, rollout 9.3e-6, chained
mixing 2.7e-3, nothing refused):
  V1  no served problem misses the tolerance on its 40 (or 20) test rollouts: R4b's four misses are either served within it or not served;
  V2  every served problem holds the tolerance under chained mixing at w = 0.01 and 0.1, except possibly hires (reduced chart, where
      chained error grows through accepted steps);
  V3  at least 9 of the 17 problems are served, among them the four controls, robertson, hires and fisher_kpp_d2;
  V4  pollu, nelson, oregonator, van_der_pol and primordial are not served (k up to the free dimension does not help without more data);
  V5  the median effective speed-up over the served problems is at least 20.
"""

import json
import sys
from pathlib import Path

from generic3 import run

from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
PROBLEMS = ("bruss_d2", "fisher_kpp_d2", "robertson", "fisher_kpp_d3", "oregonator", "van_der_pol", "lorenz96_d4", "allen_cahn_d5",
            "chemakzo", "fisher_kpp_d5", "hires", "primordial", "nelson", "bruss_d20", "burgers_d20", "lorenz96_d20", "pollu")  # fmt: skip


def case(name: str) -> dict:
    train, test = data(name)
    return run(STEP_MAPS[name], train, test, solve_us=baseline_cost(name)["solve_us"])


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    store = RecordStore(OUT / "records" / "r4c", path_of=lambda key: OUT / "records" / "r4c" / f"{key['problem']}.json")
    fields = ("status", "d", "k", "reduction", "degree", "served", "e_hat", "b_max", "one_step_worst", "refused_percent", "rollout_worst",
              "rollout_worst_20", "rollout_solver_share", "mixing_chained", "mixing_chained_without_residual", "predict_us", "gate_us", "solve_us",
              "speedup_nominal", "speedup_effective", "train_s")  # fmt: skip
    for name in names:
        rec = recorded(store, {"problem": name, "method": "generic3", "tol": 0.01}, case, name, limit=1800)
        log(f"{name}: served={rec.get('served')} rollout={rec.get('rollout_worst')} refused={rec.get('refused_percent')} speedup={rec.get('speedup_effective')}")
    summary = {r["key"]["problem"]: {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "r4c_rules.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
