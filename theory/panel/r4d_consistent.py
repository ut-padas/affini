"""
The third recipe with consistent calibration (generic3.py, ``consistent=True``) on R4c's reduced-chart problems, at tolerance 0.01, M = 0.
On a reduced chart the gate refuses when the bound's term or the decoder-residual term does; R4c calibrated the bound's e_hat on every
calibration state, so mixed states off a thin manifold, which the residual term refuses anyway, set it. Post hoc on HIRES (R4c's
configuration, before this run): e_hat 4.7e-9 on plain calibration states, 3.3e-3 on mixed ones, 1.9e-4 on the mixed states the residual
term accepts (a third of them); primordial and nelson have e_hat 3.3e-2 and 8.6e-2 on plain states already. Everything else as R4c.
One record per problem under data/records/r4d/; a case is limited to 1,800 s (3,600 s for chemakzo). Usage: r4d_consistent.py [problem ...].

Predictions, committed before the run:
  W1  hires is served, its 40 test rollouts within the tolerance, chained mixing within it at both weights, and a solver share under
      chained mixing above 50% at w = 0.1 (the residual term refuses the mixed states the calibration no longer counts);
  W2  every problem R4c served stays served, and no served problem's 40 test rollouts exceed the tolerance;
  W3  primordial and nelson stay not served (their fits are coarse on plain states: the change does not touch that);
  W4  pollu stays not served (R4c failed it on a chain of 1.25e-2, not on refusal);
  W5  on bruss_d20, burgers_d20 and fisher_kpp_d10 the refused share on the test inputs does not rise.
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
PROBLEMS = ("hires", "primordial", "nelson", "bruss_d20", "burgers_d20", "fisher_kpp_d10", "pollu", "chemakzo")
LIMITS = {"chemakzo": 3600}


def case(name: str) -> dict:
    train, test = data(name)
    return run(STEP_MAPS[name], train, test, solve_us=baseline_cost(name)["solve_us"], consistent=True)


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    store = RecordStore(OUT / "records" / "r4d", path_of=lambda key: OUT / "records" / "r4d" / f"{key['problem']}.json")
    fields = ("status", "d", "k", "degree", "served", "e_hat", "b_max", "refused_percent", "rollout_worst", "rollout_solver_share",
              "mixing_chained", "speedup_effective", "train_s")  # fmt: skip
    for name in names:
        rec = recorded(store, {"problem": name, "method": "generic3-consistent", "tol": 0.01}, case, name, limit=LIMITS.get(name, 1800))
        log(f"{name}: served={rec.get('served')} rollout={rec.get('rollout_worst')} refused={rec.get('refused_percent')} speedup={rec.get('speedup_effective')}")
    summary = {r["key"]["problem"]: {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "r4d_consistent.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
