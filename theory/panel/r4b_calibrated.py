"""
The generic recipe's second version (generic2.py: a held-out-calibrated gate, the residual term, "served" fixed before the run, the
per-factor diagnosis) on every step map of the benchmark and on KIDA uva 2024 (dense range; diffuse when its trajectories are complete),
at tolerance 0.01, with no extra data (M = 0). One record per problem under data/records/r4b/, written when the case ends; a case is
limited to 900 s (2,400 s for KIDA). Usage: r4b_calibrated.py [problem ...] (default: every problem, smallest input dimension first).

Predictions, committed before the run. They are informed by review round 10's post hoc checks (which scored a similar calibrated
threshold on nine problems' test trajectories) and by R4's records; no run of generic2 preceded them except a correctness smoke test on
bruss_d2 (served at degree 7).
  C1  "served" (the definition of generic2) on at least 12 of the 37 step maps, among them allen_cahn_d2, bruss_d2, lorenz96_d4,
      lorenz96_d5, fisher_kpp_d2, bruss_d4, burgers_d10, allen_cahn_d3, chemakzo and hires;
  C2  every served problem holds the tolerance over all 40 (or 20) test rollouts;
  C3  on the served problems with k < free, the chained mixing rollouts with the residual term stay within the tolerance at w = 0.01 and
      0.1, and on hires the run without the term exceeds it (the blind spot);
  C4  not served: oregonator, van_der_pol, pollu, nelson, allen_cahn_d10 and both KIDA ranges (propagation, no reduction, or both);
  C5  the median rollout solver share over the served problems is below 5%, and the median effective speed-up over them is above 20.
"""

import json
import sys
from pathlib import Path

from generic2 import log, run

from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
import r3_kida  # noqa: E402,F401 -- registers the KIDA system before any solver pool is forked
from common import recorded  # noqa: E402
from r4_generic import kida  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"


def case(name: str) -> dict:
    if name.startswith("kida_"):
        P, train, test = kida(name[5:])
        return run(P, train, test, kmax=20, qmax=4)
    train, test = data(name)
    return run(STEP_MAPS[name], train, test, solve_us=baseline_cost(name)["solve_us"])


def main() -> None:
    names = sys.argv[1:] or [*sorted(STEP_MAPS, key=lambda n: (data(n)[0].shape[2], n)), "kida_dense", "kida_diffuse"]
    store = RecordStore(OUT / "records" / "r4b", path_of=lambda key: OUT / "records" / "r4b" / f"{key['problem']}.json")
    fields = ("status", "d", "k", "reduction", "degree", "served", "stopped_at", "e_hat", "b_max", "lift_floor", "one_step_worst", "refused_percent",
              "rollout_worst", "rollout_solver_share", "mixing_chained", "mixing_chained_without_residual", "mixing_once", "predict_us", "gate_us",
              "solve_us", "speedup_nominal", "speedup_effective", "train_s")  # fmt: skip
    for name in names:
        if name.startswith("kida_") and kida(name[5:]) is None:
            log(f"{name}: data not complete, skipped")
            continue
        rec = recorded(store, {"problem": name, "method": "generic2", "tol": 0.01}, case, name, limit=2400 if name.startswith("kida_") else 900)
        log(f"{name}: {json.dumps({k: rec.get(k) for k in fields if not k.startswith('mixing')}, default=str)}")
    summary = {r["key"]["problem"]: {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "r4b_calibrated.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
