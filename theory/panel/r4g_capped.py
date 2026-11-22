"""
The fourth recipe with the council's fourth-round gate (generic4.py, variant gate="capped", sketch=True, account=True; council4-fable.md,
section 2): a label-free cap b <= 3 sqrt(rank) (with the offset and the out-of-span share at their calibration ranges), the additive
envelope fitted by a linear program on the calibration states inside the cap, and every chain deployed with an account charged with a
typical-defect model (non-negative least squares) scaled by lambda, lambda the smallest of 0.5, 1, 1.5, 2, 3, 5, 8 for which at most 5% of
the calibration chains (plain, w = 0.01, w = 0.1) exceed tol / 2. The deployed fit (all training trajectories) is the one calibrated, on
100 fresh calibration trajectories (seed 11 of p13's draws). Two sizes: the benchmark's 100 training trajectories, and 340 (p13's seeds 12
and 13 added). On R4e's 18 problems at R4e's chart dimension. One record per (problem, size) under data/records/r4g/; a case is limited to
1,800 s (3,600 s for chemakzo). Usage: r4g_capped.py [problem ...].

Smoke tests before the predictions (fisher_kpp_d5, burgers_d20 at 100): both served (rollouts 1.1e-3 at 6.8% and 3.3% solver, chained
mixing at most 1.2e-3), lambda 0.5, 93% and 97% of calibration states inside the cap, no test trajectory with an accepted defect above
the envelope.

Predictions, committed before the run (council4-fable.md's H1-H6, as implemented here):
  H1  all ten of R4e's served problems are served at 100 trajectories, with their 40 test rollouts and chained mixing at both weights
      within tol / 2;
  H2  bruss_d20, fisher_kpp_d5 and fisher_kpp_d10 are served with a rollout solver share below 20% at 100 trajectories and below 8% at
      340;
  H3  burgers_d20's accepted one-step worst on the test inputs is below 2e-3;
  H4  on every served problem at most 8% of the test trajectories have an accepted one-step defect above the envelope;
  H5  hires is served on plain rollouts with under 15% solver, and its lambda (chosen with the mixing scripts) is at least 2.5 or
      infinite, its solver share under chained mixing at w = 0.01 above 80%;
  H6  lambda is at most 2 on the lattices, robertson, chemakzo and lorenz96_d4, and above 2 on allen_cahn_d5.
"""

import json
import sys
from pathlib import Path

import numpy as np
from generic4 import run
from p13_learning import extra_trajectories

from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
PROBLEMS = ("bruss_d2", "fisher_kpp_d2", "robertson", "fisher_kpp_d3", "oregonator", "van_der_pol", "lorenz96_d4", "allen_cahn_d5",
            "fisher_kpp_d5", "hires", "primordial", "nelson", "bruss_d20", "burgers_d20", "lorenz96_d20", "pollu", "fisher_kpp_d10", "chemakzo")  # fmt: skip
LIMITS = {"chemakzo": 3600}
VARIANT = {"gate": "capped", "sketch": True, "account": True}


def case(name: str, size: int, dstore: RecordStore) -> dict:
    P = STEP_MAPS[name]
    train, test = data(name)
    pool, _ = extra_trajectories(name, 360, dstore)  # seeds 11, 12, 13: the first 100 calibrate, the next 240 train at size 340
    cal, more = pool[:100], pool[120:360]
    if size > 100:
        train = np.concatenate([train, more[: size - 100]])
    ref = json.loads((OUT / "records" / "r4e" / f"{name}.json").read_text())
    out = run(P, train, test, name, solve_us=baseline_cost(name)["solve_us"], variant=VARIANT, k=(ref["k"], ref["decoder_degree"]), cal_trajs=cal)
    out["trajectories"], out["calibration_trajectories"] = int(len(train)), int(len(cal))
    return out


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    dstore = RecordStore(OUT / "records" / "p13_data", path_of=lambda k: OUT / "records" / "p13_data" / f"{k['problem']}_{k['chunk']}.json")
    store = RecordStore(OUT / "records" / "r4g", path_of=lambda k: OUT / "records" / "r4g" / f"{k['problem']}_{k['size']}.json")
    fields = ("status", "trajectories", "served", "degree", "rollout_worst", "rollout_solver_share", "refused_percent", "one_step_worst", "mixing_chained",
              "fresh_calibration", "trajectories_above_envelope_percent", "account_check", "evaluate_us", "speedup_effective", "train_s")  # fmt: skip
    for size in (100, 340):
        for name in names:
            rec = recorded(store, {"problem": name, "size": size, "method": "generic4-capped", "tol": 0.01}, case, name, size, dstore, limit=LIMITS.get(name, 1800))
            fc = rec.get("fresh_calibration") or {}
            log(f"{name} {size}: served={rec.get('served')} rollout={rec.get('rollout_worst')} solver={rec.get('rollout_solver_share')} lambda={fc.get('lambda')}")
    summary = {f"{r['key']['problem']}/{r['key']['size']}": {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "r4g_capped.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
