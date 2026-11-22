"""
The fourth recipe with the council's third-round gate (generic4.py, variant gate="envelope", sketch=True, account=True): the additive
envelope a0 + a1 b + a2 rho + a3 rho^2 fitted by a linear program on the calibration states (plain and mixed), with the caps at their
ranges; the gate's two products sketched (32 columns); and every chain deployed with an error account per cell (a step is refused once
it would take the cell's sum of bounds past tol / 2). On R4e's 18 problems, 100 training trajectories, R4e's chart dimension. One record
per problem under data/records/r4f/; a case is limited to 1,800 s (3,600 s for chemakzo). Usage: r4f_envelope.py [problem ...].

Smoke tests before the predictions (hires, fisher_kpp_d3): hires not served (its plain rollouts refuse 0% one step at a time, but the
envelope's floor of 9.3e-5 per step spends the account within about 55 of 170 steps), chained mixing 9.9e-3 and 4.2e-3 at 96-97%
solver, no cell's error above its account; fisher_kpp_d3 served at degree 10, rollout 5.4e-3, the account exceeded on 0.94% of cells and
steps, evaluation 2.8 us.

Predictions, committed before the run:
  G1  no served problem's 40 test rollouts or chained mixing (both weights) exceed the tolerance;
  G2  the account holds (|e_n| <= S_n) on at least 99% of cells and steps of the plain rollouts on every served problem;
  G3  at least 10 problems are served (R4e's count), hires not among them;
  G4  the median effective speed-up over the served problems is at least 30 (R4e: 54; the account spends solver time);
  G5  on the problems whose fit has at least 300 terms the evaluation costs at most half of R4e's (the sketched gate).
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
VARIANT = {"gate": "envelope", "sketch": True, "account": True}


def case(name: str) -> dict:
    train, test = data(name)
    ref = json.loads((OUT / "records" / "r4e" / f"{name}.json").read_text())
    return run(STEP_MAPS[name], train, test, name, solve_us=baseline_cost(name)["solve_us"], variant=VARIANT, k=(ref["k"], ref["decoder_degree"]))


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    store = RecordStore(OUT / "records" / "r4f", path_of=lambda key: OUT / "records" / "r4f" / f"{key['problem']}.json")
    fields = ("status", "d", "k", "degree", "rank", "terms", "laws", "served", "e_hat", "refused_percent", "rollout_worst", "rollout_solver_share",
              "rollout_solver_share_from_step_10", "mixing_chained", "evaluate_us", "solve_us", "speedup_effective", "speedup_effective_from_step_10",
              "budgets", "account_check", "envelope", "train_s")  # fmt: skip
    for name in names:
        rec = recorded(store, {"problem": name, "method": "generic4-envelope-account", "tol": 0.01}, case, name, limit=LIMITS.get(name, 1800))
        log(f"{name}: served={rec.get('served')} rollout={rec.get('rollout_worst')} refused={rec.get('refused_percent')} speedup={rec.get('speedup_effective')}")
    summary = {r["key"]["problem"]: {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "r4f_envelope.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
