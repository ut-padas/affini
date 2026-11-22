"""
Robustness of "served" (council round 4: Fable's point that POLLU and Allen-Cahn d5 flip at the margin). R4g's recipe (generic4,
gate="capped", sketch, account) at R4g's chart dimension, trained on 340 trajectories and calibrated on 100, in three independent draws
(seed blocks 20-23, 24-27 and 28-31 of p13's generator, none used before), on the problems served near the margin: pollu, allen_cahn_d5,
fisher_kpp_d3, fisher_kpp_d5, fisher_kpp_d10. Scored on the benchmark's 40 test trajectories as R4g. One record per (problem, draw) under
data/records/p19/; a case is limited to 1,800 s. Usage: p19_seeds.py [problem ...].

Predictions, committed before the run:
  S1  every problem's served decision is the same in all three draws and the same as R4g's at 340;
  S2  every served draw's 40 test rollouts and chained mixing are within tol / 2;
  S3  on each problem the rollout solver share varies by less than 2x across the three draws.
"""

import json
import sys
from pathlib import Path

from generic4 import run
from p13_learning import extra_trajectories

from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
PROBLEMS = ("pollu", "allen_cahn_d5", "fisher_kpp_d3", "fisher_kpp_d5", "fisher_kpp_d10")
VARIANT = {"gate": "capped", "sketch": True, "account": True}


def case(name: str, draw: int, dstore: RecordStore) -> dict:
    P = STEP_MAPS[name]
    _, test = data(name)
    trajs, _ = extra_trajectories(name, 21 * 120, dstore)  # chunks 0-20 (seeds 11-31); draw d uses chunks 9 + 4d .. 12 + 4d
    counts = [json.loads((OUT / "records" / "p13_data" / f"{name}_{c}.json").read_text())["kept"] for c in range(21)]
    start = sum(counts[: 9 + 4 * draw])
    block = trajs[start : start + sum(counts[9 + 4 * draw : 13 + 4 * draw])]
    train, cal = block[:340], block[-100:]
    ref = json.loads((OUT / "records" / "r4e" / f"{name}.json").read_text())
    out = run(P, train, test, name, solve_us=baseline_cost(name)["solve_us"], variant=VARIANT, k=(ref["k"], ref["decoder_degree"]), cal_trajs=cal)
    out["trajectories"], out["calibration_trajectories"], out["draw"] = int(len(train)), int(len(cal)), draw
    log(f"  {name} draw {draw}: served {out['served']}, rollout {out['rollout_worst']:.2e}, solver {100 * out['rollout_solver_share']:.1f}%")
    return out


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    dstore = RecordStore(OUT / "records" / "p13_data", path_of=lambda k: OUT / "records" / "p13_data" / f"{k['problem']}_{k['chunk']}.json")
    store = RecordStore(OUT / "records" / "p19", path_of=lambda k: OUT / "records" / "p19" / f"{k['problem']}_{k['draw']}.json")
    fields = ("status", "trajectories", "served", "degree", "rollout_worst", "rollout_solver_share", "mixing_chained", "fresh_calibration", "speedup_effective")
    for name in names:
        for draw in range(3):
            recorded(store, {"problem": name, "draw": draw, "tol": 0.01}, case, name, draw, dstore, limit=1800)
    summary = {f"{r['key']['problem']}/{r['key']['draw']}": {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "p19_seeds.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
