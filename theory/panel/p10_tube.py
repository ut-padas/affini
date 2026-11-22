"""
The tube experiment (council round 2, Fable's section 5; Sol's and Astra's tube with feasible split samples): can a surrogate serve the
inputs a coupled run produces, slightly off the trajectories' bundle, by owning the first-order term of the off-bundle defect? The fourth
recipe (generic4.py) in five arms at the chart dimension R4e chose (k and decoder degree from its record), on the reduced charts hires,
pollu, bruss_d20, burgers_d20, fisher_kpp_d10 and the full charts chemakzo and robertson:
  T0    the fourth recipe as in R4e;
  T0+   T0 with 1,600 fresh plain one-step pairs (r5_data_scaling.extra_pairs; the control for data volume);
  T1m   T0 with 1,600 of the fit trajectories' own mixed pairs (w = 0.01 and 0.1, solver-labelled) in the fit;
  T1    T1m with the off-chart coordinates as linear terms of the design (a constant normal term; Proposition 5.1 applies to the enlarged
        design, whose Lebesgue function says whether the tube was sampled in that direction);
  T2    T1m with the chart from a reduced-rank regression of successors on inputs (plain and mixed), the directions the step responds to.
Scores, beside generic4's: on the test trajectories' states mixed once (w = 0.01, 0.1; seed 9, solver-labelled), the refused share and
the accepted one-step defect (median, maximum, sign coherence); the closed column of p6 at D = 0.005, 0.05 and 0.25. One record per
(problem, arm) under data/records/p10/; a case is limited to 1,800 s. Usage: p10_tube.py [problem ...].

Predictions, committed before the run (Fable's, informed by its post hoc check on hires, so the first is a replication):
  U1  on hires T1 cuts the median accepted mixed defect at least 20x against T0 and leaves its maximum within a factor 3 of T0's;
  U2  on hires T1's chained mixing at w = 0.01 is at least 5x below T0's and T1 refuses at least 20 points less of the mixed inputs at
      w = 0.01;
  U3  T2 matches T1 within a factor 3 on hires's median accepted mixed defect;
  U4  T0+ lowers the median accepted mixed defect by less than 2x against T0 on every reduced-chart problem (plain pairs carry no
      normal information), and T1m by less than T1 does on hires;
  U5  on chemakzo and robertson (full charts) every arm's scores are within a factor 2 of T0's;
  U6  every served arm holds the tolerance in the column at all three exchanges.
"""

import json
import sys
from pathlib import Path

import numpy as np
from generic2 import mixed_once
from generic4 import rebuild, run, verified_laws
from p6_column import column
from r5_data_scaling import extra_pairs

from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
TOL = 0.01
PROBLEMS = ("hires", "robertson", "bruss_d20", "burgers_d20", "pollu", "fisher_kpp_d10", "chemakzo")
ARMS = {"T0": None, "T0+": None, "T1m": {"mixed": 1600}, "T1": {"mixed": 1600, "normal": True}, "T2": {"mixed": 1600, "chart": "sensitivity"}}


def mixed_scores(P, sur, test) -> dict:
    out = {}
    rng = np.random.default_rng(9)
    for w in (0.01, 0.1):
        X = mixed_once(P, test, w, rng)
        Y = P.step(X)
        nxt, r = sur.evaluate(X)
        acc = ~np.asarray(r, dtype=bool)
        res = {"refused_percent": float(100 * np.mean(~acc))}
        if acc.any():
            e = np.where(P.scored(Y[acc]), nxt[acc] - Y[acc], 0.0)
            m = np.max(np.abs(e), axis=1)
            res.update({"defect_median": float(np.median(m)), "defect_max": float(np.max(m)),
                        "coherence": float(np.max(np.abs(e.mean(0)) / np.maximum(np.abs(e).mean(0), 1e-300)))})  # fmt: skip
        out[str(w)] = res
    return out


def case(name: str, arm: str) -> dict:
    P = STEP_MAPS[name]
    train, test = data(name)
    ref = json.loads((OUT / "records" / "r4e" / f"{name}.json").read_text())
    variant = ARMS[arm]
    extra = extra_pairs(name, P, 1600) if arm == "T0+" else None
    out = run(P, train, test, name, extra=extra, solve_us=baseline_cost(name)["solve_us"], variant=variant, k=(ref["k"], ref["decoder_degree"]))
    sur = rebuild(P, train, out, extra, variant)
    out["mixed_inputs"] = mixed_scores(P, sur, test)
    W = verified_laws(P, train)
    out["columns"] = {str(D): {k: v for k, v in column(P, sur, test, D, W).items() if not k.endswith("_by_step")} for D in (0.005, 0.05, 0.25)}
    ms, cs = out["mixed_inputs"], out["columns"]
    log(f"  {name} {arm}: mixed defect median {ms['0.01'].get('defect_median', float('nan')):.1e} / {ms['0.1'].get('defect_median', float('nan')):.1e}, "
        f"max {ms['0.01'].get('defect_max', float('nan')):.1e} / {ms['0.1'].get('defect_max', float('nan')):.1e}, refused {ms['0.01']['refused_percent']:.0f} / "
        f"{ms['0.1']['refused_percent']:.0f}%; column {cs['0.005']['worst']:.1e} / {cs['0.05']['worst']:.1e} / {cs['0.25']['worst']:.1e}")  # fmt: skip
    return out


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    store = RecordStore(OUT / "records" / "p10", path_of=lambda key: OUT / "records" / "p10" / f"{key['problem']}_{key['arm']}.json")
    fields = ("status", "served", "degree", "rank", "rollout_worst", "refused_percent", "mixing_chained", "mixed_inputs", "columns", "evaluate_us",
              "speedup_effective", "speedup_effective_from_step_10", "budgets", "train_s")  # fmt: skip
    for name in names:
        for arm in ARMS:
            recorded(store, {"problem": name, "arm": arm, "tol": TOL}, case, name, arm, limit=1800)
    summary = {f"{r['key']['problem']}/{r['key']['arm']}": {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "p10_tube.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
