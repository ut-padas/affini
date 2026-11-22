"""
A basis for larger charts (council round 2: Fable's C4 and step 5): R4e leaves POLLU (k = 13), Nelson (12 coordinates and 4 held inputs),
primordial (6 and 4) and lorenz96_d20 (20) unserved with total-degree fits at degrees 1-3, the largest the term budget allows. The fourth
recipe (generic4.py) at R4e's chart dimension with three index sets at each level q of the selection:
  total       total degree q (R4e);
  hyperbolic  the hyperbolic cross prod_j (alpha_j + 1) <= 2^q: higher degrees per coordinate, fewer high-order interactions;
  adaptive    a downward-closed set of total degree q's size, grown greedily by correlation with the fit's residuals (each degree at most
              min(3q, 12)).
On the four and on three served controls (bruss_d20, fisher_kpp_d10, burgers_d20). The selection, gate and scores are generic4's; the
records keep each level's held-out calibration worst. One record per (problem, basis) under data/records/p12/; a case is limited to
1,800 s. Usage: p12_basis.py [problem ...]. A trajectory-fitted tensor train (SYNTHESIS 9.6, T-a) is the next arm, not run here.

Predictions, committed before the run:
  B1  on at least 2 of POLLU, Nelson, primordial and lorenz96_d20 the adaptive set's best held-out calibration worst is at least 2x
      below total degree's, and it serves at most one of the four;
  B2  on the controls every basis that serves holds its 40 test rollouts within the tolerance, and adaptive serves wherever total degree
      does;
  B3  on lorenz96_d20 (chaotic, strongly coupled) the hyperbolic cross's best held-out calibration worst is above total degree's.
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
PROBLEMS = ("pollu", "nelson", "primordial", "lorenz96_d20", "bruss_d20", "fisher_kpp_d10", "burgers_d20")
BASES = ("total", "hyperbolic", "adaptive")


def case(name: str, basis: str) -> dict:
    P = STEP_MAPS[name]
    train, test = data(name)
    ref = json.loads((OUT / "records" / "r4e" / f"{name}.json").read_text())
    out = run(P, train, test, name, solve_us=baseline_cost(name)["solve_us"], variant={"basis": basis}, k=(ref["k"], ref["decoder_degree"]))
    out["best_heldout_worst"] = min(r["heldout_worst"] for r in out["selection"])
    log(f"  {name} {basis}: best held-out worst {out['best_heldout_worst']:.2e} at levels {[r['degree'] for r in out['selection']]}, served {out['served']}")
    return out


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    store = RecordStore(OUT / "records" / "p12", path_of=lambda key: OUT / "records" / "p12" / f"{key['problem']}_{key['basis']}.json")
    fields = ("status", "served", "degree", "terms", "rank", "best_heldout_worst", "rollout_worst", "refused_percent", "mixing_chained", "evaluate_us",
              "speedup_effective", "train_s")  # fmt: skip
    for name in names:
        for basis in BASES:
            recorded(store, {"problem": name, "basis": basis, "tol": 0.01}, case, name, basis, limit=1800)
    summary = {f"{r['key']['problem']}/{r['key']['basis']}": {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "p12_basis.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
