"""
The generic recipe (generic.py) on every step map of the benchmark and on KIDA uva 2024 in its two environment ranges (the trajectories
of theory/recipe/r3_kida.py, when complete), at tolerance 0.01. One record per problem under data/records/r4/, written when the case
ends; a case is limited to 900 s and recorded as failed past it. Usage: r4_generic.py [problem ...] (default: every problem, smallest
input dimension first, KIDA last).

Problem classes, from Proposition 5.1's computable factor before any R4 outcome: Lambda_max, the largest Lebesgue function over the
test inputs of R2's chart fit at degree 3 (p1_descriptors.json; test inputs only, no test value). covered (Lambda_max <= 100):
allen_cahn_d2, bruss_d2, fisher_kpp_d2, van_der_pol, allen_cahn_d3, fisher_kpp_d3, oregonator, robertson, bruss_d4, lorenz96_d4,
lorenz96_d5, bruss_d6, hires, bruss_d10, bruss_d20, lorenz96_d20, bruss_d40, lorenz96_d40, bruss_d80 (the last five only because R2's
unvalidated fallback chose k = 1-3, so their approximation factor is large); sparse (Lambda_max > 100): allen_cahn_d5, chemakzo,
fisher_kpp_d5, allen_cahn_d10, burgers_d10, fisher_kpp_d10, lorenz96_d10, nelson, allen_cahn_d20, burgers_d20, fisher_kpp_d20, pollu,
allen_cahn_d40, burgers_d40, fisher_kpp_d40, allen_cahn_d80, fisher_kpp_d80; primordial unclassified (R2 failed).

Predictions, committed before the run (a correctness smoke test on bruss_d2 preceded them: served at degree 6, rollout 6.5e-4):
  G1  the recipe serves (some configuration's held-out chains within the tolerance) at most 2 of the 17 sparse problems;
  G2  it serves at least 8 of the 19 covered problems, among them bruss_d2, fisher_kpp_d2, allen_cahn_d2, robertson and hires, and
      serves neither oregonator nor van_der_pol (no reduction, accumulation A_n of 2,700 and 3e7);
  G3  where it serves, the test rollout over all test trajectories is within the tolerance in at least 80% of the problems: held-out
      selection generalises where in-sample selection did not (R2);
  G4  where it serves, the rollout solver share is below 10% in at least half the problems;
  G5  with refusal honoured, the mixing rollouts are within the tolerance wherever the recipe serves, and at w = 0.1 the solver share
      exceeds the unmixed rollout's by at least 10 percentage points in most of them (mixed states leave the samples' cover);
  G6  the minimum-norm projection removes R2's singular-matrix failure on primordial (the case finishes);
  G7  Nelson and both KIDA ranges (driven environments, four held inputs as chart coordinates) are not served at 0.01.
"""

import json
import sys
from pathlib import Path

from generic import log, run

from src.benchmark.step_maps import STEP_MAPS, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
R3_DATA = Path(__file__).resolve().parents[1] / "recipe" / "data" / "records" / "r3_data"


def kida(name: str):
    """(problem, train, test) of a KIDA range from r3's chunk records, or None while they are incomplete."""
    import numpy as np
    import r3_kida

    parts = {}
    for role, n in (("train", r3_kida.N_TRAIN), ("test", r3_kida.N_TEST)):
        chunks = []
        for c in range(n // r3_kida.CHUNK + (n % r3_kida.CHUNK > 0)):
            f = R3_DATA / f"kida_{name}_{role}_{c}.npz"
            if not f.exists():
                return None
            chunks.append(np.load(f)["Z"])
        parts[role] = np.concatenate(chunks)[:n]
    return r3_kida.problem(name), parts["train"], parts["test"]


def case(name: str) -> dict:
    if name.startswith("kida_"):
        P, train, test = kida(name[5:])
        return run(P, train, test, kmax=20, qmax=4)
    train, test = data(name)
    return run(STEP_MAPS[name], train, test)


def main() -> None:
    names = sys.argv[1:] or [*sorted(STEP_MAPS, key=lambda n: (data(n)[0].shape[2], n)), "kida_diffuse", "kida_dense"]
    store = RecordStore(OUT / "records" / "r4", path_of=lambda key: OUT / "records" / "r4" / f"{key['problem']}.json")
    fields = ("status", "d", "k", "reduction", "degree", "ridge", "served", "one_step_worst", "one_step_refused", "rollout_worst",
              "rollout_solver_share", "rollout_worst_20", "mixing", "seconds")  # fmt: skip
    for name in names:
        if name.startswith("kida_") and kida(name[5:]) is None:
            log(f"{name}: data not complete, skipped")
            continue
        rec = recorded(store, {"problem": name, "method": "generic", "tol": 0.01}, case, name, limit=900)
        log(f"{name}: {json.dumps({k: rec.get(k) for k in fields}, default=str)}")
    summary = {r["key"]["problem"]: {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "r4_generic.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
