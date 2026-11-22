"""
Is the recipe limited by its data? The generic recipe (generic.py) of r4 with M extra one-step pairs: fresh draws from each problem's
initial law (after its burn-in, with the held inputs of a tracer's first step), each stepped once by the solver. M = 0 is r4 itself;
here M = 400 and 1,600 (4 and 16 times the training trajectories' starts), nested (the first 400 of the 1,600). One record per (problem,
M) under data/records/r5/, written when the case ends; a case is limited to 900 s. Usage: r5_data_scaling.py [problem ...].

Why the first step: in r4 the held-out failures sit at the trajectories' first states. A trajectory contributes one scattered state (its
start) and then states that the dynamics bring together, so a fold of 80 trajectories has about 80 independent inputs where a chart fit of
degree 3-4 in 6-9 coordinates has 84-715 terms, and Proposition 5.1's Lebesgue function explodes between them (p1: 2e3 to 4.5e11 at
degree 3-4 on the d = 10 ladders, whose training residuals at those degrees are 3e-4 to 3e-2).

Predictions, committed before the run (from p1 and r4's records, which use no extra data):
  D1  on the ladders whose degree-3 or degree-4 training residual is below 1e-3 in p1 (fisher_kpp_d10, allen_cahn_d10, burgers_d10), the
      held-out one-step error at the selected degree falls at least 10-fold from M = 0 to M = 1,600, and at least two of the three become
      served with a solver share below 50%;
  D2  on those ladders the selected degree rises with M, and the test rollout of every one served is within the tolerance;
  D3  lorenz96_d10 is not served at any M (held-out one-step 4.5e-2 at its best degree with 1,001 terms at degree 4: approximation and
      term count, not only sampling);
  D4  POLLU and Nelson (no validated reduction: k = 20 and 12) are not served at M = 1,600;
  D5  the low-dimensional controls fisher_kpp_d3 and lorenz96_d5 (solver only in r4) are served with a solver share below 50% at M = 400.
"""

import json
import sys
from pathlib import Path

import numpy as np
from generic import log, run
from r4_generic import OUT, kida

from src.benchmark.step_maps import STEP_MAPS, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import recorded  # noqa: E402

PROBLEMS = ("fisher_kpp_d3", "lorenz96_d5", "fisher_kpp_d10", "allen_cahn_d10", "burgers_d10", "bruss_d10", "lorenz96_d10",
            "fisher_kpp_d20", "allen_cahn_d20", "burgers_d20", "pollu", "nelson")  # fmt: skip
SIZES = (400, 1600)


def problem(name):
    if name.startswith("kida_"):
        return kida(name[5:])
    return (STEP_MAPS[name], *data(name))


def extra_pairs(name: str, P, m: int) -> tuple[np.ndarray, np.ndarray]:
    """The first m of 1,600 fresh one-step pairs from the initial law (cached in data/r5_extra_<name>.npz, not committed)."""
    path = OUT / f"r5_extra_{name}.npz"
    if not path.exists():
        n = max(SIZES)
        log(f"  generating {n} fresh one-step pairs for {name}")
        Y = P.initial(n, 500)
        if P.burn > 0:
            Y = P.system.step(Y, P.burn)
        Z0 = P.system.coords(Y)
        h = P.system.held
        if h and P.held_path is not None:
            Z0[:, -h:] = P.system.coords(P.held_path(n, 500, 1)[:, 0])
        Z1 = P.step(Z0)
        ok = np.all(np.isfinite(Z1), axis=1) & np.all(np.isfinite(Z0), axis=1)
        np.savez_compressed(path, Z0=Z0[ok], Z1=Z1[ok])
    z = np.load(path)
    return z["Z0"][:m], z["Z1"][:m]


def case(name: str, m: int) -> dict:
    P, train, test = problem(name)
    extra = extra_pairs(name, P, m)
    out = run(P, train, test, extra=extra, **({"kmax": 20, "qmax": 4} if name.startswith("kida_") else {}))
    return {"extra": int(len(extra[0])), **out}


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    store = RecordStore(OUT / "records" / "r5", path_of=lambda key: OUT / "records" / "r5" / f"{key['problem']}_m{key['extra']}.json")
    for name in names:
        for m in SIZES:
            recorded(store, {"problem": name, "method": "generic", "extra": m, "tol": 0.01}, case, name, m, limit=900)
    fields = ("status", "k", "degree", "served", "one_step_worst", "one_step_refused", "rollout_worst", "rollout_solver_share", "mixing", "seconds")
    summary = {f"{r['key']['problem']}_m{r['key']['extra']}": {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "r5_data_scaling.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
