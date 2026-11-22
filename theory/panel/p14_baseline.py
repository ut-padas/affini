"""
A faster exact baseline (council round 3: Fable's and Sol's point that every effective speed-up so far is quoted against SciPy's
solve_ivp called per state, whose cost is mostly Python): the time per state and step of a compiled integrator, SciPy's VODE (BDF with
the system's analytic Jacobian, Fortran; the same rtol and atol as the system's own integrator), on 40 test states at steps 0, 10 and 20
of every problem R4e serves and on the unserved chemistry, against the benchmark's baseline (src.benchmark.step_maps.baseline_cost),
with VODE's agreement with the stored next states (the worst scored error). R4e's effective speed-ups are restated against it. Writes
data/p14_baseline.json. Serial, one core.

Prediction, committed before the run: VODE is at least 3x faster than the benchmark's baseline on every problem with fewer than 25
components, its states agree with the stored ones within 1e-4 (dex or absolute), and R4e's median effective speed-up over its served
problems falls below 15 when restated against VODE.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import ode

from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data

OUT = Path(__file__).resolve().parent / "data"
PROBLEMS = ("bruss_d2", "fisher_kpp_d2", "fisher_kpp_d3", "fisher_kpp_d5", "robertson", "lorenz96_d4", "chemakzo", "bruss_d20", "burgers_d20",
            "fisher_kpp_d10", "hires", "pollu", "nelson", "primordial")  # fmt: skip


def vode_step(system, y, dt):
    r = ode(lambda t, v: system.rhs(v), lambda t, v: system.jac(v))
    r.set_integrator("vode", method="bdf", rtol=system.rtol, atol=system.atol, nsteps=100000, with_jacobian=True)
    r.set_initial_value(y, 0.0)
    r.integrate(dt)
    return r.y if r.successful() else np.full_like(y, np.nan)


def main() -> None:
    out = {}
    for name in PROBLEMS:
        P = STEP_MAPS[name]
        _, test = data(name)
        steps = [k for k in (0, 10, 20) if k < test.shape[1] - 1]
        X = np.concatenate([test[:, k] for k in steps])
        Y = np.concatenate([test[:, k + 1] for k in steps])
        Ys = 10.0**X if P.system.log else X
        t0 = time.perf_counter()
        nxt = np.array([vode_step(P.system, y, P.dt) for y in Ys])
        us = 1e6 * (time.perf_counter() - t0) / len(Ys)
        Z = P.system.coords(nxt)
        err = np.where(P.scored(Y), np.abs(Z - Y), 0.0)
        worst = float(np.max(err)) if np.all(np.isfinite(err)) else float("inf")
        base = baseline_cost(name)["solve_us"]
        rec = OUT / "records" / "r4e" / f"{name}.json"
        r = json.loads(rec.read_text()) if rec.exists() else {}
        res = {"vode_us": us, "baseline_us": base, "ratio": base / us, "worst_vs_stored": worst, "components": int(test.shape[2])}
        if r.get("served") and r.get("evaluate_us"):
            rho = r["rollout_solver_share"]
            res["r4e_speedup_against_baseline"] = r.get("speedup_effective")
            res["r4e_speedup_against_vode"] = us / (r["evaluate_us"] + rho * us)
        out[name] = res
        print(f"{name}: VODE {us:.0f} us against the baseline's {base:.0f} us ({base / us:.1f}x), worst against stored {worst:.1e}"
              + (f"; R4e's speed-up {res['r4e_speedup_against_baseline']:.1f} -> {res['r4e_speedup_against_vode']:.1f}" if "r4e_speedup_against_vode" in res else ""))  # fmt: skip
    served = [v["r4e_speedup_against_vode"] for v in out.values() if "r4e_speedup_against_vode" in v]
    out["median_r4e_speedup_against_vode"] = float(np.median(served)) if served else None
    print("median R4e speed-up against VODE:", out["median_r4e_speedup_against_vode"])
    (OUT / "p14_baseline.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
