"""
The exact baselines, re-measured (council round 4: Fable's and Astra's denominators). On each problem's 40 test trajectories' states at
steps 0, 10 and 20 (KIDA dark: 40 states at steps 0, 8, 16, as p9), one core, three exact solvers at the system's own tolerances:
  library  the benchmark's step (solve_ivp with the system's method and analytic Jacobian), timed here, not read from its baseline record;
  vode     SciPy's VODE, BDF with the analytic Jacobian (Fortran);
  lsoda    SciPy's LSODA with the analytic Jacobian (Fortran; switches between Adams and BDF).
Each is scored against the stored next states (the worst scored error); the fastest solver whose states agree within QUALIFY is the
problem's denominator. R4e's served surrogates are re-timed with the sketched gate (generic4, the same fit; one evaluation per batch) and
their effective speed-ups restated against the denominator, with R4e's solver share. Writes data/p15_denominators.json.

Prediction, committed before the run: LSODA qualifies on every problem, including Nelson and primordial where p14's VODE failed, and is
the fastest qualified solver on at least 8 of the 15; the sketched gate lowers the evaluation at least 2x on every served problem whose fit
has at least 300 terms; the median restated speed-up over R4e's served problems is between 10 and 30.
"""

import json
import time
from pathlib import Path

import numpy as np
from generic4 import rebuild
from scipy.integrate import ode

from src.benchmark.step_maps import STEP_MAPS, data

OUT = Path(__file__).resolve().parent / "data"
QUALIFY = 1e-5
PROBLEMS = ("bruss_d2", "fisher_kpp_d2", "fisher_kpp_d3", "fisher_kpp_d5", "robertson", "lorenz96_d4", "chemakzo", "bruss_d20", "burgers_d20",
            "fisher_kpp_d10", "hires", "pollu", "nelson", "primordial", "kida_dark")  # fmt: skip


def fortran(system, y, dt, name):
    r = ode(lambda t, v: system.rhs(v), lambda t, v: system.jac(v))
    kw = {"rtol": system.rtol, "atol": system.atol, "nsteps": 200000}
    if name == "vode":
        r.set_integrator("vode", method="bdf", with_jacobian=True, **kw)
    else:
        r.set_integrator("lsoda", **kw)
    r.set_initial_value(y, 0.0)
    r.integrate(dt)
    return r.y if r.successful() else np.full_like(y, np.nan)


def problem(name):
    if name == "kida_dark":
        from p5_kida_screen import Data

        D = Data()
        X = np.concatenate([D.test[:, k] for k in (0, 8, 16)])
        Y = np.concatenate([D.test[:, k + 1] for k in (0, 8, 16)])
        return D.P, X, Y
    P = STEP_MAPS[name]
    _, test = data(name)
    steps = [k for k in (0, 10, 20) if k < test.shape[1] - 1]
    return P, np.concatenate([test[:, k] for k in steps]), np.concatenate([test[:, k + 1] for k in steps])


def main() -> None:
    import warnings

    warnings.filterwarnings("ignore")
    out = {}
    for name in PROBLEMS:
        P, X, Y = problem(name)
        sysm = P.system
        Ys = 10.0**X if sysm.log else X
        res = {}
        for solver in ("library", "vode", "lsoda"):
            t0 = time.perf_counter()
            if solver == "library":
                nxt = np.array([s.y[:, -1] if (s := sysm.solve(y, P.dt)).success else np.full_like(y, np.nan) for y in Ys])
            else:
                nxt = np.array([fortran(sysm, y, P.dt, solver) for y in Ys])
            us = 1e6 * (time.perf_counter() - t0) / len(Ys)
            with np.errstate(all="ignore"):
                Z = sysm.coords(np.where(np.isfinite(nxt), nxt, np.nan))
                e = np.where(P.scored(Y), np.abs(Z - Y), 0.0)
            worst = float(np.max(e)) if np.all(np.isfinite(e)) else float("inf")
            res[solver] = {"us": us, "worst_vs_stored": worst, "qualified": bool(worst <= QUALIFY)}
        ok = {k: v["us"] for k, v in res.items() if v["qualified"]}
        res["denominator"] = min(ok, key=ok.get) if ok else None
        res["denominator_us"] = ok[res["denominator"]] if ok else None
        rec = OUT / "records" / "r4e" / f"{name}.json"
        r = json.loads(rec.read_text()) if rec.exists() else {}
        if r.get("served") and res["denominator_us"]:
            _, test = data(name)
            sur = rebuild(STEP_MAPS[name], data(name)[0], r, None, {"sketch": True})
            Xe = test[:, :-1].reshape(-1, test.shape[2])[:2000]
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                sur.evaluate(Xe, cache=False)
                ts.append(time.perf_counter() - t0)
            ev = 1e6 * float(np.median(ts)) / len(Xe)
            rho, den = r["rollout_solver_share"], res["denominator_us"]
            res.update({"terms": r["terms"], "evaluate_us_r4e": r["evaluate_us"], "evaluate_us_sketched": ev,
                        "speedup_r4e_against_denominator": den / (r["evaluate_us"] + rho * den),
                        "speedup_sketched_against_denominator": den / (ev + rho * den)})  # fmt: skip
        out[name] = res
        print(f"{name}: " + ", ".join(f"{k} {v['us']:.0f} us ({v['worst_vs_stored']:.1e})" for k, v in res.items() if isinstance(v, dict))
              + f" -> {res['denominator']}" + (f"; speed-up {res['speedup_r4e_against_denominator']:.1f} -> sketched {res['speedup_sketched_against_denominator']:.1f}"
                                                 if "speedup_sketched_against_denominator" in res else ""), flush=True)  # fmt: skip
    s = [v["speedup_sketched_against_denominator"] for v in out.values() if isinstance(v, dict) and "speedup_sketched_against_denominator" in v]
    out["median_sketched_speedup"] = float(np.median(s)) if s else None
    print("median sketched speed-up against the denominator:", out["median_sketched_speedup"])
    (OUT / "p15_denominators.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
