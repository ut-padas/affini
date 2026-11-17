"""
Accuracy check of the ensemble integrator (``integrate_ensemble``) on Lorenz-96 against SciPy, on
the active backend. Run from the repository root:

    SURROGATE_DEVICE=gpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python scripts/ode_accuracy_check.py out.json

Part 1 (accuracy): for each forcing, horizon, d (5 to 1024, so the GPU's thread and warp paths) and
tolerance, the error of integrate_ensemble and of SciPy RK45 at the same rtol = atol, both against
DOP853 at 1e-13, and |ours - RK45|. With DOPRI5 the integrator takes RK45's steps, so the error
ratio is 1 and |ours - RK45| is rounding, amplified only where the flow is chaotic.
Part 2 (regimes): max |ours - RK45| / (atol + rtol |y|) and NaN rows over 288 regimes (state size
0.01 to 500, forcing 1 to 32, horizon 1e-3 to 10, tolerance 1e-3 to 1e-10). Where the flow is
chaotic (horizon 10 at forcing >= 8, states >= 50) rounding grows to O(1), for SciPy's own solvers
too; forcing 32 is chaotic enough by horizon 1 to reach a few tenths of a 1e-10 tolerance. Elsewhere
the ratio stays far below 1. The SciPy references run on 6 CPU processes.
"""

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import solve_ivp


def rhs(t, x, F):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def scipy_rows(X, F, tf, method, tol):
    return np.array([solve_ivp(rhs, (0, tf), x, method=method, rtol=tol, atol=tol, args=(F,)).y[:, -1] for x in X])


def references(cfg):
    kind, scale, F, tf, d, tol, n = cfg
    X = np.random.default_rng(d).uniform(-scale, scale, (n, d))
    out = {"X": X, "rk45": scipy_rows(X, F, tf, "RK45", tol)}
    if kind == "accuracy":
        out["truth"] = scipy_rows(X, F, tf, "DOP853", 1e-13)
    return out


def main(path):
    from src import to_host, xp
    from src.numerics import integrate_ensemble

    acc = [("accuracy", 5.0, F, tf, d, tol, 4) for (F, tf), d, tol in itertools.product(
        [(1.0, 0.1), (8.0, 0.1), (8.0, 1.0), (8.0, 5.0)], [5, 10, 32, 33, 40, 80, 200, 1024], [1e-3, 1e-6, 1e-9])]  # fmt: skip
    reg = [("regime", s, F, tf, d, tol, 8) for s, F, tf, d, tol in itertools.product(
        [0.01, 5.0, 50.0, 500.0], [1.0, 8.0, 32.0], [1e-3, 0.1, 1.0, 10.0], [5, 40], [1e-3, 1e-6, 1e-10])]  # fmt: skip
    cfgs = acc + reg
    with ProcessPoolExecutor(6) as ex:  # SciPy references on the CPU, in parallel
        refs = list(ex.map(references, cfgs))
    rows = []
    for (kind, scale, F, tf, d, tol, _), r in zip(cfgs, refs, strict=True):
        ours = np.asarray(to_host(integrate_ensemble("lorenz96", xp.asarray(r["X"]), tf, [F], tol, tol)))
        row = {"kind": kind, "scale": scale, "F": F, "tf": tf, "d": d, "tol": tol,
               "ours_vs_rk45": float(np.abs(ours - r["rk45"]).max()),
               "in_tol": float((np.abs(ours - r["rk45"]) / (tol + tol * np.abs(r["rk45"]))).max()),
               "nan": int(np.isnan(ours).any(1).sum())}  # fmt: skip
        if kind == "accuracy":
            eo, es = np.abs(ours - r["truth"]).max(1), np.abs(r["rk45"] - r["truth"]).max(1)
            row.update(ours_err=float(eo.max()), rk45_err=float(es.max()), ratio_median=float(np.median(eo / np.maximum(es, 1e-300))),
                       ratio_max=float((eo / np.maximum(es, 1e-300)).max()))  # fmt: skip
        rows.append(row)
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"{'F':>3} {'tf':>4} {'d':>4} {'tol':>6} | {'ours err':>9} {'rk45 err':>9} | {'ratio med':>9} {'max':>6} | {'|ours-rk45|':>11}")
    for r in rows:
        if r["kind"] == "accuracy":
            print(f"{r['F']:>3.0f} {r['tf']:>4} {r['d']:>4} {r['tol']:>6.0e} | {r['ours_err']:9.2e} {r['rk45_err']:9.2e} | "
                  f"{r['ratio_median']:9.2f} {r['ratio_max']:6.2f} | {r['ours_vs_rk45']:11.2e}")  # fmt: skip
    reg_rows = [r for r in rows if r["kind"] == "regime"]
    print("regimes:", len(reg_rows), "NaN rows:", sum(r["nan"] for r in reg_rows))
    print("scale F tf | max |ours - RK45| / tol at tol 1e-3, 1e-6, 1e-10")
    for k in sorted({(r["scale"], r["F"], r["tf"]) for r in reg_rows}):
        v = [max(r["in_tol"] for r in reg_rows if (r["scale"], r["F"], r["tf"]) == k and r["tol"] == t) for t in (1e-3, 1e-6, 1e-10)]
        print(f"{k[0]:>6} {k[1]:>4} {k[2]:>6} | " + "  ".join(f"{x:9.1e}" for x in v))


if __name__ == "__main__":
    main(sys.argv[1])
