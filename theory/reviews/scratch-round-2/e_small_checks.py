"""
Round-2 small checks.
(1) Proposition 2.3's "stays bounded when the averaged defect vanishes": a planar rotation by an irrational angle with
    the tangential defect eps cos(q phi) (zero orbit average, a trigonometric polynomial). First order predicts a
    bounded error; the phase map phi -> phi + theta + eps cos(q phi) has a rotation number that differs from theta at
    order eps^2, so the error grows linearly at rate about q eps^2 / 2 per step... measured here.
(2) Proposition 6.6 on Robertson at long steps: sigma_2 of DPhi_tau against the frozen exp(tau lambda_slow(x0)) and
    against the path average exp(int_0^tau lambda_slow(x(t)) dt).
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

out = {"rotation": [], "robertson": []}
theta = 2 * np.pi * (np.sqrt(5) - 1) / 2
for eps, q in ((1e-2, 1), (1e-2, 3), (1e-3, 3), (1e-2, 8)):
    n = int(200 / eps**2 / q) if eps > 5e-3 else 2_000_000
    phi = 0.3
    errs = {}
    for k in range(1, n + 1):
        phi += theta + eps * np.cos(q * phi)
        if k in (10, 100, 1000, 10_000, 100_000, n):
            errs[k] = abs(phi - 0.3 - k * theta)
    row = {"eps": eps, "q": q, "error_over_eps": {k: v / eps for k, v in errs.items()}, "drift_per_step_over_q_eps2": (errs[n] / n) / (q * eps**2)}
    out["rotation"].append(row)
    print(f"rotation eps={eps} q={q}: |phase error| / eps at n = " + ", ".join(f"{k}: {v / eps:.2f}" for k, v in errs.items()) + f";  drift per step / (q eps^2) = {row['drift_per_step_over_q_eps2']:.3f}")

K1, K2, K3 = 0.04, 3e7, 1e4
rhs = lambda _t, y: [-K1 * y[0] + K3 * y[1] * y[2], K1 * y[0] - K3 * y[1] * y[2] - K2 * y[1] ** 2, K2 * y[1] ** 2]
jac = lambda _t, y: np.array([[-K1, K3 * y[2], K3 * y[1]], [K1, -K3 * y[2] - 2 * K2 * y[1], -K3 * y[1]], [0.0, 2 * K2 * y[1], 0.0]])
for t0 in (0.01, 1.0, 100.0):
    x0 = solve_ivp(rhs, (0, t0), [1.0, 0, 0], method="Radau", jac=jac, rtol=1e-12, atol=1e-20).y[:, -1]
    for tau in (1.0, 10.0, 100.0, 1000.0):
        base = solve_ivp(rhs, (0, tau), x0, method="Radau", jac=jac, rtol=1e-11, atol=1e-20, dense_output=True)
        ts = np.concatenate([np.linspace(a, b, 8, endpoint=False) for a, b in zip(base.t[:-1], base.t[1:])] + [[tau]])
        D, g, integral = np.eye(3), np.sqrt(3) / 6, 0.0
        for a, b in zip(ts[:-1], ts[1:]):
            h = b - a
            A1, A2 = jac(0, base.sol(a + (0.5 - g) * h)), jac(0, base.sol(a + (0.5 + g) * h))
            D = expm(h * (A1 + A2) / 2 - np.sqrt(3) / 12 * h * h * (A1 @ A2 - A2 @ A1)) @ D
            integral += h * 0.5 * (np.sort(np.linalg.eigvals(A1).real)[1] + np.sort(np.linalg.eigvals(A2).real)[1])
        s2 = np.linalg.svd(D, compute_uv=False)[1]
        frozen = np.exp(tau * np.sort(np.linalg.eigvals(jac(0, x0)).real)[1])
        row = {"t0": t0, "tau": tau, "sigma2": float(s2), "frozen": float(frozen), "path_average": float(np.exp(integral))}
        out["robertson"].append(row)
        print(f"robertson t0={t0:g} tau={tau:g}: sigma_2={s2:.3e} frozen={frozen:.3e} path-averaged={np.exp(integral):.3e}")
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
