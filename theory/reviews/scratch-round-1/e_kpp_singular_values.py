"""
Review test E (for proposed theorem N7): singular values of the Fisher-KPP step map's Jacobian (tau = 0.01, D = 1,
r = 10, Neumann, n cells on [0, 1]) for n = 20, 40, 80, at a rough state (independent uniform values in
[0.05, 0.95]) and at a smooth one (0.5 + 0.3 cos(pi x)). N7's bound: sigma_{m+1} <= exp(-lam_{m+1} tau) +
c L_f / lam_{m+1} with lam_m = 4 n^2 sin^2(m pi / 2n), L_f = 10: a count of modes above a tolerance that does not
grow with n. Jacobian by the variational equation (expm of the frozen-in-time linearisation is not used).
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

TAU, R = 0.01, 10.0
out = {}
for n in (20, 40, 80):
    L = np.diag(np.ones(n - 1), -1) - 2 * np.eye(n) + np.diag(np.ones(n - 1), 1)
    L[0, 0] += 1; L[-1, -1] += 1; L *= n**2
    x = (np.arange(n) + 0.5) / n
    for name, u0 in {"rough": np.random.default_rng(0).uniform(0.05, 0.95, n), "smooth": 0.5 + 0.3 * np.cos(np.pi * x)}.items():
        def f(_t, y):
            u, M = y[:n], y[n:].reshape(n, n)
            return np.concatenate([L @ u + R * u * (1 - u), ((L + np.diag(R * (1 - 2 * u))) @ M).ravel()])
        sol = solve_ivp(f, (0, TAU), np.concatenate([u0, np.eye(n).ravel()]), method="LSODA", rtol=1e-10, atol=1e-13)
        s = np.linalg.svd(sol.y[n:, -1].reshape(n, n), compute_uv=False)
        lam = 4 * n**2 * np.sin(np.arange(n) * np.pi / (2 * n)) ** 2
        heat = np.exp((R - lam) * TAU)  # the bound without the coupling term, e^{(L_f - lam_m) tau}
        cnt = {tol: int((s > tol).sum()) for tol in (1e-2, 1e-4, 1e-8)}
        out[f"{n}-{name}"] = {"sigma_first_16": s[:16].tolist(), "count_above": cnt, "max_sigma_over_heat": float(np.max(s[:12] / heat[:12]))}
        print(f"n={n:2d} {name:6s} modes above 1e-2/1e-4/1e-8: {cnt[1e-2]}/{cnt[1e-4]}/{cnt[1e-8]}  sigma_1..8=" + " ".join(f"{v:.2e}" for v in s[:8])
              + f"  max sigma_m / e^((L_f-lam_m)tau), m<=12: {out[f'{n}-{name}']['max_sigma_over_heat']:.2f}", flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
