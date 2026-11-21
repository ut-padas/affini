"""
Review test D: is there a d-independent light cone on the method-of-lines ladder (P3.2)?

Fisher-KPP exactly as src/problems/ode/reaction_diffusion.py (u' = D u_xx + r u (1 - u) on n cell centres of
[0, 1], Neumann, D = 1, r = 10) at the ladder's step tau = 0.01. For n = 10, 20, 40, 80: the window error of
Corollary 3.2 for the middle output (coordinates farther than r sites frozen at the box centre), on the box
u in [0.05, 0.95]^n, and the smallest r with error <= 1e-3 and <= 1e-5. Lemma 3.1's constants are l = D n^2,
delta <= -2 D n^2 + r: both grow with n. Also the lemma's own bound for the same window (linear majorant).
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

TAU, DD, R = 0.01, 1.0, 10.0

def lap(n):
    L = np.diag(np.ones(n - 1), -1) - 2 * np.eye(n) + np.diag(np.ones(n - 1), 1)
    L[0, 0] += 1; L[-1, -1] += 1
    return DD * L * n**2

def flow(X, n):
    L = lap(n)
    f = lambda _t, y: (lambda u: (u @ L.T + R * u * (1 - u)).ravel())(y.reshape(-1, n))
    return solve_ivp(f, (0, TAU), X.ravel(), method="LSODA", rtol=1e-10, atol=1e-12).y[:, -1].reshape(X.shape)

out = {}
for n in (10, 20, 40, 80):
    rng = np.random.default_rng(1)
    X = rng.uniform(0.05, 0.95, (48, n))
    i0 = n // 2
    dist = np.abs(np.arange(n) - i0)
    ref = flow(X, n)[:, i0]
    errs = []
    for r in range(0, n // 2 + 1):
        Xw = X.copy(); Xw[:, dist > r] = 0.5
        errs.append(float(np.max(np.abs(flow(Xw, n)[:, i0] - ref))))
    need = {tol: next((r for r, e in enumerate(errs) if e <= tol), None) for tol in (1e-3, 1e-5)}
    # Lemma 3.1's bound: e^{delta tau} expm(tau L), delta = sup Re J_ii = -2 D n^2 + R (u -> 0), L = D n^2 on the two neighbours
    Lm = DD * n**2 * (np.diag(np.ones(n - 1), -1) + np.diag(np.ones(n - 1), 1))
    bound = np.exp((-2 * DD * n**2 + R) * TAU) * expm(TAU * Lm)[i0]
    wb = [float(0.45 * bound[dist > r].sum()) for r in range(0, n // 2 + 1)]
    need_b = {tol: next((r for r, e in enumerate(wb) if e <= tol), None) for tol in (1e-3, 1e-5)}
    out[n] = {"window_error": errs, "radius_needed": need, "radius_needed_by_lemma_bound": need_b, "half_system": n // 2}
    print(f"n={n:3d}: radius for 1e-3: {need[1e-3]} (lemma bound {need_b[1e-3]}), for 1e-5: {need[1e-5]} (lemma bound {need_b[1e-5]}); half the system = {n//2}", flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
