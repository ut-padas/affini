"""
Round-3 test B: Proposition 6.5's lower bound on Robertson (log10 coordinates, tau = 1) against the error of the best
one-coordinate fit that was found.

At a visited state p (the benchmark-like start (0.8, 1e-8, 0.2 - 1e-8) after t = 10): the singular values of DPhi(p)
(central differences), also restricted to the tangent plane of the leaf sum y = const; K_2(r), the largest
|D^2Phi(x)[u, u]| over 30 sampled x in B(p, r) and unit u (second differences; a lower estimate of the true supremum, so the
bound computed from it is an upper estimate of the proposition's); the bound s r - K_2 r^2 / 2 with s = sigma_2; and, on
400 points of B(p, r), the error of the best function of ONE linear coordinate w.x: half the largest 2-norm diameter of
the images of 10 slabs of constant w.x, minimised over 2,000 directions w. The same slab error for the coordinate z1
alone (the memo's chart) on the ball, and on the leaf through p (points of the ball projected to the leaf).
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

K1, K2, K3 = 0.04, 3e7, 1e4
HERE = Path(__file__).resolve().parent


def rhs(_t, y):
    return [-K1 * y[0] + K3 * y[1] * y[2], K1 * y[0] - K3 * y[1] * y[2] - K2 * y[1] ** 2, K2 * y[1] ** 2]


def jac(_t, y):
    return [[-K1, K3 * y[2], K3 * y[1]], [K1, -K3 * y[2] - 2 * K2 * y[1], -K3 * y[1]], [0.0, 2 * K2 * y[1], 0.0]]


def flow(y, t):
    return solve_ivp(rhs, (0, t), y, method="Radau", jac=jac, rtol=1e-9, atol=1e-22).y[:, -1]


def phi(Z):
    return np.array([np.log10(flow(10.0**z, 1.0)) for z in np.atleast_2d(Z)])


def slab_error(c, F, bins=10):
    """Half the largest diameter of F over slabs of equal count in the coordinate c."""
    worst = 0.0
    for part in np.array_split(np.argsort(c), bins):
        P = F[part]
        worst = max(worst, 0.5 * np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1).max()))
    return worst


rng = np.random.default_rng(0)
p = np.log10(flow(np.array([0.8, 1e-8, 0.2 - 1e-8]), 10.0))
h = 1e-3
D = np.array([(phi(p + h * e) - phi(p - h * e))[0] / (2 * h) for e in np.eye(3)]).T
sig = np.linalg.svd(D, compute_uv=False)
y = 10.0**p
n = y / np.linalg.norm(y)  # normal of the leaf sum 10^z = const in log coordinates (gradient of sum y is ln10 * y)
T = np.linalg.svd(np.eye(3) - np.outer(n, n))[0][:, :2]
sig_leaf = np.linalg.svd(D @ T, compute_uv=False)
print("p (log10):", p, "\nsigma(DPhi):", sig, "\nsigma(DPhi restricted to the leaf's tangent plane):", sig_leaf)
out = {"p": p.tolist(), "sigma": sig.tolist(), "sigma_leaf": sig_leaf.tolist(), "radii": []}
W = rng.normal(size=(2000, 3))
W /= np.linalg.norm(W, axis=1, keepdims=True)
for r in (0.05, 0.2):
    X = rng.normal(size=(400, 3))
    X = p + r * X / np.linalg.norm(X, axis=1, keepdims=True) * rng.uniform(0, 1, (400, 1)) ** (1 / 3)
    F = phi(X)
    # K_2 on the ball: second differences along random unit directions at 150 of the points
    k2 = 0.0
    for x in X[:30]:
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        hh = min(0.02, r / 5)
        k2 = max(k2, np.linalg.norm(phi(x + hh * u) + phi(x - hh * u) - 2 * phi(x)) / hh**2)
    bound = sig[1] * r - 0.5 * k2 * r * r
    r_star = min(r, sig[1] / k2)
    best = sig[1] * r_star - 0.5 * k2 * r_star**2
    errs = np.array([slab_error(X @ w, F) for w in W])
    w_best = W[errs.argmin()]
    # the leaf through p: move each point along the normal direction until sum 10^z = sum 10^p (Newton in the scalar shift)
    L = X.copy()
    for _ in range(30):
        g = (10.0**L).sum(1) - y.sum()
        L -= (g / (np.log(10) * ((10.0**L) * n).sum(1)))[:, None] * n
    FL = phi(L)
    row = {"r": r, "K2": k2, "bound_at_r": bound, "bound_best_radius": best, "best_linear_coordinate_error": float(errs.min()), "best_w": w_best.tolist(),
           "z1_error_ball": slab_error(X[:, 0], F), "z1_error_leaf": slab_error(L[:, 0], FL), "half_diameter_image": 0.5 * float(np.sqrt(((F[:, None] - F[None]) ** 2).sum(-1).max()))}
    out["radii"].append(row)
    print(f"r = {r}: K2 = {k2:.2f}  bound s r - K2 r^2/2 = {bound:.4f} (best over smaller radii {best:.4f})  best one linear coordinate: {errs.min():.4f} (w = {np.round(w_best, 3)})"
          f"  z1 on the ball: {row['z1_error_ball']:.4f}  z1 on the leaf: {row['z1_error_leaf']:.2e}  constant (0 coordinates): {row['half_diameter_image']:.4f}", flush=True)
(HERE / "b_borsuk_ulam.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
