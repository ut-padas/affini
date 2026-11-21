"""
Proposition 2.4 and Theorem 8 (memo sections T2 and 8): a surrogate fitted on the visited states only is stable in a
rollout if it is a function of the slow coordinate and predicts the next state (its image then lies on the slow curve,
so the chain's distance from the curve cannot accumulate), and unstable if it depends on the fast coordinate or
predicts an increment z + g(z), whose derivative across the curve is the identity.

Robertson in log10 abundances, dt = 1, 40 steps, the benchmark's initial states (y1 in [0.5, 1], y2 log-uniform in
[1e-12, 1e-6], y3 the rest), states from step 1 on (past the initial layer; there they lie on one curve: the slow
manifold of the leaf sum y = 1, parametrised by z1). Integrated here with SciPy's Radau (no library import).
Surrogates, all least-squares polynomials fitted on the 100 training trajectories' visited states only:
  V      total degree 3 in (z1, z2, z3), increment form z + g(z)          (the review's failing case, round 2 test A)
  V-full total degree 3 in (z1, z2, z3), next state directly
  S      degree 8 in z1 alone, next state directly                          (the slow-coordinate surrogate)
  S-incr degree 8 in z1 alone, increment form z + g(z1)
Measured on 40 held-out trajectories: the worst one-step error on visited states, on states displaced by up to
0.05 dex in every coordinate (the tube), and the worst rollout error over 39 steps.
For S, which has the form G(pi(z)) with pi(z) = z1, Theorem 8's rollout bound (the max norm; L_pi = 1) is evaluated
from the fitted model: eps_G (worst residual on all visited states), L_G = sup |dG/du| and L_g = sup |dg/du| for the
reduced map g = pi o G on the visited range of z1 widened by rho = 0.05 (a grid of 4,001 points, central differences),
and the bound eps_G (1 + L_G A_{n-1}), A_n = sum_{j<n} L_g^j, valid while eps_G A_{n-1} <= rho.
"""

import numpy as np
from common import write
from scipy.integrate import solve_ivp

K1, K2, K3, STEPS, R = 0.04, 3e7, 1e4, 40, 0.05


def rhs(_t, y):
    return [-K1 * y[0] + K3 * y[1] * y[2], K1 * y[0] - K3 * y[1] * y[2] - K2 * y[1] ** 2, K2 * y[1] ** 2]


def jac(_t, y):
    return [[-K1, K3 * y[2], K3 * y[1]], [K1, -K3 * y[2] - 2 * K2 * y[1], -K3 * y[1]], [0.0, 2 * K2 * y[1], 0.0]]


def trajectory(y0: np.ndarray, steps: int) -> np.ndarray:
    sol = solve_ivp(rhs, (0, steps), y0, method="Radau", jac=jac, t_eval=np.arange(steps + 1.0), rtol=1e-10, atol=1e-22)
    return np.log10(np.maximum(sol.y.T, 1e-30))


def step(Z: np.ndarray) -> np.ndarray:
    return np.array([trajectory(10.0**z, 1)[-1] for z in Z])


def initial(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y1, y2 = rng.uniform(0.5, 1.0, n), 10.0 ** rng.uniform(-12, -6, n)
    return np.stack([y1, y2, 1.0 - y1 - y2], axis=1)


def features(X: np.ndarray, cols, q: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Chebyshev features of total degree <= q in the chosen columns, scaled to the visited box."""
    U = np.clip(2 * (X[:, cols] - lo[cols]) / (hi[cols] - lo[cols]) - 1, -3, 3)
    V = np.polynomial.chebyshev.chebvander(U, q)  # (n, len(cols), q + 1)
    idx = [a for a in np.ndindex(*([q + 1] * len(cols))) if sum(a) <= q]
    return np.stack([np.prod([V[:, c, a[c]] for c in range(len(cols))], axis=0) for a in idx], axis=1)


def fit(X, Y, cols, q, increment, lo, hi):
    target = Y - X if increment else Y
    coef = np.linalg.lstsq(features(X, cols, q, lo, hi), target, rcond=1e-12)[0]
    return lambda Z: (Z if increment else 0.0) + features(Z, cols, q, lo, hi) @ coef


def rollout_bound(P, X, Y, lo, hi, rho=0.05) -> dict:
    """Theorem 8's bound for a surrogate of z1 alone, from the fitted model (see the module docstring)."""
    eps_G = float(np.abs(P(X) - Y).max())
    u = np.linspace(lo[0] - rho, hi[0] + rho, 4001)
    h = 1e-6

    def lift(v):
        Z = np.zeros((len(v), 3))
        Z[:, 0] = v
        return P(Z)

    dG = (lift(u + h) - lift(u - h)) / (2 * h)
    L_G, L_g = float(np.abs(dG).max()), float(np.abs(dG[:, 0]).max())
    bound, A = [], 0.0
    for _n in range(1, STEPS):
        if eps_G * A > rho:
            break
        bound.append(eps_G * (1 + L_G * A))
        A = A * L_g + 1  # A_n = sum_{j<n} L_g^j
    return {"eps_G": eps_G, "L_G": L_G, "L_g": L_g, "rho": rho, "bound_by_step": bound, "bound_max": max(bound), "steps_valid": len(bound)}


def main() -> None:
    train = np.array([trajectory(y, STEPS) for y in initial(100, 1)])[:, 1:]
    test = np.array([trajectory(y, STEPS) for y in initial(40, 2)])[:, 1:]
    X, Y = train[:, :-1].reshape(-1, 3), train[:, 1:].reshape(-1, 3)
    Xe, Ye = test[:, :-1].reshape(-1, 3), test[:, 1:].reshape(-1, 3)
    lo, hi = X.min(0), X.max(0)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(Xe), 300, replace=False)
    Xt = Xe[pick] + rng.uniform(-R, R, (300, 3))
    Yt = step(Xt)
    out = {"tube_halfwidth": R, "surrogates": []}
    for name, cols, q, inc in (("V", [0, 1, 2], 3, True), ("V-full", [0, 1, 2], 3, False), ("S", [0], 8, False), ("S-incr", [0], 8, True)):
        P = fit(X, Y, cols, q, inc, lo, hi)
        Zc = test[:, 0].copy()
        roll = 0.0
        for k in range(1, STEPS):
            Zc = np.clip(P(Zc), -40, 5)
            roll = max(roll, float(np.abs(Zc - test[:, k]).max()))
        row = {"name": name, "one_step_visited": float(np.abs(P(Xe) - Ye).max()), "one_step_tube": float(np.abs(P(Xt) - Yt).max()), "rollout_39": roll}
        if name == "S":
            row["theorem_8"] = rollout_bound(P, np.vstack([X, Xe]), np.vstack([Y, Ye]), lo, hi)
        out["surrogates"].append(row)
    path = write("t2_slow_surrogate", out)
    for s in out["surrogates"]:
        print(f"{s['name']:7s} one-step visited {s['one_step_visited']:.2e}  tube {s['one_step_tube']:.2e}  rollout {s['rollout_39']:.2e}")
        if "theorem_8" in s:
            t = s["theorem_8"]
            print(f"        Theorem 8: eps_G {t['eps_G']:.2e}, L_G {t['L_G']:.3f}, L_g {t['L_g']:.3f}; bound max {t['bound_max']:.2e} over {t['steps_valid']} steps")
    print("wrote", path)


if __name__ == "__main__":
    main()
