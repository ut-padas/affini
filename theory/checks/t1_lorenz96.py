"""
Theorem 1's prediction for the step map of Lorenz-96 (memo section T1): how fast the Chebyshev coefficients of
Phi_tau decay along a fibre, as the step tau grows, in three regimes (d = 8).

  F = 0, box [-a, a]^d: dissipative. The theorem applies with an explicit bound: in the max-norm,
        mu(J(z)) <= -1 + 4 max_j |z_j|, and |x_j(t)| <= ||x(t)||_2 <= sqrt(d) a e^{-t} (the quadratic term
        conserves sum x_i^2), so m(t) <= -1 + 4 (sqrt(d) a + r) < 0 for r < 1/4 - sqrt(d) a: Lambda = 0 and
        rho_thm does not depend on tau. Checked: |c_k| <= 2 M rho_thm^-k on the fibre, for every tau.
  F = 1, box [0, 2]^d (the benchmark's setting): neutral (a travelling wave). Measured only.
  F = 8, a box of half-width 0.5 around a point of the attractor: chaotic. Prediction: rho_eff - 1 shrinks
        roughly like e^{-lambda tau}; lambda_max is estimated here (Benettin) to compare.
The fibre varies x_0 over the box with every other coordinate fixed at a random point of it; the output is x_0.
"""

import numpy as np
from common import cheb_coefficients, cheb_points, decay_rate, rho_from_halfwidth, write
from scipy.integrate import solve_ivp

D = 8
N = 128
THETA = 0.999
TAUS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]


def rhs(F: float):
    def f(_t, y):
        x = y.reshape(-1, D)
        dx = (np.roll(x, -1, 1) - np.roll(x, 2, 1)) * np.roll(x, 1, 1) - x + F
        return dx.ravel()

    return f


def flow(X: np.ndarray, tau: float, F: float) -> np.ndarray:
    sol = solve_ivp(rhs(F), (0, tau), X.ravel(), method="DOP853", rtol=1e-13, atol=1e-15)
    return sol.y[:, -1].reshape(X.shape)


def lyapunov(F: float, x0: np.ndarray, T: float = 200.0, dt: float = 0.5) -> float:
    """Largest Lyapunov exponent by Benettin's method (two nearby trajectories, renormalised every dt)."""
    x, eps, acc = x0.copy(), 1e-8, 0.0
    y = x + eps * np.random.default_rng(0).standard_normal(D) / np.sqrt(D)
    for _ in range(int(T / dt)):
        x, y = flow(x[None], dt, F)[0], flow(y[None], dt, F)[0]
        dist = np.linalg.norm(y - x)
        acc += np.log(dist / eps)
        y = x + (y - x) * eps / dist
    return acc / T


def fibre(lo: np.ndarray, hi: np.ndarray, tau: float, F: float, seed: int) -> np.ndarray:
    base = np.random.default_rng(seed).uniform(lo, hi)
    X = np.repeat(base[None], N + 1, 0)
    X[:, 0] = cheb_points(N, lo[0], hi[0])
    return cheb_coefficients(flow(X, tau, F)[:, 0])


def main() -> None:
    out = {"d": D, "n": N, "taus": TAUS, "regimes": {}}
    # F = 0: the dissipative box and the theorem's bound
    a = 0.05
    r = 0.999 * (0.25 - np.sqrt(D) * a)
    lo, hi = -a * np.ones(D), a * np.ones(D)
    rho_thm = rho_from_halfwidth(THETA * r / a)
    rows = []
    for t in TAUS:
        c = fibre(lo, hi, t, 0.0, 1)
        M = np.sqrt(D) * a * np.exp(-t) + THETA * r
        k = np.arange(len(c))
        bound = np.where(k == 0, M, 2 * M * rho_thm ** (-k.astype(float)))
        resolved = np.abs(c) > 1e-13 * np.max(np.abs(c))
        rows.append({"tau": t, **decay_rate(c), "max_coef_over_bound": float(np.max(np.abs(c[resolved]) / bound[resolved]))})
    out["regimes"]["F0_dissipative"] = {"box": [-a, a], "r": r, "rho_thm": rho_thm, "m_max": -1 + 4 * (np.sqrt(D) * a + r), "rows": rows}
    # F = 1: the benchmark's box
    lo, hi = np.zeros(D), 2 * np.ones(D)
    rows = [{"tau": t, **decay_rate(fibre(lo, hi, t, 1.0, 1))} for t in TAUS]
    big = [q for q in rows if q["tau"] >= 0.25]
    # Corollary 1.2 (neutral, m(r) <= c r): rho - 1 ~ 1/tau, a slope of -1 in log-log
    slope_loglog = float(np.polyfit(np.log([q["tau"] for q in big]), np.log([q["rho"] - 1 for q in big]), 1)[0])
    out["regimes"]["F1_neutral"] = {"box": [0.0, 2.0], "slope_loglog_rho_minus_1": slope_loglog, "rows": rows}
    # F = 8: around a point of the attractor
    x = flow(np.random.default_rng(2).uniform(-1, 1, (1, D)) + 8.0, 50.0, 8.0)[0]
    lo, hi = x - 0.5, x + 0.5
    lam = lyapunov(8.0, x)
    rows = [{"tau": t, **decay_rate(fibre(lo, hi, t, 8.0, 1))} for t in TAUS]
    taus = np.array(TAUS)
    rho = np.array([q["rho"] for q in rows])
    ok = np.isfinite(rho) & (rho > 1.0)
    slope = float(np.polyfit(taus[ok][-4:], np.log(rho[ok][-4:] - 1), 1)[0]) if ok.sum() >= 4 else None
    out["regimes"]["F8_chaotic"] = {"center": x.tolist(), "halfwidth": 0.5, "lyapunov": lam, "slope_log_rho_minus_1": slope, "rows": rows}
    path = write("t1_lorenz96", out)
    for name, res in out["regimes"].items():
        extra = {k: v for k, v in res.items() if k in ("rho_thm", "m_max", "lyapunov", "slope_log_rho_minus_1", "slope_loglog_rho_minus_1")}
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in extra.items()})
        for q in res["rows"]:
            print("  ", {k: round(v, 4) if isinstance(v, float) else v for k, v in q.items()})
    print("wrote", path)


if __name__ == "__main__":
    main()
