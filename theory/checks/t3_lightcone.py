"""
Theorem 3 (memo section T3): a light cone for locally coupled ODEs, on Lorenz-96 at F = 1 (the benchmark's
setting: box [0, 2]^d, the ladder's step tau = 0.25, and longer steps).

x_i' = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F: J = -I + N, where row i of N holds x_{i-1} at column i+1,
-x_{i-1} at column i-2 and x_{i+1} - x_{i-2} at column i-1. Lemma 3.1: entrywise
|dPhi_tau / dx_0| <= e^{-tau} expm(tau L), with L the majorant of |N| (l1 = sup |x_j| at columns i+1, i-2;
l2 = sup |x_{i+1} - x_{i-2}| at column i-1), the sups taken over the trajectories from the box. The sups here
are measured on sampled trajectories, not proved, so the bound is checked as stated with measured constants.
Checks: (1) the measured Jacobian entries never exceed the bound, as a function of the graph distance;
(2) the error of the window surrogate (coordinates farther than r from output 0 frozen at the box centre) is the
same for d = 20, 40, 80 (Corollary 3.2: it does not depend on d) and lies below the corollary's bound.
"""

import numpy as np
from common import write
from scipy.integrate import solve_ivp
from scipy.linalg import expm

F = 1.0
TAUS = [0.25, 1.0]
SAMPLES = 8


def rhs_batch(d):
    def f(_t, y):
        x = y.reshape(-1, d)
        return ((np.roll(x, -1, 1) - np.roll(x, 2, 1)) * np.roll(x, 1, 1) - x + F).ravel()

    return f


def flow(X, tau, d):
    return solve_ivp(rhs_batch(d), (0, tau), X.ravel(), method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1].reshape(X.shape)


def jacobian_flow(x0, tau, d):
    """dPhi_tau/dx0 by the variational equation M' = J(x) M, M(0) = I; also the trajectory's sup constants."""
    i = np.arange(d)
    ip, im, im2 = (i + 1) % d, (i - 1) % d, (i - 2) % d

    def f(_t, y):
        x, M = y[:d], y[d:].reshape(d, d)
        J = -np.eye(d)
        J[i, ip] += x[im]
        J[i, im2] -= x[im]
        J[i, im] += x[ip] - x[im2]
        return np.concatenate([(x[ip] - x[im2]) * x[im] - x + F, (J @ M).ravel()])

    sol = solve_ivp(f, (0, tau), np.concatenate([x0, np.eye(d).ravel()]), method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True)
    ts = np.linspace(0, tau, 101)
    xs = sol.sol(ts)[:d].T
    l1 = np.max(np.abs(xs))
    l2 = np.max(np.abs(np.roll(xs, -1, 1) - np.roll(xs, 2, 1)))
    return sol.y[d:, -1].reshape(d, d), l1, l2


def distance(d):
    """Directed graph distance D[i, j]: steps of +1, -1, -2 (i depends on i+1, i-1, i-2) from i to j."""
    off = (np.arange(d)[None, :] - np.arange(d)[:, None]) % d
    back = (-off) % d
    # forward offsets need +1 steps; backward ones -2 steps (one -1 if odd); combine: min over k forward steps
    best = np.full((d, d), d, dtype=int)
    for k in range(d):
        rem = (back + k) % d  # after k forward steps, the remaining backward offset
        best = np.minimum(best, k + (rem + 1) // 2)
    return best


def window_error(d, tau, r, rng, n=64):
    X = rng.uniform(0.0, 2.0, (n, d))
    D0 = distance(d)[0]
    Xw = X.copy()
    Xw[:, r < D0] = F
    return float(np.max(np.abs(flow(X, tau, d)[:, 0] - flow(Xw, tau, d)[:, 0])))


def main() -> None:
    rng = np.random.default_rng(0)
    out = {"F": F, "taus": TAUS, "results": {}}
    d = 40
    Dm = distance(d)
    for tau in TAUS:
        Ms, l1s, l2s = [], [], []
        for _ in range(SAMPLES):
            M, l1, l2 = jacobian_flow(rng.uniform(0.0, 2.0, d), tau, d)
            Ms.append(np.abs(M))
            l1s.append(l1)
            l2s.append(l2)
        l1, l2 = max(l1s), max(l2s)
        i = np.arange(d)
        L = np.zeros((d, d))
        L[i, (i + 1) % d] = l1
        L[i, (i - 2) % d] = l1
        L[i, (i - 1) % d] = l2
        bound = np.exp(-tau) * expm(tau * L)
        Mmax = np.max(Ms, axis=0)
        rows = []
        for D in range(12):
            sel = Dm == D
            if not sel.any():
                continue
            rows.append({"D": D, "measured_max": float(Mmax[sel].max()), "bound_max": float(bound[sel].max()), "valid": bool(np.all(Mmax[sel] <= bound[sel]))})
        werr = {}
        for dd in (20, 40, 80):
            werr[dd] = [window_error(dd, tau, r, np.random.default_rng(1)) for r in range(9)]
        # Corollary 3.2's bound on the window error: sum over j outside the window of h_j sup|dPhi_0/dx_j|, h_j = 1
        wb = [float(np.sum(bound[0][Dm[0] > r])) for r in range(9)]
        out["results"][str(tau)] = {"l1": l1, "l2": l2, "jacobian_by_distance": rows, "window_error": {str(k): v for k, v in werr.items()}, "window_bound_d40": wb}
    path = write("t3_lightcone", out)
    for tau, res in out["results"].items():
        print(f"tau={tau}: l1={res['l1']:.3f} l2={res['l2']:.3f}")
        for q in res["jacobian_by_distance"]:
            print(f"   D={q['D']:2d} measured={q['measured_max']:.2e} bound={q['bound_max']:.2e} valid={q['valid']}")
        for dd, v in res["window_error"].items():
            print(f"   window error d={dd}: " + " ".join(f"{e:.1e}" for e in v))
        print("   window bound d=40:  " + " ".join(f"{e:.1e}" for e in res["window_bound_d40"]))
    print("wrote", path)


if __name__ == "__main__":
    main()
