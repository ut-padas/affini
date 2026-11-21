"""
First demonstration of Theorem 8 (memo section 8): a reduced surrogate of the Robertson step map on one invariant leaf,
with the theorem's constants computed, and its bound compared with measured rollouts.

Setting: Robertson in log10 abundances, dt = 1, the leaf y1 + y2 + y3 = 1, horizon n = 39 steps from step 1 (past the
initial layer, as the benchmark's initial states are on this leaf). M is the slow curve of the leaf, traced by one
reference trajectory from (1, 0, 0) (Radau, rtol 1e-12) and parametrised by z1 = log10 y1, which decreases along it.

The surrogate: chart pi(z) = z1 (L_pi = 1 in the max norm); G(u) = (g1(u), g2(u), log10(1 - 10^g1 - 10^g2)), with g1, g2 the
Chebyshev interpolants of degree q of z1 and z2 one step later, at Chebyshev points of U_rho (the inputs' z1 range widened
by rho), and the third component from exact conservation on the leaf. So every output lies on the leaf by construction.

The constants, and how each is obtained:
  L_g = sup |g1'|, L_G = max(sup |g1'|, sup |g2'|, sup |G3'|) on U_rho: RIGOROUS for the fitted polynomials (in exact
      arithmetic): the derivative's maximum on a grid of spacing h plus h/2 times a bound on the second derivative from the
      Chebyshev coefficients (|T_k''| <= k^2 (k^2 - 1) / 3), and an interval bound for G3' from bounds on g1, g2, g1', g2'.
  eps_G, eps_u on M: ESTIMATED: the maximum of the defect on a grid of 20,001 curve points, plus a margin of (grid spacing / 2)
      times a Lipschitz constant of the defect, plus the reference error estimated by comparing the reference trajectory at
      rtol 1e-12 and 1e-10. Two margins: "conservative" with L_G plus the exact curve map's Lipschitz constant (estimated from
      the grid), and "modulus" with the defect's own Lipschitz constant estimated from the grid (small when the fit is good).
Measured: the worst rollout error per step over 40 held-out trajectories from the benchmark's initial distribution
(y1 in [0.5, 1], y2 log-uniform in [1e-12, 1e-6]), started at step 1, and the offset of their states from the curve.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

K1, K2, K3, STEPS, RHO = 0.04, 3e7, 1e4, 40, 0.01
OUT = Path(__file__).resolve().parent / "data"


def rhs(_t, y):
    return [-K1 * y[0] + K3 * y[1] * y[2], K1 * y[0] - K3 * y[1] * y[2] - K2 * y[1] ** 2, K2 * y[1] ** 2]


def jac(_t, y):
    return [[-K1, K3 * y[2], K3 * y[1]], [K1, -K3 * y[2] - 2 * K2 * y[1], -K3 * y[1]], [0.0, 2 * K2 * y[1], 0.0]]


def solve(y0, t_end, rtol, dense=False, t_eval=None):
    return solve_ivp(rhs, (0, t_end), y0, method="Radau", jac=jac, rtol=rtol, atol=1e-24, dense_output=dense, t_eval=t_eval)


class Curve:
    """The slow curve of the leaf from one reference trajectory: z1 -> (time on the trajectory, state in log10)."""

    def __init__(self, rtol: float, t_end: float = 2e4):
        self.sol = solve([1.0, 0.0, 0.0], t_end, rtol, dense=True)
        t = np.geomspace(1e-2, t_end - 1.5, 200_001)  # from 1e-2: past the initial layer (fast rate ~ 2e3)
        self.t, self.z1 = t, np.log10(self.sol.sol(t)[0])

    def time(self, u):
        return np.interp(-np.asarray(u), -self.z1, self.t)  # z1 decreases along the trajectory

    def state(self, t):
        return np.log10(np.maximum(self.sol.sol(np.atleast_1d(t)).T, 1e-300))


def cheb_fit(f_vals, q):
    """Chebyshev coefficients of the degree-q interpolant at the q + 1 Chebyshev-Lobatto points (values ordered from s = 1 to s = -1)."""
    n = q
    ext = np.concatenate([f_vals, f_vals[-2:0:-1]])
    c = np.real(np.fft.fft(ext))[: n + 1] / n
    c[0] /= 2
    c[n] /= 2
    return c


def deriv_bounds(c, lo, hi, grid=200_001):
    """Rigorous sup |p| and sup |p'| on [lo, hi] for p(u) = sum c_k T_k(s(u)) (exact arithmetic), from a grid and second-derivative bounds."""
    k = np.arange(len(c))
    scale = 2.0 / (hi - lo)
    s = np.linspace(-1, 1, grid)
    ds = s[1] - s[0]
    cheb = np.polynomial.chebyshev
    p, dp = cheb.chebval(s, c), cheb.chebval(s, cheb.chebder(c)) * scale
    sup_dp_s = np.sum(np.abs(c) * k**2)  # |T_k'| <= k^2
    sup_d2p_s = np.sum(np.abs(c) * k**2 * (k**2 - 1) / 3.0)  # |T_k''| <= k^2 (k^2 - 1) / 3
    p_hi = np.max(p) + ds / 2 * sup_dp_s
    p_lo = np.min(p) - ds / 2 * sup_dp_s
    dp_sup = np.max(np.abs(dp)) + ds / 2 * sup_d2p_s * scale**2 / scale  # d/du of p' = scale^2 p_ss; step in u is ds / scale
    return {"min": float(p_lo), "max": float(p_hi), "dsup": float(dp_sup)}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ref, ref2 = Curve(1e-12), Curve(1e-10)
    rng = np.random.default_rng(2)
    y1 = rng.uniform(0.5, 1.0, 40)
    y2 = 10.0 ** rng.uniform(-12, -6, 40)
    tests = np.array([np.log10(np.maximum(solve([a, b, 1 - a - b], STEPS, 1e-12, t_eval=np.arange(STEPS + 1.0)).y.T, 1e-300)) for a, b in zip(y1, y2, strict=True)])[:, 1:]
    inputs = tests[:, :-1].reshape(-1, 3)
    lo, hi = inputs[:, 0].min() - RHO, inputs[:, 0].max() + RHO
    if hi > ref.z1.max():
        raise RuntimeError("U_rho leaves the reference curve")
    # offset of the held-out states from the curve (same z1)
    offset = float(np.abs(ref.state(ref.time(inputs[:, 0])) - inputs).max())

    rows = []
    for q in (8, 12, 16, 20, 24):
        s = np.cos(np.pi * np.arange(q + 1) / q)
        u = lo + (s + 1) * (hi - lo) / 2
        nxt = ref.state(ref.time(u) + 1.0)
        c1, c2 = cheb_fit(nxt[:, 0], q), cheb_fit(nxt[:, 1], q)
        b1, b2 = deriv_bounds(c1, lo, hi), deriv_bounds(c2, lo, hi)
        y1max, y2max = 10.0 ** b1["max"], 10.0 ** b2["max"]
        if y1max + y2max >= 1:
            raise RuntimeError("conservation reconstruction not bounded away from 0")
        d3 = (y1max * b1["dsup"] + y2max * b2["dsup"]) / (1 - y1max - y2max)
        L_g, L_G = b1["dsup"], max(b1["dsup"], b2["dsup"], d3)

        def lift(v, c1=c1, c2=c2):
            sv = (2 * v - lo - hi) / (hi - lo)
            g1, g2 = np.polynomial.chebyshev.chebval(sv, c1), np.polynomial.chebyshev.chebval(sv, c2)
            return np.stack([g1, g2, np.log10(1 - 10.0**g1 - 10.0**g2)], axis=1)

        # defects on M (estimated): the inputs' range without the widening
        ug = np.linspace(lo + RHO, hi - RHO, 20_001)
        exact = ref.state(ref.time(ug) + 1.0)
        dG = np.abs(lift(ug) - exact).max(axis=1)
        du = np.abs(lift(ug)[:, 0] - exact[:, 0])
        lip_exact = float(np.max(np.abs(np.diff(exact, axis=0)).max(axis=1) / np.diff(ug).max()))
        h = ug[1] - ug[0]
        ref_err = float(np.abs(ref.state(ref.time(ug) + 1.0) - ref2.state(ref2.time(ug) + 1.0)).max())
        defect = lift(ug) - exact
        L_delta = float(np.max(np.abs(np.diff(defect, axis=0)).max(axis=1)) / h)
        L_delta_u = float(np.max(np.abs(np.diff(defect[:, 0]))) / h)
        eps = {"conservative": (float(dG.max() + h / 2 * (L_G + lip_exact) + ref_err), float(du.max() + h / 2 * (L_g + lip_exact) + ref_err)),
               "modulus": (float(dG.max() + h / 2 * L_delta + ref_err), float(du.max() + h / 2 * L_delta_u + ref_err))}  # fmt: skip
        bounds = {}
        for key, (eG, eu) in eps.items():
            bound, A = [], 0.0
            for _n in range(1, STEPS):
                bound.append(eG + L_G * eu * A)
                A = A * L_g + 1.0
            A_prev = (A - 1.0) / L_g  # A_{n-1} for n = STEPS - 1
            bounds[key] = {"eps_G": eG, "eps_u": eu, "bound_by_step": bound, "bound_max": max(bound), "horizon_condition": bool(eu * A_prev <= RHO)}
        which = ["g1", "g2", "G3 (conservation)"][int(np.argmax([b1["dsup"], b2["dsup"], d3]))]
        # measured rollouts from step 1 with the surrogate
        Z = tests[:, 0].copy()
        meas = []
        for k in range(1, STEPS):
            Z = lift(Z[:, 0])
            meas.append(float(np.abs(Z - tests[:, k]).max()))
        rows.append({"degree": q, "oracle_steps": q + 1, "L_g": L_g, "L_G": L_G, "L_G_from": which, "eps_G_grid": float(dG.max()),
                     "reference_error_estimate": ref_err, "lip_exact_estimate": lip_exact, "L_delta_estimate": L_delta, "bounds": bounds,
                     "measured_by_step": meas, "measured_max": max(meas),
                     "chain_mass_offset_max": float(np.abs(np.log10(np.sum(10.0 ** Z, axis=1))).max())})  # fmt: skip
    out = {"rho": RHO, "U_rho": [lo, hi], "inputs_offset_from_curve": offset, "rows": rows}
    (OUT / "robertson_leaf.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"U_rho = [{lo:.4f}, {hi:.4f}]; held-out inputs' offset from the curve {offset:.2e} dex")
    for r in rows:
        b, c = r["bounds"]["modulus"], r["bounds"]["conservative"]
        print(f"q={r['degree']:2d} ({r['oracle_steps']} steps): L_g {r['L_g']:.4f} L_G {r['L_G']:.2f} ({r['L_G_from']}); defect on grid {r['eps_G_grid']:.1e}, ref {r['reference_error_estimate']:.1e}; "
              f"bound (modulus) {b['bound_max']:.2e} [ok {b['horizon_condition']}], bound (conservative) {c['bound_max']:.2e}; measured {r['measured_max']:.2e}; mass offset {r['chain_mass_offset_max']:.1e}")  # fmt: skip


if __name__ == "__main__":
    main()
