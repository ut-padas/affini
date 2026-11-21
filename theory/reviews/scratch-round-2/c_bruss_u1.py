"""
Round-2 test C (the memo's first open item, the test of U1): the 4-D Brusselator's step map over tau = 0.25 and 0.5.

Part 1 (what makes the 2 dt map dearer). On the bounding box of the benchmark's visited states (100 training
trajectories of 40 steps of 0.25, log10 coordinates) the map is sampled on the full tensor Chebyshev-Lobatto grid of
degree 12 (13^4 solves per tau). From its Chebyshev coefficients, per output: the degree each axis needs for a sup
error of 1e-2 and 1e-3 dex (the smallest n with the sum of |c_k| over k_j > n below the tolerance), the decay rate
rho_eff per axis (envelope of max |c| over the other indices), the unfolding ranks of the coefficient tensor at the
tolerance (singular values above tol / 10 after scaling the grid values by 1 / sqrt(number of points)), and the cost
proxy sum_j r_{j-1} r_j (n_j + 1) of a tensor-train cross. Also a lower bound on the degree of any polynomial within
tol on the box, from Bernstein's inequality on each fibre:
  n_j >= (|f(b) - f(a)| - 2 tol) sqrt(1 - max(a^2, b^2)) / (|b - a| (osc / 2 + tol)).
Part 2 (Theorem 1 and Proposition 1.3). In the system's own variables on the same box: an upper bound of
m(t) = sup mu_beta(J) on the complex tube (entrywise bounds; trajectories from the 16 corners, the centre and 24
random points), Lambda, the certified rho per coordinate with beta optimised (two scales, one for u and one for v),
the measured rho_eff of the fibres through the centre (33 points), and Proposition 1.3's ceiling with the derivative
taken on those fibres (so the true ceiling is at most this).
"""
import itertools, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "checks"))
warnings.filterwarnings("ignore")
from common import cheb_coefficients, cheb_points, decay_rate, rho_from_halfwidth
from src.problems.ode.reaction_diffusion import BRUSS_A, BRUSS_B, BRUSS_ALPHA, bruss, bruss_initial
from src.problems.ode.system import SYSTEMS

SYS = bruss(2)
SYSTEMS[SYS.name] = SYS
N, THETA = 12, 0.999
out = {}

# the benchmark's visited states
Z = np.log10(bruss_initial(2)(100, 1))
states = [Z]
for _ in range(40):
    Z = SYS.coords_step(Z, 0.25)
    states.append(Z)
states = np.concatenate(states)
lo, hi = states.min(0), states.max(0)
out["box_log10"] = [lo.tolist(), hi.tolist()]
print("box (log10):", np.round(lo, 3), np.round(hi, 3), flush=True)

# ---- part 1
t = cheb_points(N)  # on [-1, 1]
grid = np.stack(np.meshgrid(*[0.5 * (lo[j] + hi[j]) + 0.5 * (hi[j] - lo[j]) * t for j in range(4)], indexing="ij"), axis=-1).reshape(-1, 4)


def coefficients(V):
    """Chebyshev coefficients of a (13, 13, 13, 13) value tensor, along every axis."""
    for ax in range(4):
        V = np.moveaxis(cheb_coefficients(np.moveaxis(V, ax, 0)), 0, ax)
    return V


out["part1"] = {}
for tau in (0.25, 0.5):
    F = SYS.coords_step(grid, tau).reshape(N + 1, N + 1, N + 1, N + 1, 4)
    res = {"degree": {}, "rho_eff": [], "ranks": {}, "cost_proxy": {}, "bernstein_lower_bound": {}}
    C = np.stack([coefficients(F[..., i]) for i in range(4)], axis=-1)
    for j in range(4):
        env = np.abs(np.moveaxis(C, j, 0)).reshape(N + 1, -1, 4).max(1)  # (k, output)
        res["rho_eff"].append([decay_rate(env[:, i], floor=1e-9)["rho"] for i in range(4)])
    for tol in (1e-2, 1e-3):
        deg = np.zeros((4, 4), int)  # (axis, output)
        for j in range(4):
            tail = np.abs(np.moveaxis(C, j, 0)).reshape(N + 1, -1, 4).sum(1)  # sum over other indices, per k_j
            tails = np.cumsum(tail[::-1], axis=0)[::-1]  # sum over k_j >= k
            for i in range(4):
                ok = np.nonzero(np.append(tails[1:, i], 0.0) <= tol)[0]
                deg[j, i] = ok[0]
        ranks = np.zeros((3, 4), int)
        for i in range(4):
            for bnd in range(1, 4):
                sv = np.linalg.svd(F[..., i].reshape((N + 1) ** bnd, -1) - F[..., i].mean(), compute_uv=False) / np.sqrt((N + 1) ** 4)
                ranks[bnd - 1, i] = int(np.sum(sv > tol / 10))
        cost = [int(sum(([1] + list(ranks[:, i]) + [1])[j] * ([1] + list(ranks[:, i]) + [1])[j + 1] * (deg[j, i] + 1) for j in range(4))) for i in range(4)]
        lower = np.zeros((4, 4))
        for j in range(4):
            Fj = np.moveaxis(F, j, 0).reshape(N + 1, -1, 4)  # fibres along axis j
            osc = Fj.max(0) - Fj.min(0)
            for a in range(N + 1):
                for b in range(a + 1, N + 1):
                    val = (np.abs(Fj[b] - Fj[a]) - 2 * tol) * np.sqrt(max(0.0, 1 - max(t[a] ** 2, t[b] ** 2))) / (abs(t[b] - t[a]) * (osc / 2 + tol))
                    lower[j] = np.maximum(lower[j], val.max(0))
        k = f"{tol:g}"
        res["degree"][k], res["ranks"][k], res["cost_proxy"][k], res["bernstein_lower_bound"][k] = deg.tolist(), ranks.tolist(), cost, lower.tolist()
        print(f"tau={tau} tol={tol:g}: degree per axis (rows) and output (columns):\n{deg}\n ranks at the three bonds (rows), per output:\n{ranks}\n"
              f" cross cost proxy per output {cost}, total {sum(cost)};  Bernstein lower bound on the degree per axis (max over outputs): {np.round(lower.max(1), 2)}", flush=True)
    res["last_coefficient_max"] = float(max(np.abs(np.moveaxis(C, j, 0)[N]).max() for j in range(4)))
    print(f"tau={tau}: rho_eff per axis (min over outputs) {np.round(np.min(res['rho_eff'], axis=1), 2)}; largest |c| at degree 12: {res['last_coefficient_max']:.1e}", flush=True)
    out["part1"][f"{tau:g}"] = res
for tol in ("0.01", "0.001"):
    a, b = sum(out["part1"]["0.25"]["cost_proxy"][tol]), sum(out["part1"]["0.5"]["cost_proxy"][tol])
    print(f"tol={tol}: cost proxy of Phi_0.5 over Phi_0.25: {b / a:.2f} (records: 11k against 3.8k solves, 2.9)")

# ---- part 2 (the system's own variables)
A, B = BRUSS_A, BRUSS_B
ylo, yhi = 10.0**lo, 10.0**hi
yc, h = (ylo + yhi) / 2, (yhi - ylo) / 2
rng = np.random.default_rng(0)
starts = [yc] + [yc + h * np.array(s) for s in itertools.product((-1, 1), repeat=4)] + list(yc + h * rng.uniform(-1, 1, (24, 4)))
L = BRUSS_ALPHA * 4.0 * np.array([[-1.0, 1.0], [1.0, -1.0]])
out["part2"] = {}
for tau in (0.25, 0.5):
    tg = np.linspace(0, tau, 161)
    S = np.stack([solve_ivp(lambda _t, y: SYS.rhs(y), (0, tau), s, method="Radau", jac=lambda _t, y: SYS.jac(y), rtol=1e-10, atol=1e-13, dense_output=True).sol(tg).T for s in starts])
    U, V = S[..., :2], S[..., 2:]  # (trajectory, time, site)

    def Lam(logb):
        bu, bv = np.exp(logb[0]) * h[:2], np.exp(logb[1]) * h[2:]
        cross = 2 * (U * bv + V * bu + bu * bv)
        row_u = 2 * U * V - (B + 1) + L[0, 0] + cross + (U + bu) ** 2 * bv / bu + L[0, 1] * bu[::-1] / bu
        row_v = -(U**2 - 2 * U * bu - bu**2) + L[0, 0] + (np.abs(B - 2 * U * V) + cross) * bu / bv + L[0, 1] * bv[::-1] / bv
        m = np.maximum(row_u, row_v).max(axis=(0, 2))
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(tg))])
        return float(cum.max()), float(m.max()), float(m.min())

    best = min((minimize(lambda lb: -(min(lb) - Lam(lb)[0]), s0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7}) for s0 in ([-2.0, -2.0], [-1.0, -1.0], [-3.0, -2.0])), key=lambda o: o.fun)
    lam, mmax, mmin = Lam(best.x)
    beta_over_h = np.repeat(np.exp(best.x), 2)
    rho_thm = np.array([rho_from_halfwidth(THETA * b * np.exp(-lam)) for b in beta_over_h])
    M = np.abs(S[:, -1]).max(0) + THETA * beta_over_h * h
    mu_real = max(np.max(np.linalg.eigvalsh((J + J.T) / 2)) for J in (SYS.jac(s) for s in S[:, 0]))
    rho_meas, ceiling, dmax = [], [], []
    for j in range(4):
        X = np.tile(yc, (33, 1))
        X[:, j] = cheb_points(32, ylo[j], yhi[j])
        vals = SYS.step(X, tau)
        c = cheb_coefficients(vals)
        rho_meas.append(min(decay_rate(c[:, i], floor=1e-9)["rho"] for i in range(4)))
        # derivative of each output along the fibre, at the nodes, in units of 1 / h_j
        k = np.arange(33)
        dT = np.array([np.polynomial.chebyshev.chebval(cheb_points(32), np.polynomial.chebyshev.chebder(np.eye(33)[q])) for q in k])  # (k, node)
        deriv = np.abs(c.T @ dT).max(1)  # per output: max |d Phi_i / d s|, s in [-1, 1]
        dmax.append(deriv.tolist())
        g = np.min(M / deriv)  # (rho - 1)^2 / (2 rho) <= g
        ceiling.append(float(1 + g + np.sqrt(g * g + 2 * g)))
    row = {"Lambda": lam, "m_max": mmax, "m_min": mmin, "beta_over_h": beta_over_h.tolist(), "rho_thm": rho_thm.tolist(), "rho_measured": rho_meas, "rho_ceiling": ceiling,
           "h_times_derivative": dmax, "degree_thm_2digits": (np.log(1e2) / np.log(rho_thm)).tolist(), "degree_measured_2digits": (np.log(1e2) / np.log(rho_meas)).tolist(),
           "mu2_real_at_starts": float(mu_real)}
    out["part2"][f"{tau:g}"] = row
    print(f"tau={tau}: Lambda={lam:.3f} (m between {mmin:.2f} and {mmax:.2f}), beta/h={np.round(beta_over_h, 4)}\n  rho_thm={np.round(rho_thm, 4)} rho_measured={np.round(rho_meas, 2)} "
          f"rho_ceiling={np.round(ceiling, 1)}\n  degree for 2 digits: certified {np.round(row['degree_thm_2digits'], 0)} measured {np.round(row['degree_measured_2digits'], 1)}; "
          f"h |dPhi/dx| max per axis {np.round(np.max(dmax, axis=1), 2)}", flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
