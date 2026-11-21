"""
Review test B: Theorem 1 evaluated on a stiff problem (the memo's checks only use non-stiff ones).

Robertson in the original coordinates (F polynomial, J affine in y). Box: centre y* = y(100) of the standard
trajectory, half-widths h = REL * y* (REL = 0.5 and 0.1). For a weight vector beta the theorem needs
  m(t) = sup { mu_beta(J(x(t; x0) + v)) : x0 in B, |v_j| <= beta_j },  Lambda = max_t int_0^t m,
  h_j (rho_j - 1/rho_j) / 2 = theta beta_j e^{-Lambda}.
m is computed over the trajectories from the box's centre and 8 corners and over the distinguished boundary
|v_j| = beta_j (mu_beta(J(x + v)) is convex in v, so its sup over the polydisc is attained there); beta is then
optimised (Nelder-Mead on log beta, started from the comparison matrix's Perron vector at several scales) to maximise min_j R_j / h_j: the best the theorem can give.
Compared with rho_eff measured from the Chebyshev coefficients of Phi_tau along each input fibre through the
centre (the smallest over the three outputs). Also printed: the spectral abscissa of the comparison matrix
(Re a_ii, |a_ij|) at y*, a lower bound for mu_beta(J(y*)) over every beta > 0 (Collatz-Wielandt).
"""
import json, sys, itertools
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "checks"))
from common import cheb_coefficients, cheb_points, decay_rate, rho_from_halfwidth

K1, K2, K3, THETA = 0.04, 3e7, 1e4, 0.999
rhs = lambda _t, y: [-K1 * y[0] + K3 * y[1] * y[2], K1 * y[0] - K3 * y[1] * y[2] - K2 * y[1] ** 2, K2 * y[1] ** 2]
jac = lambda _t, y: [[-K1, K3 * y[2], K3 * y[1]], [K1, -K3 * y[2] - 2 * K2 * y[1], -K3 * y[1]], [0.0, 2 * K2 * y[1], 0.0]]
YC = solve_ivp(rhs, (0, 100.0), [1.0, 0.0, 0.0], method="Radau", jac=jac, rtol=1e-12, atol=1e-20).y[:, -1]
PH = np.exp(1j * np.linspace(0, 2 * np.pi, 16, endpoint=False))

def trajectories(h, tau):
    tg = np.concatenate([[0.0], np.logspace(-7, np.log10(tau), 160)])
    starts = [YC] + [YC + h * np.array(s) for s in itertools.product((-1, 1), repeat=3)]
    S = np.stack([solve_ivp(rhs, (0, tau), s, method="Radau", jac=jac, rtol=1e-11, atol=1e-18, dense_output=True).sol(tg).T for s in starts])
    return tg, S  # S: (traj, time, 3)

def Lambda(logb, tg, S):
    b1, b2, b3 = np.exp(logb)
    y2 = S[:, :, 1][..., None, None] + b2 * PH[None, None, :, None]
    y3 = S[:, :, 2][..., None, None] + b3 * PH[None, None, None, :]
    r1 = -K1 + (K3 * np.abs(y3) * b2 + K3 * np.abs(y2) * b3) / b1
    r2 = np.real(-K3 * y3 - 2 * K2 * y2) + (K1 * b1 + K3 * np.abs(y2) * b3) / b2
    r3 = 2 * K2 * np.abs(y2) * b2 / b3
    m = np.max(np.maximum(np.maximum(r1, r2), r3 + 0 * r1), axis=(0, 2, 3))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(tg))])
    return float(cum.max())

def rho_thm(logb, h, tg, S):
    lam = Lambda(logb, tg, S)
    return np.array([rho_from_halfwidth(THETA * np.exp(lb - lam) / hj) for lb, hj in zip(logb, h)]), lam

def flow(y0, tau):
    return solve_ivp(rhs, (0, tau), y0, method="Radau", jac=jac, rtol=1e-12, atol=1e-20).y[:, -1]

def measured(h, tau, n=32):
    out = []
    for j in range(3):
        vals = []
        for x in cheb_points(n, YC[j] - h[j], YC[j] + h[j]):
            y0 = YC.copy(); y0[j] = x
            vals.append(flow(y0, tau))
        c = cheb_coefficients(np.array(vals))
        out.append(min(decay_rate(c[:, i], floor=1e-8)["rho"] for i in range(3)))
    return out

C = np.abs(np.array(jac(0, YC))); C[np.diag_indices(3)] = np.diag(np.array(jac(0, YC)))
w_, V_ = np.linalg.eig(C)
alpha = float(np.max(w_.real))
PERRON = np.abs(V_[:, np.argmax(w_.real)].real)
print("centre state", YC, " eig J:", np.sort(np.linalg.eigvals(np.array(jac(0, YC))).real))
print("spectral abscissa of the comparison matrix at y* (inf over beta of mu_beta(J(y*))):", alpha)
res = {"center": YC.tolist(), "comparison_abscissa": alpha, "cases": []}
for rel in (0.5, 0.1):
    h = rel * YC
    for tau in (0.01, 0.1, 1.0, 10.0):
        tg, S = trajectories(h, tau)
        best = None
        # smooth objective: log(R_j / h_j) = log beta_j - Lambda - log h_j (rho_j - 1 ~ R_j / h_j when small), worst coordinate;
        # starts: the comparison matrix's Perron vector (the weights minimising mu_beta(J(y*))), scaled so max_j p_j / h_j = c
        for c0 in (1e-2, 1e-1, 0.5, 2.0):
            start = np.log(c0 * PERRON / np.max(PERRON / h))
            o = minimize(lambda lb: -(np.min(lb - np.log(h)) - Lambda(lb, tg, S)), start, method="Nelder-Mead",
                         options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400})
            if best is None or o.fun < best.fun:
                best = o
        rho, lam = rho_thm(best.x, h, tg, S)
        meas = measured(h, tau)
        deg = lambda r: float(np.log(1e3) / np.log(r))  # degree for three digits, per coordinate
        row = {"rel": rel, "tau": tau, "beta_over_h": (np.exp(best.x) / h).tolist(), "Lambda": lam, "rho_thm": rho.tolist(), "rho_measured": meas,
               "degree_thm_3digits": [deg(r) for r in rho], "degree_measured_3digits": [deg(r) if np.isfinite(r) else 1.0 for r in meas]}
        res["cases"].append(row)
        print(f"rel={rel} tau={tau:5}: Lambda={lam:.3f} beta/h={np.round(np.exp(best.x)/h,4)} rho_thm-1={np.array2string(rho-1,precision=3)} "
              f"rho_meas={np.array2string(np.array(meas),precision=3)}", flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
