"""
Theorem 6 (memo section T6): the effective input dimension of a stiff step map.

(a) Michaelis-Menten in Segel-Slemrod form, s' = -s + (s + kappa - lam) c, eps c' = s - (s + kappa) c
    (kappa = 1, lam = 0.5; box s in [0.2, 1], c in [0, 1]; slow dimension k = 1). Predictions:
    (i)  sigma_2(DPhi_tau) <= C exp(-gamma tau / eps), gamma near the fast rate's lower bound s + kappa;
    (ii) the best surrogate that ignores the fast variable c (a function of s alone) has error ~ eps (the fast
         fibres are O(eps)-tilted from the lines s = const), once exp(-gamma tau / eps) is negligible.
(b) Robertson in log coordinates (the benchmark's chemistry coordinates), at states along its standard
    trajectory: the singular values of DPhi_tau collapse to k = 2 large ones within a few fast times
    1 / |lambda_fast|.
"""

import numpy as np
from common import write
from scipy.integrate import solve_ivp

KAPPA, LAM = 1.0, 0.5


def mm_rhs(eps):
    def f(_t, y):
        s, c = y[0], y[1]
        return [-s + (s + KAPPA - LAM) * c, (s - (s + KAPPA) * c) / eps]

    def jac(_t, y):
        s, c = y[0], y[1]
        return [[-1 + c, s + KAPPA - LAM], [(1 - c) / eps, -(s + KAPPA) / eps]]

    return f, jac


def mm_var(eps):
    f, jac = mm_rhs(eps)

    def g(t, y):
        J = np.array(jac(t, y[:2]))
        return np.concatenate([f(t, y[:2]), (J @ y[2:].reshape(2, 2)).ravel()])

    return g


def mm_flow(y0, tau, eps):
    f, jac = mm_rhs(eps)
    return solve_ivp(f, (0, tau), y0, method="Radau", jac=jac, rtol=1e-11, atol=1e-13).y[:, -1]


def mm_jacobian(y0, tau, eps):
    sol = solve_ivp(mm_var(eps), (0, tau), np.concatenate([y0, np.eye(2).ravel()]), method="Radau", rtol=1e-11, atol=1e-14)
    return sol.y[2:, -1].reshape(2, 2)


def michaelis_menten() -> dict:
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0.2, 1.0, 12), rng.uniform(0.0, 1.0, 12)])
    eps = 0.01
    sig = []
    for tau in (0.01, 0.02, 0.05, 0.1, 0.2):
        s2 = [np.linalg.svd(mm_jacobian(p, tau, eps), compute_uv=False)[1] for p in pts]
        sig.append({"tau": tau, "tau_over_eps": tau / eps, "sigma2_max": float(np.max(s2)), "sigma2_min": float(np.min(s2))})
    ok = [q for q in sig if q["sigma2_max"] > 1e-12]
    slope = float(np.polyfit([q["tau_over_eps"] for q in ok], np.log([q["sigma2_max"] for q in ok]), 1)[0])
    red = []
    S, C = np.linspace(0.2, 1.0, 9), np.linspace(0.0, 1.0, 9)
    for e in (0.1, 0.03, 0.01, 0.003, 0.001):
        vals = np.array([[mm_flow([s, c], 0.5, e) for c in C] for s in S])  # (s, c, component)
        spread = (vals.max(axis=1) - vals.min(axis=1)) / 2  # best error of a function of s alone, per s and component
        red.append({"eps": e, "reduction_error_s": float(spread[:, 0].max()), "reduction_error_c": float(spread[:, 1].max()),
                    "exp_term": float(np.exp(-(0.2 + KAPPA) * 0.5 / e))})  # fmt: skip
    big = [q for q in red if q["exp_term"] < 1e-6]
    slope_eps = float(np.polyfit(np.log([q["eps"] for q in big]), np.log([q["reduction_error_s"] for q in big]), 1)[0])
    return {"eps_for_sigma": eps, "sigma2": sig, "slope_log_sigma2_vs_tau_over_eps": slope, "reduction": red, "slope_loglog_reduction_vs_eps": slope_eps}


def rob_rhs(_t, y, k1=0.04, k2=3e7, k3=1e4):
    y1, y2, y3 = y
    return [-k1 * y1 + k3 * y2 * y3, k1 * y1 - k3 * y2 * y3 - k2 * y2 * y2, k2 * y2 * y2]


def rob_jac(y, k1=0.04, k2=3e7, k3=1e4):
    y1, y2, y3 = y
    return np.array([[-k1, k3 * y3, k3 * y2], [k1, -k3 * y3 - 2 * k2 * y2, -k3 * y2], [0.0, 2 * k2 * y2, 0.0]])


def rob_var(t, y):
    J = rob_jac(y[:3])
    return np.concatenate([rob_rhs(t, y[:3]), (J @ y[3:].reshape(3, 3)).ravel()])


def robertson() -> dict:
    out = []
    for t0 in (1e-2, 1.0, 1e2, 1e4):
        y0 = solve_ivp(rob_rhs, (0, t0), [1.0, 0.0, 0.0], method="Radau", rtol=1e-12, atol=1e-20).y[:, -1]
        lam = np.sort(np.linalg.eigvals(rob_jac(y0)).real)
        rows = []
        for tau in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            sol = solve_ivp(rob_var, (0, tau), np.concatenate([y0, np.eye(3).ravel()]), method="Radau", rtol=1e-12, atol=1e-22)
            y1, M = sol.y[:3, -1], sol.y[3:, -1].reshape(3, 3)
            Mlog = np.diag(1 / y1) @ M @ np.diag(y0)  # log coordinates: d log y(tau) / d log y0
            rows.append({"tau": tau, "tau_times_fast_rate": float(tau * abs(lam[0])), "sigma": np.linalg.svd(Mlog, compute_uv=False).tolist()})
        out.append({"t0": t0, "state": y0.tolist(), "eigenvalues_J": lam.tolist(), "rows": rows})
    return {"points": out}


def main() -> None:
    out = {"michaelis_menten": michaelis_menten(), "robertson": robertson()}
    path = write("t6_slowfast", out)
    mm = out["michaelis_menten"]
    print("MM sigma_2 (eps=0.01):", [(q["tau_over_eps"], f"{q['sigma2_max']:.2e}") for q in mm["sigma2"]], "slope", round(mm["slope_log_sigma2_vs_tau_over_eps"], 3))
    print("MM reduction error:", [(q["eps"], f"{q['reduction_error_s']:.2e}", f"{q['reduction_error_c']:.2e}") for q in mm["reduction"]],
          "slope", round(mm["slope_loglog_reduction_vs_eps"], 3))  # fmt: skip
    for p in out["robertson"]["points"]:
        print(f"Robertson t0={p['t0']:g} fast eig={p['eigenvalues_J'][0]:.3g}")
        for q in p["rows"]:
            print(f"   tau={q['tau']:g} (tau*|lam|={q['tau_times_fast_rate']:.2g}): sigma=" + ", ".join(f"{s:.2e}" for s in q["sigma"]))
    print("wrote", path)


if __name__ == "__main__":
    main()
