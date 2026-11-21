"""
Theorem 6.5 (memo section T6; the compound-matrix bound, review round 1's N7) and Corollary 6.1 where more than one
direction is fast: HIRES (8 species, Test Set for IVP Solvers; the right-hand side and Jacobian transcribed from
src/problems/ode/chemistry.py so this check does not import the library).

At states along the standard trajectory (y0 = (1, 0, 0, 0, 0, 0, 0, 0.0057)), for steps tau:
  the singular values sigma_1 >= ... >= sigma_8 of DPhi_tau (variational equation, Radau), in linear coordinates;
  the bound sigma_m <= exp((1/m) int_0^tau s_m), with s_m(t) the sum of the m largest eigenvalues of the symmetric
  part of J(x(t)) on that trajectory (Euclidean norm, no weighting);
  how many singular values exceed 1e-3 and 1e-6 (absolute), against Proposition 6.6's frozen-Jacobian count
  #{i : Re lambda_i(J(x0)) tau > ln eta} (exact up to the eigenvector condition number for a constant J).
The N7 bound is checked only on resolved singular values (above 1e-12 sigma_1; below that they are rounding).
Liouville fixes only the product of all eight; the individual small values are a test of the slow-fast picture.
"""

import numpy as np
from common import write
from scipy.integrate import solve_ivp


def rhs(_t, y):
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    f7 = 280.0 * y6 * y8 - 1.81 * y7
    return np.array([-1.71 * y1 + 0.43 * y2 + 8.32 * y3 + 0.0007, 1.71 * y1 - 8.75 * y2, -10.03 * y3 + 0.43 * y4 + 0.035 * y5,
                     8.32 * y2 + 1.71 * y3 - 1.12 * y4, -1.745 * y5 + 0.43 * (y6 + y7),
                     -280.0 * y6 * y8 + 0.69 * y4 + 1.71 * y5 - 0.43 * y6 + 0.69 * y7, f7, -f7])  # fmt: skip


def jac(y):
    J = np.zeros((8, 8))
    J[0, :3] = [-1.71, 0.43, 8.32]
    J[1, :2] = [1.71, -8.75]
    J[2, 2:5] = [-10.03, 0.43, 0.035]
    J[3, 1:4] = [8.32, 1.71, -1.12]
    J[4, 4:7] = [-1.745, 0.43, 0.43]
    J[5, 3:8] = [0.69, 1.71, -280.0 * y[7] - 0.43, 0.69, -280.0 * y[5]]
    J[6, 5:8] = [280.0 * y[7], -1.81, 280.0 * y[5]]
    J[7, 5:8] = [-280.0 * y[7], 1.81, -280.0 * y[5]]
    return J


def var(t, z):
    return np.concatenate([rhs(t, z[:8]), (jac(z[:8]) @ z[8:].reshape(8, 8)).ravel()])


def main() -> None:
    y0 = np.array([1.0, 0, 0, 0, 0, 0, 0, 0.0057])
    out = {"points": []}
    for t0 in (1.0, 10.0, 100.0):
        x0 = solve_ivp(rhs, (0, t0), y0, method="Radau", jac=lambda t, y: jac(y), rtol=1e-12, atol=1e-16).y[:, -1]
        lam = np.sort(np.linalg.eigvals(jac(x0)).real)
        rows = []
        for tau in (0.01, 0.1, 1.0, 10.0):
            sol = solve_ivp(var, (0, tau), np.concatenate([x0, np.eye(8).ravel()]), method="Radau", rtol=1e-12, atol=1e-16, dense_output=True)
            sig = np.linalg.svd(sol.y[8:, -1].reshape(8, 8), compute_uv=False)
            ts = np.linspace(0, tau, 401)
            ev = np.array([np.sort(np.linalg.eigvalsh((jac(sol.sol(t)[:8]) + jac(sol.sol(t)[:8]).T) / 2))[::-1] for t in ts])
            integral = np.trapezoid(np.cumsum(ev, axis=1), ts, axis=0)  # int_0^tau s_m, m = 1..8
            with np.errstate(over="ignore"):
                bound = np.exp(integral / np.arange(1, 9))
            resolved = sig > 1e-12 * sig[0]
            rows.append({"tau": tau, "sigma": sig.tolist(), "bound": bound.tolist(), "valid_resolved": bool(np.all(sig[resolved] <= bound[resolved] * (1 + 1e-6))),
                         "count_1e-3": int(np.sum(sig > 1e-3)), "count_1e-6": int(np.sum(sig > 1e-6)),
                         "predicted_1e-3": int(np.sum(lam * tau > np.log(1e-3))), "predicted_1e-6": int(np.sum(lam * tau > np.log(1e-6))),
                         "exp_lambda_tau": np.sort(np.exp(lam * tau))[::-1].tolist()})  # fmt: skip
        out["points"].append({"t0": t0, "eigenvalues_J": lam.tolist(), "rows": rows})
    path = write("t6_hires", out)
    for p in out["points"]:
        print(f"t0={p['t0']:g}: eig(J) = " + ", ".join(f"{v:.3g}" for v in p["eigenvalues_J"]))
        for r in p["rows"]:
            print(f"  tau={r['tau']:5g} N7 valid={r['valid_resolved']} counts(1e-3,1e-6)=({r['count_1e-3']},{r['count_1e-6']}) "
                  f"predicted=({r['predicted_1e-3']},{r['predicted_1e-6']})")  # fmt: skip
            print("     sigma      " + " ".join(f"{s:.1e}" for s in r["sigma"]))
            print("     e^(lam tau)" + " ".join(f"{s:.1e}" for s in r["exp_lambda_tau"]))
    print("wrote", path)


if __name__ == "__main__":
    main()
