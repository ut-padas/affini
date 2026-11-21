"""
Theorem 1 on two scalar ODEs with closed-form flow maps (memo section T1).

x' = -x - x^3 (dissipative) and x' = x - x^3 (expanding at 0), on B = [-1, 1]. The exact maps are
Phi(z) = z e^{-tau} / sqrt(1 + z^2 (1 - e^{-2 tau})) and Phi(z) = z e^{tau} / sqrt(1 + z^2 (e^{2 tau} - 1)), whose
only singularities are the branch points +- i r*. Checks, for each tau:
  1. the theorem's log-norm bound m(t) = sup Re J(x(t) + v) over |v| <= r equals the closed form used below
     (brute force over the disk, at the worst trajectory x = 0);
  2. validity: rho_thm <= rho_exact, the largest Bernstein ellipse of [-1, 1] avoiding +- i r*;
  3. the Chebyshev coefficients of the exact map obey |c_k| <= 2 M rho_thm^-k (c_0: M);
  4. the predicted tau-dependence: rho tends to a constant (dissipative) or rho - 1 ~ e^{-tau} (expanding).
"""

import numpy as np
from common import cheb_coefficients, cheb_points, rho_from_halfwidth, write

THETA = 0.999
TAUS = [0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 4.0, 8.0]
R_GRID = np.linspace(1e-3, 30.0, 300_001)

CASES = {
    "dissipative": {
        "sign": -1.0,  # x' = sign * x - x^3
        "phi": lambda z, t: z * np.exp(-t) / np.sqrt(1 + z * z * (1 - np.exp(-2 * t))),
        "r_star": lambda t: 1 / np.sqrt(1 - np.exp(-2 * t)),
    },
    "expanding": {
        "sign": 1.0,
        "phi": lambda z, t: z * np.exp(t) / np.sqrt(1 + z * z * (np.exp(2 * t) - 1)),
        "r_star": lambda t: 1 / np.sqrt(np.exp(2 * t) - 1),
    },
}


def m_closed(sign: float, r: np.ndarray) -> np.ndarray:
    """sup over x in [-1, 1], |v| <= r of Re(sign - 3 (x + v)^2): attained at x = 0, v = i r."""
    return sign + 3 * r * r


def m_brute(sign: float, r: float) -> float:
    """The same supremum by brute force over x on a grid and v on the closed disk |v| <= r."""
    x = np.linspace(-1, 1, 201)[:, None]
    rad = np.sqrt(np.linspace(0, 1, 60))[None, :, None] * r
    ang = np.linspace(0, 2 * np.pi, 241)[None, None, :]
    v = rad * np.exp(1j * ang)
    z = x[:, :, None] + v
    return float(np.max(np.real(sign - 3 * z * z)))


def main() -> None:
    out = {"theta": THETA, "taus": TAUS, "cases": {}}
    for name, case in CASES.items():
        sign = case["sign"]
        checks_m = [{"r": r, "closed": float(m_closed(sign, np.array(r))), "brute": m_brute(sign, r)} for r in (0.1, 0.5, 1.0, 2.0)]
        rows = []
        for t in TAUS:
            lam = np.maximum(0.0, m_closed(sign, R_GRID) * t)  # Lambda = max over s in [0, t] of int_0^s m = max(0, m t)
            radius = R_GRID * np.exp(-lam)
            i = int(np.argmax(radius))
            r_opt, R = float(R_GRID[i]), float(radius[i])
            rho_thm = rho_from_halfwidth(THETA * R)
            r_star = float(case["r_star"](t))
            rho_exact = rho_from_halfwidth(r_star)
            x = cheb_points(256)
            c = cheb_coefficients(case["phi"](x, t))
            M = float(np.max(np.abs(case["phi"](np.linspace(-1, 1, 20001), t)))) + THETA * r_opt
            k = np.arange(len(c))
            bound = np.where(k == 0, M, 2 * M * rho_thm ** (-k.astype(float)))
            resolved = np.abs(c) > 1e-13 * np.max(np.abs(c))  # below this the coefficients are rounding noise
            ratio = float(np.max(np.abs(c[resolved]) / bound[resolved]))
            rows.append({"tau": t, "r_opt": r_opt, "radius_thm": R, "r_star": r_star, "rho_thm": rho_thm, "rho_exact": rho_exact,
                         "valid": rho_thm <= rho_exact, "log_rho_ratio": float(np.log(rho_thm) / np.log(rho_exact)),
                         "M": M, "max_coef_over_bound": ratio})  # fmt: skip
        out["cases"][name] = {"m_check": checks_m, "rows": rows}
    taus = np.array(TAUS)
    big = taus >= 1.0
    for name in out["cases"]:
        rows = out["cases"][name]["rows"]
        thm = np.log(np.array([r["rho_thm"] for r in rows]) - 1)
        ex = np.log(np.array([r["rho_exact"] for r in rows]) - 1)
        out["cases"][name]["slope_log_rho_minus_1_vs_tau_thm"] = float(np.polyfit(taus[big], thm[big], 1)[0])
        out["cases"][name]["slope_log_rho_minus_1_vs_tau_exact"] = float(np.polyfit(taus[big], ex[big], 1)[0])
    path = write("t1_scalar", out)
    for name, res in out["cases"].items():
        print(name, "m check", [(c["r"], round(c["closed"], 4), round(c["brute"], 4)) for c in res["m_check"]])
        for r in res["rows"]:
            print(f"  tau={r['tau']:5}: rho_thm={r['rho_thm']:.4f} rho_exact={r['rho_exact']:.4f} valid={r['valid']} "
                  f"max|c|/bound={r['max_coef_over_bound']:.3g}")  # fmt: skip
        print("  slope of log(rho-1) vs tau (tau>=1): thm", round(res["slope_log_rho_minus_1_vs_tau_thm"], 3),
              "exact", round(res["slope_log_rho_minus_1_vs_tau_exact"], 3))  # fmt: skip
    print("wrote", path)


if __name__ == "__main__":
    main()
