"""
Theorem 7 (memo section T7): the cost gap between sup-norm and measure-sense accuracy for functions of an
aggregate coordinate, f(x) = g(s), s = mean(x), x uniform on [-1, 1]^d, g(s) = 1 / (1 + a s) with a = 0.8
(a pole at s = -1.25, so g is analytic in the Bernstein ellipse of [-1, 1] with rho = 1.25 + sqrt(1.25^2 - 1) = 2).

Hoeffding: P(|s| >= t) <= 2 exp(-d t^2 / 2), so with t_eta = sqrt(2 ln(2 / eta) / d) a polynomial accurate to tol on
[-t_eta, t_eta] violates tol on a share of the box at most eta. Checks, per d:
  q_sup  = the smallest Chebyshev degree in s with sup over s in [-1, 1] of |g - p| <= tol;
  q_meas = the smallest degree with sup over [-t_eta, t_eta] of |g - p| <= tol (p interpolates on that interval);
  the measured violation share of the q_meas polynomial on 10^5 uniform points (must be <= eta) and its error at
  the vertices s = +-1 (in units of tol). A polynomial of degree q in s has TT ranks <= q + 1 (Lemma 7.2), so the
  rank needed in the two senses follows q_sup and q_meas.
"""

import numpy as np
from common import cheb_points, write

A_POLE = 0.8
TOL = 1e-3
ETA = 1e-3
DIMS = [2, 5, 10, 20, 50, 100]


def g(s):
    return 1.0 / (1.0 + A_POLE * s)


def interp(lo: float, hi: float, q: int):
    nodes = cheb_points(q, lo, hi) if q > 0 else np.array([(lo + hi) / 2])
    coef = np.polynomial.chebyshev.chebfit((2 * nodes - lo - hi) / (hi - lo), g(nodes), q)
    return lambda s: np.polynomial.chebyshev.chebval((2 * s - lo - hi) / (hi - lo), coef)


def min_degree(lo: float, hi: float) -> int:
    grid = np.linspace(lo, hi, 20001)
    for q in range(200):
        if np.max(np.abs(g(grid) - interp(lo, hi, q)(grid))) <= TOL:
            return q
    raise RuntimeError("no degree below 200")


def main() -> None:
    rng = np.random.default_rng(0)
    q_sup = min_degree(-1.0, 1.0)
    rows = []
    for d in DIMS:
        t = min(1.0, float(np.sqrt(2 * np.log(2 / ETA) / d)))
        q = min_degree(-t, t)
        p = interp(-t, t, q)
        s = rng.uniform(-1, 1, (100_000, d)).mean(axis=1)
        share = float(np.mean(np.abs(g(s) - p(s)) > TOL))
        vertex = float(max(abs(g(1.0) - p(1.0)), abs(g(-1.0) - p(-1.0))) / TOL)
        t_emp = float(np.quantile(np.abs(s), 1 - ETA))  # the sharp radius (not a bound): the measured (1 - eta) quantile
        q_emp = min_degree(-t_emp, t_emp)
        rows.append({"d": d, "t_eta": t, "q_meas": q, "t_empirical": t_emp, "q_meas_empirical": q_emp, "q_sup": q_sup, "violation_share": share, "eta": ETA, "vertex_error_over_tol": vertex,
                     "rank_bound_meas": q + 1, "rank_bound_sup": q_sup + 1})  # fmt: skip
    path = write("t7_concentration", {"a": A_POLE, "tol": TOL, "eta": ETA, "rows": rows})
    for r in rows:
        print(f"d={r['d']:3d} t_eta={r['t_eta']:.3f} q_meas={r['q_meas']:2d} (empirical t={r['t_empirical']:.3f}: {r['q_meas_empirical']}) q_sup={r['q_sup']} share={r['violation_share']:.1e} (eta {ETA}) "
              f"vertex error/tol={r['vertex_error_over_tol']:.3g}")  # fmt: skip
    print("wrote", path)


if __name__ == "__main__":
    main()
