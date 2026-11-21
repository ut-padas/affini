"""
Function-suite pass, smooth and kinked classes (memo Part I): scaling laws predicted from where each function's
singularities are, checked against fresh ChebTT runs at d = 2 and against the benchmark's records.

Predictions (tier 1: exponents and counts, not constants):
  product peak  prod_i 1/(a_i^-2 + (x_i - w_i)^2) on [0, 1]^d: separable, so TT rank 1; each factor has poles at w_i +- i/a_i,
                so the degree in x_i grows by 1/log10(rho_i) per decade of tolerance (Bernstein), rho_i the ellipse through the pole.
  oscillatory   cos(2 pi w_1 + a.x): the cosine of a sum, so TT rank exactly 2 at every tolerance.
  corner peak   (1 + a.x)^-(d+1): a function of the aggregate s = a.x in [0, sum a] = [0, 1.85] with a pole at s = -1, so by
                Lemma 7.2 the TT rank grows by at most 1/log10(rho_s) = 1.69 per decade, in every dimension.
  continuous    exp(-sum_i a_i |x_i - w_i|): separable with a kink per axis. Measured directly on one factor, exp(-a |x - w|): the
                degree-n Chebyshev interpolant's sup error falls like n^-1 (coefficients ~ k^-2), so the degree grows like 1/tol;
                with a breakpoint at w each piece is entire and the error falls geometrically. ChebTT with breakpoints on the
                whole function: rank 1 and a cost nearly flat in the tolerance. (ChebTT without breakpoints did not finish at a
                relative tolerance of 0.1 within 170 s, so it is not run here.)
  discontinuous exp(a.x) inside a corner, 0 outside: on one axis a jump of height J; no polynomial gets within J/2 of it uniformly,
                so the interpolant's sup error does not tend to 0; with a breakpoint at the jump it falls geometrically.
Tolerances are relative to max |f| on the box.
"""

import json
import math
from pathlib import Path

import numpy as np

from src.problems.registry import FUNCTIONS
from src.schemes import chebtt

OUT = Path(__file__).resolve().parent / "data"


def bernstein_rho(z: complex, lo: float, hi: float) -> float:
    zeta = (2 * z - lo - hi) / (hi - lo)
    r = zeta + np.sqrt(zeta * zeta - 1)
    return float(max(abs(r), 1 / abs(r)))


def genz_params(kind_sum: float, d: int):
    rng = np.random.default_rng(7_000 + d)
    a = rng.uniform(0.0, 1.0, d)
    a = a * kind_sum / a.sum()
    w = rng.uniform(0.0, 1.0, d)
    return a, w


def fresh(name: str, d: int, tols, **kw) -> list:
    prob = FUNCTIONS[name]
    f = lambda X: np.asarray(prob.f(X)).reshape(len(X), -1)[:, 0]  # noqa: E731
    lo, hi = np.full(d, prob.lo), np.full(d, prob.hi)
    scale = float(np.abs(f(np.random.default_rng(0).uniform(prob.lo, prob.hi, (20_000, d)))).max())
    rows = []
    for t in tols:
        m = chebtt.build(f, lo, hi, t * scale, seed=1, max_evals=400_000, **kw)
        rows.append({"tol": t, "evals": int(m.evals), "degrees": [int(x) for x in m.degrees], "ranks": [int(x) for x in m.ranks],
                     "pieces": [m.pieces(k) for k in range(d)]})  # fmt: skip
    return rows


def slope(x, y) -> float:
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    out = {}
    # product peak, d = 2: degree per decade against Bernstein's prediction
    a, w = genz_params(7.25, 2)
    rho = [bernstein_rho(w[i] + 1j / a[i], 0.0, 1.0) for i in range(2)]
    tols = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    rows = fresh("genz_product_peak", 2, tols, edges=False)
    dec = [math.log10(1 / r["tol"]) for r in rows]
    out["product_peak_d2"] = {"rho": rho, "predicted_degree_per_decade": [1 / math.log10(r) for r in rho],
                              "measured_degree_per_decade": [slope(dec, [r["degrees"][i] for r in rows]) for i in range(2)],
                              "ranks": sorted({x for r in rows for x in r["ranks"]}), "rows": rows}  # fmt: skip
    # oscillatory and corner peak: the train's ranks (fresh runs; the records' "ranks" field is the MPI rank count)
    rtols = [1e-2, 1e-4, 1e-6, 1e-8]
    out["oscillatory_ranks"] = {d: [max(r["ranks"]) for r in fresh("genz_oscillatory", d, rtols, edges=False)] for d in (2, 3, 5)}
    rho_s = bernstein_rho(-1.0 + 0j, 0.0, 1.85)
    cp = {d: [max(r["ranks"]) for r in fresh("genz_corner_peak", d, rtols, edges=False)] for d in (2, 3, 5)}
    out["corner_peak"] = {"rho_s": rho_s, "predicted_rank_per_decade_at_most": 1 / math.log10(rho_s), "tols": rtols, "max_ranks": cp,
                          "measured_rank_per_decade": {d: slope([math.log10(1 / t) for t in rtols], v) for d, v in cp.items()}}  # fmt: skip
    # continuous and discontinuous: one axis's profile, interpolated with and without a breakpoint
    a, w = genz_params(20.4, 2)
    xs = np.linspace(0, 1, 200_001)
    kink = lambda x: np.exp(-a[0] * np.abs(x - w[0]))  # noqa: E731
    jump = lambda x: np.where(x < w[0], np.exp(x), 0.0)  # noqa: E731
    degs = [8, 16, 32, 64, 128, 256]

    def interp_err(f, n, pieces):
        """Sup error of the piecewise Chebyshev interpolant; each piece samples f just inside its ends (one-sided limits at a jump)."""
        err = 0.0
        for lo_, hi_ in pieces:
            nodes = 0.5 * (lo_ + hi_) + 0.5 * (hi_ - lo_) * np.cos(np.pi * np.arange(n + 1) / n)
            inner = np.clip(nodes, lo_ + 1e-12, hi_ - 1e-12)
            c = np.polynomial.chebyshev.chebfit((2 * nodes - lo_ - hi_) / (hi_ - lo_), f(inner), n)
            sel = (xs > lo_ + 1e-12) & (xs < hi_ - 1e-12)
            u = (2 * xs[sel] - lo_ - hi_) / (hi_ - lo_)
            err = max(err, float(np.abs(np.polynomial.chebyshev.chebval(u, c) - f(xs[sel])).max()))
        return err

    one, two = [(0.0, 1.0)], [(0.0, w[0]), (w[0], 1.0)]
    prof = {name: {"one_piece": [interp_err(f, n, one) for n in degs], "breakpoint": [interp_err(f, n, two) for n in degs[:4]]} for name, f in (("kink", kink), ("jump", jump))}
    prof["kink"]["slope_one_piece"] = slope(np.log10(degs), np.log10(prof["kink"]["one_piece"]))
    prof["jump_height"] = float(np.exp(w[0]))
    with_edges = fresh("genz_continuous", 2, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], edges=True)
    out["profiles"] = {"degrees": degs, **prof}
    out["continuous_d2_chebtt_breakpoints"] = {"rows": with_edges, "slope_log_evals": slope(np.log10([1 / r["tol"] for r in with_edges]), np.log10([r["evals"] for r in with_edges]))}
    (OUT / "f1_smooth_kinks.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    pp = out["product_peak_d2"]
    print("product peak d=2: rho", np.round(pp["rho"], 3), "predicted deg/decade", np.round(pp["predicted_degree_per_decade"], 2),
          "measured", np.round(pp["measured_degree_per_decade"], 2), "ranks", pp["ranks"])  # fmt: skip
    for r in pp["rows"]:
        print("   ", r["tol"], r["degrees"], r["ranks"], r["evals"])
    print("oscillatory max TT ranks at tol", rtols, ":", out["oscillatory_ranks"], "(predicted 2)")
    c = out["corner_peak"]
    print("corner peak max TT ranks:", c["max_ranks"], "; rank per decade", {d: round(v, 2) for d, v in c["measured_rank_per_decade"].items()},
          "predicted at most", round(c["predicted_rank_per_decade_at_most"], 2))  # fmt: skip
    pr = out["profiles"]
    print("kink profile, one piece, errors at n =", pr["degrees"], ":", [f"{e:.1e}" for e in pr["kink"]["one_piece"]], "slope", round(pr["kink"]["slope_one_piece"], 2), "(predicted -1)")
    print("   with the breakpoint:", [f"{e:.1e}" for e in pr["kink"]["breakpoint"]])
    print("jump profile (height", round(pr["jump_height"], 2), "), one piece:", [f"{e:.2f}" for e in pr["jump"]["one_piece"]], "(predicted >= half the jump)")
    print("   with the breakpoint:", [f"{e:.1e}" for e in pr["jump"]["breakpoint"]])
    k = out["continuous_d2_chebtt_breakpoints"]
    print("ChebTT with breakpoints, genz_continuous d=2: slope of log evals vs log(1/tol)", round(k["slope_log_evals"], 2))
    for r in k["rows"]:
        print("   ", r["tol"], "degrees", r["degrees"], "pieces", r["pieces"], "ranks", r["ranks"], "evals", r["evals"])


if __name__ == "__main__":
    main()
