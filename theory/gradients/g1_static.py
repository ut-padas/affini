"""
The gradient tier on static functions (memo section G): what values plus gradients buy over values alone.

Predictions, committed before the run:
  G1.1 (Hermite interpolation, one variable). For f analytic in the Bernstein ellipse E_rho, the Chebyshev-Hermite interpolant through
       n points (values and first derivatives, degree 2n - 1) has error ~ rho^-2n, so the points needed per decade of accuracy are
       1 / (2 log10 rho), half of the value-only interpolant's 1 / log10 rho. Tested on the product peak's two factors (rho = 2.075,
       1.638) and a Runge function: the measured points per decade, values only against values and derivatives, ratio about 2.
  G1.2 (gradient-enhanced least squares, d dimensions). For the total-degree-q polynomials (p terms) on [-1, 1]^d with N uniform points,
       each point gives 1 + d equations (the value and the gradient). The fit needs about (1 + d) times fewer points than values alone
       for the same accuracy and conditioning: at d = 5, q = 4 (p = 126) the gradient-enhanced fit at N = 30 points (180 equations)
       is as accurate as the value-only fit at N = 180, and both are near the best polynomial error; the value-only fit at N = 30
       is underdetermined. Tested on the product peak and the corner peak (d = 5), with the error measured on 20,000 uniform points.
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data"


def genz_params(total: float, d: int):
    rng = np.random.default_rng(7_000 + d)
    a = rng.uniform(0.0, 1.0, d)
    return a * total / a.sum(), rng.uniform(0.0, 1.0, d)


def bernstein_rho(z: complex, lo: float, hi: float) -> float:
    zeta = (2 * z - lo - hi) / (hi - lo)
    r = zeta + np.sqrt(zeta * zeta - 1)
    return float(max(abs(r), 1 / abs(r)))


def hermite_error(f, df, n: int, lo: float, hi: float, grid: np.ndarray) -> float:
    """Sup error of the degree-(2n-1) polynomial through f and f' at n Chebyshev points (solved in the Chebyshev basis)."""
    t = np.cos(np.pi * (np.arange(n) + 0.5) / n)  # Chebyshev points of the first kind
    x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * t
    deg = 2 * n - 1
    cheb = np.polynomial.chebyshev
    V = cheb.chebvander(t, deg)
    dV = np.stack([cheb.chebval(t, cheb.chebder(np.eye(deg + 1)[k])) for k in range(deg + 1)], axis=1) * 2 / (hi - lo)
    coef = np.linalg.lstsq(np.vstack([V, dV]), np.concatenate([f(x), df(x)]), rcond=None)[0]
    s = (2 * grid - lo - hi) / (hi - lo)
    return float(np.max(np.abs(cheb.chebval(s, coef) - f(grid))))


def value_error(f, n: int, lo: float, hi: float, grid: np.ndarray) -> float:
    """Sup error of the degree-(n-1) interpolant at n Chebyshev points of the first kind."""
    t = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * t
    coef = np.polynomial.chebyshev.chebfit(t, f(x), n - 1)
    return float(np.max(np.abs(np.polynomial.chebyshev.chebval((2 * grid - lo - hi) / (hi - lo), coef) - f(grid))))


def points_per_decade(errs, ns) -> float:
    # stop at the first error below 1e-11: past it the errors sit on the rounding floor and would flatten the slope
    stop = next((i for i, e in enumerate(errs) if e < 1e-11), len(errs))
    ok = list(zip(ns[:stop], errs[:stop], strict=True))
    n, e = np.array([o[0] for o in ok]), np.array([o[1] for o in ok])
    return float(-1.0 / np.polyfit(n, np.log10(e), 1)[0])


def g1_hermite() -> dict:
    a, w = genz_params(7.25, 2)
    grid = np.linspace(0, 1, 20_001)
    cases = {}
    for i in range(2):
        f = lambda x, i=i: 1.0 / (a[i] ** -2 + (x - w[i]) ** 2)  # noqa: E731
        df = lambda x, i=i: -2 * (x - w[i]) / (a[i] ** -2 + (x - w[i]) ** 2) ** 2  # noqa: E731
        cases[f"product_peak_axis{i}"] = (f, df, 0.0, 1.0, bernstein_rho(w[i] + 1j / a[i], 0.0, 1.0))
    cases["runge_b0.3"] = (lambda x: 1 / (0.09 + x * x), lambda x: -2 * x / (0.09 + x * x) ** 2, -1.0, 1.0, bernstein_rho(0.3j, -1.0, 1.0))
    out = {}
    for name, (f, df, lo, hi, rho) in cases.items():
        g = np.linspace(lo, hi, 20_001)
        ns = list(range(4, 41, 2))
        ev = [value_error(f, n, lo, hi, g) for n in ns]
        eh = [hermite_error(f, df, n, lo, hi, g) for n in ns]
        out[name] = {"rho": rho, "predicted_values_per_decade": 1 / math.log10(rho), "predicted_hermite_per_decade": 1 / (2 * math.log10(rho)),
                     "measured_values_per_decade": points_per_decade(ev, ns), "measured_hermite_per_decade": points_per_decade(eh, ns),
                     "n": ns, "value_errors": ev, "hermite_errors": eh}  # fmt: skip
        out[name]["measured_ratio"] = out[name]["measured_values_per_decade"] / out[name]["measured_hermite_per_decade"]
    del grid
    return out


def g1_least_squares(d: int = 5, q: int = 4) -> dict:
    idx = [m for m in itertools.product(range(q + 1), repeat=d) if sum(m) <= q]
    p = len(idx)
    cheb = np.polynomial.chebyshev
    E = np.eye(q + 1)

    def design(X):
        T = np.stack([cheb.chebval(X, E[k]) for k in range(q + 1)], axis=-1)  # (n, d, q+1)
        dT = np.stack([cheb.chebval(X, cheb.chebder(E[k])) if k else np.zeros_like(X) for k in range(q + 1)], axis=-1)
        Vv = np.stack([np.prod([T[:, j, m[j]] for j in range(d)], axis=0) for m in idx], axis=1)
        Vg = [np.stack([np.prod([(dT if j == jj else T)[:, j, m[j]] for j in range(d)], axis=0) for m in idx], axis=1) for jj in range(d)]
        return Vv, Vg

    out = {"d": d, "degree": q, "terms": p, "functions": {}}
    rng = np.random.default_rng(0)
    Xt = rng.uniform(-1, 1, (20_000, d))
    Vt, _ = design(Xt)
    for name, total in (("genz_product_peak", 7.25), ("genz_corner_peak", 1.85)):
        a, w = genz_params(total, d)
        # the functions on [0, 1]^d mapped to [-1, 1]^d: x01 = (x + 1) / 2
        if name == "genz_product_peak":
            f = lambda X, a=a, w=w: np.prod(1.0 / (a ** -2.0 + ((X + 1) / 2 - w) ** 2), axis=1)  # noqa: E731
            grad = lambda X, a=a, w=w, f=f: f(X)[:, None] * (-((X + 1) / 2 - w) / (a ** -2.0 + ((X + 1) / 2 - w) ** 2))  # noqa: E731
        else:
            f = lambda X, a=a: (1.0 + ((X + 1) / 2) @ a) ** (-(d + 1))  # noqa: E731
            grad = lambda X, a=a: (-(d + 1) * (1.0 + ((X + 1) / 2) @ a) ** (-(d + 2)))[:, None] * (a / 2)[None, :]  # noqa: E731
        ft = f(Xt)
        scale = float(np.abs(ft).max())
        best = np.linalg.lstsq(Vt, ft, rcond=None)[0]
        rows = {"best_L2_projection_sup_error": float(np.abs(Vt @ best - ft).max() / scale)}
        for N in (30, 60, 126, 180, 360):
            X = rng.uniform(-1, 1, (N, d))
            Vv, Vg = design(X)
            entry = {}
            if p <= N:
                c = np.linalg.lstsq(Vv, f(X), rcond=None)[0]
                entry["values_only"] = float(np.abs(Vt @ c - ft).max() / scale)
            A = np.vstack([Vv] + Vg)
            b = np.concatenate([f(X)] + [grad(X)[:, j] for j in range(d)])
            c = np.linalg.lstsq(A, b, rcond=None)[0]
            entry["values_and_gradients"] = float(np.abs(Vt @ c - ft).max() / scale)
            entry["equations"] = int(A.shape[0])
            rows[f"N={N}"] = entry
        out["functions"][name] = rows
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    out = {"hermite": g1_hermite(), "least_squares": g1_least_squares()}
    (OUT / "g1_static.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    for name, r in out["hermite"].items():
        print(f"{name}: rho {r['rho']:.3f}; points per decade values {r['measured_values_per_decade']:.2f} (pred {r['predicted_values_per_decade']:.2f}), "
              f"values+derivatives {r['measured_hermite_per_decade']:.2f} (pred {r['predicted_hermite_per_decade']:.2f}); ratio {r['measured_ratio']:.2f} (pred 2)")  # fmt: skip
    ls = out["least_squares"]
    print(f"least squares d={ls['d']} degree {ls['degree']} ({ls['terms']} terms):")
    for name, rows in ls["functions"].items():
        print(f"  {name}: best L2-projection sup error {rows['best_L2_projection_sup_error']:.2e}")
        for k, v in rows.items():
            if k.startswith("N="):
                print(f"    {k}: values only {v.get('values_only', 'underdetermined')}, values+gradients {v['values_and_gradients']:.2e} ({v['equations']} equations)")


if __name__ == "__main__":
    main()
