"""
The gradient tier on HIRES with a designed point set (memo section G): as g2_hires.py, but the oracle points are the tensor
Chebyshev points (first kind) of the chart box, n per axis, instead of random visited states, whose worst-case coverage was poor.

Predictions, committed before the run (tolerance 0.01 dex, k = 3):
  G3.1 with n points per axis, the fit with values and Jacobians reaches the accuracy the value-only fit reaches with about
       4^(1/3) n = 1.6 n points per axis (4 times the points: 1 + k equations per point; with first derivatives only, not 2^k);
  G3.2 some configuration holds a quarter of the tolerance (2.5e-3) in one step on the test states with fewer than 1,000 solves, nine
       times fewer than the reduced ChebTT's 9,477 (Demonstration 8.5);
  G3.3 with a measured sensitivity cost of c plain solves (6.7 in g2_hires), the gradient-enhanced fit is cheaper in total than the
       value-only fit of the same accuracy if c < 4; in this Python implementation it is not, with a compiled sensitivity solver
       (c of about 1.1-1.75) it would be.
"""

import json

import numpy as np
from g2_hires import OUT, ChartPoly, P, Reduced, poly, sensitivity_step

from src.benchmark.step_maps import (
    ROLLOUTS,
    Projected,
    data,
    invariants_of,
    rollout,
    survived,
    visited,
)


def main() -> None:
    train, test = data("hires")
    X = visited(train)
    mu = X.mean(axis=0)
    V = np.linalg.svd(X - mu, full_matrices=False)[2][:3].T
    U_all = (X - mu) @ V
    dec = np.linalg.lstsq(poly(U_all, 4), X, rcond=None)[0]
    decode = lambda U: poly(U, 4) @ dec  # noqa: E731
    lo, hi = U_all.min(0), U_all.max(0)
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    W = invariants_of(P, train)
    Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    mask = P.scored(Ye)
    h = 1e-6
    rows = []
    for n in (5, 7, 9, 11, 13):
        t = np.cos(np.pi * (np.arange(n) + 0.5) / n)
        axes = [0.5 * (lo[j] + hi[j]) + 0.5 * (hi[j] - lo[j]) * t for j in range(3)]
        U = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        Z0 = decode(U)
        dD = np.stack([(decode(U + h * e) - decode(U - h * e)) / (2 * h) for e in np.eye(3)], axis=2)
        vals, jacs = [], []
        for z, dz in zip(Z0, dD, strict=True):
            y0 = 10.0**z
            y1, S1 = sensitivity_step(y0, y0[:, None] * dz, P.dt)
            vals.append(np.log10(np.maximum(y1, 1e-30)))
            jacs.append(S1 / y1[:, None])
        vals, jacs = np.array(vals), np.array(jacs)
        for grad in (False, True):
            for q in range(2, 2 * n):
                G = ChartPoly(q, lo, hi)
                if len(U) * (4 if grad else 1) < 1.5 * len(G.idx) or len(G.idx) > 900:
                    break
                G.fit(U, vals, jacs if grad else None)
                sur = Projected(Reduced(G, mu, V, lo, hi, X.min(0) - 2.0, X.max(0) + 0.5), P, W)
                e = float(np.max(np.where(mask, np.abs(Xe + sur.predict(Xe) - Ye), 0.0)))
                rows.append({"points_per_axis": n, "solves": len(U), "gradients": grad, "degree": q, "terms": len(G.idx), "one_step_worst": e, "_sur": sur})
        best = {g: min((r for r in rows if r["points_per_axis"] == n and r["gradients"] == g), key=lambda r: r["one_step_worst"]) for g in (False, True)}
        for g, r in best.items():
            r["best_for_n"] = True
            r["rollout_worst"] = float(rollout(P, r["_sur"], test[:ROLLOUTS])["rollout_worst"])
            r["survived"] = float(survived(P, r["_sur"], test))
            print(f"n={n} ({len(U)} solves) {'values+Jacobians' if g else 'values only     '}: best degree {r['degree']} ({r['terms']} terms) one-step {r['one_step_worst']:.2e}, "
                  f"survived {r['survived']:.2e}, rollout {r['rollout_worst']:.2e}", flush=True)  # fmt: skip
    for r in rows:
        r.pop("_sur")
    (OUT / "g3_hires_grid.json").write_text(json.dumps({"rows": rows}, indent=1, sort_keys=True, default=float) + "\n")


if __name__ == "__main__":
    main()
