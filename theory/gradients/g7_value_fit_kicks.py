"""
The value-only HIRES fit off the visited set (review 8, finding 16): the plain projected recipe of g5_hires_selection.py at degrees 8-12 on
g6_hires_off_trajectory.py's 300 kicked test states (seed 7, kicks of 0.01 and 0.03 dex), the same states and reference solves as g6.
A measurement after the fact, with no prediction: it replaces a corrected-fit figure that section G had placed on the plain row.
"""

import json

import numpy as np
from g2_hires import OUT, ChartPoly, P, poly
from g4_hires_recipe import Recipe

from src.benchmark.step_maps import Projected, data, invariants_of, visited


def main() -> None:
    train, test = data("hires")
    X = visited(train)
    Y = train[:, 1:].transpose(1, 0, 2).reshape(-1, train.shape[2])
    mu = X.mean(axis=0)
    V = np.linalg.svd(X - mu, full_matrices=False)[2][:3].T
    U = (X - mu) @ V
    dec = np.linalg.lstsq(poly(U, 4), X, rcond=None)[0]

    def decode(W):
        return poly(W, 4) @ dec

    lo, hi = U.min(0), U.max(0)
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    W = invariants_of(P, train)
    Xe = test[:, :-1].reshape(-1, test.shape[2])
    krng = np.random.default_rng(7)  # g6's kick construction
    idx = krng.choice(len(Xe), 300, replace=False)
    kicks = {eps: Xe[idx] + eps * krng.choice([-1.0, 1.0], Xe[idx].shape) for eps in (0.01, 0.03)}
    out = {"degrees": []}
    for q in range(8, 13):
        sur = Projected(Recipe(ChartPoly(q, lo, hi).fit(U, Y), None, decode, mu, V, lo, hi, X.min(0) - 2.0, X.max(0) + 0.5), P, W)
        row = {"degree": q}
        for eps, Zk in kicks.items():
            Yk = P.step(Zk)
            row[f"kicked_{eps}"] = float(np.max(np.where(P.scored(Yk), np.abs(Zk + sur.predict(Zk) - Yk), 0.0)))
        out["degrees"].append(row)
        print(f"q={q}: plain value-only fit, kicked 0.01 {row['kicked_0.01']:.3g}, 0.03 {row['kicked_0.03']:.3g}", flush=True)
    (OUT / "g7_value_fit_kicks.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
