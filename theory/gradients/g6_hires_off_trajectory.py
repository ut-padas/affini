"""
One fit for both regimes (review 7's second experiment): the HIRES gradient fit of g5_hires_selection.py, plus values and chart-derivative
rows of Phi(D(u)) at chart values off the training trajectories.

g5 found the corrected gradient fit monotone in the degree on the visited set (40-trajectory rollout 6.3e-4 to 1.0e-3 at degree 12) but
worse off it as the degree grows: kicked at every step by 0.003 dex it diverges at degree 12 and errs 2.3-8.0e-2 at degree 8. Its
Jacobian rows sit on the trajectories, which are curves in the 3-D chart, so between them a high-degree fit is unconstrained. Here 300
more chart values u' = u_j + delta, delta uniform in [-0.05, 0.05]^3 dex around the 300 Jacobian states (clipped to the chart box),
each give the value Phi(D(u')) (1 solve) and the chart derivative J(D(u')) DD(u') (16 solves by central differences): 5,100 plain
solves in all with the on-trajectory Jacobians. At a decoded point the normal part is zero, so its target is Phi(D(u')) itself.
Same chart, decoder, correction B (fitted to the on-trajectory Jacobians), rule (smallest worst training-chain error of the corrected
surrogate over the 100 training trajectories, degrees 8-12) and the same three draws as g5; scored on the 40 test trajectories, by the
corrected one-step error on 300 test states kicked by 0.01 and 0.03 dex, and by g5's rollout kicked at every step by 0.003 dex.

Predictions, committed before the run (tolerance 0.01 dex):
  Q1  the corrected one-step error at a kick of 0.01 dex is within the tolerance at every degree 8-12 (without the off-trajectory rows
      it grew from 8.1e-3 at degree 8 to 0.24 at degree 12, review 7);
  Q2  the 40-trajectory rollout stays within the tolerance from degree 9 on, in every draw (no loss on the visited set);
  Q3  kicked at every step, the selected fit is within three times the tolerance in every draw, and no worse than g5's corrected degree-8
      fit of the same draw (8.0e-2, 3.8e-2, 2.3e-2);
  Q4  kicked at every step, the selected fit is within the tolerance in every draw.
"""

import json

import numpy as np
from g2_hires import OUT, P, poly
from g4_hires_recipe import TOL, Recipe, fit_chart
from g5_hires_selection import difference_jacobians, kicked_rollout

from src.benchmark.step_maps import Projected, data, invariants_of, rollout, visited

DEGREES = range(8, 13)


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

    def chart_derivative(Uq):
        return np.stack([(decode(Uq + h * e) - decode(Uq - h * e)) / (2 * h) for e in np.eye(3)], axis=2)

    lo, hi = U.min(0), U.max(0)
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    zlo, zhi = X.min(0) - 2.0, X.max(0) + 0.5
    W = invariants_of(P, train)
    signs = np.random.default_rng(99).choice([-1.0, 1.0], (test.shape[1] - 1, *test[:, 0].shape))  # g5's kicks
    Xe = test[:, :-1].reshape(-1, test.shape[2])
    krng = np.random.default_rng(7)
    kick_idx = krng.choice(len(Xe), 300, replace=False)
    kicks = {eps: Xe[kick_idx] + eps * krng.choice([-1.0, 1.0], Xe[kick_idx].shape) for eps in (0.01, 0.03)}
    kicked_next = {eps: P.step(Z) for eps, Z in kicks.items()}
    h = 1e-6

    def deployed(G, B):
        return Projected(Recipe(G, B, decode, mu, V, lo, hi, zlo, zhi), P, W)

    def worst(sur, Z, Ynext):
        return float(np.max(np.where(P.scored(Ynext), np.abs(Z + sur.predict(Z) - Ynext), 0.0)))

    out = {"draws": []}
    for seed in (3, 11, 12):
        rng = np.random.default_rng(seed)
        js = rng.choice(len(X), 300, replace=False)
        Jd = difference_jacobians(X[js])
        B = np.linalg.lstsq(poly(U[js], 2), Jd.reshape(len(js), -1), rcond=None)[0]
        normal = Recipe(None, B, decode, mu, V, lo, hi, zlo, zhi).normal_term
        Uoff = np.clip(U[js] + rng.uniform(-0.05, 0.05, (len(js), 3)), lo, hi)
        Zoff = decode(Uoff)
        Yoff, Joff = P.step(Zoff), difference_jacobians(Zoff)
        Uv, Tv = np.vstack([U, Uoff]), np.vstack([Y - normal(X), Yoff])
        Ug = np.vstack([U[js], Uoff])
        dT = np.vstack([np.einsum("nij,njk->nik", Jd, chart_derivative(U[js])), np.einsum("nij,njk->nik", Joff, chart_derivative(Uoff))])
        draw = {"seed": seed, "plain_solves": 16 * len(js) + 17 * len(Uoff), "degrees": []}
        fits = {}
        for q in DEGREES:
            fits[q] = fit_chart(q, lo, hi, Uv, Tv, Ug, dT)
            sur = deployed(fits[q], B)
            row = {"degree": q, "train_chains_worst": float(rollout(P, sur, train)["rollout_worst"]), "test40_worst": float(rollout(P, sur, test)["rollout_worst"]),
                   **{f"kicked_{eps}": worst(sur, kicks[eps], kicked_next[eps]) for eps in kicks}}  # fmt: skip
            draw["degrees"].append(row)
            print(f"[seed {seed}] q={q}: training chains {row['train_chains_worst']:.2e}, test 40 {row['test40_worst']:.2e}, "
                  f"kicked 0.01 {row['kicked_0.01']:.2e}, 0.03 {row['kicked_0.03']:.2e}", flush=True)  # fmt: skip
        best = min(draw["degrees"], key=lambda r: r["train_chains_worst"])
        draw.update(selected=best["degree"], passes=bool(best["train_chains_worst"] <= TOL))
        draw["kicked_every_step"] = {f"q{q}": kicked_rollout(deployed(fits[q], B), test, signs) for q in sorted({8, best["degree"]})}
        out["draws"].append(draw)
        print(f"[seed {seed}] selected q={best['degree']}; kicked at every step: " + ", ".join(f"{k} {v:.2e}" for k, v in draw["kicked_every_step"].items()), flush=True)
    (OUT / "g6_hires_off_trajectory.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
