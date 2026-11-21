"""
The HIRES recipe's selection as it should run (review 7, F1-F3), and the rollout it exists for: kicked at every step.

Setup as g4_hires_recipe.py (k = 3 chart, degree-4 decoder, invariant projection, tolerance 0.01 dex), with three changes:
  degrees  4..12 (g4 stopped at 8, the edge where the value-only fit diverges and recovers at 9-10);
  rule     the degree with the smallest worst training-chain error of the DEPLOYED surrogate (plain for the value-only fit, corrected
           for the gradient fit) over all 100 training trajectories; it passes when that error is within the tolerance. The
           single-state Jacobian condition (ii) is reported, not used (it is not necessary, F3);
  Jacobians by central differences of the benchmark's step (16 plain solves each, h = 1e-5; they agree with the forward sensitivities
           to 2-8e-7, g4_diagnosis.json), at 300 training states, three draws (seeds 3, 11, 12).
Fits: "data" (values only, no solve) and "gradients" (targets Y - B n plus chart-derivative rows J DD(u), corrected by B n when deployed).
Scored on the benchmark's 20 test trajectories and on all 40, and by a rollout kicked at every step: each of the 40 test trajectories
chained for its 170 steps with a kick of eps = 0.003 dex (random signs in every coordinate) added before every step, against the solver
chained with the same kicks (the operator-split proxy: another process moves the state between steps).

Predictions, committed before the run. P1 and P2 are informed by review 7's exploratory runs (seed 3 and two other draws at degrees
9-12); P3 and P4 test what no run has measured.
  P1  the value-only fit: the rule picks degree 10 and it holds the tolerance on all 40 test trajectories;
  P2  the gradient fit: in each draw the corrected training-chain error falls monotonically from degree 8 to 12, the rule picks 11 or 12,
      and the selected fit holds the tolerance on all 40 test trajectories;
  P3  kicked at every step, the ordering is corrected gradient fit at degree 8 < plain gradient fit < value-only fit at degree 10, the
      value-only fit at least 10 times worse than the corrected gradient fit at degree 8;
  P4  kicked at every step, the corrected gradient fit at degree 8 stays within three times the tolerance, and the selected
      (degree 11-12) gradient fit is worse than it (its kicked one-step error grows with the degree, review 7 F2).
"""

import json
import time

import numpy as np
from g2_hires import OUT, P, poly
from g4_hires_recipe import TOL, Recipe, fit_chart

from src.benchmark.step_maps import Projected, data, invariants_of, rollout, visited

EPS = 0.003
DEGREES = range(4, 13)


def difference_jacobians(Z, h=1e-5):
    """d z(dt) / d z(0) in log10 coordinates at every row of Z, by central differences of the benchmark's step (one batched call)."""
    n, d = Z.shape
    E = np.eye(d)
    pts = np.concatenate([Z[:, None, :] + h * E[None], Z[:, None, :] - h * E[None]], axis=1).reshape(-1, d)
    F = P.step(pts).reshape(n, 2, d, d)
    return np.transpose((F[:, 0] - F[:, 1]) / (2 * h), (0, 2, 1))  # [state, output i, input j]


def kicked_rollout(sur, true, signs):
    """Worst scored error over the steps of the surrogate and the solver chained from the same states, both kicked before every step."""
    Zs, Zt, worst = true[:, 0].copy(), true[:, 0].copy(), 0.0
    for k in range(true.shape[1] - 1):
        kick = EPS * signs[k]
        Zt = P.step(Zt + kick)
        Zin = Zs + kick
        Zs = Zin + sur.predict(Zin)
        worst = max(worst, float(np.max(np.where(P.scored(Zt), np.abs(Zs - Zt), 0.0))))
        if not np.isfinite(worst) or worst > 10:
            return float("inf") if not np.isfinite(worst) else worst
    return worst


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
    zlo, zhi = X.min(0) - 2.0, X.max(0) + 0.5
    W = invariants_of(P, train)
    signs = np.random.default_rng(99).choice([-1.0, 1.0], (test.shape[1] - 1, *test[:, 0].shape))
    h = 1e-6

    def deployed(G, B):
        return Projected(Recipe(G, B, decode, mu, V, lo, hi, zlo, zhi), P, W)

    def scores(sur):
        return {"train_chains_worst": float(rollout(P, sur, train)["rollout_worst"]), "test20_worst": float(rollout(P, sur, test[:20])["rollout_worst"]),
                "test40_worst": float(rollout(P, sur, test)["rollout_worst"])}  # fmt: skip

    def choose(rows):
        best = min(rows, key=lambda r: r["train_chains_worst"])
        return best["degree"], bool(best["train_chains_worst"] <= TOL)

    out = {"eps": EPS, "data": {"degrees": []}, "draws": []}
    for q in DEGREES:
        row = {"degree": q, **scores(deployed(fit_chart(q, lo, hi, U, Y), None))}
        out["data"]["degrees"].append(row)
        print(f"[data] q={q}: training chains {row['train_chains_worst']:.2e}, test 20 {row['test20_worst']:.2e}, test 40 {row['test40_worst']:.2e}", flush=True)
    q_data, ok = choose(out["data"]["degrees"])
    out["data"].update(selected=q_data, passes=ok, kicked=kicked_rollout(deployed(fit_chart(q_data, lo, hi, U, Y), None), test, signs))
    print(f"[data] selected q={q_data} ({'passes' if ok else 'fails'}); kicked at every step {out['data']['kicked']:.2e}", flush=True)
    for seed in (3, 11, 12):
        js = np.random.default_rng(seed).choice(len(X), 300, replace=False)
        t0 = time.perf_counter()
        Jd = difference_jacobians(X[js])
        t_jac = time.perf_counter() - t0
        B = np.linalg.lstsq(poly(U[js], 2), Jd.reshape(len(js), -1), rcond=None)[0]
        normal = Recipe(None, B, decode, mu, V, lo, hi, zlo, zhi).normal_term
        dD = np.stack([(decode(U[js] + h * e) - decode(U[js] - h * e)) / (2 * h) for e in np.eye(3)], axis=2)
        T, dT = Y - normal(X), np.einsum("nij,njk->nik", Jd, dD)
        draw = {"seed": seed, "jacobian_seconds": t_jac, "plain_solves": 16 * len(js), "degrees": []}
        fits = {}
        for q in DEGREES:
            fits[q] = fit_chart(q, lo, hi, U, T, U[js], dT)
            row = {"degree": q, **scores(deployed(fits[q], B))}
            draw["degrees"].append(row)
            print(f"[gradients seed {seed}] q={q}: training chains {row['train_chains_worst']:.2e}, test 20 {row['test20_worst']:.2e}, test 40 {row['test40_worst']:.2e}", flush=True)
        q_g, ok = choose(draw["degrees"])
        draw.update(selected=q_g, passes=ok)
        draw["kicked"] = {f"corrected_q{q}": kicked_rollout(deployed(fits[q], B), test, signs) for q in sorted({8, q_g})}
        draw["kicked"]["plain_q8"] = kicked_rollout(deployed(fits[8], None), test, signs)
        out["draws"].append(draw)
        print(f"[gradients seed {seed}] selected q={q_g} ({'passes' if ok else 'fails'}); Jacobians {t_jac:.1f} s; kicked at every step: "
              + ", ".join(f"{k} {v:.2e}" for k, v in draw["kicked"].items()), flush=True)  # fmt: skip
    (OUT / "g5_hires_selection.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
