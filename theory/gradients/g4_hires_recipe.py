"""
The reduced-surrogate recipe of the memo's section 10 item 1, end to end on HIRES (k = 3), with gradients where they pay.

  chart      pi(z) = V^T (z - mu), the top-3 principal directions of the visited log-states; decoder D(u) (total degree 4, least squares);
  data fit   G(u), total degree q, least squares from chart values to next states on all training pairs (no solve);
  selection  q in 4..8 by the two rollout conditions on the TRAINING chains (no test data): (i) chained from each training trajectory's
             first state, the worst error over its steps within the tolerance; (ii) the surrogate's own Jacobian norm max ||D Psi||
             (central differences, projection included) no larger than the dynamics' max ||D Phi|| (forward sensitivities at 200 training states);
  correction Psi_1(z) = G(pi z) + B(pi z) (z - D(pi z)), B(u) ~ D Phi(D(u)): the full log-coordinate Jacobian at 300 training states
             (forward sensitivities), fitted entrywise by a total-degree-2 polynomial in u;
  tracking   the decoder residual r(z) = max |z - D(pi z)| (no solve); refuse when r > tol / 4 (the solver steps the state);
  invariant  every output projected onto y7 + y8 (the benchmark's Projected).
Scored on the test trajectories: one-step worst, the error on test states kicked by eps = 0.003, 0.01, 0.03, 0.05 dex (random signs in every
coordinate) with and without the correction, the share refused, the worst of 20 rollouts over 170 steps.

Predictions, committed before the run (tolerance 0.01 dex):
  R1  the selection picks q = 6 and its test rollout is within twice the tolerance (review 6 measured 1.5e-2 for the degree-6 fit);
  R2  with the fitted correction the error on states kicked by 0.01 dex is within the tolerance (1.9e-2 without, review 6), and by 0.05 dex
      within twice the tolerance;
  R3  the corrected error grows like r^2 with the kick (slope near 2 in log-log against eps), the uncorrected like r (slope near 1).
"""

import json

import numpy as np
from g2_hires import OUT, ChartPoly, P, poly, sensitivity_step

from src.benchmark.step_maps import (
    ROLLOUTS,
    Projected,
    data,
    invariants_of,
    rollout,
    visited,
)

TOL = 0.01


def log_jacobian(z):
    """d z(dt) / d z(0) in log10 coordinates: the full forward sensitivity with S(0) = diag(y0)."""
    y0 = 10.0**z
    y1, S1 = sensitivity_step(y0, np.diag(y0), P.dt)
    return S1 / y1[:, None]


class Recipe:
    """The reduced surrogate as an increment model: data fit, optional normal correction, refusal by the decoder residual."""

    def __init__(self, G, B, decode, mu, V, lo, hi, zlo, zhi, refuse_at=None):
        self.G, self.B, self.decode, self.mu, self.V = G, B, decode, mu, V
        self.lo, self.hi, self.zlo, self.zhi, self.refuse_at, self.evals = (
            lo,
            hi,
            zlo,
            zhi,
            refuse_at,
            0,
        )

    def chart(self, Z):
        return np.clip((Z - self.mu) @ self.V, self.lo, self.hi)

    def residual(self, Z):
        return np.max(np.abs(Z - self.decode(self.chart(Z))), axis=1)

    def refuses(self, Z):
        if self.refuse_at is None:
            return np.zeros(len(Z), dtype=bool)
        return self.residual(Z) > self.refuse_at

    def predict(self, Z):
        U = self.chart(Z)
        out = self.G(U)
        if self.B is not None:
            n = Z - self.decode(U)
            Bm = (poly(U, 2) @ self.B).reshape(len(Z), Z.shape[1], Z.shape[1])
            out = out + np.einsum("nij,nj->ni", Bm, n)
        return np.clip(out, self.zlo, self.zhi) - Z


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train, test = data("hires")
    X = visited(train)
    Y = train[:, 1:].transpose(1, 0, 2).reshape(-1, train.shape[2])
    mu = X.mean(axis=0)
    V = np.linalg.svd(X - mu, full_matrices=False)[2][:3].T
    U = (X - mu) @ V
    dec = np.linalg.lstsq(poly(U, 4), X, rcond=None)[0]
    decode = lambda W: poly(W, 4) @ dec  # noqa: E731
    lo, hi = U.min(0), U.max(0)
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    zlo, zhi = X.min(0) - 2.0, X.max(0) + 0.5
    W = invariants_of(P, train)
    rng = np.random.default_rng(3)
    # the dynamics' Jacobians at training states (forward sensitivities)
    js = rng.choice(len(X), 300, replace=False)
    Jd = np.array([log_jacobian(z) for z in X[js]])
    dyn_norm = float(np.max(np.linalg.norm(Jd[:200], ord=2, axis=(1, 2))))
    # selection by the two conditions on the training chains
    sel = []
    for q in range(4, 9):
        G = ChartPoly(q, lo, hi).fit(U, Y)
        sur = Projected(Recipe(G, None, decode, mu, V, lo, hi, zlo, zhi), P, W)
        chain = rollout(P, sur, train[:40])["rollout_worst"]
        h = 1e-6
        Zs = X[js[:200]]
        DPsi = np.stack(
            [
                (
                    sur.predict(Zs + h * e)
                    + Zs
                    + h * e
                    - sur.predict(Zs - h * e)
                    - Zs
                    + h * e
                )
                / (2 * h)
                for e in np.eye(X.shape[1])
            ],
            axis=2,
        )
        psi_norm = float(np.max(np.linalg.norm(DPsi, ord=2, axis=(1, 2))))
        sel.append({"degree": q, "train_chain_worst": float(chain), "surrogate_jacobian_norm": psi_norm, "dynamics_jacobian_norm": dyn_norm,
                    "passes": bool(chain <= TOL and psi_norm <= dyn_norm)})  # fmt: skip
        print(
            f"q={q}: training chains worst {chain:.2e}, max ||D Psi|| {psi_norm:.2f} (dynamics {dyn_norm:.2f}) -> {'pass' if sel[-1]['passes'] else 'fail'}",
            flush=True,
        )
    passing = [r["degree"] for r in sel if r["passes"]]
    q = (
        passing[-1]
        if passing
        else min(sel, key=lambda r: r["train_chain_worst"])["degree"]
    )
    G = ChartPoly(q, lo, hi).fit(U, Y)
    # the normal correction: B(u) fitted entrywise (degree 2 in u) to the dynamics' Jacobians at the 300 training states
    Bcoef = np.linalg.lstsq(poly(U[js], 2), Jd.reshape(len(js), -1), rcond=None)[0]
    Xe, Ye = (
        test[:, :-1].reshape(-1, test.shape[2]),
        test[:, 1:].reshape(-1, test.shape[2]),
    )
    plain = Projected(Recipe(G, None, decode, mu, V, lo, hi, zlo, zhi), P, W)
    corr = Projected(Recipe(G, Bcoef, decode, mu, V, lo, hi, zlo, zhi), P, W)
    tracked = Projected(
        Recipe(G, Bcoef, decode, mu, V, lo, hi, zlo, zhi, refuse_at=TOL / 4), P, W
    )
    def one(sur, Z, Ynext):
        return float(
            np.max(np.where(P.scored(Ynext), np.abs(Z + sur.predict(Z) - Ynext), 0.0))
        )
    out = {"selection": sel, "degree": q, "one_step_worst": one(plain, Xe, Ye), "one_step_worst_corrected": one(corr, Xe, Ye),
           "rollout_worst": float(rollout(P, plain, test[:ROLLOUTS])["rollout_worst"]),
           "rollout_worst_corrected": float(rollout(P, corr, test[:ROLLOUTS])["rollout_worst"]), "kicks": []}  # fmt: skip
    idx = rng.choice(len(Xe), 300, replace=False)
    for eps in (0.003, 0.01, 0.03, 0.05):
        Zk = Xe[idx] + eps * rng.choice([-1.0, 1.0], Xe[idx].shape)
        Yk = P.step(Zk)
        r = Recipe(G, None, decode, mu, V, lo, hi, zlo, zhi).residual(Zk)
        ref = np.asarray(tracked.refuses(Zk), dtype=bool)
        e_c = np.max(
            np.where(P.scored(Yk), np.abs(Zk + corr.predict(Zk) - Yk), 0.0), axis=1
        )
        out["kicks"].append({"eps": eps, "uncorrected": one(plain, Zk, Yk), "corrected": float(e_c.max()), "residual_max": float(r.max()),
                             "refused_percent": float(100 * ref.mean()), "corrected_worst_answered": float(e_c[~ref].max()) if (~ref).any() else None})  # fmt: skip
        k = out["kicks"][-1]
        print(f"kick {eps}: uncorrected {k['uncorrected']:.2e}, corrected {k['corrected']:.2e}, residual {k['residual_max']:.2e}, "
              f"refused {k['refused_percent']:.0f}% (worst answered {k['corrected_worst_answered']})", flush=True)  # fmt: skip
    ks = out["kicks"]
    out["slope_uncorrected"] = float(
        np.polyfit(
            np.log([k["eps"] for k in ks]), np.log([k["uncorrected"] for k in ks]), 1
        )[0]
    )
    out["slope_corrected"] = float(
        np.polyfit(
            np.log([k["eps"] for k in ks]), np.log([k["corrected"] for k in ks]), 1
        )[0]
    )
    (OUT / "g4_hires_recipe.json").write_text(
        json.dumps(out, indent=1, sort_keys=True, default=float) + "\n"
    )
    print(f"selected q={q}: one-step {out['one_step_worst']:.2e} (corrected {out['one_step_worst_corrected']:.2e}); rollout {out['rollout_worst']:.2e} "
          f"(corrected {out['rollout_worst_corrected']:.2e}); slopes vs eps: uncorrected {out['slope_uncorrected']:.2f}, corrected {out['slope_corrected']:.2f}")  # fmt: skip


if __name__ == "__main__":
    main()
