"""
The reduced-surrogate recipe of the memo's section 10 item 1, end to end on HIRES (k = 3), with gradients where they pay.

  chart      pi(z) = V^T (z - mu), the top-3 principal directions of the visited log-states; decoder D(u) (total degree 4, least squares);
  data fit   G(u), total degree q, least squares from chart values to next states on all training pairs (no solve);
  selection  q in 4..8 by the two rollout conditions on the TRAINING chains (no test data): (i) chained from each training trajectory's
             first state, the worst error over its steps within the tolerance; (ii) the surrogate's own Jacobian norm max ||D Psi||
             (central differences, projection included) no larger than the dynamics' max ||D Phi|| (forward sensitivities at 200
             training states);
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

The first run (fit "data") refuted R2 and R3 and passed no degree in the selection; a diagnosis at 40 kicked test states found the
correction itself exact to second order (Phi(D(u)) + J n errs 1.3e-5 at eps = 0.003, 1.7e-4 at 0.01), and the data fit G the failure:
fitted to Phi(X) it already holds J n_X on the set, so adding B n counts it twice, and the training states lie on a few trajectories (curves
in the 3-D chart), so G is unconstrained between them (3.4e-2 at kicked chart values). Two further fits, predictions committed before
their run:
  "consistent"  G fitted to Y - B(u) n (the targets with the normal part removed, so G(u) estimates Phi(D(u)));
  "gradients"   as consistent, plus the chart-derivative rows d/du Phi(D(u)) = J DD(u) at the 300 sensitivity states (rows weighted by
                the chart box half-width, as in g2_hires);
  R4  with consistent targets the corrected test rollout is no worse than the uncorrected one, and the one-step error on the test states
      is within twice the data fit's 6.4e-4;
  R5  with the gradient rows the selected degree passes condition (ii) (max ||D Psi|| <= the dynamics' 1.86), the uncorrected kicked
      error is within twice Phi(D(u))'s own (5.5e-3 at eps = 0.003, 1.9e-2 at 0.01), and the corrected error at eps = 0.01 is within
      the tolerance with a slope in eps of at least 1.7.
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
EPS = (0.003, 0.01, 0.03, 0.05)


def log_jacobian(z):
    """d z(dt) / d z(0) in log10 coordinates: the full forward sensitivity with S(0) = diag(y0)."""
    y0 = 10.0**z
    y1, S1 = sensitivity_step(y0, np.diag(y0), P.dt)
    return S1 / y1[:, None]


class Recipe:
    """The reduced surrogate as an increment model: data fit, optional normal correction, refusal by the decoder residual."""

    def __init__(self, G, B, decode, mu, V, lo, hi, zlo, zhi, refuse_at=None):
        self.G, self.B, self.decode, self.mu, self.V = G, B, decode, mu, V
        self.lo, self.hi, self.zlo, self.zhi, self.refuse_at, self.evals = lo, hi, zlo, zhi, refuse_at, 0

    def chart(self, Z):
        return np.clip((Z - self.mu) @ self.V, self.lo, self.hi)

    def residual(self, Z):
        return np.max(np.abs(Z - self.decode(self.chart(Z))), axis=1)

    def refuses(self, Z):
        if self.refuse_at is None:
            return np.zeros(len(Z), dtype=bool)
        return self.residual(Z) > self.refuse_at

    def normal_term(self, Z):
        U = self.chart(Z)
        Bm = (poly(U, 2) @ self.B).reshape(len(Z), Z.shape[1], Z.shape[1])
        return np.einsum("nij,nj->ni", Bm, Z - self.decode(U))

    def predict(self, Z):
        out = self.G(self.chart(Z))
        if self.B is not None:
            out = out + self.normal_term(Z)
        return np.clip(out, self.zlo, self.zhi) - Z


def fit_chart(q, lo, hi, U, Y, Ug=None, dY=None):
    """Total-degree-q chart polynomial from values at U and, when given, chart derivatives dY (n, d, k) at Ug."""
    G = ChartPoly(q, lo, hi)
    if dY is None:
        return G.fit(U, Y)
    Vv, _ = G.design(U, False)
    _, Gg = G.design(Ug, True)
    w = (hi - lo) / 2
    A = np.vstack([Vv] + [w[j] * Gg[j] for j in range(len(Gg))])
    B = np.vstack([Y] + [w[j] * dY[:, :, j] for j in range(len(Gg))])
    G.coef = np.linalg.lstsq(A, B, rcond=None)[0]
    return G


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
    # the normal correction: B(u) fitted entrywise (degree 2 in u) to the dynamics' Jacobians at the 300 training states
    Bcoef = np.linalg.lstsq(poly(U[js], 2), Jd.reshape(len(js), -1), rcond=None)[0]
    normal = Recipe(None, Bcoef, decode, mu, V, lo, hi, zlo, zhi).normal_term
    h = 1e-6
    dD = np.stack([(decode(U[js] + h * e) - decode(U[js] - h * e)) / (2 * h) for e in np.eye(3)], axis=2)
    targets = {
        "data": (Y, None),
        "consistent": (Y - normal(X), None),
        "gradients": (Y - normal(X), np.einsum("nij,njk->nik", Jd, dD)),
    }
    Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    kick_idx = rng.choice(len(Xe), 300, replace=False)
    signs = {eps: rng.choice([-1.0, 1.0], Xe[kick_idx].shape) for eps in EPS}

    def worst(sur, Z, Ynext):
        return float(np.max(np.where(P.scored(Ynext), np.abs(Z + sur.predict(Z) - Ynext), 0.0)))

    def jacobian_norm(sur, Zs):
        DPsi = np.stack([(sur.predict(Zs + h * e) - sur.predict(Zs - h * e)) / (2 * h) + e for e in np.eye(X.shape[1])], axis=2)
        return float(np.max(np.linalg.norm(DPsi, ord=2, axis=(1, 2))))

    out = {"dynamics_jacobian_norm": dyn_norm, "fits": {}}
    for name, (T, dT) in targets.items():
        sel = []
        for q in range(4, 9):
            G = fit_chart(q, lo, hi, U, T, U[js], dT)
            sur = Projected(Recipe(G, None, decode, mu, V, lo, hi, zlo, zhi), P, W)
            chain = float(rollout(P, sur, train[:40])["rollout_worst"])
            psi = jacobian_norm(sur, X[js[:200]])
            sel.append({"degree": q, "train_chain_worst": chain, "surrogate_jacobian_norm": psi, "passes": bool(chain <= TOL and psi <= dyn_norm)})
            print(f"[{name}] q={q}: training chains worst {chain:.2e}, max ||D Psi|| {psi:.2f} (dynamics {dyn_norm:.2f}) -> {'pass' if sel[-1]['passes'] else 'fail'}", flush=True)
        passing = [r["degree"] for r in sel if r["passes"]]
        q = passing[-1] if passing else min(sel, key=lambda r: r["train_chain_worst"])["degree"]
        G = fit_chart(q, lo, hi, U, T, U[js], dT)
        plain = Projected(Recipe(G, None, decode, mu, V, lo, hi, zlo, zhi), P, W)
        corr = Projected(Recipe(G, Bcoef, decode, mu, V, lo, hi, zlo, zhi), P, W)
        res = {"selection": sel, "degree": q, "selected_by_conditions": bool(passing), "one_step_worst": worst(plain, Xe, Ye),
               "one_step_worst_corrected": worst(corr, Xe, Ye), "rollout_worst": float(rollout(P, plain, test[:ROLLOUTS])["rollout_worst"]),
               "rollout_worst_corrected": float(rollout(P, corr, test[:ROLLOUTS])["rollout_worst"]), "kicks": []}  # fmt: skip
        for eps in EPS:
            Zk = Xe[kick_idx] + eps * signs[eps]
            Yk = P.step(Zk)
            r = plain.model.residual(Zk)
            res["kicks"].append({"eps": eps, "uncorrected": worst(plain, Zk, Yk), "corrected": worst(corr, Zk, Yk), "residual_max": float(r.max()),
                                 "refused_percent_at_tol_over_4": float(100 * np.mean(r > TOL / 4))})  # fmt: skip
        ks = res["kicks"]
        for key in ("uncorrected", "corrected"):
            res[f"slope_{key}"] = float(np.polyfit(np.log([k["eps"] for k in ks]), np.log([k[key] for k in ks]), 1)[0])
        out["fits"][name] = res
        print(f"[{name}] q={q}{'' if passing else ' (no degree passed; smallest training-chain error)'}: one-step {res['one_step_worst']:.2e} "
              f"(corrected {res['one_step_worst_corrected']:.2e}); rollout {res['rollout_worst']:.2e} (corrected {res['rollout_worst_corrected']:.2e})", flush=True)  # fmt: skip
        for k in ks:
            print(f"[{name}]   kick {k['eps']}: uncorrected {k['uncorrected']:.2e}, corrected {k['corrected']:.2e}, residual {k['residual_max']:.2e}, "
                  f"refused at tol/4 {k['refused_percent_at_tol_over_4']:.0f}%", flush=True)  # fmt: skip
        print(f"[{name}]   slopes in eps: uncorrected {res['slope_uncorrected']:.2f}, corrected {res['slope_corrected']:.2f}", flush=True)
    (OUT / "g4_hires_recipe.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")


if __name__ == "__main__":
    main()
