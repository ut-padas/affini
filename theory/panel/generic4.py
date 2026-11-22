"""
The generic recipe, fourth version, after the council's second round (theory/reviews/council2-*.md): generic3.py with the fit, the gate and
the selection repaired. Everything not listed is generic3's (the thirds split, calibration for the maximum with mixed states, consistent
calibration of the residual term, the propagation margin, k up to the free dimension).

  fit         truncated SVD of the scaled Chebyshev design A = U S V^T, singular values below RCOND times the largest dropped: the
              coefficients are the minimum-norm fit on the retained space, the estimator Proposition 5.1's bound describes (generic3
              back-substituted through R whenever no diagonal entry was exactly zero, so a numerically singular design deployed large
              arbitrary coefficients along its undetermined directions). No ridge arm: ridge 1e-6 served nothing in R4c or R4d;
  gate        from one factor M = V_r S_r^-1 computed once: the bound b(t) = sqrt(N) |phi(t) M| (the Lebesgue function of the truncated
              fit), and the out-of-span share s(t) = |phi(t) - V_r V_r^T phi(t)| / |phi(t)|, where the data do not determine the fit
              and b is blind; refused when (1 + b) e_hat > tol / 2, b > b_max, s > s_max (the largest calibration share), the residual
              term (consistent), or overflow. One chart and one design per batch serve the prediction and the gate (the rollouts call
              refuses then predict on the same states), and the invariant projection is batched;
  selection   among configurations within the margin, the one with the smallest estimated cost per step, c = gate + (1 - rho) predict
              + rho solve (rho the selection chains' solver share, the times measured on the configuration's own surrogate): the fastest
              configuration that holds, not the most accurate (CHEMAKZO's degree 9 at an effective speed-up of 0.7 in R4c). A
              selection chain stops as soon as it cannot be served (an error above the margin, or a solver share above one half after
              ten steps);
  laws        the system's declared conservation laws, else the numerical search's laws only if they also hold at states drawn
              uniformly in the visited states' box (a linear law satisfies W F(y) = 0 at every y; laws found on the data alone include
              symmetries of the initial law and laws of the sampled states, which the off-data states break);
  budgets     recorded, not used by the gate: the share of test inputs the gate would refuse with the per-step budget tol / (2 A_n)
              (A_n the benchmark's accumulation factor, src.benchmark.step_maps.dynamics) and with tol / (2 (c n + sqrt n)) (c the
              sign coherence of the accepted one-step errors, n the steps), beside tol / 2.
"""

import time

import numpy as np
from generic import (
    TOL,
    choose_k,
    design,
    log,
    multi_indices,
    n_terms,
    pairs,
    worst_scored,
)
from generic2 import SPLITS, Decoder, timed
from generic2 import Surrogate as Surrogate2
from generic3 import MARGIN, SERVED_SHARE, calibration_sets, kept, thirds

from src.benchmark.step_maps import dynamics, invariants_of, visited

RCOND = 1e-10  # relative singular-value cutoff of the fit (numpy's lstsq default is eps * max(N, p), about 1e-12 at these sizes)


def verified_laws(P, train, n=600, tol=1e-9) -> np.ndarray:
    """Conservation laws (module docstring), rows orthonormal, in the system's own variables with zeros on the held inputs."""
    if P.system.conserved is not None:
        return invariants_of(P, train)
    Z = visited(train)
    free = Z.shape[1] - P.system.held
    lo, hi = Z.min(0), Z.max(0)
    Zb = np.random.default_rng(0).uniform(lo, hi, size=(n, Z.shape[1]))
    Z = np.concatenate([Z[np.random.default_rng(0).choice(len(Z), min(n, len(Z)), replace=False)], Zb])
    Y = 10.0**Z if P.system.log else Z
    F = np.array([np.asarray(P.system.rhs(y))[:free] for y in Y])
    F = F[np.all(np.isfinite(F), axis=1)]
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-300)
    _, sv, Vt = np.linalg.svd(F, full_matrices=True)
    sv = np.concatenate([sv, np.zeros(free - len(sv))])
    null = Vt[sv < tol * max(sv.max(), 1e-300)]
    W = np.zeros((len(null), Z.shape[1]))
    W[:, :free] = null
    return W


class Fit4:
    """The truncated-SVD fit and its gate factor (module docstring)."""

    def __init__(self, U, Y, q):
        self.q = q
        self.lo, self.hi = U.min(0), U.max(0)
        self.idx = multi_indices(U.shape[1], q)
        A = design(U, q, self.lo, self.hi, self.idx)
        self.N = len(A)
        Uu, s, Vt = np.linalg.svd(A, full_matrices=False)
        r = int(np.sum(s > RCOND * s[0]))
        self.rank, self.s = r, s
        self.Vr = Vt[:r].T
        self.M = self.Vr / s[:r]
        self.coef = self.M @ (Uu[:, :r].T @ Y)

    def phi(self, U):
        return design(U, self.q, self.lo, self.hi, self.idx)

    def __call__(self, U):
        return self.phi(U) @ self.coef

    def gate_terms(self, Phi):
        """(bound, out-of-span share) per row of the design Phi."""
        with np.errstate(all="ignore"):
            b = np.sqrt(self.N * np.sum((Phi @ self.M) ** 2, axis=1))
            inside = (Phi @ self.Vr) @ self.Vr.T
            s = np.linalg.norm(Phi - inside, axis=1) / np.maximum(np.linalg.norm(Phi, axis=1), 1e-300)
        return b, s


class Surrogate4(Surrogate2):
    """generic2.Surrogate with the repaired gate, one evaluation per batch shared by prediction and gate, and a batched projection."""

    def __init__(self, T, k, G, decoder):
        super().__init__(T, k, G, decoder)
        self.s_max = np.inf
        self._key, self._val = None, None

    def project(self, Z, Yhat):
        P, W = self.T.P, self.T.W
        if not len(W):
            return Yhat
        log_ = P.system.log
        y, yh = (10.0**Z, 10.0**Yhat) if log_ else (Z, Yhat)
        D = yh if log_ else np.ones_like(yh)
        Dw = W[None, :, :] * D[:, None, :]  # (n, m, d)
        r = np.einsum("md,nd->nm", W, y - yh)
        # z = Dw^+ r = Dw^T (Dw Dw^T)^+ r: the minimum-norm correction, batched (generic2 solved one lstsq per state)
        G = np.einsum("nmd,nkd->nmk", Dw, Dw)
        z = np.einsum("nmd,nm->nd", Dw, np.einsum("nmk,nk->nm", np.linalg.pinv(G), r))
        out = yh + D * z
        return np.log10(np.maximum(out, 1e-300)) if log_ else out

    def evaluate(self, Z, cache=True):
        """(next state, refused) for a batch, cached for the same array (the rollouts call refuses, then predict, on one batch)."""
        if cache and self._key is not None and self._key[0] is Z and self._key[1].shape == Z.shape and np.array_equal(self._key[1], Z):
            return self._val
        T = self.T
        U = T.chart(Z, self.k)
        Phi = self.G.phi(U)
        out = np.array(Z, dtype=float)
        with np.errstate(all="ignore"):
            g = Phi @ self.G.coef
            bad = ~np.all(np.isfinite(g), axis=1) | np.any(np.abs(g) > 1e6, axis=1)
            out[:, : T.free] = np.clip(np.where(np.isfinite(g), g, 0.0), T.zlo[: T.free], T.zhi[: T.free])
            out = self.project(Z, out)
            bad |= ~np.all(np.isfinite(out), axis=1)
            b, s = self.G.gate_terms(Phi)
        r = ~np.isfinite(b) | ((1 + b) * self.e_hat > TOL / 2) | (b > self.b_max) | (s > self.s_max) | bad
        if self.decoder is not None and self.use_residual:
            r |= self.decoder.residual(Z) > self.decoder.r_max
        self._key, self._val = (Z, np.array(Z, copy=True)), (out, r)
        return out, r

    def raw(self, Z):
        out, _ = self.evaluate(Z)
        return out, np.zeros(len(Z), dtype=bool)

    def refuses(self, Z):
        return self.evaluate(Z)[1]

    def predict(self, Z):
        return self.evaluate(Z)[0] - Z


def transform(P, fit_trajs, extra_X=None):
    from generic import Transform

    T = Transform.__new__(Transform)
    T.P, T.h = P, P.system.held
    T.free = fit_trajs.shape[2] - T.h
    T.cut = np.log10(P.floor) - 2.0 if (P.system.log and P.floor > 0) else -np.inf
    X = T.clip(visited(fit_trajs) if extra_X is None else np.concatenate([visited(fit_trajs), extra_X]))
    T.mu = X[:, : T.free].mean(axis=0)
    T.V = np.linalg.svd(X[:, : T.free] - T.mu, full_matrices=False)[2].T
    T.W = verified_laws(P, fit_trajs)
    T.zlo, T.zhi = X.min(0) - 2.0, X.max(0) + 0.5
    return T


def build(P, fit_trajs, k, q, dec_degree, extra=None):
    T = transform(P, fit_trajs, None if extra is None else extra[0])
    X, Y = pairs(fit_trajs)
    if extra is not None:
        X, Y = np.concatenate([X, extra[0]]), np.concatenate([Y, extra[1]])
    G = Fit4(T.chart(X, k), T.clip(Y)[:, : T.free], q)
    dec = Decoder(T, k, X, dec_degree) if k < T.free else None
    return Surrogate4(T, k, G, dec), X, Y


def calibrate(sur, P, cal_sets, consistent=True):
    """e_hat, b_max and s_max as the largest score, bound and out-of-span share over the calibration states the residual term accepts."""
    es, bs, ss = [], [], []
    for Xc, Yc in cal_sets:
        if not len(Xc):
            continue
        with np.errstate(all="ignore"):
            e = np.max(np.where(P.scored(Yc), np.abs(Xc + (sur.evaluate(Xc)[0] - Xc) - Yc), 0.0), axis=1)
        b, s = sur.G.gate_terms(sur.G.phi(sur.T.chart(Xc, sur.k)))
        ok = kept(sur, Xc, consistent)
        es.append(e[ok]), bs.append(b[ok]), ss.append(s[ok])
    e, b, s = np.concatenate(es), np.concatenate(bs), np.concatenate(ss)
    sc = e / (1 + b)
    sc = sc[np.isfinite(sc)]
    sur.e_hat = float(np.max(sc)) if len(sc) else np.inf
    sur.b_max = float(np.max(b[np.isfinite(b)])) if np.isfinite(b).any() else -np.inf
    sur.s_max = float(np.max(s[np.isfinite(s)])) if np.isfinite(s).any() else -np.inf
    sur._key = None


def chains(P, sur, true) -> dict:
    """The selection's held-out chains with fallback, stopped as soon as they cannot be served (module docstring)."""
    Z = true[:, 0].copy()
    h = P.system.held
    worst, solved = 0.0, []
    for k in range(true.shape[1] - 1):
        if h:
            Z[:, -h:] = true[:, k, -h:]
        nxt, r = sur.evaluate(Z)
        nxt = np.array(nxt)
        if r.any():
            nxt[r] = P.step(Z[r])
        Z = nxt
        worst = max(worst, worst_scored(P, Z, true[:, k + 1]))
        solved.append(float(r.mean()))
        if worst > MARGIN * TOL or (k >= 9 and np.mean(solved) > SERVED_SHARE):
            return {"rollout_worst": float(worst), "rollout_solved_by_step": solved, "stopped_early": True}
    return {"rollout_worst": float(worst), "rollout_solved_by_step": solved, "stopped_early": False}


def select(P, train, k, dec_degree, solve_us, extra=None, qmax=12) -> dict:
    m = k + P.system.held
    n_pairs = len(train) * 4 // 5 * (train.shape[1] - 1) + (0 if extra is None else len(extra[0]))
    cal_cache = {s: calibration_sets(P, thirds(train, s)[1], True, np.random.default_rng(1000 + s)) for s in SPLITS}
    rows, best_err, rising = [], np.inf, 0
    for q in range(1, qmax + 1):
        if n_terms(m, q) * 1.5 > n_pairs or n_terms(m, q) > 3000:
            break
        per = []
        for s in SPLITS:
            fit, cal, sel = thirds(train, s)
            sur, _, _ = build(P, fit, k, q, dec_degree, extra)
            calibrate(sur, P, cal_cache[s])
            share = float(np.mean(sur.refuses(pairs(sel)[0])))
            if 2 * sur.e_hat > TOL / 2:
                res = {"rollout_worst": 0.0, "rollout_solved_by_step": [1.0], "stopped_early": False}
            elif share >= SERVED_SHARE:
                res = {"rollout_worst": float("nan"), "rollout_solved_by_step": [share], "stopped_early": True}
            else:
                res = chains(P, sur, sel)
            Xs = pairs(sel)[0][:2000]
            cost = timed(lambda X, f=sur.evaluate: f(X, cache=False), Xs) if len(Xs) else np.nan  # one evaluation: prediction and gate
            Xc, Yc = cal_cache[s][0]
            with np.errstate(all="ignore"):
                e_cal = np.max(np.where(P.scored(Yc), np.abs(sur.evaluate(Xc)[0] - Yc), 0.0), axis=1)
            per.append({"split": s, "chain_worst": float(res["rollout_worst"]), "solver_share": float(np.mean(res["rollout_solved_by_step"])),
                        "stopped_early": res["stopped_early"], "heldout_worst": float(np.nanmax(e_cal)), "e_hat": sur.e_hat, "b_max": sur.b_max,
                        "s_max": sur.s_max, "rank": sur.G.rank, "terms": len(sur.G.idx), "evaluate_us": cost})  # fmt: skip
        chains_ = [p["chain_worst"] for p in per]
        rho = float(np.mean([p["solver_share"] for p in per]))
        ev = float(np.nanmedian([p["evaluate_us"] for p in per]))
        row = {"degree": q, "terms": n_terms(m, q), "chain_worst": float("inf") if any(np.isnan(chains_)) else max(chains_), "solver_share": rho,
               "heldout_worst": max(p["heldout_worst"] for p in per), "evaluate_us": ev, "cost_us": ev + rho * (solve_us or 0.0), "splits": per}  # fmt: skip
        row["served"] = bool(row["chain_worst"] <= MARGIN * TOL and rho < SERVED_SHARE and not any(p["stopped_early"] for p in per))
        rows.append(row)
        log(f"  q={q}: chains {row['chain_worst']:.2e} solver {rho:.2f} | calibration worst {row['heldout_worst']:.2e} | evaluate {ev:.0f} us, "
            f"cost {row['cost_us']:.0f} us | {'SERVED' if row['served'] else '-'}")  # fmt: skip
        best_err = min(best_err, row["heldout_worst"])
        rising = rising + 1 if row["heldout_worst"] > 10 * best_err else 0
        if rising >= 2:
            break
    ok = [r for r in rows if r["served"]]
    best = min(ok, key=lambda r: (r["cost_us"], r["chain_worst"])) if ok else min(rows, key=lambda r: r["heldout_worst"])
    return {"configs": rows, "degree": best["degree"], "served": bool(ok), "cal_cache": cal_cache}


def final(P, train, k, degree, dec_degree, cal_cache, extra=None):
    """Fitted on all of ``train``; its gate from each split's fit scored on that split's calibration states, pooled."""
    sur, _, _ = build(P, train, k, degree, dec_degree, extra)
    es, bs, ss = [], [], []
    for s in SPLITS:
        sv, _, _ = build(P, thirds(train, s)[0], k, degree, dec_degree, extra)
        calibrate(sv, P, cal_cache[s])
        es.append(sv.e_hat), bs.append(sv.b_max), ss.append(sv.s_max)
    sur.e_hat, sur.b_max, sur.s_max = max(es), max(bs), max(ss)
    return sur


def rebuild(P, train, rec: dict, extra=None):
    cal_cache = {s: calibration_sets(P, thirds(train, s)[1], True, np.random.default_rng(1000 + s)) for s in SPLITS}
    return final(P, train, rec["k"], rec["degree"], rec["decoder_degree"], cal_cache, extra)


def budget_shares(sur, Xe, Ye, P, A_n, steps) -> dict:
    """Refused shares of the test inputs under the three per-step budgets (module docstring), and the accepted errors' sign coherence."""
    nxt, r = sur.evaluate(Xe)
    b, _ = sur.G.gate_terms(sur.G.phi(sur.T.chart(Xe, sur.k)))
    acc = ~r
    if acc.any():
        e = np.where(P.scored(Ye[acc]), nxt[acc] - Ye[acc], 0.0)
        c = float(np.max(np.abs(e.mean(0)) / np.maximum(np.abs(e).mean(0), 1e-300)))
    else:
        c = float("nan")
    out = {"coherence": c, "accumulation": A_n, "steps": steps}
    for label, B in (("half_tol", TOL / 2), ("accumulation", TOL / (2 * A_n)), ("coherence", TOL / (2 * (c * steps + np.sqrt(steps))) if np.isfinite(c) else np.nan)):
        out[label] = {"budget": float(B), "refused_percent": float(100 * np.mean(r | ((1 + b) * sur.e_hat > B)))}
    return out


def run(P, train, test, name, seed=0, extra=None, solve_us=None, qmax=12, kmax=40) -> dict:
    from generic import mixing

    from src.benchmark.step_maps import rollout

    t0 = time.perf_counter()
    free = train.shape[2] - P.system.held
    kc = choose_k(P, train, min(kmax, free), extra)
    dec_degree = next((c["decoder_degree"] for c in kc["curve"] if c["k"] == kc["k"]), 1)
    log(f"  k={kc['k']} ({kc['reduction']}), decoder degree {dec_degree}")
    sel = select(P, train, kc["k"], dec_degree, solve_us, extra, qmax)
    sur = final(P, train, kc["k"], sel["degree"], dec_degree, sel["cal_cache"], extra)
    train_s = time.perf_counter() - t0
    Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    nxt, r = sur.evaluate(Xe)
    nxt = np.array(nxt)
    nxt[r] = Ye[r]
    everything_refused = 2 * sur.e_hat > TOL / 2
    if everything_refused:
        r40 = r20 = {"rollout_worst": 0.0, "rollout_solved_by_step": [1.0] * (test.shape[1] - 1)}
        mix_chain = {str(w): {"worst": 0.0, "solver_share": 1.0} for w in (0.01, 0.1)}
    else:
        r40, r20 = rollout(P, sur, test), rollout(P, sur, test[:20])
        rng = np.random.default_rng(seed + 1)
        mix_chain = {str(w): mixing(P, sur, test, w, rng) for w in (0.01, 0.1)}
    n = min(len(Xe), 2000)
    evaluate_us = timed(lambda X: sur.evaluate(X, cache=False), Xe[:n])
    rho = float(r.mean())
    by_step = np.asarray(r40["rollout_solved_by_step"])
    acc = dynamics(name)["accumulation"]
    out = {"d": int(test.shape[2]), "held": int(P.system.held), "n_train": int(len(train)), "k": kc["k"], "reduction": kc["reduction"],
           "k_curve": kc["curve"], "decoder_degree": dec_degree, "selection": sel["configs"], "degree": sel["degree"], "served": sel["served"],
           "e_hat": sur.e_hat, "b_max": sur.b_max, "s_max": sur.s_max, "rank": sur.G.rank, "terms": len(sur.G.idx), "laws": int(len(sur.T.W)),
           "residual_max": None if sur.decoder is None else sur.decoder.r_max, "train_s": train_s, "one_step_worst": worst_scored(P, nxt, Ye),
           "refused_percent": 100 * rho, "refused_by_step": r.reshape(test.shape[0], -1).mean(axis=0).tolist(),
           "rollout_worst": float(r40["rollout_worst"]), "rollout_solver_share": float(by_step.mean()), "rollout_solver_share_from_step_10": float(by_step[10:].mean()) if len(by_step) > 10 else None,
           "rollout_solved_by_step": r40["rollout_solved_by_step"], "rollout_worst_20": float(r20["rollout_worst"]), "mixing_chained": mix_chain,
           "evaluate_us": evaluate_us, "timing_batch": n, "solve_us": solve_us,
           "budgets": budget_shares(sur, Xe, Ye, P, float(np.max(acc)), test.shape[1] - 1)}  # fmt: skip
    if solve_us:
        for label, share in (("", float(by_step.mean())), ("_from_step_10", out["rollout_solver_share_from_step_10"])):
            if share is not None:
                out[f"speedup_effective{label}"] = solve_us / (evaluate_us + share * solve_us)
        out["speedup_nominal"] = solve_us / evaluate_us
    log(f"  served={out['served']} k={out['k']} q={out['degree']} rank {sur.G.rank}/{len(sur.G.idx)}: rollout {out['rollout_worst']:.2e} "
        f"(solver {100 * out['rollout_solver_share']:.1f}%), chained mixing {mix_chain['0.01']['worst']:.2e} / {mix_chain['0.1']['worst']:.2e}, "
        f"evaluate {evaluate_us:.0f} us, speed-up {out.get('speedup_effective', float('nan')):.1f} (from step 10: {out.get('speedup_effective_from_step_10', float('nan')):.1f}), "
        f"refused under the accumulation budget {out['budgets']['accumulation']['refused_percent']:.0f}% ({time.perf_counter() - t0:.0f} s)")  # fmt: skip
    return out


__all__ = ["Fit4", "Surrogate4", "build", "calibrate", "final", "rebuild", "run", "select", "verified_laws"]
