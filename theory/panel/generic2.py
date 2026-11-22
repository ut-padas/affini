"""
The generic recipe, second version (memo section P), after review round 10: the same chart fit as generic.py (fold-fitted transforms, a
lift-validated k, total-degree Chebyshev fits from trajectories and optional extra one-step pairs), with the refusal rule calibrated on
held-out data instead of inserted into Proposition 5.1, a residual term for inputs displaced across the chart's fibres, a fixed meaning
of "served", and the per-factor diagnosis recorded at every degree.

  gate        b(t) is the Lebesgue bound of Proposition 5.1 at the input (generic.Fit.lebesgue_bound; a pseudo-inverse where the design
              is rank-deficient). Every held-out state of the three splits is scored s(v) = err(v) / (1 + b(v)); e_hat is the 99%
              quantile of the scores and b_max the largest held-out bound (the range over which errors were validated).
              An input is accepted when (1 + b(t)) e_hat <= tol / 2 and b(t) <= b_max; when the chart drops coordinates (k < free),
              also when its decoder residual r(t) = max |c(t) - D(pi t)| (the decoder of the degree the k rule validated) is at most
              the largest residual over the fitted states. Overflowing fits (before the output margin) are refused. This is a
              distributional calibration (T4), not a worst-case bound: e_hat estimates e* through held-out states;
  served      the worst held-out chain over the three splits within the tolerance (refusal and solver fallback active) and a mean
              solver share below 50%, fixed before the run; when no configuration is served, the one with the smallest held-out
              one-step error is reported;
  degrees     q = 1, 2, ... while the terms are at most 3,000 and at most two thirds of the fitted pairs, and until the held-out worst
              error has exceeded ten times its best for two consecutive degrees (qmax 12);
  diagnosis   per degree: the in-sample worst (an estimate of the approximation factor on the samples), the held-out median and worst,
              the bound and the step index of the worst held-out state (the sampling factor), the lift floor of the chart;
  scores      as generic.run, plus the one-step error and refusal on mixed inputs (each test state mixed once, w = 0.01, 0.1), the
              chained mixing rollouts with and without the residual term, and median-of-three timings for the fit, the gate and the
              solve (effective and nominal speed-up, Corollary 5.3 (c), with the batch size).
"""

import time

import numpy as np
from generic import (
    TOL,
    Fit,
    Transform,
    choose_k,
    design,
    log,
    multi_indices,
    n_terms,
    pairs,
    split,
    worst_scored,
)

SPLITS = (0, 1, 2)
RIDGES = (0.0, 1e-6)
SERVED_SHARE = 0.5


def bound(G: Fit, U):
    Phi = design(U, G.q, G.lo, G.hi, G.idx)
    with np.errstate(all="ignore"):
        if G.rank < len(G.idx):
            X = np.linalg.lstsq(G.R.T, Phi.T, rcond=None)[0]
        else:
            X = np.linalg.solve(G.R.T, Phi.T)
        return np.sqrt(G.N * np.sum(X * X, axis=0))


class Decoder:
    """The chart's decoder D(u) at a given total degree, fitted on the fitted states; residual r(t) = max |c(t) - D(pi t)| (free part)."""

    def __init__(self, T: Transform, k: int, X: np.ndarray, deg: int):
        self.T, self.k, self.deg = T, k, deg
        U = T.chart(X, k)
        self.lo, self.hi = U.min(0), U.max(0)
        self.idx = multi_indices(U.shape[1], deg)
        self.coef = np.linalg.lstsq(design(U, deg, self.lo, self.hi, self.idx), T.clip(X)[:, : T.free], rcond=None)[0]
        self.r_max = float(np.max(self.residual(X)))

    def residual(self, Z):
        with np.errstate(all="ignore"):
            D = design(self.T.chart(Z, self.k), self.deg, self.lo, self.hi, self.idx) @ self.coef
            return np.max(np.abs(self.T.clip(Z)[:, : self.T.free] - D), axis=1)


class Surrogate:
    """The deployed increment model of generic.Surrogate with the calibrated gate and the residual term."""

    def __init__(self, T: Transform, k: int, G: Fit, decoder: Decoder | None):
        self.T, self.k, self.G, self.decoder, self.evals = T, k, G, decoder, 0
        self.e_hat, self.b_max, self.use_residual = np.inf, -np.inf, True

    def project(self, Z, Yhat):
        P, W = self.T.P, self.T.W
        if not len(W):
            return Yhat
        log_ = P.system.log
        y, yh = (10.0**Z, 10.0**Yhat) if log_ else (Z, Yhat)
        out = np.empty_like(yh)
        for i in range(len(yh)):
            Dw = W * (yh[i] if log_ else 1.0)
            z = np.linalg.lstsq(Dw, W @ (y[i] - yh[i]), rcond=None)[0]
            out[i] = yh[i] + (yh[i] if log_ else 1.0) * z
        return np.log10(np.maximum(out, 1e-300)) if log_ else out

    def raw(self, Z):
        """(next state, whether the fit overflowed before the output margin)."""
        T = self.T
        out = np.array(Z, dtype=float)
        with np.errstate(all="ignore"):
            g = self.G(T.chart(Z, self.k))
            bad = ~np.all(np.isfinite(g), axis=1) | np.any(np.abs(g) > 1e6, axis=1)
            out[:, : T.free] = np.clip(np.where(np.isfinite(g), g, 0.0), T.zlo[: T.free], T.zhi[: T.free])
            out = self.project(Z, out)
        return out, bad | ~np.all(np.isfinite(out), axis=1)

    def refuses(self, Z):
        b = bound(self.G, self.T.chart(Z, self.k))
        _, bad = self.raw(Z)
        r = ~np.isfinite(b) | ((1 + b) * self.e_hat > TOL / 2) | (b > self.b_max) | bad
        if self.decoder is not None and self.use_residual:
            r |= self.decoder.residual(Z) > self.decoder.r_max
        return r

    def predict(self, Z):
        nxt, _ = self.raw(Z)
        return np.where(np.isfinite(nxt), nxt, Z) - Z


def build(P, fit_trajs, k, q, ridge, dec_degree, extra=None):
    """A surrogate fitted on ``fit_trajs`` (plus ``extra`` one-step pairs), with its decoder when k < free; the gate is not yet calibrated."""
    T = Transform(P, fit_trajs, None if extra is None else extra[0])
    X, Y = pairs(fit_trajs)
    if extra is not None:
        X, Y = np.concatenate([X, extra[0]]), np.concatenate([Y, extra[1]])
    G = Fit(T.chart(X, k), T.clip(Y)[:, : T.free], q, ridge)
    dec = Decoder(T, k, X, dec_degree) if k < T.free else None
    return Surrogate(T, k, G, dec), X, Y


def scores(P, sur, Xv, Yv):
    """Held-out one-step errors, bounds and the scores err / (1 + b)."""
    with np.errstate(all="ignore"):
        err = np.max(np.where(P.scored(Yv), np.abs(Xv + sur.predict(Xv) - Yv), 0.0), axis=1)
    b = bound(sur.G, sur.T.chart(Xv, sur.k))
    return err, b


def calibrate(sur, errs, bs):
    s = np.concatenate(errs) / (1 + np.concatenate(bs))
    s = s[np.isfinite(s)]
    sur.e_hat = float(np.quantile(s, 0.99)) if len(s) else np.inf
    allb = np.concatenate(bs)
    allb = allb[np.isfinite(allb)]
    sur.b_max = float(np.max(allb)) if len(allb) else -np.inf  # a quantile would refuse that share of in-distribution inputs by construction


def select(P, train, k, dec_degree, rollout, extra=None, qmax=12) -> dict:
    m = k + P.system.held
    n_pairs = len(train) * 4 // 5 * (train.shape[1] - 1) + (0 if extra is None else len(extra[0]))
    rows, best_err, rising = [], np.inf, 0
    for q in range(1, qmax + 1):
        if n_terms(m, q) * 1.5 > n_pairs or n_terms(m, q) > 3000:
            break
        q_err = np.inf
        for ridge in RIDGES:
            per, errs, bs = [], [], []
            for s in SPLITS:
                fit, val = split(train, s)
                sur, X, Y = build(P, fit, k, q, ridge, dec_degree, extra)
                Xv, Yv = pairs(val)
                err, b = scores(P, sur, Xv, Yv)
                calibrate(sur, [err], [b])
                if 2 * sur.e_hat > TOL / 2:
                    # the bound is at least 1, so every input is refused: the chain is the solver's own (error 0, fully solved)
                    res = {"rollout_worst": 0.0, "rollout_solved_by_step": [1.0]}
                else:
                    res = rollout(P, sur, val)
                with np.errstate(all="ignore"):
                    ins = float(np.max(np.where(P.scored(Y), np.abs(X + sur.predict(X) - Y), 0.0)))
                iw = int(np.nanargmax(np.where(np.isfinite(err), err, -1)))
                per.append({"split": s, "chain_worst": float(res["rollout_worst"]), "solver_share": float(np.mean(res["rollout_solved_by_step"])),
                            "heldout_worst": float(np.nanmax(err)), "heldout_median": float(np.nanmedian(err)), "insample_worst": ins,
                            "bound_at_worst": float(b[iw]), "step_of_worst": int(iw // len(val)), "e_hat": sur.e_hat, "b_max": sur.b_max,
                            "rank": sur.G.rank, "terms": len(sur.G.idx)})  # fmt: skip
                errs.append(err)
                bs.append(b)
            row = {"degree": q, "ridge": ridge, "terms": n_terms(m, q), "chain_worst": max(p["chain_worst"] for p in per),
                   "solver_share": float(np.mean([p["solver_share"] for p in per])), "heldout_worst": max(p["heldout_worst"] for p in per),
                   "heldout_median": float(np.median([p["heldout_median"] for p in per])), "insample_worst": max(p["insample_worst"] for p in per),
                   "splits": per}  # fmt: skip
            row["served"] = bool(row["chain_worst"] <= TOL and row["solver_share"] < SERVED_SHARE)
            rows.append(row)
            q_err = min(q_err, row["heldout_worst"])
            log(f"  q={q} ridge={ridge:g}: chains {row['chain_worst']:.2e} solver {row['solver_share']:.2f} | held-out worst {row['heldout_worst']:.2e} "
                f"median {row['heldout_median']:.1e} in-sample {row['insample_worst']:.1e} | {'SERVED' if row['served'] else '-'}")  # fmt: skip
        best_err = min(best_err, q_err)
        rising = rising + 1 if q_err > 10 * best_err else 0
        if rising >= 2:
            break
    ok = [r for r in rows if r["served"]]
    best = min(ok, key=lambda r: (r["solver_share"], r["degree"])) if ok else min(rows, key=lambda r: r["heldout_worst"])
    return {"configs": rows, "degree": best["degree"], "ridge": best["ridge"], "served": bool(ok), "stopped_at": rows[-1]["degree"] if rows else None}


def mixed_once(P, test, w, rng):
    h = P.system.held
    free = test.shape[2] - h
    X = test[:, :-1]
    part = np.array([[rng.choice([j for j in range(len(test)) if j != i]) for _ in range(X.shape[1])] for i in range(len(test))])
    Xp = X[part, np.arange(X.shape[1])[None, :]]
    out = X.copy()
    if P.system.log:
        out[..., :free] = np.log10(np.maximum((1 - w) * 10.0 ** X[..., :free] + w * 10.0 ** Xp[..., :free], 1e-300))
    else:
        out[..., :free] = (1 - w) * X[..., :free] + w * Xp[..., :free]
    return out.reshape(-1, test.shape[2])


def timed(fn, Z, rounds=3):
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn(Z)
        ts.append(time.perf_counter() - t0)
    return 1e6 * float(np.median(ts)) / len(Z)


def run(P, train, test, kmax=12, seed=0, extra=None, solve_us=None, qmax=12) -> dict:
    from generic import mixing

    from src.benchmark.step_maps import rollout

    t0 = time.perf_counter()
    kc = choose_k(P, train, kmax, extra)
    dec_degree = next((c["decoder_degree"] for c in kc["curve"] if c["k"] == kc["k"]), 1)
    log(f"  k={kc['k']} ({kc['reduction']}), decoder degree {dec_degree}")
    sel = select(P, train, kc["k"], dec_degree, rollout, extra, qmax)
    sur, X, Y = build(P, train, kc["k"], sel["degree"], sel["ridge"], dec_degree, extra)
    errs, bs = [], []
    for s in SPLITS:  # the final gate: scores of the chosen configuration's split fits on their held-out states, pooled
        fit, val = split(train, s)
        sv, _, _ = build(P, fit, kc["k"], sel["degree"], sel["ridge"], dec_degree, extra)
        e, b = scores(P, sv, *pairs(val))
        errs.append(e)
        bs.append(b)
    calibrate(sur, errs, bs)
    train_s = time.perf_counter() - t0
    Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    r = np.asarray(sur.refuses(Xe), dtype=bool)
    nxt = Xe + sur.predict(Xe)
    nxt[r] = Ye[r]
    everything_refused = 2 * sur.e_hat > TOL / 2  # as in select: the chains are the solver's own
    if everything_refused:
        r40 = r20 = {"rollout_worst": 0.0, "rollout_solved_by_step": [1.0] * (test.shape[1] - 1)}
        mix_chain = {str(w): {"worst": 0.0, "solver_share": 1.0} for w in (0.01, 0.1)}
    else:
        r40, r20 = rollout(P, sur, test), rollout(P, sur, test[:20])
        rng = np.random.default_rng(seed + 1)
        mix_chain = {str(w): mixing(P, sur, test, w, rng) for w in (0.01, 0.1)}
    mix_once = {}
    for w in () if everything_refused else (0.01, 0.1):
        Z = mixed_once(P, test, w, np.random.default_rng(seed + 5))
        Zn = P.step(Z)
        rz = np.asarray(sur.refuses(Z), dtype=bool)
        with np.errstate(all="ignore"):
            ez = np.max(np.where(P.scored(Zn), np.abs(Z + sur.predict(Z) - Zn), 0.0), axis=1)
        mix_once[str(w)] = {"refused_percent": float(100 * rz.mean()), "worst_accepted": float(np.max(ez[~rz])) if (~rz).any() else 0.0}
    no_res = None
    if sur.decoder is not None and not everything_refused:
        sur.use_residual = False
        no_res = {str(w): mixing(P, sur, test, w, np.random.default_rng(seed + 1)) for w in (0.01, 0.1)}
        sur.use_residual = True
    n = min(len(Xe), 2000)
    predict_us = timed(sur.predict, Xe[:n])
    gate_us = timed(sur.refuses, Xe[:n])
    rho = float(r.mean())
    out = {"d": int(test.shape[2]), "held": int(P.system.held), "n_train": int(len(train)), "extra": 0 if extra is None else int(len(extra[0])),
           "k": kc["k"], "reduction": kc["reduction"], "k_curve": kc["curve"], "k_solves": kc["solves"], "decoder_degree": dec_degree,
           "lift_floor": next((c["lift_error"] for c in kc["curve"] if c["k"] == kc["k"]), None),
           "selection": sel["configs"], "degree": sel["degree"], "ridge": sel["ridge"], "served": sel["served"], "stopped_at": sel["stopped_at"],
           "e_hat": sur.e_hat, "b_max": sur.b_max, "residual_max": None if sur.decoder is None else sur.decoder.r_max,
           "rank": sur.G.rank, "terms": len(sur.G.idx), "train_s": train_s,
           "one_step_worst": worst_scored(P, nxt, Ye), "refused_percent": 100 * rho,
           "refused_by_step": r.reshape(test.shape[0], -1).mean(axis=0).tolist(),
           "rollout_worst": float(r40["rollout_worst"]), "rollout_solver_share": float(np.mean(r40["rollout_solved_by_step"])),
           "rollout_solved_by_step": r40["rollout_solved_by_step"], "rollout_worst_20": float(r20["rollout_worst"]),
           "rollout_solver_share_20": float(np.mean(r20["rollout_solved_by_step"])),
           "mixing_chained": mix_chain, "mixing_chained_without_residual": no_res, "mixing_once": mix_once,
           "predict_us": predict_us, "gate_us": gate_us, "timing_batch": n, "solve_us": solve_us}  # fmt: skip
    if solve_us:
        out["speedup_nominal"] = solve_us / predict_us
        out["speedup_effective"] = solve_us / (gate_us + (1 - rho) * predict_us + rho * solve_us)
    log(f"  served={out['served']} k={out['k']} q={out['degree']}: one-step {out['one_step_worst']:.2e} (refused {out['refused_percent']:.1f}%), "
        f"rollout {out['rollout_worst']:.2e} (solver {100 * out['rollout_solver_share']:.1f}%); chained mixing 0.01 {mix_chain['0.01']['worst']:.2e} "
        f"({100 * mix_chain['0.01']['solver_share']:.0f}%), 0.1 {mix_chain['0.1']['worst']:.2e} ({100 * mix_chain['0.1']['solver_share']:.0f}%); "
        f"speed-up {out.get('speedup_effective', float('nan')):.1f} ({time.perf_counter() - t0:.0f} s)")  # fmt: skip
    return out
