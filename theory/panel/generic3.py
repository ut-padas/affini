"""
The generic recipe, third version (memo section P), after the council (theory/reviews/council-*.md): generic2.py's chart fit with the
gate calibrated for the maximum on states set apart for it, mixed inputs in the calibration, a propagation margin in the selection, and
the chart's search range up to the free dimension.

  splits      each of the three splits divides its held-out fifth of the training trajectories into a calibration half and a selection
              half (whole trajectories), so the states that set the gate are not the chains that choose the configuration;
  gate        as generic2 (the Lebesgue bound b and, on a reduced chart, the decoder residual), with e_hat the *largest* score
              err / (1 + b) over the calibration states and b_max the largest calibration bound: a control of the observed maximum, not
              of a quantile. With ``mixed`` the calibration states also include each calibration state mixed once with another
              calibration trajectory's state at the same step (w = 0.01 and 0.1, the mixing of the scores), labelled by the solver, so
              the gate is calibrated on the inputs a coupled run produces;
  selection   served when the worst held-out chain over the three selection halves is within half the tolerance (a propagation margin)
              and the mean solver share is below 50%; among served configurations the one with the largest margin (the smallest worst
              chain), then the smaller solver share;
  chart       k searched up to the free dimension (at most 40).
Scores and timings as generic2.run.
"""

import time

import numpy as np
from generic import TOL, choose_k, log, n_terms, pairs, split
from generic2 import RIDGES, SPLITS, bound, build, mixed_once, scores, timed

MARGIN = 0.5  # the selection's held-out chains must stay within MARGIN * tol
SERVED_SHARE = 0.5


def thirds(train, s):
    fit, val = split(train, s)
    half = max(1, len(val) // 2)
    return fit, val[:half], val[half:]


def mixed_pairs(P, trajs, rng):
    """Each state of ``trajs`` (but the last) mixed once with another of the trajectories at the same step, w = 0.01 and 0.1, and its solver step."""
    if len(trajs) < 2:
        return np.empty((0, trajs.shape[2])), np.empty((0, trajs.shape[2]))
    X = np.concatenate([mixed_once(P, trajs, w, rng) for w in (0.01, 0.1)])
    return X, P.step(X)


def calibrate(sur, P, cal_sets):
    """e_hat and b_max as the largest score and bound over the calibration states (module docstring)."""
    errs, bs = [], []
    for Xc, Yc in cal_sets:
        if len(Xc):
            e, b = scores(P, sur, Xc, Yc)
            errs.append(e)
            bs.append(b)
    e, b = np.concatenate(errs), np.concatenate(bs)
    s = e / (1 + b)
    s = s[np.isfinite(s)]
    sur.e_hat = float(np.max(s)) if len(s) else np.inf
    b = b[np.isfinite(b)]
    sur.b_max = float(np.max(b)) if len(b) else -np.inf


def calibration_sets(P, cal, mixed, rng):
    sets = [pairs(cal)]
    if mixed:
        sets.append(mixed_pairs(P, cal, rng))
    return sets


def select(P, train, k, dec_degree, rollout, extra=None, qmax=12, mixed=True) -> dict:
    m = k + P.system.held
    n_pairs = len(train) * 4 // 5 * (train.shape[1] - 1) + (0 if extra is None else len(extra[0]))
    cal_cache = {}
    rows, best_err, rising = [], np.inf, 0
    for q in range(1, qmax + 1):
        if n_terms(m, q) * 1.5 > n_pairs or n_terms(m, q) > 3000:
            break
        q_err = np.inf
        for ridge in RIDGES:
            per = []
            for s in SPLITS:
                fit, cal, sel = thirds(train, s)
                if s not in cal_cache:
                    cal_cache[s] = calibration_sets(P, cal, mixed, np.random.default_rng(1000 + s))
                sur, X, Y = build(P, fit, k, q, ridge, dec_degree, extra)
                calibrate(sur, P, cal_cache[s])
                if 2 * sur.e_hat > TOL / 2:  # the bound is at least 1: every input refused, the chain is the solver's own
                    res = {"rollout_worst": 0.0, "rollout_solved_by_step": [1.0]}
                else:
                    res = rollout(P, sur, sel)
                e_cal, b_cal = scores(P, sur, *cal_cache[s][0])
                per.append({"split": s, "chain_worst": float(res["rollout_worst"]), "solver_share": float(np.mean(res["rollout_solved_by_step"])),
                            "heldout_worst": float(np.nanmax(e_cal)), "heldout_median": float(np.nanmedian(e_cal)), "e_hat": sur.e_hat,
                            "b_max": sur.b_max, "rank": sur.G.rank, "terms": len(sur.G.idx)})  # fmt: skip
            row = {"degree": q, "ridge": ridge, "terms": n_terms(m, q), "chain_worst": max(p["chain_worst"] for p in per),
                   "solver_share": float(np.mean([p["solver_share"] for p in per])), "heldout_worst": max(p["heldout_worst"] for p in per),
                   "heldout_median": float(np.median([p["heldout_median"] for p in per])), "splits": per}  # fmt: skip
            row["served"] = bool(row["chain_worst"] <= MARGIN * TOL and row["solver_share"] < SERVED_SHARE)
            rows.append(row)
            q_err = min(q_err, row["heldout_worst"])
            log(f"  q={q} ridge={ridge:g}: chains {row['chain_worst']:.2e} solver {row['solver_share']:.2f} | calibration worst "
                f"{row['heldout_worst']:.2e} median {row['heldout_median']:.1e} | {'SERVED' if row['served'] else '-'}")  # fmt: skip
        best_err = min(best_err, q_err)
        rising = rising + 1 if q_err > 10 * best_err else 0
        if rising >= 2:
            break
    ok = [r for r in rows if r["served"]]
    best = min(ok, key=lambda r: (r["chain_worst"], r["solver_share"])) if ok else min(rows, key=lambda r: r["heldout_worst"])
    return {"configs": rows, "degree": best["degree"], "ridge": best["ridge"], "served": bool(ok), "stopped_at": rows[-1]["degree"] if rows else None,
            "cal_cache": cal_cache}  # fmt: skip


def run(P, train, test, seed=0, extra=None, solve_us=None, qmax=12, mixed=True, kmax=40) -> dict:
    from generic import mixing

    from src.benchmark.step_maps import rollout

    t0 = time.perf_counter()
    free = train.shape[2] - P.system.held
    kc = choose_k(P, train, min(kmax, free), extra)
    dec_degree = next((c["decoder_degree"] for c in kc["curve"] if c["k"] == kc["k"]), 1)
    log(f"  k={kc['k']} ({kc['reduction']}), decoder degree {dec_degree}")
    sel = select(P, train, kc["k"], dec_degree, rollout, extra, qmax, mixed)
    sur, _, _ = build(P, train, kc["k"], sel["degree"], sel["ridge"], dec_degree, extra)
    cal_all = []
    for s in SPLITS:  # the final gate: each split's fit scored on its own calibration states, pooled (the largest score and bound)
        fit, cal, _ = thirds(train, s)
        sv, _, _ = build(P, fit, kc["k"], sel["degree"], sel["ridge"], dec_degree, extra)
        for Xc, Yc in sel["cal_cache"][s]:
            if len(Xc):
                e, b = scores(P, sv, Xc, Yc)
                cal_all.append((e, b))
    e = np.concatenate([x[0] for x in cal_all])
    b = np.concatenate([x[1] for x in cal_all])
    s_ = (e / (1 + b))[np.isfinite(e / (1 + b))]
    sur.e_hat = float(np.max(s_)) if len(s_) else np.inf
    sur.b_max = float(np.max(b[np.isfinite(b)])) if np.isfinite(b).any() else -np.inf
    train_s = time.perf_counter() - t0
    Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    r = np.asarray(sur.refuses(Xe), dtype=bool)
    nxt = Xe + sur.predict(Xe)
    nxt[r] = Ye[r]
    everything_refused = 2 * sur.e_hat > TOL / 2
    if everything_refused:
        r40 = r20 = {"rollout_worst": 0.0, "rollout_solved_by_step": [1.0] * (test.shape[1] - 1)}
        mix_chain = {str(w): {"worst": 0.0, "solver_share": 1.0} for w in (0.01, 0.1)}
    else:
        r40, r20 = rollout(P, sur, test), rollout(P, sur, test[:20])
        rng = np.random.default_rng(seed + 1)
        mix_chain = {str(w): mixing(P, sur, test, w, rng) for w in (0.01, 0.1)}
    no_res = None
    if sur.decoder is not None and not everything_refused:
        sur.use_residual = False
        no_res = {str(w): mixing(P, sur, test, w, np.random.default_rng(seed + 1)) for w in (0.01, 0.1)}
        sur.use_residual = True
    from generic import worst_scored

    n = min(len(Xe), 2000)
    predict_us = timed(sur.predict, Xe[:n])
    gate_us = timed(sur.refuses, Xe[:n])
    rho = float(r.mean())
    out = {"d": int(test.shape[2]), "held": int(P.system.held), "n_train": int(len(train)), "k": kc["k"], "reduction": kc["reduction"],
           "k_curve": kc["curve"], "decoder_degree": dec_degree, "selection": sel["configs"], "degree": sel["degree"], "ridge": sel["ridge"],
           "served": sel["served"], "stopped_at": sel["stopped_at"], "mixed_calibration": mixed, "e_hat": sur.e_hat, "b_max": sur.b_max,
           "residual_max": None if sur.decoder is None else sur.decoder.r_max, "terms": len(sur.G.idx), "train_s": train_s,
           "one_step_worst": worst_scored(P, nxt, Ye), "refused_percent": 100 * rho, "refused_by_step": r.reshape(test.shape[0], -1).mean(axis=0).tolist(),
           "rollout_worst": float(r40["rollout_worst"]), "rollout_solver_share": float(np.mean(r40["rollout_solved_by_step"])),
           "rollout_solved_by_step": r40["rollout_solved_by_step"], "rollout_worst_20": float(r20["rollout_worst"]),
           "rollout_solver_share_20": float(np.mean(r20["rollout_solved_by_step"])), "mixing_chained": mix_chain,
           "mixing_chained_without_residual": no_res, "predict_us": predict_us, "gate_us": gate_us, "timing_batch": n, "solve_us": solve_us}  # fmt: skip
    if solve_us:
        out["speedup_nominal"] = solve_us / predict_us
        out["speedup_effective"] = solve_us / (gate_us + (1 - rho) * predict_us + rho * solve_us)
    log(f"  served={out['served']} k={out['k']} q={out['degree']}: one-step {out['one_step_worst']:.2e} (refused {out['refused_percent']:.1f}%), "
        f"rollout {out['rollout_worst']:.2e} (solver {100 * out['rollout_solver_share']:.1f}%); chained mixing 0.01 {mix_chain['0.01']['worst']:.2e} "
        f"({100 * mix_chain['0.01']['solver_share']:.0f}%), 0.1 {mix_chain['0.1']['worst']:.2e} ({100 * mix_chain['0.1']['solver_share']:.0f}%); "
        f"speed-up {out.get('speedup_effective', float('nan')):.1f} ({time.perf_counter() - t0:.0f} s)")  # fmt: skip
    return out


__all__ = ["bound", "run", "select"]
