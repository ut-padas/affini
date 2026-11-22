"""
The generic recipe (memo section P): a reduced or full chart fit for any step map, selected on held-out trajectories and deployed with
a refusal rule from Proposition 5.1, the solver stepping what it refuses. It applies review recipe-round-1's fixes to theory/recipe's
first version, which was built on HIRES and overfitted elsewhere.

  transforms  fitted inside each fold: the floor clip (free components of a log system with a floor only, never held inputs), the
              principal directions of the visited free states, the chart bounds and the linear invariants;
  chart       u = (V_k^T (c(z_free) - mu), z_held), evaluated as it is (no clamp to the training box: the refusal rule decides);
  k           on fold 0: the smallest k whose lift error max |Phi(D(u), held) - Phi(z)| on 100 validation states (half from the
              trajectories' first steps, half later) is within half the tolerance for the better of an affine, a quadratic and a quartic
              decoder (the decoder only validates the chart; the deployed fit does not call it); none up to kmax: k = all free components
              (an exact affine chart) when there are at most 40, else the smallest-lift k, flagged "reduction not validated";
  fit         G(u): total-degree-q Chebyshev polynomials in the chart coordinates scaled by the training range, ridge least squares
              (lambda relative to the largest squared singular value of the design), from u to the next free state;
  refusal     the Lebesgue bound of Proposition 5.1 at the input, Lambda_X(t) <= sqrt(N phi(t)^T (A^T A + lambda I)^-1 phi(t)), against
              Lambda* = tol / (2 e_v) - 1, e_v the held-out one-step error; also every non-finite output or failed projection;
  projection  onto the input's linear invariants, as the benchmark's `Projected` but solved as the minimum-norm solution of the weighted
              constraint (W diag(y_hat)) z = W y - W y_hat by least squares, which stays defined when the Gram matrix loses rank;
  selection   three 80/20 splits of the training trajectories; for each (q, lambda) the worst over the splits of the held-out chained
              error with refusal and fallback; among the configurations within the tolerance, the one with the smallest solver share
              (then the smaller q); none within it: the problem is reported as not served by the recipe (the solver, or a solver-queried
              scheme, is the answer) together with the best configuration's numbers;
  extra       optional one-step pairs at further inputs (fresh draws from the problem's initial law), added to every fit and to the
              chart's principal directions; the held-out chains stay those of the training trajectories;
  scores      on the test trajectories with the refit on all training trajectories: one-step error and the share refused, rollouts over the
              first 20 and all test trajectories with their solver shares, and mixing rollouts (w = 0.01, 0.1) with refusal honoured.
"""

import time

import numpy as np

from src.benchmark.step_maps import invariants_of, visited
from src.problems.ode.system import FLOOR

TOL = 0.01
SPLITS = (0, 1, 2)
RIDGES = (0.0, 1e-6)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def multi_indices(m: int, q: int) -> list[tuple]:
    if m == 0:
        return [()]
    return [(a, *rest) for a in range(q + 1) for rest in multi_indices(m - 1, q - a)]


def n_terms(m: int, q: int) -> int:
    from math import comb

    return comb(m + q, q)


def design(U: np.ndarray, q: int, lo: np.ndarray, hi: np.ndarray, idx: list[tuple]) -> np.ndarray:
    """Chebyshev products of the chart coordinates scaled so the training range is [-1, 1]; evaluated beyond it without clamping."""
    s = (2 * U - lo - hi) / np.maximum(hi - lo, 1e-12)
    T = [np.polynomial.chebyshev.chebvander(s[:, j], q) for j in range(s.shape[1])]
    out = np.empty((len(U), len(idx)))
    for t, m in enumerate(idx):
        col = T[0][:, m[0]].copy()
        for j in range(1, len(m)):
            if m[j]:
                col *= T[j][:, m[j]]
        out[:, t] = col
    return out


class Transform:
    """Fold-fitted pieces: floor clip on free components, principal directions, invariants."""

    def __init__(self, P, train, extra_X=None):
        self.P = P
        self.h = P.system.held
        self.free = train.shape[2] - self.h
        self.cut = np.log10(P.floor) - 2.0 if (P.system.log and P.floor > 0) else -np.inf
        X = self.clip(visited(train) if extra_X is None else np.concatenate([visited(train), extra_X]))
        self.mu = X[:, : self.free].mean(axis=0)
        self.V = np.linalg.svd(X[:, : self.free] - self.mu, full_matrices=False)[2].T
        self.W = invariants_of(P, train)
        self.zlo, self.zhi = X.min(0) - 2.0, X.max(0) + 0.5

    def clip(self, Z):
        if not np.isfinite(self.cut):
            return Z
        out = np.array(Z, dtype=float)
        out[:, : self.free] = np.maximum(out[:, : self.free], self.cut)
        return out

    def chart(self, Z, k):
        Zc = self.clip(Z)
        return np.concatenate([(Zc[:, : self.free] - self.mu) @ self.V[:, :k], Zc[:, self.free :]], axis=1)


class Fit:
    """G with its Lebesgue bound: ridge least squares through the QR factor of the (augmented) design."""

    def __init__(self, U, Y, q, ridge):
        self.q, self.ridge = q, ridge
        self.lo, self.hi = U.min(0), U.max(0)
        self.idx = multi_indices(U.shape[1], q)
        A = design(U, q, self.lo, self.hi, self.idx)
        self.N = len(A)
        if ridge > 0:
            smax = np.linalg.norm(A, 2)
            A = np.vstack([A, np.sqrt(ridge) * smax * np.eye(A.shape[1])])
            Y = np.vstack([Y, np.zeros((A.shape[1], Y.shape[1]))])
        Q, self.R = np.linalg.qr(A)
        self.coef = np.linalg.solve(self.R, Q.T @ Y) if np.all(np.abs(np.diag(self.R)) > 0) else np.linalg.lstsq(A, Y, rcond=None)[0]
        self.rank = int(np.sum(np.abs(np.diag(self.R)) > 1e-10 * np.abs(np.diag(self.R)).max()))

    def __call__(self, U):
        return design(U, self.q, self.lo, self.hi, self.idx) @ self.coef

    def lebesgue_bound(self, U):
        Phi = design(U, self.q, self.lo, self.hi, self.idx)
        with np.errstate(all="ignore"):
            X = np.linalg.solve(self.R.T, Phi.T)
            return np.sqrt(self.N * np.sum(X * X, axis=0))


class Surrogate:
    """The deployed increment model: fit, output margin, robust invariant projection, refusal by the Lebesgue bound."""

    def __init__(self, T: Transform, k: int, G: Fit, lam_star: float):
        self.T, self.k, self.G, self.lam_star, self.evals = T, k, G, lam_star, 0

    def project(self, Z, Yhat):
        P, W = self.T.P, self.T.W
        if not len(W):
            return Yhat
        log_ = P.system.log
        y, yh = (10.0**Z, 10.0**Yhat) if log_ else (Z, Yhat)
        out = np.empty_like(yh)
        for i in range(len(yh)):
            Dw = W * (yh[i] if log_ else 1.0)
            # D^2 W^T (W D^2 W^T)^-1 r = D pinv(W D) r: the minimum-norm solution, defined when the Gram matrix loses rank
            z = np.linalg.lstsq(Dw, W @ (y[i] - yh[i]), rcond=None)[0]
            out[i] = yh[i] + (yh[i] if log_ else 1.0) * z
        return np.log10(np.maximum(out, FLOOR)) if log_ else out

    def _raw(self, Z):
        T = self.T
        out = np.array(Z, dtype=float)
        with np.errstate(all="ignore"):
            out[:, : T.free] = np.clip(self.G(T.chart(Z, self.k)), T.zlo[: T.free], T.zhi[: T.free])
            return self.project(Z, out)

    def refuses(self, Z):
        with np.errstate(all="ignore"):
            lam = self.G.lebesgue_bound(self.T.chart(Z, self.k))
            nxt = self._raw(Z)
        return ~np.isfinite(lam) | (lam > self.lam_star) | ~np.all(np.isfinite(nxt), axis=1)

    def predict(self, Z):
        nxt = self._raw(Z)
        nxt = np.where(np.isfinite(nxt), nxt, Z)
        return nxt - Z


def worst_scored(P, Zpred, Ztrue) -> float:
    if not len(Ztrue):
        return 0.0
    e = np.where(P.scored(Ztrue), np.abs(Zpred - Ztrue), 0.0)
    return float(np.max(e)) if np.all(np.isfinite(e)) else float("inf")


def pairs(train):
    d = train.shape[2]
    return visited(train), train[:, 1:].transpose(1, 0, 2).reshape(-1, d)


def lift_error(P, T: Transform, k: int, Xf, Xv, Yv, deg: int) -> float:
    U = T.chart(Xf, k)
    lo, hi = U.min(0), U.max(0)
    idx = multi_indices(U.shape[1], deg)
    if len(idx) * 1.5 > len(U):
        return float("inf")
    coef = np.linalg.lstsq(design(U, deg, lo, hi, idx), T.clip(Xf)[:, : T.free], rcond=None)[0]
    Zd = np.array(Xv, dtype=float)
    with np.errstate(all="ignore"):
        Zd[:, : T.free] = design(T.chart(Xv, k), deg, lo, hi, idx) @ coef
        if not np.all(np.isfinite(Zd)) or (P.system.log and np.any(Zd > 300)):
            return float("inf")  # a decoder that leaves the representable states cannot be a lift: no solve
        return worst_scored(P, T.clip(P.step(Zd)), T.clip(Yv))


def split(train, seed):
    order = np.random.default_rng(100 + seed).permutation(len(train))
    nv = max(1, len(train) // 5)
    return train[order[nv:]], train[order[:nv]]


def choose_k(P, train, kmax: int, extra=None) -> dict:
    fit, val = split(train, 0)
    T = Transform(P, fit, None if extra is None else extra[0])
    Xf = visited(fit) if extra is None else np.concatenate([visited(fit), extra[0]])
    rng = np.random.default_rng(7)
    first = val[:, 0]
    later = visited(val[:, 1:]) if val.shape[1] > 2 else first
    Xv = np.concatenate([first[rng.choice(len(first), min(50, len(first)), replace=False)], later[rng.choice(len(later), min(50, len(later)), replace=False)]])
    Yv = P.step(Xv)
    curve, solves = [], len(Xv)
    for k in range(1, min(kmax, T.free) + 1):
        errs = {deg: lift_error(P, T, k, Xf, Xv, Yv, deg) for deg in (1, 2, 4)}
        solves += len(Xv) * sum(np.isfinite(e) for e in errs.values())
        best = min(errs, key=errs.get)
        curve.append({"k": k, "lift_error": errs[best], "decoder_degree": best, "by_degree": {str(a): b for a, b in errs.items()}})
        log(f"  k={k}: lift {errs[best]:.3e} (decoder degree {best})")
        if errs[best] <= TOL / 2:
            return {"k": k, "reduction": "validated", "curve": curve, "solves": int(solves)}
    if T.free <= 40:
        return {"k": T.free, "reduction": "none (full chart)", "curve": curve, "solves": int(solves)}
    return {"k": min(curve, key=lambda r: r["lift_error"])["k"], "reduction": "not validated", "curve": curve, "solves": int(solves)}


def build(P, fit_trajs, k, q, ridge, val_pairs=None, extra=None):
    """
    A surrogate fitted on ``fit_trajs`` (and on the one-step pairs ``extra`` = (inputs, next states), when given); its Lambda* from the
    held-out one-step error on ``val_pairs`` (or the given e_v).
    """
    T = Transform(P, fit_trajs, None if extra is None else extra[0])
    X, Y = pairs(fit_trajs)
    if extra is not None:
        X, Y = np.concatenate([X, extra[0]]), np.concatenate([Y, extra[1]])
    U = T.chart(X, k)
    G = Fit(U, T.clip(Y)[:, : T.free], q, ridge)
    sur = Surrogate(T, k, G, np.inf)
    if isinstance(val_pairs, float):
        e_v = val_pairs
    else:
        Xv, Yv = val_pairs
        with np.errstate(all="ignore"):
            e_v = worst_scored(P, T.clip(Xv + sur.predict(Xv)), T.clip(Yv))
    sur.lam_star = TOL / (2 * e_v) - 1 if e_v > 0 else np.inf
    return sur, e_v


def select(P, train, k, qmax: int, rollout, extra=None) -> dict:
    """(q, ridge) by the worst over the splits of the held-out chained error with refusal and fallback (module docstring)."""
    m = k + P.system.held
    n_pairs = len(train) * 4 // 5 * (train.shape[1] - 1) + (0 if extra is None else len(extra[0]))
    configs = [(q, r) for q in range(1, qmax + 1) if n_terms(m, q) * 1.5 <= n_pairs and n_terms(m, q) <= 3000 for r in RIDGES]
    rows = []
    for q, r in configs:
        per = []
        for s in SPLITS:
            fit, val = split(train, s)
            sur, e_v = build(P, fit, k, q, r, pairs(val), extra)
            if sur.lam_star < 0:
                # the Lebesgue bound is nonnegative, so every input is refused: the chain is the solver's own, exact and fully solved
                res = {"rollout_worst": 0.0, "rollout_solved_by_step": [1.0]}
            else:
                res = rollout(P, sur, val)
            per.append({"split": s, "one_step_heldout": e_v, "lambda_star": sur.lam_star, "chain_worst": float(res["rollout_worst"]),
                        "solver_share": float(np.mean(res["rollout_solved_by_step"])), "rank": sur.G.rank, "terms": len(sur.G.idx)})  # fmt: skip
        row = {"degree": q, "ridge": r, "chain_worst": max(p["chain_worst"] for p in per), "solver_share": float(np.mean([p["solver_share"] for p in per])),
               "one_step_heldout": max(p["one_step_heldout"] for p in per), "splits": per}  # fmt: skip
        rows.append(row)
        log(f"  q={q} ridge={r:g}: held-out chains {row['chain_worst']:.3e}, solver share {row['solver_share']:.2f}, one-step {row['one_step_heldout']:.2e}")
    ok = [r for r in rows if r["chain_worst"] <= TOL]
    best = min(ok, key=lambda r: (r["solver_share"], r["degree"])) if ok else min(rows, key=lambda r: (r["chain_worst"] if np.isfinite(r["chain_worst"]) else np.inf))
    return {"configs": rows, "degree": best["degree"], "ridge": best["ridge"], "served": bool(ok), "e_v": best["one_step_heldout"]}


def mixing(P, sur, true, w, rng) -> dict:
    """The mixing rollout of theory/recipe with refusal honoured: refused states are stepped by the solver."""
    h = P.system.held
    free = true.shape[2] - h
    n = len(true)
    partners = np.array([[rng.choice([j for j in range(n) if j != i]) for i in range(n)] for _ in range(true.shape[1] - 1)])
    lin = (lambda Z: 10.0**Z) if P.system.log else (lambda Z: Z)
    back = (lambda Y: np.log10(np.maximum(Y, 1e-300))) if P.system.log else (lambda Y: Y)

    def mix(Z, k):
        out = Z.copy()
        out[:, :free] = back((1 - w) * lin(Z[:, :free]) + w * lin(true[partners[k], k, :free]))
        if h:
            out[:, free:] = true[:, k, free:]
        return out

    Zs, Zt, worst, solved = true[:, 0].copy(), true[:, 0].copy(), 0.0, []
    for k in range(true.shape[1] - 1):
        Zt = P.step(mix(Zt, k))
        Zin = mix(Zs, k)
        r = np.asarray(sur.refuses(Zin), dtype=bool)
        Zs = Zin + sur.predict(Zin)
        if r.any():
            Zs[r] = P.step(Zin[r])
        solved.append(float(r.mean()))
        worst = max(worst, worst_scored(P, Zs, Zt))
    return {"worst": worst, "solver_share": float(np.mean(solved))}


def run(P, train, test, kmax=12, qmax=8, seed=0, extra=None) -> dict:
    """The generic recipe end to end on one problem (module docstring); every number the memo quotes is in the returned dict."""
    from src.benchmark.step_maps import rollout

    t0 = time.perf_counter()
    kc = choose_k(P, train, kmax, extra)
    log(f"  k={kc['k']} ({kc['reduction']})")
    sel = select(P, train, kc["k"], qmax, rollout, extra)
    sur, _ = build(P, train, kc["k"], sel["degree"], sel["ridge"], float(sel["e_v"]), extra)
    Xe = test[:, :-1].reshape(-1, test.shape[2])
    Ye = test[:, 1:].reshape(-1, test.shape[2])
    r = np.asarray(sur.refuses(Xe), dtype=bool)
    nxt = Xe + sur.predict(Xe)
    nxt[r] = Ye[r]  # a refused state is stepped by the solver, whose step is the reference
    if sur.lam_star < 0:  # every input refused (see select): every chain is the solver's own
        r40 = r20 = {"rollout_worst": 0.0, "rollout_solved_by_step": [1.0]}
        mix = {str(w): {"worst": 0.0, "solver_share": 1.0} for w in (0.01, 0.1)}
    else:
        r40 = rollout(P, sur, test)
        r20 = rollout(P, sur, test[:20])
        rng = np.random.default_rng(seed + 1)
        mix = {str(w): mixing(P, sur, test, w, rng) for w in (0.01, 0.1)}
    out = {"d": int(test.shape[2]), "held": int(P.system.held), "n_train": int(len(train)), "k": kc["k"], "reduction": kc["reduction"],
           "k_curve": kc["curve"], "k_solves": kc["solves"], "selection": sel["configs"], "degree": sel["degree"], "ridge": sel["ridge"],
           "served": sel["served"], "lambda_star": float(sur.lam_star), "rank": sur.G.rank, "terms": len(sur.G.idx),
           "one_step_worst": worst_scored(P, nxt, Ye), "one_step_refused": float(r.mean()),
           "rollout_worst": float(r40["rollout_worst"]), "rollout_solver_share": float(np.mean(r40["rollout_solved_by_step"])),
           "rollout_worst_20": float(r20["rollout_worst"]), "rollout_solver_share_20": float(np.mean(r20["rollout_solved_by_step"])),
           "mixing": mix, "seconds": time.perf_counter() - t0}  # fmt: skip
    log(f"  served={out['served']} k={out['k']} q={out['degree']} ridge={out['ridge']:g}: one-step {out['one_step_worst']:.2e} "
        f"(refused {out['one_step_refused']:.2f}), rollout {out['rollout_worst']:.2e} (solver {out['rollout_solver_share']:.2f}); "
        f"mixing 0.01: {mix['0.01']['worst']:.2e} (solver {mix['0.01']['solver_share']:.2f}), 0.1: {mix['0.1']['worst']:.2e} "
        f"(solver {mix['0.1']['solver_share']:.2f}) ({out['seconds']:.0f} s)")  # fmt: skip
    return out
