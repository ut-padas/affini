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
from generic3 import MARGIN, SERVED_SHARE, calibration_sets, kept, mixed_pairs, thirds

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

    def __init__(self, U, Y, q, R=None, idx=None):
        self.lo, self.hi = U.min(0), U.max(0)
        self.idx = multi_indices(U.shape[1], q) if idx is None else idx
        self.q = max((max(a) for a in self.idx if len(a)), default=0)  # the Chebyshev tables' degree
        self.rs = None if R is None or not R.shape[1] else np.maximum(np.abs(R).max(0), 1e-300)
        A = self.phi(U, R)
        self.N = len(A)
        Uu, s, Vt = np.linalg.svd(A, full_matrices=False)
        r = int(np.sum(s > RCOND * s[0]))
        self.rank, self.s = r, s
        self.Vr = Vt[:r].T
        self.M = self.Vr / s[:r]
        self.coef = self.M @ (Uu[:, :r].T @ Y)

    def phi(self, U, R=None):
        """The design: Chebyshev products of the chart coordinates, then (a tube variant) the normal coordinates, linear, scaled."""
        A = design(U, self.q, self.lo, self.hi, self.idx)
        return A if self.rs is None else np.concatenate([A, R / self.rs], axis=1)

    def sketch(self, cols=32, seed=0):
        """Replace the gate's two products by Gaussian sketches with ``cols`` columns (E |x S|^2 = |x|^2): the bound from Phi (M S) and
        the out-of-span share from Phi (I - V_r V_r^T) S', each a p x cols product instead of p x r. The gate is calibrated on the
        sketched statistics, so the statistic deployed is the one calibrated."""
        rng = np.random.default_rng(seed)
        S = rng.standard_normal((self.M.shape[1], cols)) / np.sqrt(cols)
        S2 = rng.standard_normal((self.M.shape[0], cols)) / np.sqrt(cols)
        self.MS = self.M @ S
        self.QS = S2 - self.Vr @ (self.Vr.T @ S2)

    def gate_terms(self, Phi):
        """(bound, out-of-span share) per row of the design Phi."""
        with np.errstate(all="ignore"):
            if getattr(self, "MS", None) is not None:
                b = np.sqrt(self.N * np.sum((Phi @ self.MS) ** 2, axis=1))
                s = np.linalg.norm(Phi @ self.QS, axis=1) / np.maximum(np.linalg.norm(Phi, axis=1), 1e-300)
                return b, s
            b = np.sqrt(self.N * np.sum((Phi @ self.M) ** 2, axis=1))
            inside = (Phi @ self.Vr) @ self.Vr.T
            s = np.linalg.norm(Phi - inside, axis=1) / np.maximum(np.linalg.norm(Phi, axis=1), 1e-300)
        return b, s


class Surrogate4(Surrogate2):
    """generic2.Surrogate with the repaired gate, one evaluation per batch shared by prediction and gate, and a batched projection."""

    def __init__(self, T, k, G, decoder, n_normal=0):
        super().__init__(T, k, G, decoder)
        self.s_max, self.n_normal = np.inf, n_normal
        self.gate, self.env, self.rho_max = "product", None, np.inf
        self.b_cap, self.charge, self.lam = np.inf, None, 1.0
        self._key, self._val = None, None

    def offset(self, Z):
        """rho: the input's offset from the bundle, the decoder residual (0 on a full chart)."""
        return self.decoder.residual(Z) if self.decoder is not None else np.zeros(len(Z))

    def gate_bound(self, Z, Phi=None):
        """(the gate's error bound, the refusals by its caps) per row: the product (1 + b) e_hat with b_max, s_max and the residual
        term; or the envelope a0 + a1 b + a2 rho + a3 rho^2 with the caps b_max, s_max, rho_max (the calibration ranges)."""
        Phi = self.design(Z) if Phi is None else Phi
        b, s = self.G.gate_terms(Phi)
        if self.gate == "capped":
            rho = self.offset(Z)
            a = self.env if self.env is not None else (np.inf, 0.0, 0.0, 0.0)
            bound = a[0] + a[1] * b + a[2] * rho + a[3] * rho**2
            caps = ~np.isfinite(b) | (b > self.b_cap) | (s > self.s_max) | (rho > self.rho_max)
            return bound, caps
        if self.gate == "envelope":
            rho = self.offset(Z)
            a = self.env if self.env is not None else (np.inf, 0.0, 0.0, 0.0)  # not yet calibrated: every input refused
            bound = a[0] + a[1] * b + a[2] * rho + a[3] * rho**2
            caps = ~np.isfinite(b) | (b > self.b_max) | (s > self.s_max) | (rho > self.rho_max)
        else:
            e_hat = self.e_hat
            if getattr(self, "B", None) is not None:  # the composite: a calibration per region, on the bundle and beyond it
                e_hat = np.where(self.decoder.residual(Z) > self.rho_switch, self.e_hat_out, self.e_hat)
            bound = (1 + b) * e_hat
            caps = ~np.isfinite(b) | (b > self.b_max) | (s > self.s_max)
            if self.decoder is not None and self.use_residual:
                caps |= self.decoder.residual(Z) > self.decoder.r_max
        return bound, caps

    @property
    def refuses_all(self) -> bool:
        """Every input is refused whatever its bound: the bound's floor (b >= 1, rho >= 0) exceeds the budget."""
        if self.gate in ("envelope", "capped"):
            return self.env is None or bool(self.env[0] + self.env[1] > TOL / 2)
        return bool(2 * self.e_hat > TOL / 2)

    def design(self, Z):
        U = self.T.chart(Z, self.k)
        if not self.n_normal:
            return self.G.phi(U)
        return self.G.phi(U, normal_coords(self.T, Z, self.k, self.n_normal))

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
        Phi = self.design(Z)
        out = np.array(Z, dtype=float)
        with np.errstate(all="ignore"):
            g = Phi @ self.G.coef
            if getattr(self, "B", None) is not None:
                beyond = self.decoder.residual(Z) > self.rho_switch
                if beyond.any():
                    n = normal_coords(T, Z[beyond], self.k, T.free - self.k)
                    g[beyond] += np.concatenate([np.ones((len(n), 1)), n], axis=1) @ self.B
            bad = ~np.all(np.isfinite(g), axis=1) | np.any(np.abs(g) > 1e6, axis=1)
            out[:, : T.free] = np.clip(np.where(np.isfinite(g), g, 0.0), T.zlo[: T.free], T.zhi[: T.free])
            out = self.project(Z, out)
            bad |= ~np.all(np.isfinite(out), axis=1)
            bound, caps = self.gate_bound(Z, Phi)
        r = caps | (bound > TOL / 2) | bad
        self._key, self._val = (Z, np.array(Z, copy=True)), (out, r)
        return out, r

    def raw(self, Z):
        out, _ = self.evaluate(Z)
        return out, np.zeros(len(Z), dtype=bool)

    def refuses(self, Z):
        return self.evaluate(Z)[1]

    def predict(self, Z):
        return self.evaluate(Z)[0] - Z


def hyperbolic_indices(m: int, q: int, cap: int = 30) -> list[tuple]:
    """The hyperbolic cross {alpha : prod_j (alpha_j + 1) <= 2^q}, each degree at most cap: a sparse set whose size grows mildly with m,
    for maps whose interactions between coordinates are weak at high orders (mixed regularity)."""
    bound, out = 2**q, []

    def rec(prefix, prod):
        if len(prefix) == m:
            out.append(tuple(prefix))
            return
        a = 0
        while prod * (a + 1) <= bound and a <= cap:
            rec([*prefix, a], prod * (a + 1))
            a += 1

    rec([], 1)
    return out


def adaptive_indices(U, Y, budget: int, cap: int, rows: int = 1500, seed: int = 0) -> list[tuple]:
    """
    A downward-closed index set of ``budget`` terms grown greedily: from degree 1, each round fits the current set (truncated SVD, all
    rows) and adds the admissible candidates (one step beyond the set in one coordinate, every backward neighbour in the set, each degree
    at most ``cap``) whose design columns correlate most with the standardised residuals, scored on ``rows`` rows; the set grows by half
    per round. The selection's held-out chains, not this in-sample score, decide which budget is served.
    """
    m = U.shape[1]
    lo, hi = U.min(0), U.max(0)
    s = (2 * U - lo - hi) / np.maximum(hi - lo, 1e-12)
    tabs = [np.polynomial.chebyshev.chebvander(s[:, j], cap) for j in range(m)]
    sub = np.random.default_rng(seed).choice(len(U), min(rows, len(U)), replace=False)
    Ys = (Y - Y.mean(0)) / np.maximum(Y.std(0), 1e-300)

    def column(a, r=None):
        col = np.ones(len(U) if r is None else len(r))
        for j, d in enumerate(a):
            if d:
                col = col * (tabs[j][:, d] if r is None else tabs[j][r, d])
        return col

    idx = [tuple([0] * m)] + [tuple(int(i == j) for i in range(m)) for j in range(m)]
    have = set(idx)
    A = np.stack([column(a) for a in idx], axis=1)
    cand_cols: dict = {}  # candidate columns on the scoring rows, computed once
    while len(idx) < budget:
        u, sv, _ = np.linalg.svd(A, full_matrices=False)
        keep = sv > RCOND * sv[0]
        Rsub = (Ys - u[:, keep] @ (u[:, keep].T @ Ys))[sub]
        cands = []
        for a in idx:
            for j in range(m):
                b = a[:j] + (a[j] + 1,) + a[j + 1 :]
                if b[j] > cap or b in have or b in cands:
                    continue
                if all(b[i] == 0 or (b[:i] + (b[i] - 1,) + b[i + 1 :]) in have for i in range(m)):
                    cands.append(b)
        if not cands:
            break
        for b in cands:
            if b not in cand_cols:
                cand_cols[b] = column(b, sub)
        C = np.stack([cand_cols[b] for b in cands], axis=1)
        score = np.linalg.norm(C.T @ Rsub, axis=1) / np.maximum(np.linalg.norm(C, axis=0), 1e-300)
        n_add = min(budget - len(idx), max(1, len(idx) // 2), len(cands))
        new = [cands[t] for t in np.argsort(-score)[:n_add]]
        # a candidate added in this round may make others admissible only in the next round, which keeps the set downward closed
        idx += new
        have.update(new)
        for b in new:
            cand_cols.pop(b, None)
        A = np.concatenate([A, np.stack([column(b) for b in new], axis=1)], axis=1)
    return idx


def normal_coords(T, Z, k, n):
    """The first n off-chart coordinates of Z (the chart's directions k, k+1, ...), the offset across the chart's fibres."""
    return (T.clip(Z)[:, : T.free] - T.mu) @ T.V[:, k : k + n]


def mixed_fit_pairs(P, fit_trajs, m, seed=77):
    """m of the fit trajectories' states mixed once (w = 0.01 and 0.1) with another fit trajectory's at the same step, solver-labelled;
    kept per set of trajectories (the selection rebuilds each split's fit at every degree)."""
    key = (id(P), fit_trajs.shape, hash(fit_trajs.tobytes()), m, seed)
    if key not in _MIXED:
        X, Y = mixed_pairs(P, fit_trajs, np.random.default_rng(seed))
        pick = np.sort(np.random.default_rng(seed + 1).choice(len(X), min(m, len(X)), replace=False))
        _MIXED[key] = (X[pick], Y[pick])
    return _MIXED[key]


_MIXED: dict = {}


def sensitivity_directions(T, X, Y, free):
    """An orthonormal basis of the free components, ordered by what the step map responds to: the left singular vectors of the least-
    squares coefficient of the successors on the inputs (both centred, clipped), so the leading directions are those the next state
    depends on (a reduced-rank regression), not those along which the states vary."""
    Xc = T.clip(X)[:, :free]
    Yc = T.clip(Y)[:, :free]
    C = np.linalg.lstsq(Xc - Xc.mean(0), Yc - Yc.mean(0), rcond=None)[0]
    return np.linalg.svd(C, full_matrices=True)[0]


def transform(P, fit_trajs, extra_X=None, chart_pairs=None):
    from generic import Transform

    T = Transform.__new__(Transform)
    T.P, T.h = P, P.system.held
    T.free = fit_trajs.shape[2] - T.h
    T.cut = np.log10(P.floor) - 2.0 if (P.system.log and P.floor > 0) else -np.inf
    X = T.clip(visited(fit_trajs) if extra_X is None else np.concatenate([visited(fit_trajs), extra_X]))
    T.mu = X[:, : T.free].mean(axis=0)
    T.V = np.linalg.svd(X[:, : T.free] - T.mu, full_matrices=False)[2].T
    if chart_pairs is not None:
        T.V = sensitivity_directions(T, *chart_pairs, T.free)
    T.W = verified_laws(P, fit_trajs)
    T.zlo, T.zhi = X.min(0) - 2.0, X.max(0) + 0.5
    return T


def build(P, fit_trajs, k, q, dec_degree, extra=None, variant=None):
    """The surrogate fitted on ``fit_trajs`` (and ``extra`` pairs); ``variant`` (the tube experiment): ``mixed`` adds that many of the fit
    trajectories' own mixed pairs to the fit, ``normal`` adds the off-chart coordinates as linear terms, ``chart="sensitivity"`` takes
    the chart from sensitivity_directions on the plain and mixed pairs; ``basis`` (``"total"``, ``"hyperbolic"``, ``"adaptive"``) sets
    the index set at level q: total degree q, the hyperbolic cross of level q, or a greedy set of total degree q's size."""
    v = variant or {}
    X, Y = pairs(fit_trajs)
    if extra is not None:
        X, Y = np.concatenate([X, extra[0]]), np.concatenate([Y, extra[1]])
    if v.get("mixed"):
        Xm, Ym = mixed_fit_pairs(P, fit_trajs, v["mixed"])
        X, Y = np.concatenate([X, Xm]), np.concatenate([Y, Ym])
    added = X[len(pairs(fit_trajs)[0]) :]
    T = transform(P, fit_trajs, added if len(added) else None, (X, Y) if v.get("chart") == "sensitivity" else None)
    n_normal = (T.free - k) if v.get("normal") else 0
    R = normal_coords(T, X, k, n_normal) if n_normal else None
    U = T.chart(X, k)
    basis = v.get("basis", "total")
    if basis == "hyperbolic":
        idx = hyperbolic_indices(U.shape[1], q)
    elif basis == "adaptive":
        idx = adaptive_indices(U, T.clip(Y)[:, : T.free], n_terms(U.shape[1], q), cap=min(3 * q, 12))
    else:
        idx = None
    G = Fit4(U, T.clip(Y)[:, : T.free], q, R, idx)
    if v.get("sketch"):
        G.sketch()
    dec = Decoder(T, k, X, dec_degree) if k < T.free else None
    sur = Surrogate4(T, k, G, dec, n_normal)
    sur.gate, sur.account = v.get("gate", "product"), bool(v.get("account"))
    if v.get("composite") and dec is not None:
        anchor(sur, P, fit_trajs, v["composite"], v.get("radius", 3e-2))
    return sur, X, Y


def anchor(sur, P, fit_trajs, m, radius):
    """
    The composite tube (council round 3: an on-bundle fit kept as it is, and a correction that vanishes on the bundle): an affine map
    of the off-chart coordinates, B [1, n(x)], fitted by least squares to the on-bundle fit's residuals on ``m`` of the fit
    trajectories' mixed pairs whose offset rho lies between the decoder's largest training residual and ``radius``, and added only to
    inputs beyond that residual; on and inside the bundle the prediction is the plain fit's. The residual cap widens to the largest
    offset the correction was fitted on (at most ``radius``: a first-order correction is identified only near the bundle), so the gate
    accepts exactly the region where the composite was identified.
    """
    T, k = sur.T, sur.k
    Xm, Ym = mixed_fit_pairs(P, fit_trajs, m)
    rho = sur.decoder.residual(Xm)
    out = (rho > sur.decoder.r_max) & (rho <= radius)
    if out.sum() < 3 * (T.free - k + 1):
        return
    with np.errstate(all="ignore"):
        g = sur.G.phi(T.chart(Xm[out], k)) @ sur.G.coef
    A = np.concatenate([np.ones((int(out.sum()), 1)), normal_coords(T, Xm[out], k, T.free - k)], axis=1)
    D = T.clip(Ym[out])[:, : T.free] - g
    ok = np.all(np.isfinite(D), axis=1)
    sur.B = np.linalg.lstsq(A[ok], D[ok], rcond=None)[0]
    sur.rho_switch, sur.e_hat_out = sur.decoder.r_max, np.inf
    sur.decoder.r_max = float(np.max(rho[out]))


def calibration_stats(sur, P, cal_sets) -> dict:
    """Per calibration state: the one-step error of ``sur``, its bound b, out-of-span share s, offset rho, and whether the residual term
    accepts it."""
    out = {"e": [], "b": [], "s": [], "rho": [], "kept": []}
    for Xc, Yc in cal_sets:
        if not len(Xc):
            continue
        with np.errstate(all="ignore"):
            e = np.max(np.where(P.scored(Yc), np.abs(sur.evaluate(Xc, cache=False)[0] - Yc), 0.0), axis=1)
        b, s = sur.G.gate_terms(sur.design(Xc))
        for key, val in (("e", e), ("b", b), ("s", s), ("rho", sur.offset(Xc)), ("kept", kept(sur, Xc, True))):
            out[key].append(val)
    return {k: np.concatenate(v) for k, v in out.items()}


def envelope(e, b, rho) -> np.ndarray:
    """The tightest additive envelope a0 + a1 b + a2 rho + a3 rho^2 >= e over the given states (a >= 0, minimising its sum): a linear
    program. Each term is Proposition 5.1's or the map's across the bundle (council3-fable.md, section 3.1)."""
    from scipy.optimize import linprog

    F = np.stack([np.ones_like(b), b, rho, rho**2], axis=1)
    ok = np.all(np.isfinite(F), axis=1) & np.isfinite(e)
    F, e = F[ok], e[ok]
    scale = np.maximum(np.abs(F).max(0), 1e-300)
    Fs = F / scale
    res = linprog(Fs.sum(0), A_ub=-Fs, b_ub=-e, bounds=[(0, None)] * 4, method="highs")
    if not res.success:
        return np.array([np.inf, 0.0, 0.0, 0.0])
    return res.x / scale


BETA = 3.0  # the cap b <= BETA sqrt(rank): the mean of b^2 over the fit is the rank, so at most 1/BETA^2 of the fitted states exceed it


def typical_charge(e, b, rho) -> np.ndarray:
    """The account's charge c = m0 + m1 b + m2 rho + m3 rho^2 >= 0, a non-negative least-squares model of the typical defect (not its
    envelope; the account's multiplier lambda, calibrated on whole chains, carries the margin)."""
    from scipy.optimize import nnls

    F = np.stack([np.ones_like(b), b, rho, rho**2], axis=1)
    ok = np.all(np.isfinite(F), axis=1) & np.isfinite(e)
    scale = np.maximum(np.abs(F[ok]).max(0), 1e-300) if ok.any() else np.ones(4)
    return nnls(F[ok] / scale, e[ok])[0] / scale if ok.any() else np.zeros(4)


def apply_calibration(sur, st: dict):
    """Set the gate from calibration statistics (module docstring): the product's e_hat over the states the residual term accepts, or
    the envelope over every calibration state; the caps from their ranges."""
    def top(x):
        x = x[np.isfinite(x)]
        return float(np.max(x)) if len(x) else -np.inf

    if sur.gate == "capped":
        sur.b_cap = BETA * np.sqrt(sur.G.rank)
        inside = np.isfinite(st["b"]) & (st["b"] <= sur.b_cap) & np.isfinite(st["e"])
        sur.env = envelope(st["e"][inside], st["b"][inside], st["rho"][inside])
        sur.charge = typical_charge(st["e"][inside], st["b"][inside], st["rho"][inside])
        sur.b_max, sur.s_max, sur.rho_max = sur.b_cap, top(st["s"][inside]), top(st["rho"][inside])
        sur.inside_share = float(np.mean(inside)) if len(inside) else 0.0
    elif sur.gate == "envelope":
        sur.env = envelope(st["e"], st["b"], st["rho"])
        sur.b_max, sur.s_max, sur.rho_max = top(st["b"]), top(st["s"]), top(st["rho"])
    else:
        k = st["kept"]
        sc = st["e"] / (1 + st["b"])
        if getattr(sur, "B", None) is not None:
            beyond = st["rho"] > sur.rho_switch
            sur.e_hat, sur.e_hat_out = top(sc[k & ~beyond]), top(sc[k & beyond])
        else:
            sur.e_hat = top(sc[k])
        sur.e_hat = sur.e_hat if np.isfinite(sur.e_hat) else np.inf  # no calibration state in a region: refuse the region
        if getattr(sur, "B", None) is not None and not np.isfinite(sur.e_hat_out):
            sur.e_hat_out = np.inf
        sur.b_max, sur.s_max = top(st["b"][k]), top(st["s"][k])
    sur._key = None


def calibrate(sur, P, cal_sets):
    apply_calibration(sur, calibration_stats(sur, P, cal_sets))


class Account:
    """
    The deployment policy with an error account per cell (council3-fable.md, section 3.3; Proposition 2.2, Corollary 5.3 (d)): each cell
    (a row of the batches, in a fixed order across steps) carries S, the sum of the gate's bounds over its accepted steps; a step is
    refused when the gate refuses it or when it would take S past the budget, and a refused step adds nothing. For neutral dynamics the
    chain's error is at most S wherever the bounds hold, whatever the coupling script. One instance per chain of batches: the rows must
    be the same cells at every call.
    """

    def __init__(self, sur, budget=TOL / 2):
        self.sur, self.budget, self.S, self.evals = sur, budget, None, 0

    def refuses(self, Z):
        _, r = self.sur.evaluate(Z)
        if self.S is None:
            self.S = np.zeros(len(Z))
        if self.sur.gate == "capped":  # charge the typical defect, scaled by lambda (council4-fable.md, section 2.1)
            b, _ = self.sur.G.gate_terms(self.sur.design(Z))
            rho = self.sur.offset(Z)
            m = self.sur.charge
            c = m[0] + m[1] * b + m[2] * rho + m[3] * rho**2
            r = np.asarray(r, dtype=bool) | (self.sur.lam * (self.S + c) > self.budget)
            self.S = self.S + np.where(r, 0.0, c)
            return r
        bound, _ = self.sur.gate_bound(Z)
        r = np.asarray(r, dtype=bool) | (self.S + bound > self.budget)
        self.S = self.S + np.where(r, 0.0, bound)
        return r

    def predict(self, Z):
        return self.sur.predict(Z)

    def evaluate(self, Z, cache=True):
        nxt, _ = self.sur.evaluate(Z, cache)
        return nxt, self.refuses(Z)


def policy(sur):
    """The surrogate as deployed: with its error account when the variant asks for one (a fresh account per chain)."""
    return Account(sur) if getattr(sur, "account", False) else sur


def account_check(P, sur, true) -> dict:
    """A plain rollout under the account: the share of (cell, step) where the chain's error exceeds the cell's account, |e_n| > S_n."""
    acc = Account(sur)
    Z = true[:, 0].copy()
    h = P.system.held
    over, total = 0, 0
    for k in range(true.shape[1] - 1):
        if h:
            Z[:, -h:] = true[:, k, -h:]
        r = acc.refuses(Z)
        nxt = Z + acc.predict(Z)
        if r.any():
            nxt[r] = P.step(Z[r])
        Z = nxt
        e = np.max(np.where(P.scored(true[:, k + 1]), np.abs(Z - true[:, k + 1]), 0.0), axis=1)
        over += int(np.sum(e > acc.S + 1e-15))
        total += len(e)
    return {"exceeded_percent": 100 * over / max(total, 1), "account_max": float(np.max(acc.S))}


def chains(P, sur, true) -> dict:
    """The selection's held-out chains with fallback, stopped as soon as they cannot be served (module docstring)."""
    Z = true[:, 0].copy()
    h = P.system.held
    worst, solved = 0.0, []
    pol = policy(sur)
    for k in range(true.shape[1] - 1):
        if h:
            Z[:, -h:] = true[:, k, -h:]
        nxt, r = pol.evaluate(Z)
        nxt = np.array(nxt)
        if r.any():
            nxt[r] = P.step(Z[r])
        Z = nxt
        worst = max(worst, worst_scored(P, Z, true[:, k + 1]))
        solved.append(float(r.mean()))
        if worst > MARGIN * TOL or (k >= 9 and np.mean(solved) > SERVED_SHARE):
            return {"rollout_worst": float(worst), "rollout_solved_by_step": solved, "stopped_early": True}
    return {"rollout_worst": float(worst), "rollout_solved_by_step": solved, "stopped_early": False}


def select(P, train, k, dec_degree, solve_us, extra=None, qmax=12, variant=None) -> dict:
    m = k + P.system.held
    n_pairs = len(train) * 4 // 5 * (train.shape[1] - 1) + (0 if extra is None else len(extra[0]))
    cal_cache = {s: calibration_sets(P, thirds(train, s)[1], True, np.random.default_rng(1000 + s)) for s in SPLITS}
    rows, best_err, rising = [], np.inf, 0
    basis = (variant or {}).get("basis", "total")
    for q in range(1, qmax + 1):
        terms = len(hyperbolic_indices(m, q)) if basis == "hyperbolic" else n_terms(m, q)
        if terms * 1.5 > n_pairs or terms > 3000:
            break
        per = []
        for s in SPLITS:
            fit, cal, sel = thirds(train, s)
            sur, _, _ = build(P, fit, k, q, dec_degree, extra, variant)
            calibrate(sur, P, cal_cache[s])
            share = float(np.mean(sur.refuses(pairs(sel)[0])))
            if sur.refuses_all:
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
        row = {"degree": q, "terms": terms, "chain_worst": float("inf") if any(np.isnan(chains_)) else max(chains_), "solver_share": rho,
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
    if ok:  # the cheapest configuration; costs within 10% of the cheapest are a tie (timing noise), broken by the smaller chain error
        c0 = min(r["cost_us"] for r in ok)
        best = min((r for r in ok if r["cost_us"] <= 1.1 * c0), key=lambda r: r["chain_worst"])
    else:
        best = min(rows, key=lambda r: r["heldout_worst"])
    return {"configs": rows, "degree": best["degree"], "served": bool(ok), "cal_cache": cal_cache}


def final(P, train, k, degree, dec_degree, cal_cache, extra=None, variant=None):
    """Fitted on all of ``train``; its gate from each split's fit scored on that split's calibration states, pooled."""
    sur, _, _ = build(P, train, k, degree, dec_degree, extra, variant)
    stats = [calibration_stats(build(P, thirds(train, s)[0], k, degree, dec_degree, extra, variant)[0], P, cal_cache[s]) for s in SPLITS]
    apply_calibration(sur, {key: np.concatenate([st[key] for st in stats]) for key in stats[0]})
    return sur


def rebuild(P, train, rec: dict, extra=None, variant=None):
    cal_cache = {s: calibration_sets(P, thirds(train, s)[1], True, np.random.default_rng(1000 + s)) for s in SPLITS}
    return final(P, train, rec["k"], rec["degree"], rec["decoder_degree"], cal_cache, extra, variant)


def budget_shares(sur, Xe, Ye, P, A_n, steps) -> dict:
    """Refused shares of the test inputs under the three per-step budgets (module docstring), and the accepted errors' sign coherence."""
    nxt, r = sur.evaluate(Xe)
    bound, caps = sur.gate_bound(Xe)
    acc = ~r
    if acc.any():
        e = np.where(P.scored(Ye[acc]), nxt[acc] - Ye[acc], 0.0)
        c = float(np.max(np.abs(e.mean(0)) / np.maximum(np.abs(e).mean(0), 1e-300)))
    else:
        c = float("nan")
    out = {"coherence_value": c, "accumulation_factor": A_n, "steps": steps}
    for label, B in (("half_tol", TOL / 2), ("accumulation", TOL / (2 * A_n)), ("coherence", TOL / (2 * (c * steps + np.sqrt(steps))) if np.isfinite(c) else np.nan)):
        out[label] = {"budget": float(B), "refused_percent": float(100 * np.mean(r | caps | (bound > B)))}
    return out


LAMBDAS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
ALPHA = 0.05  # the share of calibration chains allowed to exceed tol / 2 anywhere (conformal risk control at the chain level)


def chain_maxima(P, pol_factory, true, w, rng) -> np.ndarray:
    """Per chain, the worst scored error of the chains with fallback, plain (w = 0) or mixed once per step with a partner's state of the
    reference chain (generic.mixing's script), under a fresh deployment policy."""
    h = P.system.held
    free = true.shape[2] - h
    n = len(true)
    pol = pol_factory()
    lin = (lambda Z: 10.0**Z) if P.system.log else (lambda Z: Z)
    back = (lambda Y: np.log10(np.maximum(Y, 1e-300))) if P.system.log else (lambda Y: Y)
    partners = np.array([[rng.choice([j for j in range(n) if j != i]) for i in range(n)] for _ in range(true.shape[1] - 1)]) if w else None

    def mix(Z, k):
        out = Z.copy()
        if w:
            out[:, :free] = back((1 - w) * lin(Z[:, :free]) + w * lin(true[partners[k], k, :free]))
        if h:
            out[:, free:] = true[:, k, free:]
        return out

    Zs, Zt = true[:, 0].copy(), true[:, 0].copy()
    worst = np.zeros(n)
    for k in range(true.shape[1] - 1):
        Zt = P.step(mix(Zt, k)) if w else true[:, k + 1]
        Zin = mix(Zs, k)
        r = np.asarray(pol.refuses(Zin), dtype=bool)
        Zs = Zin + pol.predict(Zin)
        if r.any():
            Zs[r] = P.step(Zin[r])
        e = np.max(np.where(P.scored(Zt), np.abs(Zs - Zt), 0.0), axis=1)
        worst = np.maximum(worst, np.where(np.isfinite(e), e, np.inf))
    return worst


def above_envelope(P, sur, test) -> float:
    """The share of test trajectories with an accepted one-step defect above the gate's envelope E (the scenario statement's event)."""
    X, Y = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    nxt, r = sur.evaluate(X, cache=False)
    E, _ = sur.gate_bound(X)
    e = np.max(np.where(P.scored(Y), np.abs(nxt - Y), 0.0), axis=1)
    bad = (~np.asarray(r, dtype=bool)) & (e > E)
    return float(100 * np.mean(bad.reshape(len(test), -1).any(axis=1)))


def calibrate_fresh(P, sur, cal_trajs, seed=7) -> dict:
    """The deployed fit's gate calibrated on fresh trajectories (plain and mixed states), then the account's lambda chosen on whole
    chains of them: the smallest lambda in LAMBDAS for which at most ALPHA of the chains (plain, w = 0.01, w = 0.1) exceed tol / 2."""
    sets = [pairs(cal_trajs), mixed_pairs(P, cal_trajs, np.random.default_rng(seed))]
    apply_calibration(sur, calibration_stats(sur, P, sets))
    table = {}
    for lam in LAMBDAS:
        sur.lam = lam
        maxima = np.concatenate([chain_maxima(P, lambda: Account(sur), cal_trajs, w, np.random.default_rng(seed + 1)) for w in (0.0, 0.01, 0.1)])
        table[str(lam)] = float(np.mean(maxima > TOL / 2))
        if table[str(lam)] <= ALPHA:
            break
    ok = [float(k) for k, v in table.items() if v <= ALPHA]
    sur.lam = min(ok) if ok else np.inf
    return {"lambda": sur.lam, "exceed_share_by_lambda": table, "inside_cap_share": getattr(sur, "inside_share", None),
            "envelope": None if sur.env is None else sur.env.tolist(), "charge": None if sur.charge is None else sur.charge.tolist(), "b_cap": sur.b_cap}  # fmt: skip


def run(P, train, test, name, seed=0, extra=None, solve_us=None, qmax=12, kmax=40, variant=None, k=None, cal_trajs=None) -> dict:
    from generic import mixing

    from src.benchmark.step_maps import rollout

    t0 = time.perf_counter()
    free = train.shape[2] - P.system.held
    kc = choose_k(P, train, min(kmax, free), extra) if k is None else {"k": k[0], "reduction": "given", "curve": [{"k": k[0], "decoder_degree": k[1]}]}
    dec_degree = next((c["decoder_degree"] for c in kc["curve"] if c["k"] == kc["k"]), 1)
    log(f"  k={kc['k']} ({kc['reduction']}), decoder degree {dec_degree}")
    sel = select(P, train, kc["k"], dec_degree, solve_us, extra, qmax, variant)
    sur = final(P, train, kc["k"], sel["degree"], dec_degree, sel["cal_cache"], extra, variant)
    fresh = calibrate_fresh(P, sur, cal_trajs) if cal_trajs is not None and sur.gate == "capped" else None
    train_s = time.perf_counter() - t0
    Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    nxt, r = sur.evaluate(Xe)
    nxt = np.array(nxt)
    nxt[r] = Ye[r]
    everything_refused = sur.refuses_all or not np.isfinite(sur.lam)
    if everything_refused:
        r40 = r20 = {"rollout_worst": 0.0, "rollout_solved_by_step": [1.0] * (test.shape[1] - 1)}
        mix_chain = {str(w): {"worst": 0.0, "solver_share": 1.0} for w in (0.01, 0.1)}
    else:
        r40, r20 = rollout(P, policy(sur), test), rollout(P, policy(sur), test[:20])
        rng = np.random.default_rng(seed + 1)
        mix_chain = {str(w): mixing(P, policy(sur), test, w, rng) for w in (0.01, 0.1)}
    n = min(len(Xe), 2000)
    evaluate_us = timed(lambda X: sur.evaluate(X, cache=False), Xe[:n])
    rho = float(r.mean())
    by_step = np.asarray(r40["rollout_solved_by_step"])
    acc = dynamics(name)["accumulation"]
    out = {"d": int(test.shape[2]), "held": int(P.system.held), "n_train": int(len(train)), "k": kc["k"], "reduction": kc["reduction"],
           "k_curve": kc["curve"], "decoder_degree": dec_degree, "selection": sel["configs"], "degree": sel["degree"], "served": sel["served"],
           "e_hat": sur.e_hat, "b_max": sur.b_max, "s_max": sur.s_max, "gate": sur.gate, "envelope": None if sur.env is None else sur.env.tolist(), "rho_max": sur.rho_max, "rank": sur.G.rank, "terms": len(sur.G.idx), "laws": int(len(sur.T.W)),
           "residual_max": None if sur.decoder is None else sur.decoder.r_max, "train_s": train_s, "one_step_worst": worst_scored(P, nxt, Ye),
           "refused_percent": 100 * rho, "refused_by_step": r.reshape(test.shape[0], -1).mean(axis=0).tolist(),
           "rollout_worst": float(r40["rollout_worst"]), "rollout_solver_share": float(by_step.mean()), "rollout_solver_share_from_step_10": float(by_step[10:].mean()) if len(by_step) > 10 else None,
           "rollout_solved_by_step": r40["rollout_solved_by_step"], "rollout_worst_20": float(r20["rollout_worst"]), "mixing_chained": mix_chain,
           "evaluate_us": evaluate_us, "timing_batch": n, "solve_us": solve_us,
           "budgets": budget_shares(sur, Xe, Ye, P, float(np.max(acc)), test.shape[1] - 1),
           "account_check": account_check(P, sur, test) if sur.account and not everything_refused else None, "fresh_calibration": fresh,
           "trajectories_above_envelope_percent": above_envelope(P, sur, test) if sur.gate == "capped" else None}  # fmt: skip
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
