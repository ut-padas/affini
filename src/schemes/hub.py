"""
The hub expansion: per group of outputs, a dense Chebyshev model in the few input directions those outputs are
steep in (the hub), and degree-1 and degree-2 terms in every other direction, whose coefficients are themselves
polynomials of the hub. In rotated coordinates (u, v) = V^T x,

    f(x) ~ P_0(u) + sum_j [P_1j(u) T_1(v_j) + P_2j(u) T_2(v_j)],

a truncated ANOVA expansion around a dense low-dimensional core, fitted by linear least squares. It suits outputs
with a few oblique steep directions and weak dependence on the rest: it is linear in d at a fixed hub dimension,
where a tensor train pays rank squared for every oblique coupling. Outside that class (isotropic, radial, many
differently oriented features, jumps) ChebTT or ChebTT-A is the tool.

Everything is chosen from the data, nothing by hand:

- **Directions:** each output's gradient covariance at pilot points. Gradients come from ``jac`` when the oracle
  supplies them (an adjoint, autodiff, or ODE sensitivities); otherwise from central differences, 2d evaluations
  per pilot point.
- **Hub dimension r:** the fewest eigenvectors holding 99% of the covariance's trace (at most ``r_max``). The weak
  terms are kept only when the remaining share exceeds 1e-4. Outputs whose hubs agree with the pooled one share a
  rotation, and with it one factorisation.
- **Samples:** half chosen by pivoted QR of the hub basis on a pool of uniform, vertex-directed and blended points
  (the extremes of the hub coordinates, which a uniform sample of a box rarely reaches), half uniform.
- **Hold-out and degree:** the hold-out is drawn from the same design. The hub degree is chosen on a sub-sample by
  the hold-out's worst error, then refitted on all samples.
- **Budget:** doubled until the hold-out's worst error is within half the tolerance, stops improving by 10% per
  doubling, or ``max_evals`` is reached. ``holdout_worst`` above ``atol`` reports a model that missed the target.

With ``jac``, every sample's gradient also enters the least squares (``grad_weight`` per unit of the box's
half-width). That cuts the samples several-fold where the model can match the gradient, and helps less where the
difficulty is coverage.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import qr

MAX_TERMS = 2500  # per group: the Gram and its solve grow as terms^2 and terms^3, and every degree tried pays them
DEGREES = {1: (8, 16, 32, 48), 2: (8, 16, 24, 32, 40), 3: (4, 6, 8, 10), 4: (3, 4, 5, 6)}  # candidates by hub dimension


@dataclass
class HubGroup:
    outputs: list
    V: np.ndarray  # (d, d) rotation, steep directions first
    r: int
    degree: int
    weak: bool
    cols: np.ndarray  # (terms, d) Chebyshev degree of every term in each rotated coordinate
    coef: np.ndarray  # (terms, len(outputs))


@dataclass
class Hub:
    lo: np.ndarray
    hi: np.ndarray
    groups: list
    m: int
    evals: int
    gradients: int = 0
    holdout_worst: float = float("nan")
    history: list = field(default_factory=list)  # (samples, hold-out worst) per budget round

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.empty((len(X), self.m))
        for g in self.groups:
            out[:, g.outputs] = _design(X, self.lo, self.hi, g.V, g.r, g.cols)[0] @ g.coef
        return out if self.m > 1 else out[:, 0]

    def describe(self) -> list:
        """Per group: its outputs, hub dimension, degree, weak terms and size (the discovered structure)."""
        return [(g.outputs, g.r, g.degree, g.weak, len(g.cols)) for g in self.groups]


def _cheb(t, n):
    """(N, n + 1) Chebyshev values and derivatives at t."""
    T, D = np.zeros((len(t), n + 1)), np.zeros((len(t), n + 1))
    T[:, 0] = 1.0
    if n:
        T[:, 1], D[:, 1] = t, 1.0
    for j in range(2, n + 1):
        T[:, j] = 2 * t * T[:, j - 1] - T[:, j - 2]
        D[:, j] = 2 * T[:, j - 1] + 2 * t * D[:, j - 1] - D[:, j - 2]
    return T, D


def _columns(d, r, degree, weak):
    hub = [a for a in itertools.product(range(degree + 1), repeat=r) if sum(a) <= degree]
    cols = [tuple(a) + (0,) * (d - r) for a in hub]
    if weak:
        for j in range(r, d):
            for k in (1, 2):
                for a in hub:
                    c = list(a) + [0] * (d - r)
                    c[j] = k
                    cols.append(tuple(c))
    return np.array(cols)


def _design(X, lo, hi, V, r, cols, gradient=False):
    """
    (design (N, terms), and with ``gradient`` its derivatives (N, terms, d) in box-scaled inputs) at X, for terms laid
    out by ``_columns``: the hub block, then for every weak coordinate j and degree k in (1, 2) the hub block times
    T_k(v_j). Built block by block, at N * terms cost rather than N * terms * d.
    """
    c0, s0 = 0.5 * (lo + hi), 0.5 * (hi - lo)
    bound = np.abs(V).sum(axis=0)  # the rotated box's half-range per rotated coordinate
    t = np.clip((((X - c0) / s0) @ V) / bound, -1.0, 1.0)
    N, d = X.shape
    hc = cols[~cols[:, r:].any(axis=1), :r]
    tables = [_cheb(t[:, i], int(hc[:, i].max())) for i in range(r)]
    vals = [T[:, hc[:, i]] for i, (T, _) in enumerate(tables)]
    Ah = np.prod(vals, axis=0)
    weak = len(cols) > len(hc)
    if weak:
        tw = t[:, r:]
        W = np.stack([tw, 2 * tw * tw - 1], axis=2)  # (N, d - r, 2): T_1, T_2 of every weak coordinate
        A = np.concatenate([Ah, (W[:, :, :, None] * Ah[:, None, None, :]).reshape(N, -1)], axis=1)
    else:
        A = Ah
    if not gradient:
        return A, None
    Wt = V / bound  # d t_i / d (scaled x)_k = Wt[k, i]
    dAh_u = [np.prod([tables[i][1][:, hc[:, i]] if i == q else vals[i] for i in range(r)], axis=0) for q in range(r)]
    dAh = np.stack(dAh_u, axis=2) @ Wt[:, :r].T  # (N, hub terms, d)
    if not weak:
        return A, dAh
    dW = np.stack([np.ones_like(tw), 4 * tw], axis=2)
    dWeak = W[:, :, :, None, None] * dAh[:, None, None, :, :]
    dWeak += (dW[:, :, :, None] * Ah[:, None, None, :])[..., None] * Wt[:, r:].T[None, :, None, None, :]
    return A, np.concatenate([dAh, dWeak.reshape(N, -1, d)], axis=1)


def _gradients(f, jac, X, lo, hi):
    """(N, d, m) gradients in box-scaled inputs (df/dx_k times the half-width), and the evaluations they cost."""
    s0 = 0.5 * (hi - lo)
    if jac is not None:
        return np.asarray(jac(X), dtype=np.float64).reshape(len(X), X.shape[1], -1) * s0[None, :, None], 0
    d = X.shape[1]
    h = 1e-4 * (hi - lo)
    E = np.repeat(X, 2 * d, axis=0)
    for i in range(d):
        E[2 * i :: 2 * d, i] += h[i]
        E[2 * i + 1 :: 2 * d, i] -= h[i]
    F = np.asarray(f(E), dtype=np.float64).reshape(len(E), -1)
    G = (F[0::2] - F[1::2]) / (2 * h[np.tile(np.arange(d), len(X))][:, None])
    return G.reshape(len(X), d, -1) * s0[None, :, None], len(E)


def _hub_rank(C, r_max):
    w = np.clip(np.linalg.eigvalsh(C)[::-1], 0.0, None)
    share = np.cumsum(w) / max(float(w.sum()), 1e-300)
    r = min(int(np.searchsorted(share, 0.99) + 1), r_max)
    return r, 1.0 - share[r - 1]


def _groups(Gp, r_max):
    """Output groups sharing a rotation: (outputs, V, r, weak) from the pilot gradients (points, d, m)."""
    m = Gp.shape[2]
    Cs = [Gp[:, :, o].T @ Gp[:, :, o] / len(Gp) for o in range(m)]
    Cpool = sum(C / max(float(np.trace(C)), 1e-300) for C in Cs)
    r_pool, _ = _hub_rank(Cpool, r_max)
    Vpool = np.linalg.eigh(Cpool)[1][:, ::-1]
    pooled, groups = [], []
    for o, C in enumerate(Cs):
        r, rest = _hub_rank(C, r_max)
        V = np.linalg.eigh(C)[1][:, ::-1]
        cos = np.linalg.svd(V[:, :r].T @ Vpool[:, :r_pool], compute_uv=False)
        if r <= r_pool and float(cos.min()) > 0.99:
            pooled.append(o)
        else:
            groups.append(([o], V, r, rest > 1e-4))
    if pooled:
        rest = max(1.0 - np.sum(np.clip(np.linalg.eigvalsh(Vpool.T @ Cs[o] @ Vpool)[::-1][:r_pool], 0, None)) / max(float(np.trace(Cs[o])), 1e-300)
                   for o in pooled)  # fmt: skip
        groups.insert(0, (pooled, Vpool, r_pool, rest > 1e-4))
    return groups, Vpool, r_pool


def _design_points(groups, lo, hi, n, rng):
    """n sample points: half by pivoted QR of the hub basis on a candidate pool (uniform, vertex-directed along each hub, blends), half uniform."""
    d = len(lo)
    c0, s0 = 0.5 * (lo + hi), 0.5 * (hi - lo)
    per = max(2000, 4 * n) // (2 * len(groups))
    pool = [rng.uniform(lo, hi, (per * len(groups), d))]
    for _, V, r, _ in groups:
        P = c0 + s0 * np.sign(rng.normal(size=(per, r)) @ V[:, :r].T)
        mix = rng.random((per, d)) < rng.random((per, 1)) * 0.5
        P[mix] = rng.uniform(lo, hi, (per, d))[mix]
        pool.append(P)
    pool = np.concatenate(pool)
    _, V0, r0, _ = groups[0]
    Phi = _design(pool, lo, hi, V0, r0, _columns(d, r0, max(DEGREES[r0]), False))[0]
    # One pivoted QR picks a well-conditioned set for the hub basis; repeating it until n // 2 points are chosen
    # costs pool * terms^2 per pass and dominated training at tens of thousands of samples, so the rest of the
    # half is drawn at random from the vertex-directed part of the pool.
    k = min(Phi.shape[1], n // 2)
    piv = qr(Phi.T, pivoting=True, mode="economic")[2][:k]
    rest = np.setdiff1d(np.arange(per * len(groups), len(pool)), piv)
    extra = rng.choice(rest, size=min(len(rest), n // 2 - k), replace=False)
    picked = pool[np.concatenate([piv, extra])]
    return np.concatenate([picked, rng.uniform(lo, hi, (n - len(picked), d))])


def _fit(X, Y, G, lo, hi, V, r, cols, grad_weight, lam=1e-10):
    """Ridge least squares of a group's outputs (one factorisation for all of them), with gradient rows when G is given."""
    A = _design(X, lo, hi, V, r, cols)[0]
    rows, rhs = A, Y
    if G is not None:
        # Gradient rows of the first k samples only, at most 10 per term and 20,000 in all: they matter while values
        # are scarce against the terms, and beyond that their d rows per sample cost d times a value row for little gain.
        k = min(len(X), max(1, min(10 * len(cols), 20_000) // X.shape[1]))
        dA = _design(X[:k], lo, hi, V, r, cols, gradient=True)[1]
        rows = np.concatenate([A, grad_weight * dA.transpose(0, 2, 1).reshape(-1, A.shape[1])])
        rhs = np.concatenate([Y, grad_weight * G[:k].reshape(-1, Y.shape[1])])
    AtA = rows.T @ rows
    AtA[np.diag_indices_from(AtA)] += lam * (np.trace(AtA) / len(AtA) + 1e-300)
    return np.linalg.solve(AtA, rows.T @ rhs)


def build(f, lo, hi, atol, *, jac=None, n0=2000, max_evals=200_000, r_max=4, pilot=100, grad_weight=0.05, seed=0) -> Hub:
    """
    Train a hub expansion of ``f`` ((N, d) -> (N,) or (N, m)) on [lo, hi] to an absolute tolerance ``atol``
    (judged on a hold-out from the same design: no max-norm guarantee). ``jac`` ((N, d) -> (N, d, m) or (N, d)), if
    given, supplies the gradients for the directions and the fit; otherwise central differences give the
    directions (2d evaluations per pilot point) and the fit uses values only.
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = len(lo)
    rng = np.random.default_rng(seed)
    ev = lambda X: np.asarray(f(X), dtype=np.float64).reshape(len(X), -1)  # noqa: E731
    Xp = rng.uniform(lo, hi, (pilot, d))
    Gp, evals = _gradients(f, jac, Xp, lo, hi)
    n_grad = pilot if jac is not None else 0
    groups, _, _ = _groups(Gp, r_max)
    m = Gp.shape[2]
    X, Y = np.zeros((0, d)), np.zeros((0, m))
    G = np.zeros((0, d, m)) if jac is not None else None
    n, history, hub = n0, [], None
    while True:
        Xn = _design_points(groups, lo, hi, n - len(X), rng) if n > len(X) else np.zeros((0, d))
        X, Y = np.concatenate([X, Xn]), np.concatenate([Y, ev(Xn)])
        evals += len(Xn)
        if jac is not None:
            G = np.concatenate([G, _gradients(f, jac, Xn, lo, hi)[0]])
            n_grad += len(Xn)
        hold = np.random.default_rng(seed + 17).random(len(X)) < 0.2
        tr, ho = ~hold, hold
        fitted, worst = [], 0.0
        for outputs, V, r, weak in groups:
            Yg = Y[:, outputs]
            Gg = None if G is None else G[:, :, outputs]
            sub = np.flatnonzero(tr)[: min(int(tr.sum()), 6000)]
            best = None
            for degree in DEGREES[r]:
                cols = _columns(d, r, degree, weak)
                if len(cols) > min(MAX_TERMS, 0.5 * len(sub) * (1 if Gg is None else 1 + d)):
                    break
                coef = _fit(X[sub], Yg[sub], None if Gg is None else Gg[sub], lo, hi, V, r, cols, grad_weight)
                err = float(np.max(np.abs(_design(X[ho], lo, hi, V, r, cols)[0] @ coef - Yg[ho]))) if ho.any() else 0.0
                if best is None or err < best[0]:
                    best = (err, degree)
            degree = best[1] if best else DEGREES[r][0]
            cols = _columns(d, r, degree, weak)
            coef = _fit(X[tr], Yg[tr], None if Gg is None else Gg[tr], lo, hi, V, r, cols, grad_weight)
            err = float(np.max(np.abs(_design(X[ho], lo, hi, V, r, cols)[0] @ coef - Yg[ho]))) if ho.any() else 0.0
            worst = max(worst, err)
            fitted.append(HubGroup(list(outputs), V, r, degree, weak, cols, coef))
        history.append((len(X), worst))
        hub = Hub(lo, hi, fitted, m, evals, n_grad, worst, history)
        # Stop at the target, at the budget, or when doubling the samples bought less than 10%: the remaining error
        # is then the model class's (hub dimension, degree cap), which more samples of the same design do not fix.
        stalled = len(history) > 1 and worst > 0.9 * history[-2][1]
        if worst <= 0.5 * atol or stalled or 2 * n + evals - len(X) > max_evals:
            break
        n *= 2
    # the final model uses every sample (the hold-out included), at the degrees chosen above
    for g in hub.groups:
        g.coef = _fit(X, Y[:, g.outputs], None if G is None else G[:, :, g.outputs], lo, hi, g.V, g.r, g.cols, grad_weight)
    return hub
