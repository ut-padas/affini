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

- **Directions:** each output's gradient covariance at pilot points. Gradients come from the Jacobian when the oracle
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

With a Jacobian, every sample's gradient also enters the least squares (``grad_weight`` per unit of the box's
half-width). That cuts the samples several-fold where the model can match the gradient, and helps less where the
difficulty is coverage.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import qr

from src.schemes.base import Settings
from src.schemes.chebyshev import cheb_tables
from src.schemes.directions import box_gradients, output_groups
from src.schemes.oracle import as_oracle
from src.schemes.profiling import Training

MAX_TERMS = 2500  # per group: the Gram and its solve grow as terms^2 and terms^3, and every degree tried pays them
DEGREES = {1: (8, 16, 32, 48), 2: (8, 16, 24, 32, 40), 3: (4, 6, 8, 10), 4: (3, 4, 5, 6)}  # candidates by hub dimension


@dataclass(frozen=True)
class HubSettings(Settings):
    n0: int = 2000  # samples of the first budget round (doubled every round)
    max_evals: int = 200_000
    r_max: int = 4  # largest hub dimension
    pilot: int = 100  # pilot points whose gradients give the directions
    grad_weight: float = 0.05  # weight of a gradient equation per unit of the box's half-width
    seed: int = 0


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
    jacobians: int = 0  # points at which the Jacobian entered the fit
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
    tables = [cheb_tables(t[:, i], int(hc[:, i].max())) for i in range(r)]
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


def fit_groups(groups, X, Y, G, lo, hi, s: HubSettings) -> tuple[list, float]:
    """
    Fit every group on the training split at the degree whose fit on a sub-sample has the smallest worst
    hold-out error; returns the fitted groups and the worst hold-out error over them.
    """
    d = X.shape[1]
    hold = np.random.default_rng(s.seed + 17).random(len(X)) < 0.2
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
            coef = _fit(X[sub], Yg[sub], None if Gg is None else Gg[sub], lo, hi, V, r, cols, s.grad_weight)
            err = float(np.max(np.abs(_design(X[ho], lo, hi, V, r, cols)[0] @ coef - Yg[ho]))) if ho.any() else 0.0
            if best is None or err < best[0]:
                best = (err, degree)
        degree = best[1] if best else DEGREES[r][0]
        cols = _columns(d, r, degree, weak)
        coef = _fit(X[tr], Yg[tr], None if Gg is None else Gg[tr], lo, hi, V, r, cols, s.grad_weight)
        err = float(np.max(np.abs(_design(X[ho], lo, hi, V, r, cols)[0] @ coef - Yg[ho]))) if ho.any() else 0.0
        worst = max(worst, err)
        fitted.append(HubGroup(list(outputs), V, r, degree, weak, cols, coef))
    return fitted, worst


def build(f, lo, hi, atol, settings: HubSettings | None = None, *, jacobian=None, stream_dir=None, **overrides) -> Hub:
    """
    Train a hub expansion of ``f`` ((N, d) -> (N,) or (N, m)) on [lo, hi] to an absolute tolerance ``atol``
    (judged on a hold-out from the same design: no max-norm guarantee). ``jacobian`` ((N, d) -> (N, m, d) or
    (N, d)), or an ``Oracle`` that has one, supplies the gradients for the directions and the fit; otherwise
    central differences give the directions (2d evaluations per pilot point) and the fit uses values only.
    """
    s = HubSettings.resolve(settings, overrides)
    oracle = as_oracle(f, jacobian)
    run = Training("hub", s, oracle, stream_dir)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = len(lo)
    rng = np.random.default_rng(s.seed)
    with_jacobian = oracle.has_jacobian
    with run.phase("explore"):
        Xp = rng.uniform(lo, hi, (s.pilot, d))
        Gp = box_gradients(oracle, Xp, lo, hi)
        groups, _, _ = output_groups(Gp, s.r_max)
    m = Gp.shape[2]
    X, Y = np.zeros((0, d)), np.zeros((0, m))
    G = np.zeros((0, d, m)) if with_jacobian else None
    n, history, hub = s.n0, [], None
    while True:
        with run.phase("sample"):
            Xn = _design_points(groups, lo, hi, n - len(X), rng) if n > len(X) else np.zeros((0, d))
        X, Y = np.concatenate([X, Xn]), np.concatenate([Y, oracle(Xn).reshape(len(Xn), -1)])
        if with_jacobian:
            G = np.concatenate([G, box_gradients(oracle, Xn, lo, hi)])
        with run.phase("fit"):
            fitted, worst = fit_groups(groups, X, Y, G, lo, hi, s)
        history.append((len(X), worst))
        hub = Hub(lo, hi, fitted, m, oracle.evals, oracle.jacobians, worst, history)
        run.round_end(samples=len(X), worst=worst / atol, degrees=",".join(str(g.degree) for g in fitted))
        # Stop at the target, at the budget, or when doubling the samples bought less than 10%: the remaining error
        # is then the model class's (hub dimension, degree cap), which more samples of the same design do not fix.
        stalled = len(history) > 1 and worst > 0.9 * history[-2][1]
        if worst <= 0.5 * atol or stalled or 2 * n + oracle.evals - len(X) > s.max_evals:
            break
        n *= 2
    # the final model uses every sample (the hold-out included), at the degrees chosen above
    with run.phase("fit"):
        for g in hub.groups:
            g.coef = _fit(X, Y[:, g.outputs], None if G is None else G[:, :, g.outputs], lo, hi, g.V, g.r, g.cols, s.grad_weight)
    return run.finish(hub)
