"""
ChebTree: an adaptive tree of tensor Chebyshev interpolants, refined by the decay of their coefficients.

Every piece is a box carrying the tensor-product interpolant of f at Chebyshev points of the
second kind, ``n_k + 1`` points along axis k. Its Chebyshev coefficients come from one DCT-I per
axis of the box's values. Along each axis, the slab sums ``s_j`` (the sum of |coefficient| over
every coefficient of degree j on that axis) bound how much dropping degree j changes the
interpolant, and they do three jobs:

- **Error estimate.** Along each axis the slab sums are fitted on the upper half of the degrees,
  geometrically (``C exp(-sigma j)``, Mavriplis) and algebraically (``Ca (j + 1)^-p``). The decay
  counts as exponential when the fit falls by ``min_drop`` across those degrees; no power law up to
  ``j^-log2(min_drop)`` does. The axis's estimate is twice the fitted tail of the omitted degrees
  (interpolation folds them back once): the larger of the two laws' tails where the decay is
  exponential, and the algebraic tail of a kink (power at most 2) otherwise; at least the last slab.
  A box is accepted when the estimates sum to at most ``safety * atol`` and the interpolant also
  meets ``safety * atol`` at the box's check points (below).
- **Degree refinement (p).** A failing axis with exponential decay gets the degree the geometric
  fit predicts, rounded up to the next power-of-two multiple of its degree: the second-kind points of
  degree 2^m n contain those of degree n, so only the new points are evaluated, and every failing
  axis grows in the same round.
- **Splitting (h).** A failing axis without exponential decay (a kink, a front, more oscillation than
  the box can follow), or whose predicted degree exceeds ``n_max``, is halved, and each half starts
  from the parent's degrees, halved along the split axis.

**Check points.** The coefficients only see the grid, so an oscillation the grid aliases can look
resolved. Every new box therefore draws ``n_check`` uniform points from a generator seeded by its
locational code (the root is 1, the children of code c are 2c and 2c + 1), so a run is
deterministic, and inherits its parent's check points and grid values inside it, which cost no
evaluation. If the estimate passes but a check point fails, every axis below ``n_max`` doubles.
A box that passes both is verified once more before it is accepted: one more round evaluates, along
each axis through its centre, the points the next nested grid would add (``fibre_points``), and
these join its check points. A kink close to a face makes the slab sums oscillate with a period
longer than the degree, so they can read as decay, and four random points rarely land in the
narrow error spike beside the kink; a fibre crosses every axis-normal feature of the box.

After acceptance, trailing coefficient slabs are chopped, smallest first, while the chopped
|coefficients| sum to at most ``chop_share * atol``: inference costs the kept coefficients only.

Each round processes every open box and evaluates all of their new points in one oracle call, so
the work is per box and batchable. A query is routed down the tree of box splits to its box and
evaluated by contracting the box's coefficient tensor with the Chebyshev polynomials at the query.

Known limits: the fit and the checks are estimates, not bounds, so a feature narrower than both
the grid and the check points can still be missed; and tensor grids cost ``prod(n_k + 1)`` points
per box, which limits plain ChebTree to low dimension.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from scipy.fft import dct


def cheb_points(n: int) -> np.ndarray:
    """Chebyshev points of the second kind on [-1, 1], ``cos(pi j / n)``, j = 0..n (descending)."""
    return np.cos(np.pi * np.arange(n + 1) / n)


def cheb_coefficients(V: np.ndarray) -> np.ndarray:
    """Tensor Chebyshev coefficients of the interpolant through values ``V`` on the second-kind grid."""
    a = np.asarray(V, dtype=np.float64)
    for k in range(a.ndim):
        n = a.shape[k] - 1
        a = dct(a, type=1, axis=k) / n
        edge = [slice(None)] * a.ndim
        for j in (0, n):
            edge[k] = j
            a[tuple(edge)] /= 2.0
    return a


def cheb_eval(coef: np.ndarray, T: np.ndarray) -> np.ndarray:
    """``sum_i coef[i] prod_k T_k[q, i_k]`` for per-axis Chebyshev values ``T`` (list of (Q, n_k + 1))."""
    Q = T[0].shape[0]
    R = T[0] @ coef.reshape(coef.shape[0], -1)  # (Q, rest)
    for k in range(1, coef.ndim):
        R = R.reshape(Q, coef.shape[k], -1)
        R = np.einsum("qi,qir->qr", T[k], R)
    return R[:, 0]


def cheb_basis(t: np.ndarray, n: int) -> np.ndarray:
    """(Q, n + 1) values T_0..T_n at ``t`` in [-1, 1]."""
    return np.cos(np.outer(np.arccos(np.clip(t, -1.0, 1.0)), np.arange(n + 1)))


@dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray
    n: list[int]
    V: np.ndarray | None = None  # values on the tensor grid, shape (n_k + 1 for k)
    depth: int = 0
    node: int = 0  # index in the tree
    code: int = 1  # locational code: the root is 1, the children of code c are 2c and 2c + 1
    Xc: np.ndarray | None = None  # check points with known values (drawn, or inherited), and the values
    fc: np.ndarray | None = None
    verified: bool = False  # its fibre points at the current degrees are among its check points


@dataclass
class ChebTree:
    """The trained tree: split nodes (axis, value, children) and leaf boxes with their coefficients."""

    axis: list[int] = field(default_factory=list)  # -1 for a leaf
    value: list[float] = field(default_factory=list)
    child: list[tuple[int, int]] = field(default_factory=list)
    leaf: list[int] = field(default_factory=list)  # leaf index or -1
    leaves: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)  # (lo, hi, coef)
    trusted: list[bool] = field(default_factory=list)  # per leaf: accepted on its estimate and checks, not at max_depth
    evals: int = 0
    check_evals: int = 0  # evaluations spent on drawn check points (included in evals)
    check_catches: int = 0  # boxes whose estimate passed but whose check points did not

    def new_node(self) -> int:
        self.axis.append(-1), self.value.append(0.0), self.child.append((-1, -1)), self.leaf.append(-1)
        return len(self.axis) - 1

    def leaf_of(self, X: np.ndarray) -> np.ndarray:
        node = np.zeros(len(X), dtype=np.int64)
        axis, value = np.asarray(self.axis), np.asarray(self.value)
        child, leaf = np.asarray(self.child), np.asarray(self.leaf)
        while True:
            act = np.nonzero(axis[node] >= 0)[0]
            if len(act) == 0:
                return leaf[node]
            na = node[act]
            right = X[act, axis[na]] >= value[na]
            node[act] = child[na, right.astype(np.int64)]

    def refuses(self, X: np.ndarray) -> np.ndarray:
        """Per query: its box was accepted only because it reached ``max_depth``, so the caller should ask the oracle."""
        return ~np.asarray(self.trusted, dtype=bool)[self.leaf_of(np.asarray(X, dtype=np.float64))]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(len(X))
        lid = self.leaf_of(X)
        for i in np.unique(lid):
            rows = np.nonzero(lid == i)[0]
            lo, hi, coef = self.leaves[i]
            t = 2.0 * (X[rows] - lo) / (hi - lo) - 1.0
            out[rows] = cheb_eval(coef, [cheb_basis(t[:, k], coef.shape[k] - 1) for k in range(len(lo))])
        return out


def _axes(box: Box) -> list[np.ndarray]:
    """
    Each axis's physical grid coordinates, with the ends and the midpoint set exactly, so that boxes
    sharing a face (siblings, neighbours, a child and its parent's grid) share its points bit for bit
    and the value table finds them instead of evaluating them again.
    """
    axes = []
    for k, nk in enumerate(box.n):
        x = box.lo[k] + (cheb_points(nk) + 1.0) * 0.5 * (box.hi[k] - box.lo[k])
        x[0], x[-1] = box.hi[k], box.lo[k]
        if nk % 2 == 0:
            x[nk // 2] = 0.5 * (box.lo[k] + box.hi[k])
        axes.append(x)
    return axes


def _known(X: np.ndarray, table: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(values, found)``: the values of the rows of X already in ``table`` (exact coordinates -> value)."""
    keys = [row.tobytes() for row in X]
    found = np.array([k in table for k in keys], dtype=bool)
    return np.array([table.get(k, np.nan) for k in keys]), found


def _grid(box: Box) -> np.ndarray:
    """Physical points of the box's tensor grid, in the C order of its value tensor."""
    mesh = np.meshgrid(*_axes(box), indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def _interpolate(coef: np.ndarray, lo: np.ndarray, hi: np.ndarray, X: np.ndarray) -> np.ndarray:
    t = 2.0 * (X - lo) / (hi - lo) - 1.0
    return cheb_eval(coef, [cheb_basis(t[:, k], coef.shape[k] - 1) for k in range(len(lo))])


class AxisFit(NamedTuple):
    """One axis's slab sums ``s`` and the two decay laws fitted on the upper half of its degrees."""

    s: np.ndarray
    C: float  # geometric law s_j ~ C exp(-sigma j)
    sigma: float  # inf: resolved to rounding level; 0: too few degrees to fit (no decay)
    Ca: float  # algebraic law s_j ~ Ca (j + 1)^-p
    p: float
    exponential: bool  # the fit falls by at least ``min_drop`` across the fitted degrees


def axis_decay(a: np.ndarray, k: int, min_drop: float) -> AxisFit:
    """
    Fit axis k's slab sums of |a| on the upper half of the degrees, geometrically (Mavriplis) and
    algebraically (the local regularity of Houston and Suli). The fits read the pairwise maxima
    ``max(s_j, s_j+1)``, so that the zero coefficients of a symmetric f do not read as decay, and
    skip entries below the rounding floor; the last pair below the floor means resolved, as
    Chebfun's chopping reads a plateau at rounding level. The decay counts as exponential when the
    geometric fit falls by ``min_drop`` across the fitted degrees: a threshold on the rate alone
    cannot tell a kink from a slowly converging analytic f on a few degrees, but a power law
    ``j^-p`` falls by only 2^p across ``[n/2, n]``, whatever n.
    """
    s = np.moveaxis(a, k, 0).reshape(a.shape[k], -1).sum(axis=1)
    n = len(s) - 1
    flat = AxisFit(s, float(s.max()), 0.0, float(s.max()), 0.0, False)
    if n < 2:
        return flat
    j = np.arange(min(n // 2, n - 2), n)
    y = np.maximum(s[:-1], s[1:])[j]
    floor = 1e-13 * max(float(s.max()), 1e-300)
    if y[-1] <= floor:
        return AxisFit(s, 0.0, np.inf, 0.0, np.inf, True)
    keep = y > floor
    if keep.sum() < 2:
        return flat
    slope, intercept = np.polyfit(j[keep], np.log(y[keep]), 1)
    power, log_ca = np.polyfit(np.log(j[keep] + 1.0), np.log(y[keep]), 1)
    sigma = float(-slope)
    return AxisFit(s, float(np.exp(intercept)), sigma, float(np.exp(log_ca)), float(-power), sigma * (j[-1] - j[0]) >= np.log(min_drop))


def tail_estimate(fit: AxisFit) -> float:
    """
    Twice the fitted tail of the omitted degrees (interpolation folds them back once), and at least
    the last slab. Where the decay is exponential the tail is the larger of the geometric and the
    algebraic fit's, since a few degrees barely tell the two laws apart. Otherwise f is assumed no
    smoother than a kink there: the algebraic tail with the power capped at 2 (a kink's
    coefficients decay as j^-2; a power fitted on a few pre-asymptotic degrees reads higher and
    underestimates the tail), anchored at the last pair of slabs. A power of at most 1 (a jump)
    never passes.
    """
    s, C, sigma, Ca, p, exponential = fit
    n = len(s) - 1
    if sigma == np.inf:
        return float(s[-1])
    algebraic = Ca * (n + 1.5) ** (1.0 - p) / (p - 1.0) if p > 1.0 else np.inf
    if exponential:
        r = np.exp(-sigma)
        tail = max(C * r ** (n + 1) / (1.0 - r), algebraic)
    else:
        tail = max(float(max(s[-2], s[-1])) * n / (min(p, 2.0) - 1.0), algebraic) if p > 1.0 else np.inf
    return max(2.0 * tail, float(s[-1]))


def degree_needed(fit: AxisFit, share: float) -> float:
    """The smallest degree at which the fitted geometric law's tail estimate meets ``share`` (used only where the decay is exponential)."""
    if fit.sigma == np.inf:
        return 0.0
    r = np.exp(-fit.sigma)
    return max(np.log(2.0 * fit.C * r / ((1.0 - r) * share)), np.log(fit.C / share)) / fit.sigma


def chop(coef: np.ndarray, budget: float) -> np.ndarray:
    """Drop trailing degree slabs, the smallest first, while the dropped |coefficients| sum to at most ``budget``; |T_j| <= 1 makes that sum a bound on the change."""
    spent = 0.0
    while True:
        w = [np.abs(np.take(coef, -1, axis=k)).sum() if coef.shape[k] > 1 else np.inf for k in range(coef.ndim)]
        k = int(np.argmin(w))
        if spent + w[k] > budget:
            return coef
        spent += w[k]
        coef = np.take(coef, np.arange(coef.shape[k] - 1), axis=k)


def build(f, lo, hi, atol: float, n0: int = 8, n_max: int = 64, min_drop: float = 16.0, safety: float = 0.5,
          n_check: int = 4, chop_share: float = 0.25, max_depth: int | None = None, max_evals: int = 5_000_000,
          n_floor: int | None = None) -> ChebTree:  # fmt: skip
    """
    Train a ChebTree of ``f`` (host (N, d) -> (N,) values) on the box [lo, hi] to tolerance ``atol``.
    ``n0`` is the root's degree per axis (an int, or one per axis); a split resets the split axis to
    at least ``n_floor`` (default: the smallest root degree). A box at ``max_depth`` (default 8 d:
    on average 8 halvings per axis, boxes down to 1/256 of the domain) is accepted as it is, so a
    jump, which no polynomial resolves, ends in a thin band of small boxes instead of endless splits;
    unless it also passed its estimate and checks, it is untrusted and ``refuses`` its queries.
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = len(lo)
    max_depth = 8 * d if max_depth is None else max_depth
    n_root = [int(n0)] * d if np.isscalar(n0) else [int(v) for v in n0]
    n_floor = min(n_root) if n_floor is None else n_floor
    tree = ChebTree()
    root = Box(lo, hi, list(n_root), node=tree.new_node(), Xc=np.empty((0, d)), fc=np.empty(0))
    pending = [(root, None)]  # (box, None: a new box | (old values, per-axis growth factors))

    table: dict[bytes, float] = {}  # every evaluated grid point, so that no box evaluates a neighbour's point again

    def evaluate(X):
        tree.evals += len(X)
        return np.asarray(f(X), dtype=np.float64).ravel()

    while pending:
        # 1. One oracle call for the points the open boxes need that no box has evaluated yet (a new
        #    box's grid, a grown box's new grid points), for the new boxes' drawn check points, and for
        #    the fibre points of the boxes about to be accepted.
        requests, plans = [], []
        for box, grow in pending:
            if grow == "verify":
                V, found, drawn = np.empty(0), np.empty(0, dtype=bool), fibre_points(box)
                requests.append(np.empty((0, d)))
            else:
                X = _grid(box)
                new = np.ones(len(X), dtype=bool) if grow is None else ~_old_mask(box.n, grow[1]).ravel()
                V, found = _known(X[new], table)
                requests.append(X[new][~found])
                drawn = np.random.default_rng(box.code).uniform(box.lo, box.hi, (n_check, d)) if grow is None else np.empty((0, d))
            requests.append(drawn)
            plans.append((V, found, drawn))
        sizes = [len(X) for X in requests]
        unique, inverse = np.unique(np.concatenate(requests), axis=0, return_inverse=True)
        if tree.evals + len(unique) > max_evals:
            raise RuntimeError(f"ChebTree would exceed {max_evals:,} evaluations.")
        fu = evaluate(unique)
        table.update(zip((row.tobytes() for row in unique), fu.tolist(), strict=True))
        values = iter(np.split(fu[inverse.ravel()], np.cumsum(sizes)[:-1]))
        nxt = []
        for (box, grow), (V, found, drawn) in zip(pending, plans, strict=True):
            V[~found] = next(values)
            box.Xc, box.fc = np.concatenate([box.Xc, drawn]), np.concatenate([box.fc, next(values)])
            tree.check_evals += len(drawn)
            shape = [nk + 1 for nk in box.n]
            if grow == "verify":
                box.verified = True
            elif grow is None:
                box.V = V.reshape(shape)
            else:
                old, factor = grow
                box.V = np.empty(shape)
                box.V[tuple(slice(None, None, fk) for fk in factor)] = old
                box.V[~_old_mask(box.n, factor)] = V
            # 2. Coefficients, per-axis decay fits, tail estimates and the check points.
            c = cheb_coefficients(box.V)
            a = np.abs(c)
            fits = [axis_decay(a, k, min_drop) for k in range(d)]
            tails = [tail_estimate(fit) for fit in fits]
            check = float(np.max(np.abs(_interpolate(c, box.lo, box.hi, box.Xc) - box.fc))) if len(box.Xc) else 0.0
            estimate_ok = sum(tails) <= safety * atol
            if estimate_ok and check <= safety * atol and not box.verified:
                nxt.append((box, "verify"))
                continue
            if (estimate_ok and check <= safety * atol) or box.depth >= max_depth:
                tree.leaf[box.node] = len(tree.leaves)
                tree.leaves.append((box.lo, box.hi, chop(c, chop_share * atol)))
                tree.trusted.append(bool(estimate_ok and check <= safety * atol))
                continue
            factor = [1] * d
            if estimate_ok:  # the grid looks resolved but a check point disagrees: aliasing
                tree.check_catches += 1
                factor = [2 if 2 * box.n[k] <= n_max else 1 for k in range(d)]
                split = None if any(fk > 1 for fk in factor) else int(np.argmax(box.hi - box.lo))
            else:
                share = safety * atol / d
                failing = [k for k in range(d) if tails[k] > share]
                for k in failing:
                    need = degree_needed(fits[k], share) if fits[k].exponential else np.inf
                    fk = 2 ** max(1, int(np.ceil(np.log2(max(need, 1.0) / box.n[k])))) if np.isfinite(need) else 0
                    factor[k] = fk if 0 < fk * box.n[k] <= n_max else 0
                unresolved = [k for k in failing if factor[k] == 0]
                split = max(unresolved, key=lambda j: tails[j]) if unresolved else None
            if split is None:
                old = box.V
                box.n = [nk * fk for nk, fk in zip(box.n, factor, strict=True)]
                box.verified = False
                nxt.append((box, (old, factor)))
                continue
            # 3. Split across the unresolved axis with the largest tail; the halves inherit every known value inside them.
            k = split
            mid = 0.5 * (box.lo[k] + box.hi[k])
            tree.axis[box.node], tree.value[box.node] = k, mid
            Xk = np.concatenate([box.Xc, _grid(box)])
            fX = np.concatenate([box.fc, box.V.ravel()])
            kids = []
            for side in (0, 1):
                clo, chi = box.lo.copy(), box.hi.copy()
                (chi if side == 0 else clo)[k] = mid
                n = list(box.n)
                n[k] = max(n_floor, n[k] // 2)
                inside = (Xk[:, k] >= mid) if side else (Xk[:, k] < mid)
                kid = Box(clo, chi, n, depth=box.depth + 1, node=tree.new_node(), code=2 * box.code + side,
                          Xc=Xk[inside], fc=fX[inside])  # fmt: skip
                kids.append(kid.node)
                nxt.append((kid, None))
            tree.child[box.node] = (kids[0], kids[1])
        pending = nxt
    return tree


def fibre_points(box: Box) -> np.ndarray:
    """
    Along each axis, through the box's centre, the points the next nested grid (degree 2 n_k) adds:
    the midpoints in angle between the nodes, where the interpolation error peaks. A kink or a front
    crossing the box crosses every fibre along its normal's axes, so these catch what the coefficients
    misread (a kink near a face makes the slab sums oscillate with a period longer than the degree).
    """
    c = 0.5 * (box.lo + box.hi)
    rows = []
    for k, nk in enumerate(box.n):
        t = np.cos(np.pi * (np.arange(nk) + 0.5) / nk)
        P = np.repeat(c[None], nk, axis=0)
        P[:, k] = box.lo[k] + (t + 1.0) * 0.5 * (box.hi[k] - box.lo[k])
        rows.append(P)
    return np.concatenate(rows)


def _old_mask(n: list[int], factor: list[int]) -> np.ndarray:
    """On the grown grid, the points the grid before growth already had: index a multiple of the axis's factor."""
    mask = np.ones([nk + 1 for nk in n], dtype=bool)
    for k, fk in enumerate(factor):
        if fk > 1:
            idx = [None] * len(n)
            idx[k] = slice(None)
            mask &= (np.arange(n[k] + 1) % fk == 0)[tuple(idx)]
    return mask


@dataclass
class RotatedChebTree:
    """A ChebTree in rotated coordinates ``y = W^T (x - c)``: the eigenvectors of the gradient outer products."""

    tree: ChebTree
    centre: np.ndarray
    W: np.ndarray
    eigenvalues: np.ndarray
    evals: int

    def predict(self, X):
        return self.tree.predict((np.asarray(X) - self.centre) @ self.W)

    def leaf_of(self, X):
        return self.tree.leaf_of((np.asarray(X) - self.centre) @ self.W)

    def refuses(self, X):
        return self.tree.refuses((np.asarray(X) - self.centre) @ self.W)


def gradient_directions(f, lo, hi, n_points: int, seed: int = 0, step: float = 1e-5):
    """
    ``(W, eigenvalues, evaluations)``: eigenvectors (descending) of ``C = mean grad f grad f^T`` from
    forward differences at ``n_points`` uniform points of the box (the active-subspace estimate).
    """
    d = len(lo)
    X = np.random.default_rng(seed).uniform(lo, hi, (n_points, d))
    h = step * (np.asarray(hi) - np.asarray(lo))
    pts = np.concatenate([X] + [X + np.eye(d)[k] * h[k] for k in range(d)])
    v = np.asarray(f(pts), dtype=np.float64).ravel().reshape(d + 1, n_points)
    G = ((v[1:] - v[0]) / h[:, None]).T  # (n_points, d)
    lam, W = np.linalg.eigh(G.T @ G / n_points)
    order = np.argsort(lam)[::-1]
    return W[:, order], lam[order], len(pts)


def build_rotated(f, lo, hi, atol: float, n_points: int | None = None, seed: int = 0, n0: int = 2, **kw) -> RotatedChebTree:
    """
    ChebTree in the active-subspace coordinates: the rotated box is the bounding box of the domain
    in those coordinates, so f is also evaluated outside [lo, hi] (at most a factor sqrt(d) out).
    Every axis starts at degree ``n0`` and grows only where its coefficients ask for it, so a
    direction along which f is constant costs ``n0 + 1`` points.
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = len(lo)
    W, lam, used = gradient_directions(f, lo, hi, n_points or 10 * d, seed)
    c = 0.5 * (lo + hi)
    half = np.abs(W).T @ (0.5 * (hi - lo))  # bounding box of the domain in y
    tree = build(lambda Y: f(c + Y @ W.T), -half, half, atol, n0=n0, **kw)
    return RotatedChebTree(tree, c, W, lam, used + tree.evals)
