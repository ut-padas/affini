"""
ChebTree: an adaptive tree of tensor Chebyshev interpolants, refined by the decay of their coefficients.

Every piece is a box carrying the tensor-product interpolant of f at Chebyshev points of the
second kind, ``n_k + 1`` points along axis k. Its Chebyshev coefficients come from one DCT-I per
axis of the box's values. They do three jobs:

- **Error estimate.** For a function analytic in the box the coefficients decay geometrically, and
  the interpolant's worst error is of the order of the first omitted coefficients. The box is
  accepted when the coefficients of the two highest degrees along every axis sum to at most
  ``safety * atol`` (``safety = 1/2``).
- **Degree refinement (p).** An axis whose coefficients decay (the top quarter of degrees at most
  ``decay`` times the bottom quarter) but are not yet small enough gets twice the degree. The
  second-kind points of degree 2n contain those of degree n, so only the new points are evaluated.
- **Splitting (h).** An axis whose coefficients do not decay, or that has reached ``n_max``, is
  unresolved at this size (a front, a kink, more oscillation than the degree can follow): the box is
  halved across it, and each half starts from the parent's degrees, halved along the split axis.

Each round processes every open box and evaluates all of their new points in one oracle call, so
the work is per box and batchable. A query is routed down the tree of box splits to its box and
evaluated by contracting the box's coefficient tensor with the Chebyshev polynomials at the query.

Known limits: the estimate trusts what the samples show, so a feature narrower than the grid
spacing can be missed (aliasing); and tensor grids cost ``prod(n_k + 1)`` points per box, which
limits plain ChebTree to low dimension.
"""

from dataclasses import dataclass, field

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


@dataclass
class ChebTree:
    """The trained tree: split nodes (axis, value, children) and leaf boxes with their coefficients."""

    axis: list[int] = field(default_factory=list)  # -1 for a leaf
    value: list[float] = field(default_factory=list)
    child: list[tuple[int, int]] = field(default_factory=list)
    leaf: list[int] = field(default_factory=list)  # leaf index or -1
    leaves: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)  # (lo, hi, coef)
    evals: int = 0

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


def _grid(box: Box, axes_new: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Physical points of the box's tensor grid (or of the sub-grid ``axes_new`` replaces on some axes)."""
    axes = []
    for k, nk in enumerate(box.n):
        t = cheb_points(nk) if axes_new is None or k not in axes_new else axes_new[k]
        axes.append(box.lo[k] + (t + 1.0) * 0.5 * (box.hi[k] - box.lo[k]))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def build(f, lo, hi, atol: float, n0: int = 8, n_max: int = 64, decay: float = 1e-3, safety: float = 0.5,
          max_depth: int | None = None, max_evals: int = 5_000_000, n_floor: int | None = None) -> ChebTree:  # fmt: skip
    """
    Train a ChebTree of ``f`` (host (N, d) -> (N,) values) on the box [lo, hi] to tolerance ``atol``.
    ``n0`` is the root's degree per axis (an int, or one per axis); a split resets the split axis to
    at least ``n_floor`` (default: the smallest root degree). A box at ``max_depth`` (default 8 d:
    on average 8 halvings per axis, boxes down to 1/256 of the domain) is accepted as it is, so a
    jump, which no polynomial resolves, ends in a thin band of small boxes instead of endless splits.
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = len(lo)
    max_depth = 8 * d if max_depth is None else max_depth
    n_root = [int(n0)] * d if np.isscalar(n0) else [int(v) for v in n0]
    n_floor = min(n_root) if n_floor is None else n_floor
    tree = ChebTree()
    root = Box(lo, hi, list(n_root), node=tree.new_node())
    pending = [(root, None)]  # (box, None: whole grid | (axis, values to merge))

    def evaluate(X):
        tree.evals += len(X)
        return np.asarray(f(X), dtype=np.float64).ravel()

    while pending:
        # 1. One oracle call for every point the open boxes need.
        requests, sizes = [], []
        for box, refine in pending:
            if refine is None:
                X = _grid(box)
            else:
                k, _ = refine
                X = _grid(box, {k: cheb_points(box.n[k])[1::2]})  # the odd points of the doubled grid
            requests.append(X)
            sizes.append(len(X))
        if tree.evals + sum(sizes) > max_evals:
            raise RuntimeError(f"ChebTree would exceed {max_evals:,} evaluations.")
        values = np.split(evaluate(np.concatenate(requests)), np.cumsum(sizes)[:-1])
        nxt = []
        for (box, refine), v in zip(pending, values, strict=True):
            if refine is None:
                box.V = v.reshape([nk + 1 for nk in box.n])
            else:  # interleave: old values at even indices along the axis, new at odd ones
                k, old = refine
                shape = list(old.shape)
                shape[k] = box.n[k] + 1
                V = np.empty(shape)
                sl = [slice(None)] * d
                sl[k] = slice(0, None, 2)
                V[tuple(sl)] = old
                sl[k] = slice(1, None, 2)
                new_shape = list(old.shape)
                new_shape[k] = box.n[k] // 2
                V[tuple(sl)] = v.reshape(new_shape)
                box.V = V
            # 2. Coefficients, per-axis tails and decay.
            a = np.abs(cheb_coefficients(box.V))
            tails, decaying = [], []
            for k in range(d):
                nk = box.n[k]
                env = np.max(np.moveaxis(a, k, 0).reshape(nk + 1, -1), axis=1)
                tails.append(float(np.moveaxis(a, k, 0)[nk - 1 :].sum()))
                q = max(1, (nk + 1) // 4)
                decaying.append(env[-q:].max() <= decay * max(env[:q].max(), 1e-300))
            if sum(tails) <= safety * atol or box.depth >= max_depth:
                tree.leaf[box.node] = len(tree.leaves)
                tree.leaves.append((box.lo, box.hi, cheb_coefficients(box.V)))
                continue
            share = safety * atol / d
            failing = [k for k in range(d) if tails[k] > share]
            grow = [k for k in failing if decaying[k] and box.n[k] < n_max]
            if grow and len(grow) == len(failing):
                k = max(grow, key=lambda j: tails[j])  # one axis per round: its doubling reuses every value
                box.n[k] *= 2
                nxt.append((box, (k, box.V)))
                continue
            # 3. Split across the unresolved axis with the largest tail.
            k = max((j for j in failing if j not in grow), key=lambda j: tails[j])
            mid = 0.5 * (box.lo[k] + box.hi[k])
            tree.axis[box.node], tree.value[box.node] = k, mid
            kids = []
            for side in (0, 1):
                clo, chi = box.lo.copy(), box.hi.copy()
                (chi if side == 0 else clo)[k] = mid
                n = list(box.n)
                n[k] = max(n_floor, n[k] // 2)
                kid = Box(clo, chi, n, depth=box.depth + 1, node=tree.new_node())
                kids.append(kid.node)
                nxt.append((kid, None))
            tree.child[box.node] = (kids[0], kids[1])
        pending = nxt
    return tree


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
