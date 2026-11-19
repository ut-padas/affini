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

import numpy as np

from src.schemes.base import Settings
from src.schemes.chebyshev import (
    axis_decay,
    cheb_basis,
    cheb_coefficients,
    cheb_eval,
    cheb_points,
    chop,
    degree_needed,
    tail_estimate,
)
from src.schemes.directions import active_subspace
from src.schemes.oracle import as_oracle
from src.schemes.profiling import Training


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


def _old_mask(n: list[int], factor: list[int]) -> np.ndarray:
    """On the grown grid, the points the grid before growth already had: index a multiple of the axis's factor."""
    mask = np.ones([nk + 1 for nk in n], dtype=bool)
    for k, fk in enumerate(factor):
        if fk > 1:
            idx = [None] * len(n)
            idx[k] = slice(None)
            mask &= (np.arange(n[k] + 1) % fk == 0)[tuple(idx)]
    return mask


@dataclass(frozen=True)
class ChebTreeSettings(Settings):
    n0: int | tuple = 8  # the root's degree per axis (one for all, or one per axis)
    n_max: int = 64  # largest degree of an axis; beyond it the box splits
    min_drop: float = 16.0  # an axis's decay is exponential when its fit falls by this across the fitted degrees
    safety: float = 0.5  # a box is accepted when its estimate and its checks are within safety * atol
    n_check: int = 4  # drawn check points per new box
    chop_share: float = 0.25  # trailing slabs are chopped while their |coefficients| sum to chop_share * atol
    max_depth: int | None = None  # boxes at this depth are accepted as they are (default 8 d)
    max_evals: int = 5_000_000
    n_floor: int | None = None  # a split resets the split axis's degree to at least this (default the smallest root degree)


@dataclass(frozen=True)
class RotatedChebTreeSettings(ChebTreeSettings):
    n0: int | tuple = 2  # every rotated axis starts low and grows only where its coefficients ask for it
    n_points: int | None = None  # points of the active-subspace estimate (default 10 d)
    seed: int = 0


@dataclass
class Decision:
    """What the check pass decided for one open box: accept it, verify it, grow its degrees, or split it."""

    action: str  # "verify", "accept", "grow" or "split"
    coef: np.ndarray | None = None
    trusted: bool = False
    factor: list | None = None
    split: int | None = None
    caught: bool = False  # the estimate passed but a check point did not (aliasing)


def request(box: Box, grow, table: dict, n_check: int, d: int):
    """The points one open box needs that no box has evaluated yet, its new check points, and what it already knows."""
    if grow == "verify":
        return np.empty((0, d)), fibre_points(box), (np.empty(0), np.empty(0, dtype=bool))
    X = _grid(box)
    new = np.ones(len(X), dtype=bool) if grow is None else ~_old_mask(box.n, grow[1]).ravel()
    V, found = _known(X[new], table)
    drawn = np.random.default_rng(box.code).uniform(box.lo, box.hi, (n_check, d)) if grow is None else np.empty((0, d))
    return X[new][~found], drawn, (V, found)


def install(box: Box, grow, V: np.ndarray, drawn: np.ndarray, fdrawn: np.ndarray) -> None:
    """Put a box's new grid values and check points in place (a grown box keeps its old values on the nested grid)."""
    box.Xc, box.fc = np.concatenate([box.Xc, drawn]), np.concatenate([box.fc, fdrawn])
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


def assess(box: Box, atol: float, s: ChebTreeSettings, d: int, max_depth: int) -> Decision:
    """The box's coefficients, per-axis decay fits, tail estimates and check points, and what follows from them."""
    c = cheb_coefficients(box.V)
    a = np.abs(c)
    fits = [axis_decay(a, k, s.min_drop) for k in range(d)]
    tails = [tail_estimate(fit) for fit in fits]
    check = float(np.max(np.abs(_interpolate(c, box.lo, box.hi, box.Xc) - box.fc))) if len(box.Xc) else 0.0
    estimate_ok = sum(tails) <= s.safety * atol
    passed = estimate_ok and check <= s.safety * atol
    if passed and not box.verified:
        return Decision("verify")
    if passed or box.depth >= max_depth:
        return Decision("accept", coef=chop(c, s.chop_share * atol), trusted=bool(passed))
    factor = [1] * d
    if estimate_ok:  # the grid looks resolved but a check point disagrees: aliasing
        factor = [2 if 2 * box.n[k] <= s.n_max else 1 for k in range(d)]
        split = None if any(fk > 1 for fk in factor) else int(np.argmax(box.hi - box.lo))
        return Decision("grow" if split is None else "split", factor=factor, split=split, caught=True)
    share = s.safety * atol / d
    failing = [k for k in range(d) if tails[k] > share]
    for k in failing:
        need = degree_needed(fits[k], share) if fits[k].exponential else np.inf
        fk = 2 ** max(1, int(np.ceil(np.log2(max(need, 1.0) / box.n[k])))) if np.isfinite(need) else 0
        factor[k] = fk if 0 < fk * box.n[k] <= s.n_max else 0
    unresolved = [k for k in failing if factor[k] == 0]
    split = max(unresolved, key=lambda j: tails[j]) if unresolved else None
    return Decision("grow" if split is None else "split", factor=factor, split=split)


def split_box(tree: ChebTree, box: Box, k: int, n_floor: int, nxt: list) -> None:
    """Halve the box across axis k; the halves inherit every known value inside them and open in ``nxt``."""
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


def grow(oracle, lo, hi, atol: float, s: ChebTreeSettings, run: Training) -> ChebTree:
    """
    Refine the tree round by round until no box is open. Each round makes one oracle call for every open
    box's new points (a new box's grid, a grown box's new grid points, new boxes' drawn check points, the
    fibre points of boxes about to be accepted), then fits, checks and adapts every open box in order.
    """
    d = len(lo)
    max_depth = 8 * d if s.max_depth is None else s.max_depth
    n_root = [int(s.n0)] * d if np.isscalar(s.n0) else [int(v) for v in s.n0]
    n_floor = min(n_root) if s.n_floor is None else s.n_floor
    tree = ChebTree()
    root = Box(lo, hi, list(n_root), node=tree.new_node(), Xc=np.empty((0, d)), fc=np.empty(0))
    pending = [(root, None)]  # (box, None: a new box | (old values, per-axis growth factors) | "verify")
    table: dict[bytes, float] = {}  # every evaluated grid point, so that no box evaluates a neighbour's point again
    while pending:
        with run.phase("sample"):
            requests, plans = [], []
            for box, how in pending:
                needed, drawn, known = request(box, how, table, s.n_check, d)
                requests += [needed, drawn]
                plans.append((known, drawn))
            sizes = [len(X) for X in requests]
            unique, inverse = np.unique(np.concatenate(requests), axis=0, return_inverse=True)
        if tree.evals + len(unique) > s.max_evals:
            raise RuntimeError(f"ChebTree would exceed {s.max_evals:,} evaluations.")
        fu = oracle(unique).ravel()
        tree.evals += len(unique)
        with run.phase("fit"):
            table.update(zip((row.tobytes() for row in unique), fu.tolist(), strict=True))
            values = iter(np.split(fu[inverse.ravel()], np.cumsum(sizes)[:-1]))
            for (box, how), ((V, found), drawn) in zip(pending, plans, strict=True):
                V[~found] = next(values)
                install(box, how, V, drawn, next(values))
                tree.check_evals += len(drawn)
        with run.phase("check"):
            decisions = [assess(box, atol, s, d, max_depth) for box, _ in pending]
        with run.phase("adapt"):
            nxt = []
            for (box, _), dec in zip(pending, decisions, strict=True):
                if dec.action == "verify":
                    nxt.append((box, "verify"))
                elif dec.action == "accept":
                    tree.leaf[box.node] = len(tree.leaves)
                    tree.leaves.append((box.lo, box.hi, dec.coef))
                    tree.trusted.append(dec.trusted)
                else:
                    tree.check_catches += dec.caught
                    if dec.split is None:
                        old = box.V
                        box.n = [nk * fk for nk, fk in zip(box.n, dec.factor, strict=True)]
                        box.verified = False
                        nxt.append((box, (old, dec.factor)))
                    else:
                        split_box(tree, box, dec.split, n_floor, nxt)
        actions = [dec.action for dec in decisions]
        run.round_end(open=len(pending), leaves=len(tree.leaves), verify=actions.count("verify"), grow=actions.count("grow"),
                      split=actions.count("split"))  # fmt: skip
        pending = nxt
    return tree


def build(f, lo, hi, atol: float, settings: ChebTreeSettings | None = None, *, jacobian=None, stream_dir=None,
          **overrides) -> ChebTree:  # fmt: skip
    """
    Train a ChebTree of ``f`` (host (N, d) -> (N,) values) on the box [lo, hi] to tolerance ``atol``.
    ``n0`` is the root's degree per axis (an int, or one per axis); a split resets the split axis to
    at least ``n_floor`` (default: the smallest root degree). A box at ``max_depth`` (default 8 d:
    on average 8 halvings per axis, boxes down to 1/256 of the domain) is accepted as it is, so a
    jump, which no polynomial resolves, ends in a thin band of small boxes instead of endless splits;
    unless it also passed its estimate and checks, it is untrusted and ``refuses`` its queries.
    """
    s = ChebTreeSettings.resolve(settings, overrides)
    oracle = as_oracle(f, jacobian)
    run = Training("chebtree", s, oracle, stream_dir)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return run.finish(grow(oracle, lo, hi, atol, s, run))


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

    def refuses(self, X):
        return self.tree.refuses((np.asarray(X) - self.centre) @ self.W)


def build_rotated(f, lo, hi, atol: float, settings: RotatedChebTreeSettings | None = None, *, jacobian=None, stream_dir=None,
                  **overrides) -> RotatedChebTree:  # fmt: skip
    """
    ChebTree in the active-subspace coordinates: the rotated box is the bounding box of the domain
    in those coordinates, so f is also evaluated outside [lo, hi] (at most a factor sqrt(d) out).
    Every axis starts at degree ``n0`` and grows only where its coefficients ask for it, so a
    direction along which f is constant costs ``n0 + 1`` points.
    """
    s = RotatedChebTreeSettings.resolve(settings, overrides)
    oracle = as_oracle(f, jacobian)
    run = Training("chebtree_rotated", s, oracle, stream_dir)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = len(lo)
    with run.phase("explore"):
        W, lam = active_subspace(oracle, lo, hi, s.n_points or 10 * d, s.seed)
    c = 0.5 * (lo + hi)
    half = np.abs(W).T @ (0.5 * (hi - lo))  # bounding box of the domain in y
    tree = grow(lambda Y: oracle(c + Y @ W.T), -half, half, atol, s, run)
    return run.finish(RotatedChebTree(tree, c, W, lam, oracle.evals))
