"""
ChebTT: a Chebyshev interpolant in tensor-train form, built by cross approximation.

f is sampled on the tensor grid of Chebyshev points of the second kind (``n_k + 1`` per axis), but
only along fibres: the TT-cross of Oseledets and Tyrtyshnikov keeps, between axes k and k + 1, a set
of ``r_k`` left multi-indices (positions on axes 1..k) and ``r_k`` right ones (axes k+1..d), and
evaluates f on the fibres ``left x (every node of axis k) x right``. Sweeping left to right, the QR
factor of each fibre matrix gives the core, and its maxvol rows (the near-largest-volume square
submatrix) become the next left set; the sweep back updates the right sets the same way. The model
is ``f(x) ~ G_1(x_1) ... G_d(x_d)``, each core's node values turned into Chebyshev coefficients along
its axis, so a query costs ``sum_k n_k r_k r_{k+1}``: linear in d.

Sampling cost is also linear in d: ``sum_k r_k (n_k + 1) r_{k+1}`` per sweep. Values are cached on
the finest nested lattice, so a fibre seen again, or an old node of an axis whose degree doubled,
costs nothing. Products of univariate functions have rank 1, ``cos(q.x)`` rank 2, and functions
with local coupling low rank; rotated ellipsoidal features, kinks along oblique planes, chaotic maps
and jumps have high rank (the format's limits).

Adaptivity and acceptance. The cross starts from degrees read off ``n_fibre`` random fibres per axis
(the smallest nested degree at which each fibre's Chebyshev tail meets the axis's share of the
tolerance), at rank ``rank0``, its right sets seeded with the node nearest the largest |f| seen and
the centre node. After each round of sweeps the model is checked at three sets:

- a scrambled Sobol set (evaluated once);
- free checks: grid points evaluated in earlier sweeps or at lower degrees, which the final cores
  do not interpolate, concentrated where the cross explored;
- a look-ahead along every axis through the worst check point and through the largest |f| seen, at
  the points the next nested grid would add (kept as check points for later rounds).

If any error exceeds ``safety * atol``: every axis whose top quarter of degrees still changes the
model by more than its share at those points doubles its degree (no rank makes up for a missing
degree); otherwise the rank grows by ``rank_step`` with random right indices, and the sweeps resume.

Piecewise axes. A sum of products of univariate functions can be non-smooth only along axis-aligned
planes, and there one global Chebyshev basis per axis converges slowly (a kink's coefficients decay
like j^-2, a jump's like j^-1) and the cross grows the rank in vain. So an axis may carry breakpoints:
its pieces share one degree, its nodes are the union of the pieces' Chebyshev points (inset by
``INSET`` at an interior breakpoint, so a jump is never sampled on the wrong side), and a query uses
its own piece's basis. Breakpoints come from ``axis_edges`` (an outlying second difference on a
uniform scan, located by zooming to 1e-12 of the width, confirmed by a two-scale test and by a
displaced fibre), searched only where it can help: an axis still unconverged at degree 32 in the
pre-pass, an axis that must refine, or before the rank grows, and in each case only if the axis's
coefficients decay algebraically. A function without edges gets exactly the model it would without
the search. Oblique and curved interfaces are not representable this way, at any rank.

Once accepted, the cores are compressed (``compress``): ranks rounded and trailing degrees chopped while
the model changes by at most ``chop_share * atol`` at the check points and ``n_probe`` uniform points,
which cuts the cores and the inference time several-fold.

The checks are estimates, not bounds: a feature that none of the three sets crosses is missed.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.fft import dct
from scipy.linalg import qr, solve
from scipy.stats import qmc

from src.structure.schemes.chebtree import cheb_points

LATTICE = (
    1 << 12
)  # the finest degree any axis may reach (a power of two; degrees are 8 * 2^m)


def maxvol(A: np.ndarray, tol: float = 1.05, max_iters: int = 200) -> np.ndarray:
    """Rows of the tall matrix A (n x r) spanning a near-maximal-volume r x r submatrix (Goreinov et al.)."""
    n, r = A.shape
    if n <= r:
        return np.arange(n)
    _, _, piv = qr(A.T, pivoting=True, mode="economic")
    idx = piv[:r].copy()
    B = solve(
        A[idx].T, A.T
    ).T  # A A[idx]^-1, the coefficients of every row in the chosen rows
    for _ in range(max_iters):
        i, j = np.unravel_index(np.argmax(np.abs(B)), B.shape)
        if abs(B[i, j]) <= tol:
            break
        idx[j] = i
        bj, bi = B[:, j].copy(), B[i].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / B[i, j])
    return idx


def _coefficients_along(G: np.ndarray, axis: int) -> np.ndarray:
    """Chebyshev coefficients along ``axis`` of node values on the second-kind grid."""
    n = G.shape[axis] - 1
    if n == 0:
        return G.copy()
    a = dct(G, type=1, axis=axis) / n
    edge = [slice(None)] * G.ndim
    for j in (0, n):
        edge[axis] = j
        a[tuple(edge)] /= 2.0
    return a


def chebyshev_tail(v: np.ndarray) -> float:
    """
    Twice the sum of |coefficients| in the top quarter of the degrees of 1-D node values (interpolation
    folds the omitted tail back once). Not the upper half: for geometric decay the error is about the
    last coefficient, a_n ~ a_{n/2}^2 / a_0, so asking the upper half to meet a share asks for about the
    square of the accuracy, and doubling overshoots further.
    """
    a = np.abs(_coefficients_along(v, 0))
    n = len(a) - 1
    return 2.0 * float(a[max(1, n - n // 4 + 1) :].sum())


@dataclass
class ChebTT:
    lo: np.ndarray
    hi: np.ndarray
    cores: list  # per axis, (r_k, n_k + 1, r_k+1) Chebyshev coefficients along the middle axis
    evals: int = 0
    ranks: tuple = ()
    degrees: tuple = ()
    breaks: (
        tuple
    ) = ()  # per axis, its interior breakpoints (empty: one piece); () for none on any axis
    bands: (
        tuple
    ) = ()  # per axis, each breakpoint's band: the model is not claimed within it

    def pieces(self, k: int) -> int:
        return len(self.breaks[k]) + 1 if self.breaks else 1

    def predict(self, X: np.ndarray, chunk: int = 16_384) -> np.ndarray:
        """In chunks that bound the temporaries (``contract``), over the active axes only (``active``)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        axes, model, const = self.active
        if model is None:
            return np.full(len(X), const)
        return np.concatenate(
            [
                model.contract(model.bases(X[s : s + chunk][:, axes]))
                for s in range(0, len(X), chunk)
            ]
        )

    @cached_property
    def active(self) -> tuple:
        """
        ``(axes, model, constant)``: the axes whose cores depend on their coordinate, and the same model
        as a train over those axes alone. A core of degree 0 (one piece) is a constant matrix C_0, so it
        is multiplied into its left neighbour (or, before the first active axis, into the next one) and
        dropped: exact, and a query then pays only for its active axes. A function of few of the inputs
        (each output of a gradient of a sum of local terms, a species coupled to few others) keeps a
        handful of its d cores. With no active axis, ``model`` is None and ``constant`` the model's value.
        """
        axes, cores, pending = [], [], None
        for k, C in enumerate(self.cores):
            if C.shape[1] == 1 and self.pieces(k) == 1:
                M = C[:, 0, :]
                if cores:
                    cores[-1] = np.einsum("ajb,bc->ajc", cores[-1], M)
                else:
                    pending = M if pending is None else pending @ M
                continue
            if pending is not None:
                C, pending = np.einsum("ab,bjc->ajc", pending, C), None
            axes.append(k)
            cores.append(C)
        if not cores:
            return [], None, float(pending[0, 0])
        if len(axes) == len(self.cores):
            return axes, self, 0.0
        breaks = tuple(self.breaks[k] for k in axes) if self.breaks else ()
        bands = tuple(self.bands[k] for k in axes) if self.bands else ()
        return (
            axes,
            ChebTT(self.lo[axes], self.hi[axes], cores, breaks=breaks, bands=bands),
            0.0,
        )

    def bases(self, X: np.ndarray) -> list:
        """
        Per axis, the basis values at the rows of X (``contract``'s input): the Chebyshev values of the
        axis, or on an axis with breakpoints those of each query's own piece in its block of rows, zero
        in the other pieces' (one product per core still serves every query).
        """
        sizes = [C.shape[1] for C in self.cores]
        if not any(len(b) for b in self.breaks):
            return chebyshev_bases(self.local(X), sizes)
        return [self.axis_values(X, k, n1) for k, n1 in enumerate(sizes)]

    def axis_values(self, X: np.ndarray, k: int, n1: int) -> np.ndarray:
        """Axis k's entry of ``bases`` (n1, Q), for a core of ``n1`` rows (``n1 / pieces`` per piece on an axis with breakpoints)."""
        if not self.breaks or not len(self.breaks[k]):
            return chebyshev_bases(
                2.0 * (X[:, k : k + 1] - self.lo[k]) / (self.hi[k] - self.lo[k]) - 1.0,
                [n1],
            )[0]
        a, b = piece_bounds(
            self.lo[k],
            self.hi[k],
            self.breaks[k],
            self.bands[k] if self.bands else None,
        )
        m1 = n1 // len(a)
        p = np.clip(
            np.searchsorted(self.breaks[k], X[:, k], side="right"), 0, len(a) - 1
        )
        T = np.zeros((n1, len(X)))
        local = chebyshev_bases(
            (2.0 * (X[:, k] - a[p]) / (b[p] - a[p]) - 1.0)[:, None], [m1]
        )[0]
        T[p[None, :] * m1 + np.arange(m1)[:, None], np.arange(len(X))[None, :]] = local
        return T

    def local(self, X: np.ndarray) -> np.ndarray:
        """X in the cores' coordinates, [-1, 1] per axis."""
        return 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0

    def contract(self, T: list) -> np.ndarray:
        """
        The model at the points whose Chebyshev values are ``T`` (``chebyshev_bases``; at least ``n_k + 1``
        rows per axis, so models of different degrees can share one set). Per core, the values times the
        core first, one matrix product (Q, n + 1) x (n + 1, r_k r_k+1), then the running row vector
        times that: about ``sum_k r_k (n_k + 1) r_k+1`` multiply-adds per point, 1.6-4x faster than
        multiplying the row vector into the core first, which leaves a strided (Q, n + 1, r) temporary.
        """
        v = np.ones((T[0].shape[1], 1))
        for C, Tk in zip(self.cores, T, strict=True):
            v = advance(v, C, Tk)
        return v[:, 0]

    def environments(self, T: list) -> tuple[list, list]:
        """
        Per core k, the train's product left of it, (Q, r_k), and right of it, (Q, r_k+1), at the points
        of ``T``: the model with only core k changed to C' is ``through(left[k], C', T[k], right[k])``, one
        core's work instead of the whole train's. ``left[d]`` holds the model itself (as ``contract``'s
        bits: the same steps), as a (Q, 1) column.
        """
        left = [np.ones((T[0].shape[1], 1))]
        for C, Tk in zip(self.cores, T, strict=True):
            left.append(advance(left[-1], C, Tk))
        right = [np.ones((T[0].shape[1], 1))]
        for C, Tk in zip(self.cores[:0:-1], T[:0:-1], strict=True):
            right.append(np.einsum("qab,qb->qa", core_values(C, Tk), right[-1]))
        return left, right[::-1]

    def leaf_of(self, X):
        return None


def core_values(C: np.ndarray, Tk: np.ndarray) -> np.ndarray:
    """(Q, r_k, r_k+1): core C at the points whose Chebyshev values on its axis are ``Tk`` (``contract``'s per-core matrix)."""
    ra, n1, rb = C.shape
    return (Tk[:n1].T @ C.transpose(1, 0, 2).reshape(n1, ra * rb)).reshape(-1, ra, rb)


def advance(v: np.ndarray, C: np.ndarray, Tk: np.ndarray) -> np.ndarray:
    """The row vectors v (Q, r_k) carried through core C: one step of ``contract``."""
    if (
        C.shape[0] == 1
    ):  # every first core, every core of a rank-1 train: a scaling, without einsum's overhead
        return v * (C[0].T @ Tk[: C.shape[1]]).T
    return np.einsum("qa,qab->qb", v, core_values(C, Tk))


def through(
    left: np.ndarray, C: np.ndarray, Tk: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """The train at Q points with core k replaced by C, from its environments (``ChebTT.environments``)."""
    return np.einsum("qb,qb->q", advance(left, C, Tk), right)


INSET = 1e-10  # a piece's end nodes sit this share of the axis inside an interior breakpoint, so a jump is never sampled on the wrong side


def piece_bounds(
    lo: float, hi: float, breaks, bands=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    The intervals [a_p, b_p] that carry each piece's Chebyshev nodes: [lo, hi] cut at ``breaks``, inset at
    each interior breakpoint by ``INSET`` of the width or twice its band (``axis_edges``), whichever is
    larger, so that no node falls on the wrong side of an edge that is located only to within its band.
    """
    e = np.concatenate([[lo], np.asarray(breaks, dtype=np.float64), [hi]])
    w = (hi - lo) * INSET
    inset = (
        np.full(len(e) - 2, w)
        if bands is None
        else np.maximum(w, 2.0 * np.asarray(bands, dtype=np.float64))
    )
    if len(
        inset
    ):  # never more than a quarter of either neighbouring piece, so no piece can invert
        inset = np.minimum(inset, 0.25 * np.minimum(np.diff(e)[:-1], np.diff(e)[1:]))
    a, b = e[:-1].copy(), e[1:].copy()
    a[1:] += inset
    b[:-1] -= inset
    return a, b


def chebyshev_bases(t: np.ndarray, sizes: list) -> list:
    """Per axis k, the (sizes[k], Q) values T_j(t[:, k]) by the three-term recurrence (no arccos or cos), rows contiguous."""
    out = []
    for k, n1 in enumerate(sizes):
        T = np.empty((n1, len(t)))
        T[0] = 1.0
        if n1 > 1:
            x = np.ascontiguousarray(t[:, k])
            T[1] = x
        for j in range(2, n1):
            np.multiply(x, T[j - 1], out=T[j])
            T[j] *= 2.0
            T[j] -= T[j - 2]
        out.append(T)
    return out


class KeyStore:
    """
    Values of f by integer lattice key (one row of d int32 per point), in insertion order. Lookup is
    vectorised: a 64-bit hash of each row, searched in a few sorted runs (merged when a run grows to
    half of the one before it, so there are O(log N) of them), and every hit checked against the full
    key, so a hash collision raises instead of returning another point's value.
    """

    def __init__(self, d: int):
        self.d = d
        # independent odd multipliers per column (a column that reused another's, scaled, would let key
        # differences cancel exactly), each column mixed nonlinearly before the sum
        self.mul = np.random.default_rng(20_261_118).integers(
            1, 2**63, size=d, dtype=np.uint64
        ) | np.uint64(1)
        self.keys = np.empty((1024, d), dtype=np.int32)
        self.vals = np.empty(1024)
        self.size = 0
        self.runs: list = []  # (sorted hashes, their positions in keys)

    def __len__(self) -> int:
        return self.size

    def _hash(self, K: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            z = (K.astype(np.uint64) + np.uint64(1)) * self.mul
            z ^= z >> np.uint64(29)
            z *= np.uint64(0xBF58476D1CE4E5B9)
            z ^= z >> np.uint64(32)
            h = z.sum(axis=1, dtype=np.uint64)
            h ^= h >> np.uint64(31)
            h *= np.uint64(0x9E3779B97F4A7C15)
            h ^= h >> np.uint64(29)
        return h

    def find(self, K: np.ndarray) -> np.ndarray:
        """Each row's position in the store, -1 where it is not stored."""
        h = self._hash(K)
        pos = np.full(len(K), -1, dtype=np.int64)
        for hs, ps in self.runs:
            i = np.searchsorted(hs, h)
            ok = (i < len(hs)) & (pos < 0)
            ok[ok] = hs[i[ok]] == h[ok]
            pos[ok] = ps[i[ok]]
        hit = pos >= 0
        if hit.any() and not np.array_equal(self.keys[pos[hit]], K[hit]):
            raise RuntimeError(
                "KeyStore: a 64-bit hash collision between two lattice keys"
            )
        return pos

    def add(self, K: np.ndarray, V: np.ndarray) -> None:
        """Append rows K (none of them stored yet, no duplicates) with values V."""
        n = len(K)
        if self.size + n > len(self.vals):
            cap = max(2 * len(self.vals), self.size + n)
            keys, vals = np.empty((cap, self.d), dtype=np.int32), np.empty(cap)
            keys[: self.size], vals[: self.size] = (
                self.keys[: self.size],
                self.vals[: self.size],
            )
            self.keys, self.vals = keys, vals
        self.keys[self.size : self.size + n], self.vals[self.size : self.size + n] = (
            K,
            V,
        )
        h = self._hash(K)
        order = np.argsort(h, kind="stable")
        self.runs.append((h[order], np.arange(self.size, self.size + n)[order]))
        self.size += n
        while len(self.runs) > 1 and 2 * len(self.runs[-1][0]) >= len(self.runs[-2][0]):
            (h1, p1), (h2, p2) = self.runs[-2], self.runs.pop()
            hh, pp = np.concatenate([h1, h2]), np.concatenate([p1, p2])
            order = np.argsort(hh, kind="stable")
            self.runs[-1] = (hh[order], pp[order])


class _Grid:
    """
    Node coordinates per axis (the pieces' Chebyshev points, one block per piece) and a cache of f by
    grid key. A node's key on axis k is ``piece * (LATTICE + 1) + j * LATTICE / n_k``: node j of degree
    n is node 2j of degree 2n, so a doubled axis finds its old points again. Values cached under earlier
    breakpoints are kept in ``history``, as free check points.
    """

    def __init__(self, f, lo, hi, n, breaks=None, bands=None):
        self.f, self.lo, self.hi = f, lo, hi
        self.breaks = (
            [np.asarray(b, dtype=np.float64) for b in breaks]
            if breaks
            else [np.zeros(0) for _ in lo]
        )
        self.bands = (
            [np.asarray(b, dtype=np.float64) for b in bands]
            if bands
            else [np.zeros(len(b)) for b in self.breaks]
        )
        self.set_degrees(n)
        self.cache = KeyStore(len(lo))
        self.history: list = []
        self.evals = 0

    def set_degrees(self, n):
        self.n = list(n)
        self.bounds = [
            piece_bounds(self.lo[k], self.hi[k], self.breaks[k], self.bands[k])
            for k in range(len(self.n))
        ]
        self.nodes = [np.concatenate([a + (cheb_points(nk) + 1.0) * 0.5 * (b - a) for a, b in zip(*self.bounds[k], strict=True)])
                      for k, nk in enumerate(self.n)]  # fmt: skip
        self.count = [len(x) for x in self.nodes]  # nodes per axis

    def set_breaks(self, k: int, breaks, bands) -> None:
        """New breakpoints (and their bands) on axis k: every cached value moves to ``history`` (its key no longer names its point)."""
        X, v = self.sample(len(self.cache), None)
        if len(v):
            self.history.append((X, v))
        self.cache = KeyStore(len(self.n))
        order = np.argsort(breaks)
        self.breaks[k] = np.asarray(breaks, dtype=np.float64)[order]
        self.bands[k] = np.asarray(bands, dtype=np.float64)[order]
        self.set_degrees(self.n)

    def sample(self, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
        """Up to n evaluated points (cached and earlier ones; all of them when there are fewer) and their values."""
        M = self.cache.keys[: len(self.cache)].astype(np.float64)
        X = np.empty_like(M)
        for k in range(len(self.n)):
            if not len(self.breaks[k]):
                X[:, k] = self.lo[k] + (
                    np.cos(np.pi * M[:, k] / LATTICE) + 1.0
                ) * 0.5 * (self.hi[k] - self.lo[k])
            else:
                p, m = np.divmod(M[:, k], LATTICE + 1)
                a, b = self.bounds[k]
                p = p.astype(np.int64)
                X[:, k] = a[p] + (np.cos(np.pi * m / LATTICE) + 1.0) * 0.5 * (
                    b[p] - a[p]
                )
        V = self.cache.vals[: len(self.cache)].copy()
        if self.history:
            X = np.concatenate([X] + [h[0] for h in self.history])
            V = np.concatenate([V] + [h[1] for h in self.history])
        if rng is not None and len(V) > n:
            pick = rng.choice(len(V), n, replace=False)
            X, V = X[pick], V[pick]
        return X, V

    def values(self, idx: np.ndarray, max_evals: int) -> np.ndarray:
        """f at the grid points with integer multi-indices ``idx`` (M, d), evaluating only the uncached ones."""
        step = np.array([LATTICE // nk for nk in self.n])
        piece, j = np.divmod(idx, np.array(self.n) + 1)
        K = (piece * (LATTICE + 1) + j * step).astype(np.int32)
        pos = self.cache.find(K)
        miss = pos < 0
        if miss.any():
            missing = np.unique(
                K[miss], axis=0
            )  # lexicographic: the order a sorted set of key tuples has
            if self.evals + len(missing) > max_evals:
                raise RuntimeError(f"ChebTT would exceed {max_evals:,} evaluations.")
            Mp, Mj = np.divmod(missing.astype(np.int64), LATTICE + 1)
            M = Mp * (np.array(self.n) + 1) + Mj // step
            X = np.stack([self.nodes[k][M[:, k]] for k in range(len(self.n))], axis=1)
            vals = np.asarray(self.f(X), dtype=np.float64).ravel()
            self.evals += len(missing)
            self.cache.add(missing, vals)
            pos[miss] = self.cache.find(K[miss])
        return self.cache.vals[pos]


def _fibre_degree(
    f, lo, hi, k, share, rng, n_fibre, n_max, n0=8, pause=None, base=None
):
    """
    The smallest nested degree at which random fibres along axis k have Chebyshev tails below
    ``share``, the points used, the fibres' base points, and whether the tails converged. With ``pause``,
    stop unconverged at that degree; calling again with ``n0 = 2 * pause`` and the same ``base`` resumes
    exactly where it stopped.
    """
    d = len(lo)
    base = rng.uniform(lo, hi, (n_fibre, d)) if base is None else base
    n, used = n0, 0
    while True:
        t = cheb_points(n)
        P = np.repeat(base, n + 1, axis=0)
        P[:, k] = np.tile(lo[k] + (t + 1.0) * 0.5 * (hi[k] - lo[k]), n_fibre)
        V = np.asarray(f(P), dtype=np.float64).reshape(n_fibre, n + 1)
        used += len(P)
        converged = max(chebyshev_tail(v) for v in V) <= share
        if converged or 2 * n > n_max or (pause is not None and n >= pause):
            return n, used, base, converged
        n *= 2


def _fibre(f, base: np.ndarray, k: int):
    """g(t) = f at ``base`` with coordinate k set to t (a 1-D slice of f)."""

    def g(t):
        P = np.repeat(base[None], len(t), axis=0)
        P[:, k] = t
        return np.asarray(f(P), dtype=np.float64).ravel()

    return g


def locate_edge(
    g, a: float, b: float, jump: bool, width: float, m: int = 9
) -> tuple[float, float, int]:
    """
    A jump in g (``jump``) or in g' inside [a, b], to ``width``; returns the location, the bracket width
    reached and the evaluations. A jump is bisected on the value (the half whose ends differ more holds
    it): one evaluation per halving. A kink is zoomed into: the cell of largest second difference on an
    m-point grid and its neighbours (chebfun's ``detectedge`` idea; the edge's second difference is
    O(h |[g']|) against O(h^2 |g''|) elsewhere, so it wins at every zoom), m evaluations per ~(m-1)/2x;
    bisecting a kink by midpoint deviations can pick the wrong half when it sits near a quarter point.
    """
    used = 0
    if jump:
        ga, gb = g(np.array([a, b]))
        used += 2
        while b - a > width:
            mid = 0.5 * (a + b)
            gm = g(np.array([mid]))[0]
            used += 1
            if abs(gm - ga) <= abs(gb - gm):
                a, ga = mid, gm
            else:
                b, gb = mid, gm
        return 0.5 * (a + b), b - a, used
    while b - a > width:
        t = np.linspace(a, b, m)
        v = g(t)
        used += m
        c = (
            int(np.argmax(np.abs(v[2:] - 2.0 * v[1:-1] + v[:-2]))) + 1
        )  # the edge is within a cell of t[c]
        a, b = t[max(c - 1, 0)], t[min(c + 1, m - 1)]
    return 0.5 * (a + b), b - a, used


def edge_at(g, c: float, delta: float, jump: bool) -> bool:
    """
    Whether g has a jump (``jump``) or a kink at c, located to within ``delta``: the one-sided value
    difference (a jump: 2 evaluations) or slope difference (a kink: 4) keeps its size when the stencil
    shrinks from 100 delta to 4 delta; at a smooth point it shrinks with the stencil.
    """
    sizes = []
    for e in (100.0 * delta, 4.0 * delta):
        if jump:
            v = g(np.array([c - e, c + e]))
            sizes.append((abs(v[1] - v[0]), max(float(np.abs(v).max()), 1e-300)))
        else:
            v = g(np.array([c - 2 * e, c - e, c + e, c + 2 * e]))
            sizes.append(
                (
                    abs((v[3] - v[2]) - (v[1] - v[0])) / e,
                    max(float(np.abs(v).max()), 1e-300),
                )
            )
    (big, _), (small, scale) = sizes
    return bool(
        small > 1e-9 * scale * (1.0 if jump else 1.0 / delta)
        and 0.5 < small / max(big, 1e-300) < 2.0
    )


def axis_edges(f, lo, hi, k: int, bases: np.ndarray, known, rng, share: float = 0.0, m: int = 65, max_edges: int = 3,
               n_confirm: int | None = None, band: float = 1e-6, bands: list | None = None,
               oblique: list | None = None) -> tuple[list, int]:  # fmt: skip
    """
    Axis-aligned jumps and kinks on axis k, and the evaluations spent. Along the fibre through the
    first of ``bases`` (the next only if it finds nothing), the cells of a 65-point scan whose second
    difference is an outlier are located (``locate_edge``: a jump to 1e-12 of the width, a kink only as
    finely as the tolerance needs, ``0.01 share / |[f']|``) and tested (``edge_at``). A candidate is
    kept as an axis-aligned edge when, at ``n_confirm`` (default 2 d) random far bases, the edge is
    present within ``band`` of the width of the same place or absent (f smooth there, the edge's factor
    vanishing), and present at one at least; a base whose fibre has an edge further away means an
    oblique interface, and rejects it. Each kept edge's band (its largest displacement plus its
    location error) is appended to ``bands``: the model is not claimed within that distance of it. A
    rejected (oblique) jump adds its two located positions, as points, to ``oblique``: they lie on the
    interface, for a caller that models interfaces.
    Edges within 1e-6 of the width of ``known`` ones are skipped.
    """
    d = len(lo)
    w = hi[k] - lo[k]
    t = np.linspace(lo[k], hi[k], m)
    used, found = 0, []
    bands = [] if bands is None else bands
    for base in bases:
        g = _fibre(f, base, k)
        v = g(t)
        used += m
        d1, d2 = np.abs(np.diff(v)), np.abs(v[2:] - 2.0 * v[1:-1] + v[:-2])
        floor = 8.0 * np.median(d2) + 1e-14 * max(float(np.abs(v).max()), 1e-300)
        for c in np.argsort(d2)[::-1][:max_edges]:
            if d2[c] <= floor:
                break
            # A jump first (bisection on the value, ~34 evaluations), else a kink: the scan cannot tell
            # them apart, since a steep kink's slope is an outlier against a tiny median difference.
            cell = c + int(np.argmax(d1[c : c + 2]))
            x0, delta, u = locate_edge(g, t[cell], t[cell + 1], True, 1e-12 * w)
            used += u + 2
            jump = edge_at(g, x0, max(delta, 1e-13 * w), True)
            if not jump:
                slope_jump = d2[c] / (t[1] - t[0])
                width = (
                    min(
                        max(1e-12 * w, 0.01 * share / max(slope_jump, 1e-300)), 1e-4 * w
                    )
                    if share > 0
                    else 1e-12 * w
                )
                x0, delta, u = locate_edge(
                    g, t[max(c - 1, 0)], t[min(c + 3, m - 1)], False, width
                )
                used += u + 8
                if not edge_at(g, x0, max(delta, 1e-13 * w), False):
                    continue
            # the same edge again (a jump makes two neighbouring cells outliers, and a kink search there
            # finds the jump, coarsely): within 10 times this location's precision of a known one
            near = max(1e-6 * w, 10.0 * delta)
            if not (lo[k] + 1e-6 * w < x0 < hi[k] - 1e-6 * w) or any(
                abs(x0 - y) <= near for y in list(known) + found
            ):
                continue
            present, tilted, shift = (
                1,
                False,
                0.0,
            )  # the fibre that found it; the far ones may only lack it (its factor vanishing there)
            for _ in range(n_confirm or 2 * d):
                far = rng.uniform(lo, hi)
                gf = _fibre(f, far, k)
                ok = edge_at(gf, x0, max(delta, 1e-13 * w), jump)
                used += 2 if jump else 8
                if ok:
                    present += 1
                    continue
                # Not at x0 to within delta: nearby (an edge tilted by less than ``band`` of the width,
                # kept with that band reported), absent (the edge's factor vanishes here), or far away
                # (an oblique interface: rejected).
                near = band * w
                y0, dy, u = locate_edge(
                    gf, x0 - near, x0 + near, jump, max(delta, 1e-13 * w)
                )
                used += u
                if edge_at(gf, y0, max(dy, 1e-13 * w), jump):
                    used += 2 if jump else 8
                    present += 1
                    shift = max(shift, abs(y0 - x0))
                    continue
                vf = gf(t)
                used += m
                d2f = np.abs(vf[2:] - 2.0 * vf[1:-1] + vf[:-2])
                if d2f.max() > 8.0 * np.median(d2f) + 1e-14 * max(
                    float(np.abs(vf).max()), 1e-300
                ):
                    tilted = True
                    if oblique is not None and jump:
                        cf = int(np.argmax(d2f))
                        cell = cf + int(np.argmax(np.abs(np.diff(vf))[cf : cf + 2]))
                        y0, dy, u = locate_edge(
                            gf, t[cell], t[cell + 1], True, 1e-12 * w
                        )
                        used += u + 2
                        if edge_at(gf, y0, max(dy, 1e-13 * w), True):
                            p0, p1 = base.copy(), far.copy()
                            p0[k], p1[k] = x0, y0
                            oblique += [p0, p1]
                    break
            if present and not tilted:
                found.append(float(x0))
                bands.append(shift + max(delta, 1e-13 * w))
        if found:
            break
    return found, used


def _algebraic(C: np.ndarray, pieces: int) -> bool:
    """
    Whether an axis's coefficients decay algebraically (a kink: ~ j^-2, a jump: ~ j^-1), read off its
    core: the upper half of the degrees holds at least 15% of the mean size of the quarter below it in
    some piece. A resolved smooth axis decays geometrically, far below; an unresolved smooth one may
    pass, and ``is_singular`` then turns its candidate edges down.
    """
    n1 = C.shape[1] // pieces
    if n1 < 16:
        return False
    A = np.abs(C).reshape(C.shape[0], pieces, n1, C.shape[2]).mean(axis=(0, 3))
    upper, lower = A[:, n1 // 2 :].mean(axis=1), A[:, n1 // 4 : n1 // 2].mean(axis=1)
    return bool(np.any(upper >= 0.15 * lower) and np.any(upper > 1e-14 * A.max()))


class ObliqueEdge(RuntimeError):
    """The edge search met a jump across an oblique interface (``build``'s ``interface`` holds points on it); the argument is the evaluations spent."""


class RankLimit(RuntimeError):
    """The cross reached ``rank_max`` without meeting the tolerance (as opposed to running out of evaluations)."""


def check_points(lo, hi, n: int, seed: int) -> np.ndarray:
    """A scrambled Sobol set of n points in [lo, hi]."""
    d = len(lo)
    return lo + qmc.Sobol(d, scramble=True, seed=seed).random(n) * (hi - lo)


def build(f, lo, hi, atol: float, safety: float = 0.5, rank0: int = 2, rank_step: int = 2, rank_max: int = 60,
          sweeps: int = 2, n_fibre: int = 4, n_max: int = 512, n_check: int | None = None, n_free: int = 20_000, chop_share: float = 0.25, n_probe: int = 4000,
          seed: int = 0, max_evals: int = 2_000_000, edges: bool = True, max_pieces: int = 4,
          interface: list | None = None) -> ChebTT:  # fmt: skip
    """
    Train a ChebTT of ``f`` (host (N, d) -> (N,)) on [lo, hi] to tolerance ``atol``. With ``edges``, an
    axis that must refine and whose coefficients decay algebraically is first searched for jumps and
    kinks (``axis_edges``); the ones found become breakpoints of that axis (at most ``max_pieces``
    pieces, one degree for all of them) instead of a doubled degree. With ``interface`` (a list), the
    build stops with ``ObliqueEdge`` as soon as the edge search meets a jump across an oblique interface,
    which no breakpoint represents, after adding the points located on it to that list.
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = len(lo)
    rng = np.random.default_rng(seed)
    share = safety * atol / d

    pre = 0
    n = []
    Xc = check_points(lo, hi, n_check or 64 + 32 * d, seed)
    fc = np.asarray(f(Xc), dtype=np.float64).ravel()
    pre += len(Xc)
    # Edge search draws from its own stream, so a function without edges gets exactly the model it
    # would without the search (only its evaluation count differs).
    edge_rng = np.random.default_rng(seed + 303)
    searched: set = (
        set()
    )  # (axis, degree, pieces) already searched for edges in the loop
    breaks, bands = [np.zeros(0) for _ in range(d)], [np.zeros(0) for _ in range(d)]
    for k in range(d):
        # An axis still unconverged at degree 32 is searched for edges through the largest |f| seen
        # before its degree grows further: a kink would otherwise take it to n_max at once.
        nk, used, base, ok = _fibre_degree(
            f, lo, hi, k, share, rng, n_fibre, n_max, pause=32 if edges else None
        )
        pre += used
        if edges and not ok and 2 * nk <= n_max:
            bk = []
            found, used = axis_edges(
                f,
                lo,
                hi,
                k,
                Xc[[int(np.argmax(np.abs(fc)))]],
                [],
                edge_rng,
                share,
                bands=bk,
                oblique=interface,
            )
            pre += used
            if interface:
                raise ObliqueEdge(pre)
            if found:
                keep = np.argsort(found)[: max_pieces - 1]
                breaks[k], bands[k], nk = (
                    np.asarray(found)[keep],
                    np.asarray(bk)[keep],
                    16,
                )
            else:
                nk, used, _, _ = _fibre_degree(
                    f, lo, hi, k, share, rng, n_fibre, n_max, n0=2 * nk, base=base
                )
                pre += used
        n.append(nk)
    grid = _Grid(f, lo, hi, n, breaks, bands)
    r = rank0
    # Seed the right sets with the grid node nearest the largest |f| seen so far and the centre node, then
    # random nodes: maxvol only moves among the fibres it is shown, so fibres that all miss a narrow peak
    # (every value near 0) would converge to a model of the empty background.
    peak = np.array(
        [
            np.argmin(np.abs(grid.nodes[k] - Xc[np.argmax(np.abs(fc)), k]))
            for k in range(d)
        ]
    )
    centre = np.array([nk // 2 for nk in n])
    right = [None] + [np.vstack([peak[k + 1 :], centre[k + 1 :], rng.integers(0, np.array(n[k + 1 :]) + 1, size=(max(r - 2, 0), d - k - 1))])[:max(r, 2)]
                      for k in range(d - 1)] + [np.zeros((1, 0), dtype=np.int64)]  # fmt: skip
    while True:
        left, cores = _sweeps(grid, right, sweeps, max_evals - pre)
        pieced = any(len(b) for b in grid.breaks)
        breaks, bands = (tuple(grid.breaks), tuple(grid.bands)) if pieced else ((), ())
        model = ChebTT(lo, hi, [_piece_coefficients(C, len(grid.breaks[k]) + 1) for k, C in enumerate(cores)], grid.evals + pre,
                       tuple(C.shape[2] for C in cores[:-1]), tuple(grid.n), breaks, bands)  # fmt: skip
        # Free checks: grid points evaluated in earlier sweeps or at lower degrees, which the final
        # cores do not interpolate (they interpolate the last sweep's fibres only), concentrated where
        # the cross explored, i.e. where f matters.
        Xg, fg = grid.sample(n_free, rng)
        Xall, fall = np.concatenate([Xc, Xg]), np.concatenate([fc, fg])
        # One set of environments on the check points serves the round: the model itself (envl[d]) and,
        # if the round fails, every axis's degree tail at one core's cost.
        TX = model.bases(Xall)
        envl, envr = model.environments(TX)
        err = np.abs(envl[d][:, 0] - fall)
        # Fibre look-ahead: the next nested grid's new points along each axis, through the worst check
        # point and through the largest |f| seen (where a narrow feature is), the points a refinement
        # would sample next. A line along axis k shares every other coordinate with its base, so the
        # model on it is the base's environments around core k at the line's values: two bases'
        # environments serve every axis, instead of a prediction of the whole train per axis.
        bases = Xall[[int(np.argmax(err)), int(np.argmax(np.abs(fall)))]]
        bl, br = model.environments(model.bases(bases))
        look = 0.0
        for k in range(d):
            nk = grid.n[k]
            t = np.cos(np.pi * (np.arange(nk) + 0.5) / nk)
            a, b = grid.bounds[k]
            line = np.concatenate(
                [
                    a_p + (t + 1.0) * 0.5 * (b_p - a_p)
                    for a_p, b_p in zip(a, b, strict=True)
                ]
            )
            P = np.repeat(bases, len(line), axis=0)
            P[:, k] = np.tile(line, len(bases))
            fP = np.asarray(f(P), dtype=np.float64).ravel()
            pre += len(P)
            rep = np.repeat(np.arange(len(bases)), len(line))
            C = model.cores[k]
            look = max(
                look,
                float(
                    np.max(
                        np.abs(
                            through(
                                bl[k][rep],
                                C,
                                model.axis_values(P, k, C.shape[1]),
                                br[k][rep],
                            )
                            - fP
                        )
                    )
                ),
            )
            Xc, fc = np.concatenate([Xc, P]), np.concatenate(
                [fc, fP]
            )  # kept: later rounds are checked there too
        model.evals = grid.evals + pre
        if max(float(err.max()), look) <= safety * atol:
            probe = np.concatenate([Xall, rng.uniform(lo, hi, (n_probe, d))])
            small = compress(model, probe, chop_share * atol)
            small.evals = model.evals
            return small
        if model.evals > max_evals:
            raise RuntimeError(f"ChebTT would exceed {max_evals:,} evaluations.")
        # Degree or rank: an axis whose top quarter of degrees still changes the model by more than its
        # share at the check points is under-resolved, and no rank can make up for it.
        tails = {
            k: float(
                np.max(np.abs(through(envl[k], _dropped(model, k), TX[k], envr[k])))
            )
            for k in range(d)
        }
        grow = [k for k in range(d) if tails[k] > share and 2 * grid.n[k] <= n_max]
        split = {}
        # Axes to search: those that must refine; and, when none must (the rank would grow next), every
        # axis whose coefficients still decay algebraically, since no rank resolves a kink whose
        # top-quarter tail happens to look small. Each axis is searched once per degree and piece count.
        candidates = grow if grow else range(d)
        for k in candidates if edges else ():
            state = (k, grid.n[k], len(grid.breaks[k]))
            if state in searched:
                continue
            if len(grid.breaks[k]) + 1 < max_pieces and _algebraic(
                model.cores[k], len(grid.breaks[k]) + 1
            ):
                searched.add(state)
                bk = []
                found, used = axis_edges(
                    f,
                    lo,
                    hi,
                    k,
                    bases,
                    grid.breaks[k],
                    edge_rng,
                    share,
                    bands=bk,
                    oblique=interface,
                )
                pre += used
                if interface:
                    raise ObliqueEdge(grid.evals + pre)
                if found:
                    allb, alls = list(grid.breaks[k]) + found, list(grid.bands[k]) + bk
                    keep = np.argsort(allb)[: max_pieces - 1]
                    split[k] = (np.asarray(allb)[keep], np.asarray(alls)[keep])
        if (
            split
        ):  # breakpoints instead of a doubled degree on those axes; the right sets move to the nearest new nodes
            old = [x.copy() for x in grid.nodes]
            for k, (b, bb) in split.items():
                grid.set_breaks(k, b, bb)
            grow = [k for k in grow if k not in split]
            grid.set_degrees(
                [2 * nk if k in grow else nk for k, nk in enumerate(grid.n)]
            )
            right = [None] + [np.stack([np.argmin(np.abs(grid.nodes[a][:, None] - old[a][R[:, i]][None]), axis=0)
                                        for i, a in enumerate(range(k + 1, d))], axis=1) for k, R in enumerate(right[1:-1])] + [right[-1]]  # fmt: skip
            continue
        if grow:
            old = [x.copy() for x in grid.nodes]
            grid.set_degrees(
                [2 * nk if k in grow else nk for k, nk in enumerate(grid.n)]
            )
            if not any(
                len(grid.breaks[a]) for a in grow
            ):  # one piece: node j of degree n is node 2j of degree 2n
                right = (
                    [None]
                    + [
                        R * (np.array([2 if a in grow else 1 for a in range(k + 1, d)]))
                        for k, R in enumerate(right[1:-1])
                    ]
                    + [right[-1]]
                )
            else:  # pieces: the same coordinates, by nearest node
                right = [None] + [np.stack([np.argmin(np.abs(grid.nodes[a][:, None] - old[a][R[:, i]][None]), axis=0)
                                            for i, a in enumerate(range(k + 1, d))], axis=1) for k, R in enumerate(right[1:-1])] + [right[-1]]  # fmt: skip
            continue
        r = min(r + rank_step, rank_max)
        if all(R.shape[0] >= r for R in right[1:-1]) and r == rank_max:
            raise RankLimit(
                f"ChebTT reached rank {rank_max} without meeting the tolerance."
            )
        right = [None] + [np.vstack([R, rng.integers(0, np.array(grid.count[k + 1 :]), size=(rank_step, d - k - 1))])
                          for k, R in enumerate(right[1:-1])] + [right[-1]]  # fmt: skip


class _SharedOracle:
    """f (host (N, d) -> (N, m)) behind one cache keyed by the point: one call gives every output, so separate trains of the outputs pay once for any point one of them asks for."""

    def __init__(self, f):
        self.f, self.cache = f, {}

    @property
    def evals(self) -> int:
        return len(self.cache)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float64)
        keys = [x.tobytes() for x in X]
        first: dict[bytes, int] = {}
        for i, key in enumerate(keys):
            if key not in self.cache and key not in first:
                first[key] = i
        if first:
            V = np.asarray(self.f(X[list(first.values())]), dtype=np.float64).reshape(
                len(first), -1
            )
            self.cache.update(zip(first, V, strict=True))
        return np.array([self.cache[key] for key in keys])


def _directions_holding(w: np.ndarray, share: float = 0.99) -> int:
    """How many of the largest entries of w (>= 0) hold ``share`` of its sum."""
    w = np.sort(np.clip(w, 0.0, None))[::-1]
    return int(np.searchsorted(np.cumsum(w) / max(float(w.sum()), 1e-300), share) + 1)


def metric_groups(J: np.ndarray) -> list[tuple[list[int], np.ndarray | None]]:
    """
    Coordinates for each output from Jacobian samples ``J`` (d, points, m): an output whose metric
    ``G_c = mean grad f_c grad f_c^T`` holds 99% of its trace in at most half as many eigenvectors as
    axes is rotated; outputs whose leading subspaces merge without growing share one rotation (the
    eigenvectors of their summed, trace-normalised metrics, largest first); the rest keep the axes
    (``None``). A separable output is never rotated, since a rotation would destroy its rank 1.
    """
    d, _, m = J.shape
    G = [np.einsum("ip,jp->ij", J[:, :, c], J[:, :, c]) for c in range(m)]
    G = [g / max(float(np.trace(g)), 1e-300) for g in G]
    k_eig = [_directions_holding(np.linalg.eigvalsh(g)) for g in G]
    rotated: list[list[int]] = []
    axes = []
    for c in sorted(range(m), key=lambda c: k_eig[c]):
        if 2 * k_eig[c] > _directions_holding(np.diag(G[c])):
            axes.append(c)
            continue
        for grp in rotated:
            joint = _directions_holding(
                np.linalg.eigvalsh(sum(G[i] for i in grp) + G[c])
            )
            if joint <= max(
                _directions_holding(np.linalg.eigvalsh(sum(G[i] for i in grp))),
                k_eig[c],
            ):
                grp.append(c)
                break
        else:
            rotated.append([c])
    out: list[tuple[list[int], np.ndarray | None]] = []
    for grp in rotated:
        w, Q = np.linalg.eigh(sum(G[i] for i in grp))
        out.append((sorted(grp), Q[:, np.argsort(w)[::-1]]))
    if axes:
        out.append((sorted(axes), None))
    return out


@dataclass
class VectorTT:
    """
    Every output its own scalar ChebTT, in its group's coordinates ``u = (x - centre) Q`` (``Q`` None:
    the box's own axes). Predictions are (N, outputs).
    """

    groups: list  # (output indices, Q or None, their ChebTTs)
    centre: np.ndarray
    outputs: int
    evals: int = 0

    def predict(self, X: np.ndarray, chunk: int = 16_384) -> np.ndarray:
        """A group's outputs share its coordinates and their Chebyshev values (``ChebTT.contract``)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.empty((len(X), self.outputs))
        for cols, Q, models in self.groups:
            shared = not any(
                m.breaks for m in models
            )  # pieces differ between outputs: each its own bases
            sizes = [
                max(m.cores[k].shape[1] for m in models)
                for k in range(len(models[0].cores))
            ]
            for s in range(0, len(X), chunk):
                U = (
                    X[s : s + chunk]
                    if Q is None
                    else (X[s : s + chunk] - self.centre) @ Q
                )
                T = chebyshev_bases(models[0].local(U), sizes) if shared else None
                for c, model in zip(cols, models, strict=True):
                    axes, small, const = model.active
                    if small is None:
                        out[s : s + chunk, c] = const
                    elif (
                        shared
                    ):  # the shared values of its active axes (at least n_k + 1 rows each)
                        out[s : s + chunk, c] = small.contract([T[k] for k in axes])
                    else:
                        out[s : s + chunk, c] = small.contract(small.bases(U[:, axes]))
        return out


def build_vector(f, lo, hi, atol: float, outputs: int, rotate: bool = False, axes_rank: int = 6, seed: int = 0,
                 **kw) -> VectorTT:  # fmt: skip
    """
    A surrogate of a vector-valued ``f`` (host (N, d) -> (N, outputs)) within ``atol`` in every output:
    one scalar ChebTT per output, all drawing on one cache of f, so a point any train asks for serves
    the others (the degree pre-pass, the Sobol checks and the first fibres coincide, as every train uses
    ``seed``). ``evals`` counts the distinct points f saw.

    Separate trains, not one train with an output axis: f gives every output at every point without
    noise, so sharing a model cannot make an interpolant more accurate at a given point set, it can only
    save points; and one train's bond ranks are those of the union of the outputs' factor spaces, up to
    the sum of theirs, which its sampling pays squared. Outputs of different structure (a ridge beside
    a product peak) therefore cost several times more in one train than apart.

    ``rotate``: what the outputs do share is coordinates. Every output is first trained on the axes, as
    without ``rotate`` but with its rank capped at ``axes_rank``; an output that meets the tolerance there
    keeps that model, at no extra cost (on the axes, low rank leaves a rotation nothing to gain, and the
    rotated box is larger and shares no points with the axes). Only the outputs that reach the cap get a
    pilot: finite differences at ``4 + 2d`` (rounded up to a power of two) Sobol points give their
    metrics, and ``metric_groups`` decides which to train in rotated coordinates, on the box bounding the
    domain in them, and which to train on the axes again at full rank (from the cache, so the capped
    attempt's fibres cost nothing twice). The rotated box extends beyond the domain, so a group stays on
    the axes when f, at ``2d`` random points of its box, is not finite or exceeds ten times its largest
    value at the pilot points (an ODE that leaves its positive orthant, a polynomial that grows fast).
    The cap trades the capped attempts' evaluations, lost when an output is rotated, against pilots for
    outputs a rotation cannot help: on six functions of two curved oblique coordinates (``shared_latent``)
    caps 4, 6 and 8 cost 40k, 49k and 74k evaluations at 5-D (the axes alone 358k), and cap 4 costs 11%
    more than the axes at 3-D, where cap 6 costs nothing extra.
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = len(lo)
    F = _SharedOracle(f)
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    rank_max = kw.pop("rank_max", 60)

    def train(g, blo, bhi, c, cap):
        return build(
            lambda X: g(X)[:, c], blo, bhi, atol, seed=seed, rank_max=cap, **kw
        )

    if not rotate:
        return VectorTT(
            [
                (
                    list(range(outputs)),
                    None,
                    [train(F, lo, hi, c, rank_max) for c in range(outputs)],
                )
            ],
            centre,
            outputs,
            F.evals,
        )
    kept, high = {}, []
    for c in range(outputs):
        try:
            kept[c] = train(F, lo, hi, c, min(axes_rank, rank_max))
        except RankLimit:
            high.append(c)
    groups = [(sorted(kept), None, [kept[c] for c in sorted(kept)])] if kept else []
    if high:
        n_p = 1 << int(np.ceil(np.log2(4 + 2 * d)))
        X = check_points(lo, hi, n_p, seed + 101)
        h = 1e-6 * (hi - lo)
        V = F(
            np.concatenate([X] + [X + h[k] * np.eye(d)[k] for k in range(d)])
        ).reshape(d + 1, n_p, outputs)[:, :, high]
        J = (V[1:] - V[0][None]) / h[:, None, None]
        top = np.max(np.abs(V[0]), axis=0)
        rng = np.random.default_rng(seed + 202)
        axes = []
        for local, Q in metric_groups(J):
            cols = [high[i] for i in local]
            if Q is not None:
                hw = np.abs(Q).T @ half
                W = F(centre + rng.uniform(-hw, hw, (2 * d, d)) @ Q.T)[:, cols]
                if np.all(np.isfinite(W)) and np.all(
                    np.abs(W) <= 10.0 * top[local] + 1e-300
                ):
                    g = lambda U, Q=Q: F(centre + U @ Q.T)
                    groups.append(
                        (cols, Q, [train(g, -hw, hw, c, rank_max) for c in cols])
                    )
                    continue
            axes += cols
        if axes:
            groups.append(
                (
                    sorted(axes),
                    None,
                    [train(F, lo, hi, c, rank_max) for c in sorted(axes)],
                )
            )
    return VectorTT(groups, centre, outputs, F.evals)


def _piece_coefficients(C: np.ndarray, pieces: int) -> np.ndarray:
    """Node values (r, pieces * (n + 1), r') to Chebyshev coefficients, piece by piece."""
    if pieces == 1:
        return _coefficients_along(C, 1)
    n1 = C.shape[1] // pieces
    return np.concatenate(
        [_coefficients_along(C[:, p * n1 : (p + 1) * n1], 1) for p in range(pieces)],
        axis=1,
    )


def _keep_degrees(C: np.ndarray, pieces: int, keep: int) -> np.ndarray:
    """The core with only the first ``keep`` coefficients of every piece."""
    if pieces == 1:
        return C[:, :keep]
    return C.reshape(C.shape[0], pieces, C.shape[1] // pieces, C.shape[2])[
        :, :, :keep
    ].reshape(C.shape[0], pieces * keep, C.shape[2])


def round_tt(cores: list, eps: float) -> list:
    """
    TT-SVD rounding (Oseledets 2011): right-orthogonalise the cores, then truncate every bond, left to
    right, to the smallest rank whose discarded singular values stay below ``eps * |TT| / sqrt(d - 1)``
    (a relative Frobenius error of at most eps overall, on the coefficient tensor).
    """
    cores = [C.copy() for C in cores]
    d = len(cores)
    for k in range(d - 1, 0, -1):
        ra, n1, rb = cores[k].shape
        Q, R = np.linalg.qr(cores[k].reshape(ra, n1 * rb).T)
        cores[k] = Q.T.reshape(-1, n1, rb)
        cores[k - 1] = np.einsum("ajb,cb->ajc", cores[k - 1], R)
    delta = eps * np.linalg.norm(cores[0]) / np.sqrt(max(d - 1, 1))
    for k in range(d - 1):
        ra, n1, rb = cores[k].shape
        U, sv, Vt = np.linalg.svd(cores[k].reshape(ra * n1, rb), full_matrices=False)
        tail = np.sqrt(np.cumsum(sv[::-1] ** 2))[
            ::-1
        ]  # tail[i]: the norm of the singular values from i on
        r = max(1, int(np.sum(tail > delta)))
        cores[k] = U[:, :r].reshape(ra, n1, r)
        cores[k + 1] = np.einsum("ab,bjc->ajc", sv[:r, None] * Vt[:r], cores[k + 1])
    return cores


def compress(model: ChebTT, probe: np.ndarray, budget: float) -> ChebTT:
    """
    The accepted model with its ranks rounded (the largest eps, by bisection) and then its trailing
    degrees chopped axis by axis, each step kept only while the predictions at ``probe`` stay within
    ``budget`` of the accepted model's. Acceptance leaves the model at most half the tolerance from f at
    its check points, usually far less, so compressing within a quarter of it cuts the cores 3-10x and
    inference with them (10-D Lotka-Volterra: 2,700 coefficients to 293, 8.7 to 0.35 us per query).
    The probe is the check set plus uniform points: model-only predictions, no evaluation of f.
    """
    T = model.bases(probe)  # once: every trial keeps or cuts degrees
    ref = model.contract(T)

    def close(values):
        return float(np.max(np.abs(values - ref))) <= budget

    # Rank rounding: the bisection on eps meets few distinct rank profiles, and a profile fixes the
    # rounded cores (the orthogonalisation does not depend on eps), so each profile is judged once.
    cores, lo_e, hi_e, judged = model.cores, 0.0, 1.0, {}
    for _ in range(16):
        mid = 0.5 * (lo_e + hi_e)
        trial = round_tt(model.cores, mid)
        ranks = tuple(C.shape[2] for C in trial[:-1])
        if ranks not in judged:
            judged[ranks] = close(ChebTT(model.lo, model.hi, trial).contract(T))
        if judged[ranks]:
            cores, lo_e = trial, mid
        else:
            hi_e = mid
    # Degree cuts, left to right, repeated while any axis shrinks. Cutting axis k's top degrees changes the
    # model by sum_j T_j(t_k) (L_k C_k[:, j, :] R_k+1) over the degrees cut, so one product of core k gives
    # every degree's contribution at the probe, and how many top degrees (in every piece at once) would
    # stay within the budget one after the other. An axis of degree <= 32 loses one degree per visit,
    # judged as that single cut; above it, a visit takes half of what fits: a high-degree axis needs
    # O(log n) passes instead of one per degree, and since no axis takes its whole share at once, the
    # axes still share the budget about as single cuts do. The right products are those of the pass's starting cores (no core right of k has changed
    # yet in this pass); the left one is carried through the cores as they are cut. An axis with
    # breakpoints is judged with its own piece layout (``axis_values`` for the core as it is now).
    layouts: dict = {}

    def values_for(k: int, n1: int) -> np.ndarray:
        """Axis k's Chebyshev values at the probe for a core of n1 rows: T's own rows, or on an axis with pieces its layout for n1 (cached)."""
        if model.pieces(k) == 1:
            return T[k]
        if (k, n1) not in layouts:
            layouts[(k, n1)] = model.axis_values(probe, k, n1)
        return layouts[(k, n1)]

    shrunk = True
    while shrunk:
        shrunk = False
        _, right = ChebTT(model.lo, model.hi, cores).environments(T)
        cores, v = list(cores), np.ones((len(ref), 1))
        for k in range(len(cores)):
            C, P = cores[k], model.pieces(k)
            m1 = C.shape[1] // P
            if (
                m1 > 33
            ):  # every degree's contribution at once: how many top degrees fit, one after the other
                Tk = values_for(k, C.shape[1])
                ra, n1, rb = C.shape
                w = np.einsum(
                    "qjb,qb->qj",
                    (v @ C.reshape(ra, n1 * rb)).reshape(-1, n1, rb),
                    right[k],
                )  # (Q, n1)
                part = Tk[:n1].T * w
                dev = part.sum(axis=1) - ref
                fits = 0
                while fits < m1 - 1:
                    dev = dev - part[
                        :, [p_ * m1 + m1 - 1 - fits for p_ in range(P)]
                    ].sum(axis=1)
                    if float(np.max(np.abs(dev))) > budget:
                        break
                    fits += 1
                if fits:
                    cores[k], shrunk = _keep_degrees(C, P, m1 - (fits + 1) // 2), True
            elif m1 > 1:  # one degree, judged as a single cut
                trial = _keep_degrees(C, P, m1 - 1)
                if close(through(v, trial, values_for(k, trial.shape[1]), right[k])):
                    cores[k], shrunk = trial, True
            v = advance(v, cores[k], values_for(k, cores[k].shape[1]))
    return ChebTT(model.lo, model.hi, cores, model.evals, tuple(C.shape[2] for C in cores[:-1]),
                  tuple(C.shape[1] // model.pieces(k) - 1 for k, C in enumerate(cores)), model.breaks, model.bands)  # fmt: skip


def _dropped(model: ChebTT, k: int) -> np.ndarray:
    """
    Axis k's core with only the top quarter of its degrees in every piece (the rest zero): the part
    of the model that truncating that axis removes (see ``chebyshev_tail``).
    """
    C = model.cores[k]
    P = model.pieces(k)
    n = C.shape[1] // P - 1
    out = C.reshape(C.shape[0], P, n + 1, C.shape[2]).copy()
    out[:, :, : n - n // 4 + 1] = 0.0
    return out.reshape(C.shape)


def _sweeps(grid: _Grid, right: list, sweeps: int, budget: int):
    """Alternating left-right maxvol sweeps; returns the left index sets and the cores (node values) of the last left-to-right sweep."""
    d = len(grid.n)
    for s in range(sweeps):
        # Left to right: cores from the current right sets, new left sets from maxvol.
        left = [np.zeros((1, 0), dtype=np.int64)]
        cores = []
        for k in range(d):
            L, R = left[k], right[k + 1]
            nk = grid.count[k]
            idx = np.concatenate([np.repeat(np.repeat(L, nk, axis=0), len(R), axis=0),
                                  np.tile(np.repeat(np.arange(nk), len(R))[:, None], (len(L), 1)),
                                  np.tile(R, (len(L) * nk, 1))], axis=1)  # fmt: skip
            C = grid.values(idx, budget).reshape(len(L) * nk, len(R))
            if k == d - 1:
                cores.append(C.reshape(len(L), nk, 1))
                break
            Q, _ = np.linalg.qr(C)
            rows = maxvol(Q)
            cores.append((Q @ np.linalg.inv(Q[rows])).reshape(len(L), nk, len(rows)))
            a, i = np.divmod(rows, nk)
            left.append(np.concatenate([L[a], i[:, None]], axis=1))
        if s == sweeps - 1:
            return left, cores
        # Right to left: new right sets from maxvol on the transposed fibre matrices.
        for k in range(d - 1, 0, -1):
            L, R = left[k], right[k + 1]
            nk = grid.count[k]
            idx = np.concatenate([np.repeat(np.repeat(L, nk, axis=0), len(R), axis=0),
                                  np.tile(np.repeat(np.arange(nk), len(R))[:, None], (len(L), 1)),
                                  np.tile(R, (len(L) * nk, 1))], axis=1)  # fmt: skip
            C = (
                grid.values(idx, budget).reshape(len(L), nk * len(R)).T
            )  # (nk r_k+1, r_k-1)
            Q, _ = np.linalg.qr(C)
            rows = maxvol(Q)
            i, b = np.divmod(rows, len(R))
            right[k] = np.concatenate([i[:, None], R[b]], axis=1)
    return left, cores
