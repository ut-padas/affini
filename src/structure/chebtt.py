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

The checks are estimates, not bounds: a feature that none of the three sets crosses is missed.
"""

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct
from scipy.linalg import qr, solve
from scipy.stats import qmc

from src.structure.chebtree import cheb_basis, cheb_points

LATTICE = 1 << 12  # the finest degree any axis may reach (a power of two; degrees are 8 * 2^m)


def maxvol(A: np.ndarray, tol: float = 1.05, max_iters: int = 200) -> np.ndarray:
    """Rows of the tall matrix A (n x r) spanning a near-maximal-volume r x r submatrix (Goreinov et al.)."""
    n, r = A.shape
    if n <= r:
        return np.arange(n)
    _, _, piv = qr(A.T, pivoting=True, mode="economic")
    idx = piv[:r].copy()
    B = solve(A[idx].T, A.T).T  # A A[idx]^-1, the coefficients of every row in the chosen rows
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        t = 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0
        v = np.ones((len(X), 1))
        for k, C in enumerate(self.cores):
            T = cheb_basis(t[:, k], C.shape[1] - 1)  # (Q, n + 1)
            v = np.einsum("qa,ajb,qj->qb", v, C, T, optimize=True)
        return v[:, 0]

    def leaf_of(self, X):
        return None


class _Grid:
    """Node coordinates per axis and a cache of f by grid index."""

    def __init__(self, f, lo, hi, n):
        self.f, self.lo, self.hi = f, lo, hi
        self.set_degrees(n)
        self.cache: dict[tuple, float] = {}
        self.evals = 0

    def set_degrees(self, n):
        self.n = list(n)
        self.nodes = [self.lo[k] + (cheb_points(nk) + 1.0) * 0.5 * (self.hi[k] - self.lo[k]) for k, nk in enumerate(self.n)]

    def sample(self, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
        """Up to n evaluated grid points (all of them when there are fewer) and their values."""
        keys = list(self.cache)
        if len(keys) > n:
            keys = [keys[i] for i in rng.choice(len(keys), n, replace=False)]
        M = np.array(keys, dtype=np.float64).reshape(-1, len(self.n))
        X = self.lo + (np.cos(np.pi * M / LATTICE) + 1.0) * 0.5 * (self.hi - self.lo)
        return X, np.array([self.cache[k] for k in keys])

    def values(self, idx: np.ndarray, max_evals: int) -> np.ndarray:
        """f at the grid points with integer multi-indices ``idx`` (M, d), evaluating only the uncached ones."""
        # Keyed on the finest lattice: node i of degree n is node i * (N / n) of degree N, so a doubled
        # axis finds its old points again.
        step = np.array([LATTICE // nk for nk in self.n])
        keys = [tuple(row) for row in (idx * step).tolist()]
        missing = sorted({k for k in keys if k not in self.cache})
        if missing:
            if self.evals + len(missing) > max_evals:
                raise RuntimeError(f"ChebTT would exceed {max_evals:,} evaluations.")
            M = np.array(missing) // step
            X = np.stack([self.nodes[j][M[:, j]] for j in range(len(self.n))], axis=1)
            vals = np.asarray(self.f(X), dtype=np.float64).ravel()
            self.evals += len(missing)
            self.cache.update(zip(missing, vals.tolist(), strict=True))
        return np.array([self.cache[k] for k in keys])


def _fibre_degree(f, lo, hi, k, share, rng, n_fibre, n_max, n0=8):
    """The smallest nested degree at which random fibres along axis k have Chebyshev tails below ``share``; and the points used."""
    d = len(lo)
    base = rng.uniform(lo, hi, (n_fibre, d))
    n, used = n0, 0
    while True:
        t = cheb_points(n)
        P = np.repeat(base, n + 1, axis=0)
        P[:, k] = np.tile(lo[k] + (t + 1.0) * 0.5 * (hi[k] - lo[k]), n_fibre)
        V = np.asarray(f(P), dtype=np.float64).reshape(n_fibre, n + 1)
        used += len(P)
        if max(chebyshev_tail(v) for v in V) <= share or 2 * n > n_max:
            return n, used
        n *= 2


def check_points(lo, hi, n: int, seed: int) -> np.ndarray:
    """A scrambled Sobol set of n points in [lo, hi]."""
    d = len(lo)
    return lo + qmc.Sobol(d, scramble=True, seed=seed).random(n) * (hi - lo)


def build(f, lo, hi, atol: float, safety: float = 0.5, rank0: int = 2, rank_step: int = 2, rank_max: int = 60,
          sweeps: int = 2, n_fibre: int = 4, n_max: int = 512, n_check: int | None = None, n_free: int = 20_000, seed: int = 0,
          max_evals: int = 2_000_000) -> ChebTT:  # fmt: skip
    """Train a ChebTT of ``f`` (host (N, d) -> (N,)) on [lo, hi] to tolerance ``atol``."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = len(lo)
    rng = np.random.default_rng(seed)
    share = safety * atol / d
    pre = 0
    n = []
    for k in range(d):
        nk, used = _fibre_degree(f, lo, hi, k, share, rng, n_fibre, n_max)
        n.append(nk)
        pre += used
    grid = _Grid(f, lo, hi, n)
    Xc = check_points(lo, hi, n_check or 64 + 32 * d, seed)
    fc = np.asarray(f(Xc), dtype=np.float64).ravel()
    pre += len(Xc)
    r = rank0
    # Seed the right sets with the grid node nearest the largest |f| seen so far and the centre node, then
    # random nodes: maxvol only moves among the fibres it is shown, so fibres that all miss a narrow peak
    # (every value near 0) would converge to a model of the empty background.
    peak = np.array([np.argmin(np.abs(grid.nodes[k] - Xc[np.argmax(np.abs(fc)), k])) for k in range(d)])
    centre = np.array([nk // 2 for nk in n])
    right = [None] + [np.vstack([peak[k + 1 :], centre[k + 1 :], rng.integers(0, np.array(n[k + 1 :]) + 1, size=(max(r - 2, 0), d - k - 1))])[:max(r, 2)]
                      for k in range(d - 1)] + [np.zeros((1, 0), dtype=np.int64)]  # fmt: skip
    while True:
        left, cores = _sweeps(grid, right, sweeps, max_evals - pre)
        model = ChebTT(lo, hi, [_coefficients_along(C, 1) for C in cores], grid.evals + pre,
                       tuple(C.shape[2] for C in cores[:-1]), tuple(grid.n))  # fmt: skip
        # Free checks: grid points evaluated in earlier sweeps or at lower degrees, which the final
        # cores do not interpolate (they interpolate the last sweep's fibres only), concentrated where
        # the cross explored, i.e. where f matters.
        Xg, fg = grid.sample(n_free, rng)
        err = np.abs(model.predict(np.concatenate([Xc, Xg])) - np.concatenate([fc, fg]))
        # Fibre look-ahead: the next nested grid's new points along each axis, through the worst check
        # point and through the largest |f| seen (where a narrow feature is), the points a refinement
        # would sample next.
        Xall, fall = np.concatenate([Xc, Xg]), np.concatenate([fc, fg])
        bases = Xall[[int(np.argmax(err)), int(np.argmax(np.abs(fall)))]]
        look = 0.0
        for k in range(d):
            nk = grid.n[k]
            t = np.cos(np.pi * (np.arange(nk) + 0.5) / nk)
            P = np.repeat(bases, nk, axis=0)
            P[:, k] = np.tile(lo[k] + (t + 1.0) * 0.5 * (hi[k] - lo[k]), len(bases))
            fP = np.asarray(f(P), dtype=np.float64).ravel()
            pre += len(P)
            look = max(look, float(np.max(np.abs(model.predict(P) - fP))))
            Xc, fc = np.concatenate([Xc, P]), np.concatenate([fc, fP])  # kept: later rounds are checked there too
        model.evals = grid.evals + pre
        if max(float(err.max()), look) <= safety * atol:
            return model
        if model.evals > max_evals:
            raise RuntimeError(f"ChebTT would exceed {max_evals:,} evaluations.")
        # Degree or rank: an axis whose top quarter of degrees still changes the model by more than its
        # share at the check points is under-resolved, and no rank can make up for it.
        full = model.predict(Xall)
        tails = [float(np.max(np.abs(full - _truncated(model, k).predict(Xall)))) for k in range(d)]
        grow = [k for k in range(d) if tails[k] > share and 2 * grid.n[k] <= n_max]
        if grow:
            grid.set_degrees([2 * nk if k in grow else nk for k, nk in enumerate(grid.n)])
            right = [None] + [R * (np.array([2 if a in grow else 1 for a in range(k + 1, d)])) for k, R in enumerate(right[1:-1])] + [right[-1]]
            continue
        r = min(r + rank_step, rank_max)
        if all(R.shape[0] >= r for R in right[1:-1]) and r == rank_max:
            raise RuntimeError(f"ChebTT reached rank {rank_max} without meeting the tolerance.")
        right = [None] + [np.vstack([R, rng.integers(0, np.array(grid.n[k + 1 :]) + 1, size=(rank_step, d - k - 1))])
                          for k, R in enumerate(right[1:-1])] + [right[-1]]  # fmt: skip


def _truncated(model: ChebTT, k: int) -> ChebTT:
    """The model with the top quarter of axis k's degrees dropped (see ``chebyshev_tail``)."""
    cores = list(model.cores)
    n = cores[k].shape[1] - 1
    cores[k] = cores[k][:, : n - n // 4 + 1]
    return ChebTT(model.lo, model.hi, cores)


def _sweeps(grid: _Grid, right: list, sweeps: int, budget: int):
    """Alternating left-right maxvol sweeps; returns the left index sets and the cores (node values) of the last left-to-right sweep."""
    d = len(grid.n)
    for s in range(sweeps):
        # Left to right: cores from the current right sets, new left sets from maxvol.
        left = [np.zeros((1, 0), dtype=np.int64)]
        cores = []
        for k in range(d):
            L, R = left[k], right[k + 1]
            nk = grid.n[k] + 1
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
            nk = grid.n[k] + 1
            idx = np.concatenate([np.repeat(np.repeat(L, nk, axis=0), len(R), axis=0),
                                  np.tile(np.repeat(np.arange(nk), len(R))[:, None], (len(L), 1)),
                                  np.tile(R, (len(L) * nk, 1))], axis=1)  # fmt: skip
            C = grid.values(idx, budget).reshape(len(L), nk * len(R)).T  # (nk r_k+1, r_k-1)
            Q, _ = np.linalg.qr(C)
            rows = maxvol(Q)
            i, b = np.divmod(rows, len(R))
            right[k] = np.concatenate([i[:, None], R[b]], axis=1)
    return left, cores
