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

Once accepted, the cores are compressed (``compress``): ranks rounded and trailing degrees chopped while
the model changes by at most ``chop_share * atol`` at the check points and ``n_probe`` uniform points,
which cuts the cores and the inference time several-fold.

The checks are estimates, not bounds: a feature that none of the three sets crosses is missed.
"""

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct
from scipy.linalg import qr, solve
from scipy.stats import qmc

from src.structure.chebtree import cheb_points

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

    def predict(self, X: np.ndarray, chunk: int = 16_384) -> np.ndarray:
        """In chunks that bound the temporaries (``contract``)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        sizes = [C.shape[1] for C in self.cores]
        return np.concatenate([self.contract(chebyshev_bases(self.local(X[s : s + chunk]), sizes)) for s in range(0, len(X), chunk)])

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
            ra, n1, rb = C.shape
            if ra == 1:  # every first core, every core of a rank-1 train: a scaling, without einsum's overhead
                v = v * (C[0].T @ Tk[:n1]).T
            else:
                M = Tk[:n1].T @ C.transpose(1, 0, 2).reshape(n1, ra * rb)
                v = np.einsum("qa,qab->qb", v, M.reshape(-1, ra, rb))
        return v[:, 0]

    def leaf_of(self, X):
        return None


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


class RankLimit(RuntimeError):
    """The cross reached ``rank_max`` without meeting the tolerance (as opposed to running out of evaluations)."""


def check_points(lo, hi, n: int, seed: int) -> np.ndarray:
    """A scrambled Sobol set of n points in [lo, hi]."""
    d = len(lo)
    return lo + qmc.Sobol(d, scramble=True, seed=seed).random(n) * (hi - lo)


def build(f, lo, hi, atol: float, safety: float = 0.5, rank0: int = 2, rank_step: int = 2, rank_max: int = 60,
          sweeps: int = 2, n_fibre: int = 4, n_max: int = 512, n_check: int | None = None, n_free: int = 20_000, chop_share: float = 0.25, n_probe: int = 4000,
          seed: int = 0, max_evals: int = 2_000_000) -> ChebTT:  # fmt: skip
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
            probe = np.concatenate([Xall, rng.uniform(lo, hi, (n_probe, d))])
            small = compress(model, probe, chop_share * atol)
            small.evals = model.evals
            return small
        if model.evals > max_evals:
            raise RuntimeError(f"ChebTT would exceed {max_evals:,} evaluations.")
        # Degree or rank: an axis whose top quarter of degrees still changes the model by more than its
        # share at the check points is under-resolved, and no rank can make up for it.
        full = model.predict(Xall)
        tails = {k: float(np.max(np.abs(full - _truncated(model, k).predict(Xall)))) for k in range(d)}
        grow = [k for k in range(d) if tails[k] > share and 2 * grid.n[k] <= n_max]
        if grow:
            grid.set_degrees([2 * nk if k in grow else nk for k, nk in enumerate(grid.n)])
            right = [None] + [R * (np.array([2 if a in grow else 1 for a in range(k + 1, d)])) for k, R in enumerate(right[1:-1])] + [right[-1]]
            continue
        r = min(r + rank_step, rank_max)
        if all(R.shape[0] >= r for R in right[1:-1]) and r == rank_max:
            raise RankLimit(f"ChebTT reached rank {rank_max} without meeting the tolerance.")
        right = [None] + [np.vstack([R, rng.integers(0, np.array(grid.n[k + 1 :]) + 1, size=(rank_step, d - k - 1))])
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
            V = np.asarray(self.f(X[list(first.values())]), dtype=np.float64).reshape(len(first), -1)
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
            joint = _directions_holding(np.linalg.eigvalsh(sum(G[i] for i in grp) + G[c]))
            if joint <= max(_directions_holding(np.linalg.eigvalsh(sum(G[i] for i in grp))), k_eig[c]):
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
            sizes = [max(m.cores[k].shape[1] for m in models) for k in range(len(models[0].cores))]
            for s in range(0, len(X), chunk):
                U = X[s : s + chunk] if Q is None else (X[s : s + chunk] - self.centre) @ Q
                T = chebyshev_bases(models[0].local(U), sizes)
                for c, model in zip(cols, models, strict=True):
                    out[s : s + chunk, c] = model.contract(T)
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
        return build(lambda X: g(X)[:, c], blo, bhi, atol, seed=seed, rank_max=cap, **kw)

    if not rotate:
        return VectorTT([(list(range(outputs)), None, [train(F, lo, hi, c, rank_max) for c in range(outputs)])], centre, outputs, F.evals)
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
        V = F(np.concatenate([X] + [X + h[k] * np.eye(d)[k] for k in range(d)])).reshape(d + 1, n_p, outputs)[:, :, high]
        J = (V[1:] - V[0][None]) / h[:, None, None]
        top = np.max(np.abs(V[0]), axis=0)
        rng = np.random.default_rng(seed + 202)
        axes = []
        for local, Q in metric_groups(J):
            cols = [high[i] for i in local]
            if Q is not None:
                hw = np.abs(Q).T @ half
                W = F(centre + rng.uniform(-hw, hw, (2 * d, d)) @ Q.T)[:, cols]
                if np.all(np.isfinite(W)) and np.all(np.abs(W) <= 10.0 * top[local] + 1e-300):
                    g = lambda U, Q=Q: F(centre + U @ Q.T)
                    groups.append((cols, Q, [train(g, -hw, hw, c, rank_max) for c in cols]))
                    continue
            axes += cols
        if axes:
            groups.append((sorted(axes), None, [train(F, lo, hi, c, rank_max) for c in sorted(axes)]))
    return VectorTT(groups, centre, outputs, F.evals)


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
        tail = np.sqrt(np.cumsum(sv[::-1] ** 2))[::-1]  # tail[i]: the norm of the singular values from i on
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
    T = chebyshev_bases(model.local(probe), [C.shape[1] for C in model.cores])  # once: every trial keeps or cuts degrees
    ref = model.contract(T)

    def close(cores):
        return float(np.max(np.abs(ChebTT(model.lo, model.hi, cores).contract(T) - ref))) <= budget

    cores, lo_e, hi_e = model.cores, 0.0, 1.0
    for _ in range(16):
        mid = 0.5 * (lo_e + hi_e)
        trial = round_tt(model.cores, mid)
        if close(trial):
            cores, lo_e = trial, mid
        else:
            hi_e = mid
    shrunk = True
    while shrunk:
        shrunk = False
        for k in range(len(cores)):
            if cores[k].shape[1] > 1:
                trial = list(cores)
                trial[k] = cores[k][:, :-1]
                if close(trial):
                    cores, shrunk = trial, True
    return ChebTT(model.lo, model.hi, cores, model.evals, tuple(C.shape[2] for C in cores[:-1]),
                  tuple(C.shape[1] - 1 for C in cores))  # fmt: skip


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
