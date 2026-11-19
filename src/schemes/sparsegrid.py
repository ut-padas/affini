"""
SparseCheb: a dimension-adaptive sparse-grid Chebyshev interpolant (Smolyak, Gerstner-Griebel).

Axis level ``idx`` means ``m(idx) = 2^idx + 1`` Chebyshev points of the second kind (one, the midpoint, at
``idx = 0``); the points of level idx contain those of every lower level. A multi-index ``idx`` (one level
per axis) has the tensor interpolant ``U_l`` on its tensor grid. For a downward-closed set ``I`` of
multi-indices the surrogate is the combination ``A_I = sum_{idx in I} c_l U_l`` with
``c_l = sum_{z in {0,1}^d, idx + z in I} (-1)^|z|``, which interpolates f on every grid of I.

Growth is dimension-adaptive. An active index's indicator is the worst surplus at its new points,
``max |f - A|`` where ``A`` is the surrogate before the index joined: how much that combination of
levels still had to correct. The active index with the largest indicator is refined, and each of its
forward neighbours whose backward neighbours are all refined becomes active. The set grows along
the directions and interactions f needs, so a function of few effective variables or low
interaction order costs few points even for large d.

Training stops when the active indicators sum to at most ``safety * atol`` and the surrogate meets
``safety * atol`` on a fixed check set. When the check fails, training looks one step ahead: every
active index is refined once, which evaluates its forward neighbours, among them the interactions
an index with a small indicator would never have opened, and the greedy growth resumes. Looking
ahead at every stop would also catch those, but it opens every forward neighbour of every active
index each time, which multiplies the evaluations at d = 5 for errors already within the tolerance.

The surpluses are a heuristic error estimate: a feature that no grid point sees, or a symmetry that
zeroes the surpluses, stops refinement early. Values are cached by grid point, so every point is
evaluated once.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np

from src.schemes.base import Settings, check_points
from src.schemes.chebyshev import cheb_basis, cheb_eval
from src.schemes.oracle import as_oracle
from src.schemes.profiling import Training

MAX_LEVEL = 12


def axis_points(level: int) -> np.ndarray:
    """The Chebyshev points of an axis level on [-1, 1]."""
    if level == 0:
        return np.zeros(1)
    n = 2**level
    return np.cos(np.pi * np.arange(n + 1) / n)


def axis_keys(level: int) -> np.ndarray:
    """Integer positions of the level's points on the finest level's lattice (shared by every level)."""
    if level == 0:
        return np.array([2 ** (MAX_LEVEL - 1)])
    return np.arange(2**level + 1) * 2 ** (MAX_LEVEL - level)


@dataclass
class SparseCheb:
    lo: np.ndarray
    hi: np.ndarray
    terms: dict = field(default_factory=dict)  # index -> Chebyshev coefficient tensor of U_l
    coef: dict = field(default_factory=dict)  # index -> combination coefficient c_l (kept up to date)
    evals: int = 0
    checks: int = 0  # times the check points sent training on

    @property
    def d(self) -> int:
        return len(self.lo)

    def insert(self, idx: tuple, U: np.ndarray) -> None:
        """
        Add index l (no index above it is in the set yet): c_l = 1, and every ``l - z`` already in the
        set gains ``(-1)^|z|``, which keeps ``c`` the combination coefficients of the enlarged set.
        """
        self.terms[idx] = U
        for z in itertools.product((0, 1), repeat=self.d):
            m = tuple(a - b for a, b in zip(idx, z, strict=True))
            if any(v < 0 for v in m) or (m not in self.terms):
                continue
            self.coef[m] = self.coef.get(m, 0) + (-1) ** sum(z)
            if self.coef[m] == 0:
                del self.coef[m]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        coef = self.coef
        t = 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0
        basis = {}
        out = np.zeros(len(X))
        for idx, c in coef.items():
            T = []
            for k, lk in enumerate(idx):
                n = 0 if lk == 0 else 2**lk
                if (k, n) not in basis:
                    basis[(k, n)] = cheb_basis(t[:, k], n) if n else np.ones((len(X), 1))
                T.append(basis[(k, n)])
            out += c * cheb_eval(self.terms[idx], T)
        return out


@dataclass(frozen=True)
class SparseGridSettings(Settings):
    safety: float = 0.5  # the surpluses and the check set must stay within safety * atol
    max_evals: int = 2_000_000
    max_level: int = 10  # deepest axis level (2^level + 1 points)
    n_check: int | None = None  # check points (default 64 + 32 d), plus the corners for d <= 8
    seed: int = 0


class SparseGridGrowth:
    """The growing index set: the refined indices (``old``), the active ones with their indicators, and every value evaluated, by grid point."""

    def __init__(self, model: SparseCheb, oracle, s: SparseGridSettings):
        self.model, self.oracle, self.s = model, oracle, s
        self.values: dict[tuple, float] = {}
        self.old: set = set()
        self.active: dict = {}

    def grid(self, idx):
        axes = [axis_points(lk) for lk in idx]
        keys = [axis_keys(lk) for lk in idx]
        P = np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], axis=1)
        K = np.stack([m.ravel() for m in np.meshgrid(*keys, indexing="ij")], axis=1)
        return P, K

    def add(self, idx) -> float:
        """Evaluate the new points of index l, store U_l, and return its indicator (worst surplus)."""
        model, lo, hi = self.model, self.model.lo, self.model.hi
        P, K = self.grid(idx)
        keys = [tuple(k) for k in K]
        new = [i for i, k in enumerate(keys) if k not in self.values]
        surplus = 0.0
        if new:
            Xn = lo + (P[new] + 1.0) * 0.5 * (hi - lo)
            fn = self.oracle(Xn).ravel()
            model.evals += len(new)
            before = model.predict(Xn) if model.terms else np.zeros(len(new))
            surplus = float(np.max(np.abs(fn - before)))
            for i, v in zip(new, fn, strict=True):
                self.values[keys[i]] = v
        V = np.array([self.values[k] for k in keys]).reshape([len(axis_points(lk)) for lk in idx])
        model.insert(idx, _coefficients(V, idx))
        return surplus

    def refine(self, idx) -> None:
        """Retire ``idx`` and activate each forward neighbour whose backward neighbours are all refined."""
        d = self.model.d
        del self.active[idx]
        self.old.add(idx)
        for k in range(d):
            if idx[k] >= self.s.max_level:
                continue
            fwd = tuple(v + (j == k) for j, v in enumerate(idx))
            back = [tuple(v - (j == i) for j, v in enumerate(fwd)) for i in range(d) if fwd[i] > 0]
            if fwd not in self.active and fwd not in self.old and all(b in self.old for b in back):
                self.active[fwd] = self.add(fwd)

    def budget(self) -> None:
        if self.model.evals > self.s.max_evals:
            raise RuntimeError(f"SparseCheb would exceed {self.s.max_evals:,} evaluations.")

    def greedy(self, atol: float) -> None:
        """Refine the largest indicator until the active indicators sum to at most ``safety * atol``."""
        while self.active and sum(self.active.values()) > self.s.safety * atol:
            self.budget()
            self.refine(max(self.active, key=self.active.get))

    def widen(self) -> None:
        """Look one step ahead: refine every active index once, which evaluates every forward neighbour."""
        for idx in list(self.active):
            self.budget()
            if idx in self.active:
                self.refine(idx)


def build(f, lo, hi, atol: float, settings: SparseGridSettings | None = None, *, jacobian=None, stream_dir=None,
          **overrides) -> SparseCheb:  # fmt: skip
    """
    Train a SparseCheb of ``f`` (host (N, d) -> (N,)) on [lo, hi] to tolerance ``atol``.

    When the surpluses say the surrogate has converged, it is checked at ``n_check`` (default
    64 + 32 d) scrambled Sobol points plus the corners for d <= 8 (``check_points``), evaluated once
    and counted as evaluations. If the worst error there exceeds ``safety * atol`` (the margin the
    surpluses get too: a check point rarely lands on the worst error), every active index is refined
    one step (the set widens instead of deepening: the surpluses missed some direction or
    interaction) and the greedy growth resumes, until the check passes.
    """
    s = SparseGridSettings.resolve(settings, overrides)
    oracle = as_oracle(f, jacobian)
    run = Training("sparsegrid", s, oracle, stream_dir)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = len(lo)
    model = SparseCheb(lo, hi)
    growth = SparseGridGrowth(model, oracle, s)
    with run.phase("fit"):
        zero = (0,) * d
        growth.add(zero)
        growth.old.add(zero)
        for k in range(d):
            e = tuple(int(j == k) for j in range(d))
            growth.active[e] = growth.add(e)
    with run.phase("sample"):
        Xc = check_points(lo, hi, s.n_check or 64 + 32 * d, s.seed, corners=True)
    fc = oracle(Xc).ravel()
    model.evals += len(Xc)
    model.checks = 0
    while growth.active:
        with run.phase("fit"):
            growth.greedy(atol)
        with run.phase("check"):
            worst = float(np.max(np.abs(model.predict(Xc) - fc))) if growth.active else 0.0
        done = not growth.active or worst <= s.safety * atol
        run.round_end(indices=len(model.terms), active=len(growth.active), worst=worst / atol, action="done" if done else "widen")
        if done:
            break
        model.checks += 1
        with run.phase("adapt"):
            growth.widen()
    return run.finish(model)


def _coefficients(V: np.ndarray, idx) -> np.ndarray:
    """Chebyshev coefficients of the tensor interpolant on index l's grid (a single-point axis is the constant)."""
    from scipy.fft import dct

    c = V.astype(np.float64)
    for k, lk in enumerate(idx):
        if lk > 0:
            a = np.moveaxis(c, k, 0)
            n = a.shape[0] - 1
            a = dct(a, type=1, axis=0) / n
            a[0] /= 2.0
            a[n] /= 2.0
            c = np.moveaxis(a, 0, k)
    return c
