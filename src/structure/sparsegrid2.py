"""
SparseCheb v2: the dimension-adaptive sparse-grid Chebyshev interpolant of ``sparsegrid``, stored as one
sparse Chebyshev expansion and checked on a low-discrepancy set.

The surrogate is the same combination ``A_I = sum_l c_l U_l`` of tensor interpolants on nested
Chebyshev grids, grown by the same Gerstner-Griebel rule. What changes is how it is held and checked:

- **One expansion.** Every ``U_l`` is a tensor of Chebyshev coefficients, so ``A_I`` is a single sum
  ``sum_k A_k prod_j T_{k_j}(x_j)`` over multi-degrees ``k``. ``A`` is updated in place whenever the
  combination coefficients change (adding an index changes ``c`` for at most 2^d indices). Evaluating
  the surrogate is then one gathered product per chunk of points, ``Phi(X) @ A``, with the per-axis
  Chebyshev values from the three-term recurrence, instead of one small tensor contraction per index.
  Training (which evaluates the surrogate at every index's new points) and inference both scale with
  the number of distinct degrees, not with the number of indices.
- **A stronger check.** Convergence is checked on a scrambled Sobol set of ``n_check`` points (default
  ``64 + 32 d``) plus the domain's corners for d <= 8, evaluated once and counted, against half the
  tolerance. A miss widens the set (every active index is refined) and growth resumes.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dct
from scipy.stats import qmc

from src.structure.sparsegrid import axis_keys, axis_points

CHUNK = 1 << 22  # entries of the gathered (points x degrees) product per chunk


def cheb_table(t: np.ndarray, n: int) -> np.ndarray:
    """(Q, n + 1) Chebyshev values T_0..T_n at ``t`` in [-1, 1], by the three-term recurrence."""
    T = np.empty((len(t), n + 1))
    T[:, 0] = 1.0
    if n >= 1:
        T[:, 1] = t
    for k in range(2, n + 1):
        T[:, k] = 2.0 * t * T[:, k - 1] - T[:, k - 2]
    return T


@dataclass
class SparseCheb2:
    lo: np.ndarray
    hi: np.ndarray
    terms: dict = field(default_factory=dict)  # index -> coefficient tensor of U_l
    coef: dict = field(default_factory=dict)  # index -> combination coefficient c_l
    evals: int = 0
    checks: int = 0
    # The expansion: multi-degrees K (n, d) and coefficients A (n,), with pending weighted tensors
    # merged into them by one np.unique pass when the expansion is next evaluated.
    K: np.ndarray | None = None
    A: np.ndarray | None = None
    _pending: list = field(default_factory=list)

    @property
    def d(self) -> int:
        return len(self.lo)

    def insert(self, idx: tuple, U: np.ndarray) -> None:
        """Add an index (none above it is in the set yet); queue the change of A for the at most 2^d indices whose c changes."""
        self.terms[idx] = U
        for z in itertools.product((0, 1), repeat=self.d):
            m = tuple(a - b for a, b in zip(idx, z, strict=True))
            if any(v < 0 for v in m) or m not in self.terms:
                continue
            w = (-1) ** sum(z)
            self.coef[m] = self.coef.get(m, 0) + w
            if self.coef[m] == 0:
                del self.coef[m]
            T = self.terms[m]
            keys = np.stack([g.ravel() for g in np.indices(T.shape)], axis=1)
            self._pending.append((keys, w * T.ravel()))

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(K, A), with every queued change merged (duplicate degrees summed, zeros dropped)."""
        if self._pending:
            parts_k = [k for k, _ in self._pending] + ([self.K] if self.K is not None else [])
            parts_a = [a for _, a in self._pending] + ([self.A] if self.A is not None else [])
            K, inv = np.unique(np.concatenate(parts_k), axis=0, return_inverse=True)
            A = np.bincount(inv.ravel(), weights=np.concatenate(parts_a), minlength=len(K))
            keep = A != 0.0
            self.K, self.A, self._pending = K[keep], A[keep], []
        if self.K is None:
            return np.zeros((0, self.d), dtype=np.int64), np.zeros(0)
        return self.K, self.A

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        K, A = self.arrays()
        out = np.empty(len(X))
        if len(K) == 0:
            out[:] = 0.0
            return out
        t = 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0
        nmax = K.max(axis=0)
        step = max(1, CHUNK // max(1, len(K)))
        for s in range(0, len(X), step):
            ts = t[s : s + step]
            Phi = np.ones((len(ts), len(K)))
            for j in range(self.d):
                if nmax[j] > 0:
                    Phi *= cheb_table(ts[:, j], int(nmax[j]))[:, K[:, j]]
            out[s : s + step] = Phi @ A
        return out

    def leaf_of(self, X):
        return None


def _coefficients(V: np.ndarray, idx) -> np.ndarray:
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


def check_points(lo, hi, n: int, seed: int) -> np.ndarray:
    """A scrambled Sobol set of n points, plus the 2^d corners for d <= 8 (where steep features hide)."""
    d = len(lo)
    S = qmc.Sobol(d, scramble=True, seed=seed).random(n)
    if d <= 8:
        S = np.vstack([S, np.array(list(itertools.product((0.0, 1.0), repeat=d)))])
    return lo + S * (hi - lo)


def build(f, lo, hi, atol: float, safety: float = 0.5, max_evals: int = 2_000_000, max_level: int = 10,
          n_check: int | None = None, seed: int = 0) -> SparseCheb2:  # fmt: skip
    """Train a SparseCheb v2 of ``f`` (host (N, d) -> (N,)) on [lo, hi] to tolerance ``atol``."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    d = len(lo)
    model = SparseCheb2(lo, hi)
    values: dict[tuple, float] = {}

    def add(idx):
        axes = [axis_points(lk) for lk in idx]
        keys = [axis_keys(lk) for lk in idx]
        P = np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], axis=1)
        Kk = [tuple(k) for k in np.stack([m.ravel() for m in np.meshgrid(*keys, indexing="ij")], axis=1)]
        new = [i for i, k in enumerate(Kk) if k not in values]
        surplus = 0.0
        if new:
            Xn = lo + (P[new] + 1.0) * 0.5 * (hi - lo)
            fn = np.asarray(f(Xn), dtype=np.float64).ravel()
            model.evals += len(new)
            before = model.predict(Xn)
            surplus = float(np.max(np.abs(fn - before)))
            for i, v in zip(new, fn, strict=True):
                values[Kk[i]] = v
        V = np.array([values[k] for k in Kk]).reshape([len(axis_points(lk)) for lk in idx])
        model.insert(idx, _coefficients(V, idx))
        return surplus

    zero = (0,) * d
    add(zero)
    old, active = {zero}, {}
    for k in range(d):
        e = tuple(int(j == k) for j in range(d))
        active[e] = add(e)

    def refine(idx):
        del active[idx]
        old.add(idx)
        for k in range(d):
            if idx[k] >= max_level:
                continue
            fwd = tuple(v + (j == k) for j, v in enumerate(idx))
            back = [tuple(v - (j == i) for j, v in enumerate(fwd)) for i in range(d) if fwd[i] > 0]
            if fwd not in active and fwd not in old and all(b in old for b in back):
                active[fwd] = add(fwd)

    Xc = check_points(lo, hi, n_check or 64 + 32 * d, seed)
    fc = np.asarray(f(Xc), dtype=np.float64).ravel()
    model.evals += len(Xc)
    while active:
        while active and sum(active.values()) > safety * atol:
            if model.evals > max_evals:
                raise RuntimeError(f"SparseCheb v2 would exceed {max_evals:,} evaluations.")
            refine(max(active, key=active.get))
        if not active or float(np.max(np.abs(model.predict(Xc) - fc))) <= safety * atol:
            break
        model.checks += 1
        for idx in list(active):
            if model.evals > max_evals:
                raise RuntimeError(f"SparseCheb v2 would exceed {max_evals:,} evaluations.")
            if idx in active:
                refine(idx)
    return model
