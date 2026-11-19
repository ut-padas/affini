"""
MetricKernel: greedy Matern-5/2 kernel interpolation in a metric learned from the model's gradients.

The kernel is ``k(|x - y|_M)`` with ``|v|_M^2 = v^T M v``. After each fit, M is replaced by the square
root of the average gradient outer product of the current model at its centres (the mechanism of
recursive feature machines), scaled to keep the length scale chosen just before, so that directions
along which f changes fast get short length scales and a rotated feature becomes isotropic. The scale
is the multiple of M with the smallest worst leave-one-out error (Rippa's closed form). Each round
evaluates a fresh scrambled Sobol batch: its worst-residual points join the centres, and training stops
when two consecutive batches stay below ``safety * atol``.

The checks are random batches: in high dimension most of the volume lies near the boundary, where a
kernel interpolant is weakest, and a batch rarely lands on the worst point (5-D: worst errors 2-11x
the tolerance on Genz oscillatory and corner peak while the batches passed). Training costs a dense
Cholesky factorisation of the centres per round, inference one kernel row per query.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import qmc

S5 = np.sqrt(5.0)


def _r(A, B, L):
    """Distances between the rows of A and B in the metric ``M = L^T L``."""
    a, b = A @ L.T, B @ L.T
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
    return np.sqrt(np.maximum(d2, 0.0))


def _k(r):
    """Matern-5/2."""
    return (1 + S5 * r + 5 * r * r / 3) * np.exp(-S5 * r)


class Model:
    """Centres, weights and metric factor of a trained MetricKernel, in [-1, 1]^d coordinates."""

    def __init__(self, lo, hi, X, alpha, L, mean):
        self.lo, self.hi, self.X, self.alpha, self.L, self.mean = lo, hi, X, alpha, L, mean

    def _t(self, Q):
        return 2 * (np.asarray(Q, float) - self.lo) / (self.hi - self.lo) - 1

    def leaf_of(self, Q):
        return None

    def predict(self, Q, chunk=20000):
        T = self._t(Q)
        out = np.empty(len(T))
        for s in range(0, len(T), chunk):
            out[s:s + chunk] = _k(_r(T[s:s + chunk], self.X, self.L)) @ self.alpha + self.mean
        return out

    def grad(self, T):
        """The model's gradient in [-1, 1]^d coordinates."""
        r = _r(T, self.X, self.L)
        w = -(5 / 3) * (1 + S5 * r) * np.exp(-S5 * r) * self.alpha[None]  # k'(r)/r * alpha
        M = self.L.T @ self.L
        diff = T[:, None, :] - self.X[None]  # (q, n, d)
        return np.einsum("qn,qnd->qd", w, diff) @ M


def _fit(X, y, L, nugget=1e-10):
    """Weights, mean and leave-one-out errors (Rippa) of the interpolant through (X, y)."""
    K = _k(_r(X, X, L)) + nugget * np.eye(len(X))
    c = cho_factor(K)
    mean = float(np.mean(y))
    alpha = cho_solve(c, y - mean)
    Kinv_diag = np.diag(cho_solve(c, np.eye(len(X))))
    return alpha, mean, np.abs(alpha / Kinv_diag)  # Rippa's leave-one-out errors


def build(f, lo, hi, atol, safety=0.5, n0=None, batch=None, add=None, max_evals=20000, seed=0, scales=(0.5, 0.7, 1.0, 1.4, 2.0), metric=True):
    """Train a MetricKernel of ``f`` (host (N, d) -> (N,)) on [lo, hi] to tolerance ``atol``."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    d = len(lo)
    sob = qmc.Sobol(d, scramble=True, seed=seed)
    n0 = n0 or 16 * d
    batch = batch or 32 * d
    add = add or 4 * d
    T = np.vstack([np.zeros((1, d)), 2 * sob.random(n0) - 1])
    y = np.asarray(f(lo + (T + 1) / 2 * (hi - lo)), float).ravel()
    evals = len(T)
    L = np.eye(d) * 2.0  # initial length scale 0.5 of the half-width
    passes = 0
    while True:
        # length scale by the worst leave-one-out error over a few multiples of L
        best = None
        for c in scales:
            try:
                alpha, mean, loo = _fit(T, y, c * L)
            except np.linalg.LinAlgError:
                continue
            if best is None or loo.max() < best[0]:
                best = (loo.max(), c, alpha, mean)
        _, c, alpha, mean = best
        L = c * L
        model = Model(lo, hi, T, alpha, L, mean)
        if metric:  # AGOP metric of the current model, normalised to keep the scale chosen above
            G = model.grad(T)
            A = G.T @ G / len(T)
            w, V = np.linalg.eigh(A)
            w = np.maximum(w, 1e-3 * w.max())
            Mn = V @ np.diag(np.sqrt(w)) @ V.T  # sqrt: a metric in which f changes at unit rate per unit length
            Mn *= np.sqrt(d) / np.linalg.norm(Mn)
            L = np.linalg.cholesky(Mn @ Mn.T * np.trace(L.T @ L) / np.trace(Mn @ Mn.T)).T
            alpha, mean, _ = _fit(T, y, L)
            model = Model(lo, hi, T, alpha, L, mean)
        Z = 2 * sob.random(batch) - 1
        fz = np.asarray(f(lo + (Z + 1) / 2 * (hi - lo)), float).ravel()
        evals += len(Z)
        res = np.abs(model.predict(lo + (Z + 1) / 2 * (hi - lo)) - fz)
        if res.max() <= safety * atol:
            passes += 1
            if passes >= 2:
                model.evals = evals
                return model
        else:
            passes = 0
        if evals > max_evals:
            raise RuntimeError(f"kernel model would exceed {max_evals:,} evaluations")
        take = np.argsort(res)[::-1][:add]
        T, y = np.vstack([T, Z[take]]), np.concatenate([y, fz[take]])
