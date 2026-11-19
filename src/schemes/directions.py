"""
Directions from gradients: the gradients themselves (the oracle's Jacobian, else central differences),
the rank of a gradient covariance, and the grouping of outputs that share their steep directions.
"""

import numpy as np


def box_gradients(oracle, X, lo, hi, step: float = 1e-4) -> np.ndarray:
    """
    (N, d, m) gradients in box-scaled inputs (df/dx_k times the half-width): the oracle's Jacobian when it
    has one, else central differences at ``step`` of the width (2d evaluations per point).
    """
    s0 = 0.5 * (hi - lo)
    if oracle.has_jacobian:
        J = oracle.jacobian(X)
        # (N, d) or (N, m, d) -> (N, d, m), C-contiguous: a transposed view would send the gradient covariance's
        # products through another summation kernel, and change their last bits.
        J = J[:, :, None] if J.ndim == 2 else np.ascontiguousarray(J.transpose(0, 2, 1))
        return J * s0[None, :, None]
    d = X.shape[1]
    h = step * (hi - lo)
    E = np.repeat(X, 2 * d, axis=0)
    for i in range(d):
        E[2 * i :: 2 * d, i] += h[i]
        E[2 * i + 1 :: 2 * d, i] -= h[i]
    F = oracle(E).reshape(len(E), -1)
    G = (F[0::2] - F[1::2]) / (2 * h[np.tile(np.arange(d), len(X))][:, None])
    return G.reshape(len(X), d, -1) * s0[None, :, None]


def covariance_rank(C, r_max):
    """The fewest leading eigenvectors of C holding 99% of its trace (at most ``r_max``), and the share left out."""
    w = np.clip(np.linalg.eigvalsh(C)[::-1], 0.0, None)
    share = np.cumsum(w) / max(float(w.sum()), 1e-300)
    r = min(int(np.searchsorted(share, 0.99) + 1), r_max)
    return r, 1.0 - share[r - 1]


def output_groups(Gp, r_max):
    """Output groups sharing a rotation: (outputs, V, r, weak) from the pilot gradients (points, d, m)."""
    m = Gp.shape[2]
    Cs = [Gp[:, :, o].T @ Gp[:, :, o] / len(Gp) for o in range(m)]
    Cpool = sum(C / max(float(np.trace(C)), 1e-300) for C in Cs)
    r_pool, _ = covariance_rank(Cpool, r_max)
    # Contiguous copies, not the reversed views: a saved model's arrays come back contiguous, and a product with
    # a reversed view sums in another order, so a loaded model would not predict bit for bit as trained.
    Vpool = np.ascontiguousarray(np.linalg.eigh(Cpool)[1][:, ::-1])
    pooled, groups = [], []
    for o, C in enumerate(Cs):
        r, rest = covariance_rank(C, r_max)
        V = np.ascontiguousarray(np.linalg.eigh(C)[1][:, ::-1])
        cos = np.linalg.svd(V[:, :r].T @ Vpool[:, :r_pool], compute_uv=False)
        if r <= r_pool and float(cos.min()) > 0.99:
            pooled.append(o)
        else:
            groups.append(([o], V, r, rest > 1e-4))
    if pooled:
        rest = max(1.0 - np.sum(np.clip(np.linalg.eigvalsh(Vpool.T @ Cs[o] @ Vpool)[::-1][:r_pool], 0, None)) / max(float(np.trace(Cs[o])), 1e-300)
                   for o in pooled)  # fmt: skip
        groups.insert(0, (pooled, Vpool, r_pool, rest > 1e-4))
    return groups, Vpool, r_pool


def active_subspace(oracle, lo, hi, n_points: int, seed: int = 0, step: float = 1e-5):
    """
    ``(W, eigenvalues)``: eigenvectors (descending) of ``C = mean grad f grad f^T`` from forward
    differences at ``n_points`` uniform points of the box (the active-subspace estimate; (d + 1) n_points
    evaluations).
    """
    d = len(lo)
    X = np.random.default_rng(seed).uniform(lo, hi, (n_points, d))
    h = step * (np.asarray(hi) - np.asarray(lo))
    pts = np.concatenate([X] + [X + np.eye(d)[k] * h[k] for k in range(d)])
    v = oracle(pts).ravel().reshape(d + 1, n_points)
    G = ((v[1:] - v[0]) / h[:, None]).T  # (n_points, d)
    lam, W = np.linalg.eigh(G.T @ G / n_points)
    order = np.argsort(lam)[::-1]
    return W[:, order], lam[order]
