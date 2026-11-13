"""Small batched dense algebra: SPD factor and solves, products with a fixed inner order, regularized fits."""

import numpy as np

from src import IS_CUPY_BACKEND, Array, kernels, xp

if IS_CUPY_BACKEND:
    pass

if IS_CUPY_BACKEND:  # sources in src/kernels/*.cu
    _linalg = kernels.module("small_linalg")
    _kern_det_matmul = _linalg.get_function("det_matmul")
    _kern_spd_solve = _linalg.get_function("spd_solve")
    _kern_spd_factor = _linalg.get_function("spd_factor")
    _kern_spd_factor_coop = _linalg.get_function("spd_factor_coop")
    _kern_spd_solve_factored = _linalg.get_function("spd_solve_factored")
    _kern_affine_reg = _linalg.get_function("fused_affine_fit_regularize")
    _kern_cholesky = _linalg.get_function("batched_inplace_cholesky_solve")


TINY_EPS = 1e-12

# From this many unknowns, SPD systems are factored by one block each (spd_factor_coop).
SPD_COOP_MIN_P = 32


def det_matmul(A: Array, B: Array) -> Array:
    """
    ``A @ B`` for stacks of small matrices (leading shapes broadcast). NumPy: exactly
    ``A @ B``. CuPy: one thread per output entry with a sequential inner sum, so every
    matrix's result is the same between runs and for any number of matrices in the stack;
    cuBLAS chooses its batched algorithm (and so the rounding) by batch count, which made
    training decisions depend on the batch size. Meant for training-time products of
    ``d``- or ``p``-sized matrices, not for large GEMMs.
    """
    if not IS_CUPY_BACKEND:
        return A @ B
    lead = xp.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    m, n, q = A.shape[-2], A.shape[-1], B.shape[-1]
    A = xp.ascontiguousarray(xp.broadcast_to(A, (*lead, m, n)), dtype=xp.float64)
    B = xp.ascontiguousarray(xp.broadcast_to(B, (*lead, n, q)), dtype=xp.float64)
    out = xp.empty((*lead, m, q))
    total = out.size
    if total:
        _kern_det_matmul(((total + 255) // 256,), (256,), (A, B, out, np.int64(total), np.int32(m), np.int32(n), np.int32(q)))
    return out


def spd_solve(A: Array, B: Array) -> Array:
    """
    ``A^-1 B`` for stacks of small symmetric positive definite systems ``(..., p, p)``,
    ``(..., p, q)`` (equal leading shapes). NumPy: ``xp.linalg.solve``. CuPy: one thread
    per system (Cholesky of the lower triangle and two triangular solves), a single kernel
    instead of the dozen launches and ~0.4 ms of a batched ``linalg.solve``, with bits that
    do not depend on the rest of the stack. A system that is not positive definite (or not
    finite) gets an all-NaN solution, which callers already treat as an invalid fit.
    """
    if IS_CUPY_BACKEND and A.shape[-1] >= SPD_COOP_MIN_P:
        return spd_solve_factored(spd_factor(A), B)
    if not IS_CUPY_BACKEND:
        return xp.linalg.solve(A, B)
    p, q = A.shape[-1], B.shape[-1]
    lead = A.shape[:-2]
    L = xp.array(A, dtype=xp.float64, order="C", copy=True)
    X = xp.array(xp.broadcast_to(B, (*lead, p, q)), dtype=xp.float64, order="C", copy=True)
    nb = int(np.prod(lead, dtype=np.int64))
    if nb:
        _kern_spd_solve(((nb + 127) // 128,), (128,), (L, X, np.int64(nb), np.int32(p), np.int32(q)))
    return X


def spd_factor(A: Array) -> Array:
    """
    The Cholesky factor of each SPD system in ``A`` (..., p, p), for ``spd_solve_factored``:
    solving several right-hand sides with one matrix factors it once. CuPy: the factorization
    of ``spd_solve`` (NaN for a non-SPD system); NumPy returns ``A`` itself, solved later.
    """
    if not IS_CUPY_BACKEND:
        return A
    p = A.shape[-1]
    L = xp.array(A, dtype=xp.float64, order="C", copy=True)
    nb = int(np.prod(A.shape[:-2], dtype=np.int64))
    if nb and p >= SPD_COOP_MIN_P:  # one block per system (same bits): one thread is too serial
        _kern_spd_factor_coop((nb,), (min(128, -(-p // 32) * 32),), (L, np.int64(nb), np.int32(p)))
    elif nb:
        _kern_spd_factor(((nb + 127) // 128,), (128,), (L, np.int64(nb), np.int32(p)))
    return L


def spd_solve_factored(L: Array, B: Array) -> Array:
    """``A^-1 B`` from ``L = spd_factor(A)``; bit-identical to ``spd_solve(A, B)`` on both backends."""
    if not IS_CUPY_BACKEND:
        return xp.linalg.solve(L, B)
    p, q = L.shape[-1], B.shape[-1]
    lead = L.shape[:-2]
    X = xp.array(xp.broadcast_to(B, (*lead, p, q)), dtype=xp.float64, order="C", copy=True)
    nb = int(np.prod(lead, dtype=np.int64))
    if nb:
        _kern_spd_solve_factored(((nb + 127) // 128,), (128,), (xp.ascontiguousarray(L), X, np.int64(nb),
                                                                 np.int32(p), np.int32(q)))  # fmt: skip
    return X


def det_einsum(spec: str, A: Array, B: Array) -> Array:
    """
    ``xp.einsum(spec, A, B)`` for the batched matrix-vector specs of training (``"xab,xb->xa"``,
    e.g. ``"nhd,nd->nh"``, and ``"np,npf->nf"``); NumPy runs the einsum itself, so its bits
    are unchanged, and CuPy routes through ``det_matmul`` for the same reason as there.
    """
    if not IS_CUPY_BACKEND:
        return xp.einsum(spec, A, B)
    ins, out = spec.split("->")
    sa, sb = ins.split(",")
    if len(sa) == 3 and len(sb) == 2 and sa[0] == sb[0] and sa[2] == sb[1] and out == sa[:2]:
        return det_matmul(A, B[..., None])[..., 0]
    if len(sa) == 2 and len(sb) == 3 and sa[0] == sb[0] and sa[1] == sb[1] and out == sa[0] + sb[2]:
        return det_matmul(A[:, None, :], B)[:, 0, :]
    raise ValueError(f"det_einsum does not support {spec!r}")


def solve_regularized_covariance_inplace(auto_cov, cross_cov, D, FD):
    """
    Solves ``(auto_cov + lambda I) X = cross_cov`` while minimizing peak GPU
    allocations. On CuPy, both inputs are overwritten in-place and ``cross_cov``
    becomes the solution. NumPy uses a vectorized solve.
    """
    if IS_CUPY_BACKEND:
        K = int(auto_cov.shape[0])
        traces = xp.trace(auto_cov, axis1=1, axis2=2)
        block = 256
        grid = (K + block - 1) // block
        _kern_affine_reg((grid,), (block,), (auto_cov, traces, TINY_EPS, D, K))
        _kern_cholesky((grid,), (block,), (auto_cov, cross_cov, D, FD, K))
        return cross_cov

    tr = xp.trace(auto_cov, axis1=1, axis2=2)
    lam = xp.maximum(1e-8 * tr, TINY_EPS)
    reg = lam[:, None, None] * xp.eye(D, dtype=xp.float64)[None, :, :]
    return xp.linalg.solve(auto_cov + reg, cross_cov)


def lsq_hessian_in_eigenbasis(C: Array, S: Array) -> tuple[Array, Array]:
    """
    Least-squares Hessian from gradient residuals, in the eigenbasis of ``C``.

    Solves ``C H + H C = S`` for symmetric ``H`` (``C``: (K, d, d) point covariance,
    ``S``: (K, d, d) symmetrized gradient-residual moments). Returns ``(U, F)``
    with ``H = U F U^T``, where ``U`` holds the eigenvectors of ``C``.
    """
    dvals, U = xp.linalg.eigh(C)
    dvals = xp.maximum(dvals, TINY_EPS)
    Gtilde = det_matmul(det_matmul(U.transpose(0, 2, 1), S), U)  # (K, d, d)
    F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])  # (K, d, d)
    return U, F_mat
