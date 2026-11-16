"""Small batched dense algebra: SPD factor and solves, products with a fixed inner order, regularized fits."""

import contextlib

import numpy as np

from src import IS_CUPY_BACKEND, Array, kernels, to_host, xp

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
        return solve_or_nan(A, B)
    p, q = A.shape[-1], B.shape[-1]
    lead = A.shape[:-2]
    L = xp.array(A, dtype=xp.float64, order="C", copy=True)
    X = xp.array(xp.broadcast_to(B, (*lead, p, q)), dtype=xp.float64, order="C", copy=True)
    nb = int(np.prod(lead, dtype=np.int64))
    if nb:
        _kern_spd_solve(((nb + 127) // 128,), (128,), (L, X, np.int64(nb), np.int32(p), np.int32(q)))
    return X


def solve_or_nan(A: Array, B: Array) -> Array:
    """
    NumPy's stacked ``solve``, with an all-NaN solution for an exactly singular system instead of
    an exception for the whole stack (the CuPy kernels' failure value); other systems keep the
    bits of one stacked call, since LAPACK solves each system on its own.
    """
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        pass
    lead = A.shape[:-2]
    A3 = A.reshape(-1, *A.shape[-2:])
    B3 = np.broadcast_to(B, (*lead, *B.shape[-2:])).reshape(-1, *B.shape[-2:])
    X = np.full(B3.shape, np.nan)
    for k in range(A3.shape[0]):
        with contextlib.suppress(np.linalg.LinAlgError):
            X[k] = np.linalg.solve(A3[k], B3[k])
    return X.reshape(*lead, *B.shape[-2:])


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
        return solve_or_nan(L, B)
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


# The reference ART's auto_rcond_threshold: a normal-equations Cholesky solution is kept when the
# 1-norm reciprocal condition number of the Gram is at least this, else the rows are refitted by QR.
LSTSQ_RCOND_THRESHOLD = 1e-10


def lstsq_auto(G: Array, B: Array, design: Array, rhs: Array, starts: Array, counts: Array) -> Array:
    """
    Unregularized least squares for K stacked systems, as the reference ART's ``ridge=0``,
    ``solver="auto"``: ``G_k^-1 B_k`` by Cholesky of the normal equations wherever the Gram's
    1-norm reciprocal condition number is at least ``LSTSQ_RCOND_THRESHOLD``; elsewhere (and
    where the Cholesky fails) system ``k`` is solved from its own rows ``design[starts[k] :
    starts[k] + counts[k]]`` (right-hand side ``rhs``) by QR (LAPACK ``gelsy``), then by SVD
    (``gelsd``) if QR fails or is not finite. ``G`` (K, p, p) and ``B`` (K, p, q) must be the
    normal equations of those rows.

    The condition number is computed exactly from the Cholesky inverse, where LAPACK's ``dpocon``
    (the reference's) estimates it; they differ only for systems near the threshold. The QR and
    SVD fallbacks run on the host (one LAPACK call per system, in cluster order), so they are
    deterministic on both backends; they are meant for the few ill-conditioned systems.
    """
    from scipy import linalg as sla

    K, p = int(G.shape[0]), int(G.shape[-1])
    X = spd_solve(G, B)
    if K == 0:
        return X
    inv = spd_solve(G, xp.broadcast_to(xp.eye(p), (K, p, p)))
    norm_1 = xp.max(xp.sum(xp.abs(G), axis=-2), axis=-1)
    norm_1_inv = xp.max(xp.sum(xp.abs(inv), axis=-2), axis=-1)
    rcond = 1.0 / (norm_1 * norm_1_inv)
    ok = xp.isfinite(rcond) & (rcond >= LSTSQ_RCOND_THRESHOLD) & xp.all(xp.isfinite(X), axis=(1, 2))
    bad = np.flatnonzero(~np.asarray(to_host(ok)))
    if bad.size == 0:
        return X
    starts_h, counts_h = np.asarray(to_host(starts)), np.asarray(to_host(counts))
    for k in bad.tolist():  # the few ill-conditioned systems, in cluster order
        rows = slice(int(starts_h[k]), int(starts_h[k] + counts_h[k]))
        A_k = np.asarray(to_host(design[rows]), dtype=np.float64)
        y_k = np.asarray(to_host(rhs[rows]), dtype=np.float64)
        try:
            sol = sla.lstsq(A_k, y_k, lapack_driver="gelsy", check_finite=False)[0]
            if not np.all(np.isfinite(sol)):
                raise sla.LinAlgError("QR solution is not finite")
        except (sla.LinAlgError, ValueError):
            sol = sla.lstsq(A_k, y_k, lapack_driver="gelsd", check_finite=False)[0]
        X[k] = xp.asarray(sol.reshape(p, -1))
    return X


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
