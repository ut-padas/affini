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
    _kern_spd_factor_shared = _linalg.get_function("spd_factor_shared")
    # The packed triangle of the largest system spd_factor_shared can hold in one block's shared memory.
    _SPD_SHARED_BYTES = int(xp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]) - int(
        _kern_spd_factor_shared.shared_size_bytes)  # the opt-in limit counts the kernel's static flag too
    _kern_spd_factor_shared.max_dynamic_shared_size_bytes = _SPD_SHARED_BYTES
    _kern_spd_solve_factored = _linalg.get_function("spd_solve_factored")
    _kern_affine_reg = _linalg.get_function("fused_affine_fit_regularize")
    _kern_cholesky = _linalg.get_function("batched_inplace_cholesky_solve")


TINY_EPS = 1e-12

# From this many unknowns, SPD systems are factored by one block each: in shared memory
# (spd_factor_shared) while the packed triangle fits, else in global memory (spd_factor_coop).
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


# leading_eigvec: 2**10 = 1,024 power steps leave an angle of about (lambda2 / lambda1)**1024 to the
# leading eigenvector: below 3e-3 degrees for lambda2 / lambda1 <= 0.99, and AFINNI's split matrices
# (4,000 recorded per d at d = 20, 60, 100) had at most 0.9934. A nearly degenerate pair still gives
# a unit vector in the span of the top two eigenvectors (within 2e-15), along which the variance is
# within a factor lambda2 / lambda1 of the largest: an equally valid split direction. The steps
# v <- C v / |C v| afterwards apply C itself, whose rounding the squarings compound.
LEADING_EIGVEC_SQUARINGS = 10
LEADING_EIGVEC_REFINE_STEPS = 3


def leading_eigvec(C: Array) -> Array:
    """
    (K, d) unit eigenvectors of the largest eigenvalue of symmetric positive semidefinite ``C``
    (K, d, d): ``LEADING_EIGVEC_SQUARINGS`` squarings ``P <- P P / max|P P|`` (a power iteration of
    ``2**LEADING_EIGVEC_SQUARINGS`` steps), the column of ``P`` at its largest diagonal entry,
    ``LEADING_EIGVEC_REFINE_STEPS`` steps ``v <- C v / |C v|``, and the sign that makes the
    first largest-magnitude component positive. A zero matrix gives the first coordinate axis.
    Products are ``det_matmul`` and the scalings maxima, so a matrix's bits do not depend on the
    rest of the stack; both backends run the same steps.
    """
    K, d = int(C.shape[0]), int(C.shape[-1])
    P = C / xp.maximum(xp.max(xp.abs(C), axis=(1, 2)), TINY_EPS)[:, None, None]
    for _ in range(LEADING_EIGVEC_SQUARINGS):
        P = det_matmul(P, P)
        P = P / xp.maximum(xp.max(xp.abs(P), axis=(1, 2)), TINY_EPS)[:, None, None]
    j = xp.argmax(xp.diagonal(P, axis1=1, axis2=2), axis=1)
    v = xp.take_along_axis(P, xp.broadcast_to(j[:, None, None], (K, d, 1)), axis=2)  # (K, d, 1)
    axis = xp.zeros((K, d, 1))
    axis[:, 0] = 1.0
    for step in range(LEADING_EIGVEC_REFINE_STEPS + 1):
        if step:
            v = det_matmul(C, v)
        n2 = det_matmul(v.transpose(0, 2, 1), v)  # (K, 1, 1)
        v = xp.where(n2 > 0, v / xp.sqrt(xp.where(n2 > 0, n2, 1.0)), axis)
    v = v[:, :, 0]
    i = xp.argmax(xp.abs(v), axis=1)
    return v * xp.where(xp.take_along_axis(v, i[:, None], axis=1) < 0, -1.0, 1.0)


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
    shared = 8 * (p * (p + 1) // 2 + p)
    if nb and p >= SPD_COOP_MIN_P and shared <= _SPD_SHARED_BYTES:  # one block per system, in shared memory (same bits)
        _kern_spd_factor_shared((nb,), (256,), (L, np.int64(nb), np.int32(p)), shared_mem=shared)
    elif nb and p >= SPD_COOP_MIN_P:  # one block per system (same bits): one thread is too serial
        _kern_spd_factor_coop((nb,), (min(128, -(-p // 32) * 32),), (L, np.int64(nb), np.int32(p)))
    elif nb:
        _kern_spd_factor(((nb + 127) // 128,), (128,), (L, np.int64(nb), np.int32(p)))
    return L


def spd_solve_factored(L: Array, B: Array) -> Array:
    """
    ``A^-1 B`` from ``L = spd_factor(A)``; bit-identical to ``spd_solve(A, B)`` on both backends.
    CuPy: one thread per (system, right-hand side), so an inverse (q = p) is not one thread's
    serial p^3 work.
    """
    if not IS_CUPY_BACKEND:
        return solve_or_nan(L, B)
    p, q = L.shape[-1], B.shape[-1]
    lead = L.shape[:-2]
    X = xp.array(xp.broadcast_to(B, (*lead, p, q)), dtype=xp.float64, order="C", copy=True)
    nb = int(np.prod(lead, dtype=np.int64))
    if nb and q:
        _kern_spd_solve_factored(((nb * q + 127) // 128,), (128,), (xp.ascontiguousarray(L), X, np.int64(nb),
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
    L = spd_factor(G)  # one factorization for the solution and the inverse
    X = spd_solve_factored(L, B)
    if K == 0:
        return X
    inv = spd_solve_factored(L, xp.broadcast_to(xp.eye(p), (K, p, p)))
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
