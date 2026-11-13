"""Anisotropic Quadratic function and its derivatives."""

import numpy as np
from array_api_compat import array_namespace

from src import Array

SEED = 0
DECAY_EXPONENT = 3.0
BETA = 1.0


def get_random_rotation(d: int, dtype) -> np.ndarray:
    """
    Creates a random rotation matrix using QR decomposition.

    Problem constants are always generated with NumPy (as a host array), so the
    function is identical on every backend; callers move it to their backend.
    A local RandomState reproduces the historical `np.random.seed(SEED)` stream
    without touching global RNG state.
    """
    rng = np.random.RandomState(SEED)
    A = rng.randn(d, d).astype(dtype)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _get_params(
    d: int, dtype, decay_exponent: float = DECAY_EXPONENT
) -> np.ndarray:
    """
    Computes the symmetric matrix H = Q * Lambda * Q^T.

    Returns:
        H: (d, d) quadratic form matrix, as a NumPy (host) array
    """
    # BL: there is no linear term (w^T x) any more, so the minimum is at zero.

    # 1. Eigenvalues: Lambda_ii = i^(-decay_exponent)
    indices = np.arange(1, d + 1, dtype=dtype)
    lambdas = np.power(indices, -decay_exponent)

    # 2. Rotation: Q
    Q = get_random_rotation(d, dtype)

    # 3. Construct H = Q @ diag(lambdas) @ Q.T
    H = (Q * lambdas[None, :]) @ Q.T

    return H


def aniso_quadratic(
    X: Array, beta: float = BETA, decay_exponent: float = DECAY_EXPONENT
) -> Array:
    """
    Computes the Rotated Anisotropic Quadratic function.
    f(x) = beta + x^T H x   (minimum f = beta at x = 0)

    Args:
        X: (N, d) data matrix
        beta: Constant offset term (default: BETA)
        decay_exponent: Eigenvalue decay exponent (default: DECAY_EXPONENT)
    Returns:
        F: (N, 1) function values
    """
    xp = array_namespace(X)
    N, d = X.shape
    H = xp.asarray(_get_params(d, X.dtype, decay_exponent))

    # Quadratic term: x^T H x
    quad_term = xp.sum((X @ H) * X, axis=1)

    # Combine: beta + quadratic
    res = beta + quad_term

    return res[:, None]  # (N, 1)


def aniso_quadratic_grad(X: Array, decay_exponent: float = DECAY_EXPONENT) -> Array:
    """
    Computes gradient.
    grad f(x) = 2Hx

    Args:
        X: (N, d) data matrix
        decay_exponent: Eigenvalue decay exponent (default: DECAY_EXPONENT)
    Returns:
        G: (N, 1, d) gradients
    """
    xp = array_namespace(X)
    N, d = X.shape
    H = xp.asarray(_get_params(d, X.dtype, decay_exponent))

    # term: 2 * H * x
    # (N, d) @ (d, d) -> (N, d)
    # Note: H is symmetric, so (H @ x) or (x @ H) results in same vector,
    # but for (N, d) shape we do X @ H.
    grad_quad = 2.0 * (X @ H)

    return grad_quad[:, None, :]  # (N, 1, d)


def aniso_quadratic_hess(X: Array, decay_exponent: float = DECAY_EXPONENT) -> Array:
    """
    Computes Hessian.
    Hess f(x) = 2H

    Args:
        X: (N, d) data matrix
        decay_exponent: Eigenvalue decay exponent (default: DECAY_EXPONENT)
    Returns:
        H_vals: (N, d, d) Hessians
    """
    xp = array_namespace(X)
    N, d = X.shape
    H = xp.asarray(_get_params(d, X.dtype, decay_exponent))

    # The Hessian is constant: 2 * H
    H_const = 2.0 * H

    # Repeat for all N points
    return xp.broadcast_to(H_const, (N, *H_const.shape))
