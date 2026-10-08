"""Anisotropic Quadratic function and its derivatives."""

from src import Array, xp

SEED = 0
DECAY_BASE = 5.0
BETA = 1.0


def _get_random_rotation(d: int, dtype) -> Array:
    """
    Creates a random rotation matrix using QR decomposition.
    """
    xp.random.seed(SEED)
    A = xp.random.randn(d, d).astype(dtype)
    Q, _ = xp.linalg.qr(A)
    if xp.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _get_params(
    d: int, dtype, decay_base: float = DECAY_BASE
) -> Array:
    """
    Computes the symmetric matrix H = Q * Lambda * Q^T.

    Returns:
        w: (d,) linear weight vector
        H: (d, d) quadratic form matrix
    """
    # BL: we are removing the linear weight vector w so the minimum is at zero.
    # 1. Linear weights: w = [1, 1, ..., 1]
    # w = xp.ones((d,), dtype=dtype)

    # 2. Eigenvalues: Lambda_ii = i^(-decay_exponent)
    indices = xp.arange(1, d + 1, dtype=dtype)
    lambdas = xp.power(DECAY_BASE, -indices)

    # 3. Rotation: Q
    Q = _get_random_rotation(d, dtype)

    # 4. Construct H = Q @ diag(lambdas) @ Q.T
    H = (Q * lambdas[None, :]) @ Q.T

    return H


def aniso_quadratic(
    X: Array, beta: float = BETA, decay_base: float = DECAY_BASE
) -> Array:
    """
    Computes the Rotated Anisotropic Quadratic function.
    f(x) = beta + w^T x + x^T H x

    Args:
        X: (N, d) data matrix
        beta: Constant offset term (default: DEFAULT_BETA)
        decay_exponent: Eigenvalue decay exponent (default: DECAY_EXPONENT)
    Returns:
        F: (N, 1) function values
    """
    N, d = X.shape
    H = _get_params(d, X.dtype, decay_base)

    # Linear term: w^T x -> (N, d) @ (d,) -> (N,)
    # linear_term = X @ w

    # Quadratic term: x^T H x
    quad_term = xp.sum((X @ H) * X, axis=1)

    # Combine: beta + linear + quadratic
    # res = beta + linear_term + quad_term
    res = beta + quad_term

    return res[:, None]  # (N, 1)


def aniso_quadratic_grad(X: Array, decay_base: float = DECAY_BASE) -> Array:
    """
    Computes gradient.
    grad f(x) = w + 2Hx

    Args:
        X: (N, d) data matrix
        decay_exponent: Eigenvalue decay exponent (default: DECAY_EXPONENT)
    Returns:
        G: (N, 1, d) gradients
    """
    N, d = X.shape
    H = _get_params(d, X.dtype, decay_base)

    # term: 2 * H * x
    # (N, d) @ (d, d) -> (N, d)
    # Note: H is symmetric, so (H @ x) or (x @ H) results in same vector,
    # but for (N, d) shape we do X @ H.
    grad_quad = 2.0 * (X @ H)

    return grad_quad[:, None, :]  # (N, 1, d)


def aniso_quadratic_hess(X: Array, decay_base: float = DECAY_BASE) -> Array:
    """
    Computes Hessian.
    Hess f(x) = 2H

    Args:
        X: (N, d) data matrix
        decay_exponent: Eigenvalue decay exponent (default: DECAY_EXPONENT)
    Returns:
        H_vals: (N, d, d) Hessians
    """
    N, d = X.shape
    H = _get_params(d, X.dtype, decay_base)

    # The Hessian is constant: 2 * H
    H_const = 2.0 * H

    # Repeat for all N points
    return xp.broadcast_to(H_const, (N, *H_const.shape))
