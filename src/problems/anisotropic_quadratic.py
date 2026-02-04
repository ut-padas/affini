"""Anisotropic Quadratic function and its derivatives."""

from src import Array, xp

C = 10.0  # Constant term
ANISO_RATIO = 100.0


def _get_scales(d: int, dtype) -> Array:
    """
    Creates diagonal scales [1, R, R^2, ...] to force high anisotropy.
    For d=2, scales=[1, 100]. The valley is 100x narrower in y than x.
    """
    exponents = xp.arange(d, dtype=dtype)
    return xp.power(ANISO_RATIO, exponents)


def aniso_quadratic(X: Array) -> Array:
    """
    Computes the Anisotropic Quadratic function.
    f(x) = C + sum(x) + sum(s_i * x_i^2)

    Args:
        X: (N, d) data matrix
    Returns:
        F: (N, 1) function values
    """
    N, d = X.shape
    scales = _get_scales(d, X.dtype)  # (d,)

    # Linear term: sum(x)
    linear_term = xp.sum(X, axis=1, keepdims=True)

    # Quadratic term: x^T Lambda x = sum(s_i * x_i^2)
    quad_term = xp.sum(scales[None, :] * (X**2), axis=1, keepdims=True)

    return C + linear_term + quad_term


def aniso_quadratic_grad(X: Array) -> Array:
    """
    Computes gradient.
    g_i(x) = 1 + 2 * s_i * x_i

    Returns:
        G: (N, 1, d) gradients
    """
    N, d = X.shape
    scales = _get_scales(d, X.dtype)  # (d,)

    # Broadcasting: (N, d) = (1, d) + (1, d) * (N, d)
    G = 1.0 + 2.0 * scales[None, :] * X
    return G[:, None, :]


def aniso_quadratic_hess(X: Array) -> Array:
    """
    Computes Hessian.
    H = diag(2 * s)

    Returns:
        H: (N, d, d) Hessians
    """
    N, d = X.shape
    scales = _get_scales(d, X.dtype)  # (d,)

    # Create diagonal matrix (d, d)
    H_base = xp.diag(2.0 * scales)

    # Repeat for all N points
    return xp.repeat(H_base[None, :, :], N, axis=0)
