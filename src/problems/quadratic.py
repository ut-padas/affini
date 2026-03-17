"""Trivial Quadratic function and its derivatives."""

from src import Array, xp

C = 10.0  # Constant term
A = 0.1  # Quadratic term coefficient


def quadratic(X: Array) -> Array:
    """
    Computes the Quadratic function for given input points.
    f(x) = c + w^T x + a x^T x, where w = ones(d)

    Args:
        X: (N, d) data matrix
    Returns:
        F: (N, 1) function values at X
    """
    N, d = X.shape
    w = xp.ones((d, 1), dtype=X.dtype)
    XTX = xp.sum(X**2, axis=1, keepdims=True)  # (N, 1)
    F = C + X @ w + A * XTX  # (N, 1)
    return F


def quadratic_grad(X: Array) -> Array:
    """
    Computes the gradient of the Quadratic function for given input points.
    grad f(x) = w + 2 a x, where w = ones(d)
    Args:
        X: (N, d) data matrix

    Returns:
        G: (N,1 d) gradients at X
    """

    N, d = X.shape
    w = xp.ones((d,), dtype=X.dtype)
    G = w[None, :] + 2 * A * X  # (N, d)
    return G[:, None, :]  # (N, 1, d)


def quadratic_hess(X: Array) -> Array:
    """
    Computes the Hessian of the Quadratic function for given input points.
    Hessian f(x) = 2 a I_d
    Args:
        X: (N, d) data matrix

    Returns:
        H: (N, d, d) Hessians at X
    """
    N, d = X.shape
    H = 2 * A * xp.eye(d, dtype=X.dtype)[None, :, :]  # (1, d, d)
    return xp.repeat(H, N, axis=0)  # (N, d, d)
