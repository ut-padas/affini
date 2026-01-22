"""Trivial Quadratic function and its derivatives."""

from src import Array, xp

C = 1.0  # Constant term
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
        G: (N, d) gradients at X
    """

    N, d = X.shape
    w = xp.ones((d,), dtype=X.dtype)
    G = w[None, :] + 2 * A * X  # (N, d)
    return G
