"""Rosenbrock function and its derivatives."""

from array_api_compat import array_namespace
from array_api_compat.common._typing import Array

ROSENBROCK_SCALE = 100.0


def rosenbrock(X: Array, a: float = ROSENBROCK_SCALE):
    """
    Computes the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix
        a: scale parameter

    Returns:
        F: (n, 1) function values at X
    """
    xp = array_namespace(X)
    x_i = X[:, :-1]
    x_next = X[:, 1:]
    F = xp.sum(
        a * (x_next - x_i**2.0) ** 2.0 + (1.0 - x_i) ** 2.0,
        axis=1,
        keepdims=True,
    )
    return F


def rosenbrock_grad(X: Array, a: float = ROSENBROCK_SCALE):
    """
    Computes the gradient of the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix
        a: scale parameter

    Returns:
        J: (n, 1, d) Jacobian at X
    """
    xp = array_namespace(X)

    n_pts, n_dim = X.shape
    J = xp.zeros((n_pts, 1, n_dim), dtype=X.dtype)

    J[:, 0, :-1] += 2 * (X[:, :-1] - 1)
    J[:, 0, :-1] -= 4 * a * X[:, :-1] * (X[:, 1:] - X[:, :-1] ** 2)
    J[:, 0, 1:] += 2 * a * (X[:, 1:] - X[:, :-1] ** 2)
    return J


def rosenbrock_hess(X: Array, a: float = ROSENBROCK_SCALE):
    """
    Computes the Hessian of the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix
        a: scale parameter

    Returns:
        H: (n, d, d) Hessian at X
    """
    xp = array_namespace(X)

    n_pts, n_dim = X.shape
    H = xp.zeros((n_pts, n_dim, n_dim), dtype=X.dtype)

    for i in range(n_dim):
        if i < n_dim - 1:
            if i > 0:
                H[:, i, i] += 2 * a
            H[:, i, i] += 12 * a * X[:, i] ** 2 - 4 * a * X[:, i + 1] + 2
            H[:, i, i + 1] += -4 * a * X[:, i]
            H[:, i + 1, i] += -4 * a * X[:, i]
        else:
            H[:, i, i] += 2 * a
    return H
