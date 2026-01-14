"""Rosenbrock function and its derivatives."""

from array_api_compat import array_namespace

ROSENBROCK_SCALE = 100.0


def rosenbrock(X):
    """
    Computes the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix

    Returns:
        F: (n, 1) function values at X
    """
    xp = array_namespace(X)
    x_i = X[:, :-1]
    x_next = X[:, 1:]
    F = xp.sum(
        ROSENBROCK_SCALE * (x_next - x_i**2.0) ** 2.0 + (1.0 - x_i) ** 2.0,
        axis=1,
        keepdims=True,
    )
    return F


def rosenbrock_grad(X):
    """
    Computes the gradient of the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix

    Returns:
        J: (n, 1, d) Jacobian at X
    """
    xp = array_namespace(X)

    n_pts, n_dim = X.shape
    J = xp.zeros((n_pts, 1, n_dim), dtype=X.dtype)

    # The term (1 - x_i)^2 only exists for i = 0 to d-2
    J[:, 0, :-1] += 2 * (X[:, :-1] - 1)

    # Term from 100*(x_{i+1} - x_i^2)^2 for i < d-1 (derivative w.r.t x_i)
    # This affects 0 to d-2
    J[:, 0, :-1] -= 4 * ROSENBROCK_SCALE * X[:, :-1] * (X[:, 1:] - X[:, :-1] ** 2)

    # Term from 100*(x_i - x_{i-1}^2)^2 for i > 0 (derivative w.r.t x_{i+1})
    # This affects 1 to d-1
    J[:, 0, 1:] += 2 * ROSENBROCK_SCALE * (X[:, 1:] - X[:, :-1] ** 2)

    return J


def rosenbrock_hess(X):
    """
    Computes the Hessian of the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix

    Returns:
        H: (n, d, d) Hessian at X
    """
    xp = array_namespace(X)

    n_pts, n_dim = X.shape
    H = xp.zeros((n_pts, n_dim, n_dim), dtype=X.dtype)

    for i in range(n_dim):
        if i < n_dim - 1:
            H[:, i, i] += (
                12 * ROSENBROCK_SCALE * X[:, i] ** 2
                - 4 * ROSENBROCK_SCALE * X[:, i + 1]
                + 2
            )
            if i > 0:
                H[:, i, i] += 2 * ROSENBROCK_SCALE
            H[:, i, i + 1] += -4 * ROSENBROCK_SCALE * X[:, i]
            H[:, i + 1, i] += -4 * ROSENBROCK_SCALE * X[:, i]
        else:
            H[:, i, i] += 2 * ROSENBROCK_SCALE
    return H
