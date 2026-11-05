"""Van der Pol oscillator and its gradient."""

import numpy as np

from src import Array


def vanderpol(X: Array, mu=5.0) -> Array:
    """
    Van der Pol oscillator implementation.

    Args:
        X: (N, 2) data matrix where each row represents a state of the system.
        mu: Nonzero oscillator parameter controlling the time scales.

    Returns:
        dXdt: (N, 2) time derivatives at each state in X.
    """
    x = X[:, 0]
    y = X[:, 1]
    dxdt = mu * (x - (x**3 / 3) - y)
    dydt = x / mu
    return np.column_stack((dxdt, dydt))


def vanderpol_grad(X, mu=5.0) -> Array:
    """
    Jacobian of the Van der Pol vector field.

    Args:
        X: (N, 2) data matrix where each row represents a state of the system.
        mu: Nonzero oscillator parameter controlling the time scales.

    Returns:
        J: (N, 2, 2) Jacobian matrices at each state in X.
    """
    N = X.shape[0]
    x = X[:, 0]
    J = np.zeros((N, 2, 2))

    # df1/dx = mu * (1 - x^2)
    J[:, 0, 0] = mu * (1 - x**2)

    # df1/dy = -mu
    J[:, 0, 1] = -mu

    # df2/dx = 1/mu
    J[:, 1, 0] = 1.0 / mu

    # df2/dy = 0 (already zero from np.zeros)
    return J
