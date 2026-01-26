"""Rastrigin function and its derivatives."""

from src import Array, xp

RASTRIGIN_A = 10.0  # Default constant term


def rastrigin(X: Array, A=RASTRIGIN_A) -> Array:
    """
    Rastrigin function (N-D).
    Global minimum at X=0, f(X)=0.
    Standard domain: [-5.12, 5.12]

    Args:
        X: (N, D) data matrix
        A: constant term

    Returns:
        F: (N, 1) function values at X
    """
    # X: (N, D)
    d = X.shape[1]

    # sum(x^2 - A * cos(2pi*x))
    sum_term = xp.sum(X**2 - A * xp.cos(2.0 * xp.pi * X), axis=1)

    return (A * d + sum_term)[:, None]  # (N, 1)


def rastrigin_grad(X: Array, A=RASTRIGIN_A) -> Array:
    """
    Analytical gradient of Rastrigin.

    Args:
        X: (N, D) data matrix
        A: constant term

    Returns:
        G: (N, 1, D) gradients at X
    """
    # df/dx_i = 2x_i + 2pi*A*sin(2pi*x_i)
    # X: (N, D)

    grad = 2.0 * X + 2.0 * xp.pi * A * xp.sin(2.0 * xp.pi * X)

    # Ensure shape (N, 1, D) for the surrogate if needed,
    return grad[:, None, :]
