"""Affine function and its gradient implementation."""

from src import Array, xp

AFFINE_DATA_DIM = 20
AFFINE_F_DIM = 10

xp.random.seed(0)
GLOBAL_A = xp.random.uniform(0, 1, size=(AFFINE_F_DIM, AFFINE_DATA_DIM))
GLOBAL_c = xp.random.uniform(0, 1, size=(AFFINE_F_DIM,))


def affine(X: Array) -> Array:
    """
    Affine function to simulate a linear transformation.

    Args:
        X: (N, 20) data matrix where each row represents a set of input parameters.

    Returns:
        F: (N, 10) function values at X
    """
    return X @ GLOBAL_A.T + GLOBAL_c


def affine_grad(X):
    """
    Gradient of the Affine function.

    Args:
        X: (N, 20) data matrix where each row represents a set of input parameters.

    Returns:
        J: (N, 10, 20) Jacobian at X
    """
    N = X.shape[0]
    J = xp.tile(GLOBAL_A, (N, 1, 1))  # (N, 10, 20)
    return J
