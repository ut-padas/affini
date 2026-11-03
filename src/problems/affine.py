"""Affine function and its gradient implementation."""

import numpy as np
from array_api_compat import array_namespace

from src import Array

AFFINE_DATA_DIM = 20
AFFINE_F_DIM = 10

# Problem constants: NumPy (host) arrays, identical on every backend. A local
# RandomState reproduces the historical `np.random.seed(0)` stream without
# reseeding the global RNG at import time.
_rng = np.random.RandomState(0)
GLOBAL_A = _rng.uniform(0, 1, size=(AFFINE_F_DIM, AFFINE_DATA_DIM))
GLOBAL_c = _rng.uniform(0, 1, size=(AFFINE_F_DIM,))
del _rng


def affine(X: Array) -> Array:
    """
    Affine function to simulate a linear transformation.

    Args:
        X: (N, 20) data matrix where each row represents a set of input parameters.

    Returns:
        F: (N, 10) function values at X
    """
    xp = array_namespace(X)
    return X @ xp.asarray(GLOBAL_A).T + xp.asarray(GLOBAL_c)


def affine_grad(X):
    """
    Gradient of the Affine function.

    Args:
        X: (N, 20) data matrix where each row represents a set of input parameters.

    Returns:
        J: (N, 10, 20) Jacobian at X
    """
    xp = array_namespace(X)
    N = X.shape[0]
    J = xp.tile(xp.asarray(GLOBAL_A), (N, 1, 1))  # (N, 10, 20)
    return J
