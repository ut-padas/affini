"""Affine function and its gradient implementation."""

import cupy as cp
import numpy as np
from array_api_compat import array_namespace

from src import _HAS_GPU

AFFINE_DATA_DIM = 20
AFFINE_F_DIM = 10

np.random.seed(0)
A_NUMPY = np.random.uniform(0, 1, size=(AFFINE_F_DIM, AFFINE_DATA_DIM))
c_NUMPY = np.random.uniform(0, 1, size=(AFFINE_F_DIM,))

if _HAS_GPU:
    A_CUPY = cp.asarray(A_NUMPY)
    c_CUPY = cp.asarray(c_NUMPY)


def affine(X):
    """
    Affine function to simulate a linear transformation.

    Args:
        X: (n_samples, 20) data matrix where each row represents a set of input parameters.

    Returns:
        F: (n_samples, 10) function values at X
    """
    xp = array_namespace(X)
    if _HAS_GPU and xp.__name__ == "array_api_compat.cupy":
        return X @ A_CUPY.T + c_CUPY  # Shape (n_samples, 10)
    else:
        return X @ A_NUMPY.T + c_NUMPY


def affine_grad(X):
    """
    Gradient of the Affine function.

    Args:
        X: (n_samples, 20) data matrix where each row represents a set of input parameters.

    Returns:
        J: (n_samples, 10, 20) Jacobian at X
    """
    xp = array_namespace(X)
    if _HAS_GPU and xp.__name__ == "array_api_compat.cupy":
        n_samples = X.shape[0]
        J = cp.tile(A_CUPY, (n_samples, 1, 1))  # Shape (n_samples, 10, 20)
        return J
    else:
        n_samples = X.shape[0]
        J = np.tile(A_NUMPY, (n_samples, 1, 1))  # Shape (n_samples, 10, 20)
        return J
