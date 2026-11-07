"""A simple two-output quadratic with axis-aligned, differently scaled Hessians, for tests."""

import numpy as np
from array_api_compat import array_namespace

from src import Array

# Problem constants: NumPy (host) arrays, moved to the backend of X on each call.
H1 = np.diag(np.array([10, 1]))
H2 = np.diag(np.array([1, 10]))


def vector_quadratic_2(X: Array) -> Array:
    """
    Computes f(x) = (<x, H1*x>, <x, H2*x>), which maps R2 to R2.
    For now assume H1 and H2 are diagonal matrices.
    Takes in X with shape (N, d) and returns matrix (N, 2).
    """
    xp = array_namespace(X)
    h1, h2 = xp.asarray(H1), xp.asarray(H2)
    N, d = X.shape
    val1 = xp.sum(X * (h1 @ X.T).T, axis=1)
    val2 = xp.sum(X * (h2 @ X.T).T, axis=1)
    return xp.stack([val1, val2], axis=1)


def vector_quadratic_2_grad(X: Array) -> Array:
    """
    Computes the gradient: f'(x) = (2*H1*x, 2*H2*x)
    Takes in X with shape (N, d) and returns tensor (N, 2, d).
    """
    xp = array_namespace(X)
    h1, h2 = xp.asarray(H1), xp.asarray(H2)
    g1 = 2.0 * (X @ h1) * X
    g2 = 2.0 * (X @ h2) * X
    return xp.stack([g1, g2], axis=1)


def vector_quadratic_2_hess(X: Array) -> Array:
    """
    Computes the Hessian: f''(x) = (2*H1, 2*H2)
    Takes in X with shape (N, d) and returns tensor (N, 2, d, d).
    """
    xp = array_namespace(X)
    h1, h2 = xp.asarray(H1), xp.asarray(H2)
    N, d = X.shape
    H1_full = 2.0 * h1  # Shape (d, d)
    H2_full = 2.0 * h2  # Shape (d, d), d is the input dimension
    H1_tiled = xp.repeat(H1_full[None, None, :, :], N, axis=0)  # Shape (N, 1, d, d)
    H2_tiled = xp.repeat(H2_full[None, None, :, :], N, axis=0)  # Shape (N, 1, d, d)
    return xp.concatenate([H1_tiled, H2_tiled], axis=1)  # Shape (N, 2, d, d)
