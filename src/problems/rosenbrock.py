"""Rosenbrock function and its derivatives."""

import cupy as cp
import numpy as np


def rosenbrock_cupy(X: cp.ndarray) -> cp.ndarray:
    """
    Computes the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix

    Returns:
        F: (n, 1) function values at X
    """
    x_i = X[:, :-1]
    x_next = X[:, 1:]
    F = cp.sum(100 * (x_next - x_i**2) ** 2 + (1 - x_i) ** 2, axis=1, keepdims=True)
    return F


def rosenbrock_numpy(X: np.ndarray) -> np.ndarray:
    """
    Computes the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix

    Returns:
        F: (n, 1) function values at X
    """
    x_i = X[:, :-1]
    x_next = X[:, 1:]
    F = np.sum(100 * (x_next - x_i**2) ** 2 + (1 - x_i) ** 2, axis=1, keepdims=True)
    return F


def rosenbrock_grad_cupy(X: cp.ndarray) -> cp.ndarray:
    """
    Computes the gradient of the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix

    Returns:
        J: (n, 1, d) Jacobian at X
    """
    n_samples, n_dims = X.shape
    J = cp.zeros((n_samples, 1, n_dims), dtype=X.dtype)

    # The term (1 - x_i)^2 only exists for i = 0 to d-2
    J[:, 0, :-1] += 2 * (X[:, :-1] - 1)

    # Term from 100*(x_{i+1} - x_i^2)^2 for i < d-1 (derivative w.r.t x_i)
    # This affects 0 to d-2
    J[:, 0, :-1] -= 400 * X[:, :-1] * (X[:, 1:] - X[:, :-1] ** 2)

    # Term from 100*(x_i - x_{i-1}^2)^2 for i > 0 (derivative w.r.t x_{i+1})
    # This affects 1 to d-1
    J[:, 0, 1:] += 200 * (X[:, 1:] - X[:, :-1] ** 2)

    return J


def rosenbrock_grad_numpy(X: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix

    Returns:
        J: (n, 1, d) Jacobian at X
    """
    n_samples, n_dims = X.shape
    J = np.zeros((n_samples, 1, n_dims), dtype=X.dtype)

    # The term (1 - x_i)^2 only exists for i = 0 to d-2
    J[:, 0, :-1] += 2 * (X[:, :-1] - 1)

    # Term from 100*(x_{i+1} - x_i^2)^2 for i < d-1 (derivative w.r.t x_i)
    # This affects 0 to d-2
    J[:, 0, :-1] -= 400 * X[:, :-1] * (X[:, 1:] - X[:, :-1] ** 2)

    # Term from 100*(x_i - x_{i-1}^2)^2 for i > 0 (derivative w.r.t x_{i+1})
    # This affects 1 to d-1
    J[:, 0, 1:] += 200 * (X[:, 1:] - X[:, :-1] ** 2)

    return J
