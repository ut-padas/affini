"""Indefinite Degenerate Quadratic function and its derivatives."""

import numpy as np
from array_api_compat import array_namespace

from src import Array

SEED = 0
BETA = 1.0


def _get_params(d: int, dtype) -> np.ndarray:
    """
    Computes the symmetric matrix H = diag(lambda), where the first
    quarter of eigenvalues are 1, the next quarter are -1, and the
    remaining half are 0.

    Returns:
        H: (d, d) quadratic form matrix, as a NumPy (host) array
    """
    # 1. Eigenvalues: 1 for first quarter, -1 for next quarter, 0 for rest
    num_pos = d // 4
    num_neg = d // 4
    lambdas = np.zeros((d,), dtype=dtype)
    lambdas[:num_pos] = 1.0
    lambdas[num_pos:num_pos + num_neg] = -1.0

    # 2. Construct H = diag(lambdas)
    H = np.diag(lambdas)

    return H


def indef_degen_quad(X: Array, beta: float = BETA) -> Array:
    """
    Computes the Indefinite Degenerate Quadratic function.
    f(x) = beta + x^T H x

    Args:
        X: (N, d) data matrix
        beta: Constant offset term (default: DEFAULT_BETA)
    Returns:
        F: (N, 1) function values
    """
    xp = array_namespace(X)
    N, d = X.shape
    H = xp.asarray(_get_params(d, X.dtype))

    # Quadratic term: x^T H x
    quad_term = xp.sum((X @ H) * X, axis=1)

    # Combine: beta + quadratic
    res = beta + quad_term

    return res[:, None]  # (N, 1)


def indef_degen_quad_grad(X: Array) -> Array:
    """
    Computes gradient.
    grad f(x) = 2Hx

    Args:
        X: (N, d) data matrix
    Returns:
        G: (N, 1, d) gradients
    """
    xp = array_namespace(X)
    N, d = X.shape
    H = xp.asarray(_get_params(d, X.dtype))

    # term: 2 * H * x
    # (N, d) @ (d, d) -> (N, d)
    # Note: H is symmetric, so (H @ x) or (x @ H) results in same vector,
    # but for (N, d) shape we do X @ H.
    grad_quad = 2.0 * (X @ H)

    return grad_quad[:, None, :]  # (N, 1, d)


def indef_degen_quad_hess(X: Array) -> Array:
    """
    Computes Hessian.
    Hess f(x) = 2H

    Args:
        X: (N, d) data matrix
    Returns:
        H_vals: (N, d, d) Hessians
    """
    xp = array_namespace(X)
    N, d = X.shape
    H = xp.asarray(_get_params(d, X.dtype))

    # The Hessian is constant: 2 * H
    H_const = 2.0 * H

    # Repeat for all N points
    return xp.broadcast_to(H_const, (N, *H_const.shape))
