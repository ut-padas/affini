"""Degenerate Quadratic function and its derivatives."""

from src import Array, xp

SEED = 0
BETA = 1.0


def _get_params(d: int, dtype) -> Array:
    """
    Computes the symmetric matrix H = diag(lambda), where the first
    floor(d/2) eigenvalues are 1 and the remaining eigenvalues are 0.

    Returns:
        H: (d, d) quadratic form matrix
    """
    # 1. Eigenvalues: 1 for the first half (rounding down), 0 for the rest
    num_ones = d // 2
    lambdas = xp.zeros((d,), dtype=dtype)
    lambdas[:num_ones] = 1.0

    # 2. Construct H = diag(lambdas)
    H = xp.diag(lambdas)

    return H


def degenerate_quadratic(X: Array, beta: float = BETA) -> Array:
    """
    Computes the Degenerate Quadratic function.
    f(x) = beta + x^T H x

    Args:
        X: (N, d) data matrix
        beta: Constant offset term (default: DEFAULT_BETA)
    Returns:
        F: (N, 1) function values
    """
    N, d = X.shape
    H = _get_params(d, X.dtype)

    # Quadratic term: x^T H x
    quad_term = xp.sum((X @ H) * X, axis=1)

    # Combine: beta + quadratic
    res = beta + quad_term

    return res[:, None]  # (N, 1)


def degenerate_quadratic_grad(X: Array) -> Array:
    """
    Computes gradient.
    grad f(x) = 2Hx

    Args:
        X: (N, d) data matrix
    Returns:
        G: (N, 1, d) gradients
    """
    N, d = X.shape
    H = _get_params(d, X.dtype)

    # term: 2 * H * x
    # (N, d) @ (d, d) -> (N, d)
    # Note: H is symmetric, so (H @ x) or (x @ H) results in same vector,
    # but for (N, d) shape we do X @ H.
    grad_quad = 2.0 * (X @ H)

    return grad_quad[:, None, :]  # (N, 1, d)


def degenerate_quadratic_hess(X: Array) -> Array:
    """
    Computes Hessian.
    Hess f(x) = 2H

    Args:
        X: (N, d) data matrix
    Returns:
        H_vals: (N, d, d) Hessians
    """
    N, d = X.shape
    H = _get_params(d, X.dtype)

    # The Hessian is constant: 2 * H
    H_const = 2.0 * H

    # Repeat for all N points
    return xp.broadcast_to(H_const, (N, *H_const.shape))