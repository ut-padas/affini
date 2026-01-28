"""Vector Quadratic function and its derivatives."""

from src import Array, xp


def _get_diagonals(d: int, dtype) -> tuple[Array, Array]:
    """
    Helper to construct the diagonal curvature vectors.
    Math index j=1 (Odd) -> Code index 0.
    Math index j=2 (Even) -> Code index 1.

    A1: 100 on Odd (0, 2...), 1 on Even (1, 3...)
    A2: 1 on Odd (0, 2...), 100 on Even (1, 3...)
    """
    diag1 = xp.ones((d,), dtype=dtype)
    diag1[0::2] = 100.0

    diag2 = xp.ones((d,), dtype=dtype)
    diag2[1::2] = 100.0  # Initialize as 1s, set Evens to 100

    return diag1, diag2


def vector_quadratic(X: Array) -> Array:
    """
    Computes the Vector Quadratic function F: R^d -> R^2.
    f1(x) = (x - 1)^T A1 (x - 1)
    f2(x) = (x + 1)^T A2 (x + 1)

    Args:
        X: (N, d) data matrix
    Returns:
        F: (N, 2) function values
    """
    N, d = X.shape
    diag1, diag2 = _get_diagonals(d, X.dtype)

    # f1: centered at +1
    diff1 = X - 1.0
    val1 = xp.sum((diff1**2) * diag1, axis=1)

    # f2: centered at -1
    diff2 = X + 1.0
    val2 = xp.sum((diff2**2) * diag2, axis=1)

    return xp.stack([val1, val2], axis=1)


def vector_quadratic_grad(X: Array) -> Array:
    """
    Computes the Jacobian of the Vector Quadratic function.
    grad f1(x) = 2 A1 (x - 1)
    grad f2(x) = 2 A2 (x + 1)

    Args:
        X: (N, d) data matrix
    Returns:
        G: (N, 2, d) gradients
    """
    N, d = X.shape
    diag1, diag2 = _get_diagonals(d, X.dtype)

    # Gradient 1: (N, d)
    g1 = 2.0 * diag1 * (X - 1.0)

    # Gradient 2: (N, d)
    g2 = 2.0 * diag2 * (X + 1.0)

    # Stack to (N, 2, d)
    return xp.stack([g1, g2], axis=1)


def vector_quadratic_hess(X: Array) -> Array:
    """
    Computes the Hessian Tensor of the Vector Quadratic function.
    H1 = 2 A1
    H2 = 2 A2

    Args:
        X: (N, d) data matrix
    Returns:
        H: (N, 2, d, d) Hessian matrices
    """
    N, d = X.shape
    diag1, diag2 = _get_diagonals(d, X.dtype)

    # Construct constant Hessian matrices (d, d)
    H1 = xp.diag(2.0 * diag1)
    H2 = xp.diag(2.0 * diag2)

    # Stack to (2, d, d)
    H_base = xp.stack([H1, H2], axis=0)

    # Broadcast to (N, 2, d, d)
    return xp.repeat(H_base[None, ...], N, axis=0)
