"""Randomly RotatedAnisotropic Quadratic function and its derivatives."""

from src import Array, xp

SEED = 0
DIAGONAL_EXPONENT = 1.5


def _get_random_rotation(d: int, dtype) -> Array:
    """
    Creates a random rotation matrix using QR decomposition.
    """
    xp.random.seed(SEED)
    A = xp.random.randn(d, d).astype(dtype)
    Q, _ = xp.linalg.qr(A)
    if xp.linalg.det(Q) < 0:
        # Ensure a proper rotation (det(Q)=1)
        Q[:, 0] = -Q[:, 0]
    return Q


def random_aniso_quadratic(X: Array) -> Array:
    """
    Computes the Randomly Rotated Anisotropic Quadratic function.
    f(x) = x^T (Q S Q^T) x
    where Q is a random rotation matrix and S = diag([(1/i)^DIAGONAL_EXPONENT for i in 1..d])
    Args:
        X: (N, d) data matrix
    Returns:
        F: (N, 1) function values
    """
    N, d = X.shape
    dtype = X.dtype

    # Create scales: S = diag([(1/i)^DIAGONAL_EXPONENT for i in 1..d])
    scales = xp.array(
        [(1.0 / (i + 1)) ** DIAGONAL_EXPONENT for i in range(d)], dtype=dtype
    )  # (d,)

    # Create random rotation matrix Q
    Q = _get_random_rotation(d, dtype)  # (d, d)

    # Rotate data: X_rot = X Q
    X_rot = X @ Q  # (N, d)

    # Quadratic term: x^T (Q S Q^T) x = sum(s_i * x_rot_i^2)
    quad_term = xp.sum(scales[None, :] * (X_rot**2), axis=1, keepdims=True)

    return quad_term


def random_aniso_quadratic_grad(X: Array) -> Array:
    """
    Computes gradient of Randomly Rotated Anisotropic Quadratic function.
    g(x) = 2 * Q S Q^T x

    Returns:
        G: (N, 1, d) gradients
    """
    N, d = X.shape
    dtype = X.dtype

    # Create scales: S = diag([(1/i)^DIAGONAL_EXPONENT for i in 1..d])
    scales = xp.array(
        [(1.0 / (i + 1)) ** DIAGONAL_EXPONENT for i in range(d)], dtype=dtype
    )  # (d,)

    # Create random rotation matrix Q
    Q = _get_random_rotation(d, dtype)  # (d, d)

    # Rotate data: X_rot = X Q
    X_rot = X @ Q  # (N, d)

    # Gradient in rotated space: g_rot = 2 * S * x_rot
    g_rot = 2.0 * scales[None, :] * X_rot  # (N, d)

    # Rotate back: g = g_rot Q^T
    G = g_rot @ Q.T  # (N, d)

    return G[:, None, :]


def random_aniso_quadratic_hess(X: Array) -> Array:
    """
    Computes Hessian of Randomly Rotated Anisotropic Quadratic function.
    H = 2 * Q S Q^T

    Returns:
        H: (N, d, d) Hessians
    """
    N, d = X.shape
    dtype = X.dtype

    # Create scales: S = diag([(1/i)^DIAGONAL_EXPONENT for i in 1..d])
    scales = xp.array(
        [(1.0 / (i + 1)) ** DIAGONAL_EXPONENT for i in range(d)], dtype=dtype
    )  # (d,)

    # Create random rotation matrix Q
    Q = _get_random_rotation(d, dtype)  # (d, d)

    # Create diagonal matrix S
    S = xp.diag(scales)  # (d, d)

    # Compute Hessian: H = 2 * Q S Q^T
    H_base = 2.0 * (Q @ S @ Q.T)  # (d, d)

    # Repeat for all N points
    return xp.repeat(H_base[None, :, :], N, axis=0)
