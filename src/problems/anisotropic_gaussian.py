"""Randomly Rotated Anisotropic Gaussian function and its derivatives."""

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
        Q[:, 0] = -Q[:, 0]
    return Q


def random_aniso_gaussian(X: Array) -> Array:
    """
    Computes the Randomly Rotated Anisotropic Gaussian function.
    f(x) = C * exp(-0.5 * x^T (Q S Q^T) x)

    Args:
        X: (N, d) data matrix
    Returns:
        F: (N, 1) function values
    """
    N, d = X.shape
    dtype = X.dtype

    # Create scales (Eigenvalues of the matrix in the exponent)
    scales = xp.array(
        [(1.0 / (i + 1)) ** DIAGONAL_EXPONENT for i in range(d)], dtype=dtype
    )

    # Create random rotation matrix Q
    Q = _get_random_rotation(d, dtype)

    # Rotate data: X_rot = X Q
    X_rot = X @ Q  # (N, d)

    # Compute Quadratic part: x^T (Q S Q^T) x
    # This is equivalent to sum(s_i * x_rot_i^2)
    quad_term = xp.sum(scales[None, :] * (X_rot**2), axis=1, keepdims=True)

    # Compute Normalization Constant C
    # log(C) = 0.5 * (sum(log(scales)) - d * log(2pi))
    log_det = xp.sum(xp.log(scales))
    log_C = 0.5 * (log_det - d * xp.log(2 * xp.pi))

    # f(x) = exp(log_C - 0.5 * quad_term)
    return xp.exp(log_C - 0.5 * quad_term)


def random_aniso_gaussian_grad(X: Array) -> Array:
    """
    Computes gradient of the Gaussian function.
    g(x) = -f(x) * (Q S Q^T x)

    Returns:
        G: (N, 1, d) gradients
    """
    N, d = X.shape
    dtype = X.dtype

    # --- 1. Compute f(x) ---
    F = random_aniso_gaussian(X)  # (N, 1)

    # --- 2. Compute Matrix-Vector product (Q S Q^T x) ---
    scales = xp.array(
        [(1.0 / (i + 1)) ** DIAGONAL_EXPONENT for i in range(d)], dtype=dtype
    )
    Q = _get_random_rotation(d, dtype)

    # M x = Q (S (Q^T x))
    X_rot = X @ Q  # X Q
    temp = scales[None, :] * X_rot  # X Q S
    Mx = temp @ Q.T  # X Q S Q^T  => (N, d)

    # --- 3. Compute Gradient ---
    # g = -f(x) * (M x)
    G = -F * Mx

    return G[:, None, :]


def random_aniso_gaussian_hess(X: Array) -> Array:
    """
    Computes Hessian of the Gaussian function.
    H(x) = f(x) * [ (Mx)(Mx)^T - M ]
    where M = Q S Q^T

    Returns:
        H: (N, d, d) Hessians
    """
    N, d = X.shape
    dtype = X.dtype

    # --- 1. Compute f(x) ---
    F = random_aniso_gaussian(X)  # (N, 1)

    # --- 2. Construct Matrix M = Q S Q^T ---
    scales = xp.array(
        [(1.0 / (i + 1)) ** DIAGONAL_EXPONENT for i in range(d)], dtype=dtype
    )
    Q = _get_random_rotation(d, dtype)
    S_mat = xp.diag(scales)

    M = Q @ S_mat @ Q.T  # (d, d)

    # --- 3. Compute Vector term v = M x ---
    # We need v for every point X_i
    X_rot = X @ Q
    temp = scales[None, :] * X_rot
    Mx = temp @ Q.T  # (N, d)

    # --- 4. Compute Hessian Terms ---
    # Term 1: Outer product (Mx)(Mx)^T for each N
    # Shape becomes (N, d, d)
    term1 = Mx[:, :, None] * Mx[:, None, :]

    # Term 2: M broadcasted to (N, d, d)
    term2 = M[None, :, :]

    # Combine: f(x) * (term1 - term2)
    # F is (N, 1), reshape to (N, 1, 1) for broadcasting
    H = F[:, :, None] * (term1 - term2)

    return H
