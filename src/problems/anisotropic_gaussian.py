"""Randomly Rotated Anisotropic Gaussian function and its derivatives."""

from src import Array, xp

SEED = 0
SIGMA_DECAY_EXPONENT = 3
BETA = 1


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


def _get_precision_eigenvalues(d: int, dtype) -> Array:
    """
    Constructs the eigenvalues of the Precision Matrix (Sigma^-1).

    Logic:
    1. Define Sigma (Standard Deviations) such that sigma_max = 1.
       sigma_i = 1 / (i + 1)^p
    2. Precision = 1 / Variance = 1 / sigma^2
       precision_i = (i + 1)^(2p)
    """
    # 1. Define Sigmas: Start at 1.0, decay by power law
    # sigma_0 = 1, sigma_1 = 1/2^p, ...
    indices = xp.arange(d, dtype=dtype) + 1
    sigmas = 1.0 / (indices**SIGMA_DECAY_EXPONENT)

    # 2. Compute Precision (inverse variance)
    # This is the diagonal matrix \Lambda^{-1}
    precision_diag = 1.0 / (sigmas)

    return precision_diag


def aniso_gaussian(X: Array) -> Array:
    """
    Computes the Randomly Rotated Anisotropic Gaussian function.
    f(x) = BETA + exp(-0.5 * x^T Sigma^-1 x)
    where Sigma^-1 = Q Lambda^-1 Q^T

    Args:
        X: (N, d) data matrix
    Returns:
        F: (N, 1) function values
    """
    N, d = X.shape
    dtype = X.dtype

    # Get Precision eigenvalues (Lambda^-1)
    prec_diag = _get_precision_eigenvalues(d, dtype)

    # Create random rotation matrix Q
    Q = _get_random_rotation(d, dtype)

    # Rotate data: X_rot = X Q
    X_rot = X @ Q  # (N, d)

    # Compute Mahalanobis distance squared: x^T (Q Lambda^-1 Q^T) x
    # = sum( (X_rot_i^2) / sigma_i^2 )
    mahalanobis_sq = xp.sum(prec_diag[None, :] * (X_rot**2), axis=1, keepdims=True)

    # f(x) = BETA + exp(-0.5 * dist^2)
    return BETA + xp.exp(-0.5 * mahalanobis_sq)


def aniso_gaussian_grad(X: Array) -> Array:
    """
    Computes gradient of the Gaussian function.
    g(x) = -exp(...) * (Sigma^-1 x)

    Returns:
        G: (N, 1, d) gradients
    """
    N, d = X.shape
    dtype = X.dtype

    # --- 1. Compute PDF components ---
    prec_diag = _get_precision_eigenvalues(d, dtype)
    Q = _get_random_rotation(d, dtype)

    X_rot = X @ Q
    mahalanobis_sq = xp.sum(prec_diag[None, :] * (X_rot**2), axis=1, keepdims=True)
    exp_part = xp.exp(-0.5 * mahalanobis_sq)  # (N, 1)

    # --- 2. Compute Sigma^-1 x ---
    # Apply diagonal precision in rotated space
    temp = prec_diag[None, :] * X_rot  # (X Q) Lambda^-1

    # Rotate back
    inv_sigma_x = temp @ Q.T  # (N, d)

    # --- 3. Compute Gradient ---
    G = -exp_part * inv_sigma_x

    return G[:, None, :]


def aniso_gaussian_hess(X: Array) -> Array:
    """
    Computes Hessian of the Gaussian function.
    H(x) = exp(...) * [ (Sigma^-1 x)(Sigma^-1 x)^T - Sigma^-1 ]

    Returns:
        H: (N, d, d) Hessians
    """
    N, d = X.shape
    dtype = X.dtype

    # --- 1. Compute PDF components ---
    prec_diag = _get_precision_eigenvalues(d, dtype)
    Q = _get_random_rotation(d, dtype)

    X_rot = X @ Q
    mahalanobis_sq = xp.sum(prec_diag[None, :] * (X_rot**2), axis=1, keepdims=True)
    exp_part = xp.exp(-0.5 * mahalanobis_sq)  # (N, 1)

    # --- 2. Construct Full Precision Matrix Sigma^-1 ---
    Prec_Mat = xp.diag(prec_diag)
    M = Q @ Prec_Mat @ Q.T  # (d, d)

    # --- 3. Compute Vector term v = Sigma^-1 x ---
    temp = prec_diag[None, :] * X_rot
    v = temp @ Q.T  # (N, d)

    # --- 4. Compute Hessian Terms ---
    term1 = v[:, :, None] * v[:, None, :]
    term2 = M[None, :, :]
    H = exp_part[:, :, None] * (term1 - term2)
    return H
