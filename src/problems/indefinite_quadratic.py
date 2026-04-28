"""Indefinite Quadratic function and its derivatives."""

from src import Array, xp
from src.problems.anisotropic_quadratic import _get_random_rotation

# Default hyperparameters matching the anisotropic case
SEED = 0
DECAY_EXPONENT = 3.0
BETA = 1.0


def _get_indefinite_params(
    d: int, dtype, decay_exponent: float = DECAY_EXPONENT
) -> tuple[Array, Array]:
    """
    Computes the linear weights and the indefinite quadratic matrix.
    H is constructed as Q * Lambda * Q^T where Lambda contains alternating signs.

    Returns:
        w: (d,) linear weight vector [1, 1, ..., 1]
        H: (d, d) indefinite symmetric matrix
    """
    # 1. Linear weights: w = [1, 1, ..., 1]
    w = xp.ones((d,), dtype=dtype)

    # 2. Eigenvalues: magnitude decays polynomially
    indices = xp.arange(1, d + 1, dtype=dtype)
    lambdas = xp.power(indices, -decay_exponent)

    # 3. Indefinite Modification: Flip the sign of alternating eigenvalues [+, -, +, -, ...]
    # This ensures the matrix is neither positive nor negative definite.
    signs = xp.ones(d, dtype=dtype)
    signs[1::2] = -1.0  # Set every second element to -1
    lambdas = lambdas * signs

    # 4. Rotation: Reuse the random orthogonal rotation logic
    Q = _get_random_rotation(d, dtype)

    # 5. Construct H = Q @ diag(lambdas) @ Q.T
    H = (Q * lambdas[None, :]) @ Q.T

    return w, H


def indefinite_quadratic(
    X: Array, beta: float = BETA, decay_exponent: float = DECAY_EXPONENT
) -> Array:
    """
    Computes the Indefinite Quadratic function values.
    f(x) = beta + w^T x + x^T H x

    Args:
        X: (N, d) data matrix
        beta: Constant offset term
        decay_exponent: Eigenvalue decay exponent
    Returns:
        F: (N, 1) function values
    """
    N, d = X.shape
    w, H = _get_indefinite_params(d, X.dtype, decay_exponent)

    # Linear term: w^T x -> (N, d) @ (d,) -> (N,)
    linear_term = X @ w

    # Quadratic term: x^T H x
    quad_term = xp.sum((X @ H) * X, axis=1)

    # Combine terms
    res = beta + linear_term + quad_term

    return res[:, None]  # (N, 1)


def indefinite_quadratic_grad(
    X: Array, decay_exponent: float = DECAY_EXPONENT
) -> Array:
    """
    Computes the gradient of the Indefinite Quadratic function.
    grad f(x) = w + 2Hx

    Args:
        X: (N, d) data matrix
        decay_exponent: Eigenvalue decay exponent
    Returns:
        G: (N, 1, d) gradients
    """
    N, d = X.shape
    w, H = _get_indefinite_params(d, X.dtype, decay_exponent)

    # Quadratic gradient: 2 * H * x -> (N, d)
    grad_quad = 2.0 * (X @ H)

    # Add linear weights: w
    G = w[None, :] + grad_quad

    return G[:, None, :]  # (N, 1, d)


def indefinite_quadratic_hess(
    X: Array, decay_exponent: float = DECAY_EXPONENT
) -> Array:
    """
    Computes the constant Hessian of the Indefinite Quadratic function.
    Hess f(x) = 2H

    Args:
        X: (N, d) data matrix
        decay_exponent: Eigenvalue decay exponent
    Returns:
        H_vals: (N, d, d) Hessians
    """
    N, d = X.shape
    _, H = _get_indefinite_params(d, X.dtype, decay_exponent)

    # The Hessian is constant across the domain: 2 * H
    H_const = 2.0 * H

    # Repeat the constant matrix for all N query points
    return xp.repeat(H_const[None, :, :], N, axis=0)
