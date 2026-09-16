"""Isotropic Gaussian function."""

from src import Array, xp

BETA = 1

def gaussian(
    X: Array, beta: float = BETA
) -> Array:
    """
    Computes the Gaussian function.
    f(x) = beta + exp(-0.5 * |x|^2 / d)
    The variance is sqrt(d) so that the bump is less localized and it isn't just 1.0 everywhere.

    Args:
        X: (N, d) data matrix
        beta: Constant offset term (default: BETA)
    Returns:
        F: (N, 1) function values
    """
    N, d = X.shape
    dist_sq = xp.sum(X ** 2, axis=1)
    f = beta + xp.exp(-0.5 * dist_sq / d)
    return xp.reshape(f, (N,1))
