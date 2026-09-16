"""Randomly Rotated Anisotropic Gaussian function and its derivatives."""

from src import Array, xp

SEED = 0
FREQUENCY = 2.0

def _get_random_direction(d: int, dtype) -> Array:
    """
    Creates a random rotation matrix using QR decomposition.
    """
    xp.random.seed(SEED)
    q = xp.random.randn(d).astype(dtype)
    q = q / xp.linalg.norm(q)
    return q


def wave(X: Array, frequency: float = FREQUENCY) -> Array:
    """
    Computes a cosine wave pointed in a random direction.
    f(x) = cos(frequency * q * x)
    where q is a random normal vector.
    frequency can be tuned; higher means more oscillatory.

    Args:
        X: (N, d) data matrix
        frequency: the frequency of the cosine.
    Returns:
        F: (N, 1) function values
    """
    N, d = X.shape
    dtype = X.dtype
    q = _get_random_direction(d, dtype)
    return xp.cos(frequency * (X @ q))

