"""Randomly Rotated Anisotropic Gaussian function and its derivatives."""

from src import Array, xp

SEED = 0
FREQUENCY = 2.0
BETA = 1.5

def _get_random_direction(d: int, dtype) -> Array:
    xp.random.seed(SEED)
    q = xp.random.randn(d).astype(dtype)
    q = q / xp.linalg.norm(q)
    return q


def wave(X: Array, frequency: float = FREQUENCY, beta: float = BETA) -> Array:
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
    f = xp.cos(frequency * (X @ q)) + beta
    return xp.reshape(f, (N, 1))


def wave_grad(
    X: Array,
    frequency: float = FREQUENCY,
) -> Array:
    """
    Computes the gradient of

        f(x) = cos(frequency * q^T x) + beta

    at each point in X.

    Args:
        X: (N, d) data matrix
        q: (d,) wave direction
        frequency: frequency of the cosine

    Returns:
        G: (N, d) gradient at each point
    """
    N, d = X.shape
    q = _get_random_direction(d, X.dtype)
    qTx = X @ q  # (N,)

    G = -frequency * xp.sin(frequency * qTx)[:, None] * q

    return G



def wave_hess(
    X: Array,
    frequency: float = FREQUENCY,
    beta: float = BETA,
) -> Array:
    """
    Computes the Hessian of the wave function

        f(x) = cos(frequency * q^T x) + beta

    at each point in X.

    Args:
        X: (N, d) data matrix
        frequency: frequency of the cosine
        beta: constant offset (does not affect the Hessian)

    Returns:
        H: (N, d, d) tensor, where H[i] is the Hessian
           at X[i].
    """
    N, d = X.shape
    dtype = X.dtype

    q = _get_random_direction(d, dtype)

    # q^T x for every point
    qTx = X @ q  # (N,)

    # q q^T
    qqT = xp.outer(q, q)  # (d, d)

    # Hessian at every point
    H = -frequency**2 * xp.cos(frequency * qTx)[:, None, None] * qqT

    return H