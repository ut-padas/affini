# A second, simpler quadratic for testing purposes

from src import Array, xp

H1 = xp.diag(xp.array([10, 1]))
H2 = xp.diag(xp.array([1, 10]))

def vector_quadratic_2(X: Array) -> Array:
    """
    Computes f(x) = (<x, H1*x>, <x, H2*x>), which maps R2 to R2.
    For now assume H1 and H2 are diagonal matrices.
    Takes in X with shape (N, d) and returns matrix (N, 2).
    """
    N, d = X.shape
    val1 = xp.sum(X * (H1 @ X.T).T, axis=1)
    val2 = xp.sum(X * (H2 @ X.T).T, axis=1)
    return xp.stack([val1, val2], axis=1)

def vector_quadratic_2_grad(X: Array) -> Array:
    """
    Computes the gradient: f'(x) = (2*H1*x, 2*H2*x)
    Takes in X with shape (N, d) and returns tensor (N, 2, d).
    """
    g1 = 2.0 * (X @ H1) * X
    g2 = 2.0 * (X @ H2) * X
    return xp.stack([g1, g2], axis=1)

def vector_quadratic_2_hess(X: Array) -> Array:
    """
    Computes the Hessian: f''(x) = (2*H1, 2*H2)
    Takes in X with shape (N, d) and returns tensor (N, 2, d, d).
    """
    N, d = X.shape
    H1_full = 2.0 * H1 # Shape (d, d)
    H2_full = 2.0 * H2 # Shape (d, d), d is the input dimension
    H1_tiled = xp.repeat(H1_full[None, None, :, :], N, axis=0) # Shape (N, 1, d, d)
    H2_tiled = xp.repeat(H2_full[None, None, :, :], N, axis=0) # Shape (N, 1, d, d)
    return xp.concatenate([H1_tiled, H2_tiled], axis=1) # Shape (N, 2, d, d)

