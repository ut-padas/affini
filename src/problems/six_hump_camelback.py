"""Six-Hump camelback function and its derivatives."""

import array_api_compat


def six_hump_camelback(X):
    """
    Computes the Six-Hump Camelback function for given input points.

    Args:
        X: (n, 2) data matrix

    Returns:
        F: (n, 1) function values at X
    """
    xp = array_api_compat.array_namespace(X)
    scale = 10
    x1, x2 = X[:, 0], X[:, 1]

    F = (
        (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
        + x1 * x2
        + (-4 + 4 * x2**2) * x2**2
        + scale
    )
    return F[:, xp.newaxis]  # Ensure shape is (n, 1)


def six_hump_camelback_grad(X):
    """
    Computes the gradient of the Six-Hump Camelback function for given input points.

    Args:
        X: (n, 2) data matrix

    Returns:
        J: (n, 1, 2) Jacobian at X
    """
    xp = array_api_compat.array_namespace(X)
    x1, x2 = X[:, 0], X[:, 1]

    # Compute gradient components
    Dxf = 8 * x1 - 8.4 * x1**3 + 2 * x1**5 + x2
    Dyf = x1 - 8 * x2 + 16 * x2**3

    # Stack gradients: shape (n, 2)
    grads = xp.stack([Dxf, Dyf], axis=1)

    # Add new axis to get shape (n, 1, 2) to follow convention: (n, fd, d) where fd=1, d=2
    return grads[:, xp.newaxis, :]


def six_hump_camelback_hess(X):
    """
    Computes the Hessian of the Six-Hump Camelback function for given input points.

    Args:
        X: (n, 2) data matrix

    Returns:
        H: (n, 2, 2) Hessian matrices at X
    """
    xp = array_api_compat.array_namespace(X)
    x1, x2 = X[:, 0], X[:, 1]

    # Compute hessian components for each point
    h11 = 8 - 25.2 * x1**2 + 10 * x1**4
    h12 = xp.ones_like(x1)  # Off-diagonal element
    h22 = -8 + 48 * x2**2

    # Stack into hessian matrices: shape (n, 2, 2)
    hessians = xp.stack(
        [xp.stack([h11, h12], axis=1), xp.stack([h12, h22], axis=1)], axis=1
    )

    return hessians
