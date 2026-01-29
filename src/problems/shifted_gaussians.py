"""Shifted Gaussian function and its derivatives."""

from src import Array, xp

CENTERS_2D = xp.array([[1.0, 1.0], [-1.0, -1.0]])
CENTERS_3D = xp.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, 1.0]])
CENTERS_4D = xp.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
    ]
)
CENTERS_5D = xp.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0, 1.0],
    ]
)


def shifted_gaussians(
    X: Array, centers: Array, shift: float = 1.0, std_dev: float = 0.5
) -> Array:
    """
    Computes the shifted Gaussian function for D input points.

    Args:
        X: (N, D) data matrix
        centers: (M, D) centers of the Gaussians
        shift: shift parameter
        std_dev: standard deviation of the Gaussians

    Returns:
        F: (N, 1) function values at X
    """
    # 1. Expand dimensions to use broadcasting
    # X: (N, D) -> (N, 1, D)
    # centers: (M, D) -> (1, M, D)
    diff = X[:, xp.newaxis, :] - centers[xp.newaxis, :, :]

    # 2. Calculate squared Euclidean distance for all pairs
    # sq_dist shape: (N, M)
    sq_dist = xp.sum(diff**2, axis=2)

    # 3. Apply Gaussian, sum over centers (axis 1), and add shift
    f_x = xp.sum(xp.exp(-sq_dist / (2 * std_dev**2)), axis=1) + shift
    return f_x[:, xp.newaxis]  # Return shape (N, 1)


def shifted_gaussians_grad(X: Array, centers: Array, std_dev: float = 0.5) -> Array:
    """
    Computes the gradient of the shifted Gaussian function.

    Args:
        X: (N, D) data matrix
        centers: (M, D) centers of the Gaussians
        std_dev: standard deviation of the Gaussians

    Returns:
        grad_f_x: (N, 1, D) gradient at each point in X
    """
    # X: (N, D), centers: (M, D)
    # diff shape: (N, M, D)
    diff = X[:, xp.newaxis, :] - centers[xp.newaxis, :, :]

    # sq_dist shape: (N, M)
    sq_dist = xp.sum(diff**2, axis=2)

    # gaussians shape: (N, M)
    gaussians = xp.exp(-sq_dist / (2 * std_dev**2))

    # 1. Compute coefficients: shape (N, M)
    # 2. Reshape coefficients to (N, M, 1) to multiply with diff (N, M, D)
    # 3. Sum across the M centers (axis 1)
    coeff = -gaussians / (std_dev**2)
    grad_f_x = xp.sum(coeff[:, :, xp.newaxis] * diff, axis=1)

    return grad_f_x[:, xp.newaxis, :]  # Return shape (N, 1, D)
