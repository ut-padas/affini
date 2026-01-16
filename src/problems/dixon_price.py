"""Dixon-Price Function and its gradient implementation."""

from src import Array, xp


def dixon_price(X: Array) -> Array:
    """
    Dixon-Price function implementation.

    Args:
        X: (N, d) data matrix where each row represents a set of input parameters.

    Returns:
        F: (N, 1) function values at X
    """
    N, d = X.shape
    term1 = (X[:, 0] - 1) ** 2
    j = xp.arange(1, d)
    term2 = xp.sum((j + 1) * (2 * X[:, j] ** 2 - X[:, j - 1]) ** 2, axis=1)
    F = term1 + term2
    return F[:, xp.newaxis]  # Ensure shape is (N, 1)


def dixon_price_grad(X: Array) -> Array:
    """
    Gradient of Dixon-Price function.

    Args:
        X: (N, d) data matrix where each row represents a set of input parameters.

    Returns:
        J: (N, 1, d) gradient at X
    """
    N, d = X.shape

    # Initialize gradient array
    J = xp.zeros((N, d), dtype=X.dtype)

    # First element (j=0)
    # g[0] = 2 * (x[0] - 1) - 4 * (2 * x[1]^2 - x[0])
    J[:, 0] = 2 * (X[:, 0] - 1) - 4 * (2 * X[:, 1] ** 2 - X[:, 0])

    # Last element (j=n-1)
    # g[-1] = 2 * n * (2 * x[-1]^2 - x[-2]) * (4 * x[-1])
    J[:, -1] = 2 * d * (2 * X[:, -1] ** 2 - X[:, -2]) * (4 * X[:, -1])

    # Middle elements (j=1 to n-2)
    # g[j] = 2 * (j+1) * (2 * x[j]^2 - x[j-1]) * (4 * x[j]) - 2 * (j+2) * (2 * x[j+1]^2 - x[j])
    for j in range(1, d - 1):
        term1 = 2 * (j + 1) * (2 * X[:, j] ** 2 - X[:, j - 1]) * (4 * X[:, j])
        term2 = 2 * (j + 2) * (2 * X[:, j + 1] ** 2 - X[:, j])
        J[:, j] = term1 - term2

    # Return shape (N, 1, d) to match expected gradient format
    return J[:, xp.newaxis, :]
