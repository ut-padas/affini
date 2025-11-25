"""Dixon-Price Function and its gradient implementation."""

from array_api_compat import array_namespace


def dixon_price(X):
    """
    Dixon-Price function implementation.

    Args:
        X: (n_samples, d) data matrix where each row represents a set of input parameters.

    Returns:
        F: (n_samples, 1) function values at X
    """
    xp = array_namespace(X)
    n_samples, n_dim = X.shape
    term1 = (X[:, 0] - 1) ** 2
    j = xp.arange(1, n_dim)
    term2 = xp.sum((j + 1) * (2 * X[:, j] ** 2 - X[:, j - 1]) ** 2, axis=1)
    F = term1 + term2
    return F[:, xp.newaxis]  # Ensure shape is (n_samples, 1)


def dixon_price_grad(X):
    """
    Gradient of Dixon-Price function.

    Args:
        X: (n_samples, d) data matrix where each row represents a set of input parameters.

    Returns:
        J: (n_samples, 1, d) gradient at X
    """
    xp = array_namespace(X)
    n_samples, n_dim = X.shape

    # Initialize gradient array
    J = xp.zeros((n_samples, n_dim), dtype=X.dtype)

    # First element (j=0)
    # g[0] = 2 * (x[0] - 1) - 4 * (2 * x[1]^2 - x[0])
    J[:, 0] = 2 * (X[:, 0] - 1) - 4 * (2 * X[:, 1] ** 2 - X[:, 0])

    # Last element (j=n-1)
    # g[-1] = 2 * n * (2 * x[-1]^2 - x[-2]) * (4 * x[-1])
    J[:, -1] = 2 * n_dim * (2 * X[:, -1] ** 2 - X[:, -2]) * (4 * X[:, -1])

    # Middle elements (j=1 to n-2)
    # g[j] = 2 * (j+1) * (2 * x[j]^2 - x[j-1]) * (4 * x[j]) - 2 * (j+2) * (2 * x[j+1]^2 - x[j])
    for j in range(1, n_dim - 1):
        term1 = 2 * (j + 1) * (2 * X[:, j] ** 2 - X[:, j - 1]) * (4 * X[:, j])
        term2 = 2 * (j + 2) * (2 * X[:, j + 1] ** 2 - X[:, j])
        J[:, j] = term1 - term2

    # Return shape (n_samples, 1, d) to match expected gradient format
    return J[:, xp.newaxis, :]
