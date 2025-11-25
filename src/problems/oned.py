"""One-dimensional example function and its gradient."""

import array_api_compat


def one_d_example(X):
    """
    One-dimensional example function: f(x) = sin(2πx) / exp(x)

    Args:
        X: (n_samples, 1) data matrix where each row represents a single input value.

    Returns:
        F: (n_samples, 1) function values at X
    """
    xp = array_api_compat.array_namespace(X)
    x = X[:, 0]
    F = xp.sin(2 * xp.pi * x) / xp.exp(x)
    return F[:, xp.newaxis]  # Ensure shape is (n_samples, 1)


def one_d_example_grad(X):
    """
    Gradient of the one-dimensional example function: f'(x) = (2πcos(2πx) - sin(2πx)) / exp(x)

    Args:
        X: (n_samples, 1) data matrix where each row represents a single input value.

    Returns:
        J: (n_samples, 1, 1) Jacobian at X
    """
    xp = array_api_compat.array_namespace(X)
    x = X[:, 0]
    grad = (2 * xp.pi * xp.cos(2 * xp.pi * x) - xp.sin(2 * xp.pi * x)) / xp.exp(x)
    return grad[:, xp.newaxis, xp.newaxis]  # Ensure shape is (n_samples, 1, 1)


def one_d_example_hess(X):
    """
    Hessian of the one-dimensional example function: f''(x) = (-4π^2 sin(2πx) - 4π cos(2πx) + sin(2πx)) / exp(x)

    Args:
        X: (n_samples, 1) data matrix where each row represents a single input value.

    Returns:
        H: (n_samples, 1, 1, 1) Hessian at X
    """
    xp = array_api_compat.array_namespace(X)
    x = X[:, 0]
    hess = (
        -4 * (xp.pi**2) * xp.sin(2 * xp.pi * x)
        - 4 * xp.pi * xp.cos(2 * xp.pi * x)
        + xp.sin(2 * xp.pi * x)
    ) / xp.exp(x)
    return hess[
        :, xp.newaxis, xp.newaxis, xp.newaxis
    ]  # Ensure shape is (n_samples, 1, 1, 1)
