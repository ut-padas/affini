"""Six-Hump Camelback Function and its Derivatives."""

import warnings

import numpy as np

try:
    import cupy as cp

    _HAS_GPU = cp.is_available()
except Exception:
    _HAS_GPU = False
    warnings.warn("CuPy or cuML not available; GPU acceleration will be disabled.")


# Six-Hump Camelback function
def SHCF_numpy(x: np.ndarray):
    """
    Vectorized Six-Hump Camelback function.

    Shape Convention:
        x.shape: (n, 2) where n=number of points, input is 2D
        Returns: (n, 1) function values
    """
    scale = 10
    x1, x2 = x[:, 0], x[:, 1]

    result = (
        (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
        + x1 * x2
        + (-4 + 4 * x2**2) * x2**2
        + scale
    )

    return result[:, np.newaxis]  # Ensure shape is (n, 1)


def grad_SHCF_numpy(x: np.ndarray):
    """
    Vectorized gradient of Six-Hump Camelback function.

    Shape Convention:
        x.shape: (n, 2) where n=number of points, input is 2D
        Returns: (n, 1, 2) gradients where each gradient is shape (1, 2)
    """
    x1, x2 = x[:, 0], x[:, 1]

    # Compute gradient components
    Dxf = 8 * x1 - 8.4 * x1**3 + 2 * x1**5 + x2
    Dyf = x1 - 8 * x2 + 16 * x2**3

    # Stack gradients: shape (n, 2)
    grads = np.stack([Dxf, Dyf], axis=1)

    # Add new axis to get shape (n, 1, 2) to follow convention: (n, fd, d) where fd=1, d=2
    return grads[:, np.newaxis, :]


def hessian_SHCF_numpy(x: np.ndarray):
    """
    Vectorized hessian of Six-Hump Camelback function.

    Shape Convention:
        x.shape: (n, 2) where n=number of points, input is 2D
        Returns: (n, 2, 2) Hessian matrices
    """
    x1, x2 = x[:, 0], x[:, 1]

    # Compute hessian components for each point
    h11 = 8 - 25.2 * x1**2 + 10 * x1**4
    h12 = np.ones_like(x1)  # Off-diagonal element
    h22 = -8 + 48 * x2**2

    # Stack into hessian matrices: shape (n, 2, 2)
    hessians = np.stack(
        [np.stack([h11, h12], axis=1), np.stack([h12, h22], axis=1)], axis=1
    )

    return hessians


def SHCF_cupy(x: cp.ndarray):
    """
    Vectorized Six-Hump Camelback function using CuPy.

    Shape Convention:
        x.shape: (n, 2) where n=number of points, input is 2D
        Returns: (n, 1) function values
    """
    scale = 10
    x1, x2 = x[:, 0], x[:, 1]

    result = (
        (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
        + x1 * x2
        + (-4 + 4 * x2**2) * x2**2
        + scale
    )

    return result[:, cp.newaxis]  # Ensure shape is (n, 1)
