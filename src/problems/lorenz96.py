"""Lorenz96 problem and its gradient."""

import numpy as np

from src import Array


def lorenz96(X: Array, forcing=8.0) -> Array:
    """
    Lorenz96 function implementation.

    Args:
        X: (N, d) data matrix where each row represents a state of the system.
        forcing: Forcing term in the Lorenz96 equations.

    Returns:
        dXdt: (n_samples, d) time derivatives at each state in X.
    """
    dXdt = (
        (np.roll(X, -1, axis=1) - np.roll(X, 2, axis=1)) * np.roll(X, 1, axis=1)
        - X
        + forcing
    )
    return dXdt


def lorenz96_grad(X) -> Array:
    """
    Gradient (Jacobian) of the Lorenz96 function.

    Args:
        X: (N, d) data matrix where each row represents a state of the system.

    Returns:
        J: (N, d, d) Jacobian matrices at each state in X.
    """
    N, d = X.shape
    J = np.zeros((N, d, d), dtype=X.dtype)

    X_im2 = np.roll(X, 2, axis=1)
    X_im1 = np.roll(X, 1, axis=1)
    X_ip1 = np.roll(X, -1, axis=1)

    # Create index arrays for the band diagonals
    i_idx = np.arange(d)
    im2_idx = (i_idx - 2) % d
    im1_idx = (i_idx - 1) % d
    ip1_idx = (i_idx + 1) % d

    # Populate Jacobian using vectorized indexing
    J[:, i_idx, im2_idx] = -X_im1
    J[:, i_idx, im1_idx] = X_ip1 - X_im2
    J[:, i_idx, i_idx] = -1
    J[:, i_idx, ip1_idx] = X_im1

    return J
