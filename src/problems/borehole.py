"""Borehole function and its gradient implementation."""

from array_api_compat import array_namespace


def borehole(X):
    """
    Borehole function to simulate water flow through a borehole.

    Args:
        X: (n_samples, 8) data matrix where each row represents a set of input parameters.

    Returns:
        F: (n_samples, 1) function values at X
    """
    xp = array_namespace(X)

    r_w = X[:, 0]  # radius of the borehole (m)
    r = X[:, 1]  # radius of influence (m)
    T_u = X[:, 2]  # transmissivity of upper aquifer (m^2/yr)
    H_u = X[:, 3]  # potentiometric head of upper aquifer (m)
    T_l = X[:, 4]  # transmissivity of lower aquifer (m^2/yr)
    H_l = X[:, 5]  # potentiometric head of lower aquifer (m)
    L = X[:, 6]  # length of the borehole (m)
    K_w = X[:, 7]  # hydraulic conductivity of the borehole (m/yr)

    term1 = -1 * H_l + H_u
    term2 = T_u
    log_term = xp.log(r / r_w)
    denom = (T_u / T_l) + log_term * (1.5 + 2 * L / (K_w * r_w**2) * T_u / log_term)
    Q = 5 * term1 * term2 / denom
    return Q[:, xp.newaxis]  # Ensure shape is (n_samples, 1)


def borehole_grad(X):
    """
    Gradient of the Borehole function.

    Args:
        X: (n_samples, 8) data matrix where each row represents a set of input parameters.

    Returns:
        J: (n_samples, 1, 8) Jacobian at X
    """
    xp = array_namespace(X)

    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    x4 = X[:, 3]
    x5 = X[:, 4]
    x6 = X[:, 5]
    x7 = X[:, 6]
    x8 = X[:, 7]

    n_samples = X.shape[0]
    g = xp.zeros((n_samples, 8), dtype=X.dtype)

    log_term = xp.log(x2 / x1)
    denom = (x3 / x5) + log_term * (1.5 + 2 * x7 / (x8 * x1**2) * x3 / log_term)
    common_factor = -5 * x3 * (x4 - x6) / denom**2

    g[:, 0] = common_factor * (
        -1 / x1 * (1.5 + 2 * x3 * x7 / (x8 * x1**2) / log_term)
        + log_term
        * (
            -4 * x3 * x7 / (x8 * x1**3) / log_term
            + 2 * x3 * x7 / (x8 * x1**3) / log_term**2
        )
    )
    g[:, 1] = common_factor * (
        -2 / (x1**2 * x2) * x3 * x7 / x8 / log_term
        + 1 / x2 * (1.5 + 2 * x3 * x7 / (x8 * x1**2) / log_term)
    )
    g[:, 2] = 5 * (x4 - x6) / denom + common_factor * (1 / x5 + 2 * x7 / (x8 * x1**2))
    g[:, 3] = 5 * x3 / denom
    g[:, 4] = common_factor * (x3**2 / x5**2)
    g[:, 5] = -5 * x3 / denom
    g[:, 6] = common_factor * (-10 * x3**2 / (x8 * x1**2))
    g[:, 7] = common_factor * (10 * x3**2 * x7 / (x8**2 * x1**2))
    return g.reshape(n_samples, 1, 8)
