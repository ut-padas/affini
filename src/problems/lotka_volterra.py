"""A competitive Lotka-Volterra network's flow map: every species (``lotka_volterra_state``) or the first (``lotka_volterra``)."""

import numpy as np

from src import xp


def lotka_volterra_state(x, tf: float = 2.0, steps: int = 200):
    """
    Every species of a competitive Lotka-Volterra network at ``tf``: ``x_i' = x_i (r_i - sum_j A_ij x_j)``,
    with growth rates and interactions drawn once from a fixed seed (A_ii = 1, off-diagonal in [0, 0.8]),
    by fixed-step RK4 (row by row, so a row's value does not depend on the others; error about 1e-10).
    """
    d = int(x.shape[1])
    rng = np.random.default_rng(1_000 + d)
    r_ = xp.asarray(rng.uniform(0.5, 1.5, d))
    A = rng.uniform(0.0, 0.8, (d, d))
    np.fill_diagonal(A, 1.0)
    A = xp.asarray(A)

    def rhs(y):
        return y * (r_ - y @ A.T)

    y, h = xp.asarray(x, dtype=xp.float64), tf / steps
    for _ in range(steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def lotka_volterra(x, tf: float = 2.0, steps: int = 200):
    """The first species of ``lotka_volterra_state``."""
    return lotka_volterra_state(x, tf, steps)[:, :1]
