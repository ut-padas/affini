"""
Classic stiff test problems outside the chemistry networks (Hairer and Wanner, Solving ODEs II), so that
a surrogate is not judged only on systems with positivity and conservation: the Van der Pol relaxation
oscillator, the Oregonator, and viscous Burgers by the method of lines.
"""

import numpy as np

from .system import System

VDP_EPS = 1e-3  # y2' = ((1 - y1^2) y2 - y1) / eps: relaxation jumps on time scale eps, slow drift on scale 1


def _vdp_rhs(y):
    return np.array([y[1], ((1.0 - y[0] ** 2) * y[1] - y[0]) / VDP_EPS])


def _vdp_jac(y):
    return np.array([[0.0, 1.0], [(-2.0 * y[0] * y[1] - 1.0) / VDP_EPS, (1.0 - y[0] ** 2) / VDP_EPS]])


VAN_DER_POL = System("van_der_pol", ("y1", "y2"), _vdp_rhs, _vdp_jac, rtol=1e-8, atol=1e-10, log=False)


def van_der_pol_initial(n: int, seed: int) -> np.ndarray:
    """y1 in [-2.2, 2.2], y2 in [-1, 1]: most states off the slow manifold, so the first step contains a fast jump."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(-2.2, 2.2, n), rng.uniform(-1.0, 1.0, n)], axis=1)


def _oreg_rhs(y):
    y1, y2, y3 = y
    return np.array([77.27 * (y2 + y1 * (1.0 - 8.375e-6 * y1 - y2)), (y3 - (1.0 + y1) * y2) / 77.27, 0.161 * (y1 - y3)])


def _oreg_jac(y):
    y1, y2, y3 = y
    return np.array([[77.27 * (1.0 - 2 * 8.375e-6 * y1 - y2), 77.27 * (1.0 - y1), 0.0],
                     [-y2 / 77.27, -(1.0 + y1) / 77.27, 1.0 / 77.27], [0.161, 0.0, -0.161]])  # fmt: skip


OREGONATOR = System("oregonator", ("y1", "y2", "y3"), _oreg_rhs, _oreg_jac, rtol=1e-8, atol=1e-10)


def oregonator_initial(n: int, seed: int) -> np.ndarray:
    """Log-uniform over the ranges the limit cycle spans: y1 in [1, 1e5], y2 in [1, 2e3], y3 in [1, 3e4]."""
    rng = np.random.default_rng(seed)
    return np.stack([10.0 ** rng.uniform(0, 5, n), 10.0 ** rng.uniform(0, np.log10(2e3), n), 10.0 ** rng.uniform(0, np.log10(3e4), n)], axis=1)


BURGERS_NU = 0.1  # with dt = 0.05: nu dt / h^2 = 0.5 at 10 points, 2 at 20, 8 at 40 (stiff from 20)


def burgers(n: int) -> System:
    """u' = -(u^2 / 2)_x + nu u_xx on n points of a periodic [0, 1) (central differences, linear coordinates)."""
    h = 1.0 / n
    i = np.arange(n)
    ip, im = (i + 1) % n, (i - 1) % n

    def rhs(u):
        return -(u[ip] ** 2 - u[im] ** 2) / (4.0 * h) + BURGERS_NU * (u[ip] - 2.0 * u + u[im]) / h**2

    def jac(u):
        J = np.zeros((n, n))
        J[i, ip] += -u[ip] / (2.0 * h) + BURGERS_NU / h**2
        J[i, im] += u[im] / (2.0 * h) + BURGERS_NU / h**2
        J[i, i] += -2.0 * BURGERS_NU / h**2
        return J

    return System(f"burgers{n}", tuple(f"u{k}" for k in range(n)), rhs, jac, atol=1e-12, log=False)


def burgers_initial(n: int):
    """0.5 + random Fourier modes k <= 3 with amplitude ~1/k (max |u - 0.5| = 1): steepening fronts that viscosity holds."""

    def initial(m: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = np.arange(n) / n
        k = np.arange(1, 4)
        a, c = rng.uniform(-1, 1, (m, 3)) / k, rng.uniform(-1, 1, (m, 3)) / k
        F = a @ np.sin(2 * np.pi * np.outer(k, x)) + c @ np.cos(2 * np.pi * np.outer(k, x))
        return 0.5 + F / np.max(np.abs(F), axis=1, keepdims=True)

    return initial
