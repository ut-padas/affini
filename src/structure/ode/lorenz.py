"""Lorenz-96, a scalable non-stiff system (linear coordinates, RK45)."""

import numpy as np

from .system import System

L96_F = 1.0  # above F = 8/9 the uniform state x = F is unstable to a travelling wave (n >= 5), and the wave is stable


def lorenz96(n: int) -> System:
    """
    Lorenz-96 on n >= 4 cyclic sites, ``x_i' = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F`` (Lorenz 1996), at
    F = 1: non-stiff (damping rate 1), and every trajectory settles onto a travelling wave, along which
    errors neither grow nor decay, so a rollout's error is the surrogate's, not chaos (linear coordinates,
    integrated by RK45).
    """
    i = np.arange(n)
    ip, im, im2 = (i + 1) % n, (i - 1) % n, (i - 2) % n

    def jac(x):
        J = -np.eye(n)
        J[i, ip] += x[im]
        J[i, im2] -= x[im]
        J[i, im] += x[ip] - x[im2]
        return J

    return System(f"lorenz96_{n}", tuple(f"x{k}" for k in range(n)), lambda x: (x[ip] - x[im2]) * x[im] - x + L96_F, jac,
                   atol=1e-12, log=False, method="RK45", params=f"F = {L96_F:g}")  # fmt: skip


def lorenz96_initial(n: int):
    return lambda m, seed: L96_F + np.random.default_rng(seed).uniform(-1.0, 1.0, (m, n))
