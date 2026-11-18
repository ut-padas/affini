"""
Scalable stiff systems: method-of-lines reaction-diffusion on n grid points of [0, 1].

Stiff through diffusion (eigenvalues up to about 4 D / h^2) and, for Allen-Cahn, a fast reaction; each
is contracting in the regime chosen, so a rollout's error is the surrogate's, not the dynamics' chaos.
"""

import numpy as np

from .system import System


def _laplacian(n: int, neumann: bool) -> np.ndarray:
    """The second-difference matrix on n points (Neumann: cell centres, h = 1/n; Dirichlet: h = 1/(n + 1))."""
    L = np.diag(np.full(n - 1, 1.0), -1) - 2.0 * np.eye(n) + np.diag(np.full(n - 1, 1.0), 1)
    if neumann:
        L[0, 0] += 1.0
        L[-1, -1] += 1.0
        return L * n**2
    return L * (n + 1) ** 2


def _smooth_fields(n_states: int, n: int, rng) -> np.ndarray:
    """(n_states, n) random fields sum_k a_k sin(k pi x) + b_k cos(k pi x), k <= 3, a_k, b_k ~ U(-1, 1)/k, scaled to max |.| = 1."""
    x = (np.arange(n) + 0.5) / n
    k = np.arange(1, 4)
    a, b = rng.uniform(-1, 1, (n_states, 3)) / k, rng.uniform(-1, 1, (n_states, 3)) / k
    F = a @ np.sin(np.pi * np.outer(k, x)) + b @ np.cos(np.pi * np.outer(k, x))
    return F / np.max(np.abs(F), axis=1, keepdims=True)


BRUSS_A, BRUSS_B, BRUSS_ALPHA = 1.0, 1.9, 0.5  # B < 1 + A^2: the uniform state (A, B/A) is stable, and near the Hopf point B = 2 it rings down slowly


def bruss(n: int) -> System:
    """
    The Brusselator with diffusion (Hairer and Wanner, Solving ODEs II, section IV.1) on n points, with
    no-flux boundaries, so that diffusion damps only the non-uniform modes (the stiff part) and the
    uniform mode rings down at Re lambda = (B - 1 - A^2) / 2; components u_1..u_n, v_1..v_n. (Held at the
    steady state, Dirichlet boundaries damp every mode within a step or two on so few points.)
    """
    A, B = BRUSS_A, BRUSS_B
    L = BRUSS_ALPHA * _laplacian(n, neumann=True) if n > 1 else np.zeros((1, 1))

    def rhs(y):
        u, v = y[:n], y[n:]
        return np.concatenate([A + u * u * v - (B + 1.0) * u + L @ u, B * u - u * u * v + L @ v])

    def jac(y):
        u, v = y[:n], y[n:]
        return np.block([[np.diag(2 * u * v - (B + 1.0)) + L, np.diag(u * u)], [np.diag(B - 2 * u * v), np.diag(-u * u) + L]])

    return System(f"bruss{n}", tuple(f"u{i}" for i in range(n)) + tuple(f"v{i}" for i in range(n)), rhs, jac, atol=1e-12)


def bruss_initial(n: int):
    def initial(m: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.concatenate([BRUSS_A + 0.6 * _smooth_fields(m, n, rng), BRUSS_B / BRUSS_A + 1.0 * _smooth_fields(m, n, rng)], axis=1)
    return initial


AC_D, AC_DELTA = 1.0, 0.05


def allen_cahn(n: int) -> System:
    """u' = D u_xx + (u - u^3) / delta on n points, Neumann: a fast reaction to +-1 and slow interfaces (linear coordinates)."""
    L = AC_D * _laplacian(n, neumann=True)
    return System(f"allen_cahn{n}", tuple(f"u{i}" for i in range(n)), lambda y: L @ y + (y - y**3) / AC_DELTA,
                   lambda y: L + np.diag((1.0 - 3.0 * y * y) / AC_DELTA), atol=1e-12, log=False)  # fmt: skip


def allen_cahn_initial(n: int):
    return lambda m, seed: 0.9 * _smooth_fields(m, n, np.random.default_rng(seed))


KPP_D, KPP_R = 1.0, 10.0


def fisher_kpp(n: int) -> System:
    """u' = D u_xx + r u (1 - u) on n points, Neumann: growth from a trace towards the stable u = 1."""
    L = KPP_D * _laplacian(n, neumann=True)
    return System(f"fisher_kpp{n}", tuple(f"u{i}" for i in range(n)), lambda y: L @ y + KPP_R * y * (1.0 - y),
                   lambda y: L + np.diag(KPP_R * (1.0 - 2.0 * y)), atol=1e-14)  # fmt: skip


def fisher_kpp_initial(n: int):
    return lambda m, seed: 10.0 ** (-1.5 + 1.5 * _smooth_fields(m, n, np.random.default_rng(seed)))
