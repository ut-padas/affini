"""
Astrochemical networks from KIDApy (``libs/KIDApy``, checked out beside ``src``; nothing here registers
without it): the Nelson network, 14 species. The state is the species' abundances per H nucleus and the
environment (n_H, T, A_v, UV flux), which is held over a step (KIDApy's piecewise-constant tracer) and
changes between steps along each trajectory, as a hydrodynamic tracer's does. All components are seen
in log10. The right-hand side is KIDApy's dx/dt = A(env) x + B(env)(x (x) x) with its analytic Jacobian;
A and B are rebuilt only when the environment changes.
"""

import sys
from pathlib import Path

import numpy as np

from .system import System

KIDAPY = Path(__file__).resolve().parents[4] / "libs" / "KIDApy"
YEAR = 3600.0 * 24.0 * 365.25
ENV = ("nH", "T", "Av", "uv")


def _load(reactions: str, abundances: str, grains: bool):
    """(network, initial abundances) of a KIDApy network, or None when KIDApy is not checked out."""
    if not (KIDAPY / "parser.py").exists():
        return None
    if str(KIDAPY) not in sys.path:
        sys.path.insert(0, str(KIDAPY))  # KIDApy's modules import each other by bare name
    from parser import Network, load_abundances  # noqa: PLC0415

    net = Network(grains=grains, dust_attenuation=True)
    net.load_from_disk(str(KIDAPY / "networks" / reactions))
    net.drop_passive_species()
    x0 = np.zeros(len(net.species))
    for name, value in load_abundances(str(KIDAPY / "networks" / abundances)).items():
        if name in net.species_map:
            x0[net.species_map[name]] = value
    return net, x0


def kidapy_system(name: str, net, rtol: float = 1e-6, atol: float = 1e-24) -> System:
    """A KIDApy network as a System whose last four components are the held environment (zero derivative)."""
    from solver import QuadraticSolver, _make_jac  # noqa: PLC0415

    N = len(net.species)
    cache: dict = {}
    q = QuadraticSolver()

    def operators(env):
        key = tuple(np.round(env, 12))
        if key not in cache:
            if len(cache) > 256:
                cache.clear()
            A, B = net.get_operators_vectorized({"nH": env[0], "T": env[1], "Av": env[2], "uv_flux": env[3], "Tcap_2body": True})
            cache[key] = (A, B, _make_jac(A.tocsr(), B, N))
        return cache[key]

    def rhs(y):
        A, B, _ = operators(y[N:])
        return np.concatenate([q.compute_rhs(y[:N], A, B), np.zeros(len(ENV))])

    def jac(y):
        _, _, jf = operators(y[N:])
        J = np.zeros((N + len(ENV), N + len(ENV)))
        J[:N, :N] = jf(0.0, y[:N]).toarray()
        return J

    return System(name, tuple(net.species) + ENV, rhs, jac, rtol=rtol, atol=atol, method="BDF", held=len(ENV))


def environment(n: int, seed: int) -> np.ndarray:
    """(n, 4) environments, log-uniform: n_H 1e2-1e4 cm^-3, T 50-3000 K, A_v 0.1-2 mag, UV 50-200 Draine."""
    rng = np.random.default_rng(seed)
    return np.stack([10.0 ** rng.uniform(2, 4, n), 10.0 ** rng.uniform(np.log10(50), np.log10(3000), n),
                     10.0 ** rng.uniform(-1, np.log10(2), n), 10.0 ** rng.uniform(np.log10(50), np.log10(200), n)], axis=1)  # fmt: skip


def environment_path(n: int, seed: int, steps: int) -> np.ndarray:
    """
    (n, steps + 1, 4) environments along n tracers: each parameter oscillates in log10 about its starting
    value (amplitudes 0.4 / 0.3 / 0.2 / 0.1 dex for n_H / T / A_v / UV, random periods of 8-30 steps and
    phases), so the chemistry is driven out of equilibrium at every step, as along a hydrodynamic tracer.
    """
    rng = np.random.default_rng(seed + 7919)
    base = np.log10(environment(n, seed))
    amp = np.array([0.4, 0.3, 0.2, 0.1])
    period = rng.uniform(8, 30, (n, 1, 4))
    phase = rng.uniform(0, 2 * np.pi, (n, 1, 4))
    k = np.arange(steps + 1)[None, :, None]
    return 10.0 ** (base[:, None, :] + amp * (np.sin(2 * np.pi * k / period + phase) - np.sin(phase)))


_NELSON = _load("nelson/gas_reactions.in", "nelson/abundances.in", grains=False)
NELSON = kidapy_system("nelson", _NELSON[0]) if _NELSON else None


def nelson_initial(n: int, seed: int) -> np.ndarray:
    """KIDApy's initial abundances (the burn-in relaxes them) beside each tracer's first environment."""
    return np.concatenate([np.tile(np.maximum(_NELSON[1], 1e-20), (n, 1)), environment(n, seed)], axis=1)
