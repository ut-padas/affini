"""
Step maps of ODE systems as surrogate problems (``STEP_MAPS``): a ``StepMapProblem`` pairs a ``System`` with
its step ``dt``, the initial states of its trajectories and how many to visit (training and test), the
abundance floor below which a component is not scored, and its regime (``REGIMES``: which overview it is
shown in). The method-of-lines families of ``FAMILIES`` appear at every size, as ``<family>_d<inputs>``.
"""

from dataclasses import dataclass

import numpy as np

from . import astro
from .astro import NELSON, environment_path, nelson_initial
from .catalogue import FAMILIES
from .chemistry import (
    CHEMAKZO,
    HIRES,
    POLLU,
    PRIMORDIAL,
    ROBERTSON,
    chemakzo_initial,
    hires_initial,
    pollu_initial,
    robertson_initial,
)
from .classic import OREGONATOR, VAN_DER_POL, oregonator_initial, van_der_pol_initial
from .system import SYSTEMS, System

YEAR = 3.156e7


@dataclass(frozen=True)
class StepMapProblem:
    system: System
    dt: float
    initial: object  # (n, seed) -> (n, components) states in the system's own variables
    steps: int
    n_train: int
    n_test: int
    burn: float = 0.0  # trajectories start after this time, past the initial transient
    floor: float = 0.0  # components below this abundance are not scored (an error in dex of ~0 means nothing)
    regime: str = "stiff"  # a key of REGIMES: which overview the problem is shown in
    held_path: object = None  # (n, seed, steps) -> (n, steps + 1, held): the held inputs along each trajectory, e.g. a tracer's environment

    @property
    def units(self) -> str:
        return "dex" if self.system.log else "absolute"

    def step(self, Z: np.ndarray) -> np.ndarray:
        return self.system.coords_step(Z, self.dt)

    def scored(self, Z_true: np.ndarray) -> np.ndarray:
        """Which entries count in the error: abundances above ``floor`` (held inputs never; every other entry of a linear system)."""
        if not self.system.log:
            mask = np.ones(Z_true.shape, dtype=bool)
        else:
            mask = Z_true > (np.log10(self.floor) if self.floor > 0 else -np.inf)
        if self.system.held:
            mask[..., -self.system.held :] = False
        return mask


def _primordial_initial(n: int, seed: int) -> np.ndarray:
    """n_H in [1, 1e4] cm^-3, T in [100, 5000] K, ionisation 1e-6-1e-2, H2 fraction 1e-8-1e-3 (log-uniform), He/H = 0.079."""
    rng = np.random.default_rng(seed)
    nH, T = 10.0 ** rng.uniform(0, 4, n), 10.0 ** rng.uniform(2, np.log10(5000.0), n)
    xe, xh2 = 10.0 ** rng.uniform(-6, -2, n), 10.0 ** rng.uniform(-8, -3, n)
    Y = np.full((n, 10), 1e-20) * nH[:, None]
    Y[:, 0], Y[:, 1], Y[:, 3], Y[:, 5] = nH * (1 - xe - 2 * xh2), nH * xe, nH * xh2, nH * xe
    Y[:, 6], Y[:, 9] = 0.079 * nH, T
    return Y


STEP_MAPS = {
    "robertson": StepMapProblem(ROBERTSON, 1.0, robertson_initial, 40, 200, 100),
    "chemakzo": StepMapProblem(CHEMAKZO, 2.0, chemakzo_initial, 85, 100, 40, burn=5.0, floor=1e-12),
    "hires": StepMapProblem(HIRES, 2.0, hires_initial, 170, 100, 40, burn=5.0, floor=1e-12),
    "primordial": StepMapProblem(PRIMORDIAL, 1e4 * YEAR, _primordial_initial, 30, 60, 30, floor=1e-12),
    "pollu": StepMapProblem(POLLU, 1.0, pollu_initial, 59, 100, 40, burn=1.0, floor=1e-12),
    "van_der_pol": StepMapProblem(VAN_DER_POL, 0.05, van_der_pol_initial, 40, 100, 40),
    "oregonator": StepMapProblem(OREGONATOR, 10.0, oregonator_initial, 40, 100, 40),
}  # fmt: skip (in order of input dimension: 3, 5, 8, 10, 20)
if NELSON is not None:  # KIDApy's Nelson network along driven tracers: 66.7 kyr hydro steps (KIDApy's example tracer)
    STEP_MAPS["nelson"] = StepMapProblem(NELSON, 1600.0e3 * astro.YEAR / 24, nelson_initial, 24, 60, 20,
                                    burn=1.0e4 * astro.YEAR, floor=1e-12, regime="longstep", held_path=environment_path)  # fmt: skip
REGIMES = {"stiff": "Stiff systems", "nonstiff": "Non-stiff systems", "chaotic": "Chaotic systems",
           "longstep": "Long-step stiff systems (many solver steps per call, driven inputs)"}
# The ladder: each family of FAMILIES at every size, named <family>_d<input dimension>.
_LADDER_STEP = {"bruss": (0.25, 40), "allen_cahn": (0.005, 40), "fisher_kpp": (0.01, 40), "lorenz96": (0.25, 40), "burgers": (0.05, 40)}  # (dt, steps): 40 steps that still move
_LADDER_REGIME = {"lorenz96": "nonstiff"}  # the others: stiff
for _family, (_make, _initial, _sizes) in FAMILIES.items():
    for _n in _sizes:
        _system = SYSTEMS[_make(_n).name]
        STEP_MAPS[f"{_family}_d{len(_system.components)}"] = StepMapProblem(_system, _LADDER_STEP[_family][0], _initial(_n), _LADDER_STEP[_family][1],
                                                                  100, 40, regime=_LADDER_REGIME.get(_family, "stiff"))  # fmt: skip
