"""
ODE systems as surrogate oracles (``system``), by kind: chemical networks (``chemistry``), method-of-lines
reaction-diffusion (``reaction_diffusion``) and Lorenz-96 (``lorenz``); ``bench`` trains and judges
surrogates of their step maps and ``fill`` climbs the ladder of problems.
"""

from .chemistry import (
                        CHEMAKZO,
                        HIRES,
                        POLLU,
                        PRIMORDIAL,
                        ROBERTSON,
                        chemakzo_initial,
                        hires_initial,
                        pollu_initial,
                        primordial_rates,
                        robertson_initial,
)
from .lorenz import lorenz96, lorenz96_initial
from .reaction_diffusion import (
                        allen_cahn,
                        allen_cahn_initial,
                        bruss,
                        bruss_initial,
                        fisher_kpp,
                        fisher_kpp_initial,
)
from .system import FLOOR, SYSTEMS, WORKERS, System, trajectories

# family: (system of size n, initial states of size n, sizes n)
FAMILIES = {"bruss": (bruss, bruss_initial, (1, 2, 3, 5, 10, 20, 40)),
            "allen_cahn": (allen_cahn, allen_cahn_initial, (2, 3, 5, 10, 20, 40, 80)),
            "fisher_kpp": (fisher_kpp, fisher_kpp_initial, (2, 3, 5, 10, 20, 40, 80)),
            "lorenz96": (lorenz96, lorenz96_initial, (4, 5, 10, 20, 40))}  # fmt: skip
SYSTEMS.update({system.name: system for make, _, sizes in FAMILIES.values() for system in map(make, sizes)})

__all__ = ["CHEMAKZO", "FAMILIES", "FLOOR", "HIRES", "POLLU", "PRIMORDIAL", "ROBERTSON", "SYSTEMS", "WORKERS", "System",
           "allen_cahn", "allen_cahn_initial", "bruss", "bruss_initial", "chemakzo_initial", "hires_initial", "pollu_initial", "fisher_kpp", "fisher_kpp_initial", "lorenz96",
           "lorenz96_initial", "primordial_rates", "robertson_initial", "trajectories"]
