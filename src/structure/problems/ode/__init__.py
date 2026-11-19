"""
ODE systems as surrogate oracles (``system``), by kind: chemical networks (``chemistry``), method-of-lines
reaction-diffusion (``reaction_diffusion``), Lorenz-96 (``lorenz``) and classic stiff test problems from other
fields (``classic``: Van der Pol, the Oregonator, viscous Burgers); ``bench`` trains and judges
surrogates of their step maps and ``fill`` climbs the ladder of problems.
"""

from .astro import NELSON, environment_path, nelson_initial
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
from .classic import (
                        OREGONATOR,
                        VAN_DER_POL,
                        burgers,
                        burgers_initial,
                        oregonator_initial,
                        van_der_pol_initial,
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
            "lorenz96": (lorenz96, lorenz96_initial, (4, 5, 10, 20, 40)),
            "burgers": (burgers, burgers_initial, (10, 20, 40))}  # fmt: skip
SYSTEMS.update({system.name: system for system in (VAN_DER_POL, OREGONATOR)})
if NELSON is not None:  # KIDApy checked out in libs/
    SYSTEMS[NELSON.name] = NELSON
SYSTEMS.update({system.name: system for make, _, sizes in FAMILIES.values() for system in map(make, sizes)})

__all__ = ["CHEMAKZO", "FAMILIES", "FLOOR", "HIRES", "NELSON", "environment_path", "nelson_initial", "OREGONATOR", "POLLU", "PRIMORDIAL", "ROBERTSON", "SYSTEMS", "VAN_DER_POL", "WORKERS", "System",
           "burgers", "burgers_initial", "oregonator_initial", "van_der_pol_initial",
           "allen_cahn", "allen_cahn_initial", "bruss", "bruss_initial", "chemakzo_initial", "hires_initial", "pollu_initial", "fisher_kpp", "fisher_kpp_initial", "lorenz96",
           "lorenz96_initial", "primordial_rates", "robertson_initial", "trajectories"]
