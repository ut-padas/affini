"""
ODE systems as surrogate oracles (``system``), by kind: chemical networks (``chemistry``), method-of-lines
reaction-diffusion (``reaction_diffusion``), Lorenz-96 (``lorenz``) and classic stiff test problems from other
fields (``classic``: Van der Pol, the Oregonator, viscous Burgers) and KIDApy's networks (``astro``); every
system by name in ``SYSTEMS`` (``catalogue``), and their step maps as surrogate problems (``step_maps``).
"""

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
from .step_maps import REGIMES, STEP_MAPS, YEAR, StepMapProblem
from .system import FLOOR, SYSTEMS, WORKERS, System, trajectories

__all__ = ["REGIMES", "STEP_MAPS", "YEAR", "StepMapProblem", "CHEMAKZO", "FAMILIES", "FLOOR", "HIRES", "NELSON", "environment_path", "nelson_initial", "OREGONATOR", "POLLU", "PRIMORDIAL", "ROBERTSON", "SYSTEMS", "VAN_DER_POL", "WORKERS", "System",
           "burgers", "burgers_initial", "oregonator_initial", "van_der_pol_initial",
           "allen_cahn", "allen_cahn_initial", "bruss", "bruss_initial", "chemakzo_initial", "hires_initial", "pollu_initial", "fisher_kpp", "fisher_kpp_initial", "lorenz96",
           "lorenz96_initial", "primordial_rates", "robertson_initial", "trajectories"]
