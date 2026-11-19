"""Every system registered in ``SYSTEMS`` by name, and the method-of-lines families (``FAMILIES``) at their sizes."""

from .astro import NELSON
from .classic import OREGONATOR, VAN_DER_POL, burgers, burgers_initial
from .lorenz import lorenz96, lorenz96_initial
from .reaction_diffusion import (
    allen_cahn,
    allen_cahn_initial,
    bruss,
    bruss_initial,
    fisher_kpp,
    fisher_kpp_initial,
)
from .system import SYSTEMS

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
