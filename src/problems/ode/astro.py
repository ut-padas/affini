"""
Astrochemical networks from KIDApy (``libs/KIDApy``, checked out beside ``src``; nothing here registers
without it): the Nelson network, 14 species. The state is the species' abundances per H nucleus and the
environment (n_H, T, A_v, UV flux), which is held over a step (KIDApy's piecewise-constant tracer) and
changes between steps along each trajectory, as a hydrodynamic tracer's does. All components are seen
in log10. The right-hand side is KIDApy's dx/dt = A(env) x + B(env)(x (x) x) with its analytic Jacobian;
A and B are rebuilt only when the environment changes.
"""

import re
import sys
from pathlib import Path

import numpy as np

from .system import System

KIDAPY = Path(__file__).resolve().parents[3] / "libs" / "KIDApy"
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


ELEMENTS = ("H", "He", "C", "N", "O", "Si", "S", "Fe", "Na", "Mg", "Cl", "P", "F")  # the elements of KIDA's species files, in their order


def composition(species: str) -> dict:
    """
    Element counts, charge and grain count of a KIDApy species from its name: ``C10H+`` is 10 C, 1 H and charge +1;
    isomer prefixes (``l-``, ``c-``, ``t-``) are dropped; ``e-`` is charge -1; the pseudo-grain species ``GRAIN0`` and
    ``GRAIN-`` are one grain each (charge 0 and -1), and ``XH`` is a hydrogen atom on a grain. Lumped species of reduced
    networks (Nelson and Langer's) are read as written: ``M`` is one atom of a lumped metal, and a trailing ``x`` (``CHx``,
    ``OHx``) is an untracked number of hydrogens, returned as ``{"H": None}``.
    """
    if species == "e-":
        return {"charge": -1}
    if species.startswith("GRAIN"):
        return {"grain": 1, "charge": -species.count("-") + species.count("+")}
    if species == "XH":
        return {"H": 1}
    body = re.sub(r"^[a-z]-", "", species)
    charge = body.count("+") - body.count("-")
    body = body.rstrip("+-")
    out: dict = {"charge": charge} if charge else {}
    if body.endswith("Hx"):
        out["H"] = None
        body = body[:-2]
    parts = re.findall(r"(He|Si|Fe|Na|Mg|Cl|[HCNOSPFM])(\d*)", body)
    if "".join(sym + n for sym, n in parts) != body:
        raise ValueError(f"cannot parse the species {species!r}")
    for sym, n in parts:
        out[sym] = out.get(sym, 0) + (int(n) if n else 1)
    return out


def conserved_totals(species, reactions) -> np.ndarray:
    """
    (laws, species) the element, charge and grain totals a network conserves: a quantity is a law when every reaction
    conserves it, counted over the network's own species (a product the network does not track, such as the ``H+`` of a
    reduced network, takes its share out). A quantity some species carries in an untracked amount (the hydrogen of a
    lumped ``CHx``) is not a law. Checked reaction by reaction from the stoichiometry, so a law is exact, not sampled.
    """
    index = {sp: j for j, sp in enumerate(species)}
    rows, untracked = {}, set()
    for j, sp in enumerate(species):
        for q, n in composition(sp).items():
            if n is None:
                untracked.add(q)
            else:
                rows.setdefault(q, np.zeros(len(species)))[j] = n
    laws = []
    for q in sorted(rows):
        if q in untracked:
            continue
        w = rows[q]
        if all(sum(w[index[x]] for x in r["products"] if x in index) == sum(w[index[x]] for x in r["reactants"] if x in index) for r in reactions):
            laws.append(w)
    return np.array(laws).reshape(-1, len(species))


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

    W = conserved_totals(net.species, net.reactions)
    W = np.concatenate([W, np.zeros((len(W), len(ENV)))], axis=1)
    return System(name, tuple(net.species) + ENV, rhs, jac, rtol=rtol, atol=atol, method="BDF", held=len(ENV), conserved=W)


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
