"""
Stiff chemical networks as surrogate oracles: the time-step map of a network and its visited states.

A network is ``dy/dt = F(y)`` for abundances ``y > 0``. The oracle a simulation would call is the step
map ``Phi_dt(y)``, integrated by an implicit Runge-Kutta method (SciPy's Radau with the analytic
Jacobian), one state at a time so that no state's result depends on the others in its batch. Inputs
and outputs are in log10 abundances: abundances span many orders of magnitude, a relative error is the
meaningful one, and an additive error of e in log10 is a factor 10^e.

The states a simulation visits lie near the network's slow manifold: after a fast transient, the fast
modes have relaxed, and the step map's Jacobian has low numerical rank (Robertson: singular values
about 1.7, 1.0 and 1e-10 for every step dt >= 0.1). So surrogates are trained and judged on states of
trajectories, ``trajectories``, not on a box: held-out trajectories give the test states.
"""

import os
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

FLOOR = 1e-30  # abundances below it are clipped before the log10, so the map is defined everywhere
WORKERS = int(os.environ.get("SURROGATE_CHEM_WORKERS", "1"))  # processes per Network.step batch


@dataclass(frozen=True)
class Network:
    """A network's right-hand side and Jacobian (functions of one state) and its species."""

    name: str
    species: tuple[str, ...]
    rhs: object
    jac: object
    rtol: float = 1e-8
    atol: float = 1e-20

    def step(self, Y: np.ndarray, dt: float) -> np.ndarray:
        """
        ``Phi_dt`` of every row of Y (abundances), each integrated on its own, so a row's result never
        depends on the others; large batches are spread over ``WORKERS`` forked processes.
        """
        Y = np.asarray(Y, dtype=np.float64)
        if WORKERS > 1 and len(Y) >= 4 * WORKERS:
            import multiprocessing

            with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
                parts = pool.starmap(_step_rows, [(self.name, part, dt) for part in np.array_split(Y, WORKERS)])
            return np.concatenate(parts)
        return _step_rows(self.name, Y, dt)

    def log_step(self, Z: np.ndarray, dt: float) -> np.ndarray:
        """The step map in log10 abundances."""
        return np.log10(np.maximum(self.step(10.0 ** np.asarray(Z, dtype=np.float64), dt), FLOOR))


def _step_rows(name: str, Y: np.ndarray, dt: float) -> np.ndarray:
    """``Network.step`` of the registered network ``name`` for the rows of Y, serially (a picklable worker)."""
    net = NETWORKS[name]
    out = np.empty_like(Y)
    for i, y in enumerate(Y):
        s = solve_ivp(lambda t, v: net.rhs(v), (0.0, dt), y, method="Radau", jac=lambda t, v: net.jac(v),
                      rtol=net.rtol, atol=net.atol)  # fmt: skip
        out[i] = s.y[:, -1] if s.success else np.nan
    return out




def _robertson_rhs(y, k1=0.04, k2=3e7, k3=1e4):
    y1, y2, y3 = y
    return np.array([-k1 * y1 + k3 * y2 * y3, k1 * y1 - k3 * y2 * y3 - k2 * y2 * y2, k2 * y2 * y2])


def _robertson_jac(y, k1=0.04, k2=3e7, k3=1e4):
    y1, y2, y3 = y
    return np.array([[-k1, k3 * y3, k3 * y2], [k1, -k3 * y3 - 2 * k2 * y2, -k3 * y2], [0.0, 2 * k2 * y2, 0.0]])


ROBERTSON = Network("robertson", ("y1", "y2", "y3"), _robertson_rhs, _robertson_jac, atol=1e-16)


# --- Primordial H/He/H2 chemistry (nine species, isothermal) --------------------------------
# Rate coefficients transcribed from Grackle's src/clib/rate_functions.c at commit 48e0733 (most fits from
# Abel, Anninos, Zhang and Norman 1997), with Grackle's defaults (case-A recombination, no three-body
# option, H2 charge exchange of Savin 2004) and the low-density k13; reaction identities from Grackle's
# species update (step_rate_g), since the comments in rate_functions.c mislabel k16-k19. Radiative rates
# and dust are left out. Check: k2(1e4 K) = 4.17e-13 cm^3/s, the case-A value.
TINY = 1e-20
TEVK = 1.1605e4


def _poly(c, x):
    """sum_i c_i x^i."""
    return sum(ci * x**i for i, ci in enumerate(c))


def primordial_rates(T: float) -> dict:
    """Rate coefficients k1-k19, k22, k57, k58 at temperature T (K), cgs (cm^3/s; k22 cm^6/s)."""
    T = float(T)
    Tev = T / 11605.0
    lT = np.log(Tev)
    k = {}
    k["k1"] = np.exp(_poly([-32.71396786375, 13.53655609057, -5.739328757388, 1.563154982022, -0.2877056004391,
                            0.03482559773736999, -0.00263197617559, 0.0001119543953861, -2.039149852002e-6], lT))
    if Tev <= 0.8:
        k["k1"] = max(TINY, k["k1"])
    k["k3"] = np.exp(_poly([-44.09864886561001, 23.91596563469, -10.75323019821, 3.058038757198, -0.5685118909884001,
                            0.06795391233790001, -0.005009056101857001, 0.0002067236157507, -3.649161410833e-6], lT)) if Tev > 0.8 else TINY
    k4_low = 3.92e-13 / Tev**0.6353
    k["k4"] = (1.54e-9 * (1.0 + 0.3 / np.exp(8.099328789667 / Tev)) / (np.exp(40.49664394833662 / Tev) * Tev**1.5) + k4_low) if Tev > 0.8 else k4_low
    if T > 5500:
        lTk = np.log(T / TEVK)
        k["k2"] = np.exp(_poly([-28.61303380689232, -0.7241125657826851, -0.02026044731984691, -0.002380861877349834,
                                -0.0003212605213188796, -0.00001421502914054107, 4.989108920299513e-6, 5.755614137575758e-7,
                                -1.856767039775261e-8, -3.071135243196595e-9], lTk))
    else:
        k["k2"] = k["k4"]  # Grackle's case-A branch below 5500 K returns k4_rate (as in the source)
    k["k5"] = np.exp(_poly([-68.71040990212001, 43.93347632635, -18.48066993568, 4.701626486759002, -0.7692466334492,
                            0.08113042097303, -0.005324020628287001, 0.0001975705312221, -3.165581065665e-6], lT)) if Tev > 0.8 else TINY
    k["k6"] = 3.36e-10 / np.sqrt(T) / (T / 1.0e3) ** 0.2 / (1.0 + (T / 1.0e6) ** 0.7)
    k["k7"] = 3.0e-16 * (T / 3.0e2) ** 0.95 * np.exp(-T / 9.32e3)
    k["k8"] = 1.35e-9 * (T**9.8493e-2 + 3.2852e-1 * T**5.5610e-1 + 2.771e-7 * T**2.1826) / (
        1.0 + 6.191e-3 * T**1.0461 + 8.9712e-11 * T**3.0424 + 3.2576e-14 * T**3.7741)
    if T < 30.0:
        k["k9"] = 2.10e-20 * (T / 30.0) ** -0.15
    else:
        L = np.log10(min(T, 3.2e4))
        k["k9"] = 10.0 ** (-18.20 - 3.194 * L + 1.786 * L**2 - 0.2072 * L**3)
    k["k10"] = 6.0e-10
    lTK = np.log(T)
    k["k11"] = (np.exp(-21237.15 / T) * _poly([-3.3232183e-07, 3.3735382e-07, -1.4491368e-07, 3.4172805e-08, -4.7813720e-09,
                                                3.9731542e-10, -1.8171411e-11, 3.5311932e-13], lTK)) if Tev > 0.3 else TINY
    k["k12"] = 4.4886e-9 * T**0.109127 * np.exp(-101858.0 / T) if Tev > 0.3 else TINY
    k["k13"] = 1.0670825e-10 * Tev**2.012 / (np.exp(4.463 / Tev) * (1.0 + 0.2472 * Tev) ** 3.512) if Tev > 0.3 else TINY
    k["k14"] = np.exp(_poly([-18.01849334273, 2.360852208681, -0.2827443061704, 0.01623316639567, -0.03365012031362999,
                             0.01178329782711, -0.001656194699504, 0.0001068275202678, -2.631285809207e-6], lT)) if Tev > 0.04 else TINY
    k["k15"] = np.exp(_poly([-20.37260896533324, 1.139449335841631, -0.1421013521554148, 0.00846445538663, -0.0014327641212992,
                             0.0002012250284791, 0.0000866396324309, -0.00002585009680264, 2.4555011970392e-6,
                             -8.06838246118e-8], lT)) if Tev > 0.1 else 2.56e-9 * Tev**1.78186
    k["k16"] = 2.4e-6 * (1.0 + T / 2.0e4) / np.sqrt(T)
    k["k17"] = 4.0e-4 * T**-1.4 * np.exp(-15100.0 / T) if T > 1.0e4 else 1.0e-8 * T**-0.4
    k["k18"] = 1.32e-6 * T**-0.76 if T > 617.0 else 1.0e-8
    k["k19"] = 5.0e-7 * np.sqrt(100.0 / T)
    k["k22"] = 1.3e-32 * (T / 300.0) ** (-0.38 if T <= 300.0 else -1.0)
    k["k57"] = 1.2e-17 * T**1.2 * np.exp(-1.578e5 / T) if T > 3.0e3 else TINY
    k["k58"] = 1.75e-17 * T**1.3 * np.exp(-1.578e5 / T) if T > 3.0e3 else TINY
    return k


PRIMORDIAL_SPECIES = ("H", "H+", "H-", "H2", "H2+", "e", "He", "He+", "He++")
# (rate name, reactants, products), mass action in number densities (cm^-3)
PRIMORDIAL_REACTIONS = (
    ("k1", ("H", "e"), ("H+", "e", "e")), ("k2", ("H+", "e"), ("H",)), ("k3", ("He", "e"), ("He+", "e", "e")),
    ("k4", ("He+", "e"), ("He",)), ("k5", ("He+", "e"), ("He++", "e", "e")), ("k6", ("He++", "e"), ("He+",)),
    ("k7", ("H", "e"), ("H-",)), ("k8", ("H", "H-"), ("H2", "e")), ("k9", ("H", "H+"), ("H2+",)),
    ("k10", ("H2+", "H"), ("H2", "H+")), ("k11", ("H2", "H+"), ("H2+", "H")), ("k12", ("H2", "e"), ("H", "H", "e")),
    ("k13", ("H2", "H"), ("H", "H", "H")), ("k14", ("H-", "e"), ("H", "e", "e")), ("k15", ("H-", "H"), ("H", "H", "e")),
    ("k16", ("H-", "H+"), ("H", "H")), ("k17", ("H-", "H+"), ("H2+", "e")), ("k18", ("H2+", "e"), ("H", "H")),
    ("k19", ("H2+", "H-"), ("H2", "H")), ("k22", ("H", "H", "H"), ("H2", "H")), ("k57", ("H", "H"), ("H+", "e", "H")),
    ("k58", ("H", "He"), ("H+", "e", "He")),
)  # fmt: skip


def mass_action(species, reactions, rate_of):
    """
    ``(rhs, jac)`` of a mass-action network whose last state entry is the temperature (held fixed):
    ``rate_of(T)`` gives the rate coefficients by name.
    """
    index = {s: i for i, s in enumerate(species)}
    ns = len(species)
    S = np.zeros((ns + 1, len(reactions)))
    reactants = []
    for j, (_, lhs, rhs_) in enumerate(reactions):
        for sp in lhs:
            S[index[sp], j] -= 1.0
        for sp in rhs_:
            S[index[sp], j] += 1.0
        reactants.append([index[sp] for sp in lhs])
    names = [name for name, _, _ in reactions]

    def rates(y):
        k = rate_of(y[-1])
        return np.array([k[name] * np.prod(y[r]) for name, r in zip(names, reactants, strict=True)])

    def rhs(y):
        return S @ rates(y)

    def jac(y):
        k = rate_of(y[-1])
        dR = np.zeros((len(reactions), ns + 1))
        for j, (name, r) in enumerate(zip(names, reactants, strict=True)):
            for a in range(len(r)):  # d/dy of k * prod(y[r]): drop one factor at a time (exact at y = 0 too)
                dR[j, r[a]] += k[name] * np.prod([y[r[b]] for b in range(len(r)) if b != a])
        return S @ dR

    return rhs, jac


_prim_rhs, _prim_jac = mass_action(PRIMORDIAL_SPECIES, PRIMORDIAL_REACTIONS, primordial_rates)
PRIMORDIAL = Network("primordial", PRIMORDIAL_SPECIES + ("T",), _prim_rhs, _prim_jac, rtol=1e-8, atol=1e-30)
NETWORKS = {net.name: net for net in (ROBERTSON, PRIMORDIAL)}


def trajectories(net: Network, Y0: np.ndarray, dt: float, steps: int) -> np.ndarray:
    """(n, steps + 1, species) log10 states of the trajectories from the rows of Y0, sampled every dt."""
    Z = np.empty((len(Y0), steps + 1, len(net.species)))
    Z[:, 0] = np.log10(np.maximum(Y0, FLOOR))
    for k in range(steps):
        Z[:, k + 1] = net.log_step(Z[:, k], dt)
    return Z


def robertson_initial(n: int, seed: int) -> np.ndarray:
    """Initial abundances: y1 in [0.5, 1], a trace of y2 in [1e-12, 1e-6], y3 the rest (they sum to 1)."""
    rng = np.random.default_rng(seed)
    y1 = rng.uniform(0.5, 1.0, n)
    y2 = 10.0 ** rng.uniform(-12, -6, n)
    return np.stack([y1, y2, 1.0 - y1 - y2], axis=1)
