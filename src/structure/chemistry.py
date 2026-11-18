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

import atexit
import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

FLOOR = 1e-30  # abundances below it are clipped before the log10, so the map is defined everywhere
WORKERS = int(os.environ.get("SURROGATE_CHEM_WORKERS", "1"))  # processes per Network.step batch


@dataclass(frozen=True)
class Network:
    """
    A stiff system's right-hand side and Jacobian (functions of one state) and its components. ``log``:
    the surrogates see log10 of the state (abundances and other positive quantities, where a relative
    error is the meaningful one); otherwise the state itself.
    """

    name: str
    species: tuple[str, ...]
    rhs: object
    jac: object
    rtol: float = 1e-8
    atol: float = 1e-20
    log: bool = True

    def step(self, Y: np.ndarray, dt: float) -> np.ndarray:
        """
        ``Phi_dt`` of every row of Y (abundances), each integrated on its own, so a row's result never
        depends on the others; a batch of at least ``WORKERS`` rows is spread over that many forked
        processes. A cross asks for one fibre at a time (tens of states), and states far from the slow
        manifold cost Radau ~10 ms each, against well under a millisecond to hand a chunk to the pool.
        """
        Y = np.asarray(Y, dtype=np.float64)
        if WORKERS > 1 and len(Y) >= WORKERS:
            # many small tasks, not one per worker: a few states (far from the slow manifold) cost many
            # times the others, and a worker holding one would leave the rest idle
            chunks = np.array_split(Y, min(len(Y), 4 * WORKERS))
            parts = _pool().starmap(_step_rows, [(self.name, part, dt) for part in chunks], chunksize=1)
            return np.concatenate(parts)
        return _step_rows(self.name, Y, dt)

    def coords(self, Y: np.ndarray) -> np.ndarray:
        """States in the surrogates' coordinates."""
        return np.log10(np.maximum(Y, FLOOR)) if self.log else np.asarray(Y, dtype=np.float64)

    def coords_step(self, Z: np.ndarray, dt: float) -> np.ndarray:
        """The step map in the surrogates' coordinates."""
        Z = np.asarray(Z, dtype=np.float64)
        return self.coords(self.step(10.0**Z if self.log else Z, dt))

    def step_cost(self, Y: np.ndarray, dt: float) -> dict:
        """
        What one step costs the solver, per state, serially: wall time in microseconds and the counts of
        right-hand sides, Jacobians and LU factorisations (a measure independent of SciPy's Python
        overhead: a compiled solver pays about ``nfev`` right-hand sides and ``nlu`` factorisations).
        """
        t, nfev, njev, nlu = [], [], [], []
        for y in np.asarray(Y, dtype=np.float64):
            t0 = time.perf_counter()
            s = solve_ivp(lambda t, v: self.rhs(v), (0.0, dt), y, method="Radau", jac=lambda t, v: self.jac(v),
                          rtol=self.rtol, atol=self.atol)  # fmt: skip
            t.append(time.perf_counter() - t0)
            nfev.append(s.nfev), njev.append(s.njev), nlu.append(s.nlu)
        return {"solve_us": 1e6 * float(np.mean(t)), "nfev": float(np.mean(nfev)), "njev": float(np.mean(njev)),
                "nlu": float(np.mean(nlu))}  # fmt: skip


_POOL = None


def _pool():
    """One forked pool for the process's lifetime: forking per batch costs more than a small batch's solves."""
    global _POOL
    if _POOL is None:
        import multiprocessing

        _POOL = multiprocessing.get_context("fork").Pool(WORKERS)
        atexit.register(_POOL.terminate)  # before interpreter teardown, which a pool's finaliser does not survive
    return _POOL


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


# --- Standard stiff test problems -------------------------------------------------------------------
# HIRES, POLLU and Chemical Akzo Nobel from the Test Set for IVP Solvers (F. Mazzia, C. Magherini, Bari;
# release 2.4), transcribed from its Fortran problem files
# https://archimede.uniba.it/~testset/src/problems/{hires,pollu,chemakzo}.f and the problem descriptions
# https://archimede.uniba.it/~testset/report/{hires,pollu,chemakzo}.pdf; line numbers are those of the .f
# files. The Jacobians keep the files' 1-based indices (the state padded with a leading zero, the matrix
# cut back), so every entry reads as its dfdy(i,j) line. Integrated from the standard initial values with
# Radau and the analytic Jacobians, they match the published reference solutions to 1.3e-12 (HIRES),
# 4.2e-12 (POLLU) and 3.3e-13 (Akzo Nobel), and the Jacobians match central differences.
# --- HIRES: hires.f (8 ODEs; Schafer 1975, named by Hairer & Wanner) ---------------------------------
# Phytochrome photomorphogenesis: y = [Pr, Pfr, PrX, PfrX, PrX', PfrX', PfrX'E, E] (hires.pdf, sec. 1.3).
# feval: lines 107-115; jeval: lines 132-156; init: lines 54-61; t(1): line 41; solut: lines 178-185
# (RADAU5 on a Cray C90, uround 1.01e-19, rtol = atol = h0 = 1.1e-18).
HIRES_SPECIES = ("Pr", "Pfr", "PrX", "PfrX", "PrX'", "PfrX'", "PfrX'E", "E")
HIRES_Y0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057])
HIRES_TEND = 321.8122
HIRES_REF = np.array([0.7371312573325668e-3, 0.1442485726316185e-3, 0.5888729740967575e-4, 0.1175651343283149e-2,
                      0.2386356198831331e-2, 0.6238968252742796e-2, 0.2849998395185769e-2, 0.2850001604814231e-2])  # fmt: skip


def hires_rhs(y):
    """HIRES right-hand side; the constant 0.0007 is the Pr source (k_oks), and y7 + y8 is conserved."""
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    f7 = 280.0 * y6 * y8 - 1.81 * y7
    return np.array([
        -1.71 * y1 + 0.43 * y2 + 8.32 * y3 + 0.0007,
        1.71 * y1 - 8.75 * y2,
        -10.03 * y3 + 0.43 * y4 + 0.035 * y5,
        8.32 * y2 + 1.71 * y3 - 1.12 * y4,
        -1.745 * y5 + 0.43 * (y6 + y7),
        -280.0 * y6 * y8 + 0.69 * y4 + 1.71 * y5 - 0.43 * y6 + 0.69 * y7,
        f7,
        -f7,
    ])  # fmt: skip


def hires_jac(y):
    y = np.concatenate(([0.0], y))
    J = np.zeros((9, 9))
    J[1, 1] = -1.71
    J[1, 2] = 0.43
    J[1, 3] = 8.32
    J[2, 1] = 1.71
    J[2, 2] = -8.75
    J[3, 3] = -10.03
    J[3, 4] = 0.43
    J[3, 5] = 0.035
    J[4, 2] = 8.32
    J[4, 3] = 1.71
    J[4, 4] = -1.12
    J[5, 5] = -1.745
    J[5, 6] = 0.43
    J[5, 7] = 0.43
    J[6, 4] = 0.69
    J[6, 5] = 1.71
    J[6, 6] = -280.0 * y[8] - 0.43
    J[6, 7] = 0.69
    J[6, 8] = -280.0 * y[6]
    J[7, 6] = 280.0 * y[8]
    J[7, 7] = -1.81
    J[7, 8] = 280.0 * y[6]
    J[8, 6] = -280.0 * y[8]
    J[8, 7] = 1.81
    J[8, 8] = -280.0 * y[6]
    return J[1:, 1:]


def hires_initial(n, seed):
    """
    Pr in [0.5, 1.5] (all phytochrome starts as Pr), the enzyme total E = y7 + y8 log-uniform in
    [0.003, 0.011] (a factor 2 around the standard 0.0057), the other forms zero, as in the standard start.
    """
    rng = np.random.default_rng(seed)
    Y = np.zeros((n, 8))
    Y[:, 0] = rng.uniform(0.5, 1.5, n)
    Y[:, 7] = 10.0 ** rng.uniform(np.log10(0.003), np.log10(0.011), n)
    return Y


# --- POLLU: pollu.f (20 ODEs, 25 reactions; Verwer 1994, RIVM air pollution chemistry) ----------------
# Rate constants: lines 111-123 (k3 = 0.123e5; pollu.pdf notes a typing error of k3 in [Ver94]); rates
# r(j): lines 125-149; feval: lines 151-171; jeval: lines 203-303; init: lines 56-65; t(1): line 41;
# solut: lines 325-344 (RADAU5 on a Cray C90, uround 1.01e-19, rtol = atol = h0 = 1.1e-18).
# Time in minutes, concentrations in ppm (Verwer 1994). The photolysis rates are constant (no diurnal cycle).
POLLU_SPECIES = ("NO2", "NO", "O3P", "O3", "HO2", "OH", "HCHO", "CO", "ALD", "MEO2", "C2O3", "CO2", "PAN",
                 "CH3O", "HNO3", "O1D", "SO2", "SO4", "NO3", "N2O5")  # fmt: skip
POLLU_K = np.array([0.0,  # index 0 unused, so POLLU_K[j] is kj of pollu.f
                    .35, .266e2, .123e5, .86e-3, .82e-3, .15e5, .13e-3, .24e5, .165e5, .9e4, .22e-1, .12e5, .188e1,
                    .163e5, .48e7, .35e-3, .175e-1, .1e9, .444e12, .124e4, .21e1, .578e1, .474e-1, .178e4, .312e1])  # fmt: skip
POLLU_Y0 = np.zeros(20)
POLLU_Y0[[1, 3, 6, 7, 8, 16]] = [0.2, 0.04, 0.1, 0.3, 0.01, 0.007]  # y2, y4, y7, y8, y9, y17
POLLU_TEND = 60.0
POLLU_REF = np.array([0.5646255480022769e-01, 0.1342484130422339e+00, 0.4139734331099427e-08, 0.5523140207484359e-02,
                      0.2018977262302196e-06, 0.1464541863493966e-06, 0.7784249118997964e-01, 0.3245075353396018e+00,
                      0.7494013383880406e-02, 0.1622293157301561e-07, 0.1135863833257075e-07, 0.2230505975721359e-02,
                      0.2087162882798630e-03, 0.1396921016840158e-04, 0.8964884856898295e-02, 0.4352846369330103e-17,
                      0.6899219696263405e-02, 0.1007803037365946e-03, 0.1772146513969984e-05, 0.5682943292316392e-04])  # fmt: skip


def pollu_rhs(y):
    """POLLU right-hand side; f(j) and r(j) read as in pollu.f (1-based)."""
    k = POLLU_K
    y = np.concatenate(([0.0], y))
    r = np.zeros(26)
    r[1] = k[1] * y[1]
    r[2] = k[2] * y[2] * y[4]
    r[3] = k[3] * y[5] * y[2]
    r[4] = k[4] * y[7]
    r[5] = k[5] * y[7]
    r[6] = k[6] * y[7] * y[6]
    r[7] = k[7] * y[9]
    r[8] = k[8] * y[9] * y[6]
    r[9] = k[9] * y[11] * y[2]
    r[10] = k[10] * y[11] * y[1]
    r[11] = k[11] * y[13]
    r[12] = k[12] * y[10] * y[2]
    r[13] = k[13] * y[14]
    r[14] = k[14] * y[1] * y[6]
    r[15] = k[15] * y[3]
    r[16] = k[16] * y[4]
    r[17] = k[17] * y[4]
    r[18] = k[18] * y[16]
    r[19] = k[19] * y[16]
    r[20] = k[20] * y[17] * y[6]
    r[21] = k[21] * y[19]
    r[22] = k[22] * y[19]
    r[23] = k[23] * y[1] * y[4]
    r[24] = k[24] * y[19] * y[1]
    r[25] = k[25] * y[20]
    f = np.zeros(21)
    f[1] = -r[1] - r[10] - r[14] - r[23] - r[24] + r[2] + r[3] + r[9] + r[11] + r[12] + r[22] + r[25]
    f[2] = -r[2] - r[3] - r[9] - r[12] + r[1] + r[21]
    f[3] = -r[15] + r[1] + r[17] + r[19] + r[22]
    f[4] = -r[2] - r[16] - r[17] - r[23] + r[15]
    f[5] = -r[3] + r[4] + r[4] + r[6] + r[7] + r[13] + r[20]
    f[6] = -r[6] - r[8] - r[14] - r[20] + r[3] + r[18] + r[18]
    f[7] = -r[4] - r[5] - r[6] + r[13]
    f[8] = r[4] + r[5] + r[6] + r[7]
    f[9] = -r[7] - r[8]
    f[10] = -r[12] + r[7] + r[9]
    f[11] = -r[9] - r[10] + r[8] + r[11]
    f[12] = r[9]
    f[13] = -r[11] + r[10]
    f[14] = -r[13] + r[12]
    f[15] = r[14]
    f[16] = -r[18] - r[19] + r[16]
    f[17] = -r[20]
    f[18] = r[20]
    f[19] = -r[21] - r[22] - r[24] + r[23] + r[25]
    f[20] = -r[25] + r[24]
    return f[1:]


def pollu_jac(y):
    k = POLLU_K
    y = np.concatenate(([0.0], y))
    J = np.zeros((21, 21))
    J[1, 1] = -k[1] - k[10] * y[11] - k[14] * y[6] - k[23] * y[4] - k[24] * y[19]
    J[1, 11] = -k[10] * y[1] + k[9] * y[2]
    J[1, 6] = -k[14] * y[1]
    J[1, 4] = -k[23] * y[1] + k[2] * y[2]
    J[1, 19] = -k[24] * y[1] + k[22]
    J[1, 2] = k[2] * y[4] + k[9] * y[11] + k[3] * y[5] + k[12] * y[10]
    J[1, 13] = k[11]
    J[1, 20] = k[25]
    J[1, 5] = k[3] * y[2]
    J[1, 10] = k[12] * y[2]

    J[2, 4] = -k[2] * y[2]
    J[2, 5] = -k[3] * y[2]
    J[2, 11] = -k[9] * y[2]
    J[2, 10] = -k[12] * y[2]
    J[2, 19] = k[21]
    J[2, 1] = k[1]
    J[2, 2] = -k[2] * y[4] - k[3] * y[5] - k[9] * y[11] - k[12] * y[10]

    J[3, 1] = k[1]
    J[3, 4] = k[17]
    J[3, 16] = k[19]
    J[3, 19] = k[22]
    J[3, 3] = -k[15]

    J[4, 4] = -k[2] * y[2] - k[16] - k[17] - k[23] * y[1]
    J[4, 2] = -k[2] * y[4]
    J[4, 1] = -k[23] * y[4]
    J[4, 3] = k[15]

    J[5, 5] = -k[3] * y[2]
    J[5, 2] = -k[3] * y[5]
    J[5, 7] = 2.0 * k[4] + k[6] * y[6]
    J[5, 6] = k[6] * y[7] + k[20] * y[17]
    J[5, 9] = k[7]
    J[5, 14] = k[13]
    J[5, 17] = k[20] * y[6]

    J[6, 6] = -k[6] * y[7] - k[8] * y[9] - k[14] * y[1] - k[20] * y[17]
    J[6, 7] = -k[6] * y[6]
    J[6, 9] = -k[8] * y[6]
    J[6, 1] = -k[14] * y[6]
    J[6, 17] = -k[20] * y[6]
    J[6, 2] = k[3] * y[5]
    J[6, 5] = k[3] * y[2]
    J[6, 16] = 2.0 * k[18]

    J[7, 7] = -k[4] - k[5] - k[6] * y[6]
    J[7, 6] = -k[6] * y[7]
    J[7, 14] = k[13]

    J[8, 7] = k[4] + k[5] + k[6] * y[6]
    J[8, 6] = k[6] * y[7]
    J[8, 9] = k[7]

    J[9, 9] = -k[7] - k[8] * y[6]
    J[9, 6] = -k[8] * y[9]

    J[10, 10] = -k[12] * y[2]
    J[10, 2] = -k[12] * y[10] + k[9] * y[11]
    J[10, 9] = k[7]
    J[10, 11] = k[9] * y[2]

    J[11, 11] = -k[9] * y[2] - k[10] * y[1]
    J[11, 2] = -k[9] * y[11]
    J[11, 1] = -k[10] * y[11]
    J[11, 9] = k[8] * y[6]
    J[11, 6] = k[8] * y[9]
    J[11, 13] = k[11]

    J[12, 11] = k[9] * y[2]
    J[12, 2] = k[9] * y[11]

    J[13, 13] = -k[11]
    J[13, 11] = k[10] * y[1]
    J[13, 1] = k[10] * y[11]

    J[14, 14] = -k[13]
    J[14, 10] = k[12] * y[2]
    J[14, 2] = k[12] * y[10]

    J[15, 1] = k[14] * y[6]
    J[15, 6] = k[14] * y[1]

    J[16, 16] = -k[18] - k[19]
    J[16, 4] = k[16]

    J[17, 17] = -k[20] * y[6]
    J[17, 6] = -k[20] * y[17]

    J[18, 17] = k[20] * y[6]
    J[18, 6] = k[20] * y[17]

    J[19, 19] = -k[21] - k[22] - k[24] * y[1]
    J[19, 1] = -k[24] * y[19] + k[23] * y[4]
    J[19, 4] = k[23] * y[1]
    J[19, 20] = k[25]

    J[20, 20] = -k[25]
    J[20, 1] = k[24] * y[19]
    J[20, 19] = k[24] * y[1]
    return J[1:, 1:]


def pollu_initial(n, seed):
    """
    The six emitted species (NO, O3, HCHO, CO, ALD, SO2) log-uniform within a factor 2 of their standard
    values, all intermediates and products zero, as in the standard start.
    """
    rng = np.random.default_rng(seed)
    Y = np.zeros((n, 20))
    idx = [1, 3, 6, 7, 8, 16]
    Y[:, idx] = POLLU_Y0[idx] * 2.0 ** rng.uniform(-1.0, 1.0, (n, len(idx)))
    return Y


# --- CHEMAKZO: chemakzo.f (Chemical Akzo Nobel; index-1 DAE, 6 components) -----------------------------
# M y' = f(y) with M = diag(1, 1, 1, 1, 1, 0) (meval, lines 237-241): y6 is algebraic, f(6) = ks*y1*y4 - y6
# (line 155). y = [FLB, CO2, FLBT, ZHU, ZLA, FLB.ZHU] (chemakzo.pdf, sec. 12.3). The source gives no ODE
# form. Because the constraint is explicit in y6, substituting y6 = ks*y1*y4 into f(1..5) is an exact
# reduction (not from the source) to a 5-species ODE: its solution is the DAE's for consistent initial
# values, and the DAE's y6 is recovered as ks*y1*y4. feval: lines 126-155; jeval: lines 188-226; init:
# lines 69-74; t(1): line 43; solut: lines 270-275 (PSIDE on a Cray C90, rtol = atol = 1e-19).
# Time in minutes, concentrations in mol/l. sqrt(y2) requires CO2 >= 0 (feval returns ierr = -1 otherwise).
AKZO_K1, AKZO_K2, AKZO_K3, AKZO_K4 = 18.7, 0.58, 0.09, 0.42
AKZO_KBIG, AKZO_KLA, AKZO_KS, AKZO_PO2, AKZO_HEN = 34.4, 3.3, 115.83, 0.9, 737.0
CHEMAKZO_SPECIES = ("FLB", "CO2", "FLBT", "ZHU", "ZLA")
CHEMAKZO_Y0 = np.array([0.444, 0.00123, 0.0, 0.007, 0.0])
CHEMAKZO_TEND = 180.0
CHEMAKZO_REF6 = np.array([0.1150794920661702e0, 0.1203831471567715e-2, 0.1611562887407974e0,
                          0.3656156421249283e-3, 0.1708010885264404e-1, 0.4873531310307455e-2])  # fmt: skip
CHEMAKZO_REF = CHEMAKZO_REF6[:5]


def chemakzo_rhs(y):
    """f(1..5) of chemakzo.f with y6 = ks*y1*y4 substituted."""
    y1, y2, y3, y4, y5 = y
    y6 = AKZO_KS * y1 * y4
    sq = np.sqrt(y2)
    r1 = AKZO_K1 * y1**4 * sq
    r2 = AKZO_K2 * y3 * y4
    r3 = AKZO_K2 / AKZO_KBIG * y1 * y5
    r4 = AKZO_K3 * y1 * y4**2
    r5 = AKZO_K4 * y6**2 * sq
    fin = AKZO_KLA * (AKZO_PO2 / AKZO_HEN - y2)
    return np.array([
        -2.0 * r1 + r2 - r3 - r4,
        -0.5 * r1 - r4 - 0.5 * r5 + fin,
        r1 - r2 + r3,
        -r2 + r3 - 2.0 * r4,
        r2 - r3 + r5,
    ])  # fmt: skip


def chemakzo_jac(y):
    """
    The DAE's jeval (6 x 6) times the tangent of y6 = ks*y1*y4: J = J6[:5, :5] + J6[:5, 5] (dy6/dy)^T,
    with dy6/dy1 = ks*y4 and dy6/dy4 = ks*y1 (J6's last row, lines 224-225).
    """
    y1, y2, y3, y4, y5 = y
    y6 = AKZO_KS * y1 * y4
    y = np.array([0.0, y1, y2, y3, y4, y5, y6])
    k1, k2, k3, k4, kbig, kla, ks = AKZO_K1, AKZO_K2, AKZO_K3, AKZO_K4, AKZO_KBIG, AKZO_KLA, AKZO_KS
    r11 = 4.0 * k1 * y[1] ** 3 * np.sqrt(y[2])
    r12 = 0.5 * k1 * y[1] ** 4 / np.sqrt(y[2])
    r23 = k2 * y[4]
    r24 = k2 * y[3]
    r31 = (k2 / kbig) * y[5]
    r35 = (k2 / kbig) * y[1]
    r41 = k3 * y[4] ** 2
    r44 = 2.0 * k3 * y[1] * y[4]
    r52 = 0.5 * k4 * y[6] ** 2 / np.sqrt(y[2])
    r56 = 2.0 * k4 * y[6] * np.sqrt(y[2])
    fin2 = -kla
    J = np.zeros((7, 7))
    J[1, 1] = -2.0 * r11 - r31 - r41
    J[1, 2] = -2.0 * r12
    J[1, 3] = r23
    J[1, 4] = r24 - r44
    J[1, 5] = -r35
    J[2, 1] = -0.5 * r11 - r41
    J[2, 2] = -0.5 * r12 - 0.5 * r52 + fin2
    J[2, 4] = -r44
    J[2, 6] = -0.5 * r56
    J[3, 1] = r11 + r31
    J[3, 2] = r12
    J[3, 3] = -r23
    J[3, 4] = -r24
    J[3, 5] = r35
    J[4, 1] = r31 - 2.0 * r41
    J[4, 3] = -r23
    J[4, 4] = -r24 - 2.0 * r44
    J[4, 5] = r35
    J[5, 1] = -r31
    J[5, 2] = r52
    J[5, 3] = r23
    J[5, 4] = r24
    J[5, 5] = -r35
    J[5, 6] = r56
    J[6, 1] = ks * y[4]
    J[6, 4] = ks * y[1]
    J[6, 6] = -1.0
    dy6 = np.zeros(5)
    dy6[0], dy6[3] = J[6, 1], J[6, 4]
    return J[1:6, 1:6] + np.outer(J[1:6, 6], dy6)


def chemakzo_initial(n, seed):
    """
    FLB in [0.3, 0.6] and ZHU in [0.0035, 0.014] mol/l (the standard 0.444 and 0.007 mixed in other
    proportions), CO2 log-uniform in [3e-4, 3e-3] around the standard 0.00123 and the saturation value
    po2/H = 1.22e-3, FLBT and ZLA zero, as in the standard start.
    """
    rng = np.random.default_rng(seed)
    Y = np.zeros((n, 5))
    Y[:, 0] = rng.uniform(0.3, 0.6, n)
    Y[:, 1] = 10.0 ** rng.uniform(np.log10(3e-4), np.log10(3e-3), n)
    Y[:, 3] = 0.007 * 2.0 ** rng.uniform(-1.0, 1.0, n)
    return Y


HIRES = Network("hires", HIRES_SPECIES, hires_rhs, hires_jac, rtol=1e-8, atol=1e-14)
POLLU = Network("pollu", POLLU_SPECIES, pollu_rhs, pollu_jac, rtol=1e-8, atol=1e-24)
CHEMAKZO = Network("chemakzo", CHEMAKZO_SPECIES, chemakzo_rhs, chemakzo_jac, rtol=1e-8, atol=1e-14)
NETWORKS.update({net.name: net for net in (HIRES, POLLU, CHEMAKZO)})


def trajectories(net: Network, Y0: np.ndarray, dt: float, steps: int) -> np.ndarray:
    """(n, steps + 1, components) states of the trajectories from the rows of Y0 in the surrogates' coordinates, every dt."""
    Z = np.empty((len(Y0), steps + 1, len(net.species)))
    Z[:, 0] = net.coords(Y0)
    for k in range(steps):
        Z[:, k + 1] = net.coords_step(Z[:, k], dt)
    return Z


def robertson_initial(n: int, seed: int) -> np.ndarray:
    """Initial abundances: y1 in [0.5, 1], a trace of y2 in [1e-12, 1e-6], y3 the rest (they sum to 1)."""
    rng = np.random.default_rng(seed)
    y1 = rng.uniform(0.5, 1.0, n)
    y2 = 10.0 ** rng.uniform(-12, -6, n)
    return np.stack([y1, y2, 1.0 - y1 - y2], axis=1)


# --- Scalable stiff systems: method-of-lines reaction-diffusion on n grid points of [0, 1] --------------
# Stiff through diffusion (eigenvalues up to about 4 D / h^2) and, for Allen-Cahn, a fast reaction; each
# is contracting in the regime chosen, so a rollout's error is the surrogate's, not the dynamics' chaos.


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


def bruss(n: int) -> Network:
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

    return Network(f"bruss{n}", tuple(f"u{i}" for i in range(n)) + tuple(f"v{i}" for i in range(n)), rhs, jac, atol=1e-12)


def bruss_initial(n: int):
    def initial(m: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.concatenate([BRUSS_A + 0.6 * _smooth_fields(m, n, rng), BRUSS_B / BRUSS_A + 1.0 * _smooth_fields(m, n, rng)], axis=1)
    return initial


AC_D, AC_DELTA = 1.0, 0.05


def allen_cahn(n: int) -> Network:
    """u' = D u_xx + (u - u^3) / delta on n points, Neumann: a fast reaction to +-1 and slow interfaces (linear coordinates)."""
    L = AC_D * _laplacian(n, neumann=True)
    return Network(f"allen_cahn{n}", tuple(f"u{i}" for i in range(n)), lambda y: L @ y + (y - y**3) / AC_DELTA,
                   lambda y: L + np.diag((1.0 - 3.0 * y * y) / AC_DELTA), atol=1e-12, log=False)  # fmt: skip


def allen_cahn_initial(n: int):
    return lambda m, seed: 0.9 * _smooth_fields(m, n, np.random.default_rng(seed))


KPP_D, KPP_R = 1.0, 10.0


def fisher_kpp(n: int) -> Network:
    """u' = D u_xx + r u (1 - u) on n points, Neumann: growth from a trace towards the stable u = 1."""
    L = KPP_D * _laplacian(n, neumann=True)
    return Network(f"fisher_kpp{n}", tuple(f"u{i}" for i in range(n)), lambda y: L @ y + KPP_R * y * (1.0 - y),
                   lambda y: L + np.diag(KPP_R * (1.0 - 2.0 * y)), atol=1e-14)  # fmt: skip


def fisher_kpp_initial(n: int):
    return lambda m, seed: 10.0 ** (-1.5 + 1.5 * _smooth_fields(m, n, np.random.default_rng(seed)))


LADDER = {"bruss": (bruss, bruss_initial, (1, 2, 3, 5, 10, 20, 40)),
          "allen_cahn": (allen_cahn, allen_cahn_initial, (2, 3, 5, 10, 20, 40, 80)),
          "fisher_kpp": (fisher_kpp, fisher_kpp_initial, (2, 3, 5, 10, 20, 40, 80))}  # fmt: skip
NETWORKS.update({net.name: net for make, _, sizes in LADDER.values() for net in map(make, sizes)})
