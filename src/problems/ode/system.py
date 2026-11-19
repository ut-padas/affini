"""
ODE systems as surrogate oracles: the time-step map of a system and its visited states.

A system is ``dy/dt = F(y)``. The oracle a simulation would call is the step map ``Phi_dt(y)``,
integrated one state at a time so that no state's result depends on the others in its batch, by the
system's own integrator (``System.method``): an implicit Runge-Kutta method (SciPy's Radau with the
analytic Jacobian) for a stiff system, an explicit one (Dormand-Prince, RK45) for a non-stiff one,
where an implicit method only pays for linear solves it does not need. A system of abundances is seen
by the surrogates in log10 (``System.log``): abundances span many orders of magnitude, a relative error
is the meaningful one, and an additive error of e in log10 is a factor 10^e.

The states a simulation visits lie near the system's attractor (for a stiff network, its slow manifold:
after a fast transient the fast modes have relaxed, and the step map's Jacobian has low numerical rank;
Robertson: singular values about 1.7, 1.0 and 1e-10 for every step dt >= 0.1). So surrogates are
trained and judged on states of trajectories, ``trajectories``, not on a box: held-out trajectories
give the test states.
"""

import atexit
import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

FLOOR = 1e-30  # abundances below it are clipped before the log10, so the map is defined everywhere
IMPLICIT = ("Radau", "BDF", "LSODA")  # the solve_ivp methods that use a Jacobian
WORKERS = int(os.environ.get("SURROGATE_ODE_WORKERS", "1"))  # processes per System.step batch
SYSTEMS: dict = {}  # name: System; the pool's workers look systems up here (filled on import of the package)


@dataclass(frozen=True)
class System:
    """
    A system's right-hand side and Jacobian (functions of one state) and the names of its components. ``log``:
    the surrogates see log10 of the state (abundances and other positive quantities, where a relative
    error is the meaningful one); otherwise the state itself. ``method``: the ``solve_ivp`` integrator
    of its step map; the Jacobian is passed only to an implicit one (an explicit one ignores it with a
    warning), and is still used by the dynamics diagnostics. ``params``: the parameter values that set the
    system's regime (Lorenz-96's forcing), printed beside its name in the plots; empty when none.
    ``held``: how many trailing components are inputs only (their right-hand side is 0: a temperature or an
    environment held over the step); they are not scored and a surrogate predicts no increment for them.
    """

    name: str
    components: tuple[str, ...]
    rhs: object
    jac: object
    rtol: float = 1e-8
    atol: float = 1e-20
    log: bool = True
    method: str = "Radau"
    params: str = ""
    held: int = 0

    def solve(self, y: np.ndarray, dt: float):
        """``solve_ivp`` of one state over dt with the system's integrator and tolerances."""
        jac = {"jac": lambda t, v: self.jac(v)} if self.method in IMPLICIT else {}
        return solve_ivp(lambda t, v: self.rhs(v), (0.0, dt), y, method=self.method, rtol=self.rtol, atol=self.atol, **jac)

    def step(self, Y: np.ndarray, dt: float) -> np.ndarray:
        """
        ``Phi_dt`` of every row of Y (in the system's own variables), each integrated on its own, so a row's result never
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
            s = self.solve(y, dt)
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
    """``System.step`` of the registered system ``name`` for the rows of Y, serially (a picklable worker)."""
    system = SYSTEMS[name]
    out = np.empty_like(Y)
    for i, y in enumerate(Y):
        s = system.solve(y, dt)
        out[i] = s.y[:, -1] if s.success else np.nan
    return out


def trajectories(system: System, Y0: np.ndarray, dt: float, steps: int, held_path: np.ndarray | None = None) -> np.ndarray:
    """
    (n, steps + 1, components) states of the trajectories from the rows of Y0 in the surrogates' coordinates,
    every dt. ``held_path`` (n, steps + 1, held), in the system's own variables, sets the held components
    before each step (an environment that changes between steps, as along a hydrodynamic tracer).
    """
    Z = np.empty((len(Y0), steps + 1, len(system.components)))
    Z[:, 0] = system.coords(Y0)
    h = system.held
    for k in range(steps):
        if held_path is not None:
            Z[:, k, -h:] = system.coords(held_path[:, k])
        Z[:, k + 1] = system.coords_step(Z[:, k], dt)
    if held_path is not None:
        Z[:, steps, -h:] = system.coords(held_path[:, steps])
    return Z
