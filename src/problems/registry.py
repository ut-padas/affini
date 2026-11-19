"""
The registered test functions (``FUNCTIONS``; the vector-valued ones also in ``VECTOR_FUNCTIONS``): each a
``FunctionProblem`` with its box domain, defined for any d, and its fixed test set.

Every function takes (N, d) points on the active backend and returns (N, 1) values, or (N, m) for a vector
problem. A vector problem's components are divided by their max |f_c| on the test set (``scale``), so one
tolerance means the same relative accuracy for every component. The test set is a 300 x 300 grid in 2-D (for
maps) and 50,000 fixed uniform points above.
"""

from dataclasses import dataclass

import numpy as np

from src import to_host, xp
from src.problems.anisotropic_gaussian import aniso_gaussian
from src.problems.genz import GENZ_DIFFICULTY, genz, genz_discontinuous, genz_triple
from src.problems.interfaces import ball_jump, front, oblique_jump
from src.problems.lotka_volterra import lotka_volterra, lotka_volterra_state
from src.problems.optimisation_tests import (
    ackley,
    griewank,
    michalewicz,
    styblinski_tang,
)
from src.problems.rastrigin import rastrigin
from src.problems.rosenbrock import rosenbrock
from src.problems.vector_fields import (
    rosenbrock_gradient,
    shared_latent,
    swirl,
    wave_pair,
)
from src.problems.wave import wave

GRID = 300
N_TEST = 50_000
_SCALES: dict = {}  # (problem, d) -> per-component max |f| of a vector problem on its test set


def _column(g):
    return lambda x: xp.reshape(g(x), (x.shape[0], -1))


def lorenz96(forcing: float, tf: float, state: bool = False):
    """
    The first state (or with ``state``, every state) of the Lorenz-96 flow map at ``tf`` (the library's
    per-trajectory DOPRI5, rtol = atol = 1e-6).
    """
    from src.problems.lorenz96_flowmap import lorenz96_flowmap

    return (lambda x: lorenz96_flowmap(x, forcing, tf)) if state else (lambda x: lorenz96_flowmap(x, forcing, tf)[:, :1])


@dataclass(frozen=True)
class FunctionProblem:
    name: str
    lo: float
    hi: float
    f: object
    vector: bool = False  # f returns (N, m): its components are scaled to max |f_c| = 1 on the test set

    def domain(self, d: int) -> tuple[np.ndarray, np.ndarray]:
        return np.full(d, self.lo), np.full(d, self.hi)

    def test_points(self, d: int) -> np.ndarray:
        lo, hi = self.domain(d)
        if d == 2:
            g = (np.arange(GRID) + 0.5) / GRID
            return np.stack(np.meshgrid(lo[0] + g * (hi[0] - lo[0]), lo[1] + g * (hi[1] - lo[1])), axis=-1).reshape(-1, 2)
        return np.random.default_rng(20_261_117).uniform(lo, hi, (N_TEST, d))

    def scale(self, d: int) -> np.ndarray:
        """
        A vector problem's per-component max |f_c| on the test set (1 for a scalar problem), cached. The
        components are divided by it, so that one tolerance is the same relative accuracy for every
        component and the error metric, max over points and components of |error_c| / max |f_c|, does
        not let a component of large range hide the errors of a small one.
        """
        if not self.vector:
            return np.ones(1)
        key = (self.name, d)
        if key not in _SCALES:
            V = np.asarray(to_host(self.f(xp.asarray(self.test_points(d)))), dtype=np.float64)
            _SCALES[key] = np.maximum(np.max(np.abs(V), axis=0), 1e-300)
        return _SCALES[key]

    def oracle(self, d: int):
        """The function the methods train on: f, or for a vector problem f with its components scaled."""
        if not self.vector:
            return self.f
        s = xp.asarray(self.scale(d))
        return lambda x: self.f(x) / s

    def values(self, X: np.ndarray) -> np.ndarray:
        """(N,) for a scalar problem, (N, m) scaled components for a vector one."""
        V = np.asarray(to_host(self.oracle(int(X.shape[1]))(xp.asarray(X))), dtype=np.float64)
        return V.reshape(len(X), -1) if self.vector else V.ravel()


FUNCTIONS = {
    "rosenbrock": FunctionProblem("rosenbrock", -2.0, 2.0, lambda x: rosenbrock(x, a=1.0)),
    "gaussian": FunctionProblem("gaussian", -3.0, 3.0, _column(aniso_gaussian)),
    "wave": FunctionProblem("wave", -5.0, 5.0, _column(wave)),
    "front": FunctionProblem("front", -2.0, 2.0, front),
    "ackley": FunctionProblem("ackley", -5.0, 5.0, ackley),
    "rastrigin": FunctionProblem("rastrigin", -5.12, 5.12, rastrigin),
    "griewank": FunctionProblem("griewank", -10.0, 10.0, griewank),
    "styblinski_tang": FunctionProblem("styblinski_tang", -5.0, 5.0, styblinski_tang),
    "michalewicz": FunctionProblem("michalewicz", 0.0, np.pi, michalewicz),
    **{f"genz_{k}": FunctionProblem(f"genz_{k}", 0.0, 1.0, genz(k)) for k in GENZ_DIFFICULTY},
    # discontinuous
    "genz_discontinuous": FunctionProblem("genz_discontinuous", 0.0, 1.0, genz_discontinuous),
    "oblique_jump": FunctionProblem("oblique_jump", -1.0, 1.0, oblique_jump),
    "ball_jump": FunctionProblem("ball_jump", -1.0, 1.0, ball_jump),
    # ODE flow maps
    "lorenz96": FunctionProblem("lorenz96", -5.0, 5.0, lorenz96(1.0, 0.1)),
    "lorenz96_chaotic": FunctionProblem("lorenz96_chaotic", -5.0, 5.0, lorenz96(8.0, 0.5)),
    "lotka_volterra": FunctionProblem("lotka_volterra", 0.1, 1.0, lotka_volterra),
}
VECTOR_FUNCTIONS = {
    "rosenbrock_gradient": FunctionProblem("rosenbrock_gradient", -2.0, 2.0, rosenbrock_gradient, vector=True),
    "genz_triple": FunctionProblem("genz_triple", 0.0, 1.0, genz_triple, vector=True),
    "lotka_volterra_state": FunctionProblem("lotka_volterra_state", 0.1, 1.0, lotka_volterra_state, vector=True),
    "swirl": FunctionProblem("swirl", -2.0, 2.0, swirl, vector=True),
    "wave_pair": FunctionProblem("wave_pair", -3.0, 3.0, wave_pair, vector=True),
    "shared_latent": FunctionProblem("shared_latent", -1.0, 1.0, shared_latent, vector=True),
    "lorenz96_state": FunctionProblem("lorenz96_state", -5.0, 5.0, lorenz96(1.0, 0.1, state=True), vector=True),
}
FUNCTIONS.update(VECTOR_FUNCTIONS)
