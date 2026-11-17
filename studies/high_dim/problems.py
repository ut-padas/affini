"""
Test functions of the high-dimensional scaling study, each defined for any dimension d on
the box [-1, 1]^d, and the fixed test sets that every method is scored on.

The four functions separate implementation limits from approximation limits:

- ``affine``: 1 + a^T x with a fixed random a (|a| = 1). Every leaf model represents it
  exactly, so one leaf should always suffice; whatever tolerance a method cannot reach here
  is a floor of the implementation (ridge bias, rounding, sampling), not of approximation.
- ``quadratic``: ``src.problems.quadratic`` (10 + sum x + 0.1 |x|^2), an isotropic Hessian.
  Exact for quadratic leaves; for affine leaves every direction is equally curved, the
  worst case for adaptive partitioning.
- ``aniso_quadratic``: ``src.problems.anisotropic_quadratic`` (1 + x^T H x, rotated, the
  eigenvalues of H decaying as i^-3): low effective dimension along oblique directions,
  which Hessian-informed oblique splits can exploit.
- ``rosenbrock``: ``src.problems.rosenbrock`` (a = 100), a generic coupled quartic.

Tolerances are given relative to the function's spread: ``atol = tol * scale`` with
``scale`` the standard deviation of f over the problem's test set, so one ``tol`` means a
comparable accuracy at every d.
"""

from dataclasses import dataclass

import numpy as np

from src.problems.anisotropic_quadratic import aniso_quadratic
from src.problems.quadratic import quadratic
from src.problems.rosenbrock import rosenbrock

LOWER, UPPER = -1.0, 1.0
TEST_SEED = 12345  # the test set depends only on d, so every method at one d sees the same points


def _affine_slope(d: int) -> np.ndarray:
    a = np.random.default_rng(1234 + d).standard_normal(d)
    return a / np.linalg.norm(a)


def affine(X):
    """1 + a^T x with a unit, fixed random slope (one per dimension)."""
    import array_api_compat

    xp = array_api_compat.array_namespace(X)
    a = xp.asarray(_affine_slope(int(X.shape[1])))
    return (1.0 + X @ a)[:, None]


@dataclass(frozen=True)
class Problem:
    """A study problem: the function, which leaf models represent it exactly, and a note."""

    name: str
    f: object
    exact_for: tuple[str, ...]  # leaf models with zero approximation error ("affine", "quadratic")
    note: str


PROBLEMS = {
    "affine": Problem("affine", affine, ("affine", "quadratic"), "exact for every leaf model: implementation floor"),
    "quadratic": Problem("quadratic", quadratic, ("quadratic",), "isotropic Hessian 0.2 I"),
    "aniso_quadratic": Problem("aniso_quadratic", aniso_quadratic, ("quadratic",),
                               "rotated, eigenvalues 2 i^-3: low effective dimension"),  # fmt: skip
    "rosenbrock": Problem("rosenbrock", rosenbrock, (), "generic coupled quartic (a = 100)"),
}


def domain(d: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full(d, LOWER), np.full(d, UPPER)


def uniform_points(n: int, d: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(LOWER, UPPER, (n, d))


def scoring_set(problem: Problem, d: int, n: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Uniform test points, f there (host arrays), and the scale (std of f) tolerances are relative to."""
    X = uniform_points(n, d, TEST_SEED + d)
    F = np.asarray(problem.f(X), dtype=np.float64)
    return X, F, float(np.std(F))
