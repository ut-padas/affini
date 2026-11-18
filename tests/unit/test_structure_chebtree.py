"""ChebTree (src/structure/chebtree.py): its error estimate, check points, growth and chopping."""

import numpy as np
import pytest

from src.structure import chebtree
from src.structure.problems import PROBLEMS


def _worst_ratio(problem: str, tol: float) -> float:
    p = PROBLEMS[problem]
    lo, hi = p.domain(2)
    X = p.test_points(2)
    fX = p.values(X)
    atol = tol * np.max(np.abs(fX))
    tree = chebtree.build(p.values, lo, hi, atol)
    return float(np.max(np.abs(tree.predict(X) - fX)) / atol)


def test_a_polynomial_is_one_exact_box():
    p = PROBLEMS["rosenbrock"]
    lo, hi = p.domain(2)
    tree = chebtree.build(p.values, lo, hi, 1e-3)
    X = np.random.default_rng(0).uniform(lo, hi, (1000, 2))
    assert len(tree.leaves) == 1
    assert np.max(np.abs(tree.predict(X) - p.values(X))) < 1e-10


@pytest.mark.parametrize(("problem", "tol"), [("rastrigin", 0.1), ("ackley", 0.1), ("michalewicz", 0.1),
                                              ("genz_continuous", 0.1), ("genz_continuous", 0.01)])  # fmt: skip
def test_aliased_and_kinked_boxes_are_not_accepted(problem, tol):
    assert _worst_ratio(problem, tol) <= 1.0


def test_a_kink_decays_algebraically_and_an_analytic_function_exponentially():
    t = chebtree.cheb_points(16)
    for g, exponential in ((np.abs(t - 0.3), False), (np.exp(np.sin(2 * t)), True)):
        a = np.abs(chebtree.cheb_coefficients(g))[:, None]
        assert chebtree.axis_decay(a, 0, 16.0).exponential == exponential


def test_growth_evaluates_every_point_once_and_runs_are_deterministic():
    seen = []

    def f(X):
        seen.append(X)
        return np.cos(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + np.exp(X[:, 0] * X[:, 1] / 4)

    a = chebtree.build(f, [-3.0, -3.0], [3.0, 3.0], 1e-6)
    pts = np.concatenate(seen)
    assert len(np.unique(pts, axis=0)) == len(pts) == a.evals
    b = chebtree.build(f, [-3.0, -3.0], [3.0, 3.0], 1e-6)
    assert a.evals == b.evals and all(np.array_equal(x[2], y[2]) for x, y in zip(a.leaves, b.leaves, strict=True))


def test_chopping_changes_the_interpolant_by_at_most_its_budget():
    c = np.random.default_rng(1).normal(size=(9, 7)) * np.exp(-np.add.outer(np.arange(9), np.arange(7)))
    kept = chebtree.chop(c, 1e-3)
    assert kept.size < c.size
    lo, hi = np.zeros(2), np.ones(2)
    X = np.random.default_rng(2).uniform(lo, hi, (2000, 2))
    change = np.abs(chebtree._interpolate(c, lo, hi, X) - chebtree._interpolate(kept, lo, hi, X))
    assert change.max() <= 1e-3
