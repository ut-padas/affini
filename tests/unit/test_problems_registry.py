"""The problem registries: every function's shape and domain, vector scaling, and the step maps' regimes."""

import numpy as np
import pytest

from src.problems import FUNCTIONS, VECTOR_FUNCTIONS
from src.problems.ode import REGIMES, STEP_MAPS, SYSTEMS

FAST = [n for n in FUNCTIONS if not n.startswith(("lorenz96", "lotka_volterra"))]


@pytest.mark.parametrize("name", FAST)
@pytest.mark.parametrize("d", [2, 3])
def test_every_function_returns_one_row_per_point_inside_its_domain(name, d):
    p = FUNCTIONS[name]
    lo, hi = p.domain(d)
    X = np.random.default_rng(0).uniform(lo, hi, (7, d))
    V = p.values(X)
    assert V.shape == ((7, V.shape[1]) if p.vector else (7,))
    assert np.all(np.isfinite(V))


def test_vector_components_are_scaled_to_unit_maximum_on_the_test_set():
    p = VECTOR_FUNCTIONS["wave_pair"]
    V = p.values(p.test_points(3))
    assert np.allclose(np.max(np.abs(V), axis=0), 1.0)


def test_scalar_problems_test_on_a_grid_in_2d_and_fixed_uniform_points_above():
    p = FUNCTIONS["rosenbrock"]
    assert p.test_points(2).shape == (300 * 300, 2)
    assert np.array_equal(p.test_points(5), p.test_points(5))


def test_every_step_map_names_a_registered_system_and_a_known_regime():
    for name, p in STEP_MAPS.items():
        assert p.regime in REGIMES, name
        assert SYSTEMS[p.system.name] is p.system, name
    assert STEP_MAPS["lorenz96_d10"].regime == "nonstiff"
