"""The batched ensemble integrator: accuracy against a tight reference, batch independence, GPU against CPU."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src import IS_CUPY_BACKEND, xp
from src.numerics import integrate_ensemble
from src.problems.lorenz96 import lorenz96
from src.problems.lorenz96_flowmap import lorenz96_flowmap
from tests._helpers import host


@pytest.mark.parametrize("d", [5, 10, 40, 80])
def test_lorenz96_flow_map_matches_dop853_within_a_few_tolerances(d):
    X = np.random.default_rng(d).uniform(-5, 5, (8, d))
    Y = host(lorenz96_flowmap(xp.asarray(X), 1.0, 0.1))
    ref = np.array([
        solve_ivp(lambda t, x: lorenz96(x[None], 1.0)[0], (0, 0.1), x, method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
        for x in X
    ])  # fmt: skip
    assert Y.dtype == np.float64
    assert np.max(np.abs(Y - ref) / (1e-6 + 1e-6 * np.abs(ref))) < 5


@pytest.mark.parametrize("d", [5, 40])
def test_a_trajectory_does_not_depend_on_its_batch(d):
    X = xp.asarray(np.random.default_rng(1).uniform(-5, 5, (64, d)))
    whole = host(integrate_ensemble("lorenz96", X, 0.1, [1.0], 1e-6, 1e-6))
    perm = np.random.default_rng(2).permutation(64)
    for n in (1, 7, 64):
        rows = perm[:n]
        part = host(integrate_ensemble("lorenz96", X[xp.asarray(rows)], 0.1, [1.0], 1e-6, 1e-6))
        np.testing.assert_array_equal(part, whole[rows])


def test_zero_rows_and_the_last_step_landing_on_t1():
    assert host(integrate_ensemble("lorenz96", xp.zeros((0, 5)), 0.1, [1.0], 1e-6, 1e-6)).shape == (0, 5)
    # x = F (every component equal to the forcing) is a fixed point of Lorenz-96: it must come back exactly
    X = xp.full((3, 6), 1.0)
    np.testing.assert_array_equal(host(integrate_ensemble("lorenz96", X, 0.37, [1.0], 1e-6, 1e-6)), np.ones((3, 6)))


@pytest.mark.skipif(not IS_CUPY_BACKEND, reason="compares the CUDA kernel with the Numba integrator")
@pytest.mark.parametrize("d", [5, 32, 33, 80])
def test_gpu_matches_cpu(monkeypatch, d):
    import src.numerics.ode as ode

    X = np.random.default_rng(d).uniform(-5, 5, (50, d))
    gpu = host(ode.integrate_ensemble("lorenz96", xp.asarray(X), 0.1, [1.0], 1e-6, 1e-6))
    monkeypatch.setattr(ode, "IS_CUPY_BACKEND", False)
    monkeypatch.setattr(ode, "xp", np)
    cpu = ode.integrate_ensemble("lorenz96", X, 0.1, [1.0], 1e-6, 1e-6)
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)
