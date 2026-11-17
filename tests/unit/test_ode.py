"""
The batched ensemble integrator: each trajectory is SciPy's RK45 solve of that trajectory alone (the
same pair and step control), in every regime below; its error against a tight reference therefore
equals RK45's; and batch independence, failure handling and GPU against CPU.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src import IS_CUPY_BACKEND, xp
from src.numerics import integrate_ensemble
from src.problems.lorenz96 import lorenz96
from src.problems.lorenz96_flowmap import lorenz96_flowmap
from tests._helpers import host


def _scipy(X, F, tf, method, tol):
    def rhs(t, x):
        return lorenz96(x[None], F)[0]

    return np.array([solve_ivp(rhs, (0, tf), x, method=method, rtol=tol, atol=tol).y[:, -1] for x in X])


# (state half-width, forcing, horizon, tolerance) where the flow is well conditioned, so two
# implementations of one algorithm agree to rounding: the study's map (5, 1, 0.1, 1e-6); forcing 8
# over a horizon of 1; states ten and a hundred times the study's (Jacobian eigenvalues ~ |x|, hundreds
# of steps); a long horizon at forcing 1; tolerances from loose to near the float64 limit; a tiny state.
REGIMES = [(5, 1, 0.1, 1e-6), (5, 8, 1.0, 1e-6), (50, 8, 0.1, 1e-6), (500, 8, 1e-3, 1e-6),
           (5, 1, 10.0, 1e-6), (5, 8, 1.0, 1e-3), (5, 8, 1.0, 1e-12), (0.01, 1, 1e-3, 1e-6)]  # fmt: skip


@pytest.mark.parametrize("d", [4, 5, 33, 80])
@pytest.mark.parametrize("regime", REGIMES)
def test_each_trajectory_is_scipy_rk45_alone(d, regime):
    scale, F, tf, tol = regime
    X = np.random.default_rng(d).uniform(-scale, scale, (6, d))
    Y = host(integrate_ensemble("lorenz96", xp.asarray(X), tf, [F], tol, tol))
    ref = _scipy(X, F, tf, "RK45", tol)
    assert Y.dtype == np.float64
    # same steps, so only rounding differs (fused multiply-adds on the GPU, the norm's summation order);
    # the bound is far below the tolerance and far below either solver's own error
    np.testing.assert_allclose(Y, ref, rtol=1e-9, atol=1e-9 * scale)


@pytest.mark.parametrize("d", [5, 80])
def test_the_study_map_is_as_accurate_as_rk45(d):
    X = np.random.default_rng(d).uniform(-5, 5, (8, d))
    Y = host(lorenz96_flowmap(xp.asarray(X), 1.0, 0.1))
    truth = _scipy(X, 1.0, 0.1, "DOP853", 1e-13)
    rk45 = _scipy(X, 1.0, 0.1, "RK45", 1e-6)
    err, err_rk45 = np.abs(Y - truth).max(1), np.abs(rk45 - truth).max(1)
    np.testing.assert_allclose(err, err_rk45, rtol=1e-6)
    assert err.max() < 1e-5  # global error at rtol = atol = 1e-6 over the study's horizon


@pytest.mark.parametrize("d", [5, 40])
def test_in_the_chaotic_regime_the_error_is_rk45s(d):
    # Forcing 8 over a horizon of 5 amplifies rounding by orders of magnitude, so states need not agree
    # to rounding with RK45's; the error against a tight reference must still be RK45's own.
    X = np.random.default_rng(d).uniform(-5, 5, (8, d))
    Y = host(integrate_ensemble("lorenz96", xp.asarray(X), 5.0, [8.0], 1e-6, 1e-6))
    truth = _scipy(X, 8.0, 5.0, "DOP853", 1e-13)
    err, err_rk45 = np.abs(Y - truth).max(1), np.abs(_scipy(X, 8.0, 5.0, "RK45", 1e-6) - truth).max(1)
    np.testing.assert_allclose(err, err_rk45, rtol=1e-2)


@pytest.mark.parametrize("d", [5, 40])
def test_a_trajectory_does_not_depend_on_its_batch(d):
    X = xp.asarray(np.random.default_rng(1).uniform(-5, 5, (64, d)))
    whole = host(integrate_ensemble("lorenz96", X, 0.1, [1.0], 1e-6, 1e-6))
    perm = np.random.default_rng(2).permutation(64)
    for n in (1, 7, 64):
        rows = perm[:n]
        part = host(integrate_ensemble("lorenz96", X[xp.asarray(rows)], 0.1, [1.0], 1e-6, 1e-6))
        np.testing.assert_array_equal(part, whole[rows])


def test_empty_batches_fixed_points_and_a_zero_horizon():
    assert host(integrate_ensemble("lorenz96", xp.zeros((0, 5)), 0.1, [1.0], 1e-6, 1e-6)).shape == (0, 5)
    # x = F (every component equal to the forcing) is a fixed point of Lorenz-96: it must come back exactly
    X = xp.full((3, 6), 1.0)
    np.testing.assert_array_equal(host(integrate_ensemble("lorenz96", X, 0.37, [1.0], 1e-6, 1e-6)), np.ones((3, 6)))
    X = np.random.default_rng(0).uniform(-5, 5, (4, 5))
    np.testing.assert_array_equal(host(integrate_ensemble("lorenz96", xp.asarray(X), 0.0, [1.0], 1e-6, 1e-6)), X)


def test_a_failing_trajectory_is_nan_and_leaves_the_others_alone():
    X = np.random.default_rng(3).uniform(-5, 5, (5, 6))
    X[1, 2], X[3, 0] = np.nan, np.inf
    Y = host(integrate_ensemble("lorenz96", xp.asarray(X), 0.1, [1.0], 1e-6, 1e-6))
    assert np.isnan(Y[[1, 3]]).all()
    ok = [0, 2, 4]
    np.testing.assert_array_equal(Y[ok], host(integrate_ensemble("lorenz96", xp.asarray(X[ok]), 0.1, [1.0], 1e-6, 1e-6)))


@pytest.mark.skipif(not IS_CUPY_BACKEND, reason="compares the CUDA kernel with the Numba integrator")
@pytest.mark.parametrize("d", [5, 32, 33, 80])
@pytest.mark.parametrize("regime", REGIMES)
def test_gpu_matches_cpu(monkeypatch, d, regime):
    import src.numerics.ode as ode

    scale, F, tf, tol = regime
    X = np.random.default_rng(d).uniform(-scale, scale, (50, d))
    gpu = host(ode.integrate_ensemble("lorenz96", xp.asarray(X), tf, [F], tol, tol))
    monkeypatch.setattr(ode, "IS_CUPY_BACKEND", False)
    monkeypatch.setattr(ode, "xp", np)
    cpu = ode.integrate_ensemble("lorenz96", X, tf, [F], tol, tol)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-10, atol=1e-10 * scale)
