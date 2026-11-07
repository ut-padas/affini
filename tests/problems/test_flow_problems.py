"""Tests for the Lorenz-96 and Van der Pol vector fields and flow maps."""

import numpy as np
import pytest

from src import _to_host
from src.problems.lorenz96 import lorenz96, lorenz96_grad
from src.problems.lorenz96_flowmap import lorenz96_flowmap, lorenz96_flowmap_grad
from src.problems.vanderpol import vanderpol, vanderpol_grad
from src.problems.vanderpol_flowmap import vanderpol_flowmap, vanderpol_flowmap_grad

rng = np.random.default_rng(0)


def _fd_jacobian(fn, X, h=1e-6):
    """Central-difference Jacobian (N, out, d) of a batched fn: (N, d) -> (N, out)."""
    X = np.asarray(X, dtype=np.float64)
    cols = []
    for j in range(X.shape[1]):
        E = np.zeros_like(X)
        E[:, j] = h
        cols.append((_to_host(fn(X + E)) - _to_host(fn(X - E))) / (2 * h))
    return np.stack(cols, axis=-1)


def test_lorenz96_matches_its_definition():
    X = rng.uniform(-5, 5, (7, 6))
    d = X.shape[1]
    expected = np.empty_like(X)
    for i in range(d):  # dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
        expected[:, i] = (X[:, (i + 1) % d] - X[:, i - 2]) * X[:, i - 1] - X[:, i] + 8.0
    np.testing.assert_allclose(lorenz96(X, forcing=8.0), expected, rtol=1e-14)


def test_lorenz96_grad_matches_finite_differences():
    X = rng.uniform(-5, 5, (5, 6))
    np.testing.assert_allclose(
        lorenz96_grad(X), _fd_jacobian(lorenz96, X), rtol=1e-6, atol=1e-6
    )


def test_vanderpol_grad_matches_finite_differences():
    X = rng.uniform(-3, 3, (5, 2))
    f = lambda Z: vanderpol(Z, mu=2.0)  # noqa: E731
    np.testing.assert_allclose(
        vanderpol_grad(X, mu=2.0), _fd_jacobian(f, X), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    "flowmap,rhs,kwargs,d",
    [
        (lorenz96_flowmap, lorenz96, {"forcing": 1.0}, 5),
        (vanderpol_flowmap, vanderpol, {"mu": 2.0}, 2),
    ],
)
def test_short_flow_map_step_follows_the_vector_field(flowmap, rhs, kwargs, d):
    X = rng.uniform(-2, 2, (4, d))
    tf = 1e-4
    Y = _to_host(flowmap(X, tf=tf, **kwargs))
    # x(tf) = x0 + tf f(x0) + O(tf^2)
    np.testing.assert_allclose((Y - X) / tf, rhs(X, **kwargs), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "flowmap,grad,kwargs,d",
    [
        (lorenz96_flowmap, lorenz96_flowmap_grad, {"forcing": 1.0, "tf": 0.1}, 4),
        (vanderpol_flowmap, vanderpol_flowmap_grad, {"mu": 2.0, "tf": 0.5}, 2),
    ],
)
def test_flow_map_grad_matches_finite_differences(flowmap, grad, kwargs, d):
    X = rng.uniform(-2, 2, (3, d))
    f = lambda Z: flowmap(Z, **kwargs)  # noqa: E731
    np.testing.assert_allclose(
        _to_host(grad(X, **kwargs)), _fd_jacobian(f, X, h=1e-5), rtol=1e-3, atol=1e-4
    )


def test_jax_lorenz96_flow_map_agrees_with_scipy():
    jax = pytest.importorskip("jax")
    pytest.importorskip("diffrax")
    from src.problems.lorenz96_flowmap_jax import lorenz96_flowmap_jax_batched

    X = rng.uniform(-5, 5, (6, 5))
    Y_jax = np.asarray(lorenz96_flowmap_jax_batched(jax.numpy.asarray(X), 1.0, 0.1))
    assert Y_jax.shape == (6, 1, 5)  # (N, saved times, d): only tf is saved
    # The JAX solve uses rtol = atol = 1e-3.
    np.testing.assert_allclose(
        Y_jax[:, 0, :], _to_host(lorenz96_flowmap(X, 1.0, 0.1)), atol=1e-2
    )
