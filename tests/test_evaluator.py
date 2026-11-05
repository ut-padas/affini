"""Tests for Evaluator, the wrapper around f_fn / g_fn / oracle_fn."""

import numpy as np
import pytest

from src.adaptive_clustering_utils import Evaluator


def f(x):
    return np.sum(x**2, axis=1)  # (N,)


def g(x):
    return 2.0 * x  # (N, d)


def oracle(x):
    return f(x), g(x)


X = np.arange(12, dtype=np.float64).reshape(6, 2)


def test_requires_f_or_oracle():
    with pytest.raises(ValueError, match="f_fn or oracle_fn"):
        Evaluator(g_fn=g)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"f_fn": f},
        {"f_fn": f, "g_fn": g},
        {"oracle_fn": oracle},
        {"f_fn": f, "oracle_fn": oracle},
    ],
)
def test_values_without_grad(kwargs):
    F, G = Evaluator(**kwargs)(X)
    assert G is None
    assert F.shape == (6, 1)
    np.testing.assert_array_equal(F[:, 0], f(X))


@pytest.mark.parametrize("kwargs", [{"f_fn": f, "g_fn": g}, {"oracle_fn": oracle}])
def test_values_with_grad(kwargs):
    F, G = Evaluator(**kwargs)(X, need_grad=True)
    assert F.shape == (6, 1) and G.shape == (6, 1, 2)
    np.testing.assert_array_equal(F[:, 0], f(X))
    np.testing.assert_array_equal(G[:, 0, :], g(X))


def test_grad_without_gradient_source_raises():
    ev = Evaluator(f_fn=f)
    assert not ev.has_grad
    with pytest.raises(ValueError, match="Gradients requested"):
        ev(X, need_grad=True)


def test_prefers_f_fn_when_no_grad_needed():
    calls = []
    ev = Evaluator(
        f_fn=lambda x: calls.append("f") or f(x),
        oracle_fn=lambda x: calls.append("oracle") or oracle(x),
    )
    ev(X)
    ev(X, need_grad=True)
    assert calls == ["f", "oracle"]


def test_counts_points():
    ev = Evaluator(oracle_fn=oracle)
    ev(X)
    ev(X[:2], need_grad=True)
    assert ev.n_evals == 8
