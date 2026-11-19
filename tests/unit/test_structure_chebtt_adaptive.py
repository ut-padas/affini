"""Adaptive ChebTT (src/schemes/chebtt_adaptive.py): smooth functions unchanged, interfaces and fronts modelled."""

import numpy as np

from src.schemes import chebtt, chebtt_adaptive


def test_a_smooth_function_gets_exactly_the_chebtt_model():
    d = 4
    lo, hi = -np.ones(d), np.ones(d)
    f = lambda X: np.exp(-np.sum(X**2, axis=1))  # noqa: E731
    m = chebtt_adaptive.build(f, lo, hi, 1e-3)
    plain = chebtt.build(f, lo, hi, 1e-3)
    X = np.random.default_rng(0).uniform(lo, hi, (2000, d))
    assert m.interface is None and m.evals <= plain.evals
    assert np.array_equal(m.predict(X), plain.predict(X))


def test_a_jump_across_an_oblique_plane_becomes_a_profile_term():
    d = 3
    q = np.array([0.6, -0.48, 0.64])
    lo, hi = -np.ones(d), np.ones(d)
    f = lambda X: np.sin(2.0 * (X @ q)) + (X @ q > 0.3)  # noqa: E731
    m = chebtt_adaptive.build(f, lo, hi, 0.01)
    assert m.interface is not None and m.interface.kind == "plane" and m.interface.profile is not None
    X = np.random.default_rng(1).uniform(lo, hi, (5000, d))
    X = X[np.abs(X @ q - 0.3) > 1e-6]  # outside the interface's band
    assert np.max(np.abs(m.predict(X) - f(X))) <= 0.01


def test_a_jump_whose_height_varies_gets_jump_fields():
    d = 2
    q = np.array([0.8, 0.6])
    lo, hi = -np.ones(d), np.ones(d)
    f = lambda X: np.sin(2.0 * (X @ q)) + (1.0 + 0.3 * X[:, 0]) * (X @ q > 0.3)  # noqa: E731
    m = chebtt_adaptive.build(f, lo, hi, 0.01)
    assert m.interface is not None and m.interface.profile is None and m.interface.degree >= 1
    X = np.random.default_rng(2).uniform(lo, hi, (5000, d))
    X = X[np.abs(X @ q - 0.3) > 1e-6]
    assert np.max(np.abs(m.predict(X) - f(X))) <= 0.01


def test_a_sharp_smooth_front_is_found_from_level_crossings():
    d = 2
    lo, hi = -2 * np.ones(d), 2 * np.ones(d)
    f = lambda X: np.tanh((np.linalg.norm(X, axis=1) - 1.0) / 0.05)  # noqa: E731
    m = chebtt_adaptive.build(f, lo, hi, 0.01)
    assert m.interface is not None and m.kind.startswith("front")
    X = np.random.default_rng(3).uniform(lo, hi, (5000, d))
    assert np.max(np.abs(m.predict(X) - f(X))) <= 0.01
