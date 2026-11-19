"""The hub expansion: discovered directions, the gradient path when the oracle supplies a Jacobian, the finite-difference path otherwise."""

import numpy as np

from src.schemes import hub

D = 8
LO, HI = -np.ones(D), np.ones(D)
A = np.linspace(1.0, 0.3, D) / np.linalg.norm(np.linspace(1.0, 0.3, D))


def ridge(X):
    return np.sin(3.0 * X @ A) + 0.05 * X[:, 0] ** 2


def ridge_jac(X):
    return 3.0 * np.cos(3.0 * X @ A)[:, None] * A[None, :] + np.outer(0.1 * X[:, 0], np.eye(D)[0])


def test_ridge_function_found_from_values_alone():
    model = hub.build(ridge, LO, HI, 0.01, n0=1000, max_evals=20_000)
    X = np.random.default_rng(5).uniform(LO, HI, (2000, D))
    assert np.max(np.abs(model.predict(X) - ridge(X))) < 0.01
    assert model.gradients == 0
    assert model.groups[0].r <= 2
    assert abs(abs(model.groups[0].V[:, 0] @ A) - 1.0) < 1e-3  # the steep direction is the ridge's


def test_oracle_gradients_replace_finite_differences_and_cut_evaluations():
    calls = []

    def counted(X):
        calls.append(len(X))
        return ridge(X)

    model = hub.build(counted, LO, HI, 0.01, jac=ridge_jac, n0=1000, max_evals=20_000)
    values_only = hub.build(ridge, LO, HI, 0.01, n0=1000, max_evals=20_000)
    assert model.gradients > 0
    assert sum(calls) == model.evals  # no finite-difference stencils: every evaluation is a training sample
    assert model.evals < values_only.evals
    X = np.random.default_rng(5).uniform(LO, HI, (2000, D))
    assert np.max(np.abs(model.predict(X) - ridge(X))) < 0.01


def test_outputs_share_a_hub_that_holds_their_directions_and_split_when_it_cannot():
    B = np.eye(D)[1]

    def f(X):
        return np.stack([np.sin(3.0 * X @ A), np.tanh(2.0 * X @ B)], axis=1)

    X = np.random.default_rng(6).uniform(LO, HI, (1000, D))
    pooled = hub.build(f, LO, HI, 0.02, n0=1000, max_evals=20_000)
    assert [(g.outputs, g.r) for g in pooled.groups] == [([0, 1], 2)]  # one rotation, one factorisation
    split = hub.build(f, LO, HI, 0.02, n0=1000, max_evals=20_000, r_max=1)
    assert sorted(g.outputs for g in split.groups) == [[0], [1]]
    for model in (pooled, split):
        assert model.predict(X).shape == (1000, 2)
        assert np.max(np.abs(model.predict(X) - f(X))) < 0.02


def test_design_derivatives_match_finite_differences():
    rng = np.random.default_rng(0)
    d, r = 7, 2
    lo, hi = -np.ones(d), 2 * np.ones(d)
    V = np.linalg.qr(rng.normal(size=(d, d)))[0]
    s0 = 0.5 * (hi - lo)
    X = rng.uniform(lo + 0.3, hi - 0.3, (30, d))
    for weak in (False, True):
        cols = hub._columns(d, r, 5, weak)
        A_, dA = hub._design(X, lo, hi, V, r, cols, gradient=True)
        fd = np.empty_like(dA)
        for k in range(d):
            e = np.zeros(d)
            e[k] = 1e-6 * s0[k]
            fd[:, :, k] = (hub._design(X + e, lo, hi, V, r, cols)[0] - hub._design(X - e, lo, hi, V, r, cols)[0]) / 2e-6
        assert A_.shape == (30, len(cols))
        assert np.max(np.abs(dA - fd)) < 1e-7 * np.max(np.abs(fd))
