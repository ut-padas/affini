"""ChebTT (src/structure/chebtt.py): maxvol, the cross on low-rank functions, and its checks."""

import numpy as np

from src.structure import chebtt


def test_maxvol_rows_dominate_every_other_row():
    A = np.random.default_rng(0).normal(size=(200, 6))
    rows = chebtt.maxvol(A)
    B = A @ np.linalg.inv(A[rows])
    assert len(set(rows.tolist())) == 6
    assert np.max(np.abs(B)) <= 1.05 + 1e-12


def test_a_product_of_univariate_functions_is_recovered_in_high_dimension():
    d = 10

    def f(X):
        return np.prod(1.0 / (1.0 + 0.5 * X**2), axis=1)

    lo, hi = -np.ones(d), np.ones(d)
    m = chebtt.build(f, lo, hi, 1e-4)
    X = np.random.default_rng(1).uniform(lo, hi, (2000, d))
    assert np.max(np.abs(m.predict(X) - f(X))) <= 1e-4
    assert m.evals < 20_000  # linear in d: a tensor grid of the same degrees has 17^10 points


def test_a_narrow_peak_away_from_random_fibres_is_found():
    d = 5

    def f(X):
        return np.exp(-np.sum((X / 0.15) ** 2, axis=1))

    lo, hi = -3 * np.ones(d), 3 * np.ones(d)
    m = chebtt.build(f, lo, hi, 0.01)
    X = np.concatenate([np.random.default_rng(2).normal(0.0, 0.1, (2000, d)), np.zeros((1, d))])
    assert np.max(np.abs(m.predict(X) - f(X))) <= 0.01


def test_separate_trains_share_one_cache_and_match_independent_trains():
    d = 4

    def f(X):
        return np.stack([np.cos(X.sum(axis=1)), np.prod(1.0 / (1.0 + X**2), axis=1)], axis=1)

    lo, hi = -np.ones(d), np.ones(d)
    v = chebtt.build_vector(f, lo, hi, 1e-3, 2)
    alone = [chebtt.build(lambda X, c=c: f(X)[:, c], lo, hi, 1e-3) for c in range(2)]
    X = np.random.default_rng(3).uniform(lo, hi, (2000, d))
    assert np.array_equal(v.predict(X), np.stack([m.predict(X) for m in alone], axis=1))
    assert v.evals < sum(m.evals for m in alone)  # the pre-pass and the Sobol checks are shared


def test_the_metric_rotates_only_high_rank_ridges_and_keeps_a_separable_output_on_the_axes():
    d = 6
    q = np.random.default_rng(4).normal(size=d)
    q /= np.linalg.norm(q)

    def f(X):
        s = X @ q
        return np.stack([np.tanh(2 * s), np.sin(3 * s), np.prod(1.0 / (1.0 + X**2), axis=1)], axis=1)

    lo, hi = -np.ones(d), np.ones(d)
    v = chebtt.build_vector(f, lo, hi, 1e-3, 3, rotate=True)
    rotated = {c for cols, Q, _ in v.groups if Q is not None for c in cols}
    assert rotated and rotated <= {0, 1}  # the ridges, never the separable product
    assert all(max(m.ranks) <= 2 for cols, Q, ms in v.groups if Q is not None for m in ms)
    X = np.random.default_rng(5).uniform(lo, hi, (4000, d))
    assert np.max(np.abs(v.predict(X) - f(X))) <= 1e-3


def test_outputs_of_low_rank_on_the_axes_get_exactly_the_plain_model():
    d = 5

    def f(X):
        return np.stack([np.cos(X.sum(axis=1)), np.prod(1.0 / (1.0 + X**2), axis=1)], axis=1)

    lo, hi = -np.ones(d), np.ones(d)
    plain, metric = chebtt.build_vector(f, lo, hi, 1e-3, 2), chebtt.build_vector(f, lo, hi, 1e-3, 2, rotate=True)
    X = np.random.default_rng(6).uniform(lo, hi, (2000, d))
    assert metric.evals == plain.evals  # no pilot
    assert np.array_equal(metric.predict(X), plain.predict(X))


def test_no_rotation_where_the_oracle_fails_outside_the_box():
    d = 4
    q = np.ones(d) / 2.0

    def f(X):  # a ridge defined only on the box
        bad = np.any(np.abs(X) > 1.0 + 1e-12, axis=1)
        return np.where(bad, np.nan, np.exp(X @ q))[:, None]

    lo, hi = -np.ones(d), np.ones(d)
    v = chebtt.build_vector(f, lo, hi, 1e-3, 1, rotate=True)
    assert [Q is None for _, Q, _ in v.groups] == [True]
