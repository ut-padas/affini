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
