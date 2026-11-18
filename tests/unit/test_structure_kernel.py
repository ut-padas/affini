"""MetricKernel (src/structure/kernel.py): the learned metric and the greedy fit."""

import numpy as np

from src.structure import kernel


def test_the_learned_metric_aligns_with_a_ridge_and_meets_the_tolerance():
    q = np.array([0.8, 0.6])

    def f(X):
        return np.cos(3.0 * X @ q)

    lo, hi = -np.ones(2), np.ones(2)
    m = kernel.build(f, lo, hi, 1e-2, seed=1)
    w, V = np.linalg.eigh(m.L.T @ m.L)
    assert abs(V[:, -1] @ q) > 0.95  # the metric's longest axis in 1/length is the ridge's direction
    X = np.random.default_rng(0).uniform(lo, hi, (4000, 2))
    assert np.max(np.abs(m.predict(X) - f(X))) <= 1e-2
