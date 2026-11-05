"""Tests for src/data_sampler.py."""

import numpy as np

from src import _to_host, xp
from src.data_sampler import DataSampler

DOMAIN_2D = xp.asarray([[0.0, 1.0], [-2.0, 3.0]])


def test_uniform_is_in_bounds_and_reproducible():
    X = _to_host(DataSampler.sample_uniform(DOMAIN_2D, N=1000, seed=3))
    assert X.shape == (1000, 2) and X.dtype == np.float64
    assert np.all(X >= [0.0, -2.0]) and np.all(X <= [1.0, 3.0])
    np.testing.assert_array_equal(
        X, _to_host(DataSampler.sample_uniform(DOMAIN_2D, N=1000, seed=3))
    )
    X1 = _to_host(DataSampler.sample_uniform(xp.asarray([2.0, 4.0]), N=10))
    assert X1.shape == (10, 1) and np.all((X1 >= 2.0) & (X1 <= 4.0))


def test_grid_covers_the_corners():
    X = _to_host(DataSampler.sample_grid(DOMAIN_2D, num_points_per_dim=3))
    assert X.shape == (9, 2)
    np.testing.assert_array_equal(np.unique(X[:, 0]), [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(np.unique(X[:, 1]), [-2.0, 0.5, 3.0])


def test_latin_hypercube_has_one_point_per_stratum():
    n = 8
    X = _to_host(DataSampler.sample_latin_hypercube(DOMAIN_2D, N=n, seed=0))
    assert X.shape == (n, 2)
    lo, hi = np.array([0.0, -2.0]), np.array([1.0, 3.0])
    strata = np.floor((X - lo) / (hi - lo) * n).astype(int)
    for j in range(2):
        np.testing.assert_array_equal(np.sort(strata[:, j]), np.arange(n))
