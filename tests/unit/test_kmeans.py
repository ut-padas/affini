"""
The custom k-means (src/kmeans.py) on the active backend: valid, reproducible labels of
k-means quality. On the CPU, results must not depend on the thread count (chunk sums
are added in chunk order); quality is compared with scikit-learn when it is installed
(it is only in the ``research`` group).
"""

import numpy as np
import pytest

from src import IS_CUPY_BACKEND, xp
from tests._helpers import host


def _inertia(X, labels, k):
    C = np.zeros((k, X.shape[1]))
    np.add.at(C, labels, X)
    counts = np.bincount(labels, minlength=k)
    C[counts > 0] /= counts[counts > 0, None]
    return float(((X - C[labels]) ** 2).sum())


@pytest.mark.parametrize(("d", "k"), [(5, 40), (30, 25)])  # direct and BLAS paths on the CPU
def test_labels_are_valid_reproducible_and_of_kmeans_quality(d, k):
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    from src.kmeans import kmeans_labels

    X = np.random.default_rng(0).uniform(-2, 3, (20_000, d))
    a = host(kmeans_labels(xp.asarray(X), k, seed=3))
    b = host(kmeans_labels(xp.asarray(X), k, seed=3))
    np.testing.assert_array_equal(a, b)  # deterministic for a given seed
    assert a.min() >= 0 and a.max() < k and len(np.unique(a)) == k
    ref = sklearn_cluster.KMeans(n_clusters=k, random_state=0, n_init=1).fit(X).inertia_
    assert _inertia(X, a, k) <= 1.02 * ref


@pytest.mark.skipif(IS_CUPY_BACKEND, reason="CPU threads")
@pytest.mark.parametrize("d", [5, 30])
def test_cpu_labels_do_not_depend_on_the_thread_count(d):
    from src.kmeans import kmeans_labels

    X = np.random.default_rng(1).normal(size=(50_000, d)) + 100.0
    one = kmeans_labels(X, 30, seed=0, max_iter=3, threads=1)
    four = kmeans_labels(X, 30, seed=0, max_iter=3, threads=4)
    np.testing.assert_array_equal(one, four)


def test_separated_blobs_are_recovered():
    from src.kmeans import kmeans_labels

    rng = np.random.default_rng(1)
    centres = rng.uniform(-50, 50, (6, 10))
    X = np.concatenate([c + rng.normal(scale=0.1, size=(500, 10)) for c in centres])
    labels = host(kmeans_labels(xp.asarray(X), 6, seed=0))
    assert all(len(np.unique(labels[i * 500 : (i + 1) * 500])) == 1 for i in range(6))
    assert len(np.unique(labels)) == 6


def test_more_clusters_than_distinct_points():
    from src.kmeans import kmeans_labels

    X = xp.asarray(np.repeat(np.eye(3), 10, axis=0))
    labels = host(kmeans_labels(X, 5, seed=0))
    assert labels.shape == (30,) and labels.min() >= 0 and labels.max() < 5
