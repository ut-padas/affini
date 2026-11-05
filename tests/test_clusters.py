"""Tests for src/clusters.py and the median split rule."""

import numpy as np
import pytest

from src import xp
from src.adaptive_clustering_utils import median_split_sizes
from src.clusters import ClusterBatch, ClusterQueueBuilder


def _parts(rng, sizes, d=2, fd=1, has_g=False):
    n = int(np.sum(sizes))
    X = rng.standard_normal((n, d))
    F = rng.standard_normal((n, fd))
    G = rng.standard_normal((n, fd, d)) if has_g else None
    return X, F, G, np.asarray(sizes, dtype=np.int64)


def test_median_split_sizes():
    np.testing.assert_array_equal(
        np.asarray(median_split_sizes(xp.asarray(np.array([5, 4, 1])))),
        [2, 3, 2, 2, 0, 1],
    )


@pytest.mark.parametrize("has_g", [False, True])
def test_builder_concatenates_and_grows(has_g):
    rng = np.random.default_rng(0)
    # Start far too small so both point and cluster buffers must grow several times.
    builder = ClusterQueueBuilder(2, 1, has_g, capacity_pts=1, capacity_clusters=1)
    parts = [_parts(rng, s, has_g=has_g) for s in ([3, 4], [5], [2, 2, 6])]
    for X, F, G, sizes in parts:
        builder.append(
            xp.asarray(X),
            xp.asarray(F),
            None if G is None else xp.asarray(G),
            xp.asarray(sizes),
        )

    batch = builder.build()
    np.testing.assert_array_equal(batch.X, np.concatenate([p[0] for p in parts]))
    np.testing.assert_array_equal(batch.F, np.concatenate([p[1] for p in parts]))
    np.testing.assert_array_equal(batch.sizes, [3, 4, 5, 2, 2, 6])
    np.testing.assert_array_equal(batch.starts, [0, 3, 7, 12, 14, 16])
    if has_g:
        np.testing.assert_array_equal(batch.G, np.concatenate([p[2] for p in parts]))
    else:
        assert batch.G is None


def test_builder_drops_empty_clusters_and_ignores_g_without_has_g():
    rng = np.random.default_rng(1)
    X, F, G, _ = _parts(rng, [3, 2], has_g=True)
    builder = ClusterQueueBuilder(
        2, 1, has_g=False, capacity_pts=5, capacity_clusters=4
    )
    builder.append(X, F, G, np.array([0, 3, 0, 2]))
    batch = builder.build()
    np.testing.assert_array_equal(batch.sizes, [3, 2])
    np.testing.assert_array_equal(batch.X, X)
    assert batch.G is None


def test_empty_builder_gives_empty_batch():
    batch = ClusterQueueBuilder(
        3, 2, has_g=True, capacity_pts=10, capacity_clusters=4
    ).build()
    assert batch.n_clusters == 0 and batch.X.shape == (0, 3) and batch.F.shape == (0, 2)


def test_cluster_batch_split():
    rng = np.random.default_rng(2)
    batch = ClusterBatch(*_parts(rng, [2, 3, 4], has_g=True))
    head, tail = batch.split(1)
    np.testing.assert_array_equal(head.sizes, [2])
    np.testing.assert_array_equal(tail.sizes, [3, 4])
    np.testing.assert_array_equal(np.concatenate([head.X, tail.X]), batch.X)
    np.testing.assert_array_equal(np.concatenate([head.G, tail.G]), batch.G)
    np.testing.assert_array_equal(tail.starts, [0, 3])
