"""Tests for src/core/clusters.py and the median split rule."""

import numpy as np
import pytest

from src import to_host, xp
from src.components.splitting import median_split_counts
from src.core.clusters import ClusterBatch, ClusterQueueBuilder


def _parts(rng, counts, d=2, fd=1, has_g=False):
    n = int(np.sum(counts))
    X = rng.standard_normal((n, d))
    F = rng.standard_normal((n, fd))
    G = rng.standard_normal((n, fd, d)) if has_g else None
    return X, F, G, np.asarray(counts, dtype=np.int64)


def test_median_split_counts():
    np.testing.assert_array_equal(
        to_host(median_split_counts(xp.asarray(np.array([5, 4, 1])))),
        [2, 3, 2, 2, 0, 1],
    )


@pytest.mark.parametrize("has_g", [False, True])
def test_builder_concatenates_and_grows(has_g):
    rng = np.random.default_rng(0)
    # Start far too small so both point and cluster buffers must grow several times.
    builder = ClusterQueueBuilder(2, 1, has_g, capacity_pts=1, capacity_clusters=1)
    parts = [_parts(rng, s, has_g=has_g) for s in ([3, 4], [5], [2, 2, 6])]
    for X, F, G, counts in parts:
        builder.append(
            xp.asarray(X),
            xp.asarray(F),
            None if G is None else xp.asarray(G),
            xp.asarray(counts),
        )

    batch = builder.build()
    np.testing.assert_array_equal(batch.X, np.concatenate([p[0] for p in parts]))
    np.testing.assert_array_equal(batch.F, np.concatenate([p[1] for p in parts]))
    np.testing.assert_array_equal(batch.counts, [3, 4, 5, 2, 2, 6])
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
    np.testing.assert_array_equal(batch.counts, [3, 2])
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
    np.testing.assert_array_equal(head.counts, [2])
    np.testing.assert_array_equal(tail.counts, [3, 4])
    np.testing.assert_array_equal(np.concatenate([head.X, tail.X]), batch.X)
    np.testing.assert_array_equal(np.concatenate([head.G, tail.G]), batch.G)
    np.testing.assert_array_equal(tail.starts, [0, 3])


class _FakeMemory:
    def __init__(self, available, total=0):
        self.available, self.total = available, total


def test_builder_checks_host_memory_on_every_append(monkeypatch):
    import src.core.clusters as clusters

    monkeypatch.setattr(clusters, "HOST_MEMORY_RESERVE_MIN", 0)
    rng = np.random.default_rng(0)
    builder = ClusterQueueBuilder(2, 1, False, capacity_pts=1000, capacity_clusters=8)
    builder.append(*_parts(rng, [2, 2]))
    # Plenty of reserved capacity, but writing 100 rows x 24 bytes needs 2,400 bytes.
    monkeypatch.setattr(clusters.psutil, "virtual_memory", lambda: _FakeMemory(1_000))
    with pytest.raises(MemoryError, match="min_pts_per_cluster"):
        builder.append(*_parts(rng, [50, 50]))
    assert builder.n_pts == 4  # the failed append left the queue unchanged
    monkeypatch.setattr(clusters.psutil, "virtual_memory", lambda: _FakeMemory(10**9))
    builder.append(*_parts(rng, [50, 50]))
    assert builder.build().n_pts == 104


def test_host_memory_check_keeps_a_reserve(monkeypatch):
    import src.core.clusters as clusters

    gib = 2**30
    monkeypatch.setattr(clusters.psutil, "virtual_memory", lambda: _FakeMemory(10 * gib, 32 * gib))
    clusters.check_host_memory(6 * gib, "x")  # 10 GiB free - 3.2 GiB reserve = 6.8 GiB usable
    with pytest.raises(MemoryError, match=r"7\.00 GiB more.*10\.00 GiB is available and 3\.20 GiB"):
        clusters.check_host_memory(7 * gib, "x")
    monkeypatch.setattr(clusters.psutil, "virtual_memory", lambda: _FakeMemory(10 * gib, 4 * gib))
    with pytest.raises(MemoryError):  # small machine: the 2 GiB minimum reserve applies
        clusters.check_host_memory(int(8.5 * gib), "x")


def test_leaf_buffer_checks_host_memory_before_it_grows(monkeypatch):
    import src.core.clusters as clusters

    monkeypatch.setattr(clusters, "HOST_MEMORY_RESERVE_MIN", 0)
    buf = clusters.ColumnBuffer({"a": clusters.ParamSpec((10,))}, capacity=2)
    buf.append({"a": np.zeros((2, 10))})
    monkeypatch.setattr(clusters.psutil, "virtual_memory", lambda: _FakeMemory(200))  # 3 rows need 240 bytes
    with pytest.raises(MemoryError, match="leaf buffer"):
        buf.append({"a": np.ones((1, 10))})
    assert buf.n_rows == 2


def test_downloads_larger_than_the_pinned_buffer_arrive_bit_for_bit(monkeypatch):
    import src

    monkeypatch.setattr(src, "PINNED_STAGING_BYTES", 8 * 7)  # 7 doubles per piece
    monkeypatch.setattr(src, "_pinned_staging", None)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((53, 3))
    for arr, out in (
        (xp.asarray(X), np.empty((53, 3))),  # 159 values: 23 pieces, the last one short
        (xp.asarray(X.T.copy()).T, np.empty((53, 3))),  # not contiguous on the device
        (xp.asarray(X[:, 0].astype(np.int32)), np.empty(53, dtype=np.int64)),  # dtype differs: plain copy
    ):
        src.copy_to_host(arr, out)
        np.testing.assert_array_equal(out, to_host(arr).astype(out.dtype))
        assert (src._pinned_staging is not None) == src.IS_CUPY_BACKEND  # the pinned path ran on the GPU
    builder = ClusterQueueBuilder(3, 1, False, capacity_pts=10, capacity_clusters=2)
    builder.append(xp.asarray(X), xp.asarray(X[:, :1]), None, xp.asarray([20, 33]))
    batch = builder.build()
    np.testing.assert_array_equal(batch.X, X)
    np.testing.assert_array_equal(batch.F, X[:, :1])
    leaves = src.core.clusters.ColumnBuffer({"c": src.core.clusters.ParamSpec((3,))}, capacity=4)
    leaves.append({"c": xp.asarray(X)})
    np.testing.assert_array_equal(leaves.result()["c"], X)


def test_training_releases_the_pinned_staging_buffer(monkeypatch):
    import src
    import src.engine.driver as driver
    from tests._helpers import train_surrogate

    held = []
    monkeypatch.setattr(driver, "release_pinned_staging", lambda: (held.append(src._pinned_staging is not None),
                                                                   src.release_pinned_staging()))  # fmt: skip
    train_surrogate(np.random.default_rng(0).uniform(-1, 1, (500, 2)), f_fn=lambda x: (1 + x[:, :1] ** 2),
                    atol=0.05, rtol=0.05, num_initial_clusters=2, min_pts_per_cluster=20, use_work_stealing=False)  # fmt: skip
    assert held == [src.IS_CUPY_BACKEND] and src._pinned_staging is None
