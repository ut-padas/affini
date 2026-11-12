"""Tests for src/clusters.py and the median split rule."""

import numpy as np
import pytest

from src import _to_host, xp
from src.clusters import ClusterBatch, ClusterQueueBuilder, ColumnBuffer, ParamSpec
from src.splitting import median_split_sizes
from tests._helpers import host


def _parts(rng, sizes, d=2, fd=1, has_g=False):
    n = int(np.sum(sizes))
    X = rng.standard_normal((n, d))
    F = rng.standard_normal((n, fd))
    G = rng.standard_normal((n, fd, d)) if has_g else None
    return X, F, G, np.asarray(sizes, dtype=np.int64)


def test_median_split_sizes():
    np.testing.assert_array_equal(
        _to_host(median_split_sizes(xp.asarray(np.array([5, 4, 1])))),
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


class _FakeMemory:
    def __init__(self, available, total=0):
        self.available, self.total = available, total


def test_builder_checks_host_memory_on_every_append(monkeypatch):
    import src.clusters as clusters

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
    import src.clusters as clusters

    gib = 2**30
    monkeypatch.setattr(clusters.psutil, "virtual_memory", lambda: _FakeMemory(10 * gib, 32 * gib))
    clusters.check_host_memory(6 * gib, "x")  # 10 GiB free - 3.2 GiB reserve = 6.8 GiB usable
    with pytest.raises(MemoryError, match=r"7\.00 GiB more.*10\.00 GiB is available and 3\.20 GiB"):
        clusters.check_host_memory(7 * gib, "x")
    monkeypatch.setattr(clusters.psutil, "virtual_memory", lambda: _FakeMemory(10 * gib, 4 * gib))
    with pytest.raises(MemoryError):  # small machine: the 2 GiB minimum reserve applies
        clusters.check_host_memory(int(8.5 * gib), "x")


def _tree(n, offset=0):
    return {
        "parent": np.arange(n, dtype=np.int64) - 1 + offset,
        "root": np.full(n, 7 + offset, dtype=np.int64),
        "morton": np.arange(n, dtype=np.int64) * 64 + offset,
    }


def test_tree_fields_follow_clusters_through_split_select_and_the_builder():
    rng = np.random.default_rng(2)
    X, F, _, sizes = _parts(rng, [3, 0, 4, 2])
    tree = _tree(4)
    batch = ClusterBatch(X, F, None, sizes, **tree)
    head, tail = batch.split(2)
    for name in tree:
        np.testing.assert_array_equal(head.tree_fields()[name], tree[name][:2])
        np.testing.assert_array_equal(tail.tree_fields()[name], tree[name][2:])

    builder = ClusterQueueBuilder(2, 1, False, capacity_pts=1, capacity_clusters=1, track_trees=True)
    builder.append(xp.asarray(X), xp.asarray(F), None, xp.asarray(sizes), **tree)
    X2, F2, _, sizes2 = _parts(rng, [5])
    builder.append(X2, F2, None, sizes2, **_tree(1, offset=100))
    built = builder.build()  # drops the empty cluster with its tree fields
    expected = {k: np.r_[np.delete(tree[k], 1), _tree(1, offset=100)[k]] for k in tree}
    for name, value in expected.items():
        np.testing.assert_array_equal(built.tree_fields()[name], value)
        assert built.tree_fields()[name].dtype == value.dtype


def test_tree_fields_are_all_or_nothing():
    rng = np.random.default_rng(3)
    X, F, _, sizes = _parts(rng, [2, 2])
    with pytest.raises(ValueError, match="all of the tree fields"):
        ClusterBatch(X, F, None, sizes, parent=np.zeros(2, dtype=np.int64))
    assert not ClusterBatch(X, F, None, sizes).has_tree
    tracking = ClusterQueueBuilder(2, 1, False, 4, 2, track_trees=True)
    with pytest.raises(ValueError, match="tree fields"):
        tracking.append(X, F, None, sizes)
    plain = ClusterQueueBuilder(2, 1, False, 4, 2)
    with pytest.raises(ValueError, match="tree fields"):
        plain.append(X, F, None, sizes, **_tree(2))


@pytest.mark.parametrize("on_device", [False, True])
def test_column_buffer_grows_on_its_backend(on_device):
    buf = ColumnBuffer({"v": ParamSpec((2,)), "i": ParamSpec((), np.int64)}, capacity=1, on_device=on_device)
    for k in range(3):
        buf.append({"v": np.full((2, 2), k, dtype=float), "i": np.array([k, k])})
    out = buf.result()
    assert isinstance(out["v"], type(xp.empty(0)) if on_device else np.ndarray)
    np.testing.assert_array_equal(host(out["i"]), [0, 0, 1, 1, 2, 2])
    assert out["i"].dtype == np.int64 and out["v"].shape == (6, 2)
