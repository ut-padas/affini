"""
Exact kNN indexes in src/knn.py (GpuExactKNN, CpuExactKNN) must return exactly the
nearest leaves, nearest first. Each test pins a lesson learned while building them:

1. Far from the origin: the fast form |q|^2 - 2 q.x + |x|^2 loses the distance to
   float32 cancellation when coordinates are large relative to the gaps between
   points (recall fell to 0.005 at d=50 with every coordinate shifted by 1e4). Both
   indexes centre the data on the centroids' mean; these tests use shifted data.
2. Every code path is exercised: GPU fused kernel (k <= MAX_K, with 64 / 32 / 16
   queries per block for k <= 32 / 128 / 256) and GPU radix-select path (k > MAX_K);
   CPU direct path (d <= CPU_DIRECT_MAX_D) and CPU BLAS-tile path. Block-size
   boundaries (k = 32/33, 128/129, 256/257) are where shared-memory layouts change.
3. Ties keep the smaller index, on every path and for any CPU thread count, so
   results do not depend on scheduling.
4. float64 queries are read in place by the GPU kernel (no float32 copy) and must
   give the same neighbours as float32 queries.
5. Edge cases: one leaf, k larger than the number of leaves (all leaves), empty
   queries, invalid k.
"""

import numpy as np
import pytest

from src import IS_CUPY_BACKEND, xp
from tests._helpers import host

gpu_only = pytest.mark.skipif(not IS_CUPY_BACKEND, reason="GPU index")
cpu_only = pytest.mark.skipif(IS_CUPY_BACKEND, reason="CPU index is tested on the CPU backend")

OFFSETS = (0.0, 100.0, 1e4)  # lesson 1: cancellation appears with large coordinates


def _exact(X, Q, k):
    """Reference neighbour sets from float64 direct differences (no expanded form)."""
    d2 = np.stack([((X - q) ** 2).sum(1) for q in Q])
    return np.argsort(d2, axis=1, kind="stable")[:, :k], np.sqrt(np.sort(d2, axis=1)[:, :k])


def _check(dist, idx, X, Q, k):
    ref_idx, ref_dist = _exact(X, Q, min(k, len(X)))
    assert idx.shape == (len(Q), min(k, len(X)))
    assert all(set(a) == set(b) for a, b in zip(idx, ref_idx, strict=True))
    assert np.all(np.diff(dist, axis=1) >= 0)  # nearest first
    scale = max(1.0, float(np.abs(X).max()))
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-4, atol=1e-5 * scale)


def _data(n, m, d, offset, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) + offset, rng.normal(size=(m, d)) + offset


# --- GPU ---------------------------------------------------------------------------
@gpu_only
@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize(
    ("n", "m", "d", "k"),
    [(1, 7, 3, 1), (63, 130, 5, 10), (1000, 257, 17, 32), (1000, 257, 17, 33),  # BM 64 -> 32
     (5000, 300, 100, 64), (3000, 100, 20, 128), (3000, 100, 20, 129),  # BM 32 -> 16
     (3000, 100, 20, 256), (3000, 100, 20, 257), (2000, 64, 8, 2000)],  # fused -> radix, all leaves
)  # fmt: skip
def test_gpu_matches_exact_neighbours(n, m, d, k, offset):
    from src.knn import GpuExactKNN

    X, Q = _data(n, m, d, offset, n + m + d + k)
    dist, idx = GpuExactKNN(xp.asarray(X)).kneighbors(xp.asarray(Q), k)
    _check(host(dist), host(idx), X, Q, k)


@gpu_only
@pytest.mark.parametrize("k", [4, 300])  # fused kernel and radix-select path
def test_gpu_ties_keep_the_smaller_index(k):
    from src.knn import GpuExactKNN

    X = xp.asarray(np.repeat(np.eye(3), 400, axis=0) + 1e4)  # hundreds of exact ties, shifted
    _, idx = GpuExactKNN(X).kneighbors(xp.asarray([[1e4 + 1.0, 1e4, 1e4]]), k)
    assert host(idx).tolist() == [list(range(k))]


@gpu_only
def test_gpu_float64_and_float32_queries_agree():
    from src.knn import GpuExactKNN

    X, Q = _data(4000, 500, 30, 50.0, 7)
    index = GpuExactKNN(xp.asarray(X))
    _, i64 = index.kneighbors(xp.asarray(Q, dtype=xp.float64), 16)
    _, i32 = index.kneighbors(xp.asarray(Q, dtype=xp.float32), 16)
    assert all(set(a) == set(b) for a, b in zip(host(i64), host(i32), strict=True))


@gpu_only
def test_gpu_empty_queries_and_invalid_k():
    from src.knn import GpuExactKNN

    index = GpuExactKNN(xp.asarray(np.eye(3)))
    dist, idx = index.kneighbors(xp.empty((0, 3)), 2)
    assert dist.shape == (0, 2) and idx.shape == (0, 2)
    with pytest.raises(ValueError):
        index.kneighbors(xp.zeros((1, 3)), 0)


# --- CPU ---------------------------------------------------------------------------
@cpu_only
@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize(
    ("n", "m", "d", "k"),
    [(1, 5, 3, 1), (300, 200, 5, 10), (3000, 150, 16, 40),  # direct path (d <= 16)
     (2000, 150, 17, 20), (2000, 150, 100, 20), (5000, 64, 20, 300), (500, 40, 50, 500)],  # BLAS tiles
)  # fmt: skip
def test_cpu_matches_exact_neighbours(n, m, d, k, offset):
    from src.knn import CpuExactKNN

    X, Q = _data(n, m, d, offset, n + d + k)
    dist, idx = CpuExactKNN(X).kneighbors(Q, k)
    _check(dist, idx, X, Q, k)


@cpu_only
@pytest.mark.parametrize("d", [3, 40])  # direct path and BLAS-tile path
def test_cpu_ties_keep_the_smaller_index_for_any_thread_count(monkeypatch, d):
    from src.knn import CpuExactKNN

    X = np.repeat(np.eye(d), 50, axis=0) + 1e4
    q = X[:1] + 0.0
    for threads in ("1", "4"):
        monkeypatch.setenv("OMP_NUM_THREADS", threads)
        _, idx = CpuExactKNN(X).kneighbors(q, 7)
        assert idx.tolist() == [list(range(7))]


@cpu_only
def test_cpu_empty_queries_and_invalid_k():
    from src.knn import CpuExactKNN

    index = CpuExactKNN(np.eye(3))
    dist, idx = index.kneighbors(np.empty((0, 3)), 2)
    assert dist.shape == (0, 2) and idx.shape == (0, 2)
    with pytest.raises(ValueError):
        index.kneighbors(np.zeros((1, 3)), 0)
