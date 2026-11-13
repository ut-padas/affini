"""
Exact kNN indexes in src/neighbors/knn.py (GpuExactKNN, CpuExactKNN) must return exactly the
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
    from src.neighbors.knn import GpuExactKNN

    X, Q = _data(n, m, d, offset, n + m + d + k)
    dist, idx = GpuExactKNN(xp.asarray(X)).kneighbors(xp.asarray(Q), k)
    _check(host(dist), host(idx), X, Q, k)


@gpu_only
@pytest.mark.parametrize("k", [4, 300])  # fused kernel and radix-select path
def test_gpu_ties_keep_the_smaller_index(k):
    from src.neighbors.knn import GpuExactKNN

    X = xp.asarray(np.repeat(np.eye(3), 400, axis=0) + 1e4)  # hundreds of exact ties, shifted
    _, idx = GpuExactKNN(X).kneighbors(xp.asarray([[1e4 + 1.0, 1e4, 1e4]]), k)
    assert host(idx).tolist() == [list(range(k))]


@gpu_only
def test_gpu_float64_and_float32_queries_agree():
    from src.neighbors.knn import GpuExactKNN

    X, Q = _data(4000, 500, 30, 50.0, 7)
    index = GpuExactKNN(xp.asarray(X))
    _, i64 = index.kneighbors(xp.asarray(Q, dtype=xp.float64), 16)
    _, i32 = index.kneighbors(xp.asarray(Q, dtype=xp.float32), 16)
    assert all(set(a) == set(b) for a, b in zip(host(i64), host(i32), strict=True))


@gpu_only
def test_gpu_empty_queries_and_invalid_k():
    from src.neighbors.knn import GpuExactKNN

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
    from src.neighbors.knn import CpuExactKNN

    X, Q = _data(n, m, d, offset, n + d + k)
    dist, idx = CpuExactKNN(X).kneighbors(Q, k)
    _check(dist, idx, X, Q, k)


@cpu_only
@pytest.mark.parametrize("d", [3, 40])  # direct path and BLAS-tile path
def test_cpu_ties_keep_the_smaller_index_for_any_thread_count(monkeypatch, d):
    from src.neighbors.knn import CpuExactKNN

    X = np.repeat(np.eye(d), 50, axis=0) + 1e4
    q = X[:1] + 0.0
    for threads in ("1", "4"):
        monkeypatch.setenv("OMP_NUM_THREADS", threads)
        _, idx = CpuExactKNN(X).kneighbors(q, 7)
        assert idx.tolist() == [list(range(7))]


@cpu_only
def test_cpu_empty_queries_and_invalid_k():
    from src.neighbors.knn import CpuExactKNN

    index = CpuExactKNN(np.eye(3))
    dist, idx = index.kneighbors(np.empty((0, 3)), 2)
    assert dist.shape == (0, 2) and idx.shape == (0, 2)
    with pytest.raises(ValueError):
        index.kneighbors(np.zeros((1, 3)), 0)


# --- Mixed scales: float32 rounding must never pick a wrong neighbour ----
def _index(points):
    from src.neighbors.knn import CpuExactKNN, GpuExactKNN

    return GpuExactKNN(xp.asarray(points)) if IS_CUPY_BACKEND else CpuExactKNN(points)


def test_points_merged_by_float32_rounding_are_still_distinguished():
    # Centred on the mean (~3.3e8), 0 and 1 round to the same float32 value; the float64
    # verification must recover the true nearest neighbour (index 1).
    points = np.array([[0.0], [1.0], [1e9]])
    dist, idx = _index(points).kneighbors(xp.asarray([[1.0]]) if IS_CUPY_BACKEND else np.array([[1.0]]), 1)
    assert host(idx).tolist() == [[1]] and float(host(dist)[0, 0]) == 0.0


@pytest.mark.parametrize("d", [1, 16, 17, 30])  # around the CPU direct/BLAS boundary
def test_mixed_scale_clouds_with_outliers_repeats_and_k_above_one(d):
    rng = np.random.default_rng(d)
    cloud = rng.normal(scale=1e-3, size=(400, d)) + 5e6  # small gaps, far from the origin
    outliers = rng.normal(scale=1e9, size=(20, d))
    repeats = np.repeat(cloud[:5], 3, axis=0)  # exact duplicates: true ties
    points = np.concatenate([cloud, outliers, repeats])
    queries = cloud[rng.choice(len(cloud), 50)] + rng.normal(scale=1e-4, size=(50, d))
    k = 7
    dist, idx = _index(points).kneighbors(xp.asarray(queries) if IS_CUPY_BACKEND else queries, k)
    dist, idx = host(dist), host(idx)
    ref = np.stack([((points - q) ** 2).sum(1) for q in queries])
    # The float64 reference itself rounds at this offset (gaps 1e-3 at 5e6: about 1e-6
    # relative), so distances agree to that level; every returned neighbour must be one
    # of the true k nearest, up to the same rounding (true ties may swap).
    np.testing.assert_allclose(dist**2, np.sort(ref, axis=1)[:, :k], rtol=1e-4, atol=0)
    for row, q_idx in zip(ref, idx, strict=True):
        kth = np.sort(row)[k - 1]
        assert np.all(row[q_idx] <= kth * (1 + 1e-5))
    assert len(set(map(tuple, idx))) > 1  # sanity: not a degenerate answer


@pytest.mark.parametrize(
    ("points", "query", "nearest"),
    [([[0.0], [1.0], [1e9]], 0.500000001, 1), ([[0.0], [1.0], [1e9]], 0.499999999, 0),
     ([[0.0], [1e-9], [1e9]], 1e-9, 1), ([[0.0], [1.0], [1e17]], 1.0, 1)],
)  # fmt: skip
def test_reference_uses_the_original_coordinates(points, query, nearest):
    """Re-ranking and fallback must use the original points, not centred float64 copies."""
    q = np.array([[query]])
    _, idx = _index(np.array(points)).kneighbors(xp.asarray(q) if IS_CUPY_BACKEND else q, 1)
    assert int(host(idx)[0, 0]) == nearest


def test_merged_nearby_leaves_next_to_an_outlier_keep_the_true_neighbour():
    """31 leaves 1e-9 apart plus an outlier at 1e12."""
    points = np.concatenate([np.arange(31)[:, None] * 1e-9, [[1e12]]])
    q = points[30:31]
    dist, idx = _index(points).kneighbors(xp.asarray(q) if IS_CUPY_BACKEND else q, 10)
    assert int(host(idx)[0, 0]) == 30 and float(host(dist)[0, 0]) == 0.0
    assert set(host(idx)[0].tolist()) == set(range(21, 31))


@pytest.mark.parametrize(("points", "query", "k", "nearest"), [
    ("overflow17", None, 1, 2),  # kk == n: fast search bypassed
    ("spread", [[1e20]], 1, 11),  # kk < n: infinite float32 cutoff must not certify
])  # fmt: skip
def test_non_finite_fast_results_are_never_trusted(points, query, k, nearest):
    """Float32 overflow on finite float64 inputs must fall back, not return a placeholder."""
    if points == "overflow17":
        P = np.zeros((3, 17))
        P[:, 0] = [0.0, 1.0, 1e20]
        q = P[2:3]
    else:
        P = (np.arange(12) * 1e15)[:, None]
        q = np.array(query)
    _, idx = _index(P).kneighbors(xp.asarray(q) if IS_CUPY_BACKEND else q, k)
    assert int(host(idx)[0, 0]) == nearest


def test_candidate_rerank_and_fallback_share_one_reduction():
    """The first neighbour must not depend on k (certified vs fallback rows)."""
    P = np.zeros((12, 17))
    P[0] = 1e-8
    P[0, 0] = P[1, 0] = 1.0
    P[2:11] = np.arange(12, 21)[:, None]
    P[11] = 1e15
    index, q = _index(P), np.zeros((1, 17))
    firsts = {int(host(index.kneighbors(xp.asarray(q) if IS_CUPY_BACKEND else q, k)[1])[0, 0]) for k in (1, 2, 5, 6)}
    assert len(firsts) == 1


def test_tree_orders_by_squared_distance_not_rounded_root():
    """Squared distances 1+2e-16 and 1 share a rounded root; index 1 is nearer."""
    P = np.array([[1.0, 1.5e-8], [1.0, 0.0], [10.0, 0.0]])
    q = np.zeros((1, 2))
    lists = [host(_index(P).kneighbors(xp.asarray(q) if IS_CUPY_BACKEND else q, k)[1])[0].tolist() for k in (1, 2, 3)]
    assert lists == [[1], [1, 0], [1, 0, 2]]


@pytest.mark.parametrize("bad", [(1, 18), (1, 16), (17,)])
def test_wrong_query_shapes_are_rejected_before_native_kernels(bad):
    """Kernels use the index's d as stride and have no bounds checks."""
    P = np.zeros((3, 17))
    q = np.zeros(bad)
    with pytest.raises(ValueError, match="queries must have shape"):
        _index(P).kneighbors(xp.asarray(q) if IS_CUPY_BACKEND else q, 1)


@cpu_only
def test_cpu_tree_path_keeps_no_float32_copy_until_unverified_search():
    from src.neighbors.knn import CPU_TREE_MAX_D, CpuExactKNN

    index = CpuExactKNN(np.random.default_rng(0).normal(size=(500, CPU_TREE_MAX_D)))
    index.kneighbors(np.zeros((3, CPU_TREE_MAX_D)), 4)
    assert index._points32 is None and index.last_search_stats["strategy"] == "kdtree"
    index.kneighbors_unverified(np.zeros((3, CPU_TREE_MAX_D)), 4)
    assert index._points32 is not None and index.last_search_stats["strategy"] == "unverified"


@cpu_only
def test_tree_sentinels_never_reach_the_scorer_and_overflow_raises(monkeypatch):
    """SciPy's "no neighbour" (index n, inf) must be rejected before the unchecked scorer."""
    import src.neighbors.knn as knn

    real = knn._candidate_d2

    def checked(xp_, q, P, cand):
        assert int(np.max(cand)) < len(P) and int(np.min(cand)) >= 0
        return real(xp_, q, P, cand)

    monkeypatch.setattr(knn, "_candidate_d2", checked)
    # Every requested squared distance overflows: a clear range error, no placeholders.
    with pytest.raises(ValueError, match="float64 range"):
        knn.CpuExactKNN(np.array([[0.0], [1.0], [2.0]])).kneighbors(np.array([[1e200]]), 2)
    # A valid nearest neighbour with an unrepresentable extra candidate is still answered.
    dist, idx = knn.CpuExactKNN(np.array([[0.0], [1e200]])).kneighbors(np.array([[0.0]]), 1)
    assert idx.tolist() == [[0]] and dist.tolist() == [[0.0]]


@cpu_only
def test_tree_sentinel_extra_candidate_from_a_test_double():
    from src.neighbors.knn import CpuExactKNN

    index = CpuExactKNN(np.array([[0.0], [1.0], [5.0]]))

    class Sentinel:
        def query(self, q, k, workers):  # one valid neighbour, then "no neighbour" entries
            return np.array([[0.0] + [np.inf] * (k - 1)]), np.array([[0] + [3] * (k - 1)])

    index._kdtree = Sentinel()
    _, idx = index.kneighbors(np.array([[0.0]]), 1)
    assert idx.tolist() == [[0]]
    with pytest.raises(ValueError, match="float64 range"):
        index.kneighbors(np.array([[0.0]]), 2)
