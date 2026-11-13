"""
Fast NumPy replacements in src/numerics must reproduce the operations they replace
exactly, and the GPU's deterministic reductions and small solves must agree with NumPy
and give each segment the same bits whatever else is in the batch.
"""

import numpy as np
import pytest

from src import IS_CUPY_BACKEND, xp
from tests._helpers import host


@pytest.mark.skipif(IS_CUPY_BACKEND, reason="NumPy contract: bit-identical to add.at (CuPy sums by segment)")
def test_segment_sum_is_bit_identical_to_add_at():
    from src.numerics import segment_sum

    rng = np.random.default_rng(0)
    for trial in range(50):
        K, n, d = int(rng.integers(1, 30)), int(rng.integers(0, 2000)), int(rng.integers(1, 5))
        labels = rng.integers(0, K, n)
        if trial % 2:
            labels = np.sort(labels)
        values = rng.normal(size=(n, d, d)) * 10 ** rng.uniform(-3, 3, (n, 1, 1))
        ref = np.zeros((K, d, d))
        np.add.at(ref, labels, values)
        np.testing.assert_array_equal(host(segment_sum(xp.asarray(values), xp.asarray(labels), K)), ref)


def test_segment_sum_matches_add_at_and_is_reproducible():
    from src.numerics import segment_sum

    rng = np.random.default_rng(5)
    for trial in range(20):
        K, n = int(rng.integers(1, 30)), int(rng.integers(0, 3000))
        labels = rng.integers(0, K, n)
        if trial % 2:
            labels = np.sort(labels)
        values = rng.normal(size=(n, 3))
        ref = np.zeros((K, 3))
        np.add.at(ref, labels, values)
        out = host(segment_sum(xp.asarray(values), xp.asarray(labels), K))
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(host(segment_sum(xp.asarray(values), xp.asarray(labels), K)), out)


def _segments(rng, K, lo=0, hi=700):
    counts = rng.integers(lo, hi, K)
    return counts, np.r_[0, np.cumsum(counts)[:-1]]


def test_segment_cross_matches_per_segment_products_with_weights_offsets_and_empty_segments():
    from src.numerics import segment_cross

    rng = np.random.default_rng(1)
    for P, Q in ((1, 1), (3, 2), (21, 1)):
        K = 9
        counts, starts = _segments(rng, K)
        counts[3] = 0
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        N = int(counts.sum())
        A, B, w = rng.normal(size=(N, P)), rng.normal(size=(N, Q)), rng.random(N)
        a_off, b_off = rng.normal(size=(K, P)), rng.normal(size=(K, Q))
        out = host(segment_cross(xp.asarray(A), xp.asarray(B), xp.asarray(starts), xp.asarray(counts), xp.asarray(w),
                                 xp.asarray(a_off), xp.asarray(b_off)))  # fmt: skip
        sym = host(segment_cross(xp.asarray(A), xp.asarray(A), xp.asarray(starts), xp.asarray(counts), symmetric=True))
        ones = host(segment_cross(xp.asarray(A), None, xp.asarray(starts), xp.asarray(counts)))
        W2 = np.stack([w, 1 - w])
        multi = host(segment_cross(xp.asarray(A), xp.asarray(B), xp.asarray(starts), xp.asarray(counts), xp.asarray(W2)))
        for k in range(K):
            s = slice(starts[k], starts[k] + counts[k])
            ref = (A[s] - a_off[k]).T @ ((B[s] - b_off[k]) * w[s][:, None])
            np.testing.assert_allclose(out[k], ref, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(sym[k], A[s].T @ A[s], rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(ones[k, :, 0], A[s].sum(axis=0), rtol=1e-11, atol=1e-11)
            for i in range(2):
                np.testing.assert_allclose(multi[i, k], A[s].T @ (B[s] * W2[i, s][:, None]), rtol=1e-11, atol=1e-11)
        assert np.all(out[3] == 0)


@pytest.mark.skipif(not IS_CUPY_BACKEND, reason="NumPy sums are sequential already")
def test_gpu_segment_sums_do_not_depend_on_the_rest_of_the_batch():
    """CuPy's add.reduceat subtracts prefix sums; segment_cross and _reduce_sum_1d must not."""
    from src.numerics import _reduce_sum_1d, segment_cross

    rng = np.random.default_rng(2)
    seg = rng.normal(size=(37, 4)) * 10 ** rng.uniform(-3, 3, (37, 1))
    ref_cross = ref_sum = None
    for before in (1, 37, 255, 256, 257, 4097, 100_000):
        X = np.vstack([rng.normal(size=(before, 4)) * 1e3, seg, rng.normal(size=(5, 4))])
        starts, counts = xp.asarray([0, before, before + 37]), xp.asarray([before, 37, 5])
        c = host(segment_cross(xp.asarray(X), xp.asarray(X), starts, counts, symmetric=True)[1])
        s = host(_reduce_sum_1d(xp.asarray(X), starts)[1])
        ref_cross = c if ref_cross is None else ref_cross
        ref_sum = s if ref_sum is None else ref_sum
        np.testing.assert_array_equal(c, ref_cross)
        np.testing.assert_array_equal(s, ref_sum)
    np.testing.assert_allclose(ref_sum, seg.sum(axis=0), rtol=1e-14)


def test_det_matmul_and_spd_solve_match_numpy_and_ignore_the_stack_size():
    from src.numerics import det_einsum, det_matmul, spd_solve

    rng = np.random.default_rng(3)
    for p, q in ((2, 2), (3, 1), (21, 1)):
        M = rng.normal(size=(300, p, p + 3))
        S = M @ np.swapaxes(M, 1, 2) + 1e-3 * np.eye(p)
        B = rng.normal(size=(300, p, q))
        prod, sol = host(det_matmul(xp.asarray(S), xp.asarray(B))), host(spd_solve(xp.asarray(S), xp.asarray(B)))
        np.testing.assert_allclose(prod, S @ B, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(sol, np.linalg.solve(S, B), rtol=1e-9, atol=1e-12)
        for g in (1, 7, 64):
            np.testing.assert_array_equal(host(det_matmul(xp.asarray(S[:g]), xp.asarray(B[:g]))), prod[:g])
            np.testing.assert_array_equal(host(spd_solve(xp.asarray(S[:g]), xp.asarray(B[:g]))), sol[:g])
        v = rng.normal(size=(300, p))
        np.testing.assert_allclose(host(det_einsum("kij,kj->ki", xp.asarray(S), xp.asarray(v))),
                                   np.einsum("kij,kj->ki", S, v), rtol=1e-12, atol=1e-12)  # fmt: skip
    assert np.all(np.isnan(host(spd_solve(xp.asarray([[[1.0, 2.0], [2.0, 1.0]]]), xp.ones((1, 2, 1)))))) or \
        not IS_CUPY_BACKEND  # fmt: skip  (an indefinite system is an invalid fit on the GPU)


def test_sort_by_block_then_value_matches_lexsort_with_ties():
    from src.numerics import sort_by_block_then_value

    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(1, 2000))
        blocks, values = np.sort(rng.integers(0, 30, n)), np.round(rng.normal(size=n), 1)
        np.testing.assert_array_equal(
            host(sort_by_block_then_value(xp.asarray(values), xp.asarray(blocks))),
            np.lexsort(np.stack([values, blocks])),
        )


def test_fused_stats_on_sorted_labels_with_empty_clusters_match_the_reference():
    """GPU contract: sorted labels (the batch layout), any clusters empty, including the first and last."""
    from src.numerics import run_fused_cross_cov, run_fused_stats

    rng = np.random.default_rng(7)
    K, d, fd = 8, 3, 2
    labels = np.sort(rng.choice([1, 2, 4, 5, 6], 500))
    X, F = rng.normal(size=(500, d)), rng.normal(size=(500, fd))
    mus, f_mus = rng.normal(size=(K, d)), rng.normal(size=(K, fd))
    C = host(run_fused_stats(xp.asarray(X), xp.asarray(mus), xp.asarray(labels), K, d, fd, None)[0])
    XF = host(run_fused_cross_cov(xp.asarray(X), xp.asarray(F), xp.asarray(mus), xp.asarray(f_mus), xp.asarray(labels), K, d, fd))
    for k in range(K):
        m = labels == k
        np.testing.assert_allclose(C[k], (X[m] - mus[k]).T @ (X[m] - mus[k]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(XF[k], (X[m] - mus[k]).T @ (F[m] - f_mus[k]), rtol=1e-12, atol=1e-12)
