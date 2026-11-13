"""Cut rules of SplitStrategy (median / sign) and the split output contract."""

import numpy as np
import pytest

from src import xp
from src.components.splitting import (
    CovarianceSplit,
    SplitData,
    median_split_sizes,
    sign_split_sizes,
)
from tests._helpers import host


def test_sign_cut_counts_negative_scores_per_cluster():
    counts = xp.asarray(np.array([4, 3, 5]))
    scores = xp.asarray(np.array([-2.0, -1.0, 0.5, 3.0, -0.1, 0.0, 0.2, -5.0, -4.0, -3.0, -1.0, 1.0]))
    np.testing.assert_array_equal(host(sign_split_sizes(counts, scores)), [2, 2, 1, 2, 4, 1])


def test_sign_cut_falls_back_to_the_median_when_one_side_is_empty():
    counts = xp.asarray(np.array([3, 4]))
    scores = xp.asarray(np.array([1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0]))
    np.testing.assert_array_equal(host(sign_split_sizes(counts, scores)), [1, 2, 2, 2])


def test_unknown_cut_rule():
    with pytest.raises(ValueError, match="Unknown cut rule"):
        CovarianceSplit(cut="mean")


def _split_data(rng, sizes, d=2):
    X = np.concatenate([rng.standard_normal((n, d)) * [3.0, 0.5] + 10 * k for k, n in enumerate(sizes)])
    labels = np.repeat(np.arange(len(sizes)), sizes)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    centroids = np.add.reduceat(X, starts) / np.asarray(sizes)[:, None]
    return SplitData(
        X=xp.asarray(X), F=xp.zeros((len(X), 1)), G=None, labels=xp.asarray(labels),
        counts=xp.asarray(np.asarray(sizes)), centroids=xp.asarray(centroids), f_c=xp.zeros((len(sizes), 1)), g_c=None,
    )  # fmt: skip


@pytest.mark.parametrize("cut", ["median", "sign"])
def test_split_returns_a_permutation_within_clusters_and_valid_children(cut):
    rng = np.random.default_rng(0)
    sizes = [9, 1, 20, 2]
    data = _split_data(rng, sizes)
    order, child = CovarianceSplit(cut=cut).split(data)
    order, child, labels = host(order), host(child), host(data.labels)
    assert sorted(order.tolist()) == list(range(sum(sizes)))  # bijection
    np.testing.assert_array_equal(labels[order], labels)  # clusters stay contiguous, in order
    assert child.shape == (2 * len(sizes),) and np.all(child >= 0)
    np.testing.assert_array_equal(child[0::2] + child[1::2], sizes)
    if cut == "median":
        np.testing.assert_array_equal(child, host(median_split_sizes(data.counts)))


def test_sign_cut_splits_through_the_centroid_along_the_principal_axis():
    # The children lie on either side of the hyperplane through the centroid
    # orthogonal to the principal axis of the cluster's covariance.
    rng = np.random.default_rng(1)
    data = _split_data(rng, [31])
    order, child = CovarianceSplit(cut="sign").split(data)
    X, c = host(data.X)[host(order)], host(data.centroids)[0]
    v = np.linalg.eigh(np.cov((X - c).T, bias=True))[1][:, -1]
    side = (X - c) @ v
    n_low = int(host(child)[0])
    assert 0 < n_low < len(X)
    low, high = side[:n_low], side[n_low:]
    assert (np.all(low < 0) and np.all(high >= 0)) or (np.all(low > 0) and np.all(high <= 0))
