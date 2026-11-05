"""Tests for src/sampling.py: samplers and cluster augmentation."""

import numpy as np
import pytest

from src import _to_host, xp
from src.adaptive_clustering_utils import Evaluator
from src.sampling import (
    SAMPLERS,
    ClusterPoints,
    ConvexCombinationSampler,
    augment_clusters,
    augmentation_targets,
    make_sampler,
)

# Three well-separated 2-D clusters (sizes 1, 4, 6), cluster k centred at (100k, 100k).
COUNTS = np.array([1, 4, 6])
_rng = np.random.default_rng(0)
X = np.concatenate(
    [100 * k + _rng.uniform(-1, 1, (n, 2)) for k, n in enumerate(COUNTS)]
)


def _clusters():
    return ClusterPoints(xp.asarray(X), xp.asarray(COUNTS))


@pytest.mark.parametrize("name", sorted(SAMPLERS))
def test_sampler_contract(name):
    """Every registered sampler: right shape, grouped by cluster, inside its cluster."""
    sampler = make_sampler(name)
    n_new = np.array([3, 0, 5])
    new = _to_host(
        sampler.sample(_clusters(), xp.asarray(n_new), xp.random.default_rng(1))
    )
    assert new.shape == (n_new.sum(), 2)

    owner = np.repeat(np.arange(len(COUNTS)), n_new)
    starts = np.r_[0, np.cumsum(COUNTS)[:-1]]
    for k in range(len(COUNTS)):
        pts = X[starts[k] : starts[k] + COUNTS[k]]
        mine = new[owner == k]
        assert np.all(mine >= pts.min(axis=0) - 1e-12)
        assert np.all(mine <= pts.max(axis=0) + 1e-12)


@pytest.mark.parametrize("name", sorted(SAMPLERS))
def test_sampler_is_reproducible_and_handles_no_points(name):
    sampler = make_sampler(name)
    n_new = xp.asarray(np.array([2, 2, 2]))
    a = sampler.sample(_clusters(), n_new, xp.random.default_rng(7))
    b = sampler.sample(_clusters(), n_new, xp.random.default_rng(7))
    np.testing.assert_array_equal(_to_host(a), _to_host(b))

    none = sampler.sample(
        _clusters(), xp.zeros(3, dtype=xp.int64), xp.random.default_rng(7)
    )
    assert none.shape == (0, 2)


def test_convex_combination_single_point_cluster_reuses_its_point():
    new = ConvexCombinationSampler().sample(
        _clusters(), xp.asarray(np.array([4, 0, 0])), xp.random.default_rng(3)
    )
    np.testing.assert_allclose(_to_host(new), np.repeat(X[:1], 4, axis=0))


def test_augmentation_targets():
    counts = xp.asarray(np.array([1, 50, 150, 200, 500]))
    np.testing.assert_array_equal(
        _to_host(augmentation_targets(counts, min_pts_per_cluster=100)),
        [200, 200, 300, 200, 500],
    )


@pytest.mark.parametrize("with_grad", [False, True])
def test_augment_clusters_layout_and_evaluation(with_grad):
    f = lambda x: x[:, :1] ** 2  # noqa: E731
    g = lambda x: (2 * x[:, :1])[:, :, None] * xp.asarray([[[1.0, 0.0]]])  # noqa: E731
    Xd = xp.asarray(X)
    F = f(Xd)
    G = g(Xd) if with_grad else None
    targets = xp.asarray(np.array([3, 4, 8]))
    evaluator = Evaluator(f_fn=f, g_fn=g)

    aug = augment_clusters(
        Xd,
        F,
        G,
        xp.asarray(COUNTS),
        targets,
        ConvexCombinationSampler(),
        evaluator,
        xp.random.default_rng(0),
    )
    Xa, labels = _to_host(aug.X), _to_host(aug.labels)
    np.testing.assert_array_equal(labels, np.repeat([0, 1, 2], [3, 4, 8]))
    # Each block starts with that cluster's original points, in order.
    for k, (s_out, s_in) in enumerate(
        zip([0, 3, 7], np.r_[0, np.cumsum(COUNTS)[:-1]], strict=True)
    ):
        np.testing.assert_array_equal(
            Xa[s_out : s_out + COUNTS[k]], X[s_in : s_in + COUNTS[k]]
        )
    assert aug.X_new.shape == (int((np.array([3, 4, 8]) - COUNTS).sum()), 2)
    # Function values (and gradients) are correct for every row, new ones included.
    np.testing.assert_allclose(_to_host(aug.F), _to_host(f(aug.X)))
    if with_grad:
        np.testing.assert_allclose(_to_host(aug.G), _to_host(g(aug.X)))
    else:
        assert aug.G is None
    assert evaluator.n_evals == aug.X_new.shape[0]


def test_augment_clusters_without_new_points():
    aug = augment_clusters(
        xp.asarray(X),
        xp.asarray(X[:, :1]),
        None,
        xp.asarray(COUNTS),
        xp.asarray(COUNTS),
        ConvexCombinationSampler(),
        Evaluator(f_fn=lambda x: x[:, :1]),
        xp.random.default_rng(0),
    )
    assert aug.X_new is None
    np.testing.assert_array_equal(_to_host(aug.X), X)


def test_unknown_sampler_name():
    with pytest.raises(ValueError, match="Unknown sampling method"):
        make_sampler("does_not_exist")


def test_config_rejects_unknown_sampling_method():
    from src.adaptive_clustering_utils import ClusteringConfig, SurrogateMode

    with pytest.raises(ValueError, match="Unknown sampling method"):
        ClusteringConfig(
            mode=SurrogateMode.NEAREST_NEIGHBOR,
            atol=1e-3,
            rtol=1e-3,
            sampling_method="typo",
        )
    ok = ClusteringConfig(mode=SurrogateMode.NEAREST_NEIGHBOR, atol=1e-3, rtol=1e-3)
    assert ok.sampling_method in SAMPLERS
