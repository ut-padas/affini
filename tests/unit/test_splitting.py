"""Tests for src/splitting.py."""

import numpy as np
import pytest

import src.adaptive_clustering as ac
from src import _to_host, xp
from src.adaptive_clustering_utils import ClusteringConfig, Evaluator, SurrogateMode
from src.splitting import (
    CovarianceHessianSplit,
    CovarianceSplit,
    ExactHessianSplit,
    GradientLSQHessianSplit,
    SplitData,
    split_strategy_for_mode,
)

# f(x, y) = 0.01 x^2 + 100 y^2 on points that are long in x but short in y.
HESS = np.diag([0.02, 200.0])


def f(X):
    return (0.01 * X[:, 0] ** 2 + 100.0 * X[:, 1] ** 2)[:, None]


def g(X):
    return np.stack([0.02 * X[:, 0], 200.0 * X[:, 1]], axis=1)[:, None, :]


def _data(sizes, seed=0, with_g=True):
    rng = np.random.default_rng(seed)
    sizes = np.asarray(sizes)
    X = np.concatenate(
        [
            rng.uniform([-2, -0.5], [2, 0.5], (n, 2)) + 10 * k
            for k, n in enumerate(sizes)
        ]
    )
    labels = np.repeat(np.arange(len(sizes)), sizes)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    centroids = np.add.reduceat(X, starts) / sizes[:, None]
    return SplitData(
        X=xp.asarray(X),
        F=xp.asarray(f(X)),
        G=xp.asarray(g(X)) if with_g else None,
        labels=xp.asarray(labels),
        counts=xp.asarray(sizes),
        centroids=xp.asarray(centroids),
        f_c=xp.asarray(f(centroids)),
        g_c=xp.asarray(g(centroids)),
        hessian_fn=lambda c: xp.asarray(np.repeat(HESS[None], len(c), axis=0)),
    )


STRATEGIES = [
    CovarianceSplit(),
    CovarianceHessianSplit("centroid", min_pts_per_cluster=300),
    CovarianceHessianSplit("local_lsq", min_pts_per_cluster=300),
    ExactHessianSplit(),
    GradientLSQHessianSplit(),
]


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: f"{s.name}")
def test_split_contract(strategy):
    sizes = [1200, 900, 1501]
    data = _data(sizes)
    order, child = strategy.split(data)
    order, child = _to_host(order), _to_host(child)
    assert sorted(order.tolist()) == list(range(sum(sizes)))  # a permutation
    labels = _to_host(data.labels)[order]
    assert np.all(np.diff(labels) >= 0)  # clusters stay contiguous, in label order
    np.testing.assert_array_equal(child, [600, 600, 450, 450, 750, 751])


@pytest.mark.parametrize(
    "strategy,axis",
    [
        (CovarianceSplit(), 0),  # follows the data: the long x direction
        (ExactHessianSplit(), 1),  # follows the curvature: y
        (CovarianceHessianSplit("centroid", 300), 1),
        (GradientLSQHessianSplit(), 1),
    ],
    ids=lambda v: getattr(v, "name", str(v)),
)
def test_split_direction_follows_curvature(strategy, axis):
    """Figure 2 of the paper: Hessian-informed splits cut across the high-curvature axis."""
    data = _data([2000])
    order, child = strategy.split(data)
    X = _to_host(data.X)[_to_host(order)]
    low, high = X[: int(_to_host(child)[0])], X[int(_to_host(child)[0]) :]
    gap = np.abs(high.mean(axis=0) - low.mean(axis=0)) / X.std(axis=0)
    assert gap[axis] > 1.5 and gap[1 - axis] < 0.2, gap


def test_missing_inputs_raise():
    with pytest.raises(RuntimeError, match="per-point gradients"):
        GradientLSQHessianSplit().split(_data([500], with_g=False))
    data = _data([500])
    data.hessian_fn = None
    with pytest.raises(RuntimeError, match="Hessian function"):
        ExactHessianSplit().split(data)


def test_exact_hessian_mode_without_hessian_fn_fails_up_front():
    config = ClusteringConfig(
        mode=SurrogateMode.AFFINE_EXACT_HESS, atol=1e-3, rtol=1e-3
    )
    with pytest.raises(ValueError, match="requires an exact Hessian"):
        ac.adaptive_clustering(xp.asarray(np.zeros((10, 2))), config, Evaluator(f_fn=f))


@pytest.mark.parametrize("mode", list(SurrogateMode))
def test_every_mode_has_a_split_and_its_requirements_hold(mode):
    config = ClusteringConfig(mode=mode, atol=1e-3, rtol=1e-3)
    split = split_strategy_for_mode(mode, config.min_pts_per_cluster)
    if split.needs_point_grad:
        assert config.requires_grad_everywhere
    if split.needs_centroid_grad:
        assert config.requires_grad_centroids
    assert split.needs_hessian_fn == ("exact_hess" in mode.value)
