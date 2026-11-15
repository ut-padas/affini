"""Tests for src/components/splitting.py."""

import numpy as np
import pytest

import src.engine.driver as driver
from src import to_host, xp
from src.components.modes import SurrogateMode
from src.components.splitting import (
    CovarianceHessianSplit,
    CovarianceSplit,
    ExactHessianSplit,
    GradientLSQHessianSplit,
    SplitData,
    split_strategy_for_mode,
)
from src.config import TrainingConfig
from src.core.evaluation import Evaluator

# f(x, y) = 0.01 x^2 + 100 y^2 on points that are long in x but short in y.
HESS = np.diag([0.02, 200.0])


def f(X):
    return (0.01 * X[:, 0] ** 2 + 100.0 * X[:, 1] ** 2)[:, None]


def g(X):
    return np.stack([0.02 * X[:, 0], 200.0 * X[:, 1]], axis=1)[:, None, :]


def _data(counts, seed=0, with_g=True):
    rng = np.random.default_rng(seed)
    counts = np.asarray(counts)
    X = np.concatenate(
        [
            rng.uniform([-2, -0.5], [2, 0.5], (n, 2)) + 10 * k
            for k, n in enumerate(counts)
        ]
    )
    labels = np.repeat(np.arange(len(counts)), counts)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    centroids = np.add.reduceat(X, starts) / counts[:, None]
    return SplitData(
        X=xp.asarray(X),
        F=xp.asarray(f(X)),
        G=xp.asarray(g(X)) if with_g else None,
        labels=xp.asarray(labels),
        counts=xp.asarray(counts),
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
    counts = [1200, 900, 1501]
    data = _data(counts)
    order, child = strategy.split(data)
    order, child = to_host(order), to_host(child)
    assert sorted(order.tolist()) == list(range(sum(counts)))  # a permutation
    labels = to_host(data.labels)[order]
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
    X = to_host(data.X)[to_host(order)]
    low, high = X[: int(to_host(child)[0])], X[int(to_host(child)[0]) :]
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
    config = TrainingConfig(
        mode=SurrogateMode.AFFINE_EXACT_HESS, atol=1e-3, rtol=1e-3
    )
    with pytest.raises(ValueError, match="requires an exact Hessian"):
        driver.refine(xp.asarray(np.zeros((10, 2))), config, Evaluator(f_fn=f))


@pytest.mark.parametrize("mode", list(SurrogateMode))
def test_every_mode_has_a_split_and_its_requirements_hold(mode):
    config = TrainingConfig(mode=mode, atol=1e-3, rtol=1e-3)
    split = split_strategy_for_mode(mode, config.min_pts_per_cluster)
    if split.needs_point_grad:
        assert config.requires_grad_everywhere
    if split.needs_centroid_grad:
        assert config.requires_grad_centroids
    assert split.needs_hessian_fn == ("exact_hess" in mode.value)
