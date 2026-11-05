"""Tests for the adaptive_clustering(X, config, evaluator) interface."""

import dataclasses

import numpy as np
import pytest

import src.adaptive_clustering as ac
from src.adaptive_clustering_utils import ClusteringConfig, Evaluator, SurrogateMode


def _x():
    return ac.xp.asarray(np.linspace(-1, 1, 128).reshape(-1, 1))


def test_config_is_keyword_only():
    with pytest.raises(TypeError):
        ClusteringConfig(SurrogateMode.NEAREST_NEIGHBOR, 1e-3, 1e-3)  # type: ignore[misc]


def test_caller_config_is_not_modified(monkeypatch):
    from src.backend import _SerialMPI

    monkeypatch.setattr(ac, "MPI", _SerialMPI)
    # A huge batch size is always capped to fit memory inside adaptive_clustering().
    config = ClusteringConfig(
        mode=SurrogateMode.AFFINE_COV_NO_HESS,
        atol=1e-4,
        rtol=1e-4,
        num_initial_clusters=1,
        min_pts_per_cluster=4,
        max_levels=1,
        batch_size=10**12,
    )
    before = dataclasses.asdict(config)
    result = ac.adaptive_clustering(_x(), config, Evaluator(f_fn=lambda x: x**2))
    assert result is not None
    assert dataclasses.asdict(config) == before


def test_mode_needing_gradients_without_gradient_raises():
    config = ClusteringConfig(mode=SurrogateMode.TAYLOR_COV_HESS, atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError, match="requires gradient information"):
        ac.adaptive_clustering(_x(), config, Evaluator(f_fn=lambda x: x**2))


# The gradients each mode needs, as the hand-written mode lists gave them before
# they were derived from the local model and split strategy.
_GRAD_NEEDS = [
    # mode, everywhere, centroids, none
    (SurrogateMode.NEAREST_NEIGHBOR, False, False, True),
    (SurrogateMode.AFFINE_COV_NO_HESS, False, False, True),
    (SurrogateMode.TAYLOR_COV_NO_HESS, False, True, False),
    (SurrogateMode.AFFINE_COV_HESS, False, False, True),
    (SurrogateMode.AFFINE_COV_HESS_EXACT_GRAD, False, True, False),
    (SurrogateMode.AFFINE_EXACT_HESS, False, False, True),
    (SurrogateMode.TAYLOR_COV_HESS, False, True, False),
    (SurrogateMode.TAYLOR_EXACT_HESS, False, True, False),
    (SurrogateMode.TAYLOR_LSQ_HESS, True, True, False),
]


@pytest.mark.parametrize("mode,everywhere,centroids,none", _GRAD_NEEDS)
def test_gradient_needs_follow_model_and_split(mode, everywhere, centroids, none):
    config = ClusteringConfig(mode=mode, atol=1e-3, rtol=1e-3)
    assert config.requires_grad_everywhere == everywhere
    assert config.requires_grad_centroids == centroids
    assert config.requires_no_grads == none


def test_every_mode_is_covered_by_the_gradient_table():
    assert {row[0] for row in _GRAD_NEEDS} == set(SurrogateMode)
