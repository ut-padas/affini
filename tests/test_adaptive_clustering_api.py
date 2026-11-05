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
