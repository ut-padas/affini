"""Tests for adaptive_clustering_utils.cluster_errors."""

import numpy as np
import pytest

from src import _to_host, xp
from src.adaptive_clustering_utils import (
    ClusteringConfig,
    SurrogateMode,
    cluster_errors,
)

# Two clusters of 2 points with fd = 2: cluster 0 is fitted exactly, cluster 1
# is off by 0.5 in the first component of its second point.
F = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0], [4.0, -2.0]])
F_HAT = F.copy()
F_HAT[3, 0] += 0.5
STARTS, SIZES = np.array([0, 2]), np.array([2, 2])
F_MAX = np.abs(F).max(axis=0)


def _errors(metric, atol, rtol):
    config = ClusteringConfig(
        mode=SurrogateMode.AFFINE_COV_HESS, atol=atol, rtol=rtol, error_metric=metric
    )
    is_good, E = cluster_errors(
        config,
        xp.asarray(F),
        xp.asarray(F_HAT),
        xp.asarray(STARTS),
        xp.asarray(SIZES),
        xp.asarray(F_MAX),
    )
    return _to_host(is_good), _to_host(E)


@pytest.mark.parametrize(
    "metric,expected_E1",
    [
        # relative L2 over the cluster: 0.5 / ||F_cluster||_F
        ("L2", 0.5 / np.sqrt(4 + 1 + 16 + 4)),
        # max over points of |diff| / max|F| over the whole batch, per component
        ("Linf-global", 0.5 / 4.0),
        # max over components of max|diff| / max|F| on the cluster
        ("Linf-local", 0.5 / 4.0),
    ],
)
def test_error_values(metric, expected_E1):
    is_good, E = _errors(metric, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(E, [0.0, expected_E1], rtol=1e-10)  # TINY_EPS
    np.testing.assert_array_equal(is_good, [True, False])


def test_absolute_tolerance_accepts_small_errors():
    # RMS error of cluster 1 is 0.5 / sqrt(2); a looser atol accepts it under L2
    # even though the relative error is above rtol.
    is_good, _ = _errors("L2", atol=0.4, rtol=1e-12)
    np.testing.assert_array_equal(is_good, [True, True])
    is_good, _ = _errors("L2", atol=0.3, rtol=1e-12)
    np.testing.assert_array_equal(is_good, [True, False])
