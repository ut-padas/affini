"""Tests for adaptive_clustering_utils.cluster_errors on a hand-computed example."""

import numpy as np
import pytest

from src import _to_host, xp
from src.components.acceptance import cluster_errors
from src.components.modes import SurrogateMode
from src.config import ClusteringConfig

# Two clusters of 2 points with fd = 2: cluster 0 is fitted exactly, cluster 1
# is off by 0.5 in the first component of its second point.
F = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0], [4.0, -2.0]])
F_HAT = F.copy()
F_HAT[3, 0] += 0.5
STARTS, SIZES = np.array([0, 2]), np.array([2, 2])


def _errors(metric, atol, rtol):
    config = ClusteringConfig(
        mode=SurrogateMode.AFFINE_COV_HESS, atol=atol, rtol=rtol, error_metric=metric
    )
    out = cluster_errors(config, xp.asarray(F), xp.asarray(F_HAT), xp.asarray(STARTS), xp.asarray(SIZES))
    return tuple(_to_host(a) for a in out)


@pytest.mark.parametrize(
    "metric,abs_1,rel_1",
    [
        # cluster 1, component 0: RMS 0.5 / sqrt(2), relative 0.5 / ||(2, 4)||
        ("L2", 0.5 / np.sqrt(2), 0.5 / np.sqrt(4 + 16)),
        # cluster 1, component 0: max |diff| 0.5, relative to max |F| = 4 on the cluster
        ("Linf", 0.5, 0.5 / 4.0),
    ],
)
def test_error_values(metric, abs_1, rel_1):
    is_good, E_abs, E_rel = _errors(metric, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(E_abs, [[0.0, 0.0], [abs_1, 0.0]], rtol=1e-10)
    np.testing.assert_allclose(E_rel, [[0.0, 0.0], [rel_1, 0.0]], rtol=1e-10)  # TINY_EPS
    np.testing.assert_array_equal(is_good, [True, False])


def test_absolute_tolerance_accepts_small_errors():
    # RMS error of cluster 1 is 0.5 / sqrt(2); a looser atol accepts it under L2
    # even though the relative error is above rtol.
    is_good, _, _ = _errors("L2", atol=0.4, rtol=1e-12)
    np.testing.assert_array_equal(is_good, [True, True])
    is_good, _, _ = _errors("L2", atol=0.3, rtol=1e-12)
    np.testing.assert_array_equal(is_good, [True, False])
