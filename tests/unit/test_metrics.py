"""The tolerance ratio: <= 1 exactly where the error metrics count a point as within tolerance."""

import numpy as np
import pytest

from src import xp
from src.surrogate import compute_error_metrics, tolerance_ratio
from tests._helpers import host


@pytest.mark.parametrize(("fd", "atol", "rtol"), [(1, 0.1, None), (1, None, 0.1), (3, 0.05, 0.1), (2, (0.1, 0.01), None)])
def test_ratio_at_most_one_is_the_within_tolerance_count(fd, atol, rtol):
    rng = np.random.default_rng(fd)
    gt = rng.normal(size=(4_000, fd))
    gt[:20] = 0.0  # relative tolerance at f = 0 accepts only an exact value
    pred = gt + rng.normal(scale=0.05, size=gt.shape) * rng.uniform(0, 3, size=(4_000, 1))
    pred[:5] = gt[:5]
    mask = rng.uniform(size=4_000) < 0.1
    m = compute_error_metrics(xp.asarray(pred), xp.asarray(gt), xp.asarray(mask), atol=atol, rtol=rtol)
    r = host(tolerance_ratio(xp.asarray(pred), xp.asarray(gt), atol, rtol))
    assert int(np.sum(r[~mask] <= 1)) == m["n_in_good"]
    h = m["tolerance_ratio"]
    assert sum(h["in"]) == 4_000 - int(mask.sum()) and sum(h["out"]) == int(mask.sum())
    assert h["max_in"] == pytest.approx(r[~mask].max())


def test_ratio_is_in_units_of_the_looser_tolerance_and_nan_is_worst():
    gt = xp.asarray([[1.0], [10.0], [0.0], [1.0]])
    pred = xp.asarray([[1.3], [11.0], [0.0], [np.nan]])
    # |e| / atol versus rel / rtol: 0.3 / 0.1 = 3 vs 0.3 / 0.2 = 1.5; 1 / 0.1 = 10 vs 0.1 / 0.2 = 0.5
    np.testing.assert_allclose(host(tolerance_ratio(pred, gt, 0.1, 0.2)), [1.5, 0.5, 0.0, np.inf])
