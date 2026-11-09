"""Edge cases of prediction and early validation of the configuration."""

import numpy as np
import pytest

from src import xp
from src.config import ClusteringConfig
from src.modes import SurrogateMode
from src.surrogate import AFINNISurrogate
from tests._helpers import assert_backend_equal, host

rng = np.random.default_rng(9)
X = rng.uniform(-1, 1, (2_000, 2))


def f(x):
    return xp.stack((1 + x[:, 0] ** 2 + 2 * x[:, 1] ** 2, x[:, 0] - x[:, 1]), axis=1)


@pytest.fixture(scope="module")
def surrogate():
    s = AFINNISurrogate()
    s.train(
        xp.asarray(X), f_fn=f, ATOL=0.05, RTOL=0.05, ERROR_METRIC="Linf",
        NUM_INITIAL_CLUSTERS=4, MIN_PTS_PER_CLUSTER=32, USE_WORK_STEALING=False,
    )  # fmt: skip
    return s


def test_zero_queries(surrogate):
    pred, mask = surrogate.predict(xp.empty((0, 2)), k_nn=4)
    assert host(pred).shape == (0, 2) and host(mask).shape == (0,)


def test_one_query_matches_the_batched_result(surrogate):
    Q = rng.uniform(-1, 1, (50, 2))
    pred_all, mask_all = surrogate.predict(xp.asarray(Q), k_nn=4)
    pred_one, mask_one = surrogate.predict(xp.asarray(Q[7:8]), k_nn=4)
    assert_backend_equal(pred_one, host(pred_all)[7:8])
    assert host(mask_one)[0] == host(mask_all)[7]


def test_batch_size_does_not_change_predictions(surrogate):
    Q = xp.asarray(rng.uniform(-1, 1, (101, 2)))
    a = surrogate.predict(Q, k_nn=4, batch_size=5000)
    b = surrogate.predict(Q, k_nn=4, batch_size=1)
    c = surrogate.predict(Q, k_nn=4, batch_size=13)
    for other in (b, c):
        assert_backend_equal(other[0], a[0])
        np.testing.assert_array_equal(host(other[1]), host(a[1]))


def test_more_neighbours_than_leaves(surrogate):
    Q = xp.asarray(rng.uniform(-1, 1, (20, 2)))
    n = surrogate.N_src
    pred, mask = surrogate.predict(Q, k_nn=n + 5, coarse_factor=n + 5)
    ref, ref_mask = surrogate.predict(Q, k_nn=n, coarse_factor=n)
    assert_backend_equal(pred, ref)
    np.testing.assert_array_equal(host(mask), host(ref_mask))


def test_k_larger_than_coarse_factor_is_rejected(surrogate):
    with pytest.raises(ValueError, match="coarse_factor"):
        surrogate.predict(xp.asarray(X[:3]), k_nn=11, coarse_factor=10)


def test_per_component_tolerances_are_used_and_must_match_the_outputs():
    s = AFINNISurrogate()
    opts = dict(f_fn=f, NUM_INITIAL_CLUSTERS=4, MIN_PTS_PER_CLUSTER=32, USE_WORK_STEALING=False)
    s.train(xp.asarray(X), ATOL=(0.05, 0.01), RTOL=None, **opts)
    E_abs = host(s.E_cluster_abs_src)
    assert E_abs.shape == (s.N_src, 2) and np.all(E_abs < [0.05, 0.01])
    assert s.config["atol"] == (0.05, 0.01) and s.config["rtol"] is None
    with pytest.raises(ValueError, match="atol has 3 components but the function has 2 outputs"):
        AFINNISurrogate().train(xp.asarray(X), ATOL=(0.05, 0.05, 0.05), **opts)
    with pytest.raises(ValueError, match="Give atol, rtol or both"):
        AFINNISurrogate().train(xp.asarray(X), **opts)


def test_untrained_surrogate_refuses_to_predict():
    with pytest.raises(RuntimeError, match="not trained"):
        AFINNISurrogate().predict(xp.asarray(X[:3]))


@pytest.mark.parametrize(
    "field,value",
    [
        ("atol", float("nan")),
        ("rtol", -1.0),
        ("q_reg", float("inf")),
        ("batch_size", 0),
        ("num_initial_clusters", 0),
        ("min_pts_per_cluster", 0),
        ("max_levels", -1),
        ("max_clusters_est", 0),
        ("num_initial_clusters", 2.5),
        ("batch_size", True),
    ],
)
def test_invalid_settings_fail_early(field, value):
    kw = dict(mode=SurrogateMode.AFFINE_COV_HESS, atol=0.1, rtol=0.1)
    kw[field] = value
    with pytest.raises(ValueError, match=field):
        ClusteringConfig(**kw)


def test_numpy_scalars_and_mode_strings_are_accepted():
    cfg = ClusteringConfig(
        mode="taylor_cov_hess", atol=np.float32(0.1), rtol=np.float64(0.1),
        num_initial_clusters=np.int64(4), max_levels=np.int32(3),
    )  # fmt: skip
    assert cfg.mode is SurrogateMode.TAYLOR_COV_HESS
