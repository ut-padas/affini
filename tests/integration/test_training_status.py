"""Convergence status: a run stopped by MAX_LEVELS must not look fully trained."""

import warnings

import numpy as np
import pytest

import src.adaptive_clustering as ac
from src import xp
from src.backend import _SerialMPI
from src.config import ClusteringConfig
from src.evaluation import Evaluator
from src.modes import SurrogateMode
from src.surrogate import AFINNISurrogate
from tests._helpers import host

X = np.random.default_rng(5).uniform(-2, 2, (3_000, 2))


def f(x):
    return (1 + x[:, 0] ** 2 + 4 * x[:, 1] ** 4)[:, None]


def train(**kw):
    s = AFINNISurrogate()
    opts = dict(
        f_fn=f, ATOL=0.01, RTOL=0.01, ERROR_METRIC="Linf", NUM_INITIAL_CLUSTERS=8,
        MIN_PTS_PER_CLUSTER=40, USE_WORK_STEALING=False,
    )  # fmt: skip
    opts.update(kw)
    s.train(xp.asarray(X), **opts)
    return s


def test_converged_run_reports_no_unresolved_work():
    s = train()
    assert s.converged is True
    assert s.num_unresolved_clusters == 0 and s.num_unresolved_pts == 0
    assert s.profiling_stats["converged"] is True


def test_partially_refined_run_warns_and_accounts_for_unresolved_points():
    with pytest.warns(RuntimeWarning, match="MAX_LEVELS=2"):
        s = train(MAX_LEVELS=2, ATOL=0.05, RTOL=0.05)
    assert s.converged is False and s.N_src > 0
    assert s.num_unresolved_clusters > 0 and s.num_unresolved_pts > 0
    ps = s.profiling_stats
    # Accepted + unresolved points = initial + generated points.
    assert int(host(s.counts_src).sum()) + s.num_unresolved_pts == ps["N0"] + ps["num_generated_pts"]


def test_run_without_any_accepted_cluster_raises():
    with pytest.raises(RuntimeError, match="accepted no clusters"):
        train(MAX_LEVELS=0, ATOL=1e-9, RTOL=1e-9)


def test_failed_retrain_leaves_the_previous_model_intact():
    s = train()
    leaves, pred = s.N_src, host(s.predict(xp.asarray(X[:50]))[0])
    with pytest.raises(RuntimeError):
        s.train(xp.asarray(X), f_fn=f, ATOL=1e-9, RTOL=1e-9, MAX_LEVELS=0, NUM_INITIAL_CLUSTERS=8, USE_WORK_STEALING=False)
    assert s.trained and s.N_src == leaves
    np.testing.assert_array_equal(host(s.predict(xp.asarray(X[:50]))[0]), pred)


def test_low_level_result_reports_status(monkeypatch):
    monkeypatch.setattr(ac, "MPI", _SerialMPI)
    cfg = ClusteringConfig(
        mode=SurrogateMode.AFFINE_COV_HESS, atol=0.01, rtol=0.01, error_metric="Linf",
        num_initial_clusters=8, min_pts_per_cluster=40, max_levels=1, use_work_stealing=False,
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ac.adaptive_clustering(xp.asarray(X), cfg, Evaluator(f_fn=f))
    assert res["converged"] is False and res["num_unresolved_clusters"] > 0
    assert res["profiling_stats"]["num_unresolved_pts"] == res["num_unresolved_pts"]
