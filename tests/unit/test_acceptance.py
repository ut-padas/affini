"""
The built-in acceptance criteria against independently written per-cluster
formulas (see the semantics in src/acceptance.py). These pin the current numerical
behaviour, including that 'Linf-global' compares its relative error with atol.
"""

import logging

import numpy as np
import pytest

from src import xp
from src.acceptance import ACCEPTANCE_CRITERIA, cluster_errors
from src.config import ClusteringConfig
from src.modes import SurrogateMode
from tests._helpers import host

EPS = 1e-12
rng = np.random.default_rng(7)
SIZES = np.array([4, 7, 1, 5])
STARTS = np.r_[0, np.cumsum(SIZES)[:-1]]
N, FD = int(SIZES.sum()), 3
F = rng.standard_normal((N, FD)) * np.array([1.0, 100.0, 0.01])
F[STARTS[1] : STARTS[1] + SIZES[1], 2] = 0.0  # a component that is zero on one cluster
F_HAT = F + rng.standard_normal((N, FD)) * np.array([0.05, 2.0, 1e-4])
F_MAX = np.max(np.abs(F), axis=0)


def _blocks():
    for s, n in zip(STARTS, SIZES, strict=True):
        yield F[s : s + n], F_HAT[s : s + n]


def reference(metric, atol, rtol):
    good, err = [], []
    for f, fh in _blocks():
        e = f - fh
        if metric == "L2":
            rel = np.sqrt(np.sum(e**2)) / (np.sqrt(np.sum(f**2)) + EPS)
            ok = rel < rtol or np.sqrt(np.sum(e**2) / len(f)) < atol
        elif metric == "Linf-local":
            rel = max(np.max(np.abs(e[:, j])) / max(np.max(np.abs(f[:, j])), EPS) for j in range(FD))
            ok = rel < rtol or np.max(np.abs(e)) < atol
        else:  # Linf-global: relative to max|F| of the initial data, compared with atol
            rel = np.max(np.abs(e) / (F_MAX + EPS))
            ok = rel < atol
        good.append(ok)
        err.append(rel)
    return np.array(good), np.array(err)


def run(metric, atol, rtol):
    crit = ACCEPTANCE_CRITERIA.create(metric, atol=atol, rtol=rtol)
    good, err = crit(xp.asarray(F), xp.asarray(F_HAT), xp.asarray(STARTS), xp.asarray(SIZES), xp.asarray(F_MAX))
    return host(good), host(err)


@pytest.mark.parametrize("metric", ["L2", "Linf-local", "Linf-global"])
@pytest.mark.parametrize("atol,rtol", [(1e-3, 0.5), (0.5, 1e-3), (0.05, 0.05), (0.0, 0.0), (10.0, 10.0)])
def test_matches_independent_formulas(metric, atol, rtol):
    good, err = run(metric, atol, rtol)
    ref_good, ref_err = reference(metric, atol, rtol)
    np.testing.assert_allclose(err, ref_err, rtol=1e-12)
    np.testing.assert_array_equal(good, ref_good)


@pytest.mark.parametrize("metric", ["L2", "Linf-local", "Linf-global"])
def test_comparisons_are_strict(metric):
    _, err = run(metric, 1.0, 1.0)
    tol = float(err[0])
    good, _ = run(metric, 0.0 if metric != "Linf-global" else tol, tol)
    assert not good[0]  # error == tolerance is rejected


def test_linf_global_uses_atol_not_rtol():
    _, err = run("Linf-global", 1.0, 1.0)
    between = float(np.median(err))
    good_loose_rtol, _ = run("Linf-global", atol=1e-9, rtol=1e9)
    good_loose_atol, _ = run("Linf-global", atol=1e9, rtol=1e-9)
    assert not good_loose_rtol.any() and good_loose_atol.all()
    good, _ = run("Linf-global", atol=between, rtol=0.0)
    np.testing.assert_array_equal(good, err < between)


def test_linf_global_with_unequal_tolerances_warns(caplog):
    with caplog.at_level(logging.WARNING):
        ClusteringConfig(mode=SurrogateMode.AFFINE_COV_HESS, atol=0.01, rtol=0.1)
    assert "rtol=0.1 is not used" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ClusteringConfig(mode=SurrogateMode.AFFINE_COV_HESS, atol=0.1, rtol=0.1)
        ClusteringConfig(mode=SurrogateMode.AFFINE_COV_HESS, atol=0.01, rtol=0.1, error_metric="L2")
    assert "not used" not in caplog.text


def test_cluster_errors_uses_the_configured_criterion():
    cfg = ClusteringConfig(mode=SurrogateMode.AFFINE_COV_HESS, atol=0.05, rtol=0.05, error_metric="Linf-local")
    good, err = cluster_errors(cfg, xp.asarray(F), xp.asarray(F_HAT), xp.asarray(STARTS), xp.asarray(SIZES), xp.asarray(F_MAX))
    ref_good, ref_err = reference("Linf-local", 0.05, 0.05)
    np.testing.assert_array_equal(host(good), ref_good)
    np.testing.assert_allclose(host(err), ref_err, rtol=1e-12)


def test_unknown_metric_is_rejected_by_the_config():
    with pytest.raises(ValueError, match="Unknown error metric"):
        ClusteringConfig(mode=SurrogateMode.AFFINE_COV_HESS, atol=0.1, rtol=0.1, error_metric="Linf")
