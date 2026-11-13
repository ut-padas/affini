"""
The built-in acceptance criteria against independently written per-cluster,
per-component formulas (see the semantics in src/components/acceptance.py), and how scalar,
per-component and omitted tolerances are interpreted.
"""

import numpy as np
import pytest

from src import xp
from src.components.acceptance import ACCEPTANCE_CRITERIA, cluster_errors
from src.components.modes import SurrogateMode
from src.config import TrainingConfig
from tests._helpers import host

EPS = 1e-12
rng = np.random.default_rng(7)
SIZES = np.array([4, 7, 1, 5])
STARTS = np.r_[0, np.cumsum(SIZES)[:-1]]
N, FD = int(SIZES.sum()), 3
F = rng.standard_normal((N, FD)) * np.array([1.0, 100.0, 0.01])
F[STARTS[1] : STARTS[1] + SIZES[1], 2] = 0.0  # a component that is zero on one cluster
F_HAT = F + rng.standard_normal((N, FD)) * np.array([0.05, 2.0, 1e-4])


def reference_errors(metric):
    """(K, FD) absolute and relative errors, one cluster and component at a time."""
    e_abs, e_rel = np.zeros((len(SIZES), FD)), np.zeros((len(SIZES), FD))
    for k, (s, n) in enumerate(zip(STARTS, SIZES, strict=True)):
        for j in range(FD):
            f, e = F[s : s + n, j], F[s : s + n, j] - F_HAT[s : s + n, j]
            if metric == "L2":
                e_abs[k, j] = np.sqrt(np.sum(e**2) / n)
                e_rel[k, j] = np.sqrt(np.sum(e**2)) / (np.sqrt(np.sum(f**2)) + EPS)
            else:
                e_abs[k, j] = np.max(np.abs(e))
                e_rel[k, j] = e_abs[k, j] / max(np.max(np.abs(f)), EPS)
    return e_abs, e_rel


def reference_good(metric, atol, rtol):
    e_abs, e_rel = reference_errors(metric)
    off = np.full(FD, -np.inf)
    a = off if atol is None else np.broadcast_to(atol, FD)
    r = off if rtol is None else np.broadcast_to(rtol, FD)
    return np.all((e_rel < r) | (e_abs < a), axis=1)


def run(metric, atol, rtol):
    crit = ACCEPTANCE_CRITERIA.create(metric, atol=atol, rtol=rtol)
    good, e_abs, e_rel = crit(xp.asarray(F), xp.asarray(F_HAT), xp.asarray(STARTS), xp.asarray(SIZES))
    return host(good), host(e_abs), host(e_rel)


@pytest.mark.parametrize("metric", ["L2", "Linf"])
@pytest.mark.parametrize(
    "atol,rtol",
    [(1e-3, 0.5), (0.5, 1e-3), (0.05, 0.05), (0.0, 0.0), (10.0, 10.0), (None, 0.1), (0.1, None),
     ((0.1, 5.0, 1e-3), None), (None, (0.5, 0.05, 1.0)), ((0.1, 5.0, 1e-3), (0.5, 0.05, 1.0))],
)  # fmt: skip
def test_matches_independent_formulas(metric, atol, rtol):
    good, e_abs, e_rel = run(metric, atol, rtol)
    ref_abs, ref_rel = reference_errors(metric)
    np.testing.assert_allclose(e_abs, ref_abs, rtol=1e-12)
    np.testing.assert_allclose(e_rel, ref_rel, rtol=1e-12)
    np.testing.assert_array_equal(good, reference_good(metric, atol, rtol))


@pytest.mark.parametrize("metric", ["L2", "Linf"])
def test_scalar_tolerance_equals_a_constant_vector(metric):
    for atol, rtol in ((0.05, 0.2), (None, 0.2), (0.05, None)):
        vec = [None if t is None else (t,) * FD for t in (atol, rtol)]
        np.testing.assert_array_equal(run(metric, atol, rtol)[0], run(metric, *vec)[0])


@pytest.mark.parametrize("metric", ["L2", "Linf"])
def test_one_failing_component_rejects_the_cluster(metric):
    _, e_abs, _ = run(metric, 1.0, 1.0)
    loose = tuple(e_abs.max(axis=0) * 2)
    assert run(metric, loose, None)[0].all()
    tight = (loose[0], loose[1], 0.0)  # only the last component fails
    assert not run(metric, tight, None)[0].any()


def test_components_may_pass_on_different_tests():
    # Cluster 0 with only component 0 passing relative and only the others passing absolute.
    _, e_abs, e_rel = run("Linf", 1.0, 1.0)
    rtol = (e_rel[0, 0] * 1.01, 0.0, 0.0)
    atol = (0.0, e_abs[0, 1] * 1.01, e_abs[0, 2] * 1.01)
    assert run("Linf", atol, rtol)[0][0]
    assert not run("Linf", None, rtol)[0][0] and not run("Linf", atol, None)[0][0]


@pytest.mark.parametrize("metric", ["L2", "Linf"])
def test_comparisons_are_strict(metric):
    _, e_abs, e_rel = run(metric, 1.0, 1.0)
    assert not run(metric, None, tuple(e_rel[0]))[0][0]  # error == tolerance is rejected
    assert not run(metric, tuple(e_abs[0]), None)[0][0]


def test_an_omitted_tolerance_never_accepts():
    # F_HAT == F: every error is 0, so only an omitted atol (-inf, not 0) can reject it.
    exact = xp.asarray(F), xp.asarray(F), xp.asarray(STARTS), xp.asarray(SIZES)
    assert host(ACCEPTANCE_CRITERIA.create("Linf", atol=1e-300, rtol=0.0)(*exact)[0]).all()
    assert not host(ACCEPTANCE_CRITERIA.create("Linf", atol=None, rtol=0.0)(*exact)[0]).any()


@pytest.mark.parametrize(
    "atol,rtol,match",
    [(None, None, "Give atol, rtol or both"), (-1.0, None, "finite and >= 0"), (None, [0.1, np.nan], "finite"),
     (np.ones((2, 2)), None, "1-D")],
)  # fmt: skip
def test_invalid_tolerances_are_rejected(atol, rtol, match):
    with pytest.raises(ValueError, match=match):
        ACCEPTANCE_CRITERIA.create("Linf", atol=atol, rtol=rtol)
    with pytest.raises(ValueError, match=match):
        TrainingConfig(mode=SurrogateMode.AFFINE_COV_HESS, atol=atol, rtol=rtol)


def test_a_tolerance_of_the_wrong_length_is_rejected():
    crit = ACCEPTANCE_CRITERIA.create("Linf", atol=(0.1, 0.1), rtol=0.1)
    with pytest.raises(ValueError, match="atol has 2 components but the function has 3 outputs"):
        crit.check_output_dim(FD)


def test_config_stores_tolerances_in_serializable_form():
    cfg = TrainingConfig(mode=SurrogateMode.AFFINE_COV_HESS, atol=np.array([0.1, 0.2]), rtol=None)
    assert cfg.atol == (0.1, 0.2) and cfg.rtol is None
    assert cfg.components().acceptance.settings() == {"atol": (0.1, 0.2), "rtol": None}


def test_cluster_errors_uses_the_configured_criterion():
    cfg = TrainingConfig(mode=SurrogateMode.AFFINE_COV_HESS, atol=0.05, rtol=0.05, error_metric="L2")
    good, e_abs, e_rel = cluster_errors(cfg, xp.asarray(F), xp.asarray(F_HAT), xp.asarray(STARTS), xp.asarray(SIZES))
    np.testing.assert_array_equal(host(good), reference_good("L2", 0.05, 0.05))
    np.testing.assert_allclose(host(e_rel), reference_errors("L2")[1], rtol=1e-12)


def test_unknown_metric_is_rejected_by_the_config():
    with pytest.raises(ValueError, match="Unknown error metric"):
        TrainingConfig(mode=SurrogateMode.AFFINE_COV_HESS, atol=0.1, rtol=0.1, error_metric="Linf-global")


def test_pointwise_acceptance_needs_every_point_to_pass():
    from src.components.acceptance import PointwiseAcceptance

    F = xp.asarray([[1.0], [100.0], [1.0], [1.0]])
    F_hat = xp.asarray([[1.005], [100.5], [1.0], [1.02]])
    good, E_abs, E_rel = PointwiseAcceptance(rtol=0.01)(F, F_hat, xp.asarray([0, 2]), xp.asarray([2, 2]))
    # Cluster 0: both points within 1% (0.5%, 0.5%); cluster 1: the second point is 2% off.
    assert host(good).tolist() == [True, False]
    np.testing.assert_allclose(host(E_rel)[:, 0], [0.005, 0.02])
    np.testing.assert_allclose(host(E_abs)[:, 0], [0.5, 0.02])
    # Linf relative to the cluster's max |F| would accept cluster 1 too: 0.02 / 1 vs 0.5 / 100.
    good_abs, _, _ = PointwiseAcceptance(atol=0.03)(F, F_hat, xp.asarray([0, 2]), xp.asarray([2, 2]))
    assert host(good_abs).tolist() == [False, True]
