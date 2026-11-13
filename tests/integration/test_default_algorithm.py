"""
Behavioural and numerical checks of the default training algorithm (fit, test,
augment rejected clusters, split) through the public AFINNISurrogate API.
"""

import warnings

import numpy as np
import pytest

from src import xp
from src.components.modes import SurrogateMode
from src.surrogate import AFINNISurrogate
from tests._helpers import assert_backend_equal, host, train_surrogate

rng = np.random.default_rng(11)
X_TRAIN = rng.uniform(-1, 1, (4_000, 2))
X_QUERY = rng.uniform(-1, 1, (1_000, 2))
A_LIN = np.array([[1.5, -2.0], [0.3, 0.7]])
B_LIN = np.array([0.25, -1.0])


def affine_f(x):
    return x @ xp.asarray(A_LIN).T + xp.asarray(B_LIN)


def affine_g(x):
    return xp.broadcast_to(xp.asarray(A_LIN), (len(x), 2, 2))


def quad_f(x):
    return (1 + x[:, 0] ** 2 + 4 * x[:, 1] ** 2 + x[:, 0] * x[:, 1])[:, None]


def quad_g(x):
    return xp.stack((2 * x[:, 0] + x[:, 1], 8 * x[:, 1] + x[:, 0]), axis=1)[:, None, :]


def quad_h(x):
    return xp.tile(xp.asarray([[2.0, 1.0], [1.0, 8.0]]), (len(x), 1, 1))


def train(f, mode=SurrogateMode.AFFINE_COV_HESS, **kw):
    s = AFINNISurrogate()
    opts = dict(
        mode=mode, atol=0.02, rtol=0.02, error_metric="Linf",
        num_initial_clusters=8, min_pts_per_cluster=40, use_work_stealing=False,
    )  # fmt: skip
    opts.update(kw)
    train_surrogate(X_TRAIN, s, f_fn=f, **opts)
    return s


@pytest.mark.parametrize(
    "mode",
    [m for m in SurrogateMode if m != SurrogateMode.NEAREST_NEIGHBOR],
)
def test_affine_functions_are_reproduced_exactly_everywhere(mode):
    # Every first-order model fits an affine function exactly, so every initial
    # cluster is accepted at level 0 and predictions are exact, even outside.
    s = train(affine_f, mode, g_fn=affine_g, hessian_fn=lambda x: xp.zeros((len(x), 2, 2)))
    assert s.N_src == 8 and s.converged
    assert s.profiling_stats["num_generated_pts"] == 0
    Q = 1.5 * X_QUERY  # beyond the training domain too
    # Taylor models are exact; the affine least-squares fit adds a small Tikhonov
    # term (1e-8 x trace of the covariance), so it is exact to ~1e-6.
    tol = 1e-6 if mode.value.startswith("affine") else 1e-12
    for k in (1, 4):
        pred, _ = s.predict(xp.asarray(Q), k_nn=k)
        np.testing.assert_allclose(host(pred), Q @ A_LIN.T + B_LIN, rtol=tol, atol=tol)


def test_training_is_deterministic():
    # Bit-identical on CPU; on GPU, atomicAdd accumulation order varies between
    # runs, so equal to rounding with the same tree.
    a, b = train(quad_f), train(quad_f)
    assert a.N_src == b.N_src
    for name in ("X_src", "f_src", "Qs_src", "A_fits_src", "b_fits_src"):
        assert_backend_equal(getattr(a, name), getattr(b, name), separate_runs=True)


@pytest.mark.parametrize(
    "mode,metric",
    [
        (SurrogateMode.AFFINE_COV_HESS, "Linf"),
        (SurrogateMode.AFFINE_COV_NO_HESS, "L2"),
        (SurrogateMode.TAYLOR_COV_HESS, "Linf"),
        (SurrogateMode.AFFINE_EXACT_HESS, "Linf"),
        (SurrogateMode.NEAREST_NEIGHBOR, "Linf"),
    ],
)
def test_work_is_conserved_and_observed_points_are_covered(mode, metric):
    s = train(quad_f, mode, error_metric=metric, g_fn=quad_g, hessian_fn=quad_h, atol=0.05, rtol=0.05)
    ps = s.profiling_stats
    assert s.converged and s.num_unresolved_pts == 0
    # Every original and generated point ends in exactly one accepted leaf.
    assert int(host(s.counts_src).sum()) == ps["N0"] + ps["num_generated_pts"] == len(X_TRAIN) + ps["num_generated_pts"]
    # Evaluations: initial points + one centroid per processed cluster + new points.
    assert ps["total_f_evals"] >= ps["N0"] + ps["num_generated_pts"] + s.N_src
    # Every accepted leaf met the tolerance on its own points, in every component.
    E_abs, E_rel = host(s.E_cluster_abs_src), host(s.E_cluster_rel_src)
    assert E_abs.shape == E_rel.shape == (s.N_src, 1)
    assert np.all(np.isfinite(E_abs)) and np.all(np.isfinite(E_rel))
    assert np.all((E_rel < 0.05) | (E_abs < 0.05))
    # The trust regions contain every training point. The farthest point of each
    # cluster lies on its boundary (q = 1 up to rounding), hence the margin.
    C, Q = host(s.X_src), host(s.Qs_src)
    dx = X_TRAIN[:, None, :] - C[None]
    q = np.einsum("nki,kij,nkj->nk", dx, Q, dx)
    assert np.all(q.min(axis=1) <= 1 + 1e-9)


def test_trust_regions_are_symmetric_positive_definite():
    s = train(quad_f)
    Q = host(s.Qs_src)
    np.testing.assert_allclose(Q, Q.transpose(0, 2, 1), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(Q) > 0)


@pytest.mark.parametrize("tol", [0.05, 0.01])
def test_in_region_held_out_error_tracks_the_tolerance(tol):
    s = train(quad_f, atol=tol, rtol=tol)
    pred, mask = s.predict(xp.asarray(X_QUERY), k_nn=1)
    pred, mask = host(pred), host(mask)
    truth = host(quad_f(xp.asarray(X_QUERY)))
    rel = np.abs(pred - truth)[~mask] / np.abs(truth)[~mask]
    assert mask.mean() < 0.2
    assert np.median(rel) < tol and np.quantile(rel, 0.95) < 3 * tol


def test_tighter_tolerance_gives_more_leaves_and_lower_error():
    errs, leaves = [], []
    for tol in (0.1, 0.02):
        s = train(quad_f, atol=tol, rtol=tol)
        pred, _ = s.predict(xp.asarray(X_QUERY), k_nn=1)
        truth = host(quad_f(xp.asarray(X_QUERY)))
        errs.append(np.linalg.norm(host(pred) - truth) / np.linalg.norm(truth))
        leaves.append(s.N_src)
    assert leaves[1] > leaves[0] and errs[1] < errs[0]


def test_inputs_and_precomputed_values_paths_agree():
    Y = quad_f(xp.asarray(X_TRAIN))
    a = train(quad_f)
    b = AFINNISurrogate()
    train_surrogate(
        X_TRAIN, b, Y_train=Y, f_fn=quad_f, mode=SurrogateMode.AFFINE_COV_HESS, atol=0.02, rtol=0.02,
        error_metric="Linf", num_initial_clusters=8, min_pts_per_cluster=40, use_work_stealing=False,
    )  # fmt: skip
    c = AFINNISurrogate()
    train_surrogate(
        X_TRAIN, c, oracle_fn=lambda x: (quad_f(x), quad_g(x)), mode=SurrogateMode.AFFINE_COV_HESS,
        atol=0.02, rtol=0.02, error_metric="Linf", num_initial_clusters=8, min_pts_per_cluster=40,
        use_work_stealing=False,
    )  # fmt: skip
    for other in (b, c):
        assert other.N_src == a.N_src
        assert_backend_equal(other.X_src, a.X_src, separate_runs=True)


def test_no_unexpected_warnings_on_a_normal_run():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        train(quad_f)
