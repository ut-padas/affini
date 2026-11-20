"""
Behavioural and numerical checks of the default training algorithm (fit, test,
augment rejected clusters, split) through the public AdaptiveSurrogate API.
"""

import warnings

import numpy as np
import pytest

from src import xp
from src.components.modes import SurrogateMode
from src.surrogate import AdaptiveSurrogate
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
    s = AdaptiveSurrogate()
    opts = {
        "mode": mode, "atol": 0.02, "rtol": 0.02, "error_metric": "Linf",
        "num_initial_clusters": 8, "min_pts_per_cluster": 40, "use_work_stealing": False,
    }  # fmt: skip
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
    for name in ("X_src", "f_src", "trust_matrices_src", "A_fits_src", "b_fits_src"):
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
    C, trust_matrices = host(s.X_src), host(s.trust_matrices_src)
    dx = X_TRAIN[:, None, :] - C[None]
    q = np.einsum("nki,kij,nkj->nk", dx, trust_matrices, dx)
    assert np.all(q.min(axis=1) <= 1 + 1e-9)


def test_trust_regions_are_symmetric_positive_definite():
    s = train(quad_f)
    trust_matrices = host(s.trust_matrices_src)
    np.testing.assert_allclose(trust_matrices, trust_matrices.transpose(0, 2, 1), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(trust_matrices) > 0)


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
    b = AdaptiveSurrogate()
    train_surrogate(
        X_TRAIN, b, Y_train=Y, f_fn=quad_f, mode=SurrogateMode.AFFINE_COV_HESS, atol=0.02, rtol=0.02,
        error_metric="Linf", num_initial_clusters=8, min_pts_per_cluster=40, use_work_stealing=False,
    )  # fmt: skip
    c = AdaptiveSurrogate()
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


def stopped_early(**kw):
    """quad_f stopped at max_levels=2, where clusters are still above tolerance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the max_levels warning
        return train(quad_f, max_levels=2, **kw)


def test_fallback_leaves_add_the_clusters_left_at_max_levels_and_change_nothing_else():
    plain, kept = stopped_early(), stopped_early(fallback_leaves=True)
    assert plain.fallback_src is None and not plain.converged and not kept.converged
    fallback = kept.fallback_src
    assert fallback.dtype == bool and fallback.shape == (kept.N_src,)
    assert int(fallback.sum()) == kept.num_unresolved_clusters > 0
    # The accepted leaves are the same leaves, in the same order: only the last level's
    # rejected clusters are handled differently.
    np.testing.assert_array_equal(host(kept.X_src)[~fallback], host(plain.X_src))
    np.testing.assert_array_equal(host(kept.trust_matrices_src)[~fallback], host(plain.trust_matrices_src))
    # They are not augmented and split only to be dropped, so they never cost more evaluations
    # (here none: these clusters already hold the 2 * min_pts_per_cluster an augmentation targets).
    assert kept.profiling_stats["total_f_evals"] <= plain.profiling_stats["total_f_evals"]
    # Every fallback leaf failed the test it was kept despite: some component beyond both tolerances.
    beyond = (host(kept.E_cluster_abs_src) > 0.02) & (host(kept.E_cluster_rel_src) > 0.02)
    assert np.all(beyond[fallback].any(axis=1)) and not np.any(beyond[~fallback].any(axis=1))


def test_fallback_leaves_lower_the_error_where_unresolved_clusters_left_no_leaf():
    truth = host(quad_f(xp.asarray(X_QUERY)))
    errors = []
    for s in (stopped_early(), stopped_early(fallback_leaves=True)):
        pred, _ = s.predict(xp.asarray(X_QUERY))
        errors.append(float(np.linalg.norm(host(pred) - truth) / np.linalg.norm(truth)))
    assert errors[1] < errors[0]


def test_fallback_marks_survive_save_and_load(tmp_path):
    kept = stopped_early(fallback_leaves=True)
    loaded = AdaptiveSurrogate.load_model(kept.save_model(tmp_path / "kept"))
    np.testing.assert_array_equal(loaded.fallback_src, kept.fallback_src)
    plain = AdaptiveSurrogate.load_model(stopped_early().save_model(tmp_path / "plain"))
    assert plain.fallback_src is None
    with np.load(tmp_path / "plain.npz") as data:
        assert "fallback_src" not in data.files


def test_a_converged_run_with_fallback_leaves_marks_none():
    s = train(quad_f, fallback_leaves=True)
    assert s.converged and s.fallback_src is not None and not s.fallback_src.any()
