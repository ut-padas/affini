"""KRR-enriched leaves (``KRREnrichedModel``): the base model's tree, with a kernel ridge correction per leaf."""

import warnings

import numpy as np
import pytest

from src import xp
from src.components.local_models import KRR_KERNELS, AffineLSQModel, KRREnrichedModel
from src.config import TrainingConfig
from src.surrogate import AdaptiveSurrogate
from tests._helpers import host, train_surrogate

rng = np.random.default_rng(5)
X_TRAIN = rng.uniform(-1, 1, (3_000, 2))
X_QUERY = rng.uniform(-1, 1, (2_000, 2))


def smooth_f(x):
    return (xp.sin(2.0 * x[:, 0]) * xp.cos(1.5 * x[:, 1]) + x[:, 0] ** 2)[:, None]


def train(local_model=None, **kw):
    opts = {"atol": 0.01, "rtol": 0.01, "num_initial_clusters": 4, "min_pts_per_cluster": 60, "max_levels": 4,
            "fallback_leaves": True, "use_work_stealing": False, **kw}  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the max_levels warning
        return train_surrogate(X_TRAIN, AdaptiveSurrogate(), f_fn=smooth_f, local_model=local_model, **opts)


def held_out_error(s) -> float:
    pred, _ = s.predict(xp.asarray(X_QUERY))
    truth = host(smooth_f(xp.asarray(X_QUERY)))
    return float(np.linalg.norm(host(pred) - truth) / np.linalg.norm(truth))


def test_the_enrichment_keeps_the_base_models_tree_and_lowers_the_held_out_error():
    base, krr = train(), train("krr_enriched")
    assert krr.N_src == base.N_src
    for name in ("X_src", "trust_matrices_src", "E_cluster_abs_src", "counts_src", "fallback_src"):
        np.testing.assert_array_equal(host(getattr(krr, name)), host(getattr(base, name)))
    for name in ("A_fit", "b_fit"):  # the base model's own parameters are untouched
        np.testing.assert_array_equal(host(krr.leaf_params[name]), host(base.leaf_params[name]))
    assert held_out_error(krr) < 0.5 * held_out_error(base)
    before, after = host(krr.leaf_params["krr_max_abs_before"]), host(krr.leaf_params["krr_max_abs_after"])
    assert np.median(after / before) < 0.5


def test_the_configuration_resolves_the_base_model_and_the_centres_per_leaf():
    model = TrainingConfig(atol=0.1, min_pts_per_cluster=73, local_model="krr_enriched").components().local_model
    assert isinstance(model, KRREnrichedModel) and isinstance(model.base, AffineLSQModel)
    assert model.n_centers == 7 and model.centers(3) == 7  # floor(0.1 * 73)
    model = TrainingConfig(atol=0.1, local_model=("krr_enriched", {"n_centers": 5, "center_fraction": 0.5})).components().local_model
    assert model.n_centers == 5
    art = TrainingConfig(atol=0.1, algorithm="art", domain=([0, 0], [1, 1]), local_model="krr_enriched").components().local_model
    assert art.points_per_feature == 50.0 and art.centers(2) == 15  # floor(0.1 * ceil(50 * 3))
    with pytest.raises(ValueError, match="kernel"):
        KRREnrichedModel(kernel="no_such_kernel")


def test_saved_models_predict_the_same_after_loading(tmp_path):
    s = train("krr_enriched")
    loaded = AdaptiveSurrogate.load_model(s.save_model(tmp_path / "krr"))
    assert isinstance(loaded.local_model, KRREnrichedModel) and loaded.local_model.settings() == s.local_model.settings()
    np.testing.assert_array_equal(host(loaded.predict(xp.asarray(X_QUERY))[0]), host(s.predict(xp.asarray(X_QUERY))[0]))


def test_training_is_deterministic():
    a, b = train("krr_enriched"), train("krr_enriched")
    for name in ("krr_alpha", "krr_centers", "krr_whiten"):
        np.testing.assert_array_equal(host(a.leaf_params[name]), host(b.leaf_params[name]))


def test_blended_neighbours_evaluate_each_candidate_as_a_single_leaf_would():
    s = train("krr_enriched")
    params = s.model_params()
    X = xp.asarray(X_QUERY[:300])
    ids = xp.asarray(np.random.default_rng(1).integers(0, s.N_src, (300, 3)))
    both = s.local_model.evaluate(params, X, ids)
    for j in range(3):
        np.testing.assert_allclose(host(both[:, j]), host(s.local_model.evaluate(params, X, ids[:, j])), rtol=1e-12, atol=1e-14)


def test_leaves_with_fewer_points_than_centres_use_all_of_them():
    s = train(("krr_enriched", {"n_centers": 300}))  # more centres than any leaf has points
    alpha, counts = host(s.leaf_params["krr_alpha"]), host(s.counts_src)
    assert alpha.shape[1] == 300 and counts.max() < 300 and np.isfinite(host(s.predict(xp.asarray(X_QUERY))[0])).all()
    for k in range(s.N_src):  # the padded slots carry no weight
        assert not np.any(alpha[k, counts[k]:])


@pytest.mark.parametrize("kernel", sorted(KRR_KERNELS))
def test_every_kernel_lowers_the_residuals_on_the_leaves(kernel):
    s = train(("krr_enriched", {"kernel": kernel}))
    before, after = host(s.leaf_params["krr_max_abs_before"]), host(s.leaf_params["krr_max_abs_after"])
    assert np.median(after / before) < 1.0


def train_art(local_model=None, **art):
    from src.config import ARTSettings

    opts = {"atol": 1e-3, "rtol": 1e-3, "algorithm": "art", "domain": ([-1.0, -1.0], [1.0, 1.0]), "max_levels": 5,
            "use_work_stealing": False, "art": ARTSettings(sample_multiplier=20, **art)}  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return train_surrogate(X_TRAIN[:60], AdaptiveSurrogate(), f_fn=smooth_f, local_model=local_model, **opts)


@pytest.mark.parametrize("store_trust_matrix", [True, False])
def test_art_leaves_keep_their_tree_and_route_to_the_enriched_model(store_trust_matrix):
    base = train_art(store_trust_matrix=store_trust_matrix)
    krr = train_art("krr_enriched", store_trust_matrix=store_trust_matrix)
    a, b = base.routing.arrays(), krr.routing.arrays()
    for name in ("normal", "offset", "leaf_locational_code"):
        np.testing.assert_array_equal(host(a[name]), host(b[name]))
    assert krr.local_model.centers(2) == 6  # floor(0.1 * ceil(20 * 3))
    assert (krr.trust_matrices_src is not None) == store_trust_matrix  # computed for the fit, stored only if asked
    assert held_out_error(krr) < 0.5 * held_out_error(base)


def jump_f(x):
    """Affine on each side of a circle, with a jump across it: a smooth kernel rings at the jump."""
    inside = (x[:, 0] ** 2 + x[:, 1] ** 2) < 0.4
    return (xp.where(inside, 1.0 + 0.3 * x[:, 0], 0.2 * x[:, 1]))[:, None]


def train_jump(**krr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return train_surrogate(X_TRAIN, AdaptiveSurrogate(), f_fn=jump_f, local_model=("krr_enriched", krr), atol=1e-3,
                               rtol=1e-3, num_initial_clusters=4, min_pts_per_cluster=60, max_levels=4,
                               fallback_leaves=True, use_work_stealing=False)  # fmt: skip


def test_the_fallback_keeps_every_leaf_at_least_as_good_as_its_base_model():
    kept, raw = train_jump(), train_jump(fallback=False)
    b, a = host(raw.leaf_params["krr_max_abs_before"]), host(raw.leaf_params["krr_max_abs_after"])
    assert np.any(a > b)  # without it, leaves across the jump get worse at their own points
    b, a, active = (host(kept.leaf_params[k]) for k in ("krr_max_abs_before", "krr_max_abs_after", "krr_active"))
    assert np.all(a <= b) and not active.all() and active.any()
    np.testing.assert_array_equal(host(kept.leaf_params["krr_alpha"])[~active[:, 0], :, 0], 0.0)


def test_a_values_target_replaces_the_base_model_on_its_active_outputs():
    base, krr = train(), train(("krr_enriched", {"target": "values"}))
    np.testing.assert_array_equal(host(krr.X_src), host(base.X_src))  # the same tree
    params, X = krr.model_params(), xp.asarray(X_QUERY)
    leaf = xp.asarray(np.random.default_rng(2).integers(0, krr.N_src, len(X_QUERY)))
    alone = host(krr.local_model.correction(params, X, leaf))
    active = host(krr.leaf_params["krr_active"])[host(leaf)]
    pred = host(krr.local_model.evaluate(params, X, leaf))
    assert active.any()
    np.testing.assert_array_equal(pred[active], alone[active])  # the KRR model alone, no base term
    np.testing.assert_array_equal(pred[~active], host(base.local_model.evaluate(base.model_params(), X, leaf))[~active])
    b, a = host(krr.leaf_params["krr_max_abs_before"]), host(krr.leaf_params["krr_max_abs_after"])
    assert np.all(a <= b) and held_out_error(krr) < held_out_error(base)


@pytest.mark.parametrize("kernel", sorted(KRR_KERNELS))
def test_the_compiled_correction_evaluates_the_kernel_formulas(kernel):
    """Prediction's compiled loop (Numba on the CPU, kernels/krr.cu on the GPU) against KRR_KERNELS on whole arrays."""
    s = train(("krr_enriched", {"kernel": kernel}))
    params, X = s.model_params(), xp.asarray(X_QUERY)
    leaf = xp.asarray(np.random.default_rng(3).integers(0, s.N_src, len(X_QUERY)))
    W, C, alpha = (host(params[k])[host(leaf)] for k in ("krr_whiten", "krr_centers", "krr_alpha"))
    Z = np.einsum("nab,nb->na", W, X_QUERY)
    q = np.sum((Z[:, None, :] - C) ** 2, axis=2)
    phi = host(KRR_KERNELS[kernel](xp.asarray(q)))
    np.testing.assert_allclose(host(s.local_model.correction(params, X, leaf)), np.einsum("np,npf->nf", phi, alpha),
                               rtol=1e-12, atol=1e-14 * np.abs(alpha).max())  # fmt: skip
