"""
The component API: modes are presets, and each component (local model, split
strategy, sampler, acceptance criterion, trust region) can be replaced through
``ClusteringConfig`` / ``AFINNISurrogate.train`` without editing the engine.

These tests check the contracts (which data a component receives, which outputs
reach prediction, that defaults are unchanged), not the accuracy of new components.
"""

import json

import numpy as np
import pytest

from src import xp
from src.acceptance import ACCEPTANCE_CRITERIA, AcceptanceCriterion, LinfAcceptance
from src.clusters import ParamSpec
from src.config import ClusteringConfig
from src.local_models import (
    LOCAL_MODELS,
    AffineLSQModel,
    ConstantModel,
    LocalModel,
    TaylorModel,
    local_model_for_mode,
)
from src.modes import SurrogateMode
from src.sampling import SAMPLERS, ConvexCombinationSampler
from src.splitting import (
    SPLIT_STRATEGIES,
    CovarianceHessianSplit,
    CovarianceSplit,
    GradientLSQHessianSplit,
    split_strategy_for_mode,
)
from src.surrogate import AFINNISurrogate
from src.trust_regions import TRUST_REGIONS, CovarianceEllipsoid, TrustRegion
from tests._helpers import assert_backend_equal, host, train_surrogate

rng = np.random.default_rng(3)
X_TRAIN = rng.uniform(-1, 1, (3_000, 2))
X_QUERY = rng.uniform(-1, 1, (500, 2))


def f(x):
    return (x[:, 0] ** 2 + 3 * x[:, 1] ** 2 + 0.5 * x[:, 0] * x[:, 1] + 1.0)[:, None]


def g(x):
    return xp.stack((2 * x[:, 0] + 0.5 * x[:, 1], 6 * x[:, 1] + 0.5 * x[:, 0]), axis=1)[
        :, None, :
    ]


def train(**kwargs):
    s = AFINNISurrogate()
    opts = dict(
        f_fn=f, atol=0.05, rtol=0.05, error_metric="Linf", num_initial_clusters=8,
        min_pts_per_cluster=40, use_work_stealing=False,
    )  # fmt: skip
    opts.update(kwargs)
    train_surrogate(X_TRAIN, s, **opts)
    return s


def predict(s, k=1):
    pred, mask = s.predict(xp.asarray(X_QUERY), k_nn=k)
    return host(pred), host(mask)


@pytest.fixture
def registered():
    """Register test components and remove them afterwards."""
    added = []

    def add(registry, cls):
        registry.register(cls)
        added.append((registry, cls.name))
        return cls

    yield add
    for registry, name in added:
        del registry[name]


# --- Defaults: modes are presets of the same components as before -------------


@pytest.mark.parametrize("mode", list(SurrogateMode))
def test_mode_presets_resolve_to_their_preset_components(mode):
    c = ClusteringConfig(mode=mode, atol=0.1, rtol=0.1, min_pts_per_cluster=37).components()
    assert type(c.local_model) is type(local_model_for_mode(mode))
    preset_split = split_strategy_for_mode(mode, 37)
    assert type(c.split_strategy) is type(preset_split)
    assert c.split_strategy.settings() == preset_split.settings()
    assert type(c.sampler) is ConvexCombinationSampler
    assert type(c.acceptance) is ACCEPTANCE_CRITERIA["Linf"]
    assert isinstance(c.trust_region, CovarianceEllipsoid) and c.trust_region.q_reg == 1e-6
    json.dumps(c.describe())  # serializable


def test_components_are_resolved_once_per_config():
    cfg = ClusteringConfig(mode=SurrogateMode.AFFINE_COV_HESS, atol=0.1, rtol=0.1)
    assert cfg.components() is cfg.components()
    assert cfg.components().local_model is cfg.components().local_model


def test_explicit_components_equal_to_the_preset_give_identical_results():
    preset = train(mode=SurrogateMode.AFFINE_COV_HESS)
    explicit = train(
        mode=SurrogateMode.AFFINE_COV_HESS,
        local_model="affine_lsq",
        split_strategy=("covariance_hessian", {"gradient": "local_lsq"}),
        sampler="convex_combination",
        acceptance="Linf",
        trust_region="covariance_ellipsoid",
    )
    # Two separate runs: bit-identical on CPU; on GPU, atomicAdd accumulation order
    # varies between runs, so equal to rounding (decisions identical).
    assert explicit.N_src == preset.N_src
    for name in ("X_src", "Qs_src", "A_fits_src", "b_fits_src"):
        assert_backend_equal(getattr(explicit, name), getattr(preset, name), separate_runs=True)
    assert_backend_equal(predict(explicit)[0], predict(preset)[0], separate_runs=True)


def test_component_settings_come_from_the_configuration():
    cfg = ClusteringConfig(
        mode=SurrogateMode.AFFINE_COV_HESS, atol=0.2, rtol=0.3, q_reg=1e-4,
        min_pts_per_cluster=55, split_strategy="covariance_hessian",
        acceptance="L2", trust_region="covariance_ellipsoid",
    )  # fmt: skip
    c = cfg.components()
    assert c.split_strategy.min_pts_per_cluster == 55
    assert (c.acceptance.atol, c.acceptance.rtol) == (0.2, 0.3)
    assert c.trust_region.q_reg == 1e-4


def test_unknown_or_invalid_components_fail_early():
    with pytest.raises(ValueError, match="Unknown local model"):
        ClusteringConfig(
            mode=SurrogateMode.AFFINE_COV_HESS, atol=0.1, rtol=0.1, local_model="nope"
        ).components()
    with pytest.raises(TypeError, match="split strategy spec"):
        ClusteringConfig(
            mode=SurrogateMode.AFFINE_COV_HESS, atol=0.1, rtol=0.1, split_strategy=3.5
        ).components()


# --- Data requirements follow from the model and the split together -----------


class GradientAffineModel(AffineLSQModel):
    """An affine model that (claims to) need per-point gradients; records what it got."""

    name = "test_gradient_affine"
    needs_point_grad = True
    seen_G_shapes: list = []

    def fit(self, data):
        type(self).seen_G_shapes.append(None if data.G is None else tuple(data.G.shape))
        return super().fit(data)


def test_a_gradient_model_with_a_gradient_free_split_receives_point_gradients():
    cfg = ClusteringConfig(
        mode=SurrogateMode.AFFINE_COV_NO_HESS, atol=0.1, rtol=0.1,
        local_model=GradientAffineModel(),
    )  # fmt: skip
    assert cfg.requires_grad_everywhere and not cfg.requires_grad_centroids
    with pytest.raises(ValueError, match="requires gradient information"):
        train(mode=SurrogateMode.AFFINE_COV_NO_HESS, local_model=GradientAffineModel())

    GradientAffineModel.seen_G_shapes = []
    s = train(mode=SurrogateMode.AFFINE_COV_NO_HESS, local_model=GradientAffineModel(), g_fn=g)
    assert GradientAffineModel.seen_G_shapes
    assert all(shape is not None and shape[1:] == (1, 2) for shape in GradientAffineModel.seen_G_shapes)
    assert s.N_src > 0


def test_a_gradient_free_model_with_a_gradient_split_requires_point_gradients():
    cfg = ClusteringConfig(
        mode=SurrogateMode.NEAREST_NEIGHBOR, atol=0.1, rtol=0.1,
        local_model=ConstantModel(), split_strategy=GradientLSQHessianSplit(),
    )  # fmt: skip
    assert cfg.requires_grad_everywhere and cfg.requires_grad_centroids


def test_gradient_free_configurations_never_call_the_gradient():
    calls = {"f": 0, "g": 0}

    def f_counted(x):
        calls["f"] += 1
        return f(x)

    def g_counted(x):
        calls["g"] += 1
        return g(x)

    for mode in (SurrogateMode.AFFINE_COV_HESS, SurrogateMode.AFFINE_COV_NO_HESS, SurrogateMode.NEAREST_NEIGHBOR):
        train(mode=mode, f_fn=f_counted, g_fn=g_counted)
    assert calls["f"] > 0 and calls["g"] == 0


# --- New components: outputs reach prediction ---------------------------------


class AffinePlusOffsetModel(LocalModel):
    """Affine LSQ fit plus a per-leaf offset added only for accepted leaves (finalize)."""

    name = "test_affine_plus_offset"

    def __init__(self, offset: float = 1.0):
        self.offset = offset
        self._affine = AffineLSQModel()

    def settings(self):
        return {"offset": self.offset}

    def columns(self, d, fd):
        return {**self._affine.columns(d, fd), "offset": ParamSpec((fd,))}

    def fit(self, data):
        return self._affine.fit(data)

    def finalize(self, data, params):
        return {"offset": xp.full((len(data.sizes), data.F.shape[1]), self.offset)}

    def evaluate(self, params, X, cluster_ids):
        out = self._affine.evaluate(params, X, cluster_ids)
        return out + params["offset"][cluster_ids] if "offset" in params else out


def test_finalized_columns_are_stored_and_reach_predict(registered):
    registered(LOCAL_MODELS, AffinePlusOffsetModel)
    base = train(mode=SurrogateMode.AFFINE_COV_HESS)
    # An offset far below the tolerance passes the re-test of the deployed model.
    s = train(mode=SurrogateMode.AFFINE_COV_HESS, local_model=("test_affine_plus_offset", {"offset": 1e-6}))
    assert s.N_src == base.N_src
    assert set(s.leaf_params) == {"A_fit", "b_fit", "offset"}
    assert host(s.leaf_params["offset"]).shape == (s.N_src, 1)
    for k in (1, 4):
        # The offset is added before Shepard weighting, so equal up to rounding.
        np.testing.assert_allclose(predict(s, k)[0], predict(base, k)[0] + 1e-6, rtol=1e-12)
    assert s.config["components"]["local_model"] == {
        "name": "test_affine_plus_offset",
        "settings": {"offset": 1e-6},
    }


def test_deployed_models_are_retested_after_finalize(registered):
    """A finalize() that breaks predictions must not be accepted on the strength of fit() alone."""
    registered(LOCAL_MODELS, AffinePlusOffsetModel)
    with pytest.raises(RuntimeError, match="accepted no clusters"):
        train(mode=SurrogateMode.AFFINE_COV_HESS, max_levels=3,
              local_model=("test_affine_plus_offset", {"offset": 10.0}))  # fmt: skip


class DuplicatingSplit(CovarianceSplit):
    name = "test_duplicating"

    def split(self, data):
        order, sizes = super().split(data)
        order = order.copy()
        order[1] = order[0]  # one row twice, another lost
        return order, sizes


def test_split_that_duplicates_rows_is_rejected():
    with pytest.raises(ValueError, match="drops, duplicates"):
        train(mode=SurrogateMode.AFFINE_COV_HESS, split_strategy=DuplicatingSplit())


def test_effective_settings_are_recorded_next_to_requested_ones():
    s = train(mode=SurrogateMode.AFFINE_COV_HESS, batch_size=10**12)
    assert s.config["batch_size"] == 10**12  # as requested
    assert s.config["effective"]["batch_size"] < 10**12  # capped to fit memory
    assert s.config["effective"]["max_levels"] == s.config["max_levels"]


class AcceptAll(AcceptanceCriterion):
    name = "test_accept_all"

    def __call__(self, F, F_hat, starts, sizes):
        zeros = xp.zeros((len(sizes), F.shape[1]))
        return xp.ones(len(sizes), dtype=bool), zeros, zeros


def test_custom_acceptance_decides_which_clusters_become_leaves():
    s = train(mode=SurrogateMode.AFFINE_COV_HESS, acceptance=AcceptAll(0.0, 0.0))
    assert s.N_src == 8  # every initial cluster accepted at level 0
    assert s.profiling_stats["num_generated_pts"] == 0
    assert s.config["components"]["acceptance"]["name"] == "test_accept_all"


class ScaledIdentity(TrustRegion):
    name = "test_scaled_identity"

    def __init__(self, scale: float = 4.0):
        self.scale = scale

    def shapes(self, X, centroids, labels, counts):
        return xp.tile(self.scale * xp.eye(X.shape[1]), (len(counts), 1, 1))


def test_custom_trust_region_shapes_are_used(registered):
    registered(TRUST_REGIONS, ScaledIdentity)
    s = train(mode=SurrogateMode.AFFINE_COV_HESS, trust_region=("test_scaled_identity", {"scale": 9.0}))
    np.testing.assert_array_equal(host(s.Qs_src), np.tile(9.0 * np.eye(2), (s.N_src, 1, 1)))
    # Inside means within distance 1/3 of some candidate centroid.
    _, mask = predict(s)
    d_min = np.min(np.linalg.norm(X_QUERY[:, None] - host(s.X_src)[None], axis=2), axis=1)
    assert np.all(mask[d_min < 1 / 3 - 1e-9] == False)  # noqa: E712


class MidpointSampler(ConvexCombinationSampler):
    """The convex-combination sampler with a (recorded, unused) setting."""

    name = "test_midpoint"

    def __init__(self, weight: float = 0.5):
        self.weight = weight

    def settings(self):
        return {"weight": self.weight}

    def sample(self, clusters, n_new, rng):
        return super().sample(clusters, n_new, rng)


def test_sampler_settings_are_passed_and_recorded(registered):
    registered(SAMPLERS, MidpointSampler)
    s = train(mode=SurrogateMode.AFFINE_COV_HESS, min_pts_per_cluster=200, sampler=("test_midpoint", {"weight": 0.25}))
    assert s.config["components"]["sampler"] == {"name": "test_midpoint", "settings": {"weight": 0.25}}
    assert s.profiling_stats["num_generated_pts"] > 0


def test_split_strategy_instance_with_sign_cut():
    s = train(mode=SurrogateMode.AFFINE_COV_HESS, split_strategy=CovarianceSplit(cut="sign"))
    assert s.converged and s.N_src > 8
    assert s.config["components"]["split_strategy"] == {"name": "covariance", "settings": {"cut": "sign"}}


# --- Contracts are enforced ------------------------------------------------------


class WrongShapeSampler(ConvexCombinationSampler):
    name = "test_wrong_shape"

    def sample(self, clusters, n_new, rng):
        return super().sample(clusters, n_new, rng)[:-1]


def test_sampler_returning_the_wrong_number_of_points_is_rejected():
    with pytest.raises(ValueError, match="test_wrong_shape"):
        train(mode=SurrogateMode.AFFINE_COV_HESS, min_pts_per_cluster=200, sampler=WrongShapeSampler())


class LopsidedSplit(CovarianceSplit):
    name = "test_lopsided"

    def child_sizes(self, counts, sorted_scores=None):
        sizes = super().child_sizes(counts, sorted_scores)
        sizes[0] += 1  # first child too large
        return sizes


def test_split_with_inconsistent_child_sizes_is_rejected():
    with pytest.raises(ValueError, match="test_lopsided"):
        train(mode=SurrogateMode.AFFINE_COV_HESS, split_strategy=LopsidedSplit())


def test_store_hess_with_an_explicit_model_is_ignored(caplog):
    cfg = ClusteringConfig(
        mode=SurrogateMode.TAYLOR_LSQ_HESS, atol=0.1, rtol=0.1, store_hess=True,
        local_model=TaylorModel(second_order=False),
    )  # fmt: skip
    assert not cfg.store_hess and "H" not in cfg.components().local_model.columns(2, 1)


def test_builtin_components_are_registered():
    assert {"constant", "taylor", "affine_lsq"} <= set(LOCAL_MODELS)
    assert {"covariance", "covariance_hessian", "exact_hessian", "gradient_lsq_hessian"} <= set(SPLIT_STRATEGIES)
    assert {"L2", "Linf"} <= set(ACCEPTANCE_CRITERIA) and "Linf-global" not in ACCEPTANCE_CRITERIA
    assert "covariance_ellipsoid" in TRUST_REGIONS and "convex_combination" in SAMPLERS
    assert isinstance(ACCEPTANCE_CRITERIA.create("Linf", atol=1, rtol=2), LinfAcceptance)
    assert isinstance(SPLIT_STRATEGIES.create("covariance_hessian"), CovarianceHessianSplit)


def test_config_to_dict_is_serializable():
    cfg = ClusteringConfig(mode=SurrogateMode.TAYLOR_COV_HESS, atol=0.1, rtol=0.1)
    d = cfg.to_dict()
    json.dumps(d)
    assert d["mode"] == "taylor_cov_hess" and d["components"]["local_model"]["name"] == "taylor"
