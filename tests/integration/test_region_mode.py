"""Region mode (algorithm="region") through the public AFINNISurrogate API."""

import numpy as np
import pytest

from src import xp
from src.config import ClusteringConfig, RegionSettings
from src.core.evaluation import Evaluator
from src.surrogate import AFINNISurrogate
from tests._helpers import assert_backend_equal, host, train_surrogate

DOMAIN = ([-1.0, -1.0], [1.0, 1.0])


def wavy(x):
    return (xp.sin(3 * x[:, 0]) * xp.cos(2 * x[:, 1]) + x[:, 0] ** 2)[:, None]


def region_options(**overrides):
    options = {
        "f_fn": wavy, "atol": 1e-2, "rtol": 1e-2, "algorithm": "region", "domain": DOMAIN, "max_levels": 20,
        "num_initial_clusters": 4, "use_work_stealing": False,
        "region": RegionSettings(sample_multiplier=10, validation_samples=10),
    }  # fmt: skip
    options.update(overrides)
    return options


@pytest.fixture(scope="module")
def points():
    return np.random.default_rng(0).uniform(-1, 1, (400, 2))


@pytest.fixture(scope="module")
def trained(points):
    return train_surrogate(points, **region_options())


def test_region_model_converges_and_routes_every_query_to_one_leaf(trained):
    assert trained.converged and trained.routing is not None
    counts = trained.leaf_status_counts()
    assert counts["terminal_reason"]["accepted"] == trained.N_src
    assert counts["validation_status"]["passed"] == trained.N_src
    leaves = host(trained.route_leaves(np.random.default_rng(3).uniform(-1, 1, (2000, 2))))
    assert leaves.min() >= 0 and leaves.max() < trained.N_src


def test_region_tree_does_not_depend_on_the_batch_size(points, trained):
    small = train_surrogate(points, **region_options(batch_size=97))
    assert small.N_src == trained.N_src
    assert small.profiling_stats["total_f_evals"] == trained.profiling_stats["total_f_evals"]
    q = np.random.default_rng(4).uniform(-1, 1, (300, 2))
    assert_backend_equal(small.predict(q)[0], trained.predict(q)[0], separate_runs=True)


def test_audit_reports_violations_and_a_binomial_bound(trained):
    audit = trained.audit(Evaluator(f_fn=wavy), 2000, seed=1)
    assert audit["n_points"] == 2000 and 0 <= audit["n_violations"] < 2000 * 0.05
    assert audit["violation_rate"] <= audit["upper_bound"] < 0.05
    clean = trained.audit(Evaluator(f_fn=wavy), 299, seed=2, atol=1e9)  # nothing can fail
    assert clean["n_violations"] == 0 and clean["upper_bound"] == pytest.approx(1 - 0.05 ** (1 / 299))


def test_audit_counts_non_finite_oracle_values_as_violations(trained):
    def poisoned(x):
        f = wavy(x)
        f[::3] = xp.nan
        f[1::3] = xp.inf
        return f

    audit = trained.audit(Evaluator(f_fn=poisoned), 300, seed=1, atol=1e9)  # finite rows cannot fail
    assert audit["n_invalid"] == audit["n_violations"] == 200
    assert np.isfinite(audit["max_scaled_error"])


def test_audit_defaults_to_the_trained_tolerances_and_an_error_at_the_tolerance_fails(trained):
    from src.components.acceptance import pointwise_violations

    default = trained.audit(Evaluator(f_fn=wavy), 500, seed=3)
    explicit = trained.audit(Evaluator(f_fn=wavy), 500, seed=3, atol=1e-2, rtol=1e-2)
    assert default["n_violations"] == explicit["n_violations"] and default["atol"] == explicit["atol"]
    F = xp.asarray([[1.0], [1.0]])
    at_tol = pointwise_violations(F, F + xp.asarray([[0.5], [0.25]]), atol=0.5, rtol=None)
    np.testing.assert_array_equal(host(at_tol), [True, False])  # strict, as in training


@pytest.mark.parametrize("kwargs", [{"n_points": 0}, {"n_points": 2.5}, {"n_points": True},
                                    {"confidence": 1.0}, {"confidence": float("nan")}])  # fmt: skip
def test_audit_rejects_invalid_arguments(trained, kwargs):
    with pytest.raises(ValueError):
        trained.audit(Evaluator(f_fn=wavy), **{"n_points": 10, **kwargs})


def test_route_details_report_each_querys_leaf_status(trained):
    from src.region import TERMINAL_REASONS, VALIDATION_STATUSES

    q = np.array([[0.3, -0.2], [1.5, 0.0]])
    details = trained.route_details(q)
    np.testing.assert_array_equal(host(details["leaf"]), host(trained.route_leaves(q)))
    np.testing.assert_array_equal(host(details["outside"]), [False, True])
    assert TERMINAL_REASONS[int(host(details["terminal_reason"])[0])] == "accepted"
    assert VALIDATION_STATUSES[int(host(details["validation_status"])[0])] == "passed"


def test_routed_predictions_match_the_leaf_models_and_flag_only_the_domain(trained):
    q = np.array([[0.3, -0.2], [1.5, 0.0], [-1.0, 1.0]])
    pred, outside = trained.predict(q)
    np.testing.assert_array_equal(host(outside), [False, True, False])
    leaf = trained.route_leaves(q)
    direct = trained.local_model.evaluate(trained._leaf_params(), xp.asarray(q), leaf)
    assert_backend_equal(pred, direct)
    knn_pred, _ = trained.predict(q, predictor="knn")  # Q is stored, so kNN still works
    assert knn_pred.shape == pred.shape


def test_loading_rejects_leaf_metadata_that_names_other_nodes(tmp_path, trained):
    path = trained.save_model(tmp_path)
    with np.load(path, allow_pickle=False) as f:
        data = dict(f)
    key = next(k for k in data if k.endswith("path_code"))
    data[key] = data[key][::-1].copy()
    np.savez(path, **data)
    with pytest.raises(ValueError, match="disagree"):
        AFINNISurrogate.load_model(path)


def test_region_model_save_and_load_round_trip(tmp_path, trained):
    loaded = AFINNISurrogate.load_model(trained.save_model(tmp_path))
    q = np.random.default_rng(5).uniform(-1.2, 1.2, (200, 2))
    assert_backend_equal(loaded.predict(q)[0], trained.predict(q)[0])
    for name, value in trained.region_leaves.items():
        np.testing.assert_array_equal(loaded.region_leaves[name], value)


@pytest.mark.filterwarnings("ignore:Training stopped:RuntimeWarning")
def test_affine_leaves_with_a_covariance_split_never_evaluate_f_at_centroids(points):
    """Neither component reads f(c), so every evaluation is a sampled point."""
    model = train_surrogate(
        points, **region_options(split_strategy="covariance", local_model="affine_lsq", max_levels=6)
    )
    stats = model.profiling_stats
    assert stats["total_f_evals"] == stats["N0"] + stats["num_generated_pts"]
    assert np.all(np.isnan(host(model.f_src)))


def test_max_levels_leaves_fallback_leaves_with_their_status(points):
    with pytest.warns(RuntimeWarning, match="keep their fitted model"):
        model = train_surrogate(points, **region_options(max_levels=1, region=RegionSettings(roots="grid", sample_multiplier=10)))
    counts = model.leaf_status_counts()
    assert not model.converged and counts["terminal_reason"]["max_depth"] > 0
    assert counts["validation_status"]["not_performed"] == model.N_src
    pred, _ = model.predict(np.zeros((3, 2)))  # every region still has a leaf
    assert np.all(np.isfinite(host(pred)))


@pytest.mark.filterwarnings("ignore:Training stopped:RuntimeWarning")
def test_region_grid_roots_without_points_are_sampled(points):
    model = train_surrogate(points[:2], **region_options(domain="data_bbox", region=RegionSettings(roots="grid", sample_multiplier=10)))
    assert set(host(model.region_leaves["root"]).tolist()) == set(range(4))
    assert model.config["domain"] == "data_bbox"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"domain": None}, "needs domain"),
        ({"domain": ([0, 0], [0, 1])}, "lower < upper"),
        ({"algorithm": "afinni"}, "only apply to algorithm='region'"),
        ({"region": RegionSettings(sample_multiplier=1)}, "no split could ever be made"),
    ],
)
def test_region_configuration_errors(points, overrides, message):
    with pytest.raises(ValueError, match=message):
        train_surrogate(points, **region_options(**overrides))


def test_region_rejects_points_outside_the_domain_and_initial_labels(points):
    with pytest.raises(ValueError, match="outside the domain"):
        train_surrogate(points * 2, **region_options())
    with pytest.raises(ValueError, match="initial_labels"):
        train_surrogate(points, initial_labels=np.zeros(len(points), dtype=np.int64), **region_options())


def test_route_predictor_needs_a_region_model(points):
    afinni = train_surrogate(points, f_fn=wavy, atol=0.05, rtol=0.05, num_initial_clusters=4, min_pts_per_cluster=16, use_work_stealing=False)
    assert afinni.routing is None
    with pytest.raises(ValueError, match="region-mode model"):
        afinni.predict(points[:3], predictor="route")


def test_region_settings_validation():
    with pytest.raises(ValueError, match="roots"):
        RegionSettings(roots="hexagon")
    with pytest.raises(ValueError, match="walk_steps"):
        RegionSettings(walk_steps=0)
    assert ClusteringConfig(atol=1, algorithm="region", domain=DOMAIN, region={"roots": "grid"}).region.roots == "grid"



@pytest.mark.parametrize("search", ["gap", "bins", "soft"])
def test_every_split_search_builds_a_valid_batch_independent_tree(points, search):
    """B2 (gap), B3 (bins) and B4 (soft) all produce routable trees, unchanged by the batch size."""
    options = region_options(region=RegionSettings(sample_multiplier=10, split_search=search, soft_iters=5), max_levels=8)
    big, small = train_surrogate(points, **options), train_surrogate(points, batch_size=97, **options)
    assert big.N_src == small.N_src and big.profiling_stats["total_f_evals"] == small.profiling_stats["total_f_evals"]
    q = np.random.default_rng(6).uniform(-1, 1, (500, 2))
    assert_backend_equal(big.predict(q)[0], small.predict(q)[0], separate_runs=True)
    counters = big.profiling_stats["region"]
    assert counters["splits"] == big.profiling_stats["num_split_nodes"]
    if search == "soft":
        stops = ("soft_gradient_converged", "soft_line_search_stalled", "soft_gate_saturated", "soft_nonfinite", "soft_iteration_limit")
        assert sum(counters[c] for c in stops) > 0  # every refined node ends with exactly one reason


def step(x):
    """An oblique discontinuity: affine pieces on either side of x0 + 0.5 x1 = 0.2."""
    return xp.where((x[:, 0] + 0.5 * x[:, 1] > 0.2)[:, None], 1.0 + x[:, :1], -1.0 + x[:, 1:])


def test_threshold_search_finds_an_oblique_interface_the_curvature_split_misses(points):
    """The fit-driven bins search needs far fewer leaves than the median gap cut on a step."""
    options = region_options(f_fn=step, atol=1e-6, rtol=None, max_levels=8)
    gap = train_surrogate(points, **{**options, "region": RegionSettings(sample_multiplier=10, split_search="gap")})
    bins = train_surrogate(points, **{**options, "region": RegionSettings(sample_multiplier=10, split_search="bins")})
    assert bins.N_src < gap.N_src
    audit = bins.audit(Evaluator(f_fn=step), 2000, seed=3)
    assert audit["violation_rate"] < 0.05


@pytest.mark.filterwarnings("ignore:Training stopped:RuntimeWarning")
def test_voronoi_roots_are_the_initial_clustering_cells(points):
    model = train_surrogate(points, **region_options(region=RegionSettings(roots="voronoi", sample_multiplier=10), max_levels=4))
    centres = model.routing.roots.centres
    assert centres.shape == (4, 2)
    # Every query is routed inside the root of its nearest centre.
    q = np.random.default_rng(7).uniform(-1, 1, (300, 2))
    U = (q + 1) / 2
    nearest = np.argmin(((U[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
    np.testing.assert_array_equal(host(model.region_leaves["root"])[host(model.route_leaves(q))], nearest)


@pytest.mark.filterwarnings("ignore:Training stopped:RuntimeWarning")
def test_b1_convex_sampling_tree_serves_both_predictors(points):
    """B1: AFINNI's augmentation on Voronoi roots, Q kept, so kNN and routing use one artifact."""
    model = train_surrogate(points, **region_options(
        region=RegionSettings(roots="voronoi", sampling="convex_combination"), min_pts_per_cluster=16, max_levels=10,
    ))  # fmt: skip
    q = np.random.default_rng(8).uniform(-1, 1, (300, 2))
    routed, outside = model.predict(q)
    knn, _ = model.predict(q, predictor="knn")
    assert not host(outside).any() and routed.shape == knn.shape
    with pytest.raises(ValueError, match="validation_samples"):
        RegionSettings(sampling="convex_combination", validation_samples=5)


def test_routed_only_models_store_no_q_and_refuse_knn(tmp_path, points):
    model = train_surrogate(points, **region_options(region=RegionSettings(sample_multiplier=10, store_Q=False), max_levels=8))
    assert model.Qs_src is None and model.knn_index is None
    with pytest.raises(ValueError, match="store_Q=False"):
        model.predict(points[:3], predictor="knn")
    loaded = AFINNISurrogate.load_model(model.save_model(tmp_path))
    assert loaded.Qs_src is None
    assert_backend_equal(loaded.predict(points[:20])[0], model.predict(points[:20])[0])


@pytest.mark.filterwarnings("ignore:Training stopped:RuntimeWarning")
def test_training_warns_when_leaves_have_insufficient_sampling(points):
    """``converged`` says nothing about sampling quality: nodes failing the drift check (forced here) are reported."""
    with pytest.warns(RuntimeWarning, match="insufficient_sampling"):
        model = train_surrogate(points, **region_options(
            region=RegionSettings(sample_multiplier=10, walk_steps=2, max_walk_steps=2, mixing_z=1e-9), max_levels=6,
        ))  # fmt: skip
    assert model.leaf_status_counts()["validation_status"]["insufficient_sampling"] > 0


def test_region_batches_hold_whole_nodes_within_the_batch_size_or_one_node_alone(points, monkeypatch):
    """An oversized node never shares a batch, even after a smaller node (the AFINNI grid would admit both)."""
    from src.region import RegionPlan

    batch_size, seen = 100, []
    original = RegionPlan.process_batch

    def spy(self, batch, **kwargs):
        seen.append(self.work_sizes(host(batch.sizes)))
        return original(self, batch, **kwargs)

    monkeypatch.setattr(RegionPlan, "process_batch", spy)
    train_surrogate(points, **region_options(batch_size=batch_size, max_levels=6,
                                             region=RegionSettings(roots="grid", sample_multiplier=10)))  # fmt: skip
    assert any(len(w) == 1 and w[0] > batch_size for w in seen)  # an oversized node occurred
    assert all(len(w) == 1 or w.sum() <= batch_size for w in seen)


def test_quadratic_leaves_represent_a_quadratic_with_one_leaf_and_round_trip(points, tmp_path):
    """ART's degree-2 leaves: the root already fits, with ART's budget of multiplier x (d+1)(d+2)/2 points."""

    def quad(x):
        return (2.0 + x[:, 0] ** 2 - 0.5 * x[:, 0] * x[:, 1] + x[:, 1])[:, None]

    model = train_surrogate(points, **region_options(f_fn=quad, local_model="quadratic_lsq", rtol=1e-6, atol=1e-6))
    assert model.N_src == 1 and model.converged
    q = np.random.default_rng(9).uniform(-1, 1, (500, 2))
    np.testing.assert_allclose(host(model.predict(q)[0]), host(quad(xp.asarray(q))), rtol=1e-8)
    loaded = AFINNISurrogate.load_model(model.save_model(tmp_path))
    assert_backend_equal(loaded.predict(q)[0], model.predict(q)[0])


@pytest.mark.parametrize("search", ["bins", "soft"])
def test_quadratic_leaves_split_in_their_own_basis_and_need_fewer_leaves(points, search):
    options = region_options(region=RegionSettings(sample_multiplier=10, split_search=search), max_levels=10)
    affine = train_surrogate(points, **options)
    quadratic = train_surrogate(points, **{**options, "local_model": "quadratic_lsq"})
    assert quadratic.converged and quadratic.N_src < affine.N_src / 2

