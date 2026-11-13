"""
Saving and loading surrogates: the versioned format, path resolution, MPI writes
and custom local models; also train()'s failure and profiler cleanup.
"""

import json

import numpy as np
import pytest

from src import xp
from src.components.local_models import LOCAL_MODELS, AffineLSQModel, LocalModel
from src.components.modes import SurrogateMode
from src.config import ClusteringConfig
from src.core.clusters import ParamSpec
from src.core.evaluation import Evaluator
from src.surrogate import MODEL_FORMAT_VERSION, AFINNISurrogate
from tests._helpers import assert_backend_equal, host

rng = np.random.default_rng(21)
X = rng.uniform(-1, 1, (2_000, 2))
Q = rng.uniform(-1.2, 1.2, (300, 2))


def f(x):
    return xp.stack((1 + x[:, 0] ** 2 + 3 * x[:, 1] ** 2, x[:, 0] * x[:, 1] - 2), axis=1)


def g(x):
    z = xp.zeros_like(x[:, 0])
    return xp.stack(
        (xp.stack((2 * x[:, 0], 6 * x[:, 1]), axis=1), xp.stack((x[:, 1] + z, x[:, 0] + z), axis=1)),
        axis=1,
    )


def train(mode=SurrogateMode.AFFINE_COV_HESS, f_fn=f, g_fn=None, **kw):
    s = AFINNISurrogate()
    opts = dict(
        mode=mode, atol=0.05, rtol=0.05, error_metric="Linf",
        num_initial_clusters=4, min_pts_per_cluster=32, use_work_stealing=False,
    )  # fmt: skip
    opts.update(kw)
    s.train(xp.asarray(X), ClusteringConfig(**opts), Evaluator(f_fn=f_fn, g_fn=g_fn))
    return s


def small_config(atol, **kw):
    """atol = rtol, and few initial clusters, for the 3000-point quadratics of the retraining and profiler tests."""
    return ClusteringConfig(**{"mode": SurrogateMode.AFFINE_COV_HESS, "atol": atol, "rtol": atol,
                               "num_initial_clusters": 4, "min_pts_per_cluster": 40, **kw})


def predictions(s):
    out = []
    for k in (1, 4):
        p, m = s.predict(xp.asarray(Q), k_nn=k)
        out += [host(p), host(m)]
    return out


def assert_same_model(a, b, separate_runs=False):
    """Same model and predictions; ``separate_runs`` allows GPU rounding differences between trainings."""
    assert a.N_src == b.N_src and a.mode == b.mode
    assert type(a.local_model) is type(b.local_model)
    assert a.local_model.settings() == b.local_model.settings()
    assert set(a.leaf_params) == set(b.leaf_params)
    for x, y in zip(predictions(a), predictions(b), strict=True):
        if separate_runs and x.dtype != bool:
            assert_backend_equal(x, y, separate_runs=True)
        else:
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize(
    "mode,kw",
    [
        (SurrogateMode.AFFINE_COV_HESS, {}),
        (SurrogateMode.NEAREST_NEIGHBOR, {}),
        (SurrogateMode.TAYLOR_COV_HESS, {"g_fn": g}),
        (SurrogateMode.TAYLOR_LSQ_HESS, {"g_fn": g, "store_hess": True}),
    ],
)
def test_round_trip_preserves_predictions_and_metadata(tmp_path, mode, kw):
    s = train(mode, **kw)
    path = s.save_model(tmp_path / "model")
    assert path.suffix == ".npz" and path.exists()
    loaded = AFINNISurrogate.load_model(path)
    assert_same_model(s, loaded)
    assert loaded.config == json.loads(json.dumps(s.config))
    assert loaded.converged is True and loaded.profiling_stats["N0"] == len(X)
    np.testing.assert_array_equal(host(loaded.E_cluster_abs_src), host(s.E_cluster_abs_src))
    np.testing.assert_array_equal(host(loaded.E_cluster_rel_src), host(s.E_cluster_rel_src))
    np.testing.assert_array_equal(host(loaded.counts_src), host(s.counts_src))
    with np.load(path, allow_pickle=False) as data:  # no pickled objects in the file
        assert int(data["format_version"]) == MODEL_FORMAT_VERSION


def test_save_into_a_directory_uses_the_default_name(tmp_path):
    path = train().save_model(tmp_path)
    assert path == tmp_path / "afinni_surrogate.npz"


@pytest.mark.parametrize("given", ["model", "model.npz", "model.txt", "."])
def test_the_path_given_to_save_also_loads(tmp_path, given):
    s = train()
    s.save_model(tmp_path / given)
    assert_same_model(s, AFINNISurrogate.load_model(tmp_path / given))


def test_only_rank_0_writes(tmp_path, monkeypatch):
    import src.surrogate.model as su

    s = train()
    monkeypatch.setattr(su, "_get_mpi_rank", lambda: 1)
    assert s.save_model(tmp_path / "m") == tmp_path / "m.npz"
    assert not list(tmp_path.iterdir())


class ScaledAffine(LocalModel):
    """A custom model with its own column name and a setting."""

    name = "test_scaled_affine"

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def settings(self):
        return {"scale": self.scale}

    def columns(self, d, fd):
        return {"W": ParamSpec((fd, d)), "c0": ParamSpec((fd,))}

    def fit(self, data):
        p = AffineLSQModel().fit(data)
        return {"W": p["A_fit"] * self.scale, "c0": p["b_fit"] * self.scale}

    def evaluate(self, params, X, cluster_ids):
        out = AffineLSQModel().evaluate({"A_fit": params["W"], "b_fit": params["c0"]}, X, cluster_ids)
        return out / self.scale


def test_custom_model_columns_and_settings_round_trip(tmp_path):
    LOCAL_MODELS.register(ScaledAffine)
    try:
        s = train(local_model=ScaledAffine(scale=2.0))
        loaded = AFINNISurrogate.load_model(s.save_model(tmp_path / "m.npz"))
        assert set(loaded.leaf_params) == {"W", "c0"}
        assert loaded.local_model.settings() == {"scale": 2.0}
        assert_same_model(s, loaded)
    finally:
        del LOCAL_MODELS[ScaledAffine.name]

    # Unregistered: a clear error, or pass the instance explicitly.
    with pytest.raises(ValueError, match="not registered"):
        AFINNISurrogate.load_model(tmp_path / "m.npz")
    loaded = AFINNISurrogate.load_model(tmp_path / "m.npz", local_model=ScaledAffine(scale=2.0))
    assert_same_model(s, loaded)


def test_loading_rejects_missing_or_misshapen_columns(tmp_path):
    s = train()
    path = s.save_model(tmp_path / "m.npz")
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["param:A_fit"] = arrays["param:A_fit"][:, :, :1]
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="'A_fit'"):
        AFINNISurrogate.load_model(tmp_path / "bad.npz")
    arrays["format_version"] = np.asarray(MODEL_FORMAT_VERSION + 1)
    np.savez(tmp_path / "future.npz", **arrays)
    with pytest.raises(ValueError, match="newer than the supported"):
        AFINNISurrogate.load_model(tmp_path / "future.npz")


def test_unversioned_files_are_refused(tmp_path):
    np.savez(tmp_path / "old.npz", X_src=np.zeros((2, 2)), trained=True)
    with pytest.raises(ValueError, match="no format version"):
        AFINNISurrogate.load_model(tmp_path / "old.npz")


def test_files_with_the_old_single_error_column_still_load(tmp_path):
    s = train()
    with np.load(s.save_model(tmp_path / "m.npz")) as data:
        arrays = {k: data[k] for k in data.files if not k.startswith("E_cluster_")}
    arrays["E_cluster_src"] = host(s.E_cluster_rel_src).max(axis=1)
    np.savez(tmp_path / "old.npz", **arrays)
    loaded = AFINNISurrogate.load_model(tmp_path / "old.npz")
    assert loaded.E_cluster_abs_src is None and loaded.E_cluster_rel_src is None
    assert_same_model(s, loaded)


def test_failed_retraining_keeps_the_previous_model_usable(monkeypatch):
    """A failure while building the new kNN index must not destroy the old model."""
    import src.surrogate.model as su
    from src.surrogate import AFINNISurrogate

    X = np.random.default_rng(0).uniform(-1, 1, (3000, 2))
    s = AFINNISurrogate()
    s.train(xp.asarray(X), small_config(atol=0.05), Evaluator(f_fn=lambda x: (x**2).sum(1)))
    before = host(s.predict(xp.asarray(X[:50]))[0])

    def failing(_):
        raise RuntimeError("index construction failed")

    monkeypatch.setattr(su, "_make_knn_index", failing)
    with pytest.raises(RuntimeError, match="index construction failed"):
        s.train(xp.asarray(X), small_config(atol=0.02), Evaluator(f_fn=lambda x: (x**2).sum(1)))
    monkeypatch.undo()
    np.testing.assert_array_equal(host(s.predict(xp.asarray(X[:50]))[0]), before)


@pytest.mark.skipif(xp.__name__.endswith("cupy"), reason="CPU KD-tree")
def test_kdtree_build_failure_during_retraining_keeps_the_old_model(monkeypatch):
    """The tree is built before the new model is installed."""
    import scipy.spatial

    from src.surrogate import AFINNISurrogate

    X = np.random.default_rng(0).uniform(-1, 1, (3000, 2))
    s = AFINNISurrogate()
    s.train(X, small_config(atol=0.05), Evaluator(f_fn=lambda x: (x**2).sum(1)))
    before = s.predict(X[:50])[0]

    def failing(*a, **k):
        raise MemoryError("tree build failed")

    monkeypatch.setattr(scipy.spatial, "cKDTree", failing)
    with pytest.raises(MemoryError):
        s.train(X, small_config(atol=0.02), Evaluator(f_fn=lambda x: (x**2).sum(1)))
    monkeypatch.undo()
    np.testing.assert_array_equal(s.predict(X[:50])[0], before)


def _record_profilers(monkeypatch):
    """Capture every Profiler the engine creates, to check its stream afterwards."""
    import src.engine.adaptive_clustering as ac

    created = []

    class Recording(ac.Profiler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(ac, "Profiler", Recording)
    return created


def _train_small(s, f, **kw):
    X = np.random.default_rng(0).uniform(-1, 1, (3000, 2))
    s.train(xp.asarray(X), small_config(0.05, **kw), Evaluator(f_fn=f))


def test_a_failed_training_closes_its_profiler_stream(tmp_path, monkeypatch):
    """The profiler file must not stay open when the oracle raises during training."""
    from src.surrogate import AFINNISurrogate

    created = _record_profilers(monkeypatch)

    def failing(_x):
        raise ValueError("oracle failed")

    with pytest.raises(ValueError, match="oracle failed"):
        AFINNISurrogate().train(xp.asarray(np.zeros((100, 2))), small_config(0.05, profile_dir=tmp_path), Evaluator(f_fn=failing))
    assert created and all(p._stream is None for p in created)


def test_a_nested_predict_does_not_release_the_outer_profiler(tmp_path, monkeypatch):
    """A predict() inside an oracle cleans up only its own scope; the outer stream is closed by the outer failure."""
    from src.surrogate import AFINNISurrogate

    inner = AFINNISurrogate()
    _train_small(inner, lambda x: (x**2).sum(1))
    created = _record_profilers(monkeypatch)

    def oracle(x):
        inner.predict(xp.asarray(x[:5]))
        raise ValueError("outer oracle failed")

    with pytest.raises(ValueError, match="outer oracle failed"):
        AFINNISurrogate().train(xp.asarray(np.zeros((100, 2))), small_config(0.05, profile_dir=tmp_path), Evaluator(f_fn=oracle))
    assert created and all(p._stream is None for p in created)


def test_an_inner_failure_caught_by_the_oracle_leaves_the_outer_operation_running(tmp_path, monkeypatch):
    from src.surrogate import AFINNISurrogate

    inner = AFINNISurrogate()
    _train_small(inner, lambda x: (x**2).sum(1))
    created = _record_profilers(monkeypatch)

    def oracle(x):
        with pytest.raises(ValueError):
            inner.predict(xp.zeros((2, 7)))  # wrong width: the inner call fails and is caught
        return (x**2).sum(1)

    _train_small(AFINNISurrogate(), oracle, profile_dir=str(tmp_path))
    assert created and all(p._stream is None for p in created)


def test_public_methods_keep_signatures_and_predict_reports_search_stats():
    import inspect

    from src.surrogate import AFINNISurrogate

    assert list(inspect.signature(AFINNISurrogate.train).parameters) == ["self", "X_train", "config", "evaluator", "Y_train", "initial_labels"]
    assert "k_nn" in inspect.signature(AFINNISurrogate.predict).parameters
    assert AFINNISurrogate.predict.__doc__ and AFINNISurrogate.train.__doc__
    X = np.random.default_rng(0).uniform(-1, 1, (3000, 2))
    s = AFINNISurrogate()
    s.train(xp.asarray(X), small_config(atol=0.05), Evaluator(f_fn=lambda x: (x**2).sum(1)))
    s.predict(xp.asarray(X[:700]), batch_size=300)  # three batches
    assert s.search_stats["queries"] == 700 and s.search_stats["fallback_rows"] >= 0


def test_nested_empty_collectives_preserve_the_parent_scope():
    import src.engine.adaptive_clustering as ac

    def outer():
        own_scope = ac._PROFILER_SCOPES[-1]
        ac.run_collective(lambda: None, "inner")
        assert ac._PROFILER_SCOPES[-1] is own_scope

    ac.run_collective(outer, "outer")
    assert not ac._PROFILER_SCOPES


def test_component_preparation_can_predict_without_corrupting_training_scope(tmp_path, monkeypatch):
    import src.engine.adaptive_clustering as ac
    from src.components.local_models import ConstantModel

    inner = AFINNISurrogate()
    _train_small(inner, lambda x: xp.ones((len(x), 1)), mode=SurrogateMode.NEAREST_NEIGHBOR)

    class PreparedConstant(ConstantModel):
        """Predict during component setup, before the outer engine creates its profiler."""

        name = "prepared_constant_scope_test"

        def __init__(self):
            inner.predict(xp.zeros((1, 2)))

    monkeypatch.setitem(LOCAL_MODELS, PreparedConstant.name, PreparedConstant)
    outer = AFINNISurrogate()
    _train_small(outer, lambda x: xp.ones((len(x), 1)), local_model=PreparedConstant.name,
                 mode=SurrogateMode.NEAREST_NEIGHBOR, profile_dir=str(tmp_path))
    assert outer.trained
    assert not ac._PROFILER_SCOPES


def test_training_writes_no_files_and_the_clustering_cache_flags_are_gone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train()
    assert not list(tmp_path.iterdir())
    for flag in ("save_initial_clustering", "use_existing_initial_clustering"):
        with pytest.raises(TypeError, match=flag):
            ClusteringConfig(atol=0.1, **{flag: True})
