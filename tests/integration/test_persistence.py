"""
Saving and loading surrogates: the versioned format, files written before it
existed, custom local models, and the deprecated train(save_model=...) adapter.
"""

import json

import numpy as np
import pytest

from src import _to_host, xp
from src.clusters import ParamSpec
from src.local_models import LOCAL_MODELS, AffineLSQModel, LocalModel
from src.modes import SurrogateMode
from src.surrogate import AFINNISurrogate, MODEL_FORMAT_VERSION
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


def train(mode=SurrogateMode.AFFINE_COV_HESS, f_fn=f, **kw):
    s = AFINNISurrogate()
    opts = dict(
        mode=mode, f_fn=f_fn, ATOL=0.05, RTOL=0.05, ERROR_METRIC="Linf-local",
        NUM_INITIAL_CLUSTERS=4, MIN_PTS_PER_CLUSTER=32, USE_WORK_STEALING=False,
    )  # fmt: skip
    opts.update(kw)
    s.train(xp.asarray(X), **opts)
    return s


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
        (SurrogateMode.TAYLOR_LSQ_HESS, {"g_fn": g, "STORE_HESSIAN": True}),
    ],
)
def test_round_trip_preserves_predictions_and_metadata(tmp_path, mode, kw):
    s = train(mode, **kw)
    path = s.save(tmp_path / "model")
    assert path.suffix == ".npz" and path.exists()
    loaded = AFINNISurrogate.load_model(path)
    assert_same_model(s, loaded)
    assert loaded.config == json.loads(json.dumps(s.config))
    assert loaded.converged is True and loaded.profiling_stats["N0"] == len(X)
    np.testing.assert_array_equal(host(loaded.E_cluster_src), host(s.E_cluster_src))
    np.testing.assert_array_equal(host(loaded.counts_src), host(s.counts_src))
    with np.load(path, allow_pickle=False) as data:  # no pickled objects in the file
        assert int(data["format_version"]) == MODEL_FORMAT_VERSION


def test_save_into_a_directory_uses_the_default_name(tmp_path):
    path = train().save(tmp_path)
    assert path == tmp_path / "afinni_surrogate.npz"


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
        loaded = AFINNISurrogate.load_model(s.save(tmp_path / "m.npz"))
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
    path = s.save(tmp_path / "m.npz")
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


def _write_legacy(s, path, shared_hessian=None):
    """A file in the layout save() wrote before versioning (pickled object arrays)."""
    H = None
    if s.Hs_src is not None:
        H = shared_hessian if shared_hessian is not None else _to_host(s.Hs_src)[:, 0]
    save_dict = {
        "trained": True,
        "mode": s.mode.value,
        "N_src": s.N_src,
        "X_src": _to_host(s.X_src),
        "f_src": _to_host(s.f_src),
        "J_src": _to_host(s.J_src) if s.J_src is not None else None,
        "trust_radii_src": np.ones(s.N_src),
        "Qs_src": _to_host(s.Qs_src),
        "Hs_src": H,
        "A_fits_src": _to_host(s.A_fits_src) if s.A_fits_src is not None else None,
        "b_fits_src": _to_host(s.b_fits_src) if s.b_fits_src is not None else None,
        "profiling_stats": {"N0": len(X)},
        "tree_data": None,
    }
    np.savez_compressed(str(path), **save_dict, allow_pickle=True)


@pytest.mark.parametrize(
    "mode,kw",
    [
        (SurrogateMode.AFFINE_COV_HESS, {}),
        (SurrogateMode.TAYLOR_COV_HESS, {"g_fn": g}),
    ],
)
def test_files_from_before_versioning_still_load(tmp_path, mode, kw):
    s = train(mode, **kw)
    _write_legacy(s, tmp_path / "old.npz")
    loaded = AFINNISurrogate.load_model(tmp_path / "old.npz")
    assert_same_model(s, loaded)


def test_legacy_scalar_hessians_gain_an_output_axis(tmp_path):
    s = train(SurrogateMode.TAYLOR_LSQ_HESS, f_fn=lambda x: f(x)[:, :1], g_fn=lambda x: g(x)[:, :1], STORE_HESSIAN=True)
    _write_legacy(s, tmp_path / "old.npz")
    loaded = AFINNISurrogate.load_model(tmp_path / "old.npz")
    assert host(loaded.Hs_src).shape == (s.N_src, 1, 2, 2)
    assert_same_model(s, loaded)


def test_legacy_shared_hessian_with_vector_output_is_refused(tmp_path):
    s = train(SurrogateMode.TAYLOR_LSQ_HESS, g_fn=g, STORE_HESSIAN=True)
    _write_legacy(s, tmp_path / "old.npz", shared_hessian=np.zeros((s.N_src, 2, 2)))
    with pytest.raises(ValueError, match="shared by all 2 outputs"):
        AFINNISurrogate.load_model(tmp_path / "old.npz")


# --- Deprecated train(save_model=..., load_if_exists=...) -------------------------


def _forbidden(x):
    raise AssertionError("the cached model should have been loaded instead of training")


def test_deprecated_save_model_saves_then_loads_instead_of_training(tmp_path):
    path = tmp_path / "cache.npz"
    with pytest.warns(DeprecationWarning, match="save_model"):
        s = train(save_model=path)
    assert path.exists()
    with pytest.warns(DeprecationWarning):
        cached = train(f_fn=_forbidden, save_model=path)
    assert_same_model(s, cached)
    with pytest.warns(DeprecationWarning):
        retrained = train(save_model=path, load_if_exists=False)
    assert_same_model(s, retrained, separate_runs=True)


def test_deprecated_save_model_false_and_true(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.warns(DeprecationWarning):
        train(save_model=False)
    assert not list(tmp_path.iterdir())
    with pytest.warns(DeprecationWarning):
        s = train(save_model=True)
    assert (tmp_path / "afinni_surrogate.npz").exists()
    # The default file name before the rename is still found.
    (tmp_path / "afinni_surrogate.npz").rename(tmp_path / "hatann_surrogate.npz")
    with pytest.warns(DeprecationWarning):
        cached = train(f_fn=_forbidden, save_model=True)
    assert_same_model(s, cached)


def test_no_deprecation_warning_without_the_old_arguments(recwarn):
    train()
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
