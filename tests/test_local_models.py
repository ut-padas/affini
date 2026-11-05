"""Tests for src/local_models.py and clusters.ColumnBuffer."""

import logging

import numpy as np
import pytest

from src import _to_host, xp
from src.adaptive_clustering_utils import ClusteringConfig, SurrogateMode
from src.clusters import ColumnBuffer, ParamSpec
from src.local_models import (
    AffineLSQModel,
    ClusterFitData,
    ConstantModel,
    TaylorModel,
    make_local_model,
    output_columns,
)

K, d, fd, n, k = 7, 3, 2, 11, 4
rng = np.random.default_rng(0)
PARAMS = {
    "centroids": rng.standard_normal((K, d)),
    "f": rng.standard_normal((K, fd)),
    "g": rng.standard_normal((K, fd, d)),
    "H": rng.standard_normal((K, d, d)),
    "A_fit": rng.standard_normal((K, fd, d)),
    "b_fit": rng.standard_normal((K, fd)),
}
Q = rng.standard_normal((n, d))
NN = rng.integers(0, K, (n, k))


def _p(*keys):
    return {key: xp.asarray(PARAMS[key]) for key in keys}


def _surrogate_reference(kind):
    """The (query, neighbour) predictions as src/surrogate.py computed them before LocalModel."""
    flat = NN.reshape(-1)
    f_nb = PARAMS["f"][flat].reshape(n, k, fd)
    dx = Q[:, None, :] - PARAMS["centroids"][NN]
    if kind == "constant":
        return f_nb
    if kind in ("taylor", "taylor2"):
        J = PARAMS["g"][flat].reshape(n, k, fd, d)
        out = f_nb + np.einsum("nkfd,nkd->nkf", J, dx)
        if kind == "taylor2":
            H = PARAMS["H"][flat].reshape(n, k, 1, d, d)
            out += 0.5 * np.einsum("nkc,nkfcd,nkd->nkf", dx, H, dx)
        return out
    A = PARAMS["A_fit"][flat].reshape(n, k, fd, d)
    return np.einsum("nkfd,nd->nkf", A, Q) + PARAMS["b_fit"][flat].reshape(n, k, fd)


@pytest.mark.parametrize(
    "kind,model,keys",
    [
        ("constant", ConstantModel(), ("centroids", "f")),
        ("taylor", TaylorModel(), ("centroids", "f", "g")),
        ("taylor2", TaylorModel(second_order=True), ("centroids", "f", "g", "H")),
        ("affine", AffineLSQModel(), ("centroids", "f", "A_fit", "b_fit")),
    ],
)
def test_inference_path_matches_previous_surrogate(kind, model, keys):
    out = _to_host(model.evaluate(_p(*keys), xp.asarray(Q), xp.asarray(NN)))
    assert out.shape == (n, k, fd)
    np.testing.assert_array_equal(out, _surrogate_reference(kind))


@pytest.mark.parametrize(
    "model,keys",
    [
        (ConstantModel(), ("centroids", "f")),
        (TaylorModel(), ("centroids", "f", "g")),
        (TaylorModel(second_order=True), ("centroids", "f", "g", "H")),
        (AffineLSQModel(), ("centroids", "f", "A_fit", "b_fit")),
    ],
)
def test_training_and_inference_paths_agree(model, keys):
    """Evaluating one (query, neighbour) pair per row equals the broadcast (n, k) call."""
    params = _p(*keys)
    pairs = model.evaluate(
        params, xp.asarray(np.repeat(Q, k, axis=0)), xp.asarray(NN.reshape(-1))
    )
    grid = model.evaluate(params, xp.asarray(Q), xp.asarray(NN))
    np.testing.assert_allclose(
        _to_host(pairs).reshape(n, k, fd), _to_host(grid), rtol=1e-12, atol=1e-12
    )


def _fit_data(X, F, sizes, g_c=None):
    sizes = np.asarray(sizes)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    centroids = np.add.reduceat(X, starts) / sizes[:, None]
    return ClusterFitData(
        X=xp.asarray(X),
        F=xp.asarray(F),
        G=None,
        labels=xp.asarray(labels),
        starts=xp.asarray(starts),
        sizes=xp.asarray(sizes),
        centroids=xp.asarray(centroids),
        f_c=xp.asarray(F[starts]),
        g_c=None if g_c is None else xp.asarray(g_c),
    )


def test_affine_fit_recovers_an_affine_function():
    sizes = [40, 25, 60]
    X = rng.standard_normal((sum(sizes), d))
    A_true = rng.standard_normal((len(sizes), fd, d))
    b_true = rng.standard_normal((len(sizes), fd))
    labels = np.repeat(np.arange(len(sizes)), sizes)
    F = np.einsum("nfd,nd->nf", A_true[labels], X) + b_true[labels]
    model = AffineLSQModel()
    data = _fit_data(X, F, sizes)
    params = model.fit(data)
    assert params.keys() == model.columns(d, fd).keys()
    np.testing.assert_allclose(_to_host(params["A_fit"]), A_true, atol=1e-5)
    np.testing.assert_allclose(_to_host(params["b_fit"]), b_true, atol=1e-5)
    F_hat = model.evaluate(
        {**params, "centroids": data.centroids, "f": data.f_c}, data.X, data.labels
    )
    np.testing.assert_allclose(_to_host(F_hat), F, atol=1e-5)


def test_fit_returns_declared_columns():
    sizes = [5, 6]
    X, F = rng.standard_normal((11, d)), rng.standard_normal((11, fd))
    g_c = rng.standard_normal((2, fd, d))
    assert ConstantModel().fit(_fit_data(X, F, sizes)) == {}
    taylor = TaylorModel().fit(_fit_data(X, F, sizes, g_c))
    assert taylor.keys() == TaylorModel().columns(d, fd).keys()
    np.testing.assert_array_equal(_to_host(taylor["g"]), g_c)
    # The second-order Hessian is only estimated for accepted clusters, so fit omits it.
    assert set(TaylorModel(True).columns(d, fd)) - set(
        TaylorModel(True).fit(_fit_data(X, F, sizes, g_c))
    ) == {"H"}


@pytest.mark.parametrize("mode", list(SurrogateMode))
def test_every_mode_has_a_model(mode):
    model = make_local_model(ClusteringConfig(mode=mode, atol=1e-3, rtol=1e-3))
    cols = output_columns(model, d, fd)
    assert {"centroids", "f", "Q", "E_cluster", "radii", "counts"} <= cols.keys()
    assert model.needs_centroid_grad == mode.value.startswith("taylor")


def test_store_hess_is_ignored_outside_taylor_lsq(caplog):
    with caplog.at_level(logging.WARNING):
        cfg = ClusteringConfig(
            mode=SurrogateMode.TAYLOR_COV_HESS, atol=1e-3, rtol=1e-3, store_hess=True
        )
    assert cfg.store_hess is False and "ignoring it" in caplog.text
    lsq = ClusteringConfig(
        mode=SurrogateMode.TAYLOR_LSQ_HESS, atol=1e-3, rtol=1e-3, store_hess=True
    )
    assert "H" in output_columns(make_local_model(lsq), d, fd)


def test_column_buffer_grows_and_checks_columns():
    cols = {"a": ParamSpec((2,)), "c": ParamSpec((), np.int64)}
    buf = ColumnBuffer(cols, capacity=1)
    chunks = [(rng.standard_normal((m, 2)), np.arange(m)) for m in (3, 1, 5)]
    for a, c in chunks:
        buf.append({"a": xp.asarray(a), "c": xp.asarray(c)})
    out = buf.result()
    np.testing.assert_array_equal(out["a"], np.concatenate([a for a, _ in chunks]))
    np.testing.assert_array_equal(out["c"], np.concatenate([c for _, c in chunks]))
    assert out["c"].dtype == np.int64 and buf.n_rows == 9

    with pytest.raises(KeyError, match="missing \\['c'\\]"):
        buf.append({"a": np.zeros((1, 2))})
    with pytest.raises(KeyError, match="unexpected \\['z'\\]"):
        buf.append(
            {"a": np.zeros((1, 2)), "c": np.zeros(1, np.int64), "z": np.zeros(1)}
        )
    with pytest.raises(ValueError, match="different numbers of rows"):
        buf.append({"a": np.zeros((2, 2)), "c": np.zeros(1, np.int64)})
