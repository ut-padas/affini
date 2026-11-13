"""Tests for src/components/local_models.py and clusters.ColumnBuffer."""

import logging

import numpy as np
import pytest

from src import _to_host, xp
from src.config import ClusteringConfig
from src.components.modes import SurrogateMode
from src.core.clusters import ColumnBuffer, ParamSpec
from src.components.local_models import (
    LOCAL_MODELS,
    AffineLSQModel,
    ClusterFitData,
    ConstantModel,
    LocalModel,
    QuadraticLSQModel,
    TaylorModel,
    make_local_model,
    output_columns,
)
from tests._helpers import assert_backend_equal

K, d, fd, n, k = 7, 3, 2, 11, 4
rng = np.random.default_rng(0)
PARAMS = {
    "centroids": rng.standard_normal((K, d)),
    "f": rng.standard_normal((K, fd)),
    "g": rng.standard_normal((K, fd, d)),
    "H": rng.standard_normal((K, fd, d, d)),  # one Hessian per output component
    "A_fit": rng.standard_normal((K, fd, d)),
    "b_fit": rng.standard_normal((K, fd)),
}
Q = rng.standard_normal((n, d))
NN = rng.integers(0, K, (n, k))


def _p(*keys):
    return {key: xp.asarray(PARAMS[key]) for key in keys}


def _surrogate_reference(kind):
    """
    The (query, neighbour) predictions as the surrogate computed them before
    LocalModel, except that the second-order term now uses one Hessian per output
    component (the old code reshaped a single (d, d) Hessian per leaf).
    """
    flat = NN.reshape(-1)
    f_nb = PARAMS["f"][flat].reshape(n, k, fd)
    dx = Q[:, None, :] - PARAMS["centroids"][NN]
    if kind == "constant":
        return f_nb
    if kind in ("taylor", "taylor2"):
        J = PARAMS["g"][flat].reshape(n, k, fd, d)
        out = f_nb + np.einsum("nkfd,nkd->nkf", J, dx)
        if kind == "taylor2":
            H = PARAMS["H"][NN]  # (n, k, fd, d, d): every component has its own H
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
    assert_backend_equal(out, _surrogate_reference(kind))


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


def _fit_data(X, F, sizes, g_c=None, G=None):
    sizes = np.asarray(sizes)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    centroids = np.add.reduceat(X, starts) / sizes[:, None]
    return ClusterFitData(
        X=xp.asarray(X),
        F=xp.asarray(F),
        G=None if G is None else xp.asarray(G),
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


def test_quadratic_fit_recovers_a_quadratic_per_cluster_far_from_the_origin():
    """Exact on quadratics, also for small clusters far from the origin (the fit is centred and scaled)."""
    sizes = [40, 25, 60]
    labels = np.repeat(np.arange(len(sizes)), sizes)
    X = 1e3 + 1e-3 * rng.standard_normal((sum(sizes), d))
    H = rng.standard_normal((len(sizes), fd, d, d))
    a, b = rng.standard_normal((len(sizes), fd, d)), rng.standard_normal((len(sizes), fd))
    u = (X - 1e3) / 1e-3
    F = 1e-3 * (np.einsum("nd,nfde,ne->nf", u, H[labels], u) + np.einsum("nfd,nd->nf", a[labels], u)) + b[labels]
    model = QuadraticLSQModel()
    data = _fit_data(X, F, sizes)
    params = model.fit(data)
    assert params.keys() == model.columns(d, fd).keys() and model.n_features(d) == (d + 1) * (d + 2) // 2
    full = {**params, "centroids": data.centroids, "f": data.f_c}
    np.testing.assert_allclose(_to_host(model.evaluate(full, data.X, data.labels)), F, rtol=0, atol=1e-9)
    # The kNN layout, (n, k) cluster ids per query, evaluates each candidate cluster.
    ids = xp.asarray(np.stack([labels, (labels + 1) % 3], axis=1))
    both = _to_host(model.evaluate(full, data.X, ids))
    np.testing.assert_allclose(both[:, 0], F, rtol=0, atol=1e-9)


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
    assert {"centroids", "f", "Q", "E_cluster_abs", "E_cluster_rel", "counts"} <= cols.keys()
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


def test_second_order_finalize_fits_one_hessian_per_output():
    # Two outputs with opposite and independent curvature, f_j = f_j(c) + g_j.dx
    # + 1/2 dx^T A_j dx on every cluster. A single Hessian shared by the outputs
    # (the old behaviour) cannot represent this; per-output Hessians recover it.
    sizes = [40, 70]
    A = rng.standard_normal((fd, d, d))
    A = A + A.transpose(0, 2, 1)
    A[1] = -A[0] + np.diag(np.arange(1.0, d + 1))
    X = rng.standard_normal((sum(sizes), d))
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    labels = np.repeat(np.arange(len(sizes)), sizes)
    centroids = np.add.reduceat(X, starts) / np.asarray(sizes)[:, None]
    g_c = rng.standard_normal((len(sizes), fd, d))
    dx = X - centroids[labels]
    F = np.einsum("nfd,nd->nf", g_c[labels], dx) + 0.5 * np.einsum("nc,fcd,nd->nf", dx, A, dx)
    G = g_c[labels] + np.einsum("fcd,nd->nfc", A, dx)  # exact gradients
    data = _fit_data(X, F, sizes, g_c=g_c, G=G)
    data.centroids, data.f_c = xp.asarray(centroids), xp.zeros((len(sizes), fd))

    model = TaylorModel(second_order=True)
    assert model.needs_point_grad and model.needs_centroid_grad
    params = model.fit(data)
    final = model.finalize(data, params)
    assert set(params) | set(final) == set(model.columns(d, fd))
    H = _to_host(final["H"])
    assert H.shape == (len(sizes), fd, d, d)
    np.testing.assert_allclose(H, np.broadcast_to(A, H.shape), atol=1e-8)

    leaf = {**params, **final, "centroids": data.centroids, "f": data.f_c}
    np.testing.assert_allclose(_to_host(model.evaluate(leaf, data.X, data.labels)), F, atol=1e-8)


def test_first_order_models_do_not_need_point_gradients():
    assert not TaylorModel().needs_point_grad
    assert not AffineLSQModel().needs_point_grad and not ConstantModel().needs_point_grad
    assert TaylorModel().finalize(None, {}) == {}


def test_second_order_finalize_requires_point_gradients():
    sizes = [5, 6]
    X, F = rng.standard_normal((11, d)), rng.standard_normal((11, fd))
    data = _fit_data(X, F, sizes, g_c=rng.standard_normal((2, fd, d)))
    with pytest.raises(RuntimeError, match="gradients at every point"):
        TaylorModel(second_order=True).finalize(data, {})


def test_evaluate_rejects_a_shared_hessian():
    params = {**_p("centroids", "f", "g"), "H": xp.asarray(rng.standard_normal((K, d, d)))}
    with pytest.raises(ValueError, match=r"\(K, fd, d, d\)"):
        TaylorModel(second_order=True).evaluate(params, xp.asarray(Q), xp.asarray(NN))


def test_builtin_models_are_registered_and_rebuildable():
    for model in (ConstantModel(), TaylorModel(), TaylorModel(True), AffineLSQModel(), QuadraticLSQModel()):
        assert LOCAL_MODELS[model.name] is type(model)
        rebuilt = LOCAL_MODELS.create((model.name, model.settings()))
        assert type(rebuilt) is type(model) and rebuilt.settings() == model.settings()


def test_reserved_column_names_are_rejected():
    class Clashing(LocalModel):
        name = "clashing"

        def columns(self, d, fd):
            return {"Q": ParamSpec((d, d))}

        def fit(self, data):
            return {}

        def evaluate(self, params, X, cluster_ids):
            return params["f"][cluster_ids]

    with pytest.raises(ValueError, match="reserved column"):
        output_columns(Clashing(), d, fd)


def test_column_buffer_checks_trailing_shapes():
    buf = ColumnBuffer({"a": ParamSpec((2, 3))}, capacity=4)
    with pytest.raises(ValueError, match="declared \\(2, 3\\)"):
        buf.append({"a": np.zeros((1, 3, 2))})
    with pytest.raises(ValueError, match="declared"):
        buf.append({"a": np.zeros((1, 2))})  # would silently broadcast otherwise
