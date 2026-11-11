"""
The function fitted on each cluster (leaf) of the AFINNI forest.

A ``LocalModel`` owns four things:

- ``fit``: fit the model on every cluster of a batch at once. The refinement loop
  evaluates the fitted model at the cluster's points for the acceptance test.
- ``finalize`` (optional): extra parameters computed only for *accepted* clusters,
  after the acceptance test (e.g. the Hessian of a second-order Taylor model).
  They are used at inference but not by the acceptance test.
- ``evaluate``: evaluate the models at given points, used both for the acceptance
  test during training and for inference;
- ``columns``: every per-cluster parameter stored for accepted clusters, i.e. the
  keys returned by ``fit`` plus those returned by ``finalize``.

Adding a model (e.g. a local quadratic or kernel ridge regression) means writing
one subclass and registering it in ``LOCAL_MODELS``; the refinement loop, output
buffers, MPI gathering, prediction and ``AFINNISurrogate.save_model``/``load_model``
only see the columns it declares. A model declares the data it needs with
``needs_centroid_value`` (f at the centroids, ``ClusterFitData.f_c``),
``needs_centroid_grad`` (gradients at the centroids, ``ClusterFitData.g_c``) and
``needs_point_grad`` (gradients at every point, ``ClusterFitData.G``); the loop
evaluates and carries gradients only when the model or the split strategy needs
them. AFINNI mode always evaluates ``f_c`` (it stores it as column ``"f"``);
region mode evaluates it only when some component declares it. ``settings()`` returns the constructor arguments, so a saved model can be
rebuilt by name.

Performance contract: ``fit``, ``finalize`` and ``evaluate`` are fully vectorized
over all clusters / points in one call, with no Python loops over points or
clusters, and keep arrays on the active backend (``xp``). ``evaluate`` accepts
``cluster_ids`` of shape (N,) (one cluster per point, in training) or (n, k) (k
candidate clusters per query, at inference); in the (n, k) case ``X`` stays
(n, d) and is broadcast, never copied k times.

Besides a model's own parameters, ``evaluate`` may read the columns every
cluster stores: ``"centroids"`` (K, d) and ``"f"`` (K, fd), the function value at
the centroid. Those names, and ``"Q"``, ``"E_cluster_abs"``, ``"E_cluster_rel"`` and
``"counts"``, are reserved (``SHARED_COLUMNS``).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src import Array, xp
from src.clusters import ParamSpec
from src.modes import SurrogateMode
from src.numerics import (
    TINY_EPS,
    _reduce_sum_1d,
    _symmetric_point_covariance,
    run_fused_cross_cov,
    run_fused_stats,
    solve_regularized_covariance_inplace,
)
from src.registry import Registry
from src.utils import repeats


@dataclass
class ClusterFitData:
    """A batch of clusters to fit, stored contiguously on the device."""

    X: Array  # (N, d) points
    F: Array  # (N, fd) function values
    G: Array | None  # (N, fd, d) gradients, when carried per point
    labels: Array  # (N,) cluster index of every point
    starts: Array  # (K,) first row of every cluster
    sizes: Array  # (K,) points per cluster
    centroids: Array  # (K, d)
    f_c: Array  # (K, fd) f at the centroids
    g_c: Array | None  # (K, fd, d) gradient at the centroids, if evaluated


class LocalModel(ABC):
    """
    The form of the approximation f_hat_c used on each leaf cluster c
    (constant, Taylor, affine least squares, ...).

    An instance holds no per-cluster state: ``fit`` fits K clusters at once and
    returns their parameters stacked along a leading (K,) axis, and ``evaluate``
    evaluates each point with the parameters of the cluster it is assigned to,
    so training and inference stay vectorized over clusters.
    """

    name: str
    needs_centroid_value: bool = True  # fit/evaluate read ClusterFitData.f_c (column "f")
    needs_centroid_grad: bool = False  # fit/evaluate need ClusterFitData.g_c
    needs_point_grad: bool = False  # fit/finalize need ClusterFitData.G

    def settings(self) -> dict:
        """Constructor arguments, so ``LOCAL_MODELS.create((name, settings))`` rebuilds it."""
        return {}

    @abstractmethod
    def columns(self, d: int, fd: int) -> dict[str, ParamSpec]:
        """Per-cluster parameters stored for every accepted cluster (fit + finalize)."""

    @abstractmethod
    def fit(self, data: ClusterFitData) -> dict[str, Array]:
        """Fit every cluster; returns parameters with a leading (K,) axis."""

    def finalize(self, data: ClusterFitData, params: dict[str, Array]) -> dict[str, Array]:
        """
        Extra parameters of *accepted* clusters, computed after the acceptance test.

        ``data`` holds only the accepted clusters and ``params`` their ``fit``
        output. Returns the columns not produced by ``fit`` (none by default).
        """
        return {}

    @abstractmethod
    def evaluate(self, params: dict[str, Array], X: Array, cluster_ids: Array) -> Array:
        """
        Evaluate cluster ``cluster_ids[i]``'s model at ``X[i]``.

        ``cluster_ids`` (N,) with ``X`` (N, d) returns (N, fd); ``cluster_ids``
        (n, k) with ``X`` (n, d) returns (n, k, fd). ``params`` holds this model's
        parameters plus the shared ``"centroids"`` and ``"f"`` columns. The
        acceptance test passes only the ``fit`` output, so columns added by
        ``finalize`` may be absent.
        """


LOCAL_MODELS: Registry[LocalModel] = Registry("local model", LocalModel)


@LOCAL_MODELS.register
class ConstantModel(LocalModel):
    """f_hat(x) = f(c): nearest-neighbor interpolation (``NEAREST_NEIGHBOR``)."""

    name = "constant"

    def columns(self, d, fd):
        return {}

    def fit(self, data):
        return {}

    def evaluate(self, params, X, cluster_ids):
        return params["f"][cluster_ids]


def lsq_hessians(
    X: Array, G: Array, centroids: Array, g_c: Array, labels: Array, counts: Array
) -> Array:
    """
    Symmetric Hessian H (K, fd, d, d) of every output component of each cluster,
    fitted by least squares to the gradient residuals ``G - g_c`` of that
    component: ``C H_j + H_j C = S_j`` (see ``lsq_hessian_in_eigenbasis``), with
    ``C`` the cluster's point covariance. Used as the second-order Taylor term.
    """
    K = len(counts)
    d, fd = X.shape[1], G.shape[1]
    Cov = _symmetric_point_covariance(X, centroids, labels, K)
    dvals, U = xp.linalg.eigh(Cov)
    dvals = xp.maximum(dvals, TINY_EPS)
    denom = dvals[:, :, None] + dvals[:, None, :]  # (K, d, d)

    Hs = xp.empty((K, fd, d, d), dtype=xp.float64)
    # One output component at a time, to keep the gradient interaction small.
    for j in range(fd):
        ac = xp.full(K, j, dtype=xp.int32)
        _, E_j = run_fused_stats(
            X, centroids, labels, K, d, fd, ac, G=G, g_mus=g_c, calc_E=True
        )
        S_j = E_j + E_j.transpose(0, 2, 1)
        F_mat = (U.transpose(0, 2, 1) @ S_j @ U) / denom  # H_j in the eigenbasis of C
        H_j = U @ F_mat @ U.transpose(0, 2, 1)
        Hs[:, j] = 0.5 * (H_j + H_j.transpose(0, 2, 1))
    return Hs


@LOCAL_MODELS.register
class TaylorModel(LocalModel):
    """
    f_hat_j(x) = f_j(c) + g_j(c) (x - c)  [+ 0.5 (x - c)^T H_j (x - c) if ``second_order``].

    Uses the gradient at the centroid (``TAYLOR_*`` modes). With ``second_order``,
    every output component ``j`` gets its own Hessian ``H_j``, column ``"H"`` of
    shape (fd, d, d), fitted to the per-point gradients of accepted clusters in
    ``finalize``: it is not part of the acceptance test, and it needs gradients at
    every point.
    """

    name = "taylor"
    needs_centroid_grad = True

    def __init__(self, second_order: bool = False):
        self.second_order = bool(second_order)

    @property
    def needs_point_grad(self) -> bool:
        return self.second_order

    def settings(self):
        return {"second_order": self.second_order}

    def columns(self, d, fd):
        cols = {"g": ParamSpec((fd, d))}
        if self.second_order:
            cols["H"] = ParamSpec((fd, d, d))
        return cols

    def fit(self, data):
        assert data.g_c is not None, "TaylorModel needs the gradient at the centroids"
        return {"g": data.g_c}

    def finalize(self, data, params):
        """Per-output Hessians of accepted clusters; needs ``data.G``."""
        if not self.second_order:
            return {}
        if data.G is None:
            raise RuntimeError("A second-order TaylorModel needs gradients at every point.")
        return {
            "H": lsq_hessians(
                data.X, data.G, data.centroids, data.g_c, data.labels, data.sizes
            )
        }

    def evaluate(self, params, X, cluster_ids):
        """First-order prediction, plus the quadratic term when an ``"H"`` column is present."""
        f = params["f"][cluster_ids]
        g = params["g"][cluster_ids]
        c = params["centroids"][cluster_ids]
        if "H" in params and params["H"].ndim != 4:
            raise ValueError(
                f"Taylor Hessians must have shape (K, fd, d, d); got {params['H'].shape}."
            )
        if cluster_ids.ndim == 1:
            dx = X - c
            out = f + xp.einsum("nmd,nd->nm", g, dx)
            if "H" in params:
                H = params["H"][cluster_ids]  # (N, fd, d, d)
                out = out + 0.5 * xp.einsum("nc,nfcd,nd->nf", dx, H, dx)
            return out
        dx = X[:, None, :] - c
        out = f + xp.einsum("nkfd,nkd->nkf", g, dx)
        if "H" in params:
            H = params["H"][cluster_ids]  # (n, k, fd, d, d)
            out += 0.5 * xp.einsum("nkc,nkfcd,nkd->nkf", dx, H, dx)
        return out


@LOCAL_MODELS.register
class AffineLSQModel(LocalModel):
    """
    f_hat(x) = A x + b, a least-squares fit to the cluster's points with adaptive
    Tikhonov regularization (``AFFINE_*`` modes; Equation (3) in the paper, with
    the intercept in absolute rather than centered coordinates).
    """

    name = "affine_lsq"
    needs_centroid_value = False

    def columns(self, d, fd):
        return {"A_fit": ParamSpec((fd, d)), "b_fit": ParamSpec((fd,))}

    def fit(self, data):
        K, (_, d), fd = len(data.sizes), data.X.shape, data.F.shape[1]
        mean_F = _reduce_sum_1d(data.F, data.starts) / data.sizes[:, None]  # (K, fd)
        auto_cov, _ = run_fused_stats(
            data.X, data.centroids, data.labels, K, d, fd, None, calc_E=False
        )  # (K, d, d)
        cross_cov = run_fused_cross_cov(
            data.X, data.F, data.centroids, mean_F, data.labels, K, d, fd
        )  # (K, d, fd)
        A_T = solve_regularized_covariance_inplace(auto_cov, cross_cov, d, fd)
        del auto_cov
        A = A_T.transpose(0, 2, 1)  # (K, fd, d)
        b = mean_F - xp.einsum("kfd,kd->kf", A, data.centroids)  # (K, fd)
        return {"A_fit": A, "b_fit": b}

    def evaluate(self, params, X, cluster_ids):
        A = params["A_fit"][cluster_ids]
        b = params["b_fit"][cluster_ids]
        if cluster_ids.ndim == 1:
            return b + xp.einsum("nfd,nd->nf", A, X)
        return xp.einsum("nkfd,nd->nkf", A, X) + b


_TAYLOR_MODES = (
    SurrogateMode.TAYLOR_LSQ_HESS,
    SurrogateMode.TAYLOR_COV_HESS,
    SurrogateMode.TAYLOR_COV_NO_HESS,
    SurrogateMode.TAYLOR_EXACT_HESS,
)
_AFFINE_MODES = (
    SurrogateMode.AFFINE_COV_HESS,
    SurrogateMode.AFFINE_COV_NO_HESS,
    SurrogateMode.AFFINE_EXACT_HESS,
    SurrogateMode.AFFINE_COV_HESS_EXACT_GRAD,
)


def local_model_for_mode(mode: SurrogateMode, second_order: bool = False) -> LocalModel:
    """The local model a ``SurrogateMode`` uses (``second_order`` only affects Taylor)."""
    if mode == SurrogateMode.NEAREST_NEIGHBOR:
        return ConstantModel()
    if mode in _TAYLOR_MODES:
        return TaylorModel(second_order=second_order)
    if mode in _AFFINE_MODES:
        return AffineLSQModel()
    raise ValueError(f"No local model for mode {mode}")


def make_local_model(config) -> LocalModel:
    """The local model ``config`` uses (``config.components().local_model``)."""
    return config.components().local_model


# Columns every accepted cluster stores, whatever its model.
SHARED_COLUMNS = ("centroids", "f", "Q", "E_cluster_abs", "E_cluster_rel", "counts")


def output_columns(model: LocalModel, d: int, fd: int) -> dict[str, ParamSpec]:
    """Every column stored per accepted cluster: the shared ones plus the model's."""
    own = model.columns(d, fd)
    clash = sorted(set(own) & set(SHARED_COLUMNS))
    if clash:
        raise ValueError(
            f"Local model {model.name!r} declares reserved column name(s) {clash}; "
            f"{list(SHARED_COLUMNS)} are stored for every model."
        )
    return {
        "centroids": ParamSpec((d,)),
        "f": ParamSpec((fd,)),
        "Q": ParamSpec((d, d)),
        # The acceptance criterion's errors per output component (see src.acceptance).
        "E_cluster_abs": ParamSpec((fd,)),
        "E_cluster_rel": ParamSpec((fd,)),
        "counts": ParamSpec((), np.int64),
        **own,
    }


@dataclass
class DeployedLeaves:
    """Leaf rows for a ``ColumnBuffer`` (shared columns plus the model's) and which clusters they are."""

    idx: Array  # (L,) positions in the batch of the clusters kept as leaves
    rows: dict[str, Array]  # every column of output_columns, one row per kept cluster
    failed: Array  # positions that failed the re-test of the finalized model


def deploy_leaves(
    model: LocalModel,
    trust_region,
    acceptance,
    data: ClusterFitData,
    idx: Array,
    params: dict[str, Array],
    E_abs: Array,
    E_rel: Array,
    retest: bool = True,
    store_Q: bool = True,
) -> DeployedLeaves:
    """
    Turn clusters ``idx`` of a fitted batch into deployed leaves: their ``Q``, the model's
    accepted-only columns (``finalize``) and, when finalize adds columns and ``retest``,
    a re-test of the model that will actually be deployed; clusters it fails are
    returned in ``failed`` (the caller splits them) instead of becoming leaves. Without
    ``retest`` (region-mode fallback leaves) every cluster is kept with its fit errors.
    Without ``store_Q`` (routed-only region models) no ``"Q"`` is computed or returned.
    Both algorithm plans call this, in AFINNI's order: preliminary acceptance first,
    then this.
    """
    idx_mask = xp.zeros(len(data.sizes), dtype=bool)
    idx_mask[idx] = True
    point_mask = idx_mask[data.labels]
    X = data.X[point_mask]
    G = data.G[point_mask] if data.G is not None else None
    F = data.F[point_mask]
    counts = data.sizes[idx]
    labels = repeats(xp.arange(int(idx.shape[0])), counts)
    starts = xp.r_[0, xp.cumsum(counts)[:-1]]
    own = {k: v[idx] for k, v in params.items()}
    centroids, f_c = data.centroids[idx], data.f_c[idx]

    # Shape matrix Q of each leaf, used by the trust-region test at inference.
    Qs = trust_region.shapes(X, centroids, labels, counts) if store_Q else None
    # Parameters the model only computes for leaves (e.g. the second-order Taylor term).
    final = model.finalize(
        ClusterFitData(
            X=X, F=F, G=G, labels=labels, starts=starts, sizes=counts, centroids=centroids,
            f_c=f_c, g_c=data.g_c[idx] if data.g_c is not None else None,
        ),
        own,
    )  # fmt: skip
    E_abs, E_rel = E_abs[idx], E_rel[idx]
    failed = idx[:0]
    if final and retest:
        # Accepted-only columns change predictions, so re-test the deployed model.
        F_final = model.evaluate({**own, **final, "centroids": centroids, "f": f_c}, X, labels)
        still, E_abs, E_rel = acceptance(F, F_final, starts, counts)
        if not bool(xp.all(still)):
            keep = xp.nonzero(still)[0]
            failed = idx[~still]
            idx, counts, E_abs, E_rel = idx[keep], counts[keep], E_abs[keep], E_rel[keep]
            Qs = Qs[keep] if Qs is not None else None
            centroids, f_c = centroids[keep], f_c[keep]
            own = {k: v[keep] for k, v in own.items()}
            final = {k: v[keep] for k, v in final.items()}
    rows = {
        "centroids": centroids,
        "f": f_c,
        **({"Q": Qs} if store_Q else {}),
        "E_cluster_abs": E_abs,
        "E_cluster_rel": E_rel,
        "counts": counts.astype(xp.int64),
        **own,
        **final,
    }
    return DeployedLeaves(idx, rows, failed)
