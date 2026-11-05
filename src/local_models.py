"""
The function fitted on each cluster (leaf) of the AFINNI forest.

A ``LocalModel`` owns three things that used to be spread over the code:

- ``fit``: fit the model on every cluster of a batch at once;
- ``evaluate``: evaluate those models at given points, used both for the
  acceptance test during training and for inference;
- ``columns``: the per-cluster parameters stored for accepted clusters.

Adding a model (e.g. a local quadratic or kernel ridge regression) means writing
one subclass; the refinement loop, output buffers and MPI gathering only see the
columns it declares.

Performance contract: ``fit`` and ``evaluate`` are fully vectorized over all
clusters / points in one call, with no Python loops over points or clusters.
``evaluate`` accepts ``cluster_ids`` of shape (N,) (one cluster per point, in
training) or (n, k) (k candidate clusters per query, at inference); in the (n, k)
case ``X`` stays (n, d) and is broadcast, never copied k times.

Besides a model's own parameters, ``evaluate`` may read the columns every
cluster stores: ``"centroids"`` (K, d) and ``"f"`` (K, fd), the function value at
the centroid.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src import Array, xp
from src.adaptive_clustering_utils import (
    ClusteringConfig,
    SurrogateMode,
    _reduce_sum_1d,
    run_fused_cross_cov,
    run_fused_stats,
    solve_regularized_covariance_inplace,
)
from src.clusters import ParamSpec


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
    """The function on one leaf, fitted and evaluated for many clusters at once."""

    name: str
    needs_centroid_grad: bool = False  # the loop must evaluate grad f at centroids

    @abstractmethod
    def columns(self, d: int, fd: int) -> dict[str, ParamSpec]:
        """Per-cluster parameters stored for every accepted cluster."""

    @abstractmethod
    def fit(self, data: ClusterFitData) -> dict[str, Array]:
        """Fit every cluster; returns parameters with a leading (K,) axis."""

    @abstractmethod
    def evaluate(self, params: dict[str, Array], X: Array, cluster_ids: Array) -> Array:
        """
        Evaluate cluster ``cluster_ids[i]``'s model at ``X[i]``.

        ``cluster_ids`` (N,) with ``X`` (N, d) returns (N, fd); ``cluster_ids``
        (n, k) with ``X`` (n, d) returns (n, k, fd). ``params`` holds this model's
        parameters plus the shared ``"centroids"`` and ``"f"`` columns.
        """


class ConstantModel(LocalModel):
    """f_hat(x) = f(c): nearest-neighbor interpolation (``NEAREST_NEIGHBOR``)."""

    name = "constant"

    def columns(self, d, fd):
        return {}

    def fit(self, data):
        return {}

    def evaluate(self, params, X, cluster_ids):
        return params["f"][cluster_ids]


class TaylorModel(LocalModel):
    """
    f_hat(x) = f(c) + g(c) (x - c)  [+ 0.5 (x - c)^T H (x - c) if ``second_order``].

    Uses the gradient at the centroid (``TAYLOR_*`` modes). The second-order term
    uses a Hessian ``"H"`` that is only estimated for accepted clusters, so it is
    not part of the acceptance test; it is added to the stored columns and used at
    inference.
    """

    name = "taylor"
    needs_centroid_grad = True

    def __init__(self, second_order: bool = False):
        self.second_order = second_order

    def columns(self, d, fd):
        cols = {"g": ParamSpec((fd, d))}
        if self.second_order:
            cols["H"] = ParamSpec((d, d))
        return cols

    def fit(self, data):
        assert data.g_c is not None, "TaylorModel needs the gradient at the centroids"
        return {"g": data.g_c}

    def evaluate(self, params, X, cluster_ids):
        f = params["f"][cluster_ids]
        g = params["g"][cluster_ids]
        c = params["centroids"][cluster_ids]
        if cluster_ids.ndim == 1:
            dx = X - c
            out = f + xp.einsum("nmd,nd->nm", g, dx)
            if "H" in params:
                H = params["H"][cluster_ids][:, None]  # (N, 1, d, d)
                out = out + 0.5 * xp.einsum("nc,nfcd,nd->nf", dx, H, dx)
            return out
        dx = X[:, None, :] - c
        out = f + xp.einsum("nkfd,nkd->nkf", g, dx)
        if "H" in params:
            n, k = cluster_ids.shape
            H = params["H"][cluster_ids].reshape(n, k, -1, X.shape[1], X.shape[1])
            out += 0.5 * xp.einsum("nkc,nkfcd,nkd->nkf", dx, H, dx)
        return out


class AffineLSQModel(LocalModel):
    """
    f_hat(x) = A x + b, a least-squares fit to the cluster's points with adaptive
    Tikhonov regularization (``AFFINE_*`` modes; Equation (3) in the paper, with
    the intercept in absolute rather than centered coordinates).
    """

    name = "affine_lsq"

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


def make_local_model(config: ClusteringConfig) -> LocalModel:
    """The local model used by ``config.mode``."""
    if config.mode == SurrogateMode.NEAREST_NEIGHBOR:
        return ConstantModel()
    if config.mode in _TAYLOR_MODES:
        return TaylorModel(second_order=config.store_hess)
    if config.mode in _AFFINE_MODES:
        return AffineLSQModel()
    raise ValueError(f"No local model for mode {config.mode}")


def output_columns(model: LocalModel, d: int, fd: int) -> dict[str, ParamSpec]:
    """Every column stored per accepted cluster: the shared ones plus the model's."""
    return {
        "centroids": ParamSpec((d,)),
        "f": ParamSpec((fd,)),
        "Q": ParamSpec((d, d)),
        "E_cluster": ParamSpec(()),
        "radii": ParamSpec(()),
        "counts": ParamSpec((), np.int64),
        **model.columns(d, fd),
    }
