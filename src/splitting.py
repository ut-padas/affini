"""
How rejected clusters are split in two (Section 2.1, Algorithms 2 and 3).

Every strategy is Hessian-informed two-means with a different estimate of the
metric ``M = |H|^{1/2}`` (the identity gives standard two-means / PCA): points
are mapped by ``M``, the leading eigenvector ``v`` of the mapped covariance
``M C M`` is the split direction, and each cluster's points are ordered by their
projection onto it. ``child_sizes`` then decides where each ordered cluster is cut
(at the median by default).

A ``SurrogateMode`` is a pair (local model, split strategy); see
``split_strategy_for_mode`` and ``src.local_models.local_model_for_mode``.
Adding a strategy means subclassing ``SplitStrategy`` and implementing ``metric``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from src import IS_CUPY_BACKEND, Array, xp
from src.adaptive_clustering_utils import (
    TINY_EPS,
    SurrogateMode,
    _reduce_sum_1d,
    get_n_smallest_per_block,
    lsq_hessian_in_eigenbasis,
    run_fused_cross_cov,
    run_fused_split_projection,
    run_fused_stats,
    solve_regularized_covariance_inplace,
)

if IS_CUPY_BACKEND:
    import cupy as cp


@dataclass
class SplitData:
    """A batch of rejected (already augmented) clusters, stored contiguously."""

    X: Array  # (N, d)
    F: Array  # (N, fd)
    G: Array | None  # (N, fd, d) per-point gradients, when carried
    labels: Array  # (N,) cluster index of every point, in contiguous order
    counts: Array  # (K,) points per cluster
    centroids: Array  # (K, d)
    f_c: Array  # (K, fd) f at the centroids
    g_c: Array | None  # (K, fd, d) gradient at the centroids, if evaluated
    hessian_fn: Callable[[Array], Array] | None = None  # exact Hessian, (K, d, d)


def median_split_sizes(counts: Array) -> Array:
    """
    Child sizes for splitting every cluster at its median along the split direction.

    Given points ordered by ``SplitStrategy.split``, a parent with ``n`` points
    becomes a low-side child of ``n // 2`` points followed by a high-side child of
    ``n - n // 2``. Returns (2K,) int64 sizes ``[k0_low, k0_high, k1_low, ...]``.

    Note: Algorithm 2 in the paper splits by the sign of the projection instead,
    which can give children of unequal size.
    """
    half = counts // 2
    sizes = xp.zeros(2 * len(counts), dtype=xp.int64)
    sizes[0::2] = half
    sizes[1::2] = counts - half
    return sizes


class SplitStrategy(ABC):
    """Splits every rejected cluster of a batch into two children."""

    name: str
    needs_point_grad: bool = False  # needs SplitData.G
    needs_centroid_grad: bool = False  # needs SplitData.g_c
    needs_hessian_fn: bool = False  # needs SplitData.hessian_fn

    @abstractmethod
    def metric(self, data: SplitData) -> tuple[Array, Array]:
        """
        Return ``(M, CovZ)``, both (K, d, d): the per-cluster metric ``M`` (an
        estimate of ``|H|^{1/2}``, or the identity) and the covariance of the
        mapped points, ``M C M``, whose leading eigenvector is the split direction.
        """

    def split(self, data: SplitData) -> tuple[Array, Array]:
        """
        Return ``(order, child_sizes)``: ``order`` permutes the rows so that every
        cluster stays contiguous (in label order) with its points sorted along its
        split direction; ``child_sizes`` (2K,) gives the two children of each
        cluster in that order.
        """
        M, CovZ = self.metric(data)
        _, V_z = xp.linalg.eigh(CovZ)
        v = V_z[:, :, -1]  # highest-variance direction, (K, d)
        scores = run_fused_split_projection(data.X, data.centroids, data.labels, M, v)
        if IS_CUPY_BACKEND:
            del CovZ, V_z, v, M
            cp.get_default_memory_pool().free_all_blocks()
        order = xp.lexsort(xp.stack([scores, data.labels]))  # (N,)
        return order, self.child_sizes(data.counts)

    def child_sizes(self, counts: Array) -> Array:
        """Where each ordered cluster is cut; the median by default."""
        return median_split_sizes(counts)


def _covariance(data: SplitData) -> Array:
    K, d = len(data.counts), data.X.shape[1]
    C, _ = run_fused_stats(
        data.X, data.centroids, data.labels, K, d, 0, None, calc_E=False
    )
    return C  # (K, d, d)


class CovarianceSplit(SplitStrategy):
    """Standard two-means via the principal direction of the covariance (M = I)."""

    name = "covariance"

    def metric(self, data):
        K, d = len(data.counts), data.X.shape[1]
        C = _covariance(data)
        CovZ = 0.5 * (C + C.transpose(0, 2, 1))
        M = xp.eye(d, dtype=xp.float64)[None, :, :].repeat(K, axis=0)
        return M, CovZ


class CovarianceHessianSplit(SplitStrategy):
    """
    Hessian from the covariance of low-Taylor-error points (Algorithm 3).

    The Taylor expansion at each centroid uses either the gradient at the centroid
    (``gradient="centroid"``) or a least-squares gradient fitted on the points
    nearest the centroid (``gradient="local_lsq"``). The ``min_pts_per_cluster // 3``
    points with the smallest Taylor error estimate the level set, and the inverse
    of their covariance is used as |H| (Theorem 2.2).
    """

    name = "covariance_hessian"

    def __init__(
        self, gradient: Literal["centroid", "local_lsq"], min_pts_per_cluster: int
    ):
        self.gradient = gradient
        self.min_pts_per_cluster = min_pts_per_cluster
        self.needs_centroid_grad = gradient == "centroid"

    def metric(self, data):
        K, d, fd = len(data.counts), data.X.shape[1], data.F.shape[1]
        dx = data.X - data.centroids[data.labels]

        if self.gradient == "centroid":
            g_mus_exp = data.g_c[data.labels]  # (N, fd, d)
            lin = xp.einsum("nmd,nd->nm", g_mus_exp, dx)  # (N, fd)
            diff = data.F - (data.f_c[data.labels] + lin)  # (N, fd)
        else:  # two-pass local fit: gradient from the points nearest the centroid
            dist_to_mu = xp.linalg.norm(dx, axis=1)  # (N,)
            n_tangent = max(d * 2, self.min_pts_per_cluster // 4)
            tangent_mask = get_n_smallest_per_block(dist_to_mu, data.labels, n_tangent)
            X_core = data.X[tangent_mask]
            F_core = data.F[tangent_mask]
            labels_core = data.labels[tangent_mask]
            M_XX, _ = run_fused_stats(
                X_core, data.centroids, labels_core, K, d, 0, None, calc_E=False
            )  # (K, d, d)
            M_XF = run_fused_cross_cov(
                X_core, F_core, data.centroids, data.f_c, labels_core, K, d, fd
            )  # (K, d, fd)
            A_T_tangent = solve_regularized_covariance_inplace(M_XX, M_XF, d, fd)
            A_tangent = A_T_tangent.transpose(0, 2, 1)  # (K, fd, d)
            lin = xp.einsum("nmd,nd->nm", A_tangent[data.labels], dx)  # (N, fd)
            diff = data.F - (data.f_c[data.labels] + lin)  # (N, fd)
        taylor_err = xp.linalg.norm(diff, axis=1)  # (N,)

        # Points where the Taylor expansion is (nearly) exact trace out a level set.
        n_est_cov = max(1, self.min_pts_per_cluster // 3)
        level_set = get_n_smallest_per_block(taylor_err, data.labels, n_est_cov)
        H_inv, _ = run_fused_stats(
            data.X[level_set],
            data.centroids,
            data.labels[level_set],
            K,
            d,
            0,
            None,
            calc_E=False,
        )  # (K, d, d)
        H_inv = 0.5 * (H_inv + H_inv.transpose(0, 2, 1))

        dvals, U = xp.linalg.eigh(H_inv)
        dvals_sqrt_inv = 1.0 / xp.sqrt(xp.maximum(xp.abs(dvals), TINY_EPS))
        M = (U * dvals_sqrt_inv[:, None]) @ U.transpose(0, 2, 1)  # |H|^{1/2}
        M = 0.5 * (M + M.transpose(0, 2, 1))

        C = _covariance(data)
        return M, M @ C @ M.transpose(0, 2, 1)


class ExactHessianSplit(SplitStrategy):
    """|H|^{1/2} from the exact Hessian at each centroid (``hessian_fn``)."""

    name = "exact_hessian"
    needs_hessian_fn = True

    def metric(self, data):
        if data.hessian_fn is None:
            raise RuntimeError(f"{self.name} split requires a Hessian function.")
        H = data.hessian_fn(data.centroids)  # (K, d, d)
        H_sym = 0.5 * (H + H.transpose(0, 2, 1))
        dvals, U = xp.linalg.eigh(H_sym)
        M = (U * xp.sqrt(xp.abs(dvals))[:, None]) @ U.transpose(0, 2, 1)
        C = _covariance(data)
        return M, M @ C @ M.transpose(0, 2, 1)


class GradientLSQHessianSplit(SplitStrategy):
    """
    Hessian from a least-squares fit to the per-point gradient residuals, using
    the output component with the largest relative Taylor error.
    """

    name = "gradient_lsq_hessian"
    needs_point_grad = True
    needs_centroid_grad = True

    def metric(self, data):
        if data.G is None:
            raise RuntimeError(f"{self.name} split requires per-point gradients.")
        K, d, fd = len(data.counts), data.X.shape[1], data.F.shape[1]
        starts = xp.r_[0, xp.cumsum(data.counts)[:-1]]
        dx = data.X - data.centroids[data.labels]

        g_mus_exp = data.g_c[data.labels]  # (N, fd, d)
        f_mus_exp = data.f_c[data.labels]  # (N, fd)
        lin = xp.einsum("nmd,nd->nm", g_mus_exp, dx)
        diff = data.F - (f_mus_exp + lin)  # (N, fd)
        sq_err = _reduce_sum_1d(diff**2, starts)  # (K,)
        sq_val = _reduce_sum_1d(data.F**2, starts)  # (K,)
        scores = xp.sqrt(sq_err) / (xp.sqrt(sq_val) + TINY_EPS)
        active_comps = xp.argmax(scores, axis=1).astype(xp.int32)  # (K,)

        # Covariance and active-component gradient interaction
        C, E = run_fused_stats(
            data.X,
            data.centroids,
            data.labels,
            K,
            d,
            fd,
            active_comps,
            G=data.G,
            g_mus=data.g_c,
            calc_E=True,
        )
        C = 0.5 * (C + C.transpose(0, 2, 1))
        S = E + E.transpose(0, 2, 1)

        U, F_mat = lsq_hessian_in_eigenbasis(C, S)  # H = U F_mat U^T
        evals, V = xp.linalg.eigh(F_mat)
        Z = U @ V  # (K, d, d)
        M = (Z * xp.sqrt(xp.abs(evals))[:, None, :]) @ Z.transpose(0, 2, 1)
        M = 0.5 * (M + M.transpose(0, 2, 1))  # |H|^{1/2}
        return M, M @ C @ M.transpose(0, 2, 1)


def split_strategy_for_mode(
    mode: SurrogateMode, min_pts_per_cluster: int
) -> SplitStrategy:
    """The split strategy a ``SurrogateMode`` uses."""
    if mode in (
        SurrogateMode.NEAREST_NEIGHBOR,
        SurrogateMode.AFFINE_COV_NO_HESS,
        SurrogateMode.TAYLOR_COV_NO_HESS,
    ):
        return CovarianceSplit()
    if mode == SurrogateMode.AFFINE_COV_HESS:
        return CovarianceHessianSplit("local_lsq", min_pts_per_cluster)
    if mode in (
        SurrogateMode.TAYLOR_COV_HESS,
        SurrogateMode.AFFINE_COV_HESS_EXACT_GRAD,
    ):
        return CovarianceHessianSplit("centroid", min_pts_per_cluster)
    if mode in (SurrogateMode.AFFINE_EXACT_HESS, SurrogateMode.TAYLOR_EXACT_HESS):
        return ExactHessianSplit()
    if mode == SurrogateMode.TAYLOR_LSQ_HESS:
        return GradientLSQHessianSplit()
    raise ValueError(f"No split strategy for mode {mode}")
