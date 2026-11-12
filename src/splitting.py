"""
How rejected clusters are split in two (Section 2.1, Algorithms 2 and 3).

Every strategy is Hessian-informed two-means with a different estimate of the
metric ``M = |H|^{1/2}`` (the identity gives standard two-means / PCA): points
are mapped by ``M``, the leading eigenvector ``v`` of the mapped covariance
``M C M`` is the split direction, and each cluster's points are ordered by their
projection onto it. ``child_sizes`` then decides where each ordered cluster is cut:
at the median by default (``cut="median"``, equal halves), or at the centroid
(``cut="sign"``, the sign of the projection as in Algorithm 2, falling back to the
median when one side would be empty).

A ``SurrogateMode`` is a pair (local model, split strategy); see
``split_strategy_for_mode`` and ``src.local_models.local_model_for_mode``.

Adding a strategy means subclassing ``SplitStrategy``, implementing ``metric``
(or overriding ``split`` entirely), and registering it in ``SPLIT_STRATEGIES``.
A different cut rule overrides ``child_sizes``, which sees the sorted projection
scores. Splits are binary: ``split`` returns a ``Split`` whose permutation keeps
every cluster contiguous, with exactly two child sizes per cluster that add up to
the cluster's size (one of them may be 0; empty children are dropped), and the
hyperplane ``w^T x + b = 0`` that separates the two children (recorded in the
binary trees, ``src.trees``).
Strategies declare the data they need with ``needs_point_grad``,
``needs_centroid_grad`` and ``needs_hessian_fn``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from src import IS_CUPY_BACKEND, Array, xp
from src.modes import SurrogateMode
from src.numerics import (
    TINY_EPS,
    _reduce_sum_1d,
    centroid_distances,
    get_n_smallest_per_block,
    lsq_hessian_in_eigenbasis,
    run_fused_cross_cov,
    run_fused_split_projection,
    run_fused_stats,
    solve_regularized_covariance_inplace,
    sort_by_block_then_value,
    taylor_residual_norms,
)
from src.registry import Registry

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


@dataclass(frozen=True)
class Split:
    """
    How a batch of K clusters is split. ``order`` (N,) permutes the rows so every
    cluster stays contiguous (in label order) with its side-0 child first;
    ``child_sizes`` (2K,) are ``[k0_side0, k0_side1, k1_side0, ...]``. Side 0 holds
    the points with ``W[k] @ x + b[k] < 0``, side 1 those with ``> 0``.
    """

    order: Array
    child_sizes: Array
    W: Array  # (K, d) hyperplane normals
    b: Array  # (K,) hyperplane offsets


def median_split_sizes(counts: Array) -> Array:
    """
    Child sizes for splitting every cluster at its median along the split direction.

    Given points ordered by ``SplitStrategy.split``, a parent with ``n`` points
    becomes a low-side child of ``n // 2`` points followed by a high-side child of
    ``n - n // 2``. Returns (2K,) int64 sizes ``[k0_low, k0_high, k1_low, ...]``.

    Note: Algorithm 2 in the paper splits by the sign of the projection instead,
    which can give children of unequal size (``sign_split_sizes``).
    """
    half = counts // 2
    sizes = xp.zeros(2 * len(counts), dtype=xp.int64)
    sizes[0::2] = half
    sizes[1::2] = counts - half
    return sizes


def sign_split_sizes(counts: Array, sorted_scores: Array) -> Array:
    """
    Child sizes for splitting every cluster by the sign of the projection score
    (Algorithm 2): the low child holds the points with a negative score. A cluster
    whose points would all fall on one side is split at the median instead, so
    every split makes progress. ``sorted_scores`` (N,) are the scores in the order
    returned by ``SplitStrategy.split`` (ascending within each cluster).
    """
    starts = xp.r_[0, xp.cumsum(counts)[:-1]]
    n_low = _reduce_sum_1d((sorted_scores < 0).astype(xp.int64), starts)
    n_low = xp.where((n_low == 0) | (n_low == counts), counts // 2, n_low)
    sizes = xp.zeros(2 * len(counts), dtype=xp.int64)
    sizes[0::2] = n_low
    sizes[1::2] = counts - n_low
    return sizes


CUT_RULES = ("median", "sign")


class SplitStrategy(ABC):
    """Splits every rejected cluster of a batch into two children."""

    name: str
    needs_point_grad: bool = False  # needs SplitData.G
    needs_centroid_grad: bool = False  # needs SplitData.g_c
    needs_hessian_fn: bool = False  # needs SplitData.hessian_fn
    cut: Literal["median", "sign"] = "median"

    def __init__(self, cut: Literal["median", "sign"] = "median"):
        if cut not in CUT_RULES:
            raise ValueError(f"Unknown cut rule {cut!r}; choose one of {CUT_RULES}.")
        self.cut = cut

    def settings(self) -> dict:
        """Constructor arguments, so ``SPLIT_STRATEGIES.create((name, settings))`` rebuilds it."""
        return {"cut": self.cut}

    @abstractmethod
    def metric(self, data: SplitData) -> tuple[Array, Array]:
        """
        Return ``(M, CovZ)``, both (K, d, d): the per-cluster metric ``M`` (an
        estimate of ``|H|^{1/2}``, or the identity) and the covariance of the
        mapped points, ``M C M``, whose leading eigenvector is the split direction.
        """

    def split(self, data: SplitData) -> Split:
        """
        Sort every cluster's points by their score ``v^T M (x - mu)`` and cut them
        with ``child_sizes``. The score is affine in ``x`` (``w = M^T v``), so the
        cut is the hyperplane ``w^T x + b = 0`` with ``b = -w^T mu - t``, where the
        threshold ``t`` is the midpoint between the last side-0 score and the first
        side-1 score. A child with no points gets a plane through the extreme point.
        """
        M, CovZ = self.metric(data)
        _, V_z = xp.linalg.eigh(CovZ)
        v = V_z[:, :, -1]  # highest-variance direction, (K, d)
        scores = run_fused_split_projection(data.X, data.centroids, data.labels, M, v)
        W = xp.einsum("kji,kj->ki", M, v)  # M^T v: scores are W[k] @ (x - mu_k)
        if IS_CUPY_BACKEND:
            del CovZ, V_z, v, M
            cp.get_default_memory_pool().free_all_blocks()
        order = sort_by_block_then_value(scores, data.labels)  # (N,)
        sorted_scores = scores[order]
        child_sizes = self.child_sizes(
            data.counts, sorted_scores if self.cut != "median" else None
        )

        starts = xp.r_[0, xp.cumsum(data.counts)[:-1]]
        ends = starts + data.counts - 1
        last_n = max(int(sorted_scores.shape[0]) - 1, 0)
        cut = starts + child_sizes[0::2]  # first side-1 row of each cluster
        below = xp.clip(xp.maximum(cut - 1, starts), 0, last_n)
        above = xp.clip(xp.minimum(cut, ends), 0, last_n)
        t = 0.5 * (sorted_scores[below] + sorted_scores[above])
        b = -xp.sum(W * data.centroids, axis=1) - t
        return Split(order=order, child_sizes=child_sizes, W=W, b=b)

    def child_sizes(self, counts: Array, sorted_scores: Array | None = None) -> Array:
        """
        Where each ordered cluster is cut: (2K,) sizes ``[k0_low, k0_high, ...]``.
        ``sorted_scores`` are the projection scores in split order (None for the
        median cut, which does not need them).
        """
        if self.cut == "sign":
            return sign_split_sizes(counts, sorted_scores)
        return median_split_sizes(counts)


SPLIT_STRATEGIES: Registry[SplitStrategy] = Registry("split strategy", SplitStrategy)


def _covariance(data: SplitData) -> Array:
    K, d = len(data.counts), data.X.shape[1]
    C, _ = run_fused_stats(
        data.X, data.centroids, data.labels, K, d, 0, None, calc_E=False
    )
    return C  # (K, d, d)


@SPLIT_STRATEGIES.register
class CovarianceSplit(SplitStrategy):
    """Standard two-means via the principal direction of the covariance (M = I)."""

    name = "covariance"

    def metric(self, data):
        K, d = len(data.counts), data.X.shape[1]
        C = _covariance(data)
        CovZ = 0.5 * (C + C.transpose(0, 2, 1))
        M = xp.eye(d, dtype=xp.float64)[None, :, :].repeat(K, axis=0)
        return M, CovZ


@SPLIT_STRATEGIES.register
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
        self,
        gradient: Literal["centroid", "local_lsq"] = "local_lsq",
        min_pts_per_cluster: int = 100,
        cut: Literal["median", "sign"] = "median",
    ):
        super().__init__(cut)
        if gradient not in ("centroid", "local_lsq"):
            raise ValueError(f"gradient must be 'centroid' or 'local_lsq'; got {gradient!r}.")
        self.gradient = gradient
        self.min_pts_per_cluster = int(min_pts_per_cluster)
        self.needs_centroid_grad = gradient == "centroid"

    def settings(self):
        return {
            "gradient": self.gradient,
            "min_pts_per_cluster": self.min_pts_per_cluster,
            "cut": self.cut,
        }

    def metric(self, data):
        K, d, fd = len(data.counts), data.X.shape[1], data.F.shape[1]

        if self.gradient == "centroid":
            taylor_err = taylor_residual_norms(
                data.X, data.F, data.centroids, data.f_c, data.labels, data.g_c
            )  # (N,)
        else:  # two-pass local fit: gradient from the points nearest the centroid
            dist_to_mu = centroid_distances(data.X, data.centroids, data.labels)  # (N,)
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
            taylor_err = taylor_residual_norms(
                data.X, data.F, data.centroids, data.f_c, data.labels, A_tangent
            )  # (N,)

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


@SPLIT_STRATEGIES.register
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


@SPLIT_STRATEGIES.register
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
