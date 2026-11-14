"""
Sampling new points inside clusters ("cluster augmentation").

A rejected cluster needs enough points to estimate its surrogate, Hessian and
split, so small clusters are topped up with new points before splitting. *Where*
the new points go is decided by a ``Sampler``; *how many* each cluster needs, and
how the new points are merged with the originals and evaluated, is shared by all
samplers (``augmentation_targets`` and ``augment_clusters``).

Adding a sampling scheme:
    1. Subclass ``Sampler`` and implement ``sample``.
    2. Register it in ``SAMPLERS`` (``@SAMPLERS.register``) so
       ``ClusteringConfig.sampling_method`` can name it; settings can be given as
       ``(name, {...})`` or by passing an instance as ``ClusteringConfig.sampler``.
Samplers receive each cluster's points, counts and centroids (``ClusterPoints``).
A sampler that sets ``needs_regions`` also receives each cluster's convex region
(``ClusterRegions``: the domain box cut by the hyperplanes of the cluster's
ancestors), which needs ``ClusteringConfig.track_trees`` and a single initial
cluster. Other per-cluster information goes in a new optional field of
``ClusterPoints``, passed through ``augment_clusters``.

All samplers work on a whole batch of clusters at once, stored contiguously, so
they can run as single vectorized passes on the GPU.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src import IS_CUPY_BACKEND, Array, xp
from src.evaluation import Evaluator
from src.registry import ComponentSpec, Registry
from src.utils import repeats

# Slack of ClusterRegions.contains, relative to the domain diagonal. Cuts sit halfway
# between neighbouring points, so points normally lie strictly inside every face;
# only ties at a cut put points on a plane, where re-expressing it from the cluster's
# centroid rounds at the scale of the ancestor that made the cut (up to the domain),
# not of the cluster.
REGION_RTOL = 1e-12


@dataclass
class ClusterRegions:
    """
    The convex region of each cluster of a batch: the domain box cut by one
    half-space per ancestor (``src.trees.BinaryTreeForest.ancestor_planes``).

    ``x`` lies in cluster ``k``'s region when ``lower <= x <= upper`` and, for every
    row ``j``, ``planes[k, j, :d] @ (x - origins[k]) + planes[k, j, d] <= 0``. The
    normals have unit length, so the left side is a signed distance; rows are
    ordered nearest ancestor first, and rows from ``depth[k]`` on are padding that
    always holds. The regions of a level's clusters share at most faces, and with
    the regions of the leaves accepted so far they cover the domain.
    """

    origins: Array  # (K, d) the clusters' centroids, which the planes are measured from
    planes: Array  # (K, D, d+1) float64: unit normal, then offset
    depth: Array  # (K,) int64 real rows of each cluster
    domain: Array  # (2, d) float64: the domain box's lower and upper corners

    def contains(self, X: Array, labels: Array, rtol: float = REGION_RTOL) -> Array:
        """
        Whether each point ``X[i]`` lies in the region of cluster ``labels[i]``, up to
        ``rtol`` times the domain diagonal (see ``REGION_RTOL``). Builds an (N, D)
        array, so it is meant for checks and tests rather than inner loops.
        """
        lower, upper = self.domain
        eps = rtol * float(xp.linalg.norm(upper - lower))
        in_box = xp.all((lower - eps <= X) & (upper + eps >= X), axis=1)
        if self.planes.shape[1] == 0:
            return in_box
        rows = self.planes[labels]
        dist = xp.einsum("nd,njd->nj", X - self.origins[labels], rows[:, :, :-1]) + rows[:, :, -1]
        return in_box & xp.all(dist <= eps, axis=1)


@dataclass
class ClusterPoints:
    """
    A batch of clusters to sample in, stored contiguously on the device.

    Cluster ``k`` owns rows ``starts[k] : starts[k] + counts[k]`` of ``X``.
    """

    X: Array  # (N, d) existing points
    counts: Array  # (K,) points per cluster
    centroids: Array | None = None  # (K, d), if the sampler needs them
    regions: ClusterRegions | None = None  # given to samplers with needs_regions
    starts: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.starts = xp.r_[0, xp.cumsum(self.counts)[:-1]]

    @property
    def n_clusters(self) -> int:
        return len(self.counts)

    @property
    def d(self) -> int:
        return int(self.X.shape[1])


class Sampler(ABC):
    """Chooses locations for new points inside clusters."""

    name: str
    needs_regions: bool = False  # needs ClusterPoints.regions

    def settings(self) -> dict:
        """Constructor arguments, so ``SAMPLERS.create((name, settings))`` rebuilds it."""
        return {}

    @abstractmethod
    def sample(self, clusters: ClusterPoints, n_new: Array, rng) -> Array:
        """
        Return ``int(sum(n_new))`` new points with shape (sum(n_new), d), grouped by
        cluster: the ``n_new[0]`` points of cluster 0 first, then cluster 1, and so
        on. ``n_new`` (K,) may contain zeros. ``rng`` is an ``xp.random.Generator``
        and is the only source of randomness, so results are reproducible.
        """


SAMPLERS: Registry[Sampler] = Registry("sampling method", Sampler)


@SAMPLERS.register
class ConvexCombinationSampler(Sampler):
    """
    Random convex combinations of two distinct points of the same cluster.

    Each new point is ``a * x_i + (1 - a) * x_j`` with ``i != j`` drawn uniformly
    from the cluster and ``a ~ U(0, 1)``. Cheap and fully vectorized, and new points
    stay inside the cluster's convex hull. The points are not uniform over the
    hull (they concentrate towards its interior) and never fill the gap between
    the hull and the cluster's Voronoi cell. See Section 2.3 / Algorithm 4.
    """

    name = "convex_combination"

    def sample(self, clusters: ClusterPoints, n_new: Array, rng) -> Array:
        n_total = int(xp.sum(n_new))
        if n_total == 0:
            return xp.empty((0, clusters.d), dtype=xp.float64)

        cluster_ids = repeats(xp.arange(clusters.n_clusters), n_new)
        c_starts = clusters.starts[cluster_ids]
        c_counts = clusters.counts[cluster_ids]

        # Two distinct parents per new point: offset the second index by
        # 1..count-1 (mod count). A single-point cluster reuses its only point.
        rand1 = (rng.random(n_total) * c_counts).astype(xp.int64)
        rand2 = (rng.random(n_total) * (c_counts - 1)).astype(xp.int64) + 1
        idx1 = c_starts + rand1
        idx2 = c_starts + (rand1 + rand2) % c_counts

        alpha = rng.random((n_total, 1)).astype(xp.float64)
        if IS_CUPY_BACKEND:
            return _convex_combination_gpu(clusters.X, idx1, idx2, alpha)
        return clusters.X[idx1] * alpha + clusters.X[idx2] * (1.0 - alpha)


_CONVEX_KERNEL = None


def _convex_combination_gpu(X, idx1, idx2, alpha):
    """
    ``X[idx1] * alpha + X[idx2] * (1 - alpha)`` in one pass, without the four (n, d)
    temporaries. The ``_rn`` intrinsics forbid fused multiply-add, so every element
    rounds exactly as in the separate CuPy operations (bit-identical output).
    """
    global _CONVEX_KERNEL
    import cupy as cp

    if _CONVEX_KERNEL is None:
        _CONVEX_KERNEL = cp.ElementwiseKernel(
            "raw float64 X, raw int64 i1, raw int64 i2, raw float64 a, int64 d",
            "float64 out",
            """
            const long long r = i / d, c = i % d;
            const double w = a[r];
            out = __dadd_rn(__dmul_rn(X[i1[r] * d + c], w),
                            __dmul_rn(X[i2[r] * d + c], __dsub_rn(1.0, w)));
            """,
            "afinni_convex_combination",
        )
    n, d = int(idx1.shape[0]), int(X.shape[1])
    out = cp.empty((n, d), dtype=cp.float64)
    _CONVEX_KERNEL(cp.ascontiguousarray(X), idx1, idx2, cp.ascontiguousarray(alpha.ravel()), d, out)
    return out


def make_sampler(spec: ComponentSpec) -> Sampler:
    """Return the sampler for ``spec``: a registered name, ``(name, settings)`` or an instance."""
    return SAMPLERS.create(spec)


def augmentation_targets(counts: Array, min_pts_per_cluster: int) -> Array:
    """
    Number of points each cluster should have after augmentation.

    A cluster with fewer than ``2 * min_pts_per_cluster`` points is grown to
    ``max(2 * count, 2 * min_pts_per_cluster)``, so each of its two children can
    still have ``min_pts_per_cluster`` points; larger clusters are left as they are.
    """
    return xp.where(
        counts < 2 * min_pts_per_cluster,
        xp.maximum(2 * counts, 2 * min_pts_per_cluster),
        counts,
    )


@dataclass
class AugmentedClusters:
    """Clusters after augmentation: each cluster's original points, then its new ones."""

    X: Array  # (sum(target_counts), d)
    F: Array  # (sum(target_counts), fd)
    G: Array | None  # (sum(target_counts), fd, d), when gradients are carried per point
    labels: Array  # (sum(target_counts),) cluster index of every row
    X_new: (
        Array | None
    )  # (n_new_total, d) the sampled points, or None if none were needed


def augment_clusters(
    X: Array,
    F: Array,
    G: Array | None,
    counts: Array,
    target_counts: Array,
    sampler: Sampler,
    evaluator: Evaluator,
    rng,
    centroids: Array | None = None,
    regions: ClusterRegions | None = None,
) -> AugmentedClusters:
    """
    Grow each cluster from ``counts`` to ``target_counts`` points.

    New points are placed by ``sampler`` and evaluated with ``evaluator`` (with
    gradients if ``G`` is given). ``centroids`` (K, d) and ``regions``, if given, are
    passed on to the sampler in ``ClusterPoints``; a sampler with ``needs_regions``
    requires ``regions`` whenever it is called (when some cluster grows). In the
    output, every cluster's block holds its original points followed by its new ones.
    """
    n_clusters = len(counts)
    total = int(xp.sum(target_counts))
    d, fd = X.shape[1], F.shape[1]

    X_aug = xp.empty((total, d), dtype=xp.float64)
    F_aug = xp.empty((total, fd), dtype=xp.float64)
    G_aug = xp.empty((total, fd, d), dtype=xp.float64) if G is not None else None

    # Rows of each output block: originals first, then the new points.
    target_starts = xp.r_[0, xp.cumsum(target_counts)[:-1]]
    labels = repeats(xp.arange(n_clusters), target_counts)
    idx_in_cluster = xp.arange(total, dtype=xp.int32) - target_starts[labels]
    is_orig = idx_in_cluster < counts[labels]

    X_aug[is_orig] = X
    F_aug[is_orig] = F
    if G is not None:
        G_aug[is_orig] = G

    is_new = ~is_orig
    if int(xp.sum(is_new)) == 0:
        return AugmentedClusters(X_aug, F_aug, G_aug, labels, None)

    if sampler.needs_regions and regions is None:
        raise ValueError(f"Sampler {sampler.name!r} needs the clusters' regions; pass regions.")
    clusters = ClusterPoints(X, counts, centroids=centroids, regions=regions)
    X_new = sampler.sample(clusters, target_counts - counts, rng)
    n_new = total - int(X.shape[0])
    if tuple(X_new.shape) != (n_new, d):
        raise ValueError(
            f"Sampler {sampler.name!r} returned shape {tuple(X_new.shape)}; "
            f"expected ({n_new}, {d})."
        )
    X_aug[is_new] = X_new
    F_new, G_new = evaluator(X_new, need_grad=G_aug is not None)
    F_aug[is_new] = F_new
    if G_aug is not None:
        G_aug[is_new] = G_new
    return AugmentedClusters(X_aug, F_aug, G_aug, labels, X_new)
