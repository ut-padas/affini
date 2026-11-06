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
If a scheme needs more information about each cluster (for example a bounding
convex polytope), add it as an optional field of ``ClusterPoints`` and pass it
through ``augment_clusters``.

All samplers work on a whole batch of clusters at once, stored contiguously, so
they can run as single vectorized passes on the GPU.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src import Array, xp
from src.evaluation import Evaluator
from src.registry import ComponentSpec, Registry
from src.utils import repeats


@dataclass
class ClusterPoints:
    """
    A batch of clusters to sample in, stored contiguously on the device.

    Cluster ``k`` owns rows ``starts[k] : starts[k] + counts[k]`` of ``X``.
    """

    X: Array  # (N, d) existing points
    counts: Array  # (K,) points per cluster
    centroids: Array | None = None  # (K, d), if the sampler needs them
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
        return clusters.X[idx1] * alpha + clusters.X[idx2] * (1.0 - alpha)


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
) -> AugmentedClusters:
    """
    Grow each cluster from ``counts`` to ``target_counts`` points.

    New points are placed by ``sampler`` and evaluated with ``evaluator`` (with
    gradients if ``G`` is given). ``centroids`` (K, d), if given, is passed on to
    the sampler in ``ClusterPoints``. In the output, every cluster's block holds its
    original points followed by its new ones.
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

    clusters = ClusterPoints(X, counts, centroids=centroids)
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
