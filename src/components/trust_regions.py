"""
Trust regions of accepted clusters.

Every accepted cluster stores a shape matrix ``Q`` (d, d), the trust-region matrix ``T`` of
the companion. At inference a query ``x`` is inside the trust region of the cluster with
centroid ``c`` when ``(x - c)^T Q (x - c) <= 1``;
``Q`` also ranks the candidate clusters of a query (smallest value first). The
default ``CovarianceEllipsoid`` scales the inverse covariance of the cluster's
points so that the farthest point lies on the boundary: the region contains every
observed point, but it is an empirical coverage region, not an error certificate.

Adding a trust region: subclass ``TrustRegion``, implement ``shapes`` (returning
symmetric positive semi-definite (K, d, d) matrices) and register it in
``TRUST_REGIONS``. Regions built by name receive ``q_reg`` as a keyword argument.
"""

from abc import ABC, abstractmethod

from src import Array, xp
from src.components.registry import Registry
from src.numerics import (
    TINY_EPS,
    det_matmul,
    reduce_max_1d,
    run_fused_mahalanobis_projection,
    symmetric_point_covariance,
)


class TrustRegion(ABC):
    """Builds the trust-region shape matrix ``Q`` of every accepted cluster."""

    name: str

    def settings(self) -> dict:
        return {}

    @abstractmethod
    def shapes(self, X: Array, centroids: Array, labels: Array, counts: Array) -> Array:
        """
        Return ``Q`` (K, d, d) for a batch of accepted clusters. ``X`` (N, d) holds
        their points contiguously by cluster, ``labels`` (N,) each point's cluster
        (0..K-1), ``counts`` (K,) the points per cluster.
        """


TRUST_REGIONS: Registry[TrustRegion] = Registry("trust region", TrustRegion)


def trust_region_shapes(
    X: Array, centroids: Array, labels: Array, counts: Array, q_reg: float
) -> Array:
    """
    Shape matrix Q (K, d, d) of each cluster's trust region.

    Q is the inverse covariance of the cluster's points (eigenvalues clamped
    below at ``q_reg``), scaled so that the farthest point lies on the boundary:
    max over the cluster's points of dx^T Q dx is 1. In the companion's notation (its
    trust-ellipsoid algorithm) ``Cov``, the unnormalized covariance, is the scatter matrix
    ``Sigma``, ``q_reg`` is ``gamma``, ``max_dist`` is ``alpha`` and ``Q`` is
    ``T = Sigma^-1 / alpha``. Points are stored
    contiguously by cluster; ``labels`` gives each point's cluster (0..K-1).
    """
    K = len(counts)
    starts = xp.r_[0, xp.cumsum(counts)[:-1]]  # (K,)
    dx = X - centroids[labels]  # (N, d)

    Cov = symmetric_point_covariance(X, centroids, labels, K)
    Lambda, U = xp.linalg.eigh(Cov)
    if q_reg > 0.0:
        Lambda = xp.maximum(Lambda, q_reg)
    Lambda_inv = 1.0 / xp.maximum(Lambda, TINY_EPS)  # (K, d)

    proj = run_fused_mahalanobis_projection(dx, labels, U)  # (N, d)
    dist_sq_mahalanobis = xp.sum(proj**2 * Lambda_inv[labels], axis=1)  # (N,)
    max_dist = reduce_max_1d(dist_sq_mahalanobis, starts)  # (K,)

    scale_fac = 1.0 / xp.maximum(max_dist, TINY_EPS)
    return (
        det_matmul(U * (scale_fac[:, None] * Lambda_inv)[:, None, :], U.transpose(0, 2, 1))
    )  # (K, d, d)


@TRUST_REGIONS.register
class CovarianceEllipsoid(TrustRegion):
    """The default: scaled, eigenvalue-floored inverse covariance (``trust_region_shapes``)."""

    name = "covariance_ellipsoid"

    def __init__(self, q_reg: float = 1e-6):
        self.q_reg = float(q_reg)

    def settings(self) -> dict:
        return {"q_reg": self.q_reg}

    def shapes(self, X, centroids, labels, counts):
        return trust_region_shapes(X, centroids, labels, counts, self.q_reg)
