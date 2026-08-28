"""Approximate and reference trust-region membership calculations."""

from __future__ import annotations

import numpy as np

from src import _to_device, _to_host, xp
from src.surrogate import AFINNISurrogate


def approximate_unguaranteed_mask(
    surrogate: AFINNISurrogate,
    points: np.ndarray,
    nearest_clusters: int,
    batch_size: int,
) -> np.ndarray:
    """Classify points with the study's nearest-centroid approximation."""

    if len(points) == 0:
        return np.empty(0, dtype=bool)
    mask = surrogate.filter_unguaranteed_local(
        _to_device(points),
        k_coarse=nearest_clusters,
        batch_size=batch_size,
    )
    return np.asarray(_to_host(mask), dtype=bool)


def exact_unguaranteed_mask(
    surrogate: AFINNISurrogate,
    points: np.ndarray,
    query_block: int = 256,
    cluster_block: int = 2_048,
) -> np.ndarray:
    """Evaluate membership against every ellipsoid in bounded-memory blocks."""

    if len(points) == 0:
        return np.empty(0, dtype=bool)

    centers = surrogate.X_src
    matrices = surrogate.Qs_src
    assert centers is not None and matrices is not None
    result = np.ones(len(points), dtype=bool)

    for query_start in range(0, len(points), query_block):
        query_end = min(query_start + query_block, len(points))
        query = xp.asarray(points[query_start:query_end])
        min_q = xp.full(len(query), xp.inf, dtype=xp.float64)
        for cluster_start in range(0, surrogate.N_src, cluster_block):
            cluster_end = min(cluster_start + cluster_block, surrogate.N_src)
            block_centers = centers[cluster_start:cluster_end]
            block_matrices = matrices[cluster_start:cluster_end]
            delta = query[:, None, :] - block_centers[None, :, :]
            q_values = xp.einsum("qci,cij,qcj->qc", delta, block_matrices, delta)
            min_q = xp.minimum(min_q, xp.min(q_values, axis=1))
        result[query_start:query_end] = np.asarray(_to_host(min_q > 1.0))
    return result


def audit_membership(
    surrogate: AFINNISurrogate,
    points: np.ndarray,
    nearest_clusters: int,
    batch_size: int,
    max_pairs: int,
) -> dict[str, float | int | bool]:
    """Compare approximate membership with the full ellipsoid union."""

    pair_count = len(points) * surrogate.N_src
    if pair_count > max_pairs:
        return {
            "membership_audit_ran": False,
            "membership_audit_pairs": pair_count,
            "membership_audit_disagreements": 0,
            "membership_false_inside": 0,
            "membership_false_outside": 0,
            "membership_disagreement_rate": float("nan"),
        }

    approx = approximate_unguaranteed_mask(
        surrogate, points, nearest_clusters, batch_size
    )
    exact = exact_unguaranteed_mask(surrogate, points)
    disagreements = approx != exact
    false_inside = (~approx) & exact
    false_outside = approx & (~exact)
    return {
        "membership_audit_ran": True,
        "membership_audit_pairs": pair_count,
        "membership_audit_disagreements": int(disagreements.sum()),
        "membership_false_inside": int(false_inside.sum()),
        "membership_false_outside": int(false_outside.sum()),
        "membership_disagreement_rate": float(disagreements.mean()),
    }
