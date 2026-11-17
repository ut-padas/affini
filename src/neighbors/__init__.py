"""Nearest-neighbour search over leaf centroids, exact trust-region containment, and the k-means initial clustering (CPU and GPU)."""

from src.neighbors.containment import (
    TrustRegionContainment,
    best_containing,
    build_box_tree,
)

__all__ = ["TrustRegionContainment", "best_containing", "build_box_tree"]
