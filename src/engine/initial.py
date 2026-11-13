"""The initial clustering that training starts from (k-means, kNN-DBSCAN or a uniform grid), computed on rank 0."""

import os

import numpy as np

from src import IS_CUPY_BACKEND, Array, to_host, xp
from src.config import TrainingConfig
from src.numerics import TINY_EPS, uniform_grid_bins


def uniform_grid_cluster_labels(
    X_host: np.ndarray, num_initial_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assigns each point in X_host to a cluster based on a uniform grid.

    The grid is constructed by dividing the bounding box of X_host into bins along each dimension. The number of bins in each dimension is determined by factorizing num_initial_clusters into prime factors and assigning them to dimensions in a way that balances the bin sizes.

    Returns:
    - labels: An array of shape (N,) containing the cluster label for each point.
    - bins: An array of shape (d,) containing the number of bins along each dimension. The total number of clusters is the product of the values in bins.
    """
    lower, upper = np.min(X_host, axis=0), np.max(X_host, axis=0)
    extents = np.maximum(upper - lower, TINY_EPS)
    bins = uniform_grid_bins(extents, num_initial_clusters)
    rel = (X_host - lower[None, :]) / extents
    rel = np.clip(rel, 0.0, 1.0 - np.finfo(np.float64).eps)
    coords = np.floor(rel * bins[None, :]).astype(np.int64)
    coords = np.minimum(coords, bins[None, :] - 1)
    labels = np.ravel_multi_index(coords.T, dims=tuple(int(b) for b in bins))
    return labels.astype(np.int64), bins


def threads_for_rank0(node_ranks: int) -> int:
    """Threads for work done by rank 0 alone: its own share (``OMP_NUM_THREADS``, default 1 under MPI) times the ranks on its node."""
    env = os.environ.get("OMP_NUM_THREADS", "")
    per_rank = int(env) if env.isdigit() else (1 if node_ranks > 1 else os.cpu_count() or 1)
    return max(1, per_rank * node_ranks)


def initial_clustering(X: Array, config: TrainingConfig, *, threads: int | None = None) -> np.ndarray:
    """
    The (N,) int64 host labels of the initial clustering that training with
    ``config`` starts from (``initial_clustering_method``, ``num_initial_clusters``,
    ``seed``); deterministic for a given seed and backend. ``threads`` (CPU k-means
    only) defaults to ``OMP_NUM_THREADS``, else every core.
    """
    method = config.initial_clustering_method
    if method == "kmeans":
        # Custom k-means (src/neighbors/kmeans.py). The CPU path keeps the previous two Lloyd
        # iterations; training passes the cores of rank 0's node, whose other ranks wait.
        from src.neighbors.kmeans import kmeans_labels

        if IS_CUPY_BACKEND:
            labels = kmeans_labels(xp.asarray(X, dtype=xp.float64), config.num_initial_clusters, config.seed)
        else:
            labels = kmeans_labels(
                np.asarray(to_host(X), dtype=np.float64), config.num_initial_clusters, config.seed,
                max_iter=2, threads=threads_for_rank0(1) if threads is None else threads,
            )  # fmt: skip
    elif method == "knndbscan":
        from knndbscan import knn_dbscan

        labels = knn_dbscan(
            xp.asarray(X, dtype=xp.float64), eps=0.5, min_samples=config.min_pts_per_cluster,
            device="gpu" if IS_CUPY_BACKEND else "cpu",
        )  # fmt: skip
    elif method == "uniform_grid":
        labels, _ = uniform_grid_cluster_labels(to_host(X), config.num_initial_clusters)
    else:
        raise ValueError(f"Invalid initial clustering method: {method!r}")
    return np.asarray(to_host(labels)).astype(np.int64)
