"""
Training utilities (timers, grid initialization, saving generated points), plus
the names that used to live here, re-exported so existing imports keep working.

The former contents now live in:

- ``src.modes``: ``SurrogateMode``
- ``src.config``: ``ClusteringConfig``
- ``src.evaluation``: ``Evaluator``
- ``src.numerics``: CUDA kernels and batched reductions/solves
- ``src.acceptance``: ``cluster_errors`` and the acceptance criteria
- ``src.trust_regions``: ``trust_region_shapes``
- ``src.local_models``: ``lsq_hessians``
"""

import glob
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from src import IS_CUPY_BACKEND, xp
from src.acceptance import cluster_errors
from src.config import ClusteringConfig, RegionSettings  # noqa: F401  (re-exported)
from src.evaluation import Evaluator
from src.local_models import lsq_hessians
from src.modes import SurrogateMode
from src.numerics import (
    TINY_EPS,
    _reduce_max_1d,
    _reduce_sum_1d,
    _symmetric_point_covariance,
    convert_to_2d_array,
    convert_to_3d_array,
    get_n_smallest_per_block,
    lsq_hessian_in_eigenbasis,
    repeats,
    run_fused_cross_cov,
    run_fused_mahalanobis_projection,
    run_fused_split_projection,
    run_fused_stats,
    solve_regularized_covariance_inplace,
)
from src.trust_regions import trust_region_shapes

logger = logging.getLogger(__name__)

__all__ = [
    "TINY_EPS",
    "ClusteringConfig",
    "CudaTimer",
    "Evaluator",
    "SurrogateMode",
    "_reduce_max_1d",
    "_reduce_sum_1d",
    "_symmetric_point_covariance",
    "cluster_errors",
    "combine_generated_data",
    "convert_to_2d_array",
    "convert_to_3d_array",
    "get_n_smallest_per_block",
    "get_prime_factors",
    "lsq_hessian_in_eigenbasis",
    "lsq_hessians",
    "repeats",
    "run_fused_cross_cov",
    "run_fused_mahalanobis_projection",
    "run_fused_split_projection",
    "run_fused_stats",
    "save_generated_data",
    "solve_regularized_covariance_inplace",
    "trust_region_shapes",
    "uniform_grid_cluster_labels",
]


# -----------------------------------------------------------------------------
# MISC UTILS
# -----------------------------------------------------------------------------
class CudaTimer:
    """Helper context manager for timing GPU operations with synchronization."""

    def __init__(self, key: str, stats: dict, enabled: bool):
        self.key = key
        self.stats = stats
        self.enabled = enabled
        self.t0 = None

    def __enter__(self):
        if self.enabled:
            if IS_CUPY_BACKEND:
                xp.cuda.Device().synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.enabled:
            if IS_CUPY_BACKEND:
                xp.cuda.Device().synchronize()
            dt = time.perf_counter() - self.t0  # type: ignore
            self.stats[self.key] += dt




def get_prime_factors(n: int) -> list[int]:
    """
    Returns the prime factors of n as a list.
    For example, get_prime_factors(12) returns [2, 2, 3].
    """
    if n < 2:
        return []
    factors: list[int] = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 2
    if n > 1:
        factors.append(n)
    return factors


def uniform_grid_bins(extents: np.ndarray, num_cells: int) -> np.ndarray:
    """
    (d,) int64 bins per dimension whose product is ``num_cells``: each prime factor,
    largest first, goes to the dimension with the widest current cell. Shared by the
    uniform-grid initial clustering of AFINNI mode and the grid roots of region mode, so both cut the same way.
    """
    if num_cells < 1:
        raise ValueError("num_initial_clusters must be >= 1")
    bins = np.ones(len(extents), dtype=np.int64)
    for p in sorted(get_prime_factors(num_cells), reverse=True):
        bins[int(np.argmax(extents / bins))] *= p
    return bins


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


"""
A new function to save the data to a single file.
I imagine this will cause problems with MPI, since two processes could read/write at the same time.
But for now I will run this without work stealing and it will be fine.
I just need the training data to do the testing with FALKON.
This will also be nice to have for later if, say, Skandan needs to compare training data.
"""


def save_generated_data(X, filename):
    """Write ``X`` to a uniquely named file in ``saved_training_data/`` (safe for concurrent ranks)."""
    path = Path("saved_training_data/")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    xp.save(f"saved_training_data/{filename}_{uuid.uuid4()}.npy", X)


def combine_generated_data(filename):
    """Stack the files written by ``save_generated_data`` into ``<filename>.npy`` and delete the folder."""
    path = Path("saved_training_data/")
    input_dir = "saved_training_data"

    # Grab all .npy files in the folder with the correct beginning
    files = glob.glob(os.path.join(input_dir, f"{filename}_*.npy"))

    # Load all files into a list
    data_list = [np.load(f) for f in files]

    if data_list:
        # Stack them into one large matrix and save
        final_matrix = np.vstack(data_list)
        if path.exists():
            shutil.rmtree(input_dir)
        np.save(f"{filename}.npy", final_matrix)
    else:
        print("ERROR: No files found to stack.")
