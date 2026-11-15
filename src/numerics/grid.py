"""Uniform-grid factorization and cell numbering shared by AFINNI's grid initial clustering and ART's grid roots."""

import numpy as np


def get_prime_factors(n: int) -> list[int]:
    """Prime factors of ``n`` in ascending order, with multiplicity (12 -> [2, 2, 3]); [] for n < 2."""
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
    uniform-grid initial clustering of AFINNI and the grid roots of ART, so both cut the same way.
    """
    if num_cells < 1:
        raise ValueError("num_initial_clusters must be >= 1")
    bins = np.ones(len(extents), dtype=np.int64)
    for p in sorted(get_prime_factors(num_cells), reverse=True):
        bins[int(np.argmax(extents / bins))] *= p
    return bins


def grid_strides(bins: np.ndarray) -> np.ndarray:
    """
    (d,) int64 row-major strides of a grid with ``bins`` cells per dimension: cell
    ``coords @ grid_strides(bins)`` is the number ``np.ravel_multi_index`` gives. Computed
    directly because NumPy's index helpers take the grid's shape as an array shape, which
    NumPy limits to 64 dimensions.
    """
    bins = np.asarray(bins, dtype=np.int64)
    return np.r_[np.cumprod(bins[::-1])[::-1][1:], 1].astype(np.int64)


def grid_cell_coords(cells: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """(n, d) int64 coordinates of the row-major cell numbers ``cells``: the inverse of ``grid_strides``."""
    bins = np.asarray(bins, dtype=np.int64)
    return (np.asarray(cells, dtype=np.int64)[:, None] // grid_strides(bins)[None, :]) % bins[None, :]
