"""Shared initial, calibration, and evaluation designs."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .randomness import derived_seed


def _scale_unit(points: np.ndarray, low: float, high: float) -> np.ndarray:
    return qmc.scale(points, [low] * points.shape[1], [high] * points.shape[1])


def screened_latin_hypercube(
    n: int,
    dimension: int,
    low: float,
    high: float,
    seed: int,
    candidates: int = 4,
) -> np.ndarray:
    """Choose the best of several LHS designs using an approximate maximin score."""

    best_points: np.ndarray | None = None
    best_score = -np.inf
    probe_size = min(n, 4_096)

    for candidate in range(max(1, candidates)):
        candidate_seed = derived_seed(seed, "initial_lhs", candidate)
        sampler = qmc.LatinHypercube(d=dimension, scramble=True, seed=candidate_seed)
        points = sampler.random(n)
        probe_rng = np.random.default_rng(derived_seed(seed, "lhs_probe", candidate))
        if n > probe_size:
            probe_ids = probe_rng.choice(n, size=probe_size, replace=False)
            probe = points[probe_ids]
        else:
            probe = points
        distances, _ = cKDTree(probe).query(probe, k=2, workers=-1)
        score = float(np.quantile(distances[:, 1], 0.05))
        if score > best_score:
            best_score = score
            best_points = points

    assert best_points is not None
    return _scale_unit(best_points, low, high).astype(np.float64, copy=False)


def scrambled_sobol(
    n: int,
    dimension: int,
    low: float,
    high: float,
    seed: int,
) -> np.ndarray:
    """Generate the first n points of an independently scrambled Sobol net."""

    power = math.ceil(math.log2(n))
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    points = sampler.random_base2(power)[:n]
    return _scale_unit(points, low, high).astype(np.float64, copy=False)
