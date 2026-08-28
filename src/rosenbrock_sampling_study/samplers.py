"""Fixed-oracle-budget acquisition methods."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

from src import _to_host, xp
from src.surrogate import AFINNISurrogate

from .config import StudyConfig
from .membership import approximate_unguaranteed_mask


@dataclass
class AcquisitionStats:
    method: str
    requested: int
    proposed: int = 0
    domain_rejected: int = 0
    covered_rejected: int = 0
    duplicate_rejected: int = 0
    survived: int = 0
    selected: int = 0
    proposal_cap: int = 0
    feasible: bool = True
    generation_seconds: float = 0.0
    filter_seconds: float = 0.0
    selection_seconds: float = 0.0
    min_selected_distance: float = float("nan")
    median_selected_distance: float = float("nan")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AcquisitionResult:
    points: np.ndarray
    stats: AcquisitionStats


class AcquisitionInfeasibleError(RuntimeError):
    def __init__(self, message: str, result: AcquisitionResult):
        super().__init__(message)
        self.result = result


def _remove_duplicates(
    candidates: np.ndarray,
    training_tree: cKDTree,
    accepted: list[np.ndarray],
    absolute_tolerance: float,
) -> tuple[np.ndarray, int]:
    if len(candidates) == 0:
        return candidates, 0

    distances, _ = training_tree.query(candidates, k=1, workers=-1)
    keep = distances > absolute_tolerance
    rejected = int((~keep).sum())
    candidates = candidates[keep]
    if len(candidates) == 0:
        return candidates, rejected

    if accepted:
        accepted_points = np.concatenate(accepted, axis=0)
        accepted_distances, _ = cKDTree(accepted_points).query(
            candidates, k=1, workers=-1
        )
        keep = accepted_distances > absolute_tolerance
        rejected += int((~keep).sum())
        candidates = candidates[keep]
    if len(candidates) == 0:
        return candidates, rejected

    keys = np.rint(candidates / absolute_tolerance).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    rejected += len(candidates) - len(first)
    return candidates[first], rejected


def _select_diverse(
    pool: np.ndarray,
    training: np.ndarray,
    count: int,
    config: StudyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pool) < count:
        raise ValueError("The candidate pool is smaller than the requested batch.")
    if len(pool) == count:
        distances, _ = cKDTree(training).query(pool, k=1, workers=-1)
        return pool, distances

    distance_to_design, _ = cKDTree(training).query(pool, k=1, workers=-1)
    if (
        count > config.max_exact_maximin_batch
        or len(pool) > config.max_exact_maximin_pool
    ):
        selected_ids = np.argpartition(distance_to_design, -count)[-count:]
        selected_ids = selected_ids[np.argsort(distance_to_design[selected_ids])[::-1]]
        return pool[selected_ids], distance_to_design[selected_ids]

    min_distances = distance_to_design.copy()
    selected_ids = np.empty(count, dtype=np.int64)
    available = np.ones(len(pool), dtype=bool)
    for i in range(count):
        masked = np.where(available, min_distances, -np.inf)
        selected = int(np.argmax(masked))
        selected_ids[i] = selected
        available[selected] = False
        new_distances = np.linalg.norm(pool - pool[selected], axis=1)
        min_distances = np.minimum(min_distances, new_distances)
    return pool[selected_ids], distance_to_design[selected_ids]


def _prepare_local_geometry(
    surrogate: AFINNISurrogate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(_to_host(surrogate.X_src), dtype=np.float64)
    eigenvalues_device, eigenvectors_device = xp.linalg.eigh(surrogate.Qs_src)
    eigenvalues = np.asarray(_to_host(eigenvalues_device), dtype=np.float64)
    eigenvectors = np.asarray(_to_host(eigenvectors_device), dtype=np.float64)
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues, 1e-12))
    return centers, eigenvectors, inverse_sqrt


def acquire(
    method: str,
    surrogate: AFINNISurrogate,
    training: np.ndarray,
    count: int,
    rng: np.random.Generator,
    config: StudyConfig,
) -> AcquisitionResult:
    """Generate, filter, deduplicate, and downselect one acquisition batch."""

    stats = AcquisitionStats(
        method=method,
        requested=count,
        proposal_cap=count * config.proposal_cap_multiplier,
    )
    accepted: list[np.ndarray] = []
    accepted_count = 0
    dimension = training.shape[1]
    low, high = config.domain_low, config.domain_high
    absolute_tolerance = config.dedup_relative_tolerance * (high - low)
    training_tree = cKDTree(training)

    geometry = None
    if method in {"shell", "gaussian"}:
        start = time.perf_counter()
        geometry = _prepare_local_geometry(surrogate)
        stats.generation_seconds += time.perf_counter() - start

    while accepted_count < count and stats.proposed < stats.proposal_cap:
        remaining = count - accepted_count
        if method == "uniform":
            chunk_size = min(remaining, stats.proposal_cap - stats.proposed)
        else:
            chunk_size = min(
                config.proposal_chunk_size,
                max(1_024, 4 * remaining),
                stats.proposal_cap - stats.proposed,
            )
        start = time.perf_counter()

        if method in {"uniform", "uniform_gap"}:
            candidates = rng.uniform(low, high, size=(chunk_size, dimension))
        elif method in {"shell", "gaussian"}:
            assert geometry is not None
            centers, eigenvectors, inverse_sqrt = geometry
            source_ids = rng.integers(0, len(centers), size=chunk_size)
            z = rng.standard_normal((chunk_size, dimension))
            if method == "shell":
                norms = np.linalg.norm(z, axis=1, keepdims=True)
                z = z / np.maximum(norms, 1e-15)
            scaled = z * inverse_sqrt[source_ids]
            offsets = np.einsum(
                "nij,nj->ni", eigenvectors[source_ids], scaled, optimize=True
            )
            candidates = centers[source_ids] + config.halo_gamma * offsets
        else:
            raise ValueError(f"Unknown sampler {method!r}")

        stats.proposed += len(candidates)
        stats.generation_seconds += time.perf_counter() - start

        domain_mask = np.all((candidates >= low) & (candidates <= high), axis=1)
        stats.domain_rejected += int((~domain_mask).sum())
        candidates = candidates[domain_mask]

        if method != "uniform" and len(candidates):
            start = time.perf_counter()
            uncovered = approximate_unguaranteed_mask(
                surrogate,
                candidates,
                nearest_clusters=config.membership_k,
                batch_size=config.membership_batch_size,
            )
            stats.filter_seconds += time.perf_counter() - start
            stats.covered_rejected += int((~uncovered).sum())
            candidates = candidates[uncovered]

        candidates, duplicate_count = _remove_duplicates(
            candidates, training_tree, accepted, absolute_tolerance
        )
        stats.duplicate_rejected += duplicate_count
        if len(candidates):
            accepted.append(candidates)
            accepted_count += len(candidates)

        if method == "uniform" and accepted_count >= count:
            break

    if accepted:
        pool = np.concatenate(accepted, axis=0)
    else:
        pool = np.empty((0, dimension), dtype=np.float64)
    stats.survived = len(pool)

    if len(pool) < count:
        stats.feasible = False
        stats.selected = len(pool)
        result = AcquisitionResult(points=pool, stats=stats)
        raise AcquisitionInfeasibleError(
            f"{method} found {len(pool):,} of {count:,} required points after "
            f"{stats.proposed:,} proposals.",
            result,
        )

    start = time.perf_counter()
    selected, selected_distances = _select_diverse(pool, training, count, config)
    stats.selection_seconds = time.perf_counter() - start
    stats.selected = len(selected)
    stats.min_selected_distance = float(np.min(selected_distances))
    stats.median_selected_distance = float(np.median(selected_distances))
    return AcquisitionResult(points=selected, stats=stats)
