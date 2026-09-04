"""Paper-matched aggressive P_out race for 5D Rosenbrock.

This driver matches the SC26 Rosenbrock construction configuration while
comparing outer-loop acquisition rules.  It deliberately uses a separate
fixed test set and appends every retained gap proposal, subject to a total
training-set safety cap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree

from src import IS_CUPY_BACKEND, _to_device, _to_host, xp
from src.adaptive_clustering_utils import SurrogateMode
from src.problems.rosenbrock import rosenbrock

from .config import StudyConfig
from .experiment import _train
from .metrics import evaluate, peak_host_memory_gib
from .randomness import derived_seed


RACE_METHODS = ("uniform", "halo", "cloud", "random_chord")
METHOD_LABELS = {
    "uniform": "Uniform",
    "halo": "Halo",
    "cloud": "Cloud",
    "random_chord": "Random chord",
}


@dataclass(frozen=True)
class SC26PoutRaceConfig:
    """Scientific and operational parameters for the SC26-style race."""

    dimension: int = 5
    seed: int = 0
    methods: tuple[str, ...] = RACE_METHODS
    tolerance: float = 0.05
    domain_low: float = -2.0
    domain_high: float = 3.0
    rosenbrock_scale: float = 100.0

    # Exact SC26 d=5 starting-design parameters: floor(factor * 1.3**5).
    initial_n: int = 185_646
    test_size: int = 92_823
    num_initial_clusters: int = 92
    min_points_per_cluster: int = 371
    error_metric: str = "Linf-local"
    error_eval_mode: str = "affine_cov_hess"
    initial_clustering_method: str = "kmeans"
    max_levels: int = 1_000

    # The original SC26 run used work stealing.  This race intentionally runs
    # one MPI rank on an idev GPU, where work stealing is inapplicable.
    use_work_stealing: bool = False
    membership_coarse_factor: int = 10

    target_pout: float = 0.02
    max_rounds: int = 10
    max_training_points: int = 5_000_000
    points_per_cluster: int = 500
    cluster_reference: int = 10_000
    min_points_per_cluster_candidate: int = 10
    candidate_chunk_size: int = 50_000
    halo_gamma: float = 1.5
    chord_neighbours: int = 5
    chord_noise: float = 0.3
    output_dir: str = "results/sc26_rosenbrock_pout_race/dim5"

    def validate(self) -> None:
        if self.dimension != 5:
            raise ValueError("This paper-matched driver is specified for d=5 only.")
        if self.error_metric != "Linf-local":
            raise ValueError("The SC26 paper configuration uses Linf-local.")
        if self.error_eval_mode != "affine_cov_hess":
            raise ValueError("The SC26 paper configuration uses affine_cov_hess.")
        if not self.methods or set(self.methods) - set(RACE_METHODS):
            raise ValueError(f"methods must be drawn from {RACE_METHODS}")
        if not 0 < self.target_pout < 1:
            raise ValueError("target_pout must be a fraction in (0, 1).")
        if self.max_rounds < 1 or self.max_training_points < self.initial_n:
            raise ValueError("Invalid round or training-point limit.")
        if self.points_per_cluster < 1 or self.chord_neighbours < 1:
            raise ValueError("Candidate counts must be positive.")

    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()


@dataclass
class ProposalStats:
    method: str
    clusters: int
    points_per_cluster: int
    proposed: int = 0
    domain_rejected: int = 0
    covered_rejected: int = 0
    retained: int = 0
    generation_seconds: float = 0.0
    filtering_seconds: float = 0.0


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=True) + "\n"


def _config_hash(config: SC26PoutRaceConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:16]


def _append_csv(path: Path, row: dict[str, Any], unique_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as handle:
            previous = list(csv.DictReader(handle))
        if previous and str(previous[-1][unique_key]) == str(row[unique_key]):
            return
        fields = list(previous[0])
    else:
        fields = list(row)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _save_state(path: Path, training: np.ndarray, next_round: int, cumulative_seconds: float, cumulative_fit_points: int) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        training=training,
        next_round=np.asarray(next_round, dtype=np.int64),
        cumulative_seconds=np.asarray(cumulative_seconds),
        cumulative_fit_points=np.asarray(cumulative_fit_points, dtype=np.int64),
    )
    temporary.replace(path)


def _load_state(path: Path) -> tuple[np.ndarray, int, float, int]:
    with np.load(path) as saved:
        return (
            np.asarray(saved["training"]),
            int(saved["next_round"]),
            float(saved["cumulative_seconds"]),
            int(saved["cumulative_fit_points"]),
        )


def _ensure_shared_designs(config: SC26PoutRaceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the SC26 train/test stream: training draw first, then test draw."""

    shared = config.output_path() / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    path = shared / "designs.npz"
    if path.exists():
        with np.load(path) as saved:
            return (
                np.asarray(saved["initial"]),
                np.asarray(saved["evaluation"]),
                np.asarray(saved["truth"]),
            )

    xp.random.seed(config.seed)
    initial = np.asarray(
        _to_host(
            xp.random.uniform(
                config.domain_low,
                config.domain_high,
                size=(config.initial_n, config.dimension),
            )
        )
    )
    evaluation = np.asarray(
        _to_host(
            xp.random.uniform(
                config.domain_low,
                config.domain_high,
                size=(config.test_size, config.dimension),
            )
        )
    )
    truth = np.asarray(
        _to_host(rosenbrock(_to_device(evaluation), a=config.rosenbrock_scale))
    )
    np.savez_compressed(path, initial=initial, evaluation=evaluation, truth=truth)
    return initial, evaluation, truth


def _study_config(config: SC26PoutRaceConfig) -> StudyConfig:
    return StudyConfig(
        preset="sc26_pout_race",
        dimensions=(config.dimension,),
        seeds=(config.seed,),
        initial_n=config.initial_n,
        batch_size=1,
        test_size=config.test_size,
        calibration_size=max(100, min(1_024, config.test_size)),
        domain_low=config.domain_low,
        domain_high=config.domain_high,
        tolerance=config.tolerance,
        rosenbrock_scale=config.rosenbrock_scale,
        halo_gamma=config.halo_gamma,
        membership_k=config.membership_coarse_factor,
        num_initial_clusters=config.num_initial_clusters,
        min_points_per_cluster=config.min_points_per_cluster,
        error_metric=config.error_metric,
        initial_clustering_method=config.initial_clustering_method,
        max_levels=config.max_levels,
        use_work_stealing=config.use_work_stealing,
        output_dir=config.output_dir,
    )


def _dynamic_ppc(config: SC26PoutRaceConfig, clusters: int) -> int:
    if clusters <= config.cluster_reference:
        return config.points_per_cluster
    return max(
        config.min_points_per_cluster_candidate,
        int(config.points_per_cluster * config.cluster_reference / clusters),
    )


def _filter_candidates(surrogate: Any, candidates: np.ndarray, coarse_factor: int) -> tuple[np.ndarray, int, float]:
    if not len(candidates):
        return candidates, 0, 0.0
    start = time.perf_counter()
    unguaranteed = surrogate.filter_unguaranteed_local(
        _to_device(candidates), k_coarse=coarse_factor, batch_size=50_000
    )
    elapsed = time.perf_counter() - start
    mask = np.asarray(_to_host(unguaranteed), dtype=bool)
    return candidates[mask], int((~mask).sum()), elapsed


def _local_geometry(surrogate: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(_to_host(surrogate.X_src), dtype=np.float64)
    matrices = np.asarray(_to_host(surrogate.Qs_src), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues, 1e-12))
    return centers, eigenvectors, inverse_sqrt


def _retained_chunks(
    config: SC26PoutRaceConfig,
    surrogate: Any,
    method: str,
    round_number: int,
) -> tuple[np.ndarray, ProposalStats]:
    clusters = int(surrogate.N_src)
    ppc = _dynamic_ppc(config, clusters)
    stats = ProposalStats(method=method, clusters=clusters, points_per_cluster=ppc)
    rng = np.random.default_rng(derived_seed(config.seed, config.dimension, method, round_number))
    survivors: list[np.ndarray] = []

    centers: np.ndarray | None = None
    vectors: np.ndarray | None = None
    inverse_sqrt: np.ndarray | None = None
    neighbours: np.ndarray | None = None
    if method != "uniform":
        geometry_start = time.perf_counter()
        centers, vectors, inverse_sqrt = _local_geometry(surrogate)
        if method == "random_chord":
            if clusters < 2:
                raise RuntimeError("Random chord requires at least two leaf clusters.")
            k = min(config.chord_neighbours + 1, clusters)
            neighbours = cKDTree(centers).query(centers, k=k)[1]
            if neighbours.ndim == 1:
                neighbours = neighbours[:, None]
            neighbours = neighbours[:, 1:]
        stats.generation_seconds += time.perf_counter() - geometry_start

    total = clusters * ppc
    if method == "uniform":
        for start_id in range(0, total, config.candidate_chunk_size):
            count = min(config.candidate_chunk_size, total - start_id)
            start = time.perf_counter()
            candidates = rng.uniform(
                config.domain_low, config.domain_high, size=(count, config.dimension)
            )
            stats.generation_seconds += time.perf_counter() - start
            stats.proposed += count
            retained, covered, elapsed = _filter_candidates(
                surrogate, candidates, config.membership_coarse_factor
            )
            stats.covered_rejected += covered
            stats.filtering_seconds += elapsed
            if len(retained):
                survivors.append(retained)
    else:
        assert centers is not None and vectors is not None and inverse_sqrt is not None
        clusters_per_chunk = max(1, config.candidate_chunk_size // ppc)
        for start_id in range(0, clusters, clusters_per_chunk):
            end_id = min(clusters, start_id + clusters_per_chunk)
            count = end_id - start_id
            start = time.perf_counter()
            z = rng.standard_normal((count, ppc, config.dimension))
            if method in {"halo", "random_chord"}:
                z /= np.maximum(np.linalg.norm(z, axis=2, keepdims=True), 1e-15)
            scaled = z * inverse_sqrt[start_id:end_id, None, :]
            offsets = np.einsum("nij,npj->npi", vectors[start_id:end_id], scaled)

            if method == "halo":
                candidates = (
                    centers[start_id:end_id, None, :]
                    + config.halo_gamma * offsets
                ).reshape(-1, config.dimension)
            elif method == "cloud":
                candidates = (
                    centers[start_id:end_id, None, :]
                    + config.halo_gamma * offsets
                ).reshape(-1, config.dimension)
            else:
                assert neighbours is not None
                local_neighbours = neighbours[start_id:end_id]
                partner_col = rng.integers(0, local_neighbours.shape[1], size=(count, ppc))
                partner_ids = np.take_along_axis(local_neighbours, partner_col, axis=1)
                partner_centers = centers[partner_ids]
                alpha = rng.random((count, ppc, 1))
                chord_base = alpha * centers[start_id:end_id, None, :] + (1.0 - alpha) * partner_centers
                candidates = (chord_base + config.chord_noise * offsets).reshape(-1, config.dimension)

            stats.proposed += len(candidates)
            inside = np.all(
                (candidates >= config.domain_low) & (candidates <= config.domain_high), axis=1
            )
            stats.domain_rejected += int((~inside).sum())
            candidates = candidates[inside]
            stats.generation_seconds += time.perf_counter() - start
            retained, covered, elapsed = _filter_candidates(
                surrogate, candidates, config.membership_coarse_factor
            )
            stats.covered_rejected += covered
            stats.filtering_seconds += elapsed
            if len(retained):
                survivors.append(retained)

    points = np.concatenate(survivors) if survivors else np.empty((0, config.dimension))
    stats.retained = len(points)
    return points, stats


def _limit_additions(points: np.ndarray, available: int, seed: int) -> np.ndarray:
    if len(points) <= available:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=available, replace=False)]


def run_arm(config: SC26PoutRaceConfig, method: str, index: int) -> str:
    root = config.output_path()
    arm = root / method
    arm.mkdir(parents=True, exist_ok=True)
    complete = arm / "COMPLETE.json"
    if complete.exists():
        print(f"[{index}/{len(config.methods)}] {METHOD_LABELS[method]}: already complete")
        return "skipped"

    initial, evaluation, truth = _ensure_shared_designs(config)
    state_path = arm / "state.npz"
    if state_path.exists():
        training, next_round, cumulative_seconds, cumulative_fit_points = _load_state(state_path)
    else:
        training, next_round, cumulative_seconds, cumulative_fit_points = initial.copy(), 0, 0.0, 0

    study = _study_config(config)
    print(f"[{index}/{len(config.methods)}] {METHOD_LABELS[method]}: resume={next_round}, n_train={len(training):,}")
    for round_number in range(next_round, config.max_rounds + 1):
        started = time.perf_counter()
        print(f"  [round {round_number}/{config.max_rounds}] fitting {len(training):,} points", flush=True)
        surrogate, train_seconds = _train(study, training, config.dimension, config.seed)
        errors, _ = evaluate(
            surrogate, evaluation, truth, config.tolerance, config.membership_coarse_factor
        )
        elapsed = time.perf_counter() - started
        cumulative_seconds += elapsed
        cumulative_fit_points += len(training)
        p_out = float(errors["p_out"])
        hit_target = p_out < config.target_pout
        before = len(training)
        row = {
            "config_hash": _config_hash(config), "method": method, "method_label": METHOD_LABELS[method],
            "round": round_number, "dimension": config.dimension, "rtol": config.tolerance,
            "n_train_before": before, "n_train_after": before, "n_added": 0,
            "N": surrogate.profiling_stats.get("total_f_evals", float("nan")),
            "n_clusters": surrogate.N_src, "p_out_percent": 100.0 * p_out,
            "e_in": errors["l2_relative_in"], "e_out": errors["l2_relative_out"],
            "e": errors["l2_relative_all"], "p_bad_given_in_percent": 100.0 * errors["p_bad_given_in"],
            "train_seconds": train_seconds, "evaluation_seconds": errors["evaluation_seconds"],
            "round_seconds": elapsed, "cumulative_seconds": cumulative_seconds,
            "cumulative_fit_points": cumulative_fit_points, "peak_host_memory_gib": peak_host_memory_gib(),
            "termination": "target" if hit_target else "", "error_metric": config.error_metric,
        }
        print(f"    p_out={row['p_out_percent']:.2f}% e_in={row['e_in']:.3e} e={row['e']:.3e} N_c={surrogate.N_src:,}")

        if hit_target or round_number == config.max_rounds or before >= config.max_training_points:
            row["termination"] = "target" if hit_target else ("max_rounds" if round_number == config.max_rounds else "max_training_points")
            _append_csv(arm / "metrics.csv", row, "round")
            complete.write_text(_json({"status": row["termination"], "method": method, "round": round_number, "n_train": before, "p_out_percent": row["p_out_percent"]}))
            return str(row["termination"])

        acquisition_started = time.perf_counter()
        candidates, stats = _retained_chunks(config, surrogate, method, round_number)
        available = config.max_training_points - before
        additions = _limit_additions(candidates, available, derived_seed(config.seed, method, round_number, "cap"))
        acquisition_seconds = time.perf_counter() - acquisition_started
        cumulative_seconds += acquisition_seconds
        row.update({"n_train_after": before + len(additions), "n_added": len(additions), "round_seconds": elapsed + acquisition_seconds, "cumulative_seconds": cumulative_seconds})
        _append_csv(arm / "metrics.csv", row, "round")
        acquisition = {"config_hash": _config_hash(config), "method": method, "round_from": round_number, "n_train_before": before, "n_train_after": before + len(additions), "training_cap_applied": len(additions) < len(candidates), "acquisition_seconds": acquisition_seconds, **asdict(stats)}
        _append_csv(arm / "acquisitions.csv", acquisition, "round_from")
        print(f"    retained={len(additions):,}/{stats.proposed:,}; next fit={before + len(additions):,}; acquisition={acquisition_seconds:.1f}s", flush=True)
        if not len(additions):
            complete.write_text(_json({"status": "no_survivors", "method": method, "round": round_number, "n_train": before, "p_out_percent": row["p_out_percent"]}))
            return "no_survivors"
        training = np.vstack([training, additions])
        _save_state(state_path, training, round_number + 1, cumulative_seconds, cumulative_fit_points)
    raise AssertionError("unreachable")


def run(config: SC26PoutRaceConfig) -> int:
    config.validate()
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("This idev race requires exactly one MPI rank.")
    root = config.output_path()
    root.mkdir(parents=True, exist_ok=True)
    serialized = asdict(config)
    serialized["methods"] = list(config.methods)
    path = root / "race_config.json"
    if path.exists() and json.loads(path.read_text()) != serialized:
        raise RuntimeError(f"{path} differs from this configuration. Choose a new --output.")
    path.write_text(_json(serialized))
    print(f"SC26-matched Rosenbrock P_out race: d=5, rtol=0.05, target={100*config.target_pout:g}%, methods={', '.join(config.methods)}, GPU={IS_CUPY_BACKEND}")
    statuses = {method: run_arm(config, method, index) for index, method in enumerate(config.methods, 1)}
    (root / "RACE_COMPLETE.json").write_text(_json(statuses))
    return 0


def _methods(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=SC26PoutRaceConfig.output_dir)
    parser.add_argument("--methods", type=_methods, default=RACE_METHODS)
    parser.add_argument("--target-pout", type=float, default=0.02)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-training-points", type=int, default=5_000_000)
    parser.add_argument("--points-per-cluster", type=int, default=500)
    parser.add_argument("--cluster-reference", type=int, default=10_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(SC26PoutRaceConfig(output_dir=args.output, methods=args.methods, target_pout=args.target_pout, max_rounds=args.max_rounds, max_training_points=args.max_training_points, points_per_cluster=args.points_per_cluster, cluster_reference=args.cluster_reference))


if __name__ == "__main__":
    raise SystemExit(main())
