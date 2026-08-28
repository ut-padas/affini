"""Run an aggressive race to a target uncovered volume.

Unlike the fixed-budget study, this driver appends every surviving proposal in
each round. It is designed to reproduce the operating regime of
``p_out_iterative_experiment.py`` while comparing several proposal schemes on
the same initial design, test set, candidate count, and membership filter.
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

from src import IS_CUPY_BACKEND, _to_device, _to_host, xp

from .config import StudyConfig
from .designs import scrambled_sobol, screened_latin_hypercube
from .experiment import _train
from .metrics import evaluate, peak_host_memory_gib
from .randomness import derived_seed


RACE_METHODS = ("uniform_gap", "shell", "gaussian", "legacy_halo")
METHOD_LABELS = {
    "uniform_gap": "Uniform gap",
    "shell": "Normalized shell",
    "gaussian": "Gaussian cloud",
    "legacy_halo": "Legacy Gaussian halo with clipping",
}


@dataclass(frozen=True)
class RaceConfig:
    dimension: int = 5
    seed: int = 0
    methods: tuple[str, ...] = RACE_METHODS
    initial_n: int = 5_000
    test_size: int = 20_000
    target_pout: float = 0.02
    max_rounds: int = 8
    points_per_cluster: int = 200
    cluster_reference: int = 10_000
    min_points_per_cluster_candidate: int = 10
    candidate_chunk_size: int = 50_000
    membership_k: int = 10
    tolerance: float = 0.05
    halo_gamma: float = 1.5
    domain_low: float = -2.0
    domain_high: float = 3.0
    design: str = "legacy_random"
    max_new_points_per_round: int = 0
    output_dir: str = "results/rosenbrock_pout_race/dim5"

    def validate(self) -> None:
        if self.dimension < 2:
            raise ValueError("dimension must be at least 2")
        if not self.methods:
            raise ValueError("at least one method is required")
        unknown = sorted(set(self.methods) - set(RACE_METHODS))
        if unknown:
            raise ValueError(f"unknown race methods: {unknown}")
        if not 0 < self.target_pout < 1:
            raise ValueError("target_pout must be a fraction in (0, 1)")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if self.points_per_cluster < 1 or self.min_points_per_cluster_candidate < 1:
            raise ValueError("candidate counts must be positive")
        if self.candidate_chunk_size < 1_024:
            raise ValueError("candidate_chunk_size must be at least 1024")
        if self.membership_k < 1:
            raise ValueError("membership_k must be positive")
        if self.design not in {"legacy_random", "space_filling"}:
            raise ValueError("design must be legacy_random or space_filling")

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


def _config_hash(config: RaceConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _append_csv(path: Path, row: dict[str, Any], unique_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
        if existing and str(existing[-1][unique_key]) == str(row[unique_key]):
            return
        fieldnames = list(existing[0])
    else:
        fieldnames = list(row)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _save_state(
    path: Path,
    training: np.ndarray,
    next_checkpoint: int,
    cumulative_compute_seconds: float,
    cumulative_fit_points: int,
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        training=training,
        next_checkpoint=np.asarray(next_checkpoint, dtype=np.int64),
        cumulative_compute_seconds=np.asarray(cumulative_compute_seconds),
        cumulative_fit_points=np.asarray(cumulative_fit_points, dtype=np.int64),
    )
    temporary.replace(path)


def _load_state(path: Path) -> tuple[np.ndarray, int, float, int]:
    with np.load(path) as data:
        return (
            np.asarray(data["training"]),
            int(data["next_checkpoint"]),
            float(data["cumulative_compute_seconds"]),
            int(data["cumulative_fit_points"]),
        )


def _ensure_shared_designs(config: RaceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared = config.output_path() / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    path = shared / "designs.npz"
    if path.exists():
        with np.load(path) as data:
            return (
                np.asarray(data["initial"]),
                np.asarray(data["evaluation"]),
                np.asarray(data["truth"]),
            )

    if config.design == "legacy_random":
        # Match the old script's backend RNG and draw order: test first, then
        # training. On Vista this uses CuPy's generator on the GH200.
        xp.random.seed(config.seed)
        evaluation = np.asarray(
            _to_host(
                xp.random.uniform(
                    config.domain_low,
                    config.domain_high,
                    size=(config.test_size, config.dimension),
                )
            )
        )
        initial = np.asarray(
            _to_host(
                xp.random.uniform(
                    config.domain_low,
                    config.domain_high,
                    size=(config.initial_n, config.dimension),
                )
            )
        )
    else:
        initial = screened_latin_hypercube(
            n=config.initial_n,
            dimension=config.dimension,
            low=config.domain_low,
            high=config.domain_high,
            seed=derived_seed(config.seed, config.dimension, "race_initial"),
            candidates=4,
        )
        evaluation = scrambled_sobol(
            n=config.test_size,
            dimension=config.dimension,
            low=config.domain_low,
            high=config.domain_high,
            seed=derived_seed(config.seed, config.dimension, "race_evaluation"),
        )

    from src.problems.rosenbrock import rosenbrock

    truth = np.asarray(_to_host(rosenbrock(_to_device(evaluation), a=100.0)))
    np.savez_compressed(path, initial=initial, evaluation=evaluation, truth=truth)
    return initial, evaluation, truth


def _study_config(config: RaceConfig) -> StudyConfig:
    return StudyConfig(
        preset="pout_race",
        dimensions=(config.dimension,),
        seeds=(config.seed,),
        initial_n=config.initial_n,
        batch_size=1,
        test_size=config.test_size,
        calibration_size=max(100, min(1_024, config.test_size)),
        tolerance=config.tolerance,
        halo_gamma=config.halo_gamma,
        membership_k=config.membership_k,
        max_levels=50,
        use_work_stealing=False,
        output_dir=config.output_dir,
    )


def _dynamic_ppc(config: RaceConfig, clusters: int) -> int:
    if clusters <= config.cluster_reference:
        return config.points_per_cluster
    scaled = int(config.points_per_cluster * config.cluster_reference / clusters)
    return max(config.min_points_per_cluster_candidate, scaled)


def _filter_chunk(surrogate: Any, candidates: Any, membership_k: int) -> tuple[np.ndarray, int, float]:
    if len(candidates) == 0:
        return np.empty((0, surrogate.X_src.shape[1])), 0, 0.0
    start = time.perf_counter()
    uncovered = surrogate.filter_unguaranteed_local(
        candidates, k_coarse=membership_k, batch_size=50_000
    )
    elapsed = time.perf_counter() - start
    rejected = int(_to_host((~uncovered).sum()))
    return np.asarray(_to_host(candidates[uncovered])), rejected, elapsed


def acquire_all_survivors(
    config: RaceConfig,
    method: str,
    surrogate: Any,
    checkpoint: int,
) -> tuple[np.ndarray, ProposalStats]:
    """Return every proposal classified as uncovered."""

    clusters = int(surrogate.N_src)
    dimension = config.dimension
    ppc = _dynamic_ppc(config, clusters)
    total = clusters * ppc
    stats = ProposalStats(method=method, clusters=clusters, points_per_cluster=ppc)
    survivors: list[np.ndarray] = []
    survivor_count = 0

    random_seed = (
        config.seed + clusters
        if method == "legacy_halo"
        else derived_seed(config.seed, config.dimension, method, checkpoint)
    )
    xp.random.seed(random_seed % (2**31 - 1))
    low, high = config.domain_low, config.domain_high

    eigenvectors = inverse_sqrt = None
    if method in {"shell", "gaussian", "legacy_halo"}:
        start = time.perf_counter()
        eigenvalues, eigenvectors = xp.linalg.eigh(surrogate.Qs_src)
        inverse_sqrt = 1.0 / xp.sqrt(xp.maximum(eigenvalues, 1e-12))
        stats.generation_seconds += time.perf_counter() - start

    if method == "uniform_gap":
        for start_id in range(0, total, config.candidate_chunk_size):
            count = min(config.candidate_chunk_size, total - start_id)
            start = time.perf_counter()
            candidates = xp.random.uniform(low, high, size=(count, dimension))
            stats.generation_seconds += time.perf_counter() - start
            stats.proposed += count
            kept, covered, elapsed = _filter_chunk(
                surrogate, candidates, config.membership_k
            )
            stats.covered_rejected += covered
            stats.filtering_seconds += elapsed
            if len(kept):
                survivors.append(kept)
                survivor_count += len(kept)
    else:
        assert eigenvectors is not None and inverse_sqrt is not None
        clusters_per_chunk = max(1, config.candidate_chunk_size // ppc)
        for cluster_start in range(0, clusters, clusters_per_chunk):
            cluster_end = min(clusters, cluster_start + clusters_per_chunk)
            block_n = cluster_end - cluster_start
            start = time.perf_counter()
            z = xp.random.standard_normal((block_n, ppc, dimension))
            if method == "shell":
                norms = xp.linalg.norm(z, axis=2, keepdims=True)
                z = z / xp.maximum(norms, 1e-15)
            scaled = z * inverse_sqrt[cluster_start:cluster_end, None, :]
            offsets = xp.einsum(
                "nij,npj->npi",
                eigenvectors[cluster_start:cluster_end],
                scaled,
            )
            candidates = (
                surrogate.X_src[cluster_start:cluster_end, None, :]
                + config.halo_gamma * offsets
            ).reshape(-1, dimension)
            stats.proposed += len(candidates)
            if method == "legacy_halo":
                candidates = xp.clip(candidates, low, high)
            else:
                inside_domain = xp.all((candidates >= low) & (candidates <= high), axis=1)
                stats.domain_rejected += int(_to_host((~inside_domain).sum()))
                candidates = candidates[inside_domain]
            stats.generation_seconds += time.perf_counter() - start
            kept, covered, elapsed = _filter_chunk(
                surrogate, candidates, config.membership_k
            )
            stats.covered_rejected += covered
            stats.filtering_seconds += elapsed
            if len(kept):
                survivors.append(kept)
                survivor_count += len(kept)

    if survivors:
        result = np.concatenate(survivors, axis=0)
    else:
        result = np.empty((0, dimension), dtype=np.float64)
    if config.max_new_points_per_round and len(result) > config.max_new_points_per_round:
        rng = np.random.default_rng(
            derived_seed(config.seed, config.dimension, method, checkpoint, "cap")
        )
        selected = rng.choice(
            len(result), size=config.max_new_points_per_round, replace=False
        )
        result = result[selected]
    stats.retained = len(result)
    return result, stats


def run_arm(config: RaceConfig, method: str, arm_index: int) -> str:
    root = config.output_path()
    arm = root / method
    arm.mkdir(parents=True, exist_ok=True)
    complete = arm / "COMPLETE.json"
    if complete.exists():
        print(f"[arm {arm_index}/{len(config.methods)}] {method}: already complete")
        return "skipped"

    initial, evaluation, truth = _ensure_shared_designs(config)
    state_path = arm / "state.npz"
    if state_path.exists():
        training, next_checkpoint, cumulative_compute, cumulative_fit_points = _load_state(state_path)
    else:
        training = initial.copy()
        next_checkpoint = 0
        cumulative_compute = 0.0
        cumulative_fit_points = 0

    study_config = _study_config(config)
    print(
        f"[arm {arm_index}/{len(config.methods)}] {METHOD_LABELS[method]} "
        f"resume={next_checkpoint} n_train={len(training):,}",
        flush=True,
    )

    for checkpoint in range(next_checkpoint, config.max_rounds + 1):
        checkpoint_start = time.perf_counter()
        print(
            f"  [checkpoint {checkpoint}/{config.max_rounds}] fitting on "
            f"{len(training):,} points",
            flush=True,
        )
        surrogate, train_seconds = _train(
            study_config, training, config.dimension, config.seed
        )
        test_metrics, _ = evaluate(
            surrogate,
            evaluation,
            truth,
            config.tolerance,
            config.membership_k,
        )
        checkpoint_seconds = time.perf_counter() - checkpoint_start
        cumulative_compute += checkpoint_seconds
        cumulative_fit_points += len(training)
        p_out = float(test_metrics["p_out"])
        target_reached = p_out <= config.target_pout
        metric_row = {
            "config_hash": _config_hash(config),
            "dimension": config.dimension,
            "seed": config.seed,
            "method": method,
            "checkpoint": checkpoint,
            "n_train": len(training),
            "n_added_unique": len(training) - config.initial_n,
            "cumulative_fit_points": cumulative_fit_points,
            "n_clusters": surrogate.N_src,
            "p_out": p_out,
            "p_bad_given_in": test_metrics["p_bad_given_in"],
            "l2_relative_all": test_metrics["l2_relative_all"],
            "l2_relative_in": test_metrics["l2_relative_in"],
            "l2_relative_out": test_metrics["l2_relative_out"],
            "train_seconds": train_seconds,
            "evaluation_seconds": test_metrics["evaluation_seconds"],
            "checkpoint_seconds": checkpoint_seconds,
            "cumulative_compute_seconds": cumulative_compute,
            "peak_host_memory_gib": peak_host_memory_gib(),
            "target_pout": config.target_pout,
            "target_reached": target_reached,
        }
        _append_csv(arm / "metrics.csv", metric_row, "checkpoint")
        print(
            f"    p_out={100*p_out:.2f}%  clusters={surrogate.N_src:,}  "
            f"fit={train_seconds:.2f}s  cumulative={cumulative_compute:.1f}s",
            flush=True,
        )

        if target_reached or checkpoint == config.max_rounds:
            status = "target_reached" if target_reached else "max_rounds"
            complete.write_text(
                _json(
                    {
                        "status": status,
                        "method": method,
                        "checkpoint": checkpoint,
                        "p_out": p_out,
                        "n_train": len(training),
                        "n_added_unique": len(training) - config.initial_n,
                        "cumulative_compute_seconds": cumulative_compute,
                        "cumulative_fit_points": cumulative_fit_points,
                    }
                )
            )
            return status

        acquisition_start = time.perf_counter()
        new_points, proposal_stats = acquire_all_survivors(
            config, method, surrogate, checkpoint
        )
        acquisition_seconds = time.perf_counter() - acquisition_start
        cumulative_compute += acquisition_seconds
        acquisition_row = {
            "config_hash": _config_hash(config),
            "dimension": config.dimension,
            "seed": config.seed,
            "method": method,
            "checkpoint_from": checkpoint,
            "n_train_before": len(training),
            **asdict(proposal_stats),
            "acquisition_seconds": acquisition_seconds,
            "cumulative_compute_seconds": cumulative_compute,
            "n_train_after": len(training) + len(new_points),
        }
        _append_csv(arm / "acquisitions.csv", acquisition_row, "checkpoint_from")
        print(
            f"    retained={len(new_points):,}/{proposal_stats.proposed:,} "
            f"proposals in {acquisition_seconds:.2f}s",
            flush=True,
        )
        if not len(new_points):
            complete.write_text(
                _json(
                    {
                        "status": "no_survivors",
                        "method": method,
                        "checkpoint": checkpoint,
                        "p_out": p_out,
                        "n_train": len(training),
                    }
                )
            )
            return "no_survivors"

        training = np.vstack([training, new_points])
        _save_state(
            state_path,
            training,
            checkpoint + 1,
            cumulative_compute,
            cumulative_fit_points,
        )

    raise AssertionError("unreachable")


def run(config: RaceConfig) -> int:
    config.validate()
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("The P_out race requires exactly one MPI rank.")
    root = config.output_path()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "race_config.json"
    serialized = asdict(config)
    serialized["methods"] = list(config.methods)
    if config_path.exists() and json.loads(config_path.read_text()) != serialized:
        raise RuntimeError(
            f"{config_path} has a different configuration. Choose a new output directory."
        )
    config_path.write_text(_json(serialized))
    print(
        f"Aggressive P_out race: d={config.dimension}, target={100*config.target_pout:.2f}%, "
        f"methods={len(config.methods)}, GPU={IS_CUPY_BACKEND}, output={root}",
        flush=True,
    )
    statuses = {}
    for index, method in enumerate(config.methods, start=1):
        statuses[method] = run_arm(config, method, index)
    (root / "RACE_COMPLETE.json").write_text(_json(statuses))
    return 0


def _tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=RaceConfig.output_dir)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", type=_tuple, default=RACE_METHODS)
    parser.add_argument("--target-pout", type=float, default=0.02)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--initial-n", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=20_000)
    parser.add_argument("--points-per-cluster", type=int, default=200)
    parser.add_argument("--membership-k", type=int, default=10)
    parser.add_argument("--halo-gamma", type=float, default=1.5)
    parser.add_argument(
        "--design", choices=("legacy_random", "space_filling"), default="legacy_random"
    )
    parser.add_argument(
        "--max-new-points-per-round",
        type=int,
        default=0,
        help="Optional safety cap. Zero keeps every survivor, matching the old regime.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RaceConfig(
        dimension=args.dimension,
        seed=args.seed,
        methods=args.methods,
        initial_n=args.initial_n,
        test_size=args.test_size,
        target_pout=args.target_pout,
        max_rounds=args.max_rounds,
        points_per_cluster=args.points_per_cluster,
        membership_k=args.membership_k,
        halo_gamma=args.halo_gamma,
        design=args.design,
        max_new_points_per_round=args.max_new_points_per_round,
        output_dir=args.output,
    )
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
