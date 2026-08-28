"""Checkpointed paired Rosenbrock sampling experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI

from src import IS_CUPY_BACKEND, _to_device, _to_host
from src.adaptive_clustering_utils import SurrogateMode
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad
from src.surrogate import AFINNISurrogate

from .config import StudyConfig, get_preset
from .designs import scrambled_sobol, screened_latin_hypercube
from .membership import audit_membership
from .metrics import (
    coverage_transitions,
    evaluate,
    gpu_memory_gib,
    peak_host_memory_gib,
)
from .randomness import derived_seed, make_rng
from .samplers import AcquisitionInfeasibleError, acquire


def _json_text(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=True) + "\n"


def _config_hash(config: StudyConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle))
        missing = set(row) - set(header)
        if missing:
            raise RuntimeError(f"New columns for existing {path}: {sorted(missing)}")
        fieldnames = header
    else:
        fieldnames = list(row)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _save_state(path: Path, training: np.ndarray, next_iteration: int) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        training=training,
        next_iteration=np.asarray(next_iteration, dtype=np.int64),
    )
    temporary.replace(path)


def _load_state(path: Path) -> tuple[np.ndarray, int]:
    with np.load(path) as data:
        return np.asarray(data["training"]), int(data["next_iteration"])


def _problem_values(points: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    device_points = _to_device(points)
    values = np.asarray(_to_host(rosenbrock(device_points, a=scale)))
    gradients = np.asarray(_to_host(rosenbrock_grad(device_points, a=scale)))
    return values, gradients


def _shared_dir(root: Path, dimension: int, seed: int) -> Path:
    return root / "shared" / f"dimension_{dimension:02d}" / f"seed_{seed:04d}"


def _arm_dir(root: Path, dimension: int, seed: int, method: str) -> Path:
    return root / f"dimension_{dimension:02d}" / f"seed_{seed:04d}" / method


def ensure_shared_designs(
    config: StudyConfig, root: Path, dimension: int, seed: int
) -> dict[str, np.ndarray]:
    directory = _shared_dir(root, dimension, seed)
    path = directory / "designs.npz"
    if path.exists():
        with np.load(path) as data:
            return {key: np.asarray(data[key]) for key in data.files}

    directory.mkdir(parents=True, exist_ok=True)
    initial_n = config.resolved_initial_n(dimension)
    initial = screened_latin_hypercube(
        n=initial_n,
        dimension=dimension,
        low=config.domain_low,
        high=config.domain_high,
        seed=derived_seed(seed, dimension, "initial"),
        candidates=config.initial_lhs_candidates,
    )
    calibration = scrambled_sobol(
        n=config.calibration_size,
        dimension=dimension,
        low=config.domain_low,
        high=config.domain_high,
        seed=derived_seed(seed, dimension, "calibration"),
    )
    evaluation = scrambled_sobol(
        n=config.test_size,
        dimension=dimension,
        low=config.domain_low,
        high=config.domain_high,
        seed=derived_seed(seed, dimension, "evaluation"),
    )
    calibration_truth, _ = _problem_values(calibration, config.rosenbrock_scale)
    evaluation_truth, _ = _problem_values(evaluation, config.rosenbrock_scale)
    np.savez_compressed(
        path,
        initial=initial,
        calibration=calibration,
        calibration_truth=calibration_truth,
        evaluation=evaluation,
        evaluation_truth=evaluation_truth,
    )
    return {
        "initial": initial,
        "calibration": calibration,
        "calibration_truth": calibration_truth,
        "evaluation": evaluation,
        "evaluation_truth": evaluation_truth,
    }


def _train(
    config: StudyConfig,
    training: np.ndarray,
    dimension: int,
    seed: int,
) -> tuple[AFINNISurrogate, float]:
    surrogate = AFINNISurrogate()

    def f_fn(points: Any) -> Any:
        return rosenbrock(points, a=config.rosenbrock_scale)

    def g_fn(points: Any) -> Any:
        return rosenbrock_grad(points, a=config.rosenbrock_scale)

    start = time.perf_counter()
    surrogate.train(
        X_train=_to_device(training),
        USE_V2=True,
        f_fn=f_fn,
        g_fn=g_fn,
        mode=SurrogateMode.AFFINE_COV_HESS,
        save_model=False,
        load_if_exists=False,
        TOL=config.tolerance,
        ERROR_METRIC=config.error_metric,
        INITIAL_CLUSTERING_METHOD=config.initial_clustering_method,
        NUM_INITIAL_CLUSTERS=config.resolved_num_initial_clusters(dimension),
        MIN_PTS_PER_CLUSTER=config.resolved_min_points_per_cluster(dimension),
        SEED=derived_seed(seed, dimension, "model") % (2**31 - 1),
        STORE_AFFINE_FIT=True,
        USE_WORK_STEALING=config.use_work_stealing,
        BATCH_SIZE=config.model_batch_size,
        MAX_LEVELS=config.max_levels,
        PROFILE_TIMING=False,
        LOG_MEM_STATS=False,
    )
    return surrogate, time.perf_counter() - start


def run_arm(
    config: StudyConfig,
    root: Path,
    dimension: int,
    seed: int,
    method: str,
    arm_number: int,
    arm_total: int,
) -> str:
    directory = _arm_dir(root, dimension, seed, method)
    directory.mkdir(parents=True, exist_ok=True)
    complete_path = directory / "COMPLETE.json"
    if complete_path.exists():
        print(
            f"[arm {arm_number}/{arm_total}] d={dimension} seed={seed} "
            f"method={method}: already complete"
        )
        return "skipped"

    shared = ensure_shared_designs(config, root, dimension, seed)
    state_path = directory / "state.npz"
    if state_path.exists():
        training, next_iteration = _load_state(state_path)
    else:
        training = shared["initial"].copy()
        next_iteration = 0

    previous_out: np.ndarray | None = None
    if next_iteration > 0:
        previous_mask_path = directory / f"evaluation_out_{next_iteration - 1:02d}.npy"
        if previous_mask_path.exists():
            previous_out = np.load(previous_mask_path)

    batch_size = config.resolved_batch_size(dimension)
    initial_n = config.resolved_initial_n(dimension)
    print(
        f"[arm {arm_number}/{arm_total}] d={dimension} seed={seed} method={method} "
        f"resume_checkpoint={next_iteration}"
    )

    for iteration in range(next_iteration, config.iterations + 1):
        checkpoint_start = time.perf_counter()
        print(
            f"  [checkpoint {iteration}/{config.iterations}] fitting on "
            f"{len(training):,} points",
            flush=True,
        )
        surrogate, train_seconds = _train(config, training, dimension, seed)

        test_metrics, current_out = evaluate(
            surrogate,
            shared["evaluation"],
            shared["evaluation_truth"],
            config.tolerance,
            config.membership_k,
        )
        calibration_metrics, _ = evaluate(
            surrogate,
            shared["calibration"],
            shared["calibration_truth"],
            config.tolerance,
            config.membership_k,
        )
        transition_metrics = coverage_transitions(previous_out, current_out)
        audit_points = shared["evaluation"][: config.exact_audit_points]
        audit_metrics = audit_membership(
            surrogate,
            audit_points,
            nearest_clusters=config.membership_k,
            batch_size=config.membership_batch_size,
            max_pairs=config.exact_audit_max_pairs,
        )

        cumulative_added = len(training) - initial_n
        row = {
            "preset": config.preset,
            "config_hash": _config_hash(config),
            "dimension": dimension,
            "seed": seed,
            "method": method,
            "iteration": iteration,
            "n_train": len(training),
            "n_initial": initial_n,
            "n_added_cumulative": cumulative_added,
            "normalized_added_budget": cumulative_added / initial_n,
            "n_clusters": surrogate.N_src,
            "train_seconds": train_seconds,
            "peak_host_memory_gib": peak_host_memory_gib(),
            "gpu_pool_memory_gib": gpu_memory_gib(),
            **test_metrics,
            **{
                f"calibration_{key}": value
                for key, value in calibration_metrics.items()
            },
            **transition_metrics,
            **audit_metrics,
            "checkpoint_seconds": time.perf_counter() - checkpoint_start,
        }
        _write_csv_row(directory / "metrics.csv", row)
        np.save(directory / f"evaluation_out_{iteration:02d}.npy", current_out)

        print(
            f"    p_out={100 * test_metrics['p_out']:.2f}%  "
            f"p_bad|in={100 * test_metrics['p_bad_given_in']:.2f}%  "
            f"clusters={surrogate.N_src:,}  train={train_seconds:.2f}s",
            flush=True,
        )

        if iteration == config.iterations:
            _save_state(state_path, training, config.iterations + 1)
            complete_path.write_text(
                _json_text(
                    {
                        "status": "complete",
                        "dimension": dimension,
                        "seed": seed,
                        "method": method,
                        "final_iteration": iteration,
                        "final_training_size": len(training),
                    }
                )
            )
            return "complete"

        acquisition_rng = make_rng(
            seed, dimension, method, iteration, "outer_acquisition"
        )
        acquisition_start = time.perf_counter()
        try:
            acquisition = acquire(
                method,
                surrogate,
                training,
                batch_size,
                acquisition_rng,
                config,
            )
        except AcquisitionInfeasibleError as exc:
            acquisition = exc.result
            acquisition_row = {
                "preset": config.preset,
                "config_hash": _config_hash(config),
                "dimension": dimension,
                "seed": seed,
                "method": method,
                "iteration_from": iteration,
                "n_train_before": len(training),
                **acquisition.stats.to_dict(),
                "oracle_seconds": 0.0,
                "acquisition_seconds": time.perf_counter() - acquisition_start,
                "n_train_after": len(training),
            }
            _write_csv_row(directory / "acquisitions.csv", acquisition_row)
            failure = {
                "status": "acquisition_infeasible",
                "message": str(exc),
                **acquisition_row,
            }
            (directory / "STOPPED.json").write_text(_json_text(failure))
            print(f"    stopped: {exc}", flush=True)
            return "infeasible"

        oracle_start = time.perf_counter()
        selected_values, selected_gradients = _problem_values(
            acquisition.points, config.rosenbrock_scale
        )
        oracle_seconds = time.perf_counter() - oracle_start
        np.savez_compressed(
            directory / f"selected_{iteration:02d}.npz",
            points=acquisition.points,
            values=selected_values,
            gradients=selected_gradients,
        )
        training_after = np.vstack([training, acquisition.points])
        acquisition_row = {
            "preset": config.preset,
            "config_hash": _config_hash(config),
            "dimension": dimension,
            "seed": seed,
            "method": method,
            "iteration_from": iteration,
            "n_train_before": len(training),
            **acquisition.stats.to_dict(),
            "oracle_seconds": oracle_seconds,
            "acquisition_seconds": time.perf_counter() - acquisition_start,
            "n_train_after": len(training_after),
        }
        _write_csv_row(directory / "acquisitions.csv", acquisition_row)
        _save_state(state_path, training_after, iteration + 1)
        training = training_after
        previous_out = current_out
        print(
            f"    selected={len(acquisition.points):,} from "
            f"{acquisition.stats.proposed:,} proposals in "
            f"{acquisition_row['acquisition_seconds']:.2f}s",
            flush=True,
        )

    return "complete"


def _prepare_root(config: StudyConfig) -> Path:
    root = config.output_path()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    config_data = config.to_dict()
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != config_data:
            raise RuntimeError(
                f"{config_path} contains a different configuration. Choose a new "
                "--output directory to protect the existing run."
            )
    else:
        config_path.write_text(_json_text(config_data))

    manifest = {
        "config_hash": _config_hash(config),
        "created_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "mpi_size": MPI.COMM_WORLD.Get_size(),
        "cupy_backend": IS_CUPY_BACKEND,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status": _git_value("status", "--short"),
    }
    (root / "manifest.json").write_text(_json_text(manifest))
    return root


def run_study(config: StudyConfig, limit_arms: int | None = None) -> int:
    config.validate()
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 1:
        raise RuntimeError(
            "This paired-study driver currently requires one MPI rank. Run it with "
            "direct `uv run python` in idev. Multi-rank model training can be added "
            "after the rank-invariant acquisition path is validated."
        )

    root = _prepare_root(config)
    arms = [
        (dimension, seed, method)
        for dimension in config.dimensions
        for seed in config.seeds
        for method in config.samplers
    ]
    if limit_arms is not None:
        arms = arms[:limit_arms]
    counts = {"complete": 0, "skipped": 0, "infeasible": 0}
    study_start = time.perf_counter()
    print(
        f"Rosenbrock sampling study: preset={config.preset}, arms={len(arms)}, "
        f"GPU_backend={IS_CUPY_BACKEND}, output={root}",
        flush=True,
    )
    for arm_number, (dimension, seed, method) in enumerate(arms, start=1):
        status = run_arm(
            config,
            root,
            dimension,
            seed,
            method,
            arm_number,
            len(arms),
        )
        counts[status] += 1

    summary = {
        "status": "complete" if counts["infeasible"] == 0 else "partial",
        "arms": len(arms),
        **counts,
        "elapsed_seconds": time.perf_counter() - study_start,
    }
    (root / "STUDY_COMPLETE.json").write_text(_json_text(summary))
    print(
        f"Study finished in {summary['elapsed_seconds'] / 60:.1f} min. "
        f"complete={counts['complete']} skipped={counts['skipped']} "
        f"infeasible={counts['infeasible']}",
        flush=True,
    )
    return 0 if counts["infeasible"] == 0 else 2


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", choices=("smoke", "idev", "pilot", "full"), default="idev"
    )
    parser.add_argument("--output", type=str)
    parser.add_argument("--dimensions", type=_parse_int_tuple)
    parser.add_argument("--seeds", type=_parse_int_tuple)
    parser.add_argument("--methods", type=_parse_str_tuple)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--initial-n", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--calibration-size", type=int)
    parser.add_argument("--limit-arms", type=int, help="Development-only arm limit.")
    return parser


def config_from_args(args: argparse.Namespace) -> StudyConfig:
    config = get_preset(args.preset)
    overrides = {
        "output_dir": args.output,
        "dimensions": args.dimensions,
        "seeds": args.seeds,
        "samplers": args.methods,
        "iterations": args.iterations,
        "initial_n": args.initial_n,
        "batch_size": args.batch_size,
        "test_size": args.test_size,
        "calibration_size": args.calibration_size,
    }
    return config.with_overrides(
        **{key: value for key, value in overrides.items() if value is not None}
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_study(config_from_args(args), limit_arms=args.limit_arms)


if __name__ == "__main__":
    raise SystemExit(main())
