"""Configuration and presets for the Rosenbrock sampling study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from src.heuristic_parameters import (
    get_MIN_PTS_PER_CLUSTER,
    get_N_TRAIN,
    get_NUM_INITIAL_CLUSTERS,
)

SUPPORTED_SAMPLERS = ("uniform", "uniform_gap", "shell", "gaussian")


@dataclass(frozen=True)
class StudyConfig:
    """All scientific and operational settings for one study."""

    preset: str
    dimensions: tuple[int, ...]
    seeds: tuple[int, ...]
    samplers: tuple[str, ...] = SUPPORTED_SAMPLERS
    domain_low: float = -2.0
    domain_high: float = 3.0
    tolerance: float = 0.05
    rosenbrock_scale: float = 100.0
    iterations: int = 5
    initial_n: int | None = 5_000
    batch_size: int | None = 500
    batch_fraction: float = 0.10
    test_size: int = 32_768
    calibration_size: int = 8_192
    initial_lhs_candidates: int = 4
    halo_gamma: float = 1.5
    membership_k: int = 50
    membership_batch_size: int = 50_000
    proposal_chunk_size: int = 16_384
    proposal_cap_multiplier: int = 1_000
    exact_audit_points: int = 512
    exact_audit_max_pairs: int = 10_000_000
    dedup_relative_tolerance: float = 1e-10
    max_exact_maximin_batch: int = 2_000
    max_exact_maximin_pool: int = 20_000
    num_initial_clusters: int | None = None
    min_points_per_cluster: int | None = None
    error_metric: str = "L2"
    initial_clustering_method: str = "kmeans"
    model_batch_size: int = 10_000_000
    max_levels: int = 50
    use_work_stealing: bool = True
    output_dir: str = "results/rosenbrock_sampling/idev"

    def validate(self) -> None:
        if not self.dimensions or min(self.dimensions) < 2:
            raise ValueError("Rosenbrock dimensions must be integers greater than one.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        unknown = sorted(set(self.samplers) - set(SUPPORTED_SAMPLERS))
        if unknown:
            raise ValueError(f"Unsupported samplers: {unknown}")
        if self.domain_low >= self.domain_high:
            raise ValueError("domain_low must be smaller than domain_high.")
        if self.iterations < 0:
            raise ValueError("iterations cannot be negative.")
        if self.initial_n is not None and self.initial_n < 100:
            raise ValueError("initial_n must be at least 100 when fixed.")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive when fixed.")
        if not 0.0 < self.batch_fraction <= 1.0:
            raise ValueError("batch_fraction must be in (0, 1].")
        if self.test_size < 100 or self.calibration_size < 100:
            raise ValueError("Evaluation designs must contain at least 100 points.")
        if self.halo_gamma <= 1.0:
            raise ValueError("halo_gamma must be greater than one.")
        if self.membership_k < 1:
            raise ValueError("membership_k must be positive.")
        if self.proposal_cap_multiplier < 1:
            raise ValueError("proposal_cap_multiplier must be positive.")

    def resolved_initial_n(self, dimension: int) -> int:
        return self.initial_n if self.initial_n is not None else get_N_TRAIN(dimension)

    def resolved_batch_size(self, dimension: int) -> int:
        if self.batch_size is not None:
            return self.batch_size
        return max(1, round(self.batch_fraction * self.resolved_initial_n(dimension)))

    def resolved_num_initial_clusters(self, dimension: int) -> int:
        if self.num_initial_clusters is not None:
            return self.num_initial_clusters
        if self.initial_n is None:
            return get_NUM_INITIAL_CLUSTERS(dimension)
        return 20 * dimension

    def resolved_min_points_per_cluster(self, dimension: int) -> int:
        if self.min_points_per_cluster is not None:
            return self.min_points_per_cluster
        if self.initial_n is None:
            return get_MIN_PTS_PER_CLUSTER(dimension)
        return 50 * dimension

    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dimensions"] = list(self.dimensions)
        data["seeds"] = list(self.seeds)
        data["samplers"] = list(self.samplers)
        data["resolved"] = {
            str(d): {
                "initial_n": self.resolved_initial_n(d),
                "batch_size": self.resolved_batch_size(d),
                "num_initial_clusters": self.resolved_num_initial_clusters(d),
                "min_points_per_cluster": self.resolved_min_points_per_cluster(d),
            }
            for d in self.dimensions
        }
        return data

    def with_overrides(self, **kwargs: Any) -> StudyConfig:
        updated = replace(self, **kwargs)
        updated.validate()
        return updated


def get_preset(name: str) -> StudyConfig:
    """Return a checked study preset."""

    presets = {
        "smoke": StudyConfig(
            preset="smoke",
            dimensions=(2,),
            seeds=(1001,),
            iterations=2,
            initial_n=1_000,
            batch_size=100,
            test_size=4_096,
            calibration_size=1_024,
            exact_audit_points=256,
            proposal_chunk_size=2_048,
            output_dir="results/rosenbrock_sampling/smoke",
        ),
        "idev": StudyConfig(
            preset="idev",
            dimensions=(2, 3, 5, 8, 10),
            seeds=(1001,),
            iterations=5,
            initial_n=5_000,
            batch_size=500,
            test_size=32_768,
            calibration_size=8_192,
            exact_audit_points=512,
            output_dir="results/rosenbrock_sampling/idev",
        ),
        "pilot": StudyConfig(
            preset="pilot",
            dimensions=(2, 5, 8),
            seeds=(1001, 1002, 1003, 1004, 1005),
            iterations=3,
            initial_n=None,
            batch_size=None,
            test_size=65_536,
            calibration_size=16_384,
            exact_audit_points=512,
            output_dir="results/rosenbrock_sampling/pilot",
        ),
        "full": StudyConfig(
            preset="full",
            dimensions=(2, 3, 5, 8, 10),
            seeds=tuple(range(1001, 1011)),
            iterations=10,
            initial_n=None,
            batch_size=None,
            test_size=131_072,
            calibration_size=32_768,
            exact_audit_points=1_024,
            exact_audit_max_pairs=25_000_000,
            output_dir="results/rosenbrock_sampling/full",
        ),
    }
    try:
        config = presets[name]
    except KeyError as exc:
        choices = ", ".join(sorted(presets))
        raise ValueError(f"Unknown preset {name!r}. Choose from {choices}.") from exc
    config.validate()
    return config
