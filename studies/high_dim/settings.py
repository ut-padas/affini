"""
The method variants of the high-dimensional study and the ``TrainingConfig`` each run uses.

Settings that the library's defaults do not scale with d are set here, and every choice is
recorded in the run's result (``config`` there is ``TrainingConfig.to_dict()``):

- ``min_pts_per_cluster = max(100, 3 (d + 1))``: the Hessian estimate's level set takes
  ``min_pts_per_cluster // 3`` points, and fewer than d + 1 make its scatter matrix singular
  (TODO.md, "Algorithms"). ``--library-min-pts`` keeps the default 100 to measure that effect.
- ``max_clusters_est = 20_000``: the leaf buffer is allocated up front for this many leaves,
  and at the default 1,000,000 its (d, d) trust-matrix column alone reserves 8 d^2 MB, which
  the kernel refuses beyond RAM + swap (d of about 85 on a 32 GB machine). It grows on demand.
- ART: one box root, ``store_trust_matrix=False`` (routing never reads it, and it costs
  8 d^2 bytes per leaf), no validation points.
- ``max_levels = 40`` for both: deep enough for any tree reachable here, and ART's batch
  charge grows with it (2 (max_levels + 1) (d + 1) doubles per point).
"""

import math
from dataclasses import dataclass, field

from src.components.modes import SurrogateMode
from src.config import ARTSettings, TrainingConfig
from studies.high_dim.problems import domain


@dataclass(frozen=True)
class Variant:
    """One method configuration: the algorithm, its leaf model and ART settings beyond the study's defaults."""

    name: str
    algorithm: str  # "afinni" or "art"
    leaf: str  # "affine" or "quadratic"
    art: dict = field(default_factory=dict)
    note: str = ""


VARIANTS = {
    "afinni": Variant("afinni", "afinni", "affine", note="AFFINE_COV_HESS, kNN + trust regions"),
    "art_affine": Variant("art_affine", "art", "affine", note="soft split search, m = 50"),
    "art_quad": Variant("art_quad", "art", "quadratic", note="quadratic leaves, soft split search, m = 50"),
    # Exploratory: fewer points per node, the change high d most needs (n = m p).
    "art_affine_m10": Variant("art_affine_m10", "art", "affine", {"sample_multiplier": 10.0}, "m = 10"),
    "art_quad_m4": Variant("art_quad_m4", "art", "quadratic", {"sample_multiplier": 4.0, "split_search": "bins"},
                           "quadratic leaves, m = 4, bins search"),  # fmt: skip
}
DEFAULT_VARIANTS = ("afinni", "art_affine", "art_quad")

MAX_LEVELS = 40
MAX_CLUSTERS_EST = 20_000
AFINNI_INITIAL_CLUSTERS = 64


def min_pts(d: int, library_default: bool = False) -> int:
    return 100 if library_default else max(100, 3 * (d + 1))


def n_coefficients(leaf: str, d: int) -> int:
    return d + 1 if leaf == "affine" else (d + 1) * (d + 2) // 2


def art_settings(variant: Variant) -> ARTSettings:
    return ARTSettings(**{"roots": "box", "store_trust_matrix": False, "validation_samples": 0, **variant.art})


def n_initial_points(variant: Variant, d: int, library_min_pts: bool = False) -> int:
    """
    Training points X_0. AFINNI: two clusters' worth per initial cluster (it starts from a given
    sample); ART: one node's budget, so the root is fitted without a top-up.
    """
    if variant.algorithm == "afinni":
        return AFINNI_INITIAL_CLUSTERS * 2 * min_pts(d, library_min_pts)
    s = art_settings(variant)
    return math.ceil(s.sample_multiplier * n_coefficients(variant.leaf, d))


def build_config(variant: Variant, d: int, atol: float, *, seed: int = 0, library_min_pts: bool = False,
                 profile: bool = True, max_levels: int = MAX_LEVELS) -> TrainingConfig:  # fmt: skip
    """The study's ``TrainingConfig`` for one run (absolute tolerance only: ``rtol`` is None)."""
    common = {
        "atol": atol, "rtol": None, "error_metric": "Linf", "mode": SurrogateMode.AFFINE_COV_HESS,
        "min_pts_per_cluster": min_pts(d, library_min_pts), "max_levels": max_levels, "seed": seed,
        "max_clusters_est": MAX_CLUSTERS_EST, "use_work_stealing": False, "profile_timing": profile,
    }  # fmt: skip
    if variant.algorithm == "afinni":
        return TrainingConfig(num_initial_clusters=AFINNI_INITIAL_CLUSTERS, **common)
    lower, upper = domain(d)
    return TrainingConfig(
        algorithm="art", domain=(tuple(lower), tuple(upper)), art=art_settings(variant), num_initial_clusters=1,
        local_model="quadratic_lsq" if variant.leaf == "quadratic" else None, **common,
    )  # fmt: skip
