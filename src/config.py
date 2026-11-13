"""
Settings for adaptive clustering, and the components they select.

``TrainingConfig`` holds every algorithm setting. The pieces of the refinement
loop that can be swapped are *components*:

=================  ================================  ===================================
component          base class / registry             default
=================  ================================  ===================================
local_model        ``LocalModel`` / LOCAL_MODELS      from ``mode`` (and ``store_hess``)
split_strategy     ``SplitStrategy`` / SPLIT_STRATEGIES  from ``mode``
sampler            ``Sampler`` / SAMPLERS             ``sampling_method``
acceptance         ``AcceptanceCriterion`` / ACCEPTANCE_CRITERIA  ``error_metric``, atol, rtol
trust_region       ``TrustRegion`` / TRUST_REGIONS    ``CovarianceEllipsoid(q_reg)``
=================  ================================  ===================================

Each component field accepts an instance, a registered name, or ``(name,
settings)`` (see ``src.components.registry``); ``None`` uses the default. A ``SurrogateMode``
is therefore only a preset pair (local model, split strategy). Components built
by name receive the matching configuration values (``min_pts_per_cluster``,
``atol``/``rtol``, ``q_reg``) when their constructor accepts them.

``components()`` resolves the components once per configuration; the data the
loop must provide (gradients at every point or at the centroids, an exact
Hessian) follows from the resolved local model and split strategy together.
Under MPI every rank builds its own configuration, so every rank must be given
the same component specs.

``algorithm`` selects one of two algorithm plans over the same refinement engine
(``src.engine``): ``"afinni"`` (the default, AFINNI's adaptive clustering, bit-identical to
earlier versions) or ``"art"``, ART's adaptive regression tree (``src.art``; settings in
``ARTSettings``), which needs an explicit ``domain``.
"""

import dataclasses
import logging
import math
import numbers
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.components.acceptance import (
    ACCEPTANCE_CRITERIA,
    AcceptanceCriterion,
    Tolerance,
    normalize_tolerance,
)
from src.components.local_models import LOCAL_MODELS, LocalModel, local_model_for_mode
from src.components.modes import SurrogateMode
from src.components.registry import ComponentSpec, component_spec
from src.components.sampling import SAMPLERS, Sampler
from src.components.splitting import (
    SPLIT_STRATEGIES,
    SplitStrategy,
    split_strategy_for_mode,
)
from src.components.trust_regions import TRUST_REGIONS, CovarianceEllipsoid, TrustRegion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Components:
    """The resolved components of one configuration."""

    local_model: LocalModel
    split_strategy: SplitStrategy
    sampler: Sampler
    acceptance: AcceptanceCriterion
    trust_region: TrustRegion

    def describe(self) -> dict[str, dict[str, Any]]:
        """``{component: {"name": ..., "settings": ...}}``, e.g. for saving with a model."""
        return {f.name: component_spec(getattr(self, f.name)) for f in dataclasses.fields(self)}


@dataclass(kw_only=True)
class ARTSettings:
    """
    Settings of ``algorithm="art"`` (see ``src.art``).

    Every node is topped up to ``ceil(sample_multiplier * p_eff)`` training points
    before it is fitted, with ``p_eff = d + 1`` (ART's ``n_node``; ART uses 50). New
    points are uniform in a box or grid root and hit-and-run endpoints (``walk_steps``
    steps from a random inherited point) elsewhere. A split must leave
    ``p_eff + min_side_margin`` points on each side. The experiment variants of the
    design (B1-B4) are settings of this one plan: B1 = ``sampling="convex_combination"``
    with ``roots="voronoi"``, B2 = ``split_search="gap"``, B3 = ``"bins"``, B4 = ``"soft"``
    (the default: on ART's benchmarks it was as accurate as ART or better on most problems,
    with similar or fewer leaves; see Section 42.3 of AFINNI_ART_DESIGN.md).
    """

    # "grid": num_initial_clusters cells over the domain; "voronoi": the Voronoi cells of
    # the initial clustering's centres (initial_clustering_method), clipped to the domain.
    roots: Literal["box", "grid", "voronoi"] = "box"
    # "hit_and_run": top every node up inside its polytope before fitting (ART's budget).
    # "convex_combination": the B1 experiment, AFINNI's convex-combination augmentation
    # of rejected nodes.
    sampling: Literal["hit_and_run", "convex_combination"] = "hit_and_run"
    sample_multiplier: float = 50.0
    # Fresh points per node on which the deployed model is checked before acceptance; a
    # diagnostic, not a certificate (the global claim is AFINNISurrogate.audit). 0: none.
    validation_samples: int = 0
    min_side_margin: int = 0
    walk_steps: int = 20
    # Chains whose endpoints still drift between the halfway and last step (standardized
    # mean difference > mixing_z) are continued, doubling, up to max_walk_steps (default
    # 4 * walk_steps); nodes that still drift get validation_status "insufficient_sampling".
    max_walk_steps: int | None = None
    mixing_z: float = 4.0
    # How a rejected node's hyperplane is chosen: "gap" (the split strategy's direction,
    # cut near the median), "bins" (B3: n_bins - 1 cuts along n_directions directions,
    # the n_screened best re-checked directly) or "soft" (B4: "bins" then the consistent
    # soft-oblique optimizer for soft_iters steps, kept only if its hard fit is better).
    split_search: Literal["gap", "bins", "soft"] = "soft"
    n_bins: int = 16
    n_directions: int = 6
    n_screened: int = 3
    # 40: on ART's benchmarks 20 stopped nearly every node at the limit, and 40 gave fewer
    # leaves or lower error on most of them for about 35% more build time (ART allows 200).
    soft_iters: int = 40
    soft_temperature: float = 0.1  # gate width as a fraction of the node's score spread
    soft_weight_floor: float = 1e-8  # delta in W = delta + (1 - 2 delta) sigma
    # Store the trust-region shape Q per leaf (needed only by predictor="knn"; d^2 per leaf).
    store_Q: bool = True  # noqa: N815  (named after the leaf column "Q")
    # Work-stealing thresholds in queued points; None: a few nodes' worth (src.art).
    steal_min_load: int | None = None
    steal_size: int | None = None

    def __post_init__(self) -> None:
        """Validate the settings."""
        for name, choices in (
            ("roots", ("box", "grid", "voronoi")),
            ("sampling", ("hit_and_run", "convex_combination")),
            ("split_search", ("gap", "bins", "soft")),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"ARTSettings.{name} must be one of {choices}; got {getattr(self, name)!r}.")
        if self.sampling == "convex_combination" and self.validation_samples:
            raise ValueError("validation_samples needs sampling='hit_and_run' (fresh points inside each node).")
        for name, low in (("n_bins", 2), ("n_directions", 1), ("n_screened", 1), ("soft_iters", 0)):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < low:
                raise ValueError(f"{name} must be an integer >= {low}; got {value!r}.")
        if not (self.soft_temperature > 0) or not (0 <= self.soft_weight_floor < 0.5):
            raise ValueError("soft_temperature must be > 0 and soft_weight_floor in [0, 0.5).")
        if not (isinstance(self.sample_multiplier, numbers.Real) and self.sample_multiplier > 0):
            raise ValueError(f"sample_multiplier must be > 0; got {self.sample_multiplier!r}.")
        for name, low in (("validation_samples", 0), ("min_side_margin", 0), ("walk_steps", 1)):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < low:
                raise ValueError(f"{name} must be an integer >= {low}; got {value!r}.")
        if self.max_walk_steps is not None and (
            not isinstance(self.max_walk_steps, numbers.Integral) or self.max_walk_steps < self.walk_steps
        ):
            raise ValueError(f"max_walk_steps must be None or an integer >= walk_steps; got {self.max_walk_steps!r}.")
        if not self.mixing_z > 0:
            raise ValueError(f"mixing_z must be > 0; got {self.mixing_z!r}.")
        for name in ("steal_min_load", "steal_size"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, numbers.Integral) or value < 1):
                raise ValueError(f"{name} must be None or an integer >= 1; got {value!r}.")


@dataclass(kw_only=True)
class TrainingConfig:
    """
    Every algorithm setting of a training run (``AFINNISurrogate.train`` or
    ``refine``); the functions to evaluate go in an ``Evaluator``.
    All fields are keyword-only and only a tolerance (or an ``acceptance`` that
    brings its own) is required.
    """

    # --- Surrogate and acceptance test ---
    mode: SurrogateMode = SurrogateMode.AFFINE_COV_HESS
    # Error tolerances for accepting a cluster: a scalar, one value per output
    # component, or None (that test never accepts). At least one is required unless
    # ``acceptance`` brings its own; see src.components.acceptance for the rule.
    atol: Tolerance = None
    rtol: Tolerance = None
    error_metric: Literal["L2", "Linf", "pointwise"] = "Linf"

    # --- Initial clustering ---
    initial_clustering_method: Literal["kmeans", "knndbscan", "uniform_grid"] = "kmeans"
    num_initial_clusters: int = 100

    # --- Refinement ---
    min_pts_per_cluster: int = 100
    sampling_method: str = "convex_combination"  # see src.components.sampling.SAMPLERS
    q_reg: float = 1e-6  # regularization added when computing the trust-region shape Q
    # Maximum number of refinement levels; clusters still unresolved after it leave
    # their region without leaves and the run reports converged=False.
    max_levels: int = 50
    batch_size: int = 10_000_000  # max points moved to the device per batch
    seed: int = 0
    max_clusters_est: int = 1_000_000  # initial capacity of the output buffers (all ranks)
    use_work_stealing: bool = True

    # --- Outputs ---
    # Store a Hessian per accepted cluster and use it in a second-order Taylor
    # prediction. Only TAYLOR_LSQ_HESS estimates one; elsewhere it is ignored.
    store_hess: bool = False
    # Deprecated and ignored: what is stored is decided by the local model (affine
    # modes always store A_fit/b_fit). Kept so existing callers still work.
    store_affine_fits: bool = False

    # --- Algorithm plan (see the module docstring) ---
    algorithm: Literal["afinni", "art"] = "afinni"
    # ART only: ((lower_1, ..., lower_d), (upper_1, ..., upper_d)), or
    # "data_bbox" to use the bounding box of the training points (recorded as such).
    domain: Any = None
    art: ARTSettings | dict | None = None  # None: ARTSettings() when algorithm='art'

    # --- Diagnostics (see src.infra.profiling) ---
    log_mem_stats: bool = False
    profile_timing: bool = False  # per-phase time totals (synchronizes the GPU)
    profile_events: bool = False  # also record a timeline of phases, steals and loads
    profile_dir: str | Path | None = None  # also stream events to <dir>/rank_<r>.jsonl; stored as str
    save_all_gen_training_data: bool = False

    # --- Components (None: the default for mode / the settings above) ---
    local_model: ComponentSpec = None
    split_strategy: ComponentSpec = None
    sampler: ComponentSpec = None
    acceptance: ComponentSpec = None
    trust_region: ComponentSpec = None

    _components: Components | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate settings early, normalize the tolerances and profile_dir, warn about save_all_gen_training_data, and drop ``store_hess`` where no model can use it."""
        self.mode = SurrogateMode(self.mode)
        if self.algorithm not in ("afinni", "art"):
            raise ValueError(f"algorithm must be 'afinni' or 'art'; got {self.algorithm!r}.")
        if isinstance(self.art, dict):
            self.art = ARTSettings(**self.art)
        if self.algorithm == "art":
            self.art = self.art or ARTSettings()
            self.domain = _normalize_domain(self.domain)
        elif self.domain is not None or self.art is not None:
            raise ValueError("domain and art only apply to algorithm='art'.")
        if self.profile_dir is not None:
            self.profile_dir = str(self.profile_dir)
        # Normalized to a float or a tuple so to_dict() stays serializable.
        self.atol = normalize_tolerance("atol", self.atol)
        self.rtol = normalize_tolerance("rtol", self.rtol)
        # An instance or (name, settings) acceptance carries its own tolerances.
        if self.atol is None and self.rtol is None and (self.acceptance is None or isinstance(self.acceptance, str)):
            raise ValueError("Give atol, rtol or both: an acceptance test needs a tolerance.")
        if not (isinstance(self.q_reg, numbers.Real) and math.isfinite(self.q_reg) and self.q_reg >= 0):
            raise ValueError(f"q_reg must be a finite number >= 0; got {self.q_reg!r}.")
        for name, low in (
            ("num_initial_clusters", 1),
            ("min_pts_per_cluster", 1),
            ("batch_size", 1),
            ("max_levels", 0),
            ("max_clusters_est", 1),
        ):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < low:
                raise ValueError(f"{name} must be an integer >= {low}; got {value!r}.")
        if self.sampler is None and self.sampling_method not in SAMPLERS:
            raise ValueError(
                f"Unknown sampling method {self.sampling_method!r}; "
                f"choose one of {sorted(SAMPLERS)}."
            )
        if self.acceptance is None and self.error_metric not in ACCEPTANCE_CRITERIA:
            raise ValueError(
                f"Unknown error metric {self.error_metric!r}; "
                f"choose one of {sorted(ACCEPTANCE_CRITERIA)}."
            )
        if self.store_hess and self.local_model is not None:
            self.store_hess = False
            logger.warning(
                "store_hess is ignored when local_model is given; configure the model "
                "instead (e.g. TaylorModel(second_order=True))."
            )
        elif self.store_hess and self.mode != SurrogateMode.TAYLOR_LSQ_HESS:
            self.store_hess = False
            logger.warning(
                f"store_hess is only supported by {SurrogateMode.TAYLOR_LSQ_HESS.value}; "
                f"ignoring it for mode {self.mode.value}."
            )
        if self.save_all_gen_training_data:
            warnings.warn(
                "save_all_gen_training_data writes one file per batch into ./saved_training_data/, "
                "then rank 0 stacks the points (X only, no f values) into ./<name>.npy and deletes "
                "the whole folder, including files of any other run using it. Run one such job per "
                "working directory, on a filesystem every MPI rank shares; runs with the same "
                "tolerances and dimension overwrite each other's output.",
                UserWarning,
                stacklevel=3,
            )

    def components(self) -> Components:
        """The local model, split strategy, sampler, acceptance criterion and trust region."""
        if self._components is None:
            self._components = Components(
                local_model=(
                    local_model_for_mode(self.mode, second_order=self.store_hess)
                    if self.local_model is None
                    else LOCAL_MODELS.create(self.local_model)
                ),
                split_strategy=(
                    split_strategy_for_mode(self.mode, self.min_pts_per_cluster)
                    if self.split_strategy is None
                    else SPLIT_STRATEGIES.create(
                        self.split_strategy, min_pts_per_cluster=self.min_pts_per_cluster
                    )
                ),
                sampler=SAMPLERS.create(
                    self.sampling_method if self.sampler is None else self.sampler
                ),
                acceptance=ACCEPTANCE_CRITERIA.create(
                    self.error_metric if self.acceptance is None else self.acceptance,
                    atol=self.atol,
                    rtol=self.rtol,
                ),
                trust_region=(
                    CovarianceEllipsoid(self.q_reg)
                    if self.trust_region is None
                    else TRUST_REGIONS.create(self.trust_region, q_reg=self.q_reg)
                ),
            )
        return self._components

    # Which data the loop must provide follows from the local model and the split
    # strategy together (their needs_* flags), so a new mode or component only has
    # to declare those.

    @property
    def requires_f_centroids(self) -> bool:
        """True if some component reads f at the centroids (AFINNI always evaluates it)."""
        c = self.components()
        return bool(c.local_model.needs_centroid_value or c.split_strategy.needs_centroid_value)

    @property
    def requires_grad_everywhere(self) -> bool:
        """True if gradients must be evaluated (and carried) at every point."""
        c = self.components()
        return bool(c.split_strategy.needs_point_grad or c.local_model.needs_point_grad)

    @property
    def requires_grad_centroids(self) -> bool:
        """True if gradients must be evaluated at the cluster centroids."""
        c = self.components()
        return bool(c.local_model.needs_centroid_grad or c.split_strategy.needs_centroid_grad)

    @property
    def requires_no_grads(self) -> bool:
        """True if the configuration needs no gradients at all."""
        return not (self.requires_grad_everywhere or self.requires_grad_centroids)

    @property
    def requires_hessian_fn(self) -> bool:
        """True if an exact Hessian function must be provided."""
        return bool(self.components().split_strategy.needs_hessian_fn)

    def to_dict(self) -> dict[str, Any]:
        """The settings and resolved components as plain, serializable values."""
        out = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.init and f.name not in Components.__dataclass_fields__
        }
        out["mode"] = self.mode.value
        out["art"] = dataclasses.asdict(self.art) if self.art is not None else None
        out["components"] = self.components().describe()
        return out

    def generated_data_name(self, d: int) -> str:
        """File prefix for ``save_all_gen_training_data``; a per-component tolerance is joined with '-'."""
        tags = [
            "none" if t is None else "-".join(f"{v:g}" for v in (t if isinstance(t, tuple) else (t,)))
            for t in (self.atol, self.rtol)
        ]
        return f"train_data_atol_{tags[0]}_rtol_{tags[1]}_d{d}"


def _normalize_domain(domain: Any) -> str | tuple[tuple[float, ...], tuple[float, ...]]:
    """``"data_bbox"`` or validated ``(lower, upper)`` float tuples with ``lower < upper``."""
    if isinstance(domain, str):
        if domain != "data_bbox":
            raise ValueError(f"domain must be (lower, upper) or 'data_bbox'; got {domain!r}.")
        return domain
    if domain is None:
        raise ValueError(
            "algorithm='art' needs domain=(lower, upper) bounds, or domain='data_bbox' to "
            "use the training points' bounding box: the domain is what routing covers and "
            "what AFINNISurrogate.audit samples, so it is never inferred silently."
        )
    import numpy as np

    bounds = np.asarray(domain, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[0] != 2 or not np.all(np.isfinite(bounds)) or not np.all(bounds[0] < bounds[1]):
        raise ValueError(f"domain must be finite (lower, upper) vectors with lower < upper; got {domain!r}.")
    return tuple(map(float, bounds[0])), tuple(map(float, bounds[1]))


# --- Size heuristics (used by the examples and the launcher) ---------------------


def get_N_TRAIN(dim: int, *, base: float = 1.3, factor: int = 50_000) -> int:
    """
    Heuristic to determine N_TRAIN based on dimension and tolerance.
        factor * (base)^dim
    """
    N_TRAIN = int(factor * (base**dim))
    return N_TRAIN


def get_NUM_INITIAL_CLUSTERS(dim: int, *, base: float = 1.3, factor: int = 25) -> int:
    """
    Heuristic to determine N_INIT_CLUSTERS based on dimension.
        factor * (base)^dim
    """
    N_INIT_CLUSTERS = int(factor * (base**dim))
    return N_INIT_CLUSTERS


def get_MIN_PTS_PER_CLUSTER(dim: int, *, base: float = 1.3, factor: int = 100) -> int:
    """
    Heuristic to determine N_MIN_PER_CLUSTER based on dimension.
        factor * (base)^dim
    """
    N_MIN_PER_CLUSTER = int(factor * (base**dim))
    return N_MIN_PER_CLUSTER
