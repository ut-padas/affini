"""
Settings for adaptive clustering, and the components they select.

``ClusteringConfig`` holds every algorithm setting. The pieces of the refinement
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
settings)`` (see ``src.registry``); ``None`` uses the default. A ``SurrogateMode``
is therefore only a preset pair (local model, split strategy). Components built
by name receive the matching configuration values (``min_pts_per_cluster``,
``atol``/``rtol``, ``q_reg``) when their constructor accepts them.

``components()`` resolves the components once per configuration; the data the
loop must provide (gradients at every point or at the centroids, an exact
Hessian) follows from the resolved local model and split strategy together.
Under MPI every rank builds its own configuration, so every rank must be given
the same component specs.
"""

import dataclasses
import logging
import math
import numbers
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.acceptance import ACCEPTANCE_CRITERIA, AcceptanceCriterion, Tolerance, normalize_tolerance
from src.local_models import LOCAL_MODELS, LocalModel, local_model_for_mode
from src.modes import SurrogateMode
from src.registry import ComponentSpec, component_spec
from src.sampling import SAMPLERS, Sampler
from src.splitting import SPLIT_STRATEGIES, SplitStrategy, split_strategy_for_mode
from src.trees import MAX_DEPTH
from src.trust_regions import TRUST_REGIONS, CovarianceEllipsoid, TrustRegion

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
class ClusteringConfig:
    """
    Every algorithm setting of a training run (``AFINNISurrogate.train`` or
    ``adaptive_clustering``); the functions to evaluate go in an ``Evaluator``.
    All fields are keyword-only and only a tolerance (or an ``acceptance`` that
    brings its own) is required.
    """

    # --- Surrogate and acceptance test ---
    mode: SurrogateMode = SurrogateMode.AFFINE_COV_HESS
    # Error tolerances for accepting a cluster: a scalar, one value per output
    # component, or None (that test never accepts). At least one is required unless
    # ``acceptance`` brings its own; see src.acceptance for the rule.
    atol: Tolerance = None
    rtol: Tolerance = None
    error_metric: Literal["L2", "Linf"] = "Linf"

    # --- Initial clustering ---
    initial_clustering_method: Literal["kmeans", "knndbscan", "uniform_grid"] = "kmeans"
    num_initial_clusters: int = 100

    # --- Refinement ---
    min_pts_per_cluster: int = 100
    sampling_method: str = "convex_combination"  # see src.sampling.SAMPLERS
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

    # --- Diagnostics (see src.profiling) ---
    log_mem_stats: bool = False
    profile_timing: bool = False  # per-phase time totals (synchronizes the GPU)
    profile_events: bool = False  # also record a timeline of phases, steals and loads
    profile_dir: str | Path | None = None  # also stream events to <dir>/rank_<r>.jsonl; stored as str
    save_all_gen_training_data: bool = False
    # Record the binary refinement trees (one per initial cluster; see src.trees).
    # Requires use_work_stealing=False (stolen clusters would carry their tree
    # across ranks) and max_levels < src.trees.MAX_DEPTH (Morton ID bits).
    track_trees: bool = False

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
        """Validate settings early (including track_trees' limits on work stealing and depth), normalize the tolerances and profile_dir, warn about save_all_gen_training_data, and drop ``store_hess`` where no model can use it."""
        self.mode = SurrogateMode(self.mode)
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
        if self.track_trees and self.use_work_stealing:
            raise ValueError(
                "track_trees requires use_work_stealing=False (the default is True)."
            )
        if self.track_trees and self.max_levels >= MAX_DEPTH:
            raise ValueError(
                f"track_trees requires max_levels < {MAX_DEPTH}: children of the deepest "
                f"clusters need a Morton ID; got max_levels={self.max_levels}."
            )
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
        out["components"] = self.components().describe()
        return out

    def generated_data_name(self, d: int) -> str:
        """File prefix for ``save_all_gen_training_data``; a per-component tolerance is joined with '-'."""
        tags = [
            "none" if t is None else "-".join(f"{v:g}" for v in (t if isinstance(t, tuple) else (t,)))
            for t in (self.atol, self.rtol)
        ]
        return f"train_data_atol_{tags[0]}_rtol_{tags[1]}_d{d}"
