"""
The public surrogate: ``AFINNISurrogate`` trains with ``adaptive_clustering``, stores
the accepted leaves on every MPI rank, predicts with a two-stage (Euclidean, then
trust-region) neighbour search (``predictor="knn"``) or, for a region-mode model, by
routing each query down the explicit tree (``predictor="route"``, its default), and
saves/loads a versioned model file. ``audit`` checks a region-mode model on fresh
uniform points of its domain.
``compute_error_metrics`` scores predictions inside and outside the trust regions.
"""

import functools
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, ParamSpec, TypeVar, Union

import numpy as np

from src import (
    IS_CUPY_BACKEND,
    Array,
    _get_mpi_rank,
    _to_device,
    _to_host,
    xp,
)
from src.acceptance import Tolerance, normalize_tolerance, tolerance_vector
from src.adaptive_clustering import _adaptive_clustering, run_collective
from src.backend import MPI  # real mpi4py.MPI, or a single-rank stand-in without MPI
from src.config import ClusteringConfig
from src.evaluation import Evaluator
from src.local_models import LOCAL_MODELS, LocalModel
from src.modes import SurrogateMode
from src.region import TERMINAL_REASONS, VALIDATION_STATUSES, RoutingTable
from src.registry import component_spec

logger = logging.getLogger(__name__)

# Version of the file format written by AFINNISurrogate.save_model(); load_model()
# reads every version up to it.
# Version 2 adds the optional region tree ("route:" arrays, "leaf:" statuses).
MODEL_FORMAT_VERSION = 2
DEFAULT_MODEL_FILENAME = "afinni_surrogate.npz"
_PARAM_PREFIX = "param:"
_ROUTE_PREFIX = "route:"
_LEAF_PREFIX = "leaf:"
# Per-leaf region-mode columns (see src.region), kept apart from the model's parameters.
REGION_LEAF_COLUMNS = ("root", "path_code", "terminal_reason", "validation_status")


def _model_path(path: Union[str, Path]) -> Path:
    """``path`` as an ``.npz`` file path; a directory gets the default file name."""
    path = Path(path)
    if path.is_dir():
        return path / DEFAULT_MODEL_FILENAME
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def _json_default(obj: Any) -> Any:
    """Make NumPy scalars/arrays and enums JSON serializable."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "value"):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _collective(what: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Run a public collective method under ``run_collective`` (MPI: a failure on any rank
    aborts the job). ``functools.wraps`` and ``ParamSpec`` keep the method's signature,
    annotations and docstring for ``help()``, ``inspect`` and static type checkers.
    """

    def decorate(method: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(method)
        def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return run_collective(lambda: method(*args, **kwargs), what)

        return guarded

    return decorate


def _make_knn_index(centroids: Array):
    """The exact kNN index of ``src.knn`` for the active backend."""
    if IS_CUPY_BACKEND:
        from src.knn import GpuExactKNN

        return GpuExactKNN(_to_device(centroids))
    from src.knn import CpuExactKNN

    return CpuExactKNN(_to_host(centroids))


class AFINNISurrogate:
    """
    Surrogate using Adaptive Clustering + kNN with Ellipsoidal Distance on GPU.
    Supports MPI-parallelized training and inference.
    """

    def __init__(self):
        self.trained = False
        self.mode: Optional[SurrogateMode] = None
        self.local_model: Optional[LocalModel] = None  # the function on each leaf

        # Storage for source points (Reference data)
        self.N_src = 0
        self.X_src = None
        self.f_src = None
        self.Qs_src = None
        # The local model's own per-leaf parameters, keyed by the columns it declares
        # (e.g. "A_fit"/"b_fit", or "g" and optionally "H").
        self.leaf_params: dict[str, Array] = {}
        # Per-leaf training diagnostics: the accepted absolute and relative errors per
        # output component, (N_src, fd) (see src.acceptance), and the number of points.
        self.E_cluster_abs_src = None
        self.E_cluster_rel_src = None
        self.counts_src = None
        # The effective settings and components of the last training run.
        self.config: dict[str, Any] = {}
        # False if refinement stopped at max_levels with clusters still unresolved;
        # their region has no leaves (see num_unresolved_clusters / _pts).
        self.converged: Optional[bool] = None
        self.num_unresolved_clusters = 0
        self.num_unresolved_pts = 0
        self.profiling_stats = {}

        self.knn_index = None
        self.search_stats: dict = {}  # last predict's kNN totals on this rank
        # Region mode: the explicit tree, and per-leaf host arrays named in
        # REGION_LEAF_COLUMNS (terminal_reason / validation_status index TERMINAL_REASONS /
        # VALIDATION_STATUSES). None / {} for AFINNI-mode models.
        self.routing: RoutingTable | None = None
        self.region_leaves: dict[str, np.ndarray] = {}

    def _leaf_params(self) -> dict:
        """The per-cluster arrays ``local_model.evaluate`` reads (see src.local_models)."""
        return {"centroids": self.X_src, "f": self.f_src, **self.leaf_params}

    def _leaf_param(self, name: str) -> Array | None:
        return self.leaf_params.get(name)

    def _set_leaf_param(self, name: str, value: Array | None) -> None:
        if value is None:
            self.leaf_params.pop(name, None)
        else:
            self.leaf_params[name] = value

    # Named views of leaf_params for the current models (None when absent).
    J_src = property(
        lambda self: self._leaf_param("g"),
        lambda self, v: self._set_leaf_param("g", v),
        doc="(K, fd, d) gradients at the centroids (Taylor models).",
    )
    Hs_src = property(
        lambda self: self._leaf_param("H"),
        lambda self, v: self._set_leaf_param("H", v),
        doc="(K, fd, d, d) second-order Taylor Hessians, one per output component.",
    )
    A_fits_src = property(
        lambda self: self._leaf_param("A_fit"),
        lambda self, v: self._set_leaf_param("A_fit", v),
        doc="(K, fd, d) affine least-squares slopes.",
    )
    b_fits_src = property(
        lambda self: self._leaf_param("b_fit"),
        lambda self, v: self._set_leaf_param("b_fit", v),
        doc="(K, fd) affine least-squares intercepts.",
    )

    @_collective("train()")
    def train(
        self,
        X_train: Array,
        config: ClusteringConfig,
        evaluator: Evaluator,
        *,
        Y_train: Optional[Array] = None,
        initial_labels: Optional[Array] = None,
    ) -> None:
        """
        Train the surrogate collectively on every MPI rank, replacing any previous model.

        Every setting, including the components and the tolerances, comes from
        ``config``; every rank must pass an equivalent one. Only rank 0's ``X_train``,
        ``Y_train`` (a precomputed f(X_train)) and ``initial_labels`` are read.
        ``initial_labels`` (e.g. from ``src.adaptive_clustering.initial_clustering``,
        computed once for several trainings on the same points) skip the initial
        clustering. ``evaluator`` may be
        reused: its ``n_evals`` keeps counting, but ``profiling_stats`` reports only
        this run's evaluations.

        If refinement stops at ``config.max_levels`` before every cluster is accepted,
        training still succeeds but ``converged`` is False and a warning is issued;
        if no cluster at all was accepted, a ``RuntimeError`` is raised and the
        previous model is kept.
        """
        comm = MPI.COMM_WORLD
        rank = _get_mpi_rank()
        t0 = time.perf_counter()

        leaf_model = config.components().local_model
        # Unguarded engine body: the failure policy is owned by train() (see run_collective).
        results = _adaptive_clustering(X_train, config, evaluator, Y=Y_train, initial_labels=initial_labels)

        # 2. Broadcast trained structures to all ranks
        bcast_data = None
        if rank == 0:
            assert results is not None
            # Push to host memory to safely pickle/broadcast over MPI
            bcast_data = {
                k: _to_host(v) if isinstance(v, xp.ndarray) else v
                for k, v in results.items()
            }

        bcast_data = comm.bcast(bcast_data, root=0)

        n_leaves = int(bcast_data["centroids"].shape[0])
        if n_leaves == 0:
            raise RuntimeError(
                f"Training accepted no clusters: refinement stopped at max_levels="
                f"{config.max_levels} with {bcast_data['num_unresolved_clusters']:,} unresolved "
                f"clusters ({bcast_data['num_unresolved_pts']:,} points). Increase "
                f"max_levels or loosen atol/rtol."
            )

        d, fd = bcast_data["centroids"].shape[1], bcast_data["f"].shape[1]
        self._install(
            {
                "mode": config.mode,
                "local_model": leaf_model,
                "config": {**config.to_dict(),
                           # Settings the run changed at runtime (GPU batch cap).
                           "effective": bcast_data["profiling_stats"].get("effective_config", {})},
                "X_src": bcast_data["centroids"],
                "f_src": bcast_data["f"],
                "Qs_src": bcast_data.get("Q"),  # absent for region models with store_Q=False
                "E_cluster_abs_src": bcast_data["E_cluster_abs"],
                "E_cluster_rel_src": bcast_data["E_cluster_rel"],
                "counts_src": bcast_data["counts"],
                "profiling_stats": bcast_data["profiling_stats"],
                "converged": bool(bcast_data["converged"]),
                "num_unresolved_clusters": int(bcast_data["num_unresolved_clusters"]),
                "num_unresolved_pts": int(bcast_data["num_unresolved_pts"]),
                "leaf_params": {name: bcast_data[name] for name in leaf_model.columns(d, fd)},
                "routing": RoutingTable.from_arrays(bcast_data["routing"]) if "routing" in bcast_data else None,
                "region_leaves": {k: bcast_data[k] for k in REGION_LEAF_COLUMNS if k in bcast_data},
            },
            source="training",
        )  # fmt: skip

        if rank == 0:
            logger.info(f"Training Time: {time.perf_counter() - t0:.4f}s")
            logger.info(f"Trained Clusters: {self.N_src:,}")
            if not self.converged:
                where = (
                    "they keep their fitted model as a leaf (see region_leaves / leaf_status_counts())"
                    if self.routing is not None else "their region has no leaves"
                )
                warnings.warn(
                    f"Training stopped at max_levels={config.max_levels} with "
                    f"{self.num_unresolved_clusters:,} clusters "
                    f"({self.num_unresolved_pts:,} points) still above tolerance; "
                    f"{where}. See AFINNISurrogate.converged.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if self.routing is not None:
                n_weak = self.leaf_status_counts()["validation_status"]["insufficient_sampling"]
                if n_weak:
                    # ``converged`` only says no node is still above tolerance; these leaves
                    # passed on samples that may not cover their region.
                    warnings.warn(
                        f"{n_weak:,} region leaves have validation_status='insufficient_sampling' "
                        "(their samples may not cover the region); see route_details() and "
                        "leaf_status_counts().",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    def save_model(self, filepath: Union[str, Path]) -> Path:
        """
        Save the trained model to an ``.npz`` file and return its path.

        A directory gets the default file name (``afinni_surrogate.npz``); another
        suffix is replaced by ``.npz``. The file stores a format version, the leaf
        arrays, every local-model parameter under its column name, the local
        model's registered name and settings, the effective training
        configuration and the profiling statistics, all without pickling.

        Under MPI only rank 0 writes (every rank holds the same model), so it may be
        called on every rank or on rank 0 alone; the other ranks return the path
        without waiting for the file.
        """
        if not self.trained:
            raise RuntimeError("Cannot save an untrained model.")

        filepath = _model_path(filepath)
        if _get_mpi_rank() != 0:
            return filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "format_version": MODEL_FORMAT_VERSION,
            "mode": self.mode.value if self.mode else None,
            "local_model": component_spec(self.local_model),
            "N_src": int(self.N_src),
            "d": int(self.X_src.shape[1]),
            "fd": int(self.f_src.shape[1]),
            "leaf_params": sorted(self.leaf_params),
            "config": self.config,
            "converged": self.converged,
            "num_unresolved_clusters": self.num_unresolved_clusters,
            "num_unresolved_pts": self.num_unresolved_pts,
        }
        try:
            metadata_json = json.dumps(metadata, default=_json_default)
        except TypeError as e:
            raise TypeError(
                f"Cannot save: the model's settings or configuration are not JSON "
                f"serializable ({e}). Give custom components plain settings()."
            ) from e
        arrays = {
            "format_version": np.asarray(MODEL_FORMAT_VERSION),
            "metadata": np.asarray(metadata_json),
            "profiling_stats": np.asarray(
                json.dumps(self.profiling_stats, default=_json_default)
            ),
            "X_src": _to_host(self.X_src),
            "f_src": _to_host(self.f_src),
            **({"Qs_src": _to_host(self.Qs_src)} if self.Qs_src is not None else {}),
            **{_PARAM_PREFIX + k: _to_host(v) for k, v in self.leaf_params.items()},
            **({_ROUTE_PREFIX + k: v for k, v in self.routing.arrays().items()} if self.routing else {}),
            **{_LEAF_PREFIX + k: np.asarray(v) for k, v in self.region_leaves.items()},
        }
        for name in ("E_cluster_abs_src", "E_cluster_rel_src"):
            if getattr(self, name) is not None:
                arrays[name] = _to_host(getattr(self, name))
        if self.counts_src is not None:
            arrays["counts_src"] = _to_host(self.counts_src)

        np.savez_compressed(str(filepath), **arrays)
        logger.info(f"Model saved to {filepath}")
        return filepath

    @classmethod
    def load_model(
        cls, filepath: Union[str, Path], local_model: Optional[LocalModel] = None
    ) -> "AFINNISurrogate":
        """
        Load a model written by ``save_model()``; ``filepath`` is resolved the same
        way, so the path given to ``save_model()`` also loads.

        The local model is rebuilt from its registered name and settings, so a
        custom model's module must be imported (registering it) before loading;
        alternatively pass the ``local_model`` instance to use. Each rank reads the
        file itself, so under MPI it must be on a filesystem every rank can see.
        """
        filepath = _model_path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with np.load(str(filepath), allow_pickle=False) as data:
            if "format_version" not in data.files:
                raise ValueError(
                    f"{filepath} has no format version: it was written before model "
                    f"files were versioned, which is no longer supported; retrain it."
                )
            version = int(data["format_version"])
            if version > MODEL_FORMAT_VERSION:
                raise ValueError(
                    f"{filepath} uses model format {version}, newer than the supported "
                    f"{MODEL_FORMAT_VERSION}; update the code to load it."
                )
            meta = json.loads(str(data["metadata"]))
            arrays = {k: data[k] for k in data.files}

        if local_model is None:
            spec = meta["local_model"]
            if spec["name"] not in LOCAL_MODELS:
                raise ValueError(
                    f"{filepath} was trained with local model {spec['name']!r}, which is "
                    f"not registered. Import the module that registers it, or pass "
                    f"local_model= to load_model()."
                )
            local_model = LOCAL_MODELS.create((spec["name"], spec["settings"]))
        instance = cls()
        instance._install(
            {
                "mode": SurrogateMode(meta["mode"]) if meta["mode"] else None,
                "local_model": local_model,
                "config": meta["config"],
                "X_src": arrays["X_src"],
                "f_src": arrays["f_src"],
                "Qs_src": arrays.get("Qs_src"),
                # Files saved before the per-component error columns have neither.
                "E_cluster_abs_src": arrays.get("E_cluster_abs_src"),
                "E_cluster_rel_src": arrays.get("E_cluster_rel_src"),
                "counts_src": arrays.get("counts_src"),
                "profiling_stats": json.loads(str(arrays["profiling_stats"])),
                "converged": meta["converged"],
                "num_unresolved_clusters": int(meta["num_unresolved_clusters"]),
                "num_unresolved_pts": int(meta["num_unresolved_pts"]),
                "leaf_params": {k[len(_PARAM_PREFIX):]: v for k, v in arrays.items() if k.startswith(_PARAM_PREFIX)},
                "routing": (
                    RoutingTable.from_arrays({k[len(_ROUTE_PREFIX):]: v for k, v in arrays.items() if k.startswith(_ROUTE_PREFIX)})
                    if any(k.startswith(_ROUTE_PREFIX) for k in arrays) else None
                ),
                "region_leaves": {k[len(_LEAF_PREFIX):]: v for k, v in arrays.items() if k.startswith(_LEAF_PREFIX)},
            },
            source=str(filepath),
        )  # fmt: skip
        return instance

    def _install(self, state: dict[str, Any], source: str) -> None:
        """
        Make ``state`` (every field of a trained model, arrays on either backend) the
        current model. The local model's columns are checked and the kNN index is
        built before any attribute changes, so a failure leaves the previous model
        usable; ``source`` names where the state came from in error messages.
        """
        state = dict(state)
        for name in ("X_src", "f_src", "Qs_src", "E_cluster_abs_src", "E_cluster_rel_src"):
            if state[name] is not None:
                state[name] = _to_device(state[name])
        if state["counts_src"] is not None:
            state["counts_src"] = np.asarray(_to_host(state["counts_src"]))  # host-side diagnostics
        state["leaf_params"] = {k: _to_device(v) for k, v in state["leaf_params"].items()}
        state.setdefault("routing", None)
        state["region_leaves"] = {k: np.asarray(_to_host(v)) for k, v in state.get("region_leaves", {}).items()}
        state["N_src"] = n = int(state["X_src"].shape[0])
        d, fd = state["X_src"].shape[1], state["f_src"].shape[1]
        local_model = state["local_model"]
        for name, spec in local_model.columns(d, fd).items():
            value = state["leaf_params"].get(name)
            if value is None or tuple(value.shape) != (n, *spec.shape):
                got = None if value is None else tuple(value.shape)
                raise ValueError(
                    f"{source}: local model {local_model.name!r} needs column "
                    f"{name!r} of shape {(n, *spec.shape)}; found {got}."
                )
        routing = state["routing"]
        if routing is not None:
            if routing.n_leaves != n or routing.domain.d != d:
                raise ValueError(
                    f"{source}: the region tree has {routing.n_leaves} leaves in {routing.domain.d}-D "
                    f"but the model has {n} in {d}-D."
                )
            # Equal counts are not enough: leaf row i must be the same node in both stores.
            tree, meta = routing.arrays(), state["region_leaves"]
            for key, col in (("leaf_root", "root"), ("leaf_path", "path_code")):
                if col in meta and not np.array_equal(np.asarray(tree[key]), meta[col]):
                    raise ValueError(f"{source}: the region tree and the leaf metadata disagree on {col!r}.")
        t0 = time.perf_counter()
        state["knn_index"] = _make_knn_index(state["X_src"]) if state["Qs_src"] is not None else None
        if _get_mpi_rank() == 0:
            logger.info(f"KNN Index built in {time.perf_counter() - t0:.4f}s")
        for name, value in state.items():
            setattr(self, name, value)
        self.trained = True

    def _query_knn(self, query_points: Array, k: int) -> tuple[Array, Array]:
        """``(indices, distances)`` of the ``min(k, N_src)`` nearest centroids, on the active backend."""
        k = min(k, self.N_src)
        queries = _to_device(query_points) if IS_CUPY_BACKEND else _to_host(query_points)
        distances, indices = self.knn_index.kneighbors(queries, k)
        stats = self.knn_index.last_search_stats
        for key in ("queries", "fallback_rows"):
            self.search_stats[key] = self.search_stats.get(key, 0) + stats[key]
        self.search_stats["candidates"] = stats["candidates"]
        self.search_stats["strategy"] = stats.get("strategy", "verified")
        return _to_device(indices), _to_device(distances)

    @_collective("predict()")
    def predict(
        self,
        query_points: Optional[Array] = None,
        k_nn: int = 1,
        coarse_factor: int = 10,
        strict_trust_region: bool = False,
        batch_size: int = 5000,
        predictor: str | None = None,
    ) -> tuple:
        """
        Predict collectively on every rank: rank 0's queries are scattered, predicted
        locally, and gathered; returns ``(predictions, outside_mask)`` on rank 0 and
        ``(None, None)`` elsewhere.

        ``predictor="knn"`` (the default for AFINNI-mode models): the ``k_nn * coarse_factor``
        nearest centroids are reranked by trust-region distance ``dx^T Q dx``; the best
        ``k_nn`` leaves' models are blended with inverse-distance weights
        (``strict_trust_region`` drops leaves with distance > 1). A query is flagged
        outside when its best candidate is > 1.

        ``predictor="route"`` (the default, and only meaningful choice, for region-mode
        models): each query is routed down the tree to exactly one leaf, whose model
        predicts it; "outside" then means outside the domain only (such queries are routed
        from the nearest domain point). ``k_nn`` and the trust-region options do not apply.
        A leaf's ``terminal_reason`` and ``validation_status`` are in ``region_leaves``.
        """
        predictor = predictor or ("route" if self.routing is not None else "knn")
        if predictor not in ("knn", "route"):
            raise ValueError(f"predictor must be 'knn' or 'route'; got {predictor!r}.")
        if predictor == "route" and self.routing is None:
            raise ValueError("predictor='route' needs a region-mode model (algorithm='region').")
        if predictor == "knn" and self.trained and self.Qs_src is None:
            raise ValueError(
                "predictor='knn' needs the trust-region shapes Q, which this region model did not "
                "store (region.store_Q=False); use predictor='route'."
            )
        # Totals over this call's batches on this rank: queries, fallback rows, strategy.
        self.search_stats = {"queries": 0, "fallback_rows": 0, "candidates": 0, "strategy": None}
        if not self.trained:
            raise RuntimeError("Model not trained.")

        if k_nn > coarse_factor:
            raise ValueError(
                "k_nn cannot be greater than coarse_factor for the two-stage kNN to work properly."
            )

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        chunks = None
        if rank == 0:
            if query_points is None:
                raise ValueError("query_points must be provided on rank 0.")
            query_host = _to_host(query_points)
            chunks = np.array_split(query_host, size)

        # Scatter evenly across ranks
        local_queries_host = comm.scatter(chunks, root=0)
        local_queries = xp.asarray(local_queries_host)

        # Run mathematical logic independently with batching
        if predictor == "route":
            local_preds, local_mask = self._predict_route_local(local_queries, batch_size)
        else:
            local_preds, local_mask = self._predict_local(
                query_points=local_queries,
                k_nn=k_nn,
                coarse_factor=coarse_factor,
                strict_trust_region=strict_trust_region,
                batch_size=batch_size,
            )

        # Bring back to host and merge
        gathered_preds = comm.gather(_to_host(local_preds), root=0)
        gathered_masks = comm.gather(_to_host(local_mask), root=0)

        if rank == 0:
            return xp.asarray(np.concatenate(gathered_preds)), xp.asarray(
                np.concatenate(gathered_masks)
            )  # type: ignore

        return None, None

    def _predict_local(
        self,
        query_points: Array,
        k_nn: int,
        coarse_factor: int,
        strict_trust_region: bool,
        batch_size: int = 5000,
        return_cluster_assignments: bool = False,
    ) -> tuple[Array, Array]:
        """
        Predict this rank's queries in chunks of ``batch_size`` to bound the
        (chunk, k, fd, d) gathers; see ``predict`` for the method.
        """
        N, D = query_points.shape
        assert (
            self.trained
            and self.X_src is not None
            and self.f_src is not None
            and self.Qs_src is not None
        )
        if N == 0:  # e.g. fewer queries than MPI ranks
            empty = (
                xp.empty((0, self.f_src.shape[1]), dtype=self.f_src.dtype),
                xp.empty((0,), dtype=bool),
            )
            if return_cluster_assignments:
                return (*empty, xp.empty((0, max(1, min(k_nn, self.N_src))), dtype=xp.int64))
            return empty

        # Determine true k values based on available centroids
        if k_nn < len(self.X_src):
            k_coarse = min(k_nn * coarse_factor, len(self.X_src))
            k_nn = min(k_nn, len(self.X_src))
        else:
            k_coarse = len(self.X_src)
            k_nn = len(self.X_src)

        assert self.local_model is not None, "no local model; was the surrogate trained?"
        leaf_params = self._leaf_params()

        all_predictions = []
        all_ung_masks = []
        all_nn_idx = []

        # Process queries in Memory-Safe Batches
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_queries = query_points[start_idx:end_idx]

            # 1. Phase 1: L2 Coarse Search
            nn_idx_coarse, _ = self._query_knn(batch_queries, k_coarse)

            # 2. Extract coarse candidate properties
            Qs_coarse = self.Qs_src[nn_idx_coarse]
            centroids_coarse = self.X_src[nn_idx_coarse]
            dx_coarse = batch_queries[:, None, :] - centroids_coarse

            # Compute Mahalanobis distance squared for all coarse candidates
            dist_sq_coarse = xp.einsum(
                "nki,nkij,nkj->nk", dx_coarse, Qs_coarse, dx_coarse
            )

            # 3. Phase 2: Refine to top k_nn based on true Trust Region distance
            sorted_local_indices = xp.argsort(dist_sq_coarse, axis=1)
            top_k_local = sorted_local_indices[:, :k_nn]

            nn_idx = xp.take_along_axis(nn_idx_coarse, top_k_local, axis=1)
            dist_sq = xp.take_along_axis(dist_sq_coarse, top_k_local, axis=1)


            # 4. Shepard's Inverse Distance Weighting
            # Weight = 1 / (d^p + eps). Points exactly at the centroid get dominating weight.
            p = 4.0
            base_weights = 1.0 / (xp.power(dist_sq, p / 2) + 1e-12)

            if strict_trust_region:
                # Zero out weights for centroids outside the trust region (d_sq > 1.0)
                final_weights = xp.where(dist_sq <= 1.0, base_weights, 0.0)
            else:
                # Pure Shepard's weighting across all k_nn points
                final_weights = base_weights

            total_weight = xp.sum(final_weights, axis=1)

            # A point is unguaranteed if its CLOSEST centroid is outside the trust region (d_sq > 1.0)
            unguaranteed_mask = xp.min(dist_sq, axis=1) > 1.0

            # Prevent divide-by-zero if strict_trust_region killed all weights
            total_weight = xp.maximum(total_weight, 1e-12)

            # 5. Evaluate each neighbour's local model at the queries: (chunk, k_nn, fd)
            predictions_k = self.local_model.evaluate(leaf_params, batch_queries, nn_idx)

            # 6. Apply Shepard's Weights
            weights_expanded = final_weights[:, :, None]
            weighted_preds = predictions_k * weights_expanded

            # If total_weight was 1e-12 (all centroids strict-rejected), this gracefully zeros out the prediction
            batch_predictions = xp.sum(weighted_preds, axis=1) / total_weight[:, None]

            # Store chunk results
            all_predictions.append(batch_predictions)
            all_ung_masks.append(unguaranteed_mask)
            if return_cluster_assignments:
                all_nn_idx.append(nn_idx)

        # Reconstruct the full arrays
        predictions = xp.concatenate(all_predictions, axis=0)
        unguaranteed_mask = xp.concatenate(all_ung_masks, axis=0)

        if return_cluster_assignments:
            nn_idx_full = xp.concatenate(all_nn_idx, axis=0)
            return predictions, unguaranteed_mask, nn_idx_full  # type: ignore
        else:
            return predictions, unguaranteed_mask

    def _predict_route_local(self, query_points: Array, batch_size: int = 5000) -> tuple[Array, Array]:
        """``(predictions, outside_domain)`` for this rank's queries by routing (see ``predict``)."""
        fd = self.f_src.shape[1]
        if query_points.shape[0] == 0:
            return xp.empty((0, fd), dtype=xp.float64), xp.empty((0,), dtype=bool)
        leaf_params = self._leaf_params()
        preds, outs = [], []
        for s in range(0, int(query_points.shape[0]), batch_size):
            q = xp.asarray(query_points[s : s + batch_size], dtype=xp.float64)
            leaf, outside = self.routing.route(q)
            preds.append(self.local_model.evaluate(leaf_params, q, leaf))
            outs.append(outside)
        return xp.concatenate(preds, axis=0), xp.concatenate(outs, axis=0)

    def route_leaves(self, query_points: Array) -> Array:
        """The (n,) leaf row each query routes to, on this rank (no MPI); region-mode models only."""
        if self.routing is None:
            raise ValueError("route_leaves needs a region-mode model (algorithm='region').")
        return self.routing.route(xp.asarray(query_points, dtype=xp.float64))[0]

    def route_details(self, query_points: Array) -> dict[str, Array]:
        """
        Per query: the leaf row, whether it lies outside the domain, and that leaf's
        ``terminal_reason`` / ``validation_status`` codes (indices into ``TERMINAL_REASONS``
        / ``VALIDATION_STATUSES``), so a caller can flag predictions from weak leaves.
        """
        if self.routing is None:
            raise ValueError("route_details needs a region-mode model (algorithm='region').")
        leaf, outside = self.routing.route(xp.asarray(query_points, dtype=xp.float64))
        out = {"leaf": leaf, "outside": outside}
        for col in ("terminal_reason", "validation_status"):
            out[col] = xp.asarray(self.region_leaves[col])[leaf]
        return out

    def leaf_status_counts(self) -> dict[str, dict[str, int]]:
        """How many leaves end with each terminal reason and validation status (region mode)."""
        out = {}
        for col, names in (("terminal_reason", TERMINAL_REASONS), ("validation_status", VALIDATION_STATUSES)):
            counts = np.bincount(self.region_leaves[col].astype(np.int64), minlength=len(names))
            out[col] = dict(zip(names, map(int, counts), strict=True))
        return out

    def audit(
        self,
        evaluator: Evaluator,
        n_points: int,
        *,
        seed: int = 0,
        confidence: float = 0.95,
        atol: Tolerance = None,
        rtol: Tolerance = None,
    ) -> dict[str, Any]:
        """
        The global accuracy claim for a region-mode model: draw ``n_points`` iid uniform
        points of its domain (independent of training), predict them by routing, and
        count tolerance violations with the same row test as pointwise training
        (``src.acceptance.pointwise_violations``: a point passes only if every component
        has ``|e| < atol`` or ``|e| / max(|f|, 1e-12) < rtol``).

        A non-finite oracle value or prediction counts as a violation (reported as
        ``n_invalid`` too): the bounded event is "fails the tolerance or cannot be
        evaluated", and invalid rows are never dropped, which would change the measure.
        Returns the count and a one-sided ``confidence`` upper bound on the violation
        probability under the uniform measure (Clopper-Pearson; ``1 - (1 - confidence)^(1/n)``
        with no violations: e.g. 299 points bound it below 1% at 95%). The bound is for the
        frozen model and this measure; it does not bound the error at every point, and a
        model chosen after inspecting audits needs a fresh, reserved audit (a new
        ``seed`` alone does not undo that selection). Tolerances default to the ones the
        trained acceptance test used. Runs on the calling rank only.
        """
        if self.routing is None:
            raise ValueError("audit needs a region-mode model (algorithm='region').")
        if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)) or n_points < 1:
            raise ValueError(f"n_points must be a positive integer; got {n_points!r}.")
        if not (isinstance(confidence, (int, float)) and np.isfinite(confidence) and 0 < confidence < 1):
            raise ValueError(f"confidence must be strictly between 0 and 1; got {confidence!r}.")
        from scipy.stats import beta

        from src.acceptance import pointwise_violations

        if atol is None and rtol is None:  # the resolved acceptance test's own tolerances
            trained = self.config.get("components", {}).get("acceptance", {}).get("settings", {})
            atol, rtol = trained.get("atol", self.config.get("atol")), trained.get("rtol", self.config.get("rtol"))
        atol, rtol = normalize_tolerance("atol", atol), normalize_tolerance("rtol", rtol)
        if atol is None and rtol is None:
            raise ValueError("audit needs atol, rtol or both (the model records none).")
        dom = self.routing.domain
        # A NumPy stream on every backend, so the audit points do not depend on the device.
        U = np.random.default_rng(np.random.SeedSequence([int(seed), 0xA0D17])).random((int(n_points), dom.d))
        X = xp.asarray(dom.lower + U * dom.width)
        F, _ = evaluator(X)
        F_hat, _ = self._predict_route_local(X)
        if tuple(F.shape) != tuple(F_hat.shape):
            raise ValueError(f"The oracle returned shape {tuple(F.shape)}; the model predicts {tuple(F_hat.shape)}.")
        violations = pointwise_violations(F, F_hat, atol, rtol)
        invalid = ~xp.all(xp.isfinite(F) & xp.isfinite(F_hat), axis=1)
        fd = F.shape[1]
        scale = xp.maximum(  # the scaled error, a diagnostic only (failure comes from the predicate)
            xp.maximum(tolerance_vector("atol", atol, fd), 0.0),
            xp.maximum(tolerance_vector("rtol", rtol, fd), 0.0) * xp.maximum(xp.abs(F), 1e-12),
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = xp.max(xp.abs(F - F_hat) / scale, axis=1)
        k, n = int(xp.sum(violations)), int(n_points)
        upper = 1.0 if k >= n else float(beta.ppf(confidence, k + 1, n - k))
        finite = scaled[~invalid]
        return {
            "n_points": n, "n_violations": k, "n_invalid": int(xp.sum(invalid)), "violation_rate": k / n,
            "upper_bound": upper, "confidence": confidence,
            "max_scaled_error": float(xp.max(finite)) if int(finite.shape[0]) else float("nan"),
            "measure": "uniform on the domain", "seed": int(seed), "atol": atol, "rtol": rtol,
        }  # fmt: skip

    def filter_unguaranteed_local(
        self, query_points: Array, k_coarse: int = 10, batch_size: int = 50000
    ) -> Array:
        """
        Outside-trust-region mask only, skipping model evaluation; for queries already
        local to this rank (no MPI scatter).
        """
        assert self.trained and self.X_src is not None and self.Qs_src is not None
        N = query_points.shape[0]
        if N == 0:
            return xp.array([], dtype=bool)

        all_ung_masks = []
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_queries = query_points[start_idx:end_idx]

            # 1. Coarse KNN Search
            nn_idx_coarse, _ = self._query_knn(batch_queries, k_coarse)

            # 2. Extract Qs and compute Mahalanobis Distance
            Qs_coarse = self.Qs_src[nn_idx_coarse]
            centroids_coarse = self.X_src[nn_idx_coarse]
            dx_coarse = batch_queries[:, None, :] - centroids_coarse

            dist_sq_coarse = xp.einsum(
                "nki,nkij,nkj->nk", dx_coarse, Qs_coarse, dx_coarse
            )

            # 3. Mask: Point is unguaranteed if its closest Mahalanobis distance > 1.0
            unguaranteed_mask = xp.min(dist_sq_coarse, axis=1) > 1.0
            all_ung_masks.append(unguaranteed_mask)

        return xp.concatenate(all_ung_masks, axis=0)


def compute_error_metrics(
    preds,
    gt,
    ug_mask,
    atol: Tolerance = None,
    rtol: Tolerance = None,
    return_ptwise_error_arrays: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Array]]:
    """
    L2/Linf errors over all queries and separately inside/outside the trust regions
    (``ug_mask`` marks outside), plus the share of inside points that are good: every
    component within ``rtol`` relative or ``atol`` absolute error (inclusive), with the
    tolerances given as in training (see ``src.acceptance``). Sections with no points
    report NaN. Optionally also returns the pointwise relative errors.
    """
    TINY_EPS = 1e-12
    atol, rtol = normalize_tolerance("atol", atol), normalize_tolerance("rtol", rtol)
    if atol is None and rtol is None:
        raise ValueError("Give atol, rtol or both to count the points within tolerance.")

    n_test = gt.shape[0]
    n_out = xp.sum(ug_mask)
    n_in = n_test - n_out

    good_idx = ~ug_mask

    if n_in > 0:
        # 1. Filter to guaranteed points
        gt_in = gt[good_idx]
        preds_in = preds[good_idx]
        diff = preds_in - gt_in
        norm_diff = xp.linalg.norm(diff)
        norm_gt = xp.linalg.norm(gt_in)
        norm_diff_inf = xp.linalg.norm(diff, ord=xp.inf)
        norm_gt_inf = xp.linalg.norm(gt_in, ord=xp.inf)

        in_l2_rel_error = float(norm_diff / (norm_gt + TINY_EPS))
        in_l2_abs_error = float(norm_diff / xp.sqrt(n_in))
        in_linf_rel_error = float(norm_diff_inf / (norm_gt_inf + TINY_EPS))
        in_linf_abs_error = float(norm_diff_inf / xp.sqrt(n_in))

        element_abs_error = xp.abs(diff)
        element_rel_error = xp.abs(diff) / (xp.abs(gt_in) + TINY_EPS)
        # A vector point's reported error is its largest component error.
        ptwise_rel_error = xp.max(element_rel_error, axis=1) if element_rel_error.ndim > 1 else element_rel_error

        in_ptwise_max_rel_error_linf = float(xp.nanmax(ptwise_rel_error))
        in_ptwise_median_rel_error_linf = float(xp.nanmedian(ptwise_rel_error))

        abs_2d = element_abs_error.reshape(len(element_abs_error), -1)
        rel_2d = element_rel_error.reshape(abs_2d.shape)
        fd = abs_2d.shape[1]
        within = (rel_2d <= tolerance_vector("rtol", rtol, fd)) | (abs_2d <= tolerance_vector("atol", atol, fd))
        n_in_good = int(xp.sum(xp.all(within, axis=1)))
        n_in_bad = int(n_in - n_in_good)
    else:
        in_l2_rel_error = float("nan")
        in_l2_abs_error = float("nan")
        in_linf_rel_error = float("nan")
        in_linf_abs_error = float("nan")
        in_ptwise_max_rel_error_linf = float("nan")
        in_ptwise_median_rel_error_linf = float("nan")
        n_in_good = 0
        n_in_bad = 0

    if n_out > 0:
        gt_out = gt[ug_mask]
        preds_out = preds[ug_mask]
        diff_out = preds_out - gt_out
        norm_diff_out = xp.linalg.norm(diff_out)
        norm_diff_out_inf = xp.linalg.norm(diff_out, ord=xp.inf)
        norm_gt_out = xp.linalg.norm(gt_out)
        norm_gt_out_inf = xp.linalg.norm(gt_out, ord=xp.inf)

        out_l2_rel_error = float(norm_diff_out / (norm_gt_out + TINY_EPS))
        out_l2_abs_error = float(norm_diff_out / xp.sqrt(n_out))
        out_linf_rel_error = float(norm_diff_out_inf / (norm_gt_out_inf + TINY_EPS))
        out_linf_abs_error = float(norm_diff_out_inf / xp.sqrt(n_out))

        element_rel_error_out = xp.abs(diff_out) / (xp.abs(gt_out) + TINY_EPS)
        if element_rel_error_out.ndim > 1:
            ptwise_rel_error_out = xp.max(element_rel_error_out, axis=1)
        else:
            ptwise_rel_error_out = element_rel_error_out

        out_ptwise_max_rel_error_linf = float(xp.nanmax(ptwise_rel_error_out))
        out_ptwise_median_rel_error_linf = float(xp.nanmedian(ptwise_rel_error_out))
    else:
        out_l2_rel_error = float("nan")
        out_l2_abs_error = float("nan")
        out_linf_rel_error = float("nan")
        out_linf_abs_error = float("nan")
        out_ptwise_max_rel_error_linf = float("nan")
        out_ptwise_median_rel_error_linf = float("nan")

    # Metrics for all points
    diff_all = preds - gt
    norm_diff_all = xp.linalg.norm(diff_all)
    norm_diff_all_inf = xp.linalg.norm(diff_all, ord=xp.inf)
    norm_gt_all = xp.linalg.norm(gt)
    norm_gt_all_inf = xp.linalg.norm(gt, ord=xp.inf)

    all_l2_rel_error = float(norm_diff_all / (norm_gt_all + TINY_EPS))
    all_l2_abs_error = float(norm_diff_all / xp.sqrt(n_test))
    all_linf_rel_error = float(norm_diff_all_inf / (norm_gt_all_inf + TINY_EPS))
    all_linf_abs_error = float(norm_diff_all_inf / xp.sqrt(n_test))
    element_rel_error_all = xp.abs(diff_all) / (xp.abs(gt) + TINY_EPS)
    if element_rel_error_all.ndim > 1:
        ptwise_rel_error_all = xp.max(element_rel_error_all, axis=1)
    else:
        ptwise_rel_error_all = element_rel_error_all
    all_ptwise_max_rel_error_linf = float(xp.nanmax(ptwise_rel_error_all))
    all_ptwise_median_rel_error_linf = float(xp.nanmedian(ptwise_rel_error_all))

    # Calculate percentages
    perct_out = float(100.0 * n_out / n_test)
    perct_in_good = float(100.0 * n_in_good / n_test)
    perct_in_bad = float(100.0 * n_in_bad / n_test)

    # Create error metrics dictionary
    error_metrics = {
        "n_test": int(n_test),
        "n_out": int(n_out),
        "perct_out": perct_out,
        "n_in_good": n_in_good,
        "perct_in_good": perct_in_good,
        "n_in_bad": n_in_bad,
        "perct_in_bad": perct_in_bad,
        "all": {
            "l2_rel_error": all_l2_rel_error,
            "l2_abs_error": all_l2_abs_error,
            "linf_rel_error": all_linf_rel_error,
            "linf_abs_error": all_linf_abs_error,
            "ptwise_max_rel_error_linf": all_ptwise_max_rel_error_linf,
            "ptwise_median_rel_error_linf": all_ptwise_median_rel_error_linf,
        },
        "in": {
            "l2_rel_error": in_l2_rel_error,
            "l2_abs_error": in_l2_abs_error,
            "linf_rel_error": in_linf_rel_error,
            "linf_abs_error": in_linf_abs_error,
            "ptwise_max_rel_error_linf": in_ptwise_max_rel_error_linf,
            "ptwise_median_rel_error_linf": in_ptwise_median_rel_error_linf,
        },
        "out": {
            "l2_rel_error": out_l2_rel_error,
            "l2_abs_error": out_l2_abs_error,
            "linf_rel_error": out_linf_rel_error,
            "linf_abs_error": out_linf_abs_error,
            "ptwise_max_rel_error_linf": out_ptwise_max_rel_error_linf,
            "ptwise_median_rel_error_linf": out_ptwise_median_rel_error_linf,
        },
    }
    if not return_ptwise_error_arrays:
        return error_metrics
    else:
        return error_metrics, {
            "all": _to_host(ptwise_rel_error_all),
            "in": _to_host(ptwise_rel_error),
            "out": _to_host(ptwise_rel_error_out),
        }
