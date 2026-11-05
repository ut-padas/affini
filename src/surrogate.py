import logging
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import numpy as np

from src import (
    HAS_CUML,
    Array,
    _get_mpi_rank,
    _to_device,
    _to_host,
    xp,
)
from src.adaptive_clustering import adaptive_clustering
from src.adaptive_clustering_utils import ClusteringConfig, Evaluator, SurrogateMode
from src.backend import MPI  # real mpi4py.MPI, or a single-rank stand-in without MPI
from src.local_models import LocalModel, local_model_for_mode, make_local_model

logger = logging.getLogger(__name__)


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
        self.profiling_stats = {}
        self.tree_data = None

        self._model_file = None
        self.knn_index = None

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
        doc="(K, d, d) second-order Taylor Hessians.",
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

    def get_trained_data_as_dict(self) -> dict:
        if not self.trained:
            raise RuntimeError("Model not trained.")

        return {
            "mode": self.mode.value if self.mode else None,
            "X_src": self.X_src,
            "f_src": self.f_src,
            "J_src": self.J_src,
            "Qs_src": self.Qs_src,
            "N_src": self.N_src,
            "profiling_stats": self.profiling_stats,
            "tree_data": self.tree_data,
            "Hs_src": self.Hs_src,
            "A_fits_src": self.A_fits_src,
            "b_fits_src": self.b_fits_src,
        }

    def train(
        self,
        X_train: Array,
        Y_train: Optional[Array] = None,
        f_fn: Optional[Callable] = None,
        g_fn: Optional[Callable] = None,
        oracle_fn: Optional[Callable[[Array], tuple[Array, Array]]] = None,
        *,
        mode: SurrogateMode = SurrogateMode.AFFINE_COV_HESS,
        ATOL: float = 1e-2,
        RTOL: float = 1e-2,
        ERROR_METRIC: Literal["L2", "Linf-local", "Linf-global"] = "Linf-global",
        INITIAL_CLUSTERING_METHOD: Literal[
            "kmeans", "knndbscan", "uniform_grid"
        ] = "kmeans",
        NUM_INITIAL_CLUSTERS: int = 100,
        MIN_PTS_PER_CLUSTER: int = 100,
        MAX_CLUSTERS_EST: int = 1_000_000,
        SEED: int = 0,
        Q_REG: float = 1e-6,
        SAMPLING_METHOD: str = "convex_combination",  # see src.sampling.SAMPLERS
        STORE_HESSIAN: bool = False,
        STORE_AFFINE_FIT: bool = False,
        USE_WORK_STEALING: bool = True,
        BATCH_SIZE: int = 10_000_000,
        MAX_LEVELS: int = 50,
        PROFILE_TIMING: bool = False,
        LOG_MEM_STATS: bool = False,
        SAVE_ALL_GEN_TRAINING_DATA: bool = False,
        HESSIAN: Optional[Callable] = None,
        SAVE_INITIAL_CLUSTERING: bool = False,
        USE_EXISTING_INITIAL_CLUSTERING: bool = False,
    ):
        comm = MPI.COMM_WORLD
        rank = _get_mpi_rank()

        if oracle_fn is not None and (f_fn is not None or g_fn is not None):
            raise ValueError("Cannot specify both oracle_fn and f_fn/g_fn.")

        t0 = time.perf_counter()

        self.mode = mode

        # 1. MPI-Distributed Adaptive Clustering
        config = ClusteringConfig(
            mode=self.mode,
            atol=ATOL,
            rtol=RTOL,
            error_metric=ERROR_METRIC,
            initial_clustering_method=INITIAL_CLUSTERING_METHOD,
            num_initial_clusters=NUM_INITIAL_CLUSTERS,
            save_initial_clustering=SAVE_INITIAL_CLUSTERING,
            use_existing_initial_clustering=USE_EXISTING_INITIAL_CLUSTERING,
            min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
            q_reg=Q_REG,
            sampling_method=SAMPLING_METHOD,
            max_levels=MAX_LEVELS,
            batch_size=BATCH_SIZE,
            seed=SEED,
            max_clusters_est=MAX_CLUSTERS_EST,
            use_work_stealing=USE_WORK_STEALING,
            store_hess=STORE_HESSIAN,
            store_affine_fits=STORE_AFFINE_FIT,
            log_mem_stats=LOG_MEM_STATS,
            profile_timing=PROFILE_TIMING,
            save_all_gen_training_data=SAVE_ALL_GEN_TRAINING_DATA,
        )
        if oracle_fn is None and (
            f_fn is None or (not config.requires_no_grads and g_fn is None)
        ):
            raise ValueError(
                "Must specify oracle_fn or required f_fn/g_fn for the chosen mode."
            )
        self.local_model = make_local_model(config)
        evaluator = Evaluator(
            f_fn=f_fn, g_fn=g_fn, oracle_fn=oracle_fn, hessian_fn=HESSIAN
        )
        results = adaptive_clustering(X_train, config, evaluator, Y=Y_train)

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

        # 3. Store Data on all ranks
        self.X_src = xp.asarray(bcast_data["centroids"])
        self.f_src = xp.asarray(bcast_data["f"])
        self.Qs_src = xp.asarray(bcast_data["Q"])
        self.N_src = self.X_src.shape[0]
        self.profiling_stats = bcast_data["profiling_stats"]

        # Whatever parameters the local model declared, under their column names.
        d, fd = self.X_src.shape[1], self.f_src.shape[1]
        self.leaf_params = {
            name: xp.asarray(bcast_data[name])
            for name in self.local_model.columns(d, fd)
        }

        # 4. Build individual KNN index per rank
        self._build_knn_index()
        self.trained = True

        if rank == 0:
            logger.info(f"Training Time: {time.perf_counter() - t0:.4f}s")
            logger.info(f"Trained Clusters: {self.N_src:,}")

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Deprecated: save the trained model to an ``.npz`` file.

        Saving and loading are being redesigned along with the per-cluster model
        refactor, and this format will change. ``train()`` no longer saves models.
        """
        warnings.warn(
            "AFINNISurrogate.save() is deprecated; the model file format will change.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.trained:
            raise RuntimeError("Cannot save an untrained model.")

        filepath = Path(filepath)
        if filepath.is_dir():
            filepath = filepath / "hatann_surrogate.npz"
        elif filepath.suffix != ".npz":
            filepath = filepath.with_suffix(".npz")

        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "trained": self.trained,
            "mode": self.mode.value if self.mode else None,
            "N_src": self.N_src,
            "X_src": _to_host(self.X_src),
            "f_src": _to_host(self.f_src),
            "J_src": _to_host(self.J_src) if self.J_src is not None else None,
            "Qs_src": _to_host(self.Qs_src),
            "Hs_src": _to_host(self.Hs_src) if self.Hs_src is not None else None,
            "A_fits_src": (
                _to_host(self.A_fits_src) if self.A_fits_src is not None else None
            ),
            "b_fits_src": (
                _to_host(self.b_fits_src) if self.b_fits_src is not None else None
            ),
            "profiling_stats": self.profiling_stats,
            "tree_data": self.tree_data,
        }

        np.savez_compressed(str(filepath), **save_dict, allow_pickle=True)
        self._model_file = filepath
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> "AFINNISurrogate":
        """
        Deprecated: load a model written by ``save()``.

        Saving and loading are being redesigned along with the per-cluster model
        refactor; files from this format will not be readable afterwards.
        """
        warnings.warn(
            "AFINNISurrogate.load_model() is deprecated; the model file format will change.",
            DeprecationWarning,
            stacklevel=2,
        )
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        data = np.load(str(filepath), allow_pickle=True)
        instance = cls()

        instance.trained = bool(data["trained"])

        mode_val = data.get("mode", None)
        if mode_val is not None and mode_val.item() is not None:
            instance.mode = SurrogateMode(mode_val.item())
            instance.local_model = local_model_for_mode(instance.mode)

        instance.N_src = int(data["N_src"])
        instance.X_src = _to_device(data["X_src"])
        instance.f_src = _to_device(data["f_src"])

        instance.J_src = _to_device(data["J_src"]) if data["J_src"].shape else None
        instance.Qs_src = _to_device(data["Qs_src"])
        instance.Hs_src = _to_device(data["Hs_src"]) if data["Hs_src"].shape else None
        instance.A_fits_src = (
            _to_device(data["A_fits_src"]) if data["A_fits_src"].shape else None
        )
        instance.b_fits_src = (
            _to_device(data["b_fits_src"]) if data["b_fits_src"].shape else None
        )

        _prof = data["profiling_stats"]
        instance.profiling_stats = _prof.item() if _prof.size > 0 else {}
        _td = data["tree_data"]
        instance.tree_data = _td.item() if _td.size > 0 else None

        instance._model_file = filepath
        instance._build_knn_index()

        return instance

    def _build_knn_index(self):
        if self.X_src is None:
            return

        t0 = time.perf_counter()
        # Keep X_src in the active array backend (device under CuPy); only the
        # kNN index itself lives on the host when cuML is unavailable.
        self.X_src = _to_device(self.X_src)
        if HAS_CUML:
            from cuml.neighbors import NearestNeighbors as cuMLNearestNeighbors

            self.knn_index = cuMLNearestNeighbors(metric="euclidean", algorithm="auto")
            self.knn_index.fit(self.X_src)
        else:
            # CPU backend, or CuPy without cuML (e.g. Windows + GPU): FAISS on host.
            import faiss

            centroids_f32 = np.ascontiguousarray(_to_host(self.X_src), dtype=np.float32)
            D = self.X_src.shape[1]

            # ANN LOGIC FOR CPU
            if self.N_src > 5000:
                # Use HNSW graph for ultra-fast approximate search on large cluster sets
                # 32 = number of links per node (higher is more accurate but uses more RAM)
                self.knn_index = faiss.IndexHNSWFlat(D, 32)
                # efConstruction controls index time vs accuracy trade-off
                self.knn_index.hnsw.efConstruction = 40
                self.knn_index.add(centroids_f32)  # type: ignore
                # efSearch controls query time vs accuracy trade-off
                self.knn_index.hnsw.efSearch = 32
            else:
                # Fallback to exact search for small models
                self.knn_index = faiss.IndexFlatL2(D)
                self.knn_index.add(centroids_f32)  # type: ignore

        if _get_mpi_rank() == 0:
            logger.info(f"KNN Index built in {time.perf_counter() - t0:.4f}s")

    def _query_knn(self, query_points: Array, k: int) -> tuple[Array, Array]:
        if self.knn_index is None:
            self._build_knn_index()

        k = min(k, self.N_src)
        if HAS_CUML:
            query_points = _to_device(query_points)
            distances, indices = self.knn_index.kneighbors(query_points, n_neighbors=k)  # type: ignore
        else:
            query_points_f32 = np.ascontiguousarray(_to_host(query_points), dtype=np.float32)
            distances_sq, indices = self.knn_index.search(query_points_f32, k)  # type: ignore
            # FAISS returns host arrays; move them to the active backend so
            # downstream indexing of device-resident X_src/f_src etc. works.
            indices = _to_device(indices)
            distances = _to_device(np.sqrt(distances_sq))

        return indices, distances

    def predict(
        self,
        query_points: Optional[Array] = None,
        k_nn: int = 1,
        coarse_factor: int = 10,
        strict_trust_region: bool = False,
        batch_size: int = 5000,
    ) -> tuple:
        """
        MPI-Parallel Inference via kNN with Memory-Safe Batching.
        Rank 0 splits and scatters query points, computes locally, and gathers results.
        """
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
        Core localized math function execution on an individual rank's data chunk,
        processed in batches to prevent Out-Of-Memory (OOM) errors during einsum.
        """
        N, D = query_points.shape
        assert (
            self.trained
            and self.X_src is not None
            and self.f_src is not None
            and self.Qs_src is not None
        )

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

    def filter_unguaranteed_local(
        self, query_points: Array, k_coarse: int = 10, batch_size: int = 50000
    ) -> Array:
        """
        Fast dedicated method to return only the unguaranteed mask.
        Bypasses all surrogate prediction math (Affine/Taylor/Shepard).
        Assumes query_points are already local to the rank.
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
    preds, gt, ug_mask, atol: float, rtol: float, return_ptwise_error_arrays: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Array]]:
    """
    Compute error metrics for the predictions.

    Args:
        preds: Predicted values (N, F_dim)
        gt: True values (N, F_dim)
        ug_mask: Boolean array indicating which points are unguaranteed (N,)
        atol: Absolute tolerance for determining "good" vs "bad" points among the guaranteed ones.
        rtol: Relative tolerance for determining "good" vs "bad" points among the guaranteed ones.
        return_ptwise_error_arrays: If True, also return the pointwise relative error arrays.

    Returns:
        Dictionary of error metrics.
    """
    TINY_EPS = 1e-12

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
        if element_rel_error.ndim > 1:
            # Vector Case: A point is "good" only if ALL components are good.
            # Equivalently, the point error is the MAX error across components.
            ptwise_abs_error = xp.max(element_abs_error, axis=1)  # Shape: (N_in,)
            ptwise_rel_error = xp.max(element_rel_error, axis=1)  # Shape: (N_in,)
        else:
            # Scalar Case: Just take the element-wise relative error.
            ptwise_abs_error = element_abs_error  # Shape: (N_in,)
            ptwise_rel_error = element_rel_error  # Shape: (N_in,)

        in_ptwise_max_rel_error_linf = float(xp.nanmax(ptwise_rel_error))
        in_ptwise_median_rel_error_linf = float(xp.nanmedian(ptwise_rel_error))

        n_in_good = int(xp.sum(xp.logical_or(ptwise_rel_error <= rtol, ptwise_abs_error <= atol)))
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
