import logging
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
from mpi4py import MPI

from src import IS_CUPY_BACKEND, Array, _get_mpi_rank, _to_device, _to_host, xp
from src.adaptive_clustering_utils import SurrogateMode
from src.adaptive_clustering import adaptive_clustering

logger = logging.getLogger(__name__)


class AFINNISurrogate:
    """
    Surrogate using Adaptive Clustering + kNN with Ellipsoidal Distance on GPU.
    Supports MPI-parallelized training and inference.
    """

    def __init__(self):
        self.trained = False
        self.mode: Optional[SurrogateMode] = None

        # Storage for source points (Reference data)
        self.N_src = 0
        self.X_src = None
        self.f_src = None
        self.J_src = None
        self.trust_radii_src = None
        self.Qs_src = None
        self.Hs_src = None
        self.A_fits_src = None
        self.b_fits_src = None
        self.profiling_stats = {}
        self.tree_data = None

        self._model_file = None
        self.knn_index = None

    def get_trained_data_as_dict(self) -> dict:
        if not self.trained:
            raise RuntimeError("Model not trained.")

        return {
            "mode": self.mode.value if self.mode else None,
            "X_src": self.X_src,
            "f_src": self.f_src,
            "J_src": self.J_src,
            "trust_radii_src": self.trust_radii_src,
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
        USE_V2: bool,
        f_fn: Optional[Callable] = None,
        g_fn: Optional[Callable] = None,
        oracle_fn: Optional[Callable[[Array], tuple[Array, Array]]] = None,
        *,
        mode: SurrogateMode = SurrogateMode.AFFINE_COV_HESS,
        save_model: Optional[Union[bool, str, Path]] = None,
        load_if_exists: bool = True,
        TOL: float = 1e-2,
        ERROR_METRIC: Literal["L2", "Linf-local", "Linf-global"] = "Linf-global",
        INITIAL_CLUSTERING_METHOD: Literal[
            "kmeans", "knndbscan", "uniform_grid"
        ] = "kmeans",
        NUM_INITIAL_CLUSTERS: int = 100,
        MIN_PTS_PER_CLUSTER: int = 100,
        MAX_CLUSTERS_EST: int = 1_000_000,
        SEED: int = 0,
        STORE_HESSIAN: bool = False,
        STORE_AFFINE_FIT: bool = False,
        USE_WORK_STEALING: bool = True,
        BATCH_SIZE: int = 10_000_000,
        MAX_LEVELS: int = 50,
        PROFILE_TIMING: bool = False,
        LOG_MEM_STATS: bool = False,
        SAVE_ALL_GEN_TRAINING_DATA: bool = False,
    ):
        comm = MPI.COMM_WORLD
        rank = _get_mpi_rank()

        model_filepath = None
        if save_model is not None and save_model is not False:
            if isinstance(save_model, bool) and save_model:
                model_filepath = Path.cwd() / "hatann_surrogate.npz"
            else:
                model_filepath = Path(save_model)
                if model_filepath.is_dir():
                    model_filepath = model_filepath / "hatann_surrogate.npz"
                elif model_filepath.suffix != ".npz":
                    model_filepath = model_filepath.with_suffix(".npz")

        if load_if_exists and model_filepath is not None and model_filepath.exists():
            try:
                loaded_model = self.load_model(model_filepath)
                self.__dict__.update(loaded_model.__dict__)
                if rank == 0:
                    logger.info(f"Loaded existing model from {model_filepath}")
                return
            except Exception as e:
                if rank == 0:
                    logger.warning(
                        f"Failed to load existing model: {e}. Proceeding with training."
                    )

        if oracle_fn is not None:
            if f_fn is not None or g_fn is not None:
                raise ValueError("Cannot specify both oracle_fn and f_fn/g_fn.")
        else:
            if (
                f_fn is None
                or mode != SurrogateMode.AFFINE_COV_HESS
                and mode != SurrogateMode.NEAREST_NEIGHBOR
                and g_fn is None
            ):
                raise ValueError(
                    "Must specify oracle_fn or required f_fn/g_fn for the chosen mode."
                )

        t0 = time.perf_counter()

        self.mode = mode

        # 1. MPI-Distributed Adaptive Clustering
        results = adaptive_clustering(
            X=X_train,
            f_fn=f_fn,
            g_fn=g_fn,
            oracle_fn=oracle_fn,
            TOL=TOL,
            ERROR_EVAL_MODE=self.mode,
            ERROR_METRIC=ERROR_METRIC,
            INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
            NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
            MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            SEED=SEED,
            Q_SHRINK=0.05,
            MAX_LEVELS=MAX_LEVELS,
            BATCH_SIZE=BATCH_SIZE,
            PROFILE_TIMING=PROFILE_TIMING,
            STORE_HESSIAN_PER_CLUSTER=STORE_HESSIAN,
            STORE_AFFINE_FIT_PER_CLUSTER=STORE_AFFINE_FIT,
            USE_WORK_STEALING=USE_WORK_STEALING,
            LOG_MEM_STATS=LOG_MEM_STATS,
            SAVE_ALL_GEN_TRAINING_DATA=SAVE_ALL_GEN_TRAINING_DATA,
        )

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

        # Guard optional arrays
        if "g" in bcast_data and bcast_data["g"] is not None:
            self.J_src = xp.asarray(bcast_data["g"])
        self.trust_radii_src = xp.asarray(bcast_data["radii"])
        self.Qs_src = xp.asarray(bcast_data["Q"])
        self.N_src = self.X_src.shape[0]
        self.profiling_stats = bcast_data["profiling_stats"]

        if "H" in bcast_data and bcast_data["H"] is not None:
            self.Hs_src = xp.asarray(bcast_data["H"])
        if "A_fit" in bcast_data and bcast_data["A_fit"] is not None:
            self.A_fits_src = xp.asarray(bcast_data["A_fit"])
            self.b_fits_src = xp.asarray(bcast_data["b_fit"])

        # 4. Build individual KNN index per rank
        self._build_knn_index()
        self.trained = True

        if rank == 0:
            logger.info(f"Training Time: {time.perf_counter()-t0:.4f}s")
            logger.info(f"Trained Clusters: {self.N_src:,}")
            if model_filepath is not None:
                self.save(model_filepath)

    def save(self, filepath: Union[str, Path]) -> None:
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
            "trust_radii_src": _to_host(self.trust_radii_src),
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
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        data = np.load(str(filepath), allow_pickle=True)
        instance = cls()

        instance.trained = bool(data["trained"])

        mode_val = data.get("mode", None)
        if mode_val is not None and mode_val.item() is not None:
            instance.mode = SurrogateMode(mode_val.item())

        instance.N_src = int(data["N_src"])
        instance.X_src = _to_device(data["X_src"])
        instance.f_src = _to_device(data["f_src"])

        instance.J_src = _to_device(data["J_src"]) if data["J_src"].shape else None
        instance.trust_radii_src = _to_device(data["trust_radii_src"])
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
        if IS_CUPY_BACKEND:
            from cuml.neighbors import NearestNeighbors as cuMLNearestNeighbors

            self.knn_index = cuMLNearestNeighbors(metric="euclidean", algorithm="auto")
            self.X_src = _to_device(self.X_src)
            self.knn_index.fit(self.X_src)
        else:
            import faiss

            self.X_src = _to_host(self.X_src)
            centroids_f32 = self.X_src.astype(np.float32)
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
            logger.info(f"KNN Index built in {time.perf_counter()-t0:.4f}s")

    def _query_knn(self, query_points: Array, k: int) -> tuple[Array, Array]:
        if self.knn_index is None:
            self._build_knn_index()

        k = min(k, self.N_src)
        if IS_CUPY_BACKEND:
            query_points = _to_device(query_points)
            distances, indices = self.knn_index.kneighbors(query_points, n_neighbors=k)  # type: ignore
        else:
            query_points = _to_host(query_points)
            query_points_f32 = query_points.astype(xp.float32)
            distances_sq, indices = self.knn_index.search(query_points_f32, k)  # type: ignore
            distances = xp.sqrt(distances_sq)

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
            return xp.asarray(np.concatenate(gathered_preds)), xp.asarray(np.concatenate(gathered_masks))  # type: ignore

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

        F_dim = self.f_src.shape[1] if self.f_src.ndim > 1 else 1

        all_predictions = []
        all_ung_masks = []
        all_nn_idx = []

        # Process queries in Memory-Safe Batches
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_queries = query_points[start_idx:end_idx]
            chunk_size = len(batch_queries)

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

            centroids = self.X_src[nn_idx]
            dx = batch_queries[:, None, :] - centroids

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

            # Extract function values for the k_nn neighbors
            flat_idx = nn_idx.reshape(-1)
            f_neighbors = xp.asarray(self.f_src)[flat_idx].reshape(
                chunk_size, k_nn, F_dim
            )

            # ---------------------------------------------------------------------
            # 5. Dynamic Mode Routing (Local Model Evaluation)
            # ---------------------------------------------------------------------
            if self.mode == SurrogateMode.NEAREST_NEIGHBOR:
                predictions_k = f_neighbors

            elif self.mode in (
                SurrogateMode.TAYLOR_LSQ_HESS,
                SurrogateMode.TAYLOR_COV_HESS,
                SurrogateMode.TAYLOR_COV_NO_HESS,
            ):
                J_neighbors = xp.asarray(self.J_src)[flat_idx].reshape(
                    chunk_size, k_nn, F_dim, D
                )
                linear_term = xp.einsum("nkfd,nkd->nkf", J_neighbors, dx)
                predictions_k = f_neighbors + linear_term

                if self.Hs_src is not None:
                    H_neighbors = xp.asarray(self.Hs_src)[flat_idx].reshape(
                        chunk_size, k_nn, F_dim, D, D
                    )
                    quadratic_term = xp.einsum(
                        "nkc,nkfcd,nkd->nkf", dx, H_neighbors, dx
                    )
                    predictions_k += 0.5 * quadratic_term

            elif self.mode in (
                SurrogateMode.AFFINE_COV_HESS,
                SurrogateMode.AFFINE_COV_NO_HESS,
            ):
                A_neighbors = xp.asarray(self.A_fits_src)[flat_idx].reshape(
                    chunk_size, k_nn, F_dim, D
                )
                b_neighbors = xp.asarray(self.b_fits_src)[flat_idx].reshape(
                    chunk_size, k_nn, F_dim
                )

                # Evaluate the affine model: A*x + b
                affine_term = xp.einsum("nkfd,nd->nkf", A_neighbors, batch_queries)
                predictions_k = affine_term + b_neighbors

            else:
                raise ValueError(
                    f"Unknown or improperly set SurrogateMode: '{self.mode}'."
                )

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
    preds, gt, ug_mask, tol: float, return_ptwise_error_arrays: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Array]]:
    """
    Compute error metrics for the predictions.

    Args:
        preds: Predicted values (N, F_dim)
        gt: True values (N, F_dim)
        ug_mask: Boolean array indicating which points are unguaranteed (N,)
        tol: Tolerance for determining "good" vs "bad" points among the guaranteed ones.
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

        in_l2_rel_error = float(norm_diff / (norm_gt + TINY_EPS))
        in_l2_abs_error = float(norm_diff / xp.sqrt(n_in))

        element_rel_error = xp.abs(diff) / (xp.abs(gt_in) + TINY_EPS)
        if element_rel_error.ndim > 1:
            # Vector Case: A point is "good" only if ALL components are good.
            # Equivalently, the point error is the MAX error across components.
            ptwise_rel_error = xp.max(element_rel_error, axis=1)  # Shape: (N_in,)
        else:
            # Scalar Case: Just take the element-wise relative error.
            ptwise_rel_error = element_rel_error  # Shape: (N_in,)

        in_ptwise_max_rel_error_linf = float(xp.nanmax(ptwise_rel_error))
        in_ptwise_median_rel_error_linf = float(xp.nanmedian(ptwise_rel_error))

        n_in_good = int(xp.sum(ptwise_rel_error <= tol))
        n_in_bad = int(n_in - n_in_good)
    else:
        in_l2_rel_error = float("nan")
        in_l2_abs_error = float("nan")
        in_ptwise_max_rel_error_linf = float("nan")
        in_ptwise_median_rel_error_linf = float("nan")
        n_in_good = 0
        n_in_bad = 0

    if n_out > 0:
        gt_out = gt[ug_mask]
        preds_out = preds[ug_mask]
        diff_out = preds_out - gt_out
        norm_diff_out = xp.linalg.norm(diff_out)
        norm_gt_out = xp.linalg.norm(gt_out)

        out_l2_rel_error = float(norm_diff_out / (norm_gt_out + TINY_EPS))
        out_l2_abs_error = float(norm_diff_out / xp.sqrt(n_out))

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
        out_ptwise_max_rel_error_linf = float("nan")
        out_ptwise_median_rel_error_linf = float("nan")

    # Metrics for all points
    diff_all = preds - gt
    norm_diff_all = xp.linalg.norm(diff_all)
    norm_gt_all = xp.linalg.norm(gt)

    all_l2_rel_error = float(norm_diff_all / (norm_gt_all + TINY_EPS))
    all_l2_abs_error = float(norm_diff_all / xp.sqrt(n_test))
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
            "ptwise_max_rel_error_linf": all_ptwise_max_rel_error_linf,
            "ptwise_median_rel_error_linf": all_ptwise_median_rel_error_linf,
        },
        "in": {
            "l2_rel_error": in_l2_rel_error,
            "l2_abs_error": in_l2_abs_error,
            "ptwise_max_rel_error_linf": in_ptwise_max_rel_error_linf,
            "ptwise_median_rel_error_linf": in_ptwise_median_rel_error_linf,
        },
        "out": {
            "l2_rel_error": out_l2_rel_error,
            "l2_abs_error": out_l2_abs_error,
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


if __name__ == "__main__":
    import json
    import time
    from functools import partial

    from src import RESULTS_DIR, _get_mpi_rank, _get_mpi_size
    from src.logging_utils import setup_logging
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging(console_level=logging.DEBUG)
    _rank = _get_mpi_rank()
    _size = _get_mpi_size()

    # Create partial Rosenbrock functions (all ranks need these for oracle calls)
    A_ROSENBROCK = 100.0
    rosenbrock = partial(rosenbrock, a=A_ROSENBROCK)
    rosenbrock_grad = partial(rosenbrock_grad, a=A_ROSENBROCK)

    N_TRAIN = 100_000
    N_TEST = 100_000
    DIM = 30
    PROB_LIMS = (-1.0, 3)
    TOL = 0.4
    NUM_INITIAL_CLUSTERS = 10
    MIN_PTS_PER_CLUSTER = 500
    USE_WORK_STEALING = True
    ERROR_METRIC = "Linf-local"
    INITIAL_CLUSTERING_METHOD = "uniform_grid"
    ERROR_EVAL_MODE = SurrogateMode.AFFINE_COV_HESS

    USE_V2 = True

    # Scale initial clusters with dimension for better
    NUM_INITIAL_CLUSTERS = NUM_INITIAL_CLUSTERS * DIM

    OUTPUT_DIR = RESULTS_DIR / "surrogate_experimental"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if USE_V2:
        json_fpath = (
            OUTPUT_DIR
            / f"mpi_sz{_size}_rosen{A_ROSENBROCK}_ntrain{N_TRAIN}_dim{DIM}_tol{TOL}_err{ERROR_EVAL_MODE.value}_v2_metric{ERROR_METRIC}_init{INITIAL_CLUSTERING_METHOD}_nclust{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}.json"
        )
    else:
        json_fpath = (
            OUTPUT_DIR
            / f"mpi_sz{_size}_rosen{A_ROSENBROCK}_ntrain{N_TRAIN}_dim{DIM}_tol{TOL}_err{ERROR_EVAL_MODE.value}_metric{ERROR_METRIC}_init{INITIAL_CLUSTERING_METHOD}_nclust{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}.json"
        )

    if _rank == 0:
        xp.random.seed(0)
        X = xp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(N_TRAIN, DIM))
    else:
        X = xp.empty((0, DIM), dtype=xp.float64)

    # Train surrogate
    surrogate = AFINNISurrogate()
    _start_time = time.perf_counter()
    surrogate.train(
        X_train=X,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        mode=ERROR_EVAL_MODE,
        TOL=TOL,
        ERROR_METRIC=ERROR_METRIC,
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        USE_WORK_STEALING=USE_WORK_STEALING,
        STORE_AFFINE_FIT=True,
        USE_V2=USE_V2,
    )
    _train_time = time.perf_counter() - _start_time

    # Evaluate Surrogate
    if _rank == 0:
        print("\n--- Evaluation ---")
        xp.random.seed(100)
        X_test = xp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(N_TEST, DIM))
        X_test = xp.asarray(X_test)
        ground_truth = rosenbrock(X_test)
        query_pts_input = X_test
    else:
        query_pts_input = None

    # Predict in parallel across MPI ranks
    t0_pred = time.perf_counter()
    _start_time = time.perf_counter()
    preds, ug_mask = surrogate.predict(
        query_points=query_pts_input,
        k_nn=1,
        strict_trust_region=False,
    )
    _pred_time = time.perf_counter() - _start_time

    # Evaluate error metrics and print only on Rank 0
    if _rank != 0:
        MPI.Finalize()
        raise SystemExit(0)

    logger.info(f"Prediction completed in {time.perf_counter() - t0_pred:.4f}s")

    # Ensure ground truth and predictions are 2D arrays (N, F_dim)
    # as expected by compute_error_metrics
    if ground_truth.ndim == 1:
        ground_truth = ground_truth.reshape(-1, 1)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    # Compute the metrics
    error_metrics = compute_error_metrics(
        preds=preds,
        gt=ground_truth,
        ug_mask=ug_mask,
        tol=TOL,
        return_ptwise_error_arrays=False,
    )

    json_data = {
        "IS_CUPY_BACKEND": IS_CUPY_BACKEND,
        "mpi_size": _size,
        "rosenbrock_a": A_ROSENBROCK,
        "dim": DIM,
        "tol": TOL,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "num_clusters": surrogate.N_src,
        "train_time_sec": _train_time,
        "pred_time_sec": _pred_time,
        "num_levels": surrogate.profiling_stats.get("num_levels", None),
        "num_generated_pts": surrogate.profiling_stats.get("num_generated_pts", None),
        "total_f_evals": surrogate.profiling_stats.get("total_f_evals", None),
        "error_metric": ERROR_METRIC,
        "error_eval_mode": (
            (ERROR_EVAL_MODE.value + "_v2") if USE_V2 else ERROR_EVAL_MODE.value
        ),
        "initial_clustering_method": INITIAL_CLUSTERING_METHOD,
        "num_initial_clusters": NUM_INITIAL_CLUSTERS,
        "min_pts_per_cluster": MIN_PTS_PER_CLUSTER,
        "use_work_stealing": USE_WORK_STEALING,
        # Add the error metrics to the JSON data
        "error_metrics": error_metrics,
    }
    # Print the metrics nicely
    print("\n=== Error Metrics ===")
    print(json.dumps(error_metrics, indent=4))

    # Save the metrics to a JSON file
    with open(json_fpath, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Error metrics saved to {json_fpath}")
