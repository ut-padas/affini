import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

from src import IS_CUPY_BACKEND, Array, xp
from src.data_sampler import DataSampler
from src.l2_distance import knn_l2_distance

logger = logging.getLogger(__name__)


class HATANNSurrogate:
    """
    Surrogate using Adaptive Clustering + kNN with Ellipsoidal Distance on GPU.
    """

    def __init__(self, version: Literal["v1", "v2"]):
        self.trained = False
        self.version = version

        # Storage for source points (Reference data)
        self.X_src = None
        self.f_src = None
        self.J_src = None
        self.trust_radii_src = None
        self.Qs_src = None
        self.N_src = 0
        self.profiling_stats = {}

        # Model file path (for saving/loading)
        self._model_file = None

    def get_trained_data_as_dict(self) -> dict:
        """
        Return the trained data as a dictionary of arrays.
        Useful for saving or inspection.

        Returns:
            Dictionary containing trained data arrays.
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")

        return {
            "X_src": self.X_src,
            "f_src": self.f_src,
            "J_src": self.J_src,
            "trust_radii_src": self.trust_radii_src,
            "Qs_src": self.Qs_src,
            "N_src": self.N_src,
            "profiling_stats": self.profiling_stats,
        }

    def train(
        self,
        X_train: Array,
        f_fn: Optional[Callable] = None,
        g_fn: Optional[Callable] = None,
        oracle_fn: Optional[Callable[[Array], tuple[Array, Array]]] = None,
        *,
        # Model persistence
        save_model: Optional[Union[bool, str, Path]] = None,
        load_if_exists: bool = True,
        # Clustering params
        TOL: float = 1e-2,
        ERROR_METRIC: Literal["L2", "Linf-local", "Linf-global"] = "Linf-global",
        NUM_INITIAL_CLUSTERS: int = 100,
        MIN_PTS_PER_CLUSTER: int = 100,
        MAX_CLUSTERS_EST: int = 1_000_000,
        SEED: int = 0,
        USE_HESSIAN_INFORMED_SPLIT: bool = True,
        IGNORE_GRADIENT: bool = False,
        BATCH_SIZE: int = 10_000_000,
        MAX_LEVELS: int = 50,
        PROFILE_TIMING: bool = False,
    ):
        """
        Train the surrogate model or load from saved file.

        Args:
            X_train: Training data points
            f_fn: Function to evaluate
            g_fn: Gradient function
            oracle_fn: Combined function that returns (f, g)
            save_model: If True, saves to default location. If str/Path, saves to that
            location. If None/False, doesn't save.
            load_if_exists: If True and save_model file exists, loads instead of training.
            TOL: Tolerance for adaptive clustering
            ... (other clustering parameters)
            PROFILE_TIMING: If True, enables detailed performance profiling (GPU only).
        """
        # Handle save_model parameter
        model_filepath = None
        if save_model is not None and save_model is not False:
            if isinstance(save_model, bool) and save_model:
                # Default save location
                model_filepath = Path.cwd() / "hatann_surrogate.npz"
            else:
                model_filepath = Path(save_model)
                if model_filepath.is_dir():
                    model_filepath = model_filepath / "hatann_surrogate.npz"
                elif model_filepath.suffix != ".npz":
                    model_filepath = model_filepath.with_suffix(".npz")

        # Try to load existing model if requested
        if load_if_exists and model_filepath is not None and model_filepath.exists():
            try:
                loaded_model = self.load_model(model_filepath)
                # Transfer state to current instance
                self.trained = loaded_model.trained
                self.X_src = loaded_model.X_src
                self.f_src = loaded_model.f_src
                self.J_src = loaded_model.J_src
                self.trust_radii_src = loaded_model.trust_radii_src
                self.Qs_src = loaded_model.Qs_src
                self.N_src = loaded_model.N_src
                self.profiling_stats = loaded_model.profiling_stats
                self._model_file = model_filepath
                logger.info(f"Loaded existing model from {model_filepath}")
                return
            except Exception as e:
                logger.warning(
                    f"Failed to load existing model: {e}. Proceeding with training."
                )

        # Validate function arguments
        if oracle_fn is not None:
            if f_fn is not None or g_fn is not None:
                raise ValueError(
                    "Cannot specify both oracle_fn and f_fn/g_fn. Use either oracle_fn OR both f_fn and g_fn."
                )
        else:
            if f_fn is None or g_fn is None:
                raise ValueError("Must specify either oracle_fn OR both f_fn and g_fn.")

        t0 = time.perf_counter()

        if self.version == "v2":
            if not IS_CUPY_BACKEND:
                from src.adaptive_clustering import adaptive_clustering

                # 1. Adaptive Clustering
                results = adaptive_clustering(
                    X=X_train,
                    f_fn=f_fn,
                    g_fn=g_fn,
                    oracle_fn=oracle_fn,
                    TOL=TOL,
                    ERROR_METRIC=ERROR_METRIC,
                    NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
                    MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                    MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
                    SEED=SEED,
                    LOG_MEM_STATS=False,
                    SYNTH_BUDGET_FACTOR=200.0,
                    USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
                    IGNORE_GRADIENT=IGNORE_GRADIENT,
                    Q_SHRINK=0.05,
                    MAX_LEVELS=MAX_LEVELS,
                )

                centroids = xp.asarray(results["centroids"])
                f_vals = xp.asarray(results["f"])
                grads = xp.asarray(results["g"])
                radii = xp.asarray(results["radii"])
                Qs = xp.asarray(results["Q"])
                profiling_stats = {
                    "num_levels": results["num_levels"],
                    "num_generated_pts": results["num_generated_pts"],
                    "total_f_evals": results["total_f_evals"],
                }
            else:
                from src.gpu_adaptive_clustering import (
                    adaptive_clustering as gpu_adaptive_clustering,
                )

                if oracle_fn is not None:
                    raise NotImplementedError(
                        "oracle_fn is not supported with GPU backend."
                    )
                if f_fn is None or g_fn is None:
                    raise ValueError(
                        "Must specify both f_fn and g_fn when using GPU backend."
                    )

                results = gpu_adaptive_clustering(
                    X=X_train,
                    f_fn=f_fn,
                    g_fn=g_fn,
                    TOL=TOL,
                    ERROR_METRIC=ERROR_METRIC,
                    NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
                    MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                    MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
                    USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
                    IGNORE_GRADIENT=IGNORE_GRADIENT,
                    SEED=SEED,
                    BATCH_SIZE=BATCH_SIZE,
                    Q_SHRINK=0.05,
                    PROFILE_TIMING=PROFILE_TIMING,
                )
                centroids = xp.asarray(results["centroids"])
                f_vals = xp.asarray(results["f"])
                grads = xp.asarray(results["g"])
                radii = xp.asarray(results["radii"])
                Qs = xp.asarray(results["Q"])
                profiling_stats = results["profiling_stats"]

            # 2. Store Data
            self.X_src = centroids
            self.f_src = f_vals
            self.J_src = grads
            self.trust_radii_src = radii
            self.Qs_src = Qs
            self.N_src = centroids.shape[0]
            self.profiling_stats = profiling_stats
        else:
            raise NotImplementedError(
                "Version v1 is deprecated. Use version v2 instead."
            )

        # 3. Training complete
        self.trained = True
        logger.info(f"Training Time: {time.perf_counter()-t0:.4f}s")
        logger.info(f"Trained Clusters: {len(self.X_src):,}")

        # Save model if requested
        if model_filepath is not None:
            self.save(model_filepath)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to a file.

        Args:
            filepath: Path to save the model. If directory, uses default filename.
        """
        if not self.trained:
            raise RuntimeError("Cannot save an untrained model.")

        filepath = Path(filepath)

        # If it's a directory, create a default filename
        if filepath.is_dir():
            filepath = filepath / "hatann_surrogate.npz"
        elif filepath.suffix != ".npz":
            filepath = filepath.with_suffix(".npz")

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for saving
        save_dict = {
            "trained": self.trained,
            "N_src": self.N_src,
            "profiling_stats": self.profiling_stats,
        }

        # Convert arrays to numpy for saving
        if IS_CUPY_BACKEND:
            save_dict.update(
                {
                    "X_src": self.X_src.get() if self.X_src is not None else None,
                    "f_src": self.f_src.get() if self.f_src is not None else None,
                    "J_src": self.J_src.get() if self.J_src is not None else None,
                    "trust_radii_src": (
                        self.trust_radii_src.get()
                        if self.trust_radii_src is not None
                        else None
                    ),
                    "Qs_src": self.Qs_src.get() if self.Qs_src is not None else None,
                }
            )
        else:
            save_dict.update(
                {
                    "X_src": self.X_src,
                    "f_src": self.f_src,
                    "J_src": self.J_src,
                    "trust_radii_src": self.trust_radii_src,
                    "Qs_src": self.Qs_src,
                }
            )

        # Save using numpy
        if IS_CUPY_BACKEND:
            import numpy as np

            np.savez_compressed(str(filepath), **save_dict, allow_pickle=True)
        else:
            xp.savez_compressed(str(filepath), **save_dict, allow_pickle=True)

        self._model_file = filepath
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> "HATANNSurrogate":
        """
        Load a trained model from a file.

        Args:
            filepath: Path to the saved model file.

        Returns:
            Loaded HATANNSurrogate instance.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load the data
        if IS_CUPY_BACKEND:
            import numpy as np

            data = np.load(str(filepath), allow_pickle=True)
        else:
            data = xp.load(str(filepath), allow_pickle=True)

        # Create instance
        instance = cls(version="v2")

        # Restore state
        instance.trained = bool(data["trained"])
        instance.N_src = int(data["N_src"])

        # profiling_stats is stored as a 0-d array object if saved via np.savez
        # We try to extract it, handling potential absence in older files
        if "profiling_stats" in data:
            ps_data = data["profiling_stats"]
            if ps_data.shape == ():
                instance.profiling_stats = ps_data.item()
            else:
                # Should not happen if saved correctly as object
                instance.profiling_stats = {}
        else:
            instance.profiling_stats = {}

        # Restore arrays (convert back to appropriate backend)
        instance.X_src = (
            xp.asarray(data["X_src"])
            if data["X_src"] is not None and data["X_src"].size > 0
            else None
        )
        instance.f_src = (
            xp.asarray(data["f_src"])
            if data["f_src"] is not None and data["f_src"].size > 0
            else None
        )
        instance.J_src = (
            xp.asarray(data["J_src"])
            if data["J_src"] is not None and data["J_src"].size > 0
            else None
        )
        instance.trust_radii_src = (
            xp.asarray(data["trust_radii_src"])
            if data["trust_radii_src"] is not None and data["trust_radii_src"].size > 0
            else None
        )
        instance.Qs_src = (
            xp.asarray(data["Qs_src"])
            if data["Qs_src"] is not None and data["Qs_src"].size > 0
            else None
        )

        instance._model_file = filepath
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Loaded model with {instance.N_src:,} clusters")

        return instance

    def get_source_centroids(self) -> Any:
        """Return the source centroids used in training."""
        if not self.trained:
            raise RuntimeError("Model not trained.")
        return self.X_src

    def get_trust_regions(self) -> Any:
        """Return the trust regions for each cluster."""
        if not self.trained:
            raise RuntimeError("Model not trained.")
        return self.Qs_src

    def get_number_of_sources(self) -> int:
        """Return the number of source centroids used in training."""
        if not self.trained:
            raise RuntimeError("Model not trained.")
        return self.N_src

    def predict(
        self,
        query_points,
        k_nn: int = 1,
        strict_trust_region: bool = False,
    ) -> tuple:
        """
        Inference via GPU kNN with L2 Distance.
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")

        if self.version != "v2":
            raise NotImplementedError(
                "Version v1 is deprecated. Use version v2 instead."
            )

        assert (
            self.X_src is not None
            and self.Qs_src is not None
            and self.f_src is not None
            and self.J_src is not None
        )

        qp = xp.asarray(query_points)
        N, D = qp.shape

        # --- 1. Coarse L2 Filter ---
        if k_nn < len(self.X_src):
            COARSE_FACTOR = 10
            k_coarse = min(k_nn * COARSE_FACTOR, len(self.X_src))
            k_nn = min(k_nn, len(self.X_src))
        else:
            k_coarse = len(self.X_src)
            k_nn = len(self.X_src)

        nn_idx_coarse, _, _ = knn_l2_distance(
            qp, self.X_src, self.trust_radii_src, k_coarse
        )

        Qs_coarse = self.Qs_src[nn_idx_coarse]
        centroids_coarse = self.X_src[nn_idx_coarse]
        dx_coarse = qp[:, None, :] - centroids_coarse

        # --- 2. Compute Q-Distance ---
        dist_sq_coarse = xp.einsum("nki,nkij,nkj->nk", dx_coarse, Qs_coarse, dx_coarse)

        # --- 3. Refine: Sort by Q-Distance ---
        sorted_local_indices = xp.argsort(dist_sq_coarse, axis=1)
        top_k_local = sorted_local_indices[:, :k_nn]

        # Gather Top-k Data
        nn_idx = xp.take_along_axis(nn_idx_coarse, top_k_local, axis=1)
        dist_sq = xp.take_along_axis(dist_sq_coarse, top_k_local, axis=1)

        centroids = self.X_src[nn_idx]
        dx = qp[:, None, :] - centroids

        # --- 4. Validity & Weights ---
        dist = xp.sqrt(dist_sq)

        # A. Calculate STRICT status (for the mask only)
        # We check if points would have been zeroed out under strict rules.
        valid_mask_strict = dist_sq <= 1.0
        weights_strict = xp.maximum(0.0, 1.0 - dist) ** 2
        weights_strict *= valid_mask_strict

        # If strict sum is effectively 0, the point is "unguaranteed"
        total_weight_strict = xp.sum(weights_strict, axis=1)
        unguaranteed_mask = total_weight_strict < 1e-12

        # B. Determine Final Weights for Prediction
        if strict_trust_region:
            final_weights = weights_strict
            total_weight = total_weight_strict
        else:
            # Simple Average: Set all weights to 1.0
            final_weights = xp.ones_like(dist_sq)
            total_weight = xp.sum(final_weights, axis=1)

        # --- 5. Taylor Prediction ---
        F_dim = self.f_src.shape[1] if self.f_src.ndim > 1 else 1
        flat_idx = nn_idx.reshape(-1)

        f_neighbors = xp.asarray(self.f_src)[flat_idx].reshape(N, k_nn, F_dim)
        J_neighbors = xp.asarray(self.J_src)[flat_idx].reshape(N, k_nn, F_dim, D)

        linear_term = xp.einsum("nkfd,nkd->nkf", J_neighbors, dx)
        predictions_k = f_neighbors + linear_term

        # --- 6. Averaging ---
        weights_expanded = final_weights[:, :, None]
        weighted_preds = predictions_k * weights_expanded

        # If strict=True and total_weight is 0 -> Predict 0.0 (safe fallback)
        # If strict=False, total_weight is k -> Simple Mean
        predictions = xp.sum(weighted_preds, axis=1) / (total_weight[:, None] + 1e-12)

        return predictions, unguaranteed_mask


if __name__ == "__main__":
    import argparse
    from functools import partial

    import matplotlib.pyplot as plt

    from src import FIGURES_DIR
    from src.logging_utils import setup_logging
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging(console_level=logging.DEBUG, file_output=True)

    SURR_EXP_DIR = FIGURES_DIR / "surrogate_experiments"
    SURR_EXP_DIR.mkdir(parents=True, exist_ok=True)

    # Parse Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dim", type=int, default=2, help="Dimensionality of the problem"
    )
    parser.add_argument(
        "--tol", type=float, default=5e-2, help="Tolerance for training and evaluation"
    )
    parser.add_argument(
        "--n-train", type=int, default=400_000, help="Number of training samples"
    )
    parser.add_argument(
        "--n-test", type=int, default=100_000, help="Number of test samples"
    )
    parser.add_argument(
        "--n-init-clusters",
        type=int,
        default=25,
        help="Number of initial clusters for adaptive clustering",
    )
    parser.add_argument(
        "--n-min-per-cluster",
        type=int,
        default=100,
        help="Minimum points per cluster for adaptive clustering",
    )
    parser.add_argument(
        "--error-metric",
        type=str,
        default="Linf-global",
        choices=["L2", "Linf-local", "Linf-global"],
        help="Error metric for adaptive clustering",
    )
    parser.add_argument(
        "--k-neighbours", type=int, default=1, help="Number of neighbors for kNN"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for training data sampling"
    )
    parser.add_argument(
        "--rosenbrock-a", type=float, default=100, help="Rosenbrock 'a' parameter"
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force re-computation of results"
    )

    # Parse arguments
    args = parser.parse_args()
    dim = args.dim
    TOL = args.tol
    N_train = args.n_train
    N_test = args.n_test
    N_INIT_CLUSTERS = args.n_init_clusters
    N_MIN_PER_CLUSTER = args.n_min_per_cluster
    ERROR_METRIC = args.error_metric
    K_NEIGHBOURS = args.k_neighbours
    SEED = args.seed
    A_ROSENBROCK = args.rosenbrock_a
    FORCE_RECOMPUTE = args.force

    # Creat partial Rosenbrock functions
    rosenbrock = partial(rosenbrock, a=A_ROSENBROCK)
    rosenbrock_grad = partial(rosenbrock_grad, a=A_ROSENBROCK)

    img_fpath = (
        SURR_EXP_DIR
        / f"rosen_nearode_surrogate_rel_error_histogram_rosen{A_ROSENBROCK}_dim{dim}_tol{TOL:.1e}_error{ERROR_METRIC}_Ntrain{N_train}_Ntest{N_test}_k{K_NEIGHBOURS}.png"
    )
    json_fpath = img_fpath.with_suffix(".json")

    # 1. Setup Domain
    DIM = 2
    domain = xp.array([[-3, 3] * DIM]).reshape(DIM, 2)

    # Sample Training Data
    X_train = DataSampler.sample_latin_hypercube(domain, N=N_train, seed=SEED)
    X_train = xp.asarray(X_train)

    # 2. Initialize Surrogate
    surrogate = HATANNSurrogate(version="v2")
    print("--- Training ---")
    _start_train_time = time.perf_counter()

    # Use save/load functionality - will load from cache if exists, otherwise train and save
    model_cache_path = (
        SURR_EXP_DIR
        / f"cached_model_rosen{A_ROSENBROCK}_dim{dim}_tol{TOL:.1e}_Ntrain{N_train}_clusters{N_INIT_CLUSTERS}.npz"
    )

    surrogate.train(
        X_train=X_train,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        save_model=model_cache_path,
        load_if_exists=not FORCE_RECOMPUTE,
        TOL=TOL,
        ERROR_METRIC=ERROR_METRIC,
        NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
        MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
        MAX_CLUSTERS_EST=50_000,
        SEED=SEED,
        PROFILE_TIMING=True,
    )
    _end_train_time = time.perf_counter()
    N_CLUSTERS = surrogate.get_number_of_sources()
    print(
        f"Trained with {N_CLUSTERS:,} clusters in {_end_train_time - _start_train_time:.2f}s"
    )

    # 4. Evaluate
    print("\n--- Evaluation ---")
    X_test = DataSampler.sample_uniform(domain, N=N_test, seed=SEED)
    X_test = xp.asarray(X_test)

    # Example utilizing the new signature with Guarantee checks
    _start_predict_time = time.perf_counter()
    preds, ungauranteed_mask = surrogate.predict(
        X_test,
        k_nn=K_NEIGHBOURS,
    )
    _end_predict_time = time.perf_counter()
    print(
        f"Predicted {N_test:,} test points in {_end_predict_time - _start_predict_time:.2f}s"
    )

    ground_truth = rosenbrock(X_test)
    good_idx = ~ungauranteed_mask
    rel_error_linf = xp.abs(preds[good_idx] - ground_truth[good_idx]) / xp.abs(
        ground_truth[good_idx]
    )
    max_rel_error_linf = xp.nanmax(rel_error_linf)
    median_rel_error_linf = xp.nanmedian(rel_error_linf)

    n_out = xp.sum(ungauranteed_mask)
    perct_out = 100.0 * n_out / N_test

    n_in_good = xp.sum(rel_error_linf <= TOL)
    n_in_bad = xp.sum(rel_error_linf > TOL)
    perct_in_good = 100.0 * n_in_good / N_test
    perct_in_bad = 100.0 * n_in_bad / N_test

    plt.figure(figsize=(8, 5))
    if IS_CUPY_BACKEND:
        plt.hist(rel_error_linf.get().flatten(), bins=50, log=True)
    else:
        plt.hist(rel_error_linf.flatten(), bins=50, log=True)

    plt.xlabel("Relative Error (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(
        f"NearODESurrogate RelError (Rosenbrock: a={A_ROSENBROCK}) (dim={dim}, tol={TOL:.1e}, N_train={N_train:.1e}, N_clusters={surrogate.N_src}, N_test={N_test:.1e}, k={K_NEIGHBOURS}, N_out={n_out} ({perct_out:.2f}%))",
        fontsize=7,
    )
    plt.axvline(
        x=TOL,
        color="red",
        linestyle="--",
        label=f"TOL={TOL:.1e} | N_in_good={int(n_in_good)} ({perct_in_good:.2f}%) | N_in_bad={int(n_in_bad)} ({perct_in_bad:.2f}%)",
    )
    plt.legend()
    plt.xscale("log")
    plt.grid(True)
    plt.savefig(img_fpath, dpi=300)
    print(f"Saved relative error histogram to {img_fpath}")

    json_data_dict = {
        "rosenbrock_a": A_ROSENBROCK,
        "dim": dim,
        "tol": TOL,
        "n_train": N_train,
        "n_test": N_test,
        "n_clusters": surrogate.N_src,
        "k_neighbours": K_NEIGHBOURS,
        "n_out": int(n_out),
        "perct_out": float(perct_out),
        "n_in_good": int(n_in_good),
        "perct_in_good": float(perct_in_good),
        "n_in_bad": int(n_in_bad),
        "perct_in_bad": float(perct_in_bad),
        "max_rel_error_linf": float(max_rel_error_linf),
        "median_rel_error_linf": float(median_rel_error_linf),
        "train_time_sec": float(_end_train_time - _start_train_time),
        "predict_time_sec": float(_end_predict_time - _start_predict_time),
        "profiling_stats": surrogate.profiling_stats,
    }

    with open(json_fpath, "w") as json_file:
        json.dump(json_data_dict, json_file, indent=4)
    print(f"Saved results to {json_fpath}")

    print("\n--- Profiling Stats ---")
    print(surrogate.profiling_stats)
