import json
import logging
import time
from typing import Any, Callable, Literal

from src import IS_CUPY_BACKEND, Array, xp
from src.data_sampler import DataSampler
from src.l2_distance import knn_l2_distance

logger = logging.getLogger(__name__)


class NearODESurrogate:
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
        self.training_num_levels = -1
        self.num_generated_pts = -1
        self.total_f_evals = -1

    def train(
        self,
        X_train: Array,
        f_fn: Callable,
        g_fn: Callable,
        *,
        # Clustering params
        TOL: float = 1e-2,
        ERROR_METRIC: Literal["L2", "Linf-local", "Linf-global"] = "Linf-global",
        NUM_INITIAL_CLUSTERS: int = 100,
        MIN_PTS_PER_CLUSTER: int = 100,
        MAX_CLUSTERS_EST: int = 1_000_000,
        SEED: int = 0,
        USE_HESSIAN_INFORMED_SPLIT: bool = True,
        IGNORE_GRADIENT: bool = False,
    ):
        """
        1. Sample X from domain.
        2. Run GPU Adaptive Clustering.
        3. Store Reference Data.
        """
        t0 = time.perf_counter()

        if self.version == "v2":
            if not IS_CUPY_BACKEND:
                from src.adaptive_clustering import adaptive_clustering

                # 1. Adaptive Clustering
                results = adaptive_clustering(
                    X=X_train,
                    f_fn=f_fn,
                    g_fn=g_fn,
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
                )

                centroids = xp.asarray(results["centroids"])
                f_vals = xp.asarray(results["f"])
                grads = xp.asarray(results["g"])
                radii = xp.asarray(results["radii"])
                Qs = xp.asarray(results["Q"])
                num_levels = results.get("num_levels", -1)
                num_generated_pts = results.get("num_generated_pts", -1)
                total_f_evals = results.get("total_f_evals", -1)
            else:
                from src.gpu_adaptive_clustering import (
                    adaptive_clustering as gpu_adaptive_clustering,
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
                    Q_SHRINK=0.05,
                )
                centroids = xp.asarray(results["centroids"])
                f_vals = xp.asarray(results["f"])
                grads = xp.asarray(results["g"])
                radii = xp.asarray(results["radii"])
                Qs = xp.asarray(results["Q"])
                num_levels = results.get("num_levels", -1)
                num_generated_pts = results.get("num_generated_pts", -1)
                total_f_evals = results.get("total_f_evals", -1)

            # 2. Store Data
            self.X_src = centroids
            self.f_src = f_vals
            self.J_src = grads
            self.trust_radii_src = radii
            self.Qs_src = Qs
            self.N_src = centroids.shape[0]
            self.training_num_levels = num_levels
            self.num_generated_pts = num_generated_pts
            self.total_f_evals = total_f_evals
        else:
            raise NotImplementedError(
                "Version v1 is deprecated. Use version v2 instead."
            )

        # 3. Training complete
        self.trained = True
        logger.info(f"Training Time: {time.perf_counter()-t0:.4f}s")
        logger.info(f"Trained Clusters: {len(self.X_src):,}")

    def get_source_centroids(self) -> Any:
        """Return the source centroids used in training."""
        if not self.trained:
            raise RuntimeError("Model not trained.")
        return self.X_src

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
        "--n-train", type=int, default=200_000, help="Number of training samples"
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
    if img_fpath.exists() and json_fpath.exists() and not FORCE_RECOMPUTE:
        print(f"Results already exist at {img_fpath} and {json_fpath}. Exiting.")
        exit(0)

    # 1. Setup Domain
    DIM = 2
    domain = xp.array([[-3, 3] * DIM]).reshape(DIM, 2)

    # Sample Training Data
    X_train = DataSampler.sample_latin_hypercube(domain, N=N_train, seed=SEED)
    X_train = xp.asarray(X_train)

    # 2. Initialize Surrogate
    surrogate = NearODESurrogate(version="v2")
    print("--- Training ---")
    _start_train_time = time.perf_counter()
    surrogate.train(
        X_train=X_train,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        TOL=TOL,
        ERROR_METRIC=ERROR_METRIC,
        NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
        MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
        MAX_CLUSTERS_EST=50_000,
        SEED=SEED,
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
    }

    with open(json_fpath, "w") as json_file:
        json.dump(json_data_dict, json_file, indent=4)
    print(f"Saved results to {json_fpath}")
