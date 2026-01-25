import json
import logging
import time
from typing import Any, Callable, Literal

from src import IS_CUPY_BACKEND, Array, xp
from src.adaptive_clustering import adaptive_clustering
from src.data_sampler import DataSampler
from src.l2_distance import knn_l2_distance
from src.v1_adaptive_clustering import adaptive_cluster_algorithm
from src.v1_gpu_adaptive_clustering import adaptive_cluster_algorithm_gpu

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
        self.trust_radii = None
        self.obb_rotation_matrices = None
        self.obb_extents = None
        self.N_src = 0
        self.training_num_levels = -1

    def train(
        self,
        X_train: Array,
        f_fn: Callable,
        g_fn: Callable,
        *,
        # Clustering params
        TOL: float = 1e-2,
        NUM_INITIAL_CLUSTERS: int = 100,
        MIN_PTS_PER_CLUSTER: int = 100,
        MAX_CLUSTERS_EST: int = 1_000_000,
        SYNTH_BUDGET_FACTOR: float = 10.0,
        SEED: int = 0,
        USE_HESSIAN_INFORMED_SPLIT: bool = False,
        ERROR_SPLIT_CRITERION: Literal["L2", "Linf"] = "L2",
    ):
        """
        1. Sample X from domain.
        2. Run GPU Adaptive Clustering.
        3. Store Reference Data.
        """
        t0 = time.perf_counter()

        if self.version == "v2":
            # 1. Adaptive Clustering
            results = adaptive_clustering(
                X=X_train,
                f_fn=f_fn,
                g_fn=g_fn,
                TOL=TOL,
                NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
                MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
                SEED=SEED,
                SYNTH_BUDGET_FACTOR=SYNTH_BUDGET_FACTOR,
                LOG_MEM_STATS=True,
                ERROR_METRIC="Linf-local",
            )

            centroids = xp.asarray(results["centroids"])
            f_vals = xp.asarray(results["f"])
            grads = xp.asarray(results["g"])
            radii = xp.asarray(results["radii"])
            num_levels = results.get("num_levels", -1)

            # 2. Store Data
            self.X_src = centroids
            self.f_src = f_vals
            self.J_src = grads
            self.trust_radii = radii
            self.obb_rotation_matrices = results["obb_rotation_matrices"]
            self.obb_extents = results["obb_extents"]
            self.N_src = centroids.shape[0]
            self.training_num_levels = num_levels
        elif self.version == "v1":
            if not IS_CUPY_BACKEND:
                clusters, num_levels = adaptive_cluster_algorithm(
                    X_train=X_train,
                    f=f_fn,
                    grad_f=g_fn,
                    cupy_f=f_fn,
                    tol=TOL,
                    num_initial_clusters=NUM_INITIAL_CLUSTERS,
                    min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
                    seed=SEED,
                    use_hessian_informed_split=USE_HESSIAN_INFORMED_SPLIT,
                    error_split_criterion=ERROR_SPLIT_CRITERION,
                )
                centroids = xp.asarray([clust.centroid for clust in clusters])
                f_vals = xp.asarray([clust.F_centroid for clust in clusters])
                grads = xp.asarray([clust.J_centroid for clust in clusters])
                trust_radii = xp.asarray([clust.radius for clust in clusters])
            else:
                result = adaptive_cluster_algorithm_gpu(
                    X_train=X_train,
                    f=f_fn,
                    grad=g_fn,
                    DATA_DIM=X_train.shape[1],
                    F_DIM=f_fn(X_train[:1]).shape[1],
                    NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
                    MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                    MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
                    ATOL=TOL,
                    RTOL=TOL,
                    SEED=SEED,
                )
                centroids = result[0]
                f_vals = result[1]
                grads = result[2]
                trust_radii = result[4]
                num_levels = result[9]

            self.X_src = centroids
            self.f_src = f_vals
            self.J_src = grads
            self.trust_radii = trust_radii
            self.N_src = centroids.shape[0]
            self.training_num_levels = num_levels
        else:
            raise ValueError(f"Unknown version: {self.version}")

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
        k_NEIGHBOURS: int = 1,
    ) -> tuple:
        """
        Inference via GPU kNN with L2 Distance.
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")

        if self.version == "v2":
            t_search_start = time.perf_counter()
            predictions, unguaranteed_mask = self._predict_obb(
                query_points, k=k_NEIGHBOURS
            )
            logger.info(
                f"OBB kNN Search + Predict Time: {time.perf_counter()-t_search_start:.4f}s"
            )
            return predictions, unguaranteed_mask

        qp = xp.asarray(query_points)
        t_search = time.perf_counter()
        nn_idx, nn_dist2, ungauranteed_mask = knn_l2_distance(
            qp,
            self.X_src,
            self.trust_radii,
            k_NEIGHBOURS,
        )
        logger.info(f"L2 kNN Search Time: {time.perf_counter()-t_search:.4f}s")

        N, D = qp.shape
        F = self.f_src.shape[1] if len(self.f_src.shape) > 1 else 1  # type: ignore

        # Flatten indices for gathers
        flat_idx = nn_idx.reshape(-1)  # (N*k,)
        X_neighbors = xp.asarray(self.X_src)[flat_idx]  # (N*k, D)
        f_neighbors = xp.asarray(self.f_src)[flat_idx]  # (N*k, F) or (N*k,)
        J_neighbors = xp.asarray(self.J_src)[flat_idx]  # (N*k, F, D)

        X_neighbors = X_neighbors.reshape(N, k_NEIGHBOURS, D)
        f_neighbors = f_neighbors.reshape(N, k_NEIGHBOURS, F)
        J_neighbors = J_neighbors.reshape(N, k_NEIGHBOURS, F, D)

        # ---------------------------------------------------------------------
        # First-order Taylor predictions around each neighbor centroid
        # ---------------------------------------------------------------------
        qp_expanded = qp[:, None, :]  # (N, 1, D)
        dx = qp_expanded - X_neighbors  # (N, k, D)

        # linear_term[n,k,f] = sum_d J[n,k,f,d] * dx[n,k,d]
        linear_term = xp.einsum("nkfd,nkd->nkf", J_neighbors, dx)  # (N, k, F)
        predictions_k = f_neighbors + linear_term  # (N, k, F)
        predictions = xp.mean(predictions_k, axis=1)  # (N, F)
        return predictions, ungauranteed_mask

    def _predict_obb(self, query_points, k=1):
        if not self.trained:
            raise RuntimeError("Model not trained.")

        qp = xp.asarray(query_points)

        # 1. Standard L2 kNN
        nn_idx, nn_dist2, _ = knn_l2_distance(qp, self.X_src, self.trust_radii, k)

        # 2. Retrieve OBB info
        Obb_rotation_matrices = self.obb_rotation_matrices[nn_idx]
        obb_extents = self.obb_extents[nn_idx]
        centroids = self.X_src[nn_idx]

        # 3. Rotate & Normalize
        dx = qp[:, None, :] - centroids
        z = xp.einsum("nk...i,nk...ij->nk...j", dx, Obb_rotation_matrices)

        # 4. Box Check (Compute "Box Distance")
        # normalized_dist_vector: (N, k, D) -- distance along each axis relative to extent
        INFLATION_FACTOR = 1.1
        box_radius = obb_extents * INFLATION_FACTOR

        normalized_dist_vector = xp.abs(z) / box_radius

        # The "distance" to the box center is the L-inf norm of the normalized vector
        # 0.0 = Center, 1.0 = Boundary
        box_dist = xp.max(normalized_dist_vector, axis=2)  # (N, k)

        valid_mask = box_dist <= 1.0

        # ---------------------------------------------------------------------
        # 5. Compute Trust Weights (NEW)
        # ---------------------------------------------------------------------
        # We use a quadratic kernel: (1 - d)^2
        # This gives high weight to points near the centroid, and decays to 0 at the boundary.
        # We perform the operation only on valid points (masking handles the rest)

        raw_weights = xp.maximum(0.0, 1.0 - box_dist) ** 2  # (N, k)

        # Apply strict validity mask (zero out invalid neighbors)
        final_weights = raw_weights * valid_mask

        # 6. Gather & Predict
        N, D = qp.shape
        F = self.f_src.shape[1] if self.f_src.ndim > 1 else 1
        flat_idx = nn_idx.reshape(-1)

        f_neighbors = xp.asarray(self.f_src)[flat_idx].reshape(N, k, F)
        J_neighbors = xp.asarray(self.J_src)[flat_idx].reshape(N, k, F, D)

        linear_term = xp.einsum("nkfd,nkd->nkf", J_neighbors, dx)
        predictions_k = f_neighbors + linear_term

        # 7. Weighted Averaging
        # Broadcast weights: (N, k) -> (N, k, 1) for F-dimension
        weights_expanded = final_weights[:, :, None]

        weighted_preds = predictions_k * weights_expanded
        total_weight = xp.sum(weights_expanded, axis=1)  # (N, 1)

        # Handle case where total_weight is 0 (all neighbors invalid)
        # We add TINY_EPS to denominator to prevent NaN, then mask later
        predictions = xp.sum(weighted_preds, axis=1) / (total_weight + 1e-12)

        # If total weight is effectively 0, the point is unguaranteed
        unguaranteed_mask = total_weight.squeeze(-1) < 1e-12

        return predictions, unguaranteed_mask


if __name__ == "__main__":
    import argparse
    from functools import partial

    import matplotlib.pyplot as plt

    from src import FIGURES_DIR
    from src.logging_config import setup_logging
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
    K_NEIGHBOURS = args.k_neighbours
    SEED = args.seed
    A_ROSENBROCK = args.rosenbrock_a
    FORCE_RECOMPUTE = args.force

    # Creat partial Rosenbrock functions
    rosenbrock = partial(rosenbrock, a=A_ROSENBROCK)
    rosenbrock_grad = partial(rosenbrock_grad, a=A_ROSENBROCK)

    img_fpath = (
        SURR_EXP_DIR
        / f"rosen_nearode_surrogate_rel_error_histogram_rosen{A_ROSENBROCK}_dim{dim}_tol{TOL:.1e}_Ntrain{N_train}_Ntest{N_test}_k{K_NEIGHBOURS}.png"
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
        NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
        MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
        MAX_CLUSTERS_EST=50_000,
        SEED=SEED,
        SYNTH_BUDGET_FACTOR=100,
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
        k_NEIGHBOURS=K_NEIGHBOURS,
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
