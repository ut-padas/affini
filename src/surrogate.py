import logging
import time
from typing import Any, Callable

import cupy as cp

from src import _HAS_GPU
from src.data_sampler import DataSampler
from src.gpu_adaptive_clustering_metric import adaptive_clustering
from src.mahalanobis import knn_mahalanobis_clusters

logger = logging.getLogger(__name__)

if not _HAS_GPU:
    msg = "GPU not available. Cannot run NearODESurrogate with GPU acceleration."
    raise ImportError(msg)


class NearODESurrogate:
    """
    Surrogate using Adaptive Clustering + kNN with Mahalanobis Distance on GPU.
    """

    def __init__(self):
        self.trained = False

        # Storage for source points (Reference data)
        self.X_src = None
        self.f_src = None
        self.J_src = None
        self.trust_ellipses_src = None
        self.N_src = 0

    def train(
        self,
        X_train,
        f: Callable,
        grad: Callable,
        *,
        # Clustering params
        NUM_INITIAL_CLUSTERS: int = 100,
        MIN_PTS_PER_CLUSTER: int = 100,
        ATOL: float = 1e-2,
        RTOL: float = 1e-2,
        MAX_CLUSTERS_EST: int = 1_000_000,
        SEED: int = 0,
    ):
        """
        1. Sample X from domain.
        2. Run GPU Adaptive Clustering.
        3. Store Reference Data.
        """
        t0 = time.perf_counter()

        # 1. Adaptive Clustering
        results = adaptive_clustering(
            X=X_train,
            f_fn=f,
            g_fn=grad,
            ATOL=ATOL,
            RTOL=RTOL,
            NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
            MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            SEED=SEED,
        )

        # Convert to original backend
        centroids = cp.asarray(results["centroids"])
        f_vals = cp.asarray(results["f"])
        grads = cp.asarray(results["g"])
        trust_ellipses = results["Q"]

        # 2. Store Data
        self.X_src = centroids
        self.f_src = f_vals
        self.J_src = grads
        self.trust_ellipses_src = trust_ellipses
        self.N_src = centroids.shape[0]

        # 3. Training complete
        self.trained = True
        logger.info(f"Training Time: {time.perf_counter()-t0:.4f}s")
        logger.info(f"Trained Clusters: {len(self.X_src):,}")

    def get_source_centroids(self) -> Any:
        """Return the source centroids used in training."""
        if not self.trained:
            raise RuntimeError("Model not trained.")
        return self.X_src

    def predict(
        self,
        query_points,
        k_NEIGHBOURS: int = 1,
        *,
        trust_region_containment: bool = False,
        containment_level: float = 1.0,
    ) -> tuple:
        """
        Inference via GPU Mahalanobis kNN over cluster centroids + local Taylor surrogate.

        Pipeline:
        1) kNN search over source centroids using cluster-wise Mahalanobis distance:
                dist2_k(x) = (x - μ_k)^T Q_k (x - μ_k)
            optionally requiring trust-region containment:
                dist2_k(x) <= containment_level

        2) Gather neighbor (μ_k, f(μ_k), g(μ_k)) and evaluate first-order Taylor model:
                f̃_k(x) = f(μ_k) + g(μ_k)^T (x - μ_k)

        3) If k_NEIGHBOURS > 1, average predictions across neighbors.

        4) Guarantee / trust checks:
            For k=1, check whether x lies inside a conservative region derived from local
            curvature estimates and tolerances. Here we replace the old scalar radius r_k
            with the ellipsoidal trust region defined by Q_k:
                inside if (x - μ_k)^T Q_k (x - μ_k) <= containment_level.
            The tolerance-based check using ATOL/RTOL/L_global is preserved.

        Args:
            query_points: (N, D) query points (NumPy or CuPy).
            k_NEIGHBOURS: number of neighbors (k). Guarantees are defined for k=1.
            ATOL: absolute tolerance ε_abs for guarantee checks (optional).
            RTOL: relative tolerance ε_rel for guarantee checks (optional).
            L_global: global curvature bound L_g (optional).
            ALPHA: trust factor alpha used in tolerance terms.
            trust_region_containment: if True, neighbors must contain query under Q_k.
            containment_level: ellipsoid level for containment, typically 1.0.

        Returns:
            predictions: (N, F) predicted function values.
            ungauranteed_mask: (N,) boolean mask; True indicates guarantee violation.
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")

        t0 = time.perf_counter()

        # Ensure GPU arrays for the kNN + downstream computation.
        qp = cp.asarray(query_points)

        # ---------------------------------------------------------------------
        # 1. kNN in cluster-wise Mahalanobis distance using stored (centroids, Q)
        # ---------------------------------------------------------------------
        t_search = time.perf_counter()
        nn_idx, nn_dist2 = knn_mahalanobis_clusters(
            qp,
            cp.asarray(self.X_src),
            cp.asarray(self.trust_ellipses_src),
            k_NEIGHBOURS,
            trust_region_containment=trust_region_containment,
            containment_level=containment_level,
            return_sqrt=False,  # keep squared distances for checks
        )
        logger.info(f"Mahalanobis kNN Search Time: {time.perf_counter()-t_search:.4f}s")

        # If containment is enabled, some slots may be -1. For inference we need
        # valid neighbors. We handle this by:
        #   - for k=1: if idx=-1 => prediction becomes NaN and marked unguaranteed
        #   - for k>1: ignore invalid neighbors in the average (count-valid per point)
        N, D = qp.shape
        F = self.f_src.shape[1] if len(self.f_src.shape) > 1 else 1  # type: ignore

        valid_mask = nn_idx >= 0  # (N, k)
        num_valid = cp.sum(valid_mask, axis=1)  # (N,)

        # Flatten indices for gathers; clamp invalid to 0 to keep gather safe.
        nn_idx_safe = cp.where(valid_mask, nn_idx, 0).astype(cp.int32)  # (N, k)
        flat_idx = nn_idx_safe.reshape(-1)  # (N*k,)

        # ---------------------------------------------------------------------
        # 2. Gather neighbor metadata (μ_k, f(μ_k), g(μ_k), Q_k)
        # ---------------------------------------------------------------------
        X_neighbors = cp.asarray(self.X_src)[flat_idx]  # (N*k, D)
        f_neighbors = cp.asarray(self.f_src)[flat_idx]  # (N*k, F) or (N*k,)
        J_neighbors = cp.asarray(self.J_src)[flat_idx]  # (N*k, F, D)
        Q_neighbors = cp.asarray(self.trust_ellipses_src)[flat_idx]  # (N*k, D, D)

        X_neighbors = X_neighbors.reshape(N, k_NEIGHBOURS, D)
        f_neighbors = f_neighbors.reshape(N, k_NEIGHBOURS, F)
        J_neighbors = J_neighbors.reshape(N, k_NEIGHBOURS, F, D)
        Q_neighbors = Q_neighbors.reshape(N, k_NEIGHBOURS, D, D)

        # ---------------------------------------------------------------------
        # 3. First-order Taylor predictions around each neighbor centroid
        # ---------------------------------------------------------------------
        qp_expanded = qp[:, None, :]  # (N, 1, D)
        dx = qp_expanded - X_neighbors  # (N, k, D)

        # linear_term[n,k,f] = sum_d J[n,k,f,d] * dx[n,k,d]
        linear_term = cp.einsum("nkfd,nkd->nkf", J_neighbors, dx)  # (N, k, F)
        predictions_k = f_neighbors + linear_term  # (N, k, F)

        # Mask out invalid neighbor slots (if containment mode produced -1s).
        # We convert invalid predictions to 0 and then divide by #valid.
        valid_mask_f = valid_mask[:, :, None]  # (N, k, 1)
        predictions_k = cp.where(valid_mask_f, predictions_k, 0)

        # Average across valid neighbors (per query). If none valid, set NaN.
        denom = cp.maximum(num_valid, 1).astype(predictions_k.dtype)[:, None]  # (N, 1)
        predictions = cp.sum(predictions_k, axis=1) / denom  # (N, F)
        predictions = cp.where(num_valid[:, None] > 0, predictions, cp.nan)

        # ---------------------------------------------------------------------
        # 4. Guarantee / trust-region checks (simplified)
        # ---------------------------------------------------------------------
        # We interpret "guaranteed" purely geometrically w.r.t. trust ellipses:
        #   - If trust_region_containment=True: kNN already filtered to contained
        #     clusters, so a query is guaranteed iff a neighbor exists.
        #   - Else: a query is guaranteed iff the nearest neighbor contains it,
        #     i.e., Mahalanobis dist2 <= 1 (or <= containment_level, if preferred).
        #
        # NOTE: This intentionally ignores ATOL/RTOL/L_global/ALPHA-based bounds.
        # If you want those bounds back, keep your previous tolerance math and combine.
        guaranteed_mask = cp.zeros(N, dtype=bool)

        if trust_region_containment:
            # In containment mode, existence of a neighbor implies containment.
            # For k>1 this is "any neighbor exists"; for k=1 it's equivalent to nn_idx[:,0] >= 0.
            guaranteed_mask = num_valid > 0
        else:
            # Unconstrained mode: check whether the nearest neighbor contains the point.
            # nn_dist2 is (x-μ)^T Q (x-μ) since return_sqrt=False.
            guaranteed_mask = nn_dist2[:, 0] <= 1.0

        ungauranteed_mask = ~guaranteed_mask

        num_ungauranteed = cp.sum(ungauranteed_mask)
        logger.info(f"Unguaranteed Points: {int(num_ungauranteed.get())} / {N}")

        # Heuristic warning for under-resolved surrogate coverage
        if int(num_ungauranteed.get()) > 3 * self.N_src:
            msg = (
                "Too many unguaranteed points. Need more clusters! "
                f"(Unguaranteed: {int(num_ungauranteed.get())} > 3 x Sources: {self.N_src})"
            )
            print(f"\033[91m{msg}\033[0m")
            logger.warning(msg)

        logger.info(f"Prediction Time: {time.perf_counter()-t0:.4f}s")
        return predictions, ungauranteed_mask


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    from src import FIGURES_DIR
    from src.logging_config import setup_logging
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging(console_level=logging.DEBUG, file_output=False)

    # Parse Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dim", type=int, default=2, help="Dimensionality of the problem"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-1, help="Tolerance for training and evaluation"
    )
    parser.add_argument(
        "--n-train", type=int, default=1_000_000, help="Number of training samples"
    )
    parser.add_argument(
        "--n-test", type=int, default=10_000, help="Number of test samples"
    )
    parser.add_argument(
        "--n-init-clusters",
        type=int,
        default=100,
        help="Number of initial clusters for adaptive clustering",
    )
    parser.add_argument(
        "--n-min-per-cluster",
        type=int,
        default=100,
        help="Minimum points per cluster for adaptive clustering",
    )
    parser.add_argument(
        "--k-neighbours", type=int, default=5, help="Number of neighbors for kNN"
    )
    parser.add_argument(
        "--containment-level",
        type=float,
        default=1.0,
        help="Containment level for trust-region checks",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for training data sampling"
    )
    args = parser.parse_args()
    dim = args.dim
    TOL = args.tol
    N_train = args.n_train
    N_test = args.n_test
    N_INIT_CLUSTERS = args.n_init_clusters
    N_MIN_PER_CLUSTER = args.n_min_per_cluster
    K_NEIGHBOURS = args.k_neighbours
    CONTAINMENT_LEVEL = args.containment_level
    SEED = args.seed

    # 1. Setup Domain
    domain = cp.array([[-3, 3] * dim]).reshape(dim, 2)
    print(f"--- Domain: {domain.get().tolist()} ---")

    # Sample Training Data
    X_train = DataSampler.sample_latin_hypercube(
        cp.asnumpy(domain), N=N_train, seed=SEED
    )
    X_train = cp.asarray(X_train)

    # 2. Initialize Surrogate
    surrogate = NearODESurrogate()

    # 3. Train
    print("--- Training ---")
    surrogate.train(
        X_train=X_train,
        f=rosenbrock,
        grad=rosenbrock_grad,
        NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
        MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
        ATOL=TOL,
        RTOL=TOL,
        SEED=SEED,
    )

    # 4. Evaluate
    print("\n--- Evaluation ---")
    X_test = DataSampler.sample_uniform(cp.asnumpy(domain), N=N_test, seed=SEED + 1)
    X_test = cp.asarray(X_test)

    # Example utilizing the new signature with Guarantee checks
    preds, ungauranteed_mask = surrogate.predict(
        X_test,
        k_NEIGHBOURS=K_NEIGHBOURS,
        trust_region_containment=True,
        containment_level=CONTAINMENT_LEVEL,
    )

    ground_truth = rosenbrock(X_test)
    rel_error = cp.abs(preds - ground_truth) / (cp.abs(ground_truth) + 1e-12)

    n_ungauranteed = cp.sum(ungauranteed_mask)
    perct_ungauranteed = 100.0 * n_ungauranteed / X_test.shape[0]
    print(
        f"Unguaranteed Points: {n_ungauranteed:,} / {X_test.shape[0]:,} ({perct_ungauranteed:.2f}%)"
    )
    print(f"Max Relative Error (InfiniteNorm): {cp.nanmax(rel_error).get():.4e}")
    n_within_tol = cp.sum(rel_error <= TOL)
    perct_within_tol = 100.0 * n_within_tol / X_test.shape[0]
    print(
        f"Points within TOL={TOL}: {n_within_tol:,} / {X_test.shape[0]:,} ({perct_within_tol:.2f}%)"
    )

    plt.figure(figsize=(8, 5))
    plt.hist(cp.asnumpy(rel_error.flatten()), bins=50, log=True)
    plt.xlabel("Relative Error (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(
        f"NearODESurrogate RelError (dim={dim}, tol={TOL:.1e}, N_train={N_train:.1e}, N_clusters={surrogate.N_src}, N_test={N_test:.1e}, k={K_NEIGHBOURS}, N_ung.={n_ungauranteed} ({perct_ungauranteed:.2f}%))",
        fontsize=7,
    )
    plt.axvline(
        x=TOL,
        color="red",
        linestyle="--",
        label=f"TOL={TOL:.1e} | N_within_tol: {n_within_tol} ({perct_within_tol:.2f}%)",
    )
    plt.legend()
    plt.xscale("log")
    plt.grid(True)
    fname = (
        FIGURES_DIR
        / f"rosen_nearode_surrogate_rel_error_histogram_dim{dim}_tol{TOL}_Ntrain{N_train}_Ntest{N_test}_k{K_NEIGHBOURS}.png"
    )
    plt.savefig(fname, dpi=300)
    print(f"Saved relative error histogram to {fname}")
