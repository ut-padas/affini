import logging
import time
from typing import Any, Callable, Optional

import cupy as cp
import faiss
import numpy as np
from array_api_compat import array_namespace

from src.adaptive_clustering import _HAS_GPU, adaptive_cluster_algorithm
from src.data_sampler import DataSampler
from src.gpu_adaptive_clustering import adaptive_cluster_algorithm_gpu

logger = logging.getLogger(__name__)


class NearODESurrogate:
    """
    Combined kNN + Taylor Expansion Surrogate.
    """

    def __init__(self, data_dim: int, f_dim: int):
        self.D = data_dim
        self.F = f_dim
        self.trained = False

        # Storage for source points (Reference data)
        # We keep X_src as the reference for indices
        self.X_src = None
        self.f_src = None
        self.J_src = None
        self.H_src = None
        self.R_src = None
        self.N_src = 0

        # FAISS Index
        self.index = None

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
        RTOL: Optional[float] = None,
        MAX_DEPTH: int = 50,
        MAX_CLUSTERS_EST: int = 1_000_000,
        SEED: int = 0,
    ):
        """
        1. Sample X from domain (NumPy or CuPy).
        2. Run GPU Adaptive Clustering (Forces CuPy).
        3. Store Reference Data (NumPy for FAISS).
        """
        xp = array_namespace(X_train)
        t0 = time.perf_counter()

        # 1. Adaptive Clustering
        if _HAS_GPU:
            (
                centroids,  # (M, D)
                f_vals,  # (M, F)
                grads,  # (M, F, D)
                hess_proxies,  # (M,)
                radii,  # (M,)
                _,
                _,
                _,
                _,  # Ignore debug stats
            ) = adaptive_cluster_algorithm_gpu(
                X_train=xp.asarray(X_train),
                f=f,
                grad=grad,
                DATA_DIM=self.D,
                F_DIM=self.F,
                NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
                MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
                ATOL=ATOL,
                RTOL=RTOL,
                MAX_DEPTH=MAX_DEPTH,
                MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
                SEED=SEED,
                MOVE_INTERMEDIATE_TO_HOST=True,
                STORE_FINAL_ARRAYS_ON_HOST=True,
                LOG_MEM_STATS=True,
            )
        else:
            # Ensure X_train is on CPU for the clustering algorithm
            X_train_cpu = np.asarray(X_train)
            clusters = adaptive_cluster_algorithm(
                X_train=X_train_cpu,
                f=f,
                grad_f=grad,
                cupy_f=f,
                atol=ATOL,
                rtol=RTOL,
                num_initial_clusters=NUM_INITIAL_CLUSTERS,
                min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
                seed=SEED,
                should_generate_extra_points=True,
                two_means_split_method="median",
            )
            centroids = np.array([c.centroid for c in clusters])
            f_vals = np.array([c.F_centroid for c in clusters])
            grads = np.array([c.J_centroid for c in clusters])
            hess_proxies = np.array([c.hessian_proxy for c in clusters])
            radii = np.array([c.radius for c in clusters])

            # Make centroids (N, 1) with np.newaxis
            if self.D == 1:
                centroids = np.expand_dims(centroids, axis=1)
                f_vals = np.expand_dims(f_vals, axis=1)
                grads = np.expand_dims(grads, axis=1)

        # Convert to original backend
        centroids = xp.asarray(centroids)
        f_vals = xp.asarray(f_vals)
        grads = xp.asarray(grads)
        hess_proxies = xp.asarray(hess_proxies)
        radii = xp.asarray(radii)

        # 2. Store Data
        self.X_src = centroids
        self.f_src = f_vals
        self.J_src = grads
        self.H_src = hess_proxies
        self.R_src = radii
        self.N_src = centroids.shape[0]

        # 3. Build Index
        if xp.__name__ == "array_api_compat.cupy":
            # Move centroids to CPU for FAISS
            centroids = xp.asnumpy(centroids)
        self.index = faiss.IndexFlatL2(self.D)
        self.index.add(centroids)  # type: ignore

        # 4. Training complete
        self.trained = True
        logger.info(f"Training Time: {time.perf_counter()-t0:.4f}s")
        logger.info(f"Trained Clusters: {len(self.X_src)}")

    def get_source_centroids(self) -> Any:
        """Return the source centroids used in training."""
        if not self.trained:
            raise RuntimeError("Model not trained.")
        return self.X_src

    def predict(
        self,
        targets,
        k_NEIGHBOURS: int = 1,
        ATOL: Optional[float] = None,
        RTOL: Optional[float] = None,
        L_global: Optional[float] = None,
        ALPHA: float = 1.0,
    ) -> tuple:
        """
        Inference: kNN Search -> Taylor Expansion -> Trust Region Check.
        Matches Algorithm 3: Nearest Neighbors Inference with Local Error Bound.

        Args:
            targets: Input points (N, DATA_DIM)
            k_NEIGHBOURS: Number of neighbors (k). Note: Algorithm guarantees are defined for k=1.
            ATOL: eps_abs (Absolute error tolerance)
            RTOL: eps_rel (Relative error tolerance)
            L_global: L_g (Global Hessian approximation)
            ALPHA: alpha (Trust factor)

        Returns:
            predictions: (N, F_DIM) predicted function values (f_tilde)
            ungauranteed_mask: (N,) boolean, True if point violates guarantees.
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")

        xp = array_namespace(targets)
        t0 = time.perf_counter()

        # 1. FAISS Search (CPU)
        # Convert targets to NumPy safely for search
        if xp.__name__ == "array_api_compat.cupy":
            targets_np = cp.asnumpy(targets)
        else:
            targets_np = np.asarray(targets)

        # Get distances and indices
        # Line 2: k <- arg min ||x - xi||_2
        # FAISS returns squared L2 distances
        distances_sq, indices = self.index.search(targets_np, k_NEIGHBOURS)  # type: ignore

        # Convert back to original backend
        distances_sq = xp.asarray(distances_sq)
        indices = xp.asarray(indices)

        # 2. Gather Neighbors
        flat_idx = indices.flatten()

        def gather(src_array, idx):
            gathered = src_array[idx]
            return gathered

        N = len(targets)
        X_neighbors = gather(self.X_src, flat_idx)  # (N*k, DATA_DIM)
        f_neighbors = gather(self.f_src, flat_idx)  # (N*k, F_DIM)
        J_neighbors = gather(self.J_src, flat_idx)  # (N*k, F_DIM, DATA_DIM)

        X_neighbors = X_neighbors.reshape(N, k_NEIGHBOURS, self.D)  # (N, k, DATA_DIM)
        f_neighbors = f_neighbors.reshape(N, k_NEIGHBOURS, self.F)  # (N, k, F_DIM)
        J_neighbors = J_neighbors.reshape(
            N, k_NEIGHBOURS, self.F, self.D
        )  # (N, k, F_DIM, DATA_DIM)

        # 3. Taylor Expansion
        targets_expanded = targets[:, None, :]  # (N, 1, DATA_DIM)
        dx = targets_expanded - X_neighbors  # (N, k, DATA_DIM)

        # Einsum: n=N, k=k, f=F_DIM, d=DATA_DIM
        linear_term = xp.einsum("nkfd,nkd->nkf", J_neighbors, dx)

        predictions_k = f_neighbors + linear_term  # (N, k, F_DIM)

        # If k > 1, then average the predictions
        predictions_sum = xp.sum(predictions_k, axis=1)
        predictions = predictions_sum / k_NEIGHBOURS  # (N, F_DIM)

        # 4. Guarantee / Trust Region Checks
        # Initialize ungauranteed_mask to all False
        ungauranteed_mask = xp.zeros(N, dtype=bool)

        # The algorithm specifically relies on Nearest Neighbor properties (k=1)
        if (k_NEIGHBOURS == 1) and (ATOL is not None):
            idx_nearest = indices[:, 0]  # (N,)
            dist_sq_nearest = distances_sq[:, 0]  # (N,)

            # Calculate Euclidean distance ||x - x_k||
            dist_nearest = xp.sqrt(dist_sq_nearest)

            # Gather Metadata for nearest neighbor
            R_k = gather(self.R_src, idx_nearest)  # (N,) Cluster radii r_k
            L_k = gather(self.H_src, idx_nearest)  # (N,) Local Hessians L_hat_k

            # Denominator: alpha * L_hat_k
            denom = ALPHA * L_k
            denom = xp.where(denom == 0, 1e-12, denom)  # Safety for div/0

            # Calculate Terms for tolerance
            # Term 1: sqrt( 2 * eps_abs / (alpha * L_hat_k) )
            logger.debug("Calculating absolute error term for guarantees.")
            term_abs_local = xp.sqrt((2.0 * ATOL) / denom)

            # Term 2: sqrt( 2 * eps_rel * |f_tilde(x)| / (alpha * L_hat_k) )
            term_rel_local = xp.zeros_like(term_abs_local)
            if RTOL is not None:
                logger.debug("Calculating relative error term for guarantees.")
                # |f_tilde(x)|: L2 norm of the prediction vector
                pred_norm = xp.linalg.norm(predictions, ord=2, axis=1)
                term_rel_local = xp.sqrt((2.0 * RTOL * pred_norm) / denom)

            # Term 3: sqrt( 2 * eps_abs / L_g )
            term_abs_global = xp.zeros_like(term_abs_local)
            if L_global is not None:
                logger.debug("Calculating global error term for guarantees.")
                _term = xp.sqrt((2.0 * ATOL) / L_global)
                term_abs_global = xp.full_like(term_abs_local, _term)

            # tol <- max { Term 1, Term 2, Term 3 }
            tol = xp.maximum(term_abs_local, term_rel_local)
            tol = xp.maximum(tol, term_abs_global)

            # Check bounds
            # if ||x - x_k|| <= min { r_k, tol }
            bound = xp.minimum(R_k, tol)
            is_valid = dist_nearest <= bound
            ungauranteed_mask = xp.logical_not(is_valid)

            num_ungauranteed = xp.sum(ungauranteed_mask)
            logger.info(f"Unguaranteed Points: {num_ungauranteed} / {N}")

            # Heuristic warning for under-resolved surrogate coverage
            if num_ungauranteed > 3 * self.N_src:
                msg = "Too many ungauranteed points. Need more clusters! "
                msg += f"(Ungauranteed: {num_ungauranteed} > 3 x Sources: {self.N_src})"
                # Print in red to console
                print(f"\033[91m{msg}\033[0m")
                logger.warning(msg)

        logger.info(f"Evaluation Time: {time.perf_counter()-t0:.4f}s")
        return predictions, ungauranteed_mask


if __name__ == "__main__":
    from src.logging_config import setup_logging
    from src.problems.six_hump_camelback import (
        six_hump_camelback,
        six_hump_camelback_grad,
    )

    setup_logging(console_level=logging.INFO)

    if _HAS_GPU:
        xp = cp
    else:
        xp = np

    # 1. Setup Domain
    domain = xp.array([[-2.0, 2.0], [-1.0, 1.0]])
    xp = array_namespace(domain)

    # Sample Training Data
    N_train = 100_000
    X_train = DataSampler.sample_latin_hypercube(domain, N=N_train, seed=0)

    # 2. Initialize Surrogate
    surrogate = NearODESurrogate(data_dim=2, f_dim=1)

    ATOL = 1e-2
    RTOL = 1e-2

    # 3. Train
    print("--- Training ---")
    surrogate.train(
        X_train=X_train,
        f=six_hump_camelback,
        grad=six_hump_camelback_grad,
        NUM_INITIAL_CLUSTERS=100,
        MIN_PTS_PER_CLUSTER=100,
        ATOL=ATOL,
        RTOL=RTOL,
        SEED=42,
    )

    # 4. Evaluate
    print("\n--- Evaluation ---")
    X_test = DataSampler.sample_uniform(domain, N=500, seed=10)

    # Example utilizing the new signature with Guarantee checks
    preds, ungauranteed_mask = surrogate.predict(
        X_test,
        k_NEIGHBOURS=1,
        ATOL=ATOL,
        RTOL=RTOL,
        L_global=10.0,
        ALPHA=1.0,
    )

    ground_truth = six_hump_camelback(X_test)
    err = xp.abs(preds - ground_truth)

    print(f"Unguaranteed Points: {xp.sum(ungauranteed_mask)} / {X_test.shape[0]}")
    print(f"Max Error: {xp.max(err)}")
