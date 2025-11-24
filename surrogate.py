import logging
import time
from typing import Callable

import cupy as cp
import faiss
import numpy as np
from array_api_compat import array_namespace
from scipy.stats import qmc

from src.gpu_adaptive_clustering import adaptive_cluster_algorithm_gpu

logger = logging.getLogger(__name__)


class DataSampler:
    """Helper to generate training data X in a backend-agnostic way."""

    @staticmethod
    def sample_uniform(domain, N: int, seed: int = 0):
        """
        Sample N points uniformly from domain.
        Backend matches domain's backend.
        """
        logger.debug("Sampling %d points uniformly from domain.", N)
        xp = array_namespace(domain)
        rng = xp.random.default_rng(seed)

        # Parse domain
        if domain.ndim == 1:
            l, r = domain[0], domain[1]
            n_dim = 1
        else:
            l = domain[:, 0]
            r = domain[:, 1]
            n_dim = len(domain)

        # Generate (N, D)
        X = rng.uniform(l, r, (N, n_dim))
        return xp.astype(X, xp.float64)

    @staticmethod
    def sample_latin_hypercube(domain, N: int, seed: int = 0):
        """
        Sample N points using Latin Hypercube Sampling.
        Uses Scipy (CPU) internally, then moves to domain's backend.
        """
        logger.debug("Sampling %d points using Latin Hypercube from domain.", N)

        xp = array_namespace(domain)

        # Parse domain dimensions
        if domain.ndim == 1:
            n_dim = 1
            l_np = xp.array([float(domain[0])])
            r_np = xp.array([float(domain[1])])
        else:
            l_np = xp.asnumpy(domain[:, 0])
            r_np = xp.asnumpy(domain[:, 1])
            n_dim = len(domain)

        # Generate LHS on CPU
        sampler = qmc.LatinHypercube(d=n_dim)
        sample = sampler.random(n=N)

        # Scale to domain bounds
        sample_scaled = qmc.scale(sample, l_np, r_np)

        # Convert to target backend
        X = xp.asarray(sample_scaled, dtype=xp.float64)
        return X


class NearODESurrogate:
    """
    Combined kNN + Taylor Expansion Surrogate.
    """

    def __init__(self, data_dim: int, f_dim: int):
        self.D = data_dim
        self.F = f_dim
        self.trained = False

        # Storage for source points (Reference data)
        self.X_src = None
        self.f_src = None
        self.J_src = None
        self.H_src = None
        self.R_src = None

        # Training metadata
        self.atol = None
        self.index = None

    def train(
        self,
        X_train,
        f: Callable,
        grad: Callable,
        # Clustering params
        num_init_clusters: int = 100,
        min_pts: int = 10,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        max_depth: int = 50,
        max_clusters: int = 1_000_000,
        seed: int = 0,
    ):
        """
        1. Sample X from domain (NumPy or CuPy).
        2. Run GPU Adaptive Clustering (Forces CuPy).
        3. Store Reference Data (NumPy for FAISS).
        """
        # Persist atol for evaluate checks
        self.atol = atol

        t0 = time.perf_counter()

        # Ensure X_train is on GPU for the clustering algorithm
        X_train_gpu = cp.asarray(X_train)

        # 2. Cluster
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
            X_train=X_train_gpu,
            f=f,
            grad=grad,
            DATA_DIM=self.D,
            F_DIM=self.F,
            NUM_INITIAL_CLUSTERS=num_init_clusters,
            MIN_PTS_PER_CLUSTER=min_pts,
            ATOL=atol,
            RTOL=rtol,
            MAX_DEPTH=max_depth,
            MAX_CLUSTERS_EST=max_clusters,
            SEED=seed,
            MOVE_INTERMEDIATE_TO_HOST=True,
            STORE_FINAL_ARRAYS_ON_HOST=True,
        )

        # 3. Store Data (Host side for FAISS compatibility)
        self.X_src = centroids
        self.f_src = f_vals
        self.J_src = grads
        self.H_src = hess_proxies
        self.R_src = radii

        # 4. Build Index
        self.index = faiss.IndexFlatL2(self.D)
        self.index.add(self.X_src.astype(np.float32))  # type: ignore

        self.trained = True
        print(
            f"Training complete. Clusters: {len(self.X_src)}. Time: {time.perf_counter()-t0:.4f}s"
        )

    def evaluate(
        self,
        targets,
        k: int = 1,
        L_global: float | None = None,
        alpha: float = 1.0,
        check_trust_region: bool = True,
    ):
        """
        Inference: kNN Search -> Taylor Expansion.
        Supports both NumPy and CuPy targets via array_api_compat.
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")

        xp = array_namespace(targets)

        # Reliable check for CuPy array
        is_cupy = hasattr(targets, "device")

        # 1. FAISS Search (CPU)
        # Convert targets to NumPy safely
        if is_cupy:
            targets_np = cp.asnumpy(targets)
        else:
            targets_np = np.asarray(targets)

        targets_np = np.ascontiguousarray(targets_np, dtype=np.float32)

        distances_sq, indices = self.index.search(targets_np, k)  # type: ignore

        # 2. Gather Neighbors
        flat_idx = indices.flatten()

        def gather(src_array, idx):
            gathered = src_array[idx]
            # If input targets are CuPy, move neighbors to GPU for math
            if is_cupy:
                return cp.asarray(gathered)
            return gathered

        X_neighbors = gather(self.X_src, flat_idx)  # (N_q * k, D)
        f_neighbors = gather(self.f_src, flat_idx)  # (N_q * k, F)
        J_neighbors = gather(self.J_src, flat_idx)  # (N_q * k, F, D)

        n_q = len(targets)
        X_neighbors = X_neighbors.reshape(n_q, k, self.D)
        f_neighbors = f_neighbors.reshape(n_q, k, self.F)
        J_neighbors = J_neighbors.reshape(n_q, k, self.F, self.D)

        # 3. Taylor Expansion
        targets_exp = targets[:, None, :]  # (N_q, 1, D)
        dx = targets_exp - X_neighbors  # (N_q, k, D)

        # Linear Term: J * dx
        linear_term = xp.einsum("nkfd,nkd->nkf", J_neighbors, dx)

        predictions_k = f_neighbors + linear_term
        predictions = xp.mean(predictions_k, axis=1)

        # 4. Trust Region Check
        valid_mask = xp.ones(n_q, dtype=bool)

        if check_trust_region:
            # We strictly verify the Nearest Neighbor (k=1)

            idx_nearest = indices[:, 0]
            dist_sq_nearest = (
                xp.asarray(distances_sq[:, 0]) if is_cupy else distances_sq[:, 0]
            )

            R_nearest = gather(self.R_src, idx_nearest)  # (N_q,)
            H_nearest = gather(self.H_src, idx_nearest)  # (N_q,)

            # --- Local Bounds ---
            # 1. Geometric limit: Cluster radius squared
            limit_geom_sq = R_nearest**2

            # 2. Hessian limit: r^2 < 2*tol / (alpha * H)
            # Avoid div by zero
            hess_denom = alpha * H_nearest
            # If using cupy, we need to ensure the scalar logic works or use xp.where
            hess_denom = xp.where(hess_denom == 0, 1e-12, hess_denom)
            limit_hess_sq = (2.0 * self.atol) / hess_denom  # type: ignore

            local_limit_sq = xp.minimum(limit_geom_sq, limit_hess_sq)

            # --- Global Limit (Fallback) ---
            if L_global is not None:
                # Derived from L * r < 2 * tol => r < 2*tol/L
                # We square it because we are comparing against squared distance
                global_limit_sq = (2.0 * self.atol / L_global) ** 2  # type: ignore

                # Point is valid if inside Local OR inside Global
                valid_mask = xp.logical_or(
                    dist_sq_nearest < local_limit_sq,
                    xp.logical_and(
                        local_limit_sq < dist_sq_nearest,
                        dist_sq_nearest < global_limit_sq,
                    ),
                )
            else:
                valid_mask = dist_sq_nearest < local_limit_sq

        return predictions, valid_mask


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from src.logging_config import setup_logging
    from src.problems.six_hump_camelback import (
        six_hump_camelback,
        six_hump_camelback_grad,
    )

    setup_logging(console_level=logging.DEBUG)

    # 1. Setup Domain (Backend determines computation device)
    # Using CuPy for GPU acceleration
    domain = cp.array([[-2.0, 2.0], [-1.0, 1.0]])

    # Sample Training Data
    N_train = 200_000
    X_train = DataSampler.sample_latin_hypercube(domain, N=N_train, seed=0)

    # 2. Initialize Surrogate
    surrogate = NearODESurrogate(data_dim=2, f_dim=1)

    # 3. Train
    print("--- Training ---")
    surrogate.train(
        X_train=X_train,
        f=six_hump_camelback,
        grad=six_hump_camelback_grad,
        num_init_clusters=50,
        atol=0.1,
        rtol=0.1,
        seed=42,
    )

    # 4. Evaluate
    print("\n--- Evaluation ---")
    # Create random test points on GPU
    X_test = DataSampler.sample_uniform(domain, N=5, seed=123)

    preds, valid = surrogate.evaluate(X_test, k=1, L_global=10.0, alpha=1.0)

    # Compare with ground truth
    ground_truth = six_hump_camelback(X_test)

    print(f"Test Points:\n{X_test}")
    print(f"Predictions:\n{preds}")
    print(f"Ground Truth:\n{ground_truth[:, 0]}")
    print(f"Valid (Trust Region): {valid}")

    err = cp.abs(preds - ground_truth[:, 0])
    print(f"Max Error: {cp.max(err)}")
