import logging
import time
from typing import Callable, Optional

import cupy as cp
import faiss
import numpy as np
from array_api_compat import array_namespace
from scipy.stats import qmc

from src.adaptive_clustering import _HAS_GPU, adaptive_cluster_algorithm
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
            l_np = np.asarray([float(domain[0])])
            r_np = np.asarray([float(domain[1])])
        else:
            # Convert domain bounds to NumPy for scipy
            l_np = np.asarray(domain[:, 0])
            r_np = np.asarray(domain[:, 1])
            n_dim = len(domain)

        # Generate LHS on CPU
        sampler = qmc.LatinHypercube(d=n_dim, rng=seed)
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
        # We keep X_src as the reference for indices
        self.X_src = None
        self.f_src = None
        self.J_src = None
        self.H_src = None
        self.R_src = None

        # FAISS Index
        self.index = None

    def train(
        self,
        X_train,
        f: Callable,
        grad: Callable,
        *,
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
        t0 = time.perf_counter()

        # 1. Adaptive Clustering
        if _HAS_GPU:
            # Ensure X_train is on GPU for the clustering algorithm
            X_train_gpu = cp.asarray(X_train)
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
        else:
            # Ensure X_train is on CPU for the clustering algorithm
            X_train_cpu = np.asarray(X_train)
            clusters = adaptive_cluster_algorithm(
                X_train=X_train_cpu,
                f=f,
                grad_f=grad,
                cupy_f=f,
                atol=atol,
                rtol=rtol,
                num_initial_clusters=num_init_clusters,
                min_pts_per_cluster=min_pts,
                seed=seed,
                should_generate_extra_points=True,
            )
            centroids = np.array([c.centroid for c in clusters])
            f_vals = np.array([c.F_centroid for c in clusters])
            grads = np.array([c.J_centroid for c in clusters])
            hess_proxies = np.array([c.hessian_proxy for c in clusters])
            radii = np.array([c.radius for c in clusters])

        # 2. Store Data (Host side for FAISS compatibility, but usable by CuPy via conversion)
        # Note: centroids coming out of the clustering with STORE_FINAL_ARRAYS_ON_HOST=True are NumPy
        self.X_src = centroids
        self.f_src = f_vals
        self.J_src = grads
        self.H_src = hess_proxies
        self.R_src = radii

        # 3. Build Index
        self.index = faiss.IndexFlatL2(self.D)
        self.index.add(self.X_src.astype(np.float32))  # type: ignore

        # 4. Training complete
        self.trained = True
        msg = f"Training complete. Clusters: {len(self.X_src)}. Time: {time.perf_counter()-t0:.4f}s"
        logger.info(msg)

    def evaluate(
        self,
        targets,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        k: int = 1,
        L_global: Optional[float] = None,
        alpha: float = 1.0,
    ) -> tuple:
        """
        Inference: kNN Search -> Taylor Expansion -> Trust Region Check.

        Args:
            targets: Input points (M, D)
            atol: Absolute tolerance for guarantee check (k=1 only)
            rtol: Relative tolerance for guarantee check (k=1 only)
            k: Neighbors to average
            L_global: Global Lipschitz constant (fallback check)
            alpha: Heuristic safety factor for Hessian bound

        Returns:
            predictions: (M, F)
            outside_mask: (M,) boolean, True if point violates guarantees.
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")

        t_start = time.perf_counter()
        xp = array_namespace(targets)
        is_cupy = xp.__name__ == "cupy"

        # 1. FAISS Search (CPU)
        # Convert targets to NumPy safely for search
        if is_cupy:
            targets_np = cp.asnumpy(targets)
        else:
            targets_np = np.asarray(targets)

        # Get distances and indices
        # distances_sq are squared Euclidean distances (L2^2)
        distances_sq, indices = self.index.search(targets_np, k)  # type: ignore

        # 2. Gather Neighbors
        flat_idx = indices.flatten()

        def gather(src_array, idx):
            gathered = src_array[idx]
            # If input targets are CuPy, move neighbors to GPU for math
            if is_cupy:
                return cp.asarray(gathered)
            return gathered

        X_neighbors = gather(self.X_src, flat_idx)  # (M*k, D)
        f_neighbors = gather(self.f_src, flat_idx)  # (M*k, F)
        J_neighbors = gather(self.J_src, flat_idx)  # (M*k, F, D)

        n_q = len(targets)
        X_neighbors = X_neighbors.reshape(n_q, k, self.D)
        f_neighbors = f_neighbors.reshape(n_q, k, self.F)
        J_neighbors = J_neighbors.reshape(n_q, k, self.F, self.D)

        # 3. Handle Unwrap Log (Linearizing space for Taylor expansion)
        # We need a copy of targets and X_neighbors to potentially modify them
        targets_calc = xp.asarray(targets) if not is_cupy else targets
        X_calc = X_neighbors

        # 4. Taylor Expansion
        # dx = x* - x_i (in the calculation space, potentially unlogged)
        targets_expanded = targets_calc[:, None, :]  # (M, 1, D)
        dx = targets_expanded - X_calc  # (M, k, D)

        # Linear Term: J * dx -> (M, k, F)
        # Einsum: n=M, k=k, f=F, d=D
        linear_term = xp.einsum("nkfd,nkd->nkf", J_neighbors, dx)

        predictions_k = f_neighbors + linear_term
        predictions_sum = xp.sum(predictions_k, axis=1)  # Sum over k
        predictions = predictions_sum / k

        # 5. Guarantee / Trust Region Checks
        # Strict checking is generally only mathematically sound for k=1 (nearest neighbor).
        outside_mask = xp.zeros(n_q, dtype=bool)  # Default: Valid (False = inside)
        if (atol is not None) and (k == 1):
            idx_nearest = indices[:, 0]
            # distances_sq is (M, k), we need nearest
            dist_sq_nearest = (
                xp.asarray(distances_sq[:, 0]) if is_cupy else distances_sq[:, 0]
            )

            # Gather Metadata for nearest neighbor
            R_nearest = gather(self.R_src, idx_nearest)  # (M,)
            H_nearest = gather(self.H_src, idx_nearest)  # (M,)

            # --- Local Bounds ---
            # 1. Geometric limit: Cluster radius squared (r^2)
            limit_geom_sq = xp.pow(R_nearest, 2.0)

            # 2. Hessian limit
            # Bound: r^2 <= 2 * tol / (alpha * H)
            denom = alpha * H_nearest
            denom = xp.where(denom == 0, 1e-12, denom)  # Avoid div/0

            # Absolute Component
            limit_hess_abs = (2.0 * atol) / denom

            # Relative Component (if rtol provided)
            if rtol is not None:
                # Norm of prediction vector (L2 norm along feature dim)
                pred_norm = xp.linalg.norm(predictions, ord=2, axis=1)
                limit_hess_rel = (2.0 * rtol * pred_norm) / denom
                # Take looser of abs/rel (Standard float comparison logic)
                limit_hess_final = xp.maximum(limit_hess_abs, limit_hess_rel)
            else:
                limit_hess_final = limit_hess_abs

            # Local Limit is strict: min(Geom, Hessian)
            local_limit_sq = xp.minimum(limit_geom_sq, limit_hess_final)

            # Check Local Validity
            is_valid_local = dist_sq_nearest < local_limit_sq

            # --- Global Limit (Fallback) ---
            if L_global is not None:
                # Bound: (L/2) * r^2 <= tol  =>  r^2 <= 2*tol / L
                # Note: NOT squared. FAISS returns d^2, so we compare against derived r^2.
                global_limit_sq = (2.0 * atol) / L_global

                # Logic: Valid if (Local OK) OR (Between Local and Global)
                # Effectively: Valid if (d^2 < Local) OR (d^2 < Global)
                # However, the snippet logic was explicit:
                # valid = local OR (local < dist < global)
                # This simplifies to valid = dist < max(local, global) IF local < global.
                # But strictly following snippet logic:
                is_valid_global = xp.logical_and(
                    local_limit_sq < dist_sq_nearest, dist_sq_nearest < global_limit_sq
                )
                is_good = xp.logical_or(is_valid_local, is_valid_global)
            else:
                is_good = is_valid_local

            # The return is "outside_mask" (True if BAD)
            outside_mask = xp.logical_not(is_good)

        logger.debug(f"Prediction time: {time.perf_counter() - t_start:.5f}s")

        return predictions, outside_mask


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

    # 1. Setup Domain
    domain = np.array([[-2.0, 2.0], [-1.0, 1.0]])
    xp = array_namespace(domain)

    # Sample Training Data
    N_train = 200_000
    X_train = DataSampler.sample_latin_hypercube(domain, N=N_train, seed=0)

    # 2. Initialize Surrogate
    surrogate = NearODESurrogate(data_dim=2, f_dim=1)

    ATOL = 1e-3
    RTOL = 1e-3

    # 3. Train
    print("--- Training ---")
    surrogate.train(
        X_train=X_train,
        f=six_hump_camelback,
        grad=six_hump_camelback_grad,
        num_init_clusters=50,
        atol=ATOL,
        rtol=RTOL,
        seed=42,
    )

    # 4. Evaluate
    print("\n--- Evaluation ---")
    X_test = DataSampler.sample_uniform(domain, N=5, seed=123)

    # Example utilizing the new signature with Guarantee checks
    preds, outside_mask = surrogate.evaluate(
        X_test,
        k=1,
        atol=ATOL,
        rtol=RTOL,
        L_global=10.0,
        alpha=1.0,
    )

    ground_truth = six_hump_camelback(X_test)
    err = xp.abs(preds - ground_truth)

    print(f"Test Points:\n{X_test}")
    print(f"Predictions:\n{preds}")
    print(f"Ground Truth:\n{ground_truth}")
    print(f"Outside Guarantee Mask: {outside_mask}")
    print(f"Max Error: {xp.max(err)}")
