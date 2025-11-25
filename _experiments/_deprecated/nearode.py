"""
kNN/Taylor surrogate wrapper for near-ODE emulation.
"""

import time
from enum import Enum
from typing import Callable

import cupy as cp
import numpy as np
from scipy.stats import qmc

from _experiments._deprecated.nn_inference import NN_Inference
from src.adaptive_clustering import adaptive_cluster_algorithm
from src.gpu_adaptive_clustering import adaptive_cluster_algorithm_gpu


class ClusteringBackend(Enum):
    COMPLETE_CUPY = "complete_cupy"
    NUMPY_CUPY_HYBRID = "numpy_cupy_hybrid"


class NearODESurrogate:
    """kNN/Taylor surrogate wrapper for near-ODE emulation on GPU."""

    def __init__(self, data_dim: int, f_dim: int) -> None:
        """Initialize an empty surrogate; call a train_* method before evaluate()."""
        self.data_dim = data_dim
        self.f_dim = f_dim
        self._model_trained = False

    def train_model_from_data(
        self,
        X: np.ndarray,
        f_vals: np.ndarray | None,
        gradients: np.ndarray | None,
        clustering_backend: ClusteringBackend,
        do_clustering: bool = True,
        f: Callable | None = None,
        grad_f: Callable | None = None,
        cupy_f: Callable | None = None,
        cupy_grad_f: Callable | None = None,
        num_initial_clusters: int = 10,
        atol: float = 0.01,
        rtol: float | None = None,
        min_pts_per_cluster: int = 1,
        check_guarantee_N_times: int = 0,
        L_global: float | None = None,
        alpha: float = 1.0,
        seed: int | None = None,
    ) -> None:
        start_time = time.perf_counter()

        # Persist tolerances/oracle/dims
        self.atol = atol
        self.rtol = rtol if rtol is not None else 0.001
        self.f = f
        self.grad_f = grad_f
        self.cupy_f = cupy_f
        self.cupy_grad_f = cupy_grad_f
        self.seed = seed if seed is not None else 0

        # Optional clustering step
        if do_clustering:
            if clustering_backend == ClusteringBackend.NUMPY_CUPY_HYBRID:
                if not callable(self.f):
                    raise ValueError("f must be a callable")

                clusters = adaptive_cluster_algorithm(
                    X_train=X,
                    f=self.f,
                    grad_f=self.grad_f,
                    cupy_f=cupy_f,
                    atol=self.atol,
                    rtol=self.rtol,
                    num_initial_clusters=num_initial_clusters,
                    min_pts_per_cluster=min_pts_per_cluster,
                    seed=self.seed,
                )
                new_X = np.array([cluster.centroid for cluster in clusters])
                new_f_vals = np.array([cluster.F_centroid for cluster in clusters])
                new_gradients = np.array([cluster.J_centroid for cluster in clusters])
                new_hessians = np.array([cluster.hessian_proxy for cluster in clusters])
                new_radii = np.array([cluster.radius for cluster in clusters])
                num_clusters = len(new_X)

            elif clustering_backend == ClusteringBackend.COMPLETE_CUPY:
                if not callable(self.cupy_f) or not callable(self.cupy_grad_f):
                    raise ValueError("cupy_f and cupy_grad_f must be callables")

                result = adaptive_cluster_algorithm_gpu(
                    X_train=cp.asarray(X),
                    f=self.cupy_f,
                    grad=self.cupy_grad_f,
                    DATA_DIM=self.data_dim,
                    F_DIM=self.f_dim,
                    NUM_INITIAL_CLUSTERS=num_initial_clusters,
                    MIN_PTS_PER_CLUSTER=min_pts_per_cluster,
                    ATOL=self.atol,
                    RTOL=self.rtol,
                    SEED=self.seed,
                    LOG_MEM_STATS=True,
                    MOVE_INTERMEDIATE_TO_HOST=True,
                    STORE_FINAL_ARRAYS_ON_HOST=True,
                )
                new_X = result[0]
                new_f_vals = result[1]
                new_gradients = result[2]
                new_hessians = result[3]
                new_radii = result[4]
                num_clusters = len(new_X)

        else:
            # Use raw data as sources (no clustering)
            if f_vals is None or gradients is None:
                raise ValueError(
                    "f_vals and gradients cannot be None when do_clustering is False"
                )

            if X.ndim != 2:
                raise ValueError(
                    f"X must be 2D array with shape (N, D), got shape {X.shape}"
                )
            if f_vals.ndim != 2:
                raise ValueError(
                    f"f_vals must be 2D array with shape (N, FD), got shape {f_vals.shape}"
                )
            if gradients.ndim != 3:
                raise ValueError(
                    f"gradients must be 3D array with shape (N, FD, D), got shape {gradients.shape}"
                )

            N, D = X.shape
            N_F, FD = f_vals.shape
            N_G, FD_G, D_G = gradients.shape

            if N != N_F or N != N_G:
                raise ValueError(
                    f"Number of samples in X ({N}), f_vals ({N_F}), and gradients ({N_G}) must match"
                )
            if D != D_G:
                raise ValueError(
                    f"Input dimension in X ({D}) and gradients ({D_G}) must match"
                )
            if FD != FD_G:
                raise ValueError(
                    f"Function output dimension in f_vals ({FD}) and gradients ({FD_G}) must match"
                )

            new_X = X
            new_f_vals = f_vals
            new_gradients = gradients
            new_hessians = None
            new_radii = None
            num_clusters = len(X)

        # Build the kNN/Taylor inference structure
        self.KNN_search = NN_Inference(
            X=new_X,
            f_vals=new_f_vals,
            gradients=new_gradients,
            hessians=new_hessians,
            trust_radii=new_radii,
        )
        self._model_trained = True
        self.N_sources = num_clusters

        # --- Optional: heuristic guarantee ---
        # When clustering and a global L are provided, we can repeatedly check
        # original X against the guarantee and add the violations back.
        if do_clustering and L_global and check_guarantee_N_times > 0:
            for _ in range(check_guarantee_N_times):
                bad_X = self._test_guarantee(
                    targets=X, tol=atol, L_global=L_global, alpha=alpha
                )
                self.add_sources(X=X[bad_X], clustering_backend=clustering_backend)

        end_time = time.perf_counter()
        print(f"Time to train: {end_time - start_time:.2f} seconds")

    def train_model_from_sampling(
        self,
        f: Callable,
        grad_f: Callable | None,
        cupy_f: Callable | None,
        cupy_grad_f: Callable | None,
        domain: np.ndarray,
        N: int,
        clustering_backend: ClusteringBackend,
        distribution: str = "uniform",
        num_initial_clusters: int = 10,
        atol: float = 0.01,
        rtol: float | None = None,
        min_pts_per_cluster: int = 1,
        check_guarantee_N_times: int = 0,
        L: float | None = None,
        alpha: float = 1,
        seed: int | None = None,
    ) -> None:
        # --- Validate/parse domain into (n, [l, r]) ---
        if domain.ndim > 2:
            raise ValueError("domain must be 1D or 2D array")
        if domain.ndim == 1:
            n = 1
            q = len(domain)
            l = domain[0]
            r = domain[1]
        else:
            n, q = domain.shape
            l = domain[:, 0]
            r = domain[:, 1]
        if q != 2:
            raise ValueError("domain must have shape (n, 2)")

        # --- Draw samples according to chosen distribution ---
        distributions = ["uniform", "LHS"]
        if distribution not in distributions:
            raise ValueError(f"distribution must be one of {distributions}")

        rng = np.random.default_rng(seed)
        assert rng is not None, "Something went wrong with RNG initialization."

        if distribution == "uniform":
            X = rng.uniform(l, r, (N, n))
        elif distribution == "LHS":
            sampler = qmc.LatinHypercube(d=n, rng=seed)
            X = sampler.random(n=N)
            if cp.is_available():
                l_np = cp.asnumpy(l) if hasattr(l, "device") else l  # type: ignore
                r_np = cp.asnumpy(r) if hasattr(r, "device") else r  # type: ignore
                X = qmc.scale(X, l_np, r_np)
                X = cp.asarray(X)  # type: ignore
            else:
                X = qmc.scale(X, l, r)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        # --- Delegate to the array-based trainer (preserving original args order) ---
        self.train_model_from_data(
            X=X,
            f_vals=None,
            gradients=None,
            do_clustering=True,
            f=f,
            grad_f=grad_f,
            cupy_f=cupy_f,
            cupy_grad_f=cupy_grad_f,
            clustering_backend=clustering_backend,
            num_initial_clusters=num_initial_clusters,
            atol=atol,
            rtol=rtol,
            min_pts_per_cluster=min_pts_per_cluster,
            check_guarantee_N_times=check_guarantee_N_times,
            L_global=L,
            alpha=alpha,
            seed=seed,
        )

    def evaluate(
        self,
        targets: np.ndarray,
        clustering_backend: ClusteringBackend,
        k: int = 1,
        L: float | None = None,
        recluster: bool = False,
        alpha: float = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._model_trained:
            raise RuntimeError(
                "Model must be trained before calling evaluate(). Run train_model_from_data() or train_model_from_sampling() first."
            )

        # Handle CuPy arrays by converting to NumPy
        is_cupy_input = cp.is_available()
        if is_cupy_input:
            targets = cp.asnumpy(targets)  # type: ignore

        if targets.ndim != 2:
            raise ValueError(
                f"targets must be 2D array with shape (n_query, d), got shape {targets.shape}"
            )

        predictions, unguaranteed_points = self.KNN_search.predict(
            targets=targets,
            k=k,
            L_global=L,
            atol=self.atol,
            rtol=self.rtol,
            alpha=alpha,
        )

        # Convert back to CuPy if input was CuPy
        if is_cupy_input:
            predictions = cp.asarray(predictions)  # type: ignore
            unguaranteed_points = cp.asarray(unguaranteed_points)  # type: ignore

        # Heuristic warning for under-resolved surrogate coverage.
        # Use appropriate sum function based on array type
        if is_cupy_input:
            sum_unguaranteed = cp.sum(unguaranteed_points)  # type: ignore
        else:
            sum_unguaranteed = np.sum(unguaranteed_points)

        if sum_unguaranteed > 3 * self.N_sources:
            print(
                "WARNING: too many points (3 * N_clusters) are outside of guarantee. Need more clusters."
            )

        # Optional enrichment: recluster the failing targets and append as sources.
        if recluster:
            self.add_sources(
                X=targets[unguaranteed_points], clustering_backend=clustering_backend
            )

        return predictions, unguaranteed_points

    def get_sources(self) -> np.ndarray:
        return np.asarray(self.KNN_search.X)

    def add_sources(self, X: np.ndarray, clustering_backend: ClusteringBackend) -> None:
        if clustering_backend == ClusteringBackend.NUMPY_CUPY_HYBRID:
            if not callable(self.f):
                raise ValueError("f must be a callable")

            clusters = adaptive_cluster_algorithm(
                X_train=X,
                f=self.f,
                grad_f=self.grad_f,
                cupy_f=self.cupy_f,
                atol=self.atol,
                rtol=self.rtol,
                num_initial_clusters=2,
                min_pts_per_cluster=10,
                seed=0,
            )
            new_X = np.array([cluster.centroid for cluster in clusters])
            new_f_vals = np.array([cluster.F_centroid for cluster in clusters])
            new_gradients = np.array([cluster.J_centroid for cluster in clusters])
            new_hessians = np.array([cluster.hessian_proxy for cluster in clusters])
            new_radii = np.array([cluster.radius for cluster in clusters])
            num_clusters = len(new_X)

        elif clustering_backend == ClusteringBackend.COMPLETE_CUPY:
            if not callable(self.cupy_f) or not callable(self.cupy_grad_f):
                raise ValueError("cupy_f and cupy_grad_f must be callables")

            result = adaptive_cluster_algorithm_gpu(
                X_train=cp.asarray(X),
                f=self.cupy_f,
                grad=self.cupy_grad_f,
                DATA_DIM=self.data_dim,
                F_DIM=self.f_dim,
                NUM_INITIAL_CLUSTERS=2,
                MIN_PTS_PER_CLUSTER=10,
                ATOL=self.atol,
                RTOL=self.rtol,
                SEED=self.seed,
                LOG_MEM_STATS=True,
                MOVE_INTERMEDIATE_TO_HOST=True,
                STORE_FINAL_ARRAYS_ON_HOST=True,
            )
            new_X = result[0]
            new_f_vals = result[1]
            new_gradients = result[2]
            new_hessians = result[3]
            new_radii = result[4]
            num_clusters = len(new_X)

        self.KNN_search.add_extra_sources(
            new_X, new_f_vals, new_gradients, new_hessians, new_radii
        )
        self.N_sources += num_clusters

    def _test_guarantee(
        self, targets: np.ndarray, tol: float, L_global: float, alpha: float
    ) -> np.ndarray:
        """Return a boolean mask of targets that violate the local/global guarantee.

        This helper queries the nearest neighbor from the FAISS index and checks
        whether each target lies within either a local trust radius (based on the
        stored cluster radius and Hessian proxy) or a global Lipschitz-based bound.

        Args:
          targets: Array of target points with shape (n_query, d).
          tol: Absolute tolerance used in the bound.
          L_global: Global Lipschitz constant for the fallback bound.
          alpha: Multiplicative factor for Hessian-based trust bounds.

        Returns:
          np.ndarray: Boolean mask of shape (n_query,), True where the target is
          **outside** the guarantee (i.e., not good).
        """
        if self.KNN_search.hessians is None or self.KNN_search.trust_radii is None:
            raise RuntimeError(
                "Hessian proxies and trust radii are required for guarantee checks."
            )

        D, I = self.KNN_search._faiss_index.search(targets, 1)  # type: ignore
        local_tols = np.minimum(
            self.KNN_search.trust_radii[I[:, 0]] ** 2,
            2 * tol / (alpha * self.KNN_search.hessians[I[:, 0]]),
        )
        global_tol = 2 * tol / L_global
        good_points = np.logical_or(
            (D[:, 0] < local_tols),
            np.logical_and(local_tols < D[:, 0], D[:, 0] < global_tol),
        )
        return np.logical_not(good_points)
