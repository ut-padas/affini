"""
kNN/Taylor surrogate wrapper for near-ODE emulation.
"""

import time
from typing import Callable

import numpy as np
from scipy.stats import qmc

from src.adaptive_clustering import adaptive_cluster_algorithm
from src.nn_inference import NN_Inference


class NearODESurrogate:
    """kNN/Taylor surrogate wrapper for near-ODE emulation on GPU."""

    def __init__(self) -> None:
        """Initialize an empty surrogate; call a train_* method before evaluate()."""
        self._model_trained = False

    def train_model_from_data(
        self,
        X: np.ndarray,
        f_vals: np.ndarray | None,
        gradients: np.ndarray | None,
        do_clustering: bool = True,
        f: Callable | None = None,
        grad_f: Callable | None = None,
        cupy_f: Callable | None = None,
        N_init_clusters: int = 10,
        atol: float = 0.01,
        rtol: float | None = None,
        min_ppc: int = 1,
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
        self.seed = seed if seed is not None else 0

        # --- Optional adaptive clustering ---
        if do_clustering:
            if not callable(self.f):
                raise ValueError("f must be a callable when do_clustering is True")

            clusters = adaptive_cluster_algorithm(
                X_train=X,
                f=self.f,
                grad_f=self.grad_f,
                cupy_f=cupy_f,
                atol=self.atol,
                rtol=self.rtol,
                N_init_clusters=N_init_clusters,
                min_ppc=min_ppc,
                seed=self.seed,
            )
            new_X = np.array([cluster.centroid for cluster in clusters])
            new_f_vals = np.array([cluster.value for cluster in clusters])
            new_gradients = np.array([cluster.gradient for cluster in clusters])
            new_hessians = np.array([cluster.hessian for cluster in clusters])
            new_radii = np.array([cluster.radius for cluster in clusters])
            k = len(new_X)

        else:
            # Use raw data as sources (no clustering).
            # --- Basic shape checks (consolidated here) ---
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
            k = len(X)

        # --- Build FAISS-backed nearest-neighbor inference object ---
        self.KNN_search = NN_Inference(
            X=new_X,
            f_vals=new_f_vals,
            gradients=new_gradients,
            hessians=new_hessians,
            trust_radii=new_radii,
        )
        self._model_trained = True
        self.N_sources = k

        # --- Optional: heuristic guarantee ---
        # When clustering and a global L are provided, we can repeatedly check
        # original X against the guarantee and add the violations back.
        if do_clustering and L_global and check_guarantee_N_times > 0:
            for _ in range(check_guarantee_N_times):
                bad_X = self._test_guarantee(
                    targets=X, tol=atol, L_global=L_global, alpha=alpha
                )
                self.add_sources(X=X[bad_X])

        end_time = time.perf_counter()
        print(f"Time to train: {end_time - start_time:.2f} seconds")

    def train_model_from_sampling(
        self,
        f: Callable,
        grad_f: Callable | None,
        cupy_f: Callable | None,
        domain: np.ndarray,
        N: int,
        distribution: str = "uniform",
        N_init_clusters: int = 10,
        atol: float = 0.01,
        rtol: float | None = None,
        min_ppc: int = 1,
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

        if distribution == "uniform":
            X = np.random.uniform(l, r, (N, n))
        elif distribution == "LHS":
            sampler = qmc.LatinHypercube(d=n, rng=seed)
            X = sampler.random(n=N)
            X = qmc.scale(X, l, r)

        # --- Delegate to the array-based trainer (preserving original args order) ---
        self.train_model_from_data(
            X=X,
            f_vals=None,
            gradients=None,
            do_clustering=True,
            f=f,
            grad_f=grad_f,
            cupy_f=cupy_f,
            N_init_clusters=N_init_clusters,
            atol=atol,
            rtol=rtol,
            min_ppc=min_ppc,
            check_guarantee_N_times=check_guarantee_N_times,
            L_global=L,
            alpha=alpha,
            seed=seed,
        )

    def evaluate(
        self,
        targets: np.ndarray,
        k: int = 1,
        L: float | None = None,
        recluster: bool = False,
        alpha: float = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._model_trained:
            raise RuntimeError(
                "Model must be trained before calling evaluate(). Run train_model_from_data() or train_model_from_sampling() first."
            )

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

        # Heuristic warning for under-resolved surrogate coverage.
        if np.sum(unguaranteed_points) > 3 * self.N_sources:
            print(
                "WARNING: too many points (3 * N_clusters) are outside of guarantee. Need more clusters."
            )

        # Optional enrichment: recluster the failing targets and append as sources.
        if recluster:
            self.add_sources(X=targets[unguaranteed_points])

        return predictions, unguaranteed_points

    def get_sources(self) -> np.ndarray:
        return np.asarray(self.KNN_search.X)

    def add_sources(self, X: np.ndarray) -> None:
        if not callable(self.f):
            raise ValueError("oracle must be a callable to add sources")

        clusters = adaptive_cluster_algorithm(
            X_train=X,
            f=self.f,
            grad_f=self.grad_f,
            cupy_f=self.cupy_f,
            atol=self.atol,
            rtol=self.rtol,
            N_init_clusters=2,
            min_ppc=10,
            seed=0,
        )
        new_X = np.array([cluster.centroid for cluster in clusters])
        new_f_vals = np.array([cluster.value for cluster in clusters])
        new_gradients = np.array([cluster.gradient for cluster in clusters])
        new_hessians = np.array([cluster.hessian for cluster in clusters])
        new_radii = np.array([cluster.radius for cluster in clusters])
        k = len(new_X)

        self.KNN_search.add_extra_sources(
            new_X, new_f_vals, new_gradients, new_hessians, new_radii
        )
        self.N_sources += len(new_X)

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
