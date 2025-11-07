"""kNN/Taylor surrogate wrapper for near-ODE emulation."""

import time
from typing import Callable

import cupy as cp
import numpy as np
from scipy.stats import qmc

from src.gpu_adaptive_clustering import adaptive_clustering_gpu
from src.nn_inference import NN_Inference


class NearODESurrogate:
    """kNN/Taylor surrogate wrapper for near-ODE emulation on GPU."""

    def __init__(self) -> None:
        """Initialize an empty surrogate; call a train_* method before evaluate()."""
        self._model_trained = False

    def train_model_from_data(
        self,
        X: np.ndarray,
        f_vals: np.ndarray,
        gradients: np.ndarray,
        do_clustering: bool = True,
        oracle: Callable | None = None,
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

        # --- Basic shape checks (consolidated here) ---
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

        # Persist tolerances/oracle/dims
        self.atol = atol
        self.rtol = rtol
        self.oracle = oracle
        self.D = D
        self.FD = FD

        # --- Optional adaptive clustering ---
        if do_clustering:
            if not callable(oracle):
                raise ValueError("oracle must be a callable when do_clustering is True")

            clusters = adaptive_clustering_gpu(
                X=X,
                f=oracle,
                n_init_clusters=N_init_clusters,
                min_pts_per_cluster=min_ppc,
                abs_tol=atol,
                rel_tol=rtol if rtol is not None else 0.0,
                seed=seed if seed is not None else 0,
                return_numpy=True,
            )
            new_X = np.vstack([cluster.centroid for cluster in clusters])
            new_f_vals = np.vstack([cluster.F_centroid for cluster in clusters])
            new_gradients = np.stack(
                [cluster.J_centroid for cluster in clusters]
            ).squeeze(axis=1)

            # Handle hessian_proxy - convert float scalars to array
            new_hessians = np.array([cluster.hessian_proxy for cluster in clusters])

            new_radii = np.array([cluster.radius for cluster in clusters])
            k = len(clusters)

        else:
            # Use raw data as sources (no clustering).
            new_X = X
            new_f_vals = f_vals
            new_gradients = gradients
            new_hessians = None
            new_radii = None
            k = len(X)

        # --- Build FAISS-backed nearest-neighbor inference object ---
        # Convert CuPy arrays to NumPy for FAISS compatibility
        self.KNN_search = NN_Inference(
            X=new_X,
            f_vals=new_f_vals,
            gradients=new_gradients,
            hessians=new_hessians if new_hessians is not None else None,
            trust_radii=new_radii if new_radii is not None else None,
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
        oracle: Callable,
        domain: np.ndarray,
        N: int,
        distribution: str = "uniform",
        do_clustering: bool = True,
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

        rng = np.random.default_rng(seed)
        assert rng is not None, "Something went wrong with RNG initialization."

        if distribution == "uniform":
            X = rng.uniform(l, r, (N, n))
        elif distribution == "LHS":
            sampler = qmc.LatinHypercube(d=n, rng=seed)
            X = sampler.random(n=N)
            l, r = cp.asnumpy(l), cp.asnumpy(r)
            X = qmc.scale(X, l, r)

        # Ensure X is a CuPy array for oracle functions that expect CuPy
        f_vals, gradients = oracle(cp.asarray(X))

        f_vals = cp.asnumpy(f_vals)
        gradients = cp.asnumpy(gradients)
        X = cp.asnumpy(X)  # Convert back to NumPy for FAISS compatibility

        # --- Delegate to the array-based trainer (preserving original args order) ---
        self.train_model_from_data(
            X=X,
            f_vals=f_vals,
            gradients=gradients,
            do_clustering=do_clustering,
            oracle=oracle,
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

        # Convert CuPy targets to NumPy for FAISS compatibility
        if hasattr(targets, "device"):  # It's a CuPy array
            targets_np = cp.asnumpy(targets)
        else:
            targets_np = targets

        # Convert CuPy scalars to Python scalars for compatibility
        L_global = float(L) if L is not None else L
        atol = float(self.atol)
        rtol = float(self.rtol) if self.rtol is not None else self.rtol
        alpha = float(alpha)

        predictions, unguaranteed_points = self.KNN_search.predict(
            targets=targets_np,
            k=k,
            L_global=L_global,
            atol=atol,
            rtol=rtol,
            alpha=alpha,
        )

        # Convert results back to CuPy arrays if original targets were CuPy
        if hasattr(targets, "device"):
            predictions = cp.asarray(predictions)
            unguaranteed_points = cp.asarray(unguaranteed_points)
        else:
            predictions = np.asarray(predictions)
            unguaranteed_points = np.asarray(unguaranteed_points)

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
        if not callable(self.oracle):
            raise ValueError("oracle must be a callable to add sources")

        clusters = adaptive_clustering_gpu(
            X=X,
            f=self.oracle,
            n_init_clusters=2,
            min_pts_per_cluster=10,
            abs_tol=self.atol,
            rel_tol=self.rtol if self.rtol is not None else 0.0001,
            seed=0,
            return_numpy=True,
        )
        new_X = np.vstack([cluster.centroid for cluster in clusters])
        new_f_vals = np.vstack([cluster.F_centroid for cluster in clusters])
        new_gradients = np.stack([cluster.J_centroid for cluster in clusters]).squeeze(
            axis=1
        )

        # Handle hessian_proxy - convert float scalars to array
        new_hessians = np.array([cluster.hessian_proxy for cluster in clusters])

        new_dims = np.array([cluster.radius for cluster in clusters])

        self.KNN_search.add_extra_sources(
            new_X,
            new_f_vals,
            new_gradients,
            new_hessians,
            new_dims,
        )
        self.N_sources += len(new_X)

    def _test_guarantee(
        self, targets: np.ndarray, tol: float, L_global: float, alpha: float
    ) -> np.ndarray:
        if self.KNN_search.hessians is None or self.KNN_search.trust_radii is None:
            raise RuntimeError(
                "Hessian proxies and trust radii are required for guarantee checks."
            )

        D, I = self.KNN_search._faiss_index.search(targets.get(), 1)  # type: ignore
        D, I = np.asarray(D), np.asarray(I)  # Convert back to CuPy
        local_tols = np.minimum(
            np.asarray(self.KNN_search.trust_radii)[I[:, 0]] ** 2,
            2 * tol / (alpha * np.asarray(self.KNN_search.hessians)[I[:, 0]]),
        )
        global_tol = 2 * tol / L_global
        good_points = np.logical_or(
            (D[:, 0] < local_tols),
            np.logical_and(local_tols < D[:, 0], D[:, 0] < global_tol),
        )
        return np.logical_not(good_points)
