"""kNN/Taylor surrogate wrapper for near-ODE emulation."""

import time
from typing import Callable

import cupy as cp
from scipy.stats import qmc

from src.gpu_adaptive_clustering import adaptive_clustering_gpu
from src.gpu_nn_inference import GPU_NN_Inference


class GPU_NearODESurrogate:
    """kNN/Taylor surrogate wrapper for near-ODE emulation on GPU."""

    def __init__(self) -> None:
        """Initialize an empty surrogate; call a train_* method before evaluate()."""
        self._model_trained = False

    def train_model_from_data(
        self,
        X: cp.ndarray,
        f_vals: cp.ndarray,
        gradients: cp.ndarray,
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
            )
            new_X = cp.vstack([cluster.centroid for cluster in clusters])
            new_f_vals = cp.vstack([cluster.F_centroid for cluster in clusters])
            new_gradients = cp.stack(
                [cluster.J_centroid for cluster in clusters]
            ).squeeze(axis=1)

            # Handle hessian_proxy - convert float scalars to array
            new_hessians = cp.array([cluster.hessian_proxy for cluster in clusters])

            new_radii = cp.array([cluster.radius for cluster in clusters])
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
        self.KNN_search = GPU_NN_Inference(
            X=new_X.get() if new_X is not None else None,
            f_vals=new_f_vals.get() if new_f_vals is not None else None,
            gradients=new_gradients.get() if new_gradients is not None else None,
            hessians=new_hessians.get() if new_hessians is not None else None,
            trust_radii=new_radii.get() if new_radii is not None else None,
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
        domain: cp.ndarray,
        m: int,
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
        start_time = time.perf_counter()

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

        rng = cp.random.default_rng(seed)
        assert rng is not None, "Something went wrong with RNG initialization."

        if distribution == "uniform":
            X = rng.uniform(l, r, (N, n))
        elif distribution == "LHS":
            sampler = qmc.LatinHypercube(d=n, rng=seed)
            X = sampler.random(n=N)
            X = qmc.scale(X, cp.asnumpy(l), cp.asnumpy(r))
            X = cp.asarray(X)

        f_vals, gradients = oracle(X)
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

        end_time = time.perf_counter()
        print(f"Time to train: {end_time - start_time:.2f} seconds")

    def evaluate(
        self,
        targets: cp.ndarray,
        k: int = 1,
        L: float | None = None,
        recluster: bool = False,
        alpha: float = 1,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        if not self._model_trained:
            raise RuntimeError(
                "Model must be trained before calling evaluate(). Run train_model_from_data() or train_model_from_sampling() first."
            )

        if targets.ndim != 2:
            raise ValueError(
                f"targets must be 2D array with shape (n_query, d), got shape {targets.shape}"
            )

        predictions, unguaranteed_points = self.KNN_search.predict(
            targets=targets.get(),
            k=k,
            L_global=L,
            atol=self.atol,
            rtol=self.rtol,
            alpha=alpha,
        )

        # Convert results back to CuPy arrays
        predictions = cp.asarray(predictions)
        unguaranteed_points = cp.asarray(unguaranteed_points)

        # Heuristic warning for under-resolved surrogate coverage.
        if cp.sum(unguaranteed_points) > 3 * self.N_sources:
            print(
                "WARNING: too many points (3 * N_clusters) are outside of guarantee. Need more clusters."
            )

        # Optional enrichment: recluster the failing targets and append as sources.
        if recluster:
            self.add_sources(X=targets[unguaranteed_points])

        return predictions, unguaranteed_points

    def get_sources(self) -> cp.ndarray:
        return cp.asarray(self.KNN_search.X)

    def add_sources(self, X: cp.ndarray) -> None:
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
        )
        new_X = cp.vstack([cluster.centroid for cluster in clusters])
        new_f_vals = cp.vstack([cluster.F_centroid for cluster in clusters])
        new_gradients = cp.stack([cluster.J_centroid for cluster in clusters]).squeeze(
            axis=1
        )

        # Handle hessian_proxy - convert float scalars to array
        new_hessians = cp.array([cluster.hessian_proxy for cluster in clusters])

        new_dims = cp.array([cluster.radius for cluster in clusters])

        self.KNN_search.add_extra_sources(
            new_X.get(),
            new_f_vals.get(),
            new_gradients.get(),
            new_hessians.get(),
            new_dims.get(),
        )
        self.N_sources += len(new_X)

    def _test_guarantee(
        self, targets: cp.ndarray, tol: float, L_global: float, alpha: float
    ) -> cp.ndarray:
        if self.KNN_search.hessians is None or self.KNN_search.trust_radii is None:
            raise RuntimeError(
                "Hessian proxies and trust radii are required for guarantee checks."
            )

        D, I = self.KNN_search._faiss_index.search(targets.get(), 1)  # type: ignore
        D, I = cp.asarray(D), cp.asarray(I)  # Convert back to CuPy
        local_tols = cp.minimum(
            cp.asarray(self.KNN_search.trust_radii)[I[:, 0]] ** 2,
            2 * tol / (alpha * cp.asarray(self.KNN_search.hessians)[I[:, 0]]),
        )
        global_tol = 2 * tol / L_global
        good_points = cp.logical_or(
            (D[:, 0] < local_tols),
            cp.logical_and(local_tols < D[:, 0], D[:, 0] < global_tol),
        )
        return cp.logical_not(good_points)
