"""
kNN/Taylor surrogate wrapper for near-ODE emulation.
"""

import time
from typing import Callable

import numpy as np
from scipy.stats import qmc

from adaptive_clustering_accelerated import (
    adaptive_cluster_algorithm,
    flatten_cluster_centers_with_gradient,
)
from nn_inference import NN_Inference


class NearODESurrogate:
    """
    kNN/Taylor surrogate wrapper for near-ODE emulation.

    Shape Conventions:
        - X.shape: (n_pts, d) where n_pts=number of points, d=input dimension
        - f_vals.shape: (n_pts, fd) where fd=function output dimension
        - gradients.shape: (n_pts, fd, d) where gradients are Jacobian matrices
        - targets.shape: (n_query, d) for evaluation points

    Note: If fd=1 (scalar function), f_vals.shape=(n_pts, 1) and gradients.shape=(n_pts, 1, d)

    This class orchestrates:
      1) (Optional) adaptive clustering to select source points/centroids,
      2) building a FAISS-backed nearest-neighbor index over source X,
      3) first-order Taylor prediction f(x_i) + J(x_i) @ (x - x_i),
      4) optional trust-region style guarantee checks using Hessian proxies.

    Notes:
      * The implementation assumes that the "oracle" callable returns both the
        function value and its Jacobian at a queried point.
      * Guarantees are computed in the inference layer; this class exposes a
        convenience method `_test_guarantee` for batch checks.

    Attributes:
      _model_trained (bool): Flag to indicate if the FAISS index is ready.
      atol (float | None): Absolute error tolerance for guarantees.
      rtol (float | None): Relative error tolerance for guarantees (optional).
      oracle (Callable | None): Function that returns (value, jacobian).
      d (int): Input dimension inferred from X/domain.
      fd (int): Output dimension inferred from f_vals.
      KNN_search (NN_Inference): Backend for nearest-neighbor + Taylor inference.
      N_sources (int): Number of source points (post-clustering or raw).
    """

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
        disp_time: bool = True,
        seed: int | None = None,
    ) -> int:
        """Train the surrogate from already-available data arrays.

        After training, use `evaluate()` to predict on new targets.

        Args:
          X: Array of input points with shape (n_pts, d).
          f_vals: Array of function values f(x) with shape (n_pts, fd).
          gradients: Array of Jacobians Df(x) with shape (n_pts, fd, d).
          do_clustering: If True, run adaptive clustering (uses `oracle`) to
            refine sources before building the index. If False, use every row in
            `X` as a source.
          oracle: Callable taking a single point vector x and returning
            (value: (fd,), jacobian: (fd, d)). Required if `do_clustering` is True.
          N_init_clusters: Initial number of clusters for the adaptive clustering.
          atol: Absolute error tolerance for guarantee checks.
          rtol: Relative error tolerance for guarantee checks (unused in this
            file but stored and passed downstream).
          min_ppc: Minimum points per cluster; clustering will not accept leaves
            below this threshold.
          check_guarantee_N_times: If > 0 and clustering is used, iterate a
            guarantee-check + add-sources loop this many times on the original
            X.
          L_global: Optional global Lipschitz constant used in guarantee logic.
          alpha: Multiplicative “fudge factor” for Hessian-based trust bounds.
          disp_time: If True, print wall-clock training time.
          seed: Optional seed forwarded to clustering / sampling routines.
        """
        start_time = time.perf_counter()

        # --- Basic shape checks (consolidated here) ---
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D array with shape (n_pts, d), got shape {X.shape}"
            )
        if f_vals.ndim != 2:
            raise ValueError(
                f"f_vals must be 2D array with shape (n_pts, fd), got shape {f_vals.shape}"
            )
        if gradients.ndim != 3:
            raise ValueError(
                f"gradients must be 3D array with shape (n_pts, fd, d), got shape {gradients.shape}"
            )

        n_pts, d = X.shape
        n_f_vals, fd = f_vals.shape
        n_grad, fd_grad, d_grad = gradients.shape

        if n_pts != n_f_vals or n_pts != n_grad:
            raise ValueError(
                f"Number of samples in X ({n_pts}), f_vals ({n_f_vals}), and gradients ({n_grad}) must match"
            )
        if d != d_grad:
            raise ValueError(
                f"Input dimension in X ({d}) and gradients ({d_grad}) must match"
            )
        if fd != fd_grad:
            raise ValueError(
                f"Function output dimension in f_vals ({fd}) and gradients ({fd_grad}) must match"
            )

        # Persist tolerances/oracle/dims
        self.atol = atol
        self.rtol = rtol
        self.oracle = oracle
        self.d = d
        self.fd = fd

        # --- Optional adaptive clustering ---
        if do_clustering:
            if not callable(oracle):
                raise ValueError("oracle must be a callable when do_clustering is True")

            k, cluster_tree = adaptive_cluster_algorithm(
                X_train=X,
                f=oracle,
                atol=atol,
                rtol=rtol if rtol is not None else 0.0001,
                N_init_clusters=N_init_clusters,
                min_ppc=min_ppc,
                use_gradient=True,
                seed=seed if seed is not None else 0,
            )

            new_X, new_f_vals, new_gradients, new_hessians, new_dims = (
                flatten_cluster_centers_with_gradient(
                    nc=k,
                    n=self.d,
                    m=self.fd,
                    tree=cluster_tree,
                )
            )

        else:
            # Use raw data as sources (no clustering).
            new_X = X
            new_f_vals = f_vals
            new_gradients = gradients
            new_hessians = None
            new_dims = None
            k = len(X)

        # --- Build FAISS-backed nearest-neighbor inference object ---
        self.KNN_search = NN_Inference(
            X=new_X,
            f_vals=new_f_vals,
            gradients=new_gradients,
            hessians=new_hessians,
            trust_radii=new_dims,
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
        if disp_time:
            print(f"Time to train: {end_time - start_time:.2f} seconds")
        return 0

    def train_model_from_sampling(
        self,
        oracle: Callable,
        domain: np.ndarray,
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
        disp_time: bool = True,
        seed: int | None = None,
    ) -> None:
        """Train the surrogate by sampling an oracle over a given domain.

        This method draws samples in `domain`, calls `oracle` to obtain
        (f_vals, gradients), and delegates to `train_model_from_data()`.

        Args:
          oracle: Callable that returns (value: (m,), jacobian: (m, n)) for a
            given input x.
          domain: Either a 1D array [l, r] for n=1 or a matrix with shape (n, 2)
            where each row is [l_i, r_i].
          m: Output dimension of the oracle.
          N: Number of samples to draw from `distribution`.
          distribution: "uniform" or "LHS" (Latin Hypercube Sampling).
          do_clustering: Whether to run adaptive clustering on the sampled data.
          N_init_clusters: Initial number of clusters for clustering.
          atol: Absolute error tolerance for guarantees.
          rtol: Relative error tolerance for guarantees (stored).
          min_ppc: Minimum points per cluster for acceptance.
          check_guarantee_N_times: Iterations of guarantee-check + enrichment.
          L: Optional global Lipschitz constant for guarantees.
          alpha: Multiplicative factor for Hessian-based trust bounds.
          disp_time: If True, print wall-clock sampling + training time.
          seed: Optional seed for reproducible sampling.

        Returns:
          None

        Notes:
          * This method preserves the original behavior, including returning -1
            early for invalid `domain` or `distribution`.
          * Output dimension `m` must match the oracle's output.
        """
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

        if distribution == "uniform":
            X = np.random.uniform(l, r, (N, n))
        elif distribution == "LHS":
            sampler = qmc.LatinHypercube(d=n, rng=seed)
            X = sampler.random(n=N)
            X = qmc.scale(X, l, r)

        # --- Evaluate oracle at sampled X to get (f_vals, gradients) ---
        # Vectorized oracle call: oracle expects shape (n, d) and returns (n, fd), (n, fd, d)
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
            disp_time=False,
            seed=seed,
        )

        end_time = time.perf_counter()
        if disp_time:
            print(f"Time to train: {end_time - start_time:.2f} seconds")

    def evaluate(
        self,
        targets: np.ndarray,
        k: int = 1,
        disp_time: bool = False,
        L: float | None = None,
        recluster: bool = False,
        alpha: float = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the trained surrogate at an array of target points.

        For 1D targets, a (N,) array is reshaped to (N, 1). Otherwise, `targets`
        must have shape (N, n). Returns both predictions and a boolean mask that
        flags points outside the trust/guarantee (if available).

        Args:
          targets: Array of evaluation points with shape (n_query, d).
          k: Number of nearest neighbors to use in prediction.
          disp_time: If True, print wall-clock inference time from backend.
          L: Optional global Lipschitz constant for guarantees; forwarded to
            the backend inference.
          recluster: If True, points outside the guarantee will be passed to
            adaptive clustering and appended to the neighbor index.
          alpha: Multiplicative factor for Hessian-based trust bounds.

        Returns:
          tuple[np.ndarray, np.ndarray]:
            - predictions: (n_query, fd) surrogate outputs,
            - unguaranteed_points: (n_query,) boolean mask; True where outside trust.

        Notes:
          * Requires that `train_model_from_*` has been called successfully.
          * Guarantee logic currently applies in the backend primarily for k==1.
          * Input targets must be 2D array with shape (n_query, d)
        """
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
            disp_time=disp_time,
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
        """Return the current source feature matrix used in kNN searches.

        Returns:
          np.ndarray: Array of source points with shape (n_sources, d).
        """
        return self.KNN_search.X

    def add_sources(self, X: np.ndarray) -> None:
        """Adaptively cluster new candidate points and append them as sources.

        This is typically called on the subset of targets that failed the
        guarantee check during `evaluate(recluster=True)`.

        Args:
          X: Array of new candidate points with shape (n_query, d).

        Returns:
          None

        Notes:
          * This method preserves the original logic, including the argument
            order to `adaptive_cluster_algorithm`.
          * The call to `flatten_cluster_centers_with_gradient` mirrors the
            original code (which may contain a dimensionality bug if `self.nf`
            is undefined / not intended). It is left unchanged intentionally.
        """
        if not callable(self.oracle):
            raise ValueError("oracle must be a callable to add sources")

        k, cluster_tree = adaptive_cluster_algorithm(
            X_train=X,
            f=self.oracle,
            atol=self.atol,
            rtol=self.rtol if self.rtol is not None else 0.0001,
            N_init_clusters=2,
            min_ppc=10,
            use_gradient=True,
            seed=0,
        )
        new_X, new_f_vals, new_gradients, new_hessians, new_dims = (
            flatten_cluster_centers_with_gradient(
                nc=k,
                n=self.d,
                m=self.fd,
                tree=cluster_tree,
            )
        )

        self.KNN_search.add_extra_sources(
            new_X, new_f_vals, new_gradients, new_hessians, new_dims
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
