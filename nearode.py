"""
kNN/Taylor surrogate wrapper for near-ODE emulation.
"""

import time
from typing import Callable

import numpy as np
from scipy.stats import qmc

import adaptive_clustering_accelerated as aca
from adaptive_clustering import (
    adaptive_cluster_algorithm,
    flatten_cluster_centers_with_gradient,
)
from nn_inference import NN_Inference


class NearODESurrogate:
    """kNN/Taylor surrogate wrapper for near-ODE emulation.

    This class orchestrates:
      1) (Optional) adaptive clustering to select source points/centroids,
      2) building a FAISS-backed nearest-neighbor index over source features,
      3) first-order Taylor prediction f(x_i) + J(x_i) @ (x - x_i),
      4) optional trust-region style guarantee checks using Hessian proxies.

    Notes:
      * The implementation assumes that the "oracle" callable returns both the
        function value and its Jacobian at a queried point.
      * Shapes used throughout:
          - features:   (P, n)
          - labels:     (P, m)
          - gradients:  (P, m, n)  # full Jacobian per sample
      * Guarantees are computed in the inference layer; this class exposes a
        convenience method `_test_guarantee` for batch checks.

    Attributes:
      _model_trained (bool): Flag to indicate if the FAISS index is ready.
      atol (float | None): Absolute error tolerance for guarantees.
      rtol (float | None): Relative error tolerance for guarantees (optional).
      oracle (Callable | None): Function that returns (value, jacobian).
      n (int): Input dimension inferred from features/domain.
      m (int): Output dimension inferred from labels.
      KNN_search (NN_Inference): Backend for nearest-neighbor + Taylor inference.
      N_sources (int): Number of source points (post-clustering or raw).
    """

    def __init__(self, use_accelerated_clustering: bool) -> None:
        """Initialize an empty surrogate; call a train_* method before evaluate()."""
        self.use_accelerated_clustering = use_accelerated_clustering
        self._model_trained = False

    def train_model_from_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
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
          features: Array of input points used as cluster centroids or raw sources
            with shape (P, n).
          labels: Array of function values f(x) with shape (P, m).
          gradients: Array of Jacobians Df(x) with shape (P, m, n).
          do_clustering: If True, run adaptive clustering (uses `oracle`) to
            refine sources before building the index. If False, use every row in
            `features` as a source.
          oracle: Callable taking a single feature vector x and returning
            (value: (m,), jacobian: (m, n)). Required if `do_clustering` is True.
          N_init_clusters: Initial number of clusters for the adaptive clustering.
          atol: Absolute error tolerance for guarantee checks.
          rtol: Relative error tolerance for guarantee checks (unused in this
            file but stored and passed downstream).
          min_ppc: Minimum points per cluster; clustering will not accept leaves
            below this threshold.
          check_guarantee_N_times: If > 0 and clustering is used, iterate a
            guarantee-check + add-sources loop this many times on the original
            features.
          L_global: Optional global Lipschitz constant used in guarantee logic.
          alpha: Multiplicative “fudge factor” for Hessian-based trust bounds.
          disp_time: If True, print wall-clock training time.
          seed: Optional seed forwarded to clustering / sampling routines.
        """
        start_time = time.perf_counter()

        # --- Basic shape checks (kept as prints, preserving original behavior) ---
        pf, n = features.shape
        pl, m = labels.shape
        pg, mg, ng = gradients.shape
        if pf != pg or pl != pg:
            raise ValueError(
                "Number of samples in features/labels/gradients must match"
            )
        if n != ng:
            raise ValueError("Length of input dimension does not match gradients")
        if m != mg:
            raise ValueError("Length of output dimension does not match gradients")

        # Persist tolerances/oracle/dimensions
        self.atol = atol
        self.rtol = rtol
        self.oracle = oracle
        self.n = n
        self.m = m

        # --- Optional adaptive clustering ---
        if do_clustering:
            if not callable(oracle):
                raise ValueError("oracle must be a callable when do_clustering is True")

            if not self.use_accelerated_clustering:
                # Build cluster tree using the provided oracle and tolerances.
                k, cluster_tree = adaptive_cluster_algorithm(
                    training_data=features,
                    eval_function=oracle,
                    atol=atol,
                    rtol=rtol,
                    N_init_clusters=N_init_clusters,
                    min_ppc=min_ppc,
                    use_gradient=True,
                    function_vals=labels,
                    seed=seed,
                )

                # Extract flattened arrays of centroids, values, gradients, hessians, radii.
                new_features, new_labels, new_gradients, new_hessians, new_dims = (
                    flatten_cluster_centers_with_gradient(
                        nc=k,
                        n=self.n,
                        m=self.m,
                        tree=cluster_tree,
                    )
                )
            else:
                k, cluster_tree = aca.adaptive_cluster_algorithm(
                    X_train=features,
                    f=oracle,
                    atol=atol,
                    rtol=rtol if rtol is not None else 0.0001,
                    N_init_clusters=N_init_clusters,
                    min_ppc=min_ppc,
                    use_gradient=True,
                    f_vals=labels,
                    seed=seed if seed is not None else 0,
                )

                new_features, new_labels, new_gradients, new_hessians, new_dims = (
                    aca.flatten_cluster_centers_with_gradient(
                        nc=k,
                        n=self.n,
                        m=self.m,
                        tree=cluster_tree,
                    )
                )

        else:
            # Use raw data as sources (no clustering).
            new_features = features
            new_labels = labels
            new_gradients = gradients
            new_hessians = None
            new_dims = None
            k = len(features)

        # --- Build FAISS-backed nearest-neighbor inference object ---
        self.KNN_search = NN_Inference(
            features=new_features,
            labels=new_labels,
            gradients=new_gradients,
            hessians=new_hessians,
            trust_radii=new_dims,
        )
        self._model_trained = True
        self.N_sources = k

        # --- Optional: heuristic guarantee ---
        # When clustering and a global L are provided, we can repeatedly check
        # original features against the guarantee and add the violations back.
        if do_clustering and L_global and check_guarantee_N_times > 0:
            for _ in range(check_guarantee_N_times):
                bad_features = self._test_guarantee(
                    targets=features, tol=atol, L_global=L_global, alpha=alpha
                )
                self.add_sources(features=features[bad_features])

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
        (labels, gradients), and delegates to `train_model_from_data()`.

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
            features = np.random.uniform(l, r, (N, n))
        elif distribution == "LHS":
            sampler = qmc.LatinHypercube(d=n, rng=seed)
            features = sampler.random(n=N)
            features = qmc.scale(features, l, r)

        # --- Evaluate oracle at sampled features to get (labels, gradients) ---
        labels = np.empty((len(features), m))
        gradients = np.empty((len(features), m, n))
        for jj, feature in enumerate(features):
            labels[jj], gradients[jj] = oracle(feature)

        # --- Delegate to the array-based trainer (preserving original args order) ---
        self.train_model_from_data(
            features=features,
            labels=labels,
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
          targets: Array of evaluation points; shape (N, n) or (n,) for 1D.
          k: Number of nearest neighbors to use in prediction.
          disp_time: If True, print wall-clock inference time from backend.
          L: Optional global Lipschitz constant for guarantees; forwarded to
            the backend inference.
          recluster: If True, points outside the guarantee will be passed to
            adaptive clustering and appended to the neighbor index.
          alpha: Multiplicative factor for Hessian-based trust bounds.

        Returns:
          tuple[np.ndarray, np.ndarray]:
            - predictions: (N, m) surrogate outputs,
            - unguaranteed_points: (N,) boolean mask; True where outside trust.

        Notes:
          * Requires that `train_model_from_*` has been called successfully.
          * Guarantee logic currently applies in the backend primarily for k==1.
        """
        if not self._model_trained:
            raise RuntimeError(
                "Model must be trained before calling evaluate(). Run train_model_from_data() or train_model_from_sampling() first."
            )

        if targets.ndim == 1:
            predictions, unguaranteed_points = self.KNN_search.predict(
                targets=targets.reshape(len(targets), 1),
                k=k,
                disp_time=disp_time,
                L_global=L,
                atol=self.atol,
                rtol=self.rtol,
                alpha=alpha,
            )
        else:
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
            self.add_sources(features=targets[unguaranteed_points])

        return predictions, unguaranteed_points

    def get_sources(self) -> np.ndarray:
        """Return the current source feature matrix used in kNN searches.

        Returns:
          np.ndarray: Array of source points with shape (P_sources, n).
        """
        return self.KNN_search.features

    def add_sources(self, features: np.ndarray) -> None:
        """Adaptively cluster new candidate points and append them as sources.

        This is typically called on the subset of targets that failed the
        guarantee check during `evaluate(recluster=True)`.

        Args:
          features: Array of new candidate points with shape (Q, n).

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

        if not self.use_accelerated_clustering:
            k, cluster_tree = adaptive_cluster_algorithm(
                training_data=features,
                eval_function=self.oracle,
                atol=self.atol,
                rtol=self.rtol,
                N_init_clusters=2,
                min_ppc=10,
                use_gradient=True,
            )
            new_features, new_labels, new_gradients, new_hessians, new_dims = (
                flatten_cluster_centers_with_gradient(
                    nc=k,
                    n=self.n,
                    m=self.m,
                    tree=cluster_tree,
                )
            )
        else:
            k, cluster_tree = aca.adaptive_cluster_algorithm(
                X_train=features,
                f=self.oracle,
                atol=self.atol,
                rtol=self.rtol if self.rtol is not None else 0.0001,
                N_init_clusters=2,
                min_ppc=10,
                use_gradient=True,
                seed=0,
            )
            new_features, new_labels, new_gradients, new_hessians, new_dims = (
                aca.flatten_cluster_centers_with_gradient(
                    nc=k,
                    n=self.n,
                    m=self.m,
                    tree=cluster_tree,
                )
            )

        self.KNN_search.add_extra_features(
            new_features, new_labels, new_gradients, new_hessians, new_dims
        )
        self.N_sources += len(new_features)

    def _test_guarantee(
        self, targets: np.ndarray, tol: float, L_global: float, alpha: float
    ) -> np.ndarray:
        """Return a boolean mask of targets that violate the local/global guarantee.

        This helper queries the nearest neighbor from the FAISS index and checks
        whether each target lies within either a local trust radius (based on the
        stored cluster radius and Hessian proxy) or a global Lipschitz-based bound.

        Args:
          targets: Array of target points with shape (N, n).
          tol: Absolute tolerance used in the bound.
          L_global: Global Lipschitz constant for the fallback bound.
          alpha: Multiplicative factor for Hessian-based trust bounds.

        Returns:
          np.ndarray: Boolean mask of shape (N,), True where the target is
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
