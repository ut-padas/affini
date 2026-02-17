"""Adaptive clustering with binary tree structure."""

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from cuml.cluster import KMeans as cuKMeans
from sklearn.cluster import KMeans
from typing_extensions import Literal

from src import IS_CUPY_BACKEND
from src.two_means import estimate_hessian_abs_sqrt, two_means_pca_threshold

if IS_CUPY_BACKEND:
    import cupy as cp

    CP_DTYPE = cp.float64
else:
    cp = None  # type: ignore
    CP_DTYPE = None


NP_DTYPE = np.float64

logger = logging.getLogger(__name__)
logger.info(f"GPU acceleration available: {IS_CUPY_BACKEND}")

# Global counters for clustering method usage
NUM_CLUSTERED_BY_CUML = 0
NUM_CLUSTERED_BY_SKLEARN = 0
NUM_CLUSTERED_BY_TWO_MEANS = 0


@dataclass
class Cluster:
    """Data class to hold cluster information."""

    centroid: np.ndarray  # (1, D)
    F_centroid: np.ndarray  # (1, FD)
    J_centroid: Optional[np.ndarray]  # (1, FD, D)
    radius: float
    hessian_proxy: float
    cluster_size: int
    max_error: float
    mean_error: float
    lipschitz_obs: float


@dataclass
class AdaptiveClusterer:
    """
    Adaptive clustering.

    Shape Conventions:
        - x.shape: (n, d) where n=number of points, d=input dimensions
        - f.shape: (n, fd) where fd=output dimensions (function values)
        - grad.shape: (n, fd, d) where gradients are Jacobian matrices

    Note: If fd=1 (scalar function), f.shape=(n, 1) and grad.shape=(n, 1, d)
    """

    def __init__(
        self,
        tol: float,
        num_initial_clusters: int = 10,
        min_points: int = 1,
        seed: int = 0,
        *,
        # Keyword arguments
        use_gpu: bool = True,
        gpu_len_threshold: int = 10_000,
        should_generate_extra_points: bool = True,
        two_means_split_method: Literal["optimal", "median"] = "median",
        use_hessian_informed_split: bool = True,
    ):
        """
        Initialize the adaptive clusterer.

        Args:
            tol: Tolerance for cluster acceptance.
            num_initial_clusters: Number of initial clusters.
            min_points: Minimum points per cluster.
            seed: Random seed for reproducibility.
            use_gpu: Whether to use GPU acceleration if available.
            gpu_len_threshold: Minimum data length to use GPU clustering. The value for this was decided based on empirical benchmarks from: research/benchmark_kmeans_cpu_vs_gpu.py
            should_generate_extra_points: Whether to generate extra points for small clusters.
            two_means_split_method: Method for splitting in TwoMeans ('optimal' or 'median').
            use_hessian_informed_split: Whether to use Hessian-informed splitting.
        """
        self.tol = tol
        self.num_initial_clusters = num_initial_clusters
        self.min_points = min_points
        self.seed = seed

        self.use_gpu = use_gpu and IS_CUPY_BACKEND
        self.gpu_len_threshold = gpu_len_threshold
        self.should_generate_extra_points = should_generate_extra_points
        logger.info(
            f"Should generate extra points: {self.should_generate_extra_points}"
        )
        self.two_means_split_method = two_means_split_method
        self.use_hessian_informed_split = use_hessian_informed_split

        self._cluster_id = 0

    def _get_function_values(
        self, X: np.ndarray, f: Callable, cupy_f: Optional[Callable]
    ) -> np.ndarray:
        """
        Compute function values.

        Args:
            X: Input points with shape (n, d)
            f: Function that returns values with shape (n, fd).

        Returns:
            Function values with shape (n, fd)
        """
        if self.use_gpu and cupy_f is not None and len(X) >= self.gpu_len_threshold:
            Xg = cp.asarray(X, dtype=CP_DTYPE)  # type: ignore
            val = cupy_f(Xg)
            return cp.asnumpy(val).astype(NP_DTYPE)  # type: ignore
        else:
            val = f(X)
            return np.array(val, dtype=NP_DTYPE)

    def _normalize_data(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Z-score normalize the given data.

        Args:
            X: Data to normalize.

        Returns:
            Normalized data and mean.
        """
        mean = X.mean(axis=0)
        return (X - mean), mean

    def _initial_cluster(self, X: np.ndarray) -> np.ndarray:
        """Perform initial clustering."""
        global NUM_CLUSTERED_BY_CUML, NUM_CLUSTERED_BY_SKLEARN

        X_norm, _ = self._normalize_data(X)

        n = len(X)
        if self.use_gpu and n >= self.gpu_len_threshold:
            Xg = cp.asarray(X_norm, dtype=CP_DTYPE)  # type: ignore
            km = cuKMeans(n_clusters=self.num_initial_clusters, random_state=self.seed)  # type: ignore
            labels = km.fit_predict(Xg).get()
            NUM_CLUSTERED_BY_CUML += 1
            return labels

        km = KMeans(
            n_clusters=self.num_initial_clusters, random_state=self.seed, n_init=1
        )
        NUM_CLUSTERED_BY_SKLEARN += 1

        labels = km.fit_predict(X_norm)
        return labels

    def _generate_extra_points(
        self,
        X_cluster: np.ndarray,
        cluster_size: int,
    ) -> np.ndarray:
        n_extra = max(1, self.min_points - cluster_size + 1)
        lambdas = np.random.random(size=n_extra)

        idx0 = np.random.randint(0, cluster_size, size=n_extra)
        offsets = np.random.randint(1, cluster_size, size=n_extra)
        idx1 = (idx0 + offsets) % cluster_size
        new_X = (
            lambdas[:, None] * X_cluster[idx0]
            + (1 - lambdas)[:, None] * X_cluster[idx1]
        )
        return new_X

    def _get_centroid(self, X: np.ndarray) -> np.ndarray:
        """Compute centroid of given points of a cluster."""
        return X.mean(axis=0)

    def _cluster_violation(
        self,
        X_cluster: np.ndarray,
        f: Callable,
        grad_f: Optional[Callable],
        cupy_f: Optional[Callable],
        error_metric: Literal["L2", "Linf-global", "Linf-local"],
        F_inf: Optional[float],
    ) -> tuple[
        bool,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
    ]:
        """
        Check if cluster violates Taylor approximation error bounds.

        Returns:
            acceptable: whether cluster is acceptable
            radius: cluster radius
            centroid: (1, D) cluster centroid
            error_vec: (n,) error scalar for each point in cluster
            F_c: (1, FD) function value at centroid
            J_c: (1, FD, D) Jacobian at centroid
        """
        cluster_size = X_cluster.shape[0]
        if cluster_size == 0:
            raise ValueError("Empty cluster encountered.")

        c = self._get_centroid(X_cluster)[None, :]
        F_cluster = self._get_function_values(X_cluster, f, cupy_f)
        F_c = f(c)
        J_c = grad_f(c) if grad_f is not None else None

        dx = X_cluster - c
        radius = float(np.max(np.linalg.norm(dx, axis=1)))
        if J_c is not None:
            F_taylor = F_c + np.einsum("ijk,nk->nj", J_c, dx)
        else:
            F_taylor = F_c

        if error_metric == "L2":
            error_vec = np.linalg.norm(F_cluster - F_taylor, axis=1)
            error_vec_norm = np.linalg.norm(error_vec, 2)

            # We use Frobenius norm, since F_cluster is (n, fd)
            # and it is equivalent to sqrt(sum(norm(v, axis=1) ** 2))
            F_cluster_norm = np.linalg.norm(F_cluster, "fro")
            if F_cluster_norm > 0:
                local_error_ratio = error_vec_norm / F_cluster_norm
            else:
                local_error_ratio = float("inf")

            # Check if relative error is within tolerance
            acceptable = bool(local_error_ratio <= self.tol)

        elif error_metric == "Linf-global" and F_inf is not None:
            error_vec = np.linalg.norm(F_cluster - F_taylor, ord=np.inf, axis=1)
            max_error = float(np.max(error_vec))
            e_rel = max_error / F_inf
            acceptable = bool(e_rel <= self.tol)
        elif error_metric == "Linf-local":
            error_vec = np.linalg.norm(F_cluster - F_taylor, ord=np.inf, axis=1)
            max_error = float(np.max(error_vec))

            # F_cluster shape is (N, F_dim). We take norm over F_dim, then max over N points.
            F_cluster_abs = np.linalg.norm(F_cluster, ord=np.inf, axis=1)
            local_F_max = float(np.max(F_cluster_abs))

            # Compute Relative Error with safety
            normalization = max(local_F_max, 1e-12)
            e_rel = max_error / normalization
            acceptable = bool((e_rel <= self.tol) or (max_error <= self.tol))
        else:
            raise ValueError(f"Invalid error_split_criterion: {error_metric}")

        return acceptable, radius, c, error_vec, F_c, J_c

    def fit(
        self,
        X: np.ndarray,
        f: Callable,
        grad_f: Optional[Callable],
        cupy_f: Optional[Callable],
        error_metric: Literal["L2", "Linf-global", "Linf-local"],
    ) -> tuple[list[Cluster], int]:
        """
        Main clustering algorithm.

        Args:
            X: Input data points with shape (n, d) where n=number of points, d=input dimensions.
            f: Function to evaluate. Should return values with shape (n, fd) where fd=output dimensions.
            grad_f: Optional function to compute gradients. Should return gradients with shape (n, fd, d).
            cupy_f: Optional function to evaluate on GPU. Should return values with shape (n, fd).
            error_split_criterion: Criterion for error splitting ('L2' or 'Linf-global' or 'Linf-local').

        Returns:
            List of tree nodes representing the clustered data.
            Number of levels in the clustering tree.
        """
        global NUM_CLUSTERED_BY_TWO_MEANS, NUM_CLUSTERED_BY_CUML, NUM_CLUSTERED_BY_SKLEARN
        # Prepare data - ensure proper shape without forcing 2D
        X = X.astype(NP_DTYPE)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D array with shape (n, d), got shape {X.shape}"
            )

        start_time = time.perf_counter()
        logger.info(f"Starting Adaptive Clustering on {X.shape}")

        # Initial clustering
        tic = time.perf_counter()
        labels = self._initial_cluster(X)
        toc = time.perf_counter()
        logger.info(f"Initial clustering done in {toc - tic:.4f}s")

        # Initialize queue with point arrays from initial clusters
        cluster_ids = np.unique(labels)
        queue: list[np.ndarray] = []
        for cid in cluster_ids:
            pts_in_cluster = X[labels == cid]
            queue.append(pts_in_cluster)

        final_clusters: list[Cluster] = []
        n_level = 0

        if error_metric != "L2":
            # Get max of f over entire dataset for Linf criterion
            F_all = self._get_function_values(X, f, cupy_f)
            # CCompute infinity norm: max_x |f(x)|_oo
            F_inf = float(np.max(np.linalg.norm(F_all, ord=np.inf, axis=1)))
        else:
            F_inf = None

        # BFS by levels
        while queue:
            level_start = time.perf_counter()
            logger.debug(f"Processing level {n_level}, queue size: {len(queue)}")
            level = queue
            queue = []

            children_left: list[np.ndarray] = []
            children_right: list[np.ndarray] = []

            children_left: list[np.ndarray] = []
            children_right: list[np.ndarray] = []

            for X_cluster in level:
                cluster_size = len(X_cluster)
                if cluster_size <= 2:
                    final_clusters.append(
                        Cluster(
                            centroid=np.squeeze(X_cluster[0]),
                            F_centroid=f(X_cluster[0:1]),
                            J_centroid=(
                                grad_f(X_cluster[0:1]) if grad_f is not None else None
                            ),
                            radius=0.0,
                            hessian_proxy=0.0,
                            cluster_size=1,
                            max_error=0.0,
                            mean_error=0.0,
                            lipschitz_obs=0.0,
                        )
                    )
                    continue

                force_add = False
                if cluster_size < self.min_points:
                    if self.should_generate_extra_points:
                        # Generate extra points
                        new_X = self._generate_extra_points(X_cluster, cluster_size)
                        X_cluster = np.vstack([X_cluster, new_X])
                        cluster_size = len(X_cluster)
                    else:
                        # Append to final clusters directly
                        force_add = True

                acceptable, radius, c, error_vec, F_c, J_c = self._cluster_violation(
                    X_cluster, f, grad_f, cupy_f, error_metric, F_inf
                )

                if acceptable or force_add:
                    max_error = float(np.max(error_vec))
                    mean_error = float(np.mean(error_vec))
                    h_proxy = 2.0 * max_error / (radius**2) if radius > 0 else 0.0

                    # Calculate Lipschitz observation - NEW STRATEGY: Scan all points to find max curvature
                    L_obs = 0.0
                    if grad_f is not None and cluster_size > 1:
                        # 1. Compute Gradients for ALL points in cluster
                        # Note: In production, you might only sample subset for speed
                        J_all = grad_f(X_cluster)  # (N, FD, D)

                        # 2. Vectorized Difference Calculation
                        # J_c is (1, FD, D) -> Broadcast subtract
                        diff_grad = np.linalg.norm(
                            (J_all - J_c).reshape(cluster_size, -1), axis=1
                        )
                        diff_x = np.linalg.norm(
                            (X_cluster - c).reshape(cluster_size, -1), axis=1
                        )

                        # 3. Compute Ratios
                        # Filter out tiny diff_x to avoid division by zero (centroid itself)
                        valid_mask = diff_x > 1e-9
                        if np.any(valid_mask):
                            ratios = diff_grad[valid_mask] / diff_x[valid_mask]
                            # The max ratio corresponds to the steepest direction
                            L_obs = float(np.max(ratios))

                    final_clusters.append(
                        Cluster(
                            centroid=np.squeeze(c),
                            F_centroid=F_c,
                            J_centroid=J_c,
                            radius=radius,
                            hessian_proxy=h_proxy,
                            cluster_size=cluster_size,
                            max_error=max_error,
                            mean_error=mean_error,
                            lipschitz_obs=float(L_obs),
                        )
                    )
                else:
                    # Split based on cluster size via TwoMeans or CuML KMeans
                    X_norm, mu = self._normalize_data(X_cluster)

                    if self.use_hessian_informed_split and grad_f is not None:
                        # Center the gradient estimates
                        Grad_norm = grad_f(X_cluster) - grad_f(mu[None, :])
                        Grad_norm = Grad_norm[:, 0, :]
                        M = estimate_hessian_abs_sqrt(
                            X=X_norm,
                            G=Grad_norm,
                        )

                        # Transform data into Hessian-informed space
                        X_norm = X_norm @ M

                    if cluster_size < self.gpu_len_threshold:
                        labels, _ = two_means_pca_threshold(
                            X_norm,
                            mean=mu,
                            split_method=self.two_means_split_method,
                        )
                        NUM_CLUSTERED_BY_TWO_MEANS += 1

                        left = X_cluster[labels == 0]
                        right = X_cluster[labels == 1]
                    elif self.use_gpu and cluster_size >= self.gpu_len_threshold:
                        Xg = cp.asarray(X_cluster, dtype=CP_DTYPE)  # type: ignore
                        km = cuKMeans(n_clusters=2, random_state=self.seed)  # type: ignore
                        labels = km.fit_predict(Xg).get()
                        NUM_CLUSTERED_BY_CUML += 1

                        left = X_cluster[labels == 0]
                        right = X_cluster[labels == 1]
                    else:
                        km = KMeans(n_clusters=2, random_state=self.seed, n_init=1)
                        labels = km.fit_predict(X_norm)
                        NUM_CLUSTERED_BY_SKLEARN += 1

                        left = X_cluster[labels == 0]
                        right = X_cluster[labels == 1]

                    children_left.append(left)
                    children_right.append(right)

            queue.extend(children_left)
            queue.extend(children_right)
            logger.debug(f"Bad Clusters: {len(children_left)}")
            elapsed_level = time.perf_counter() - level_start
            logger.debug(f"Level {n_level} done in {elapsed_level:.4f}s")
            n_level += 1
        elapsed = time.perf_counter() - start_time
        logger.info(f"Finished in {elapsed:.4f}s")
        logger.info(f"Final Cluster Count: {len(final_clusters)}")
        return final_clusters, n_level


def adaptive_cluster_algorithm(
    X_train: np.ndarray,
    f: Callable,
    grad_f: Optional[Callable],
    cupy_f: Optional[Callable] = None,
    tol: float = 0.01,
    num_initial_clusters=10,
    min_pts_per_cluster=1,
    seed=0,
    should_generate_extra_points: bool = True,
    error_metric: Literal["L2", "Linf-global", "Linf-local"] = "L2",
    two_means_split_method: Literal["optimal", "median"] = "median",
    use_hessian_informed_split: bool = True,
) -> tuple[list[Cluster], int]:
    """Run the adaptive clustering algorithm and return the list of clusters."""
    clusterer = AdaptiveClusterer(
        tol=tol,
        num_initial_clusters=num_initial_clusters,
        min_points=min_pts_per_cluster,
        seed=seed,
        should_generate_extra_points=should_generate_extra_points,
        two_means_split_method=two_means_split_method,
        use_hessian_informed_split=use_hessian_informed_split,
    )
    clusters, num_levels = clusterer.fit(
        X=X_train,
        f=f,
        grad_f=grad_f,
        cupy_f=cupy_f,
        error_metric=error_metric,
    )
    return clusters, num_levels


def plot_clusters(
    clusters: list[Cluster], f, domain, fname: str = "adaptive_clusters.png"
):
    """Plot 2D clusters from the tree nodes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    x = np.linspace(domain[0], domain[1], 200)
    y = np.linspace(domain[0], domain[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = f(np.vstack([X.ravel(), Y.ravel()]).T).reshape(X.shape)

    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar()

    ax = plt.gca()
    for cluster in clusters:
        centroid = cluster.centroid
        radius = cluster.radius
        circle = Circle(
            (centroid[0], centroid[1]),
            radius,
            color="blue",
            fill=False,
            alpha=0.5,
        )
        ax.add_patch(circle)
        plt.plot(centroid[0], centroid[1], "ro")  # plot centroid

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f"Adaptive Clusters (N={len(clusters)})")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(fname)
    plt.close()


if __name__ == "__main__":
    import time
    from functools import partial
    from pathlib import Path

    from src.problems.shifted_gaussians import (
        CENTERS_2D,
        shifted_gaussians,
        shifted_gaussians_grad,
    )

    sg_fn = partial(shifted_gaussians, centers=CENTERS_2D)
    sg_grad_fn = partial(shifted_gaussians_grad, centers=CENTERS_2D)

    FILE_NAME = Path(__file__).resolve().stem

    # Generate sample data
    seed = 0
    np.random.seed(seed)
    N_SAMPLES = 100_000
    N_INIT_CLUSTERS = 1
    MIN_PTS_PER_CLUSTER = 5
    USE_HESSIAN_INFORMED_SPLIT = False
    ERROR_SPLIT_CRITERION = "L2"
    TOL = 0.01
    print(f"{TOL=}, {TOL=}")
    print(f"{N_INIT_CLUSTERS=}, {MIN_PTS_PER_CLUSTER=}")
    D = 2

    DOMAIN = [-2, 2]

    X_samples = np.random.uniform(
        low=DOMAIN[0], high=DOMAIN[1], size=(N_SAMPLES, D)
    ).astype(np.float64)
    print(f"Generated {N_SAMPLES:,} sample data points.")

    # Run adaptive clustering
    tic = time.perf_counter()
    clusters, num_levels = adaptive_cluster_algorithm(
        X_train=X_samples,
        f=sg_fn,
        grad_f=sg_grad_fn,
        cupy_f=sg_fn,
        tol=TOL,
        num_initial_clusters=N_INIT_CLUSTERS,
        min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
        seed=seed,
        should_generate_extra_points=True,
        use_hessian_informed_split=USE_HESSIAN_INFORMED_SPLIT,
        error_metric=ERROR_SPLIT_CRITERION,
    )
    toc = time.perf_counter()
    print(f"Total clusters formed: {len(clusters):,}")
    total = (
        NUM_CLUSTERED_BY_CUML + NUM_CLUSTERED_BY_SKLEARN + NUM_CLUSTERED_BY_TWO_MEANS
    )
    logger.info(f"Clusters by CUML: {NUM_CLUSTERED_BY_CUML}")
    logger.info(f"Clusters by SKLearn KMeans: {NUM_CLUSTERED_BY_SKLEARN}")
    logger.info(f"Clusters by TwoMeans: {NUM_CLUSTERED_BY_TWO_MEANS}")
    logger.info(f"Total clustering calls: {total}")
    logger.info(f"Time taken: {toc - tic:.4f} seconds")
    logger.info(f"Time taken: {toc - tic:.4f} seconds")
    logger.info(f"Time taken: {toc - tic:.4f} seconds")

    if D == 2 and len(clusters) < 1000:
        fname = f"old_ac_clusters_gaussian_{N_SAMPLES}_pts_tol{TOL}_{USE_HESSIAN_INFORMED_SPLIT}_{ERROR_SPLIT_CRITERION}.png"
        plot_clusters(clusters, sg_fn, domain=DOMAIN, fname=fname)
        print(f"Cluster plot saved to {fname}")

    npz_fpath = f"v1_adaptive_clusters_gaussian_train_{N_SAMPLES}_pts_tol_{TOL}.npz"
    results = {
        "centroids": np.array([cluster.centroid for cluster in clusters]),
        "f": np.array([cluster.F_centroid for cluster in clusters]),
        "g": np.array([cluster.J_centroid for cluster in clusters]),
        "radii": np.array([cluster.radius for cluster in clusters]),
    }
    np.savez_compressed(npz_fpath, **results)
    print(f"Cluster data saved to {npz_fpath}")
