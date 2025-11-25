"""Adaptive clustering with binary tree structure."""

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.cluster import KMeans

from src.two_means import two_means_pca_threshold

try:
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans

    _HAS_GPU = cp.is_available()
except Exception:
    _HAS_GPU = False
    warnings.warn("CuPy or cuML not available; GPU acceleration will be disabled.")

CP_DTYPE = cp.float64 if _HAS_GPU else None  # type: ignore
NP_DTYPE = np.float64

logger = logging.getLogger(__name__)
logger.info(f"GPU acceleration available: {_HAS_GPU}")

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
        atol: float = 1e-2,
        rtol: float | None = None,
        num_initial_clusters: int = 10,
        min_points: int = 1,
        seed: int = 0,
        *,
        # Keyword arguments
        use_gpu: bool = True,
        gpu_len_threshold: int = 10_000,
        should_generate_extra_points: bool = True,
        two_means_split_method: str = "optimal",
    ):
        """
        Initialize the adaptive clusterer.

        Args:
            atol: Absolute tolerance for cluster acceptance.
            rtol: Relative tolerance for cluster acceptance. If None, only absolute tolerance is used.
            num_initial_clusters: Number of initial clusters.
            min_points: Minimum points per cluster.
            seed: Random seed for reproducibility.
            use_gpu: Whether to use GPU acceleration if available.
            gpu_len_threshold: Minimum data length to use GPU clustering. The value for this was decided based on empirical benchmarks from: research/benchmark_kmeans_cpu_vs_gpu.py
            should_generate_extra_points: Whether to generate extra points for small clusters.
            two_means_split_method: Method for splitting in TwoMeans ('optimal' or 'median').
        """
        self.atol = atol
        self.rtol = rtol
        self.num_initial_clusters = num_initial_clusters
        self.min_points = min_points
        self.seed = seed

        self.use_gpu = use_gpu and _HAS_GPU
        self.gpu_len_threshold = gpu_len_threshold
        self.should_generate_extra_points = should_generate_extra_points
        logger.info(
            f"Should generate extra points: {self.should_generate_extra_points}"
        )
        self.two_means_split_method = two_means_split_method

        self._cluster_id = 0

    def _get_function_values(
        self, X: np.ndarray, f: Callable, cupy_f: Callable | None
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
        grad_f: Callable | None,
        cupy_f: Callable | None,
    ) -> tuple[bool, float, float, float, np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Check if cluster violates Taylor approximation error bounds.

        Returns:
            violated: whether cluster violates tolerances
            max_error: maximum error in cluster
            mean_error: mean error in cluster
            radius: cluster radius
            centroid: (1, D) cluster centroid
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
        if J_c is not None:
            F_taylor = F_c + np.einsum("ijk,nk->nj", J_c, dx)
        else:
            F_taylor = F_c

        err = np.linalg.norm(F_cluster - F_taylor, axis=1)

        # Tolerance per point
        if self.rtol is None:
            thr = self.atol
        else:
            thr = np.maximum(self.atol, self.rtol * np.linalg.norm(F_cluster, axis=1))
        violated_mask = err > thr
        violated = bool(violated_mask.any())

        radius = (
            float(np.max(np.linalg.norm(X_cluster - c, axis=1)))
            if cluster_size > 0
            else 0.0
        )
        max_error = float(err.max()) if cluster_size > 0 else 0.0
        mean_error = float(err.mean()) if cluster_size > 0 else 0.0
        return violated, max_error, mean_error, radius, c, F_c, J_c

    def fit(
        self,
        X: np.ndarray,
        f: Callable,
        grad_f: Callable | None,
        cupy_f: Callable | None,
        debug: bool = False,
    ) -> list[Cluster]:
        """
        Main clustering algorithm.

        Args:
            X: Input data points with shape (n, d) where n=number of points, d=input dimensions.
            f: Function to evaluate. Should return values with shape (n, fd) where fd=output dimensions.
            grad_f: Optional function to compute gradients. Should return gradients with shape (n, fd, d).

        Returns:
            List of tree nodes representing the clustered data.
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
                if cluster_size <= 1:
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

                violated, max_error, mean_error, radius, c, F_c, J_c = (
                    self._cluster_violation(X_cluster, f, grad_f, cupy_f)
                )

                if not violated or force_add:
                    # Accept cluster
                    h_proxy = 2.0 * max_error / (radius**2) if radius > 0 else 0.0
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
                        )
                    )
                else:
                    # Split based on cluster size via TwoMeans or CuML KMeans
                    if cluster_size < self.gpu_len_threshold:
                        X_norm, mu = self._normalize_data(X_cluster)
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
                        X_norm, mu = self._normalize_data(X_cluster)
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
        return final_clusters


def adaptive_cluster_algorithm(
    X_train: np.ndarray,
    f: Callable,
    grad_f: Callable | None,
    cupy_f: Callable | None = None,
    atol: float = 0.01,
    rtol: float | None = 0.01,
    num_initial_clusters=10,
    min_pts_per_cluster=1,
    seed=0,
    should_generate_extra_points: bool = True,
    debug: bool = False,
    two_means_split_method: str = "optimal",
) -> list[Cluster]:
    """Run the adaptive clustering algorithm and return the list of clusters."""
    clusterer = AdaptiveClusterer(
        atol=atol,
        rtol=rtol,
        num_initial_clusters=num_initial_clusters,
        min_points=min_pts_per_cluster,
        seed=seed,
        should_generate_extra_points=should_generate_extra_points,
        two_means_split_method=two_means_split_method,
    )
    clusters = clusterer.fit(X=X_train, f=f, grad_f=grad_f, cupy_f=cupy_f, debug=debug)
    return clusters


def plot_clusters(clusters: list[Cluster], fname: str = "adaptive_clusters.png"):
    """Plot 2D clusters from the tree nodes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    plt.figure(figsize=(8, 8))
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
    plt.title("Adaptive Clusters")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(fname)
    plt.close()


if __name__ == "__main__":
    import time
    from pathlib import Path

    FILE_NAME = Path(__file__).resolve().stem

    def sample_function(X) -> np.ndarray:
        """
        f(x) = [x1^2 + x2^2, x1 - x2 + x3]
        """
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        val0 = x1**2 + x2**2
        val1 = x1 - x2 + x3
        values = np.stack([val0, val1], axis=1)
        return values

    def sample_gradient(X) -> np.ndarray:
        """
        Gradient of sample_function.
        Jacobian/Gradient:
        [[2*x1, 2*x2, 0],
         [1,    -1,  1]]
        """
        x1, x2, _ = X[:, 0], X[:, 1], X[:, 2]
        g0 = np.stack([2 * x1, 2 * x2, np.zeros_like(x1)], axis=1)
        g1 = np.stack([np.ones_like(x1), -np.ones_like(x1), np.ones_like(x1)], axis=1)
        return np.stack([g0, g1], axis=1)  # (B, 2, 3)

    def sample_cupy_function(X) -> cp.ndarray:  # type: ignore
        """
        f(x) = [x1^2 + x2^2, x1 - x2 + x3] using CuPy
        """
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        val0 = x1**2 + x2**2
        val1 = x1 - x2 + x3
        values = cp.stack([val0, val1], axis=1)  # type: ignore
        return values

    # Generate sample data
    seed = 0
    np.random.seed(seed)
    N_SAMPLES = 1_000_000
    N_INIT_CLUSTERS = 100
    MIN_PTS_PER_CLUSTER = 5
    ABS_TOL = 0.01
    REL_TOL = 0.01
    print(f"{ABS_TOL=}, {REL_TOL=}")
    print(f"{N_INIT_CLUSTERS=}, {MIN_PTS_PER_CLUSTER=}")
    D = 3

    X_samples = np.random.random((N_SAMPLES, D))
    print(f"Generated {N_SAMPLES:,} sample data points.")

    # cprofile = cProfile.Profile()
    # cprofile.enable()

    # Run adaptive clustering
    tic = time.perf_counter()
    clusters = adaptive_cluster_algorithm(
        X_train=X_samples,
        f=sample_function,
        grad_f=sample_gradient,
        cupy_f=sample_cupy_function,
        atol=ABS_TOL,
        rtol=REL_TOL,
        num_initial_clusters=N_INIT_CLUSTERS,
        min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
        seed=seed,
        should_generate_extra_points=False,
    )
    toc = time.perf_counter()
    logger.info(f"Total clusters formed: {len(clusters)}")
    total = (
        NUM_CLUSTERED_BY_CUML + NUM_CLUSTERED_BY_SKLEARN + NUM_CLUSTERED_BY_TWO_MEANS
    )
    logger.info(f"Clusters by CUML: {NUM_CLUSTERED_BY_CUML}")
    logger.info(f"Clusters by SKLearn KMeans: {NUM_CLUSTERED_BY_SKLEARN}")
    logger.info(f"Clusters by TwoMeans: {NUM_CLUSTERED_BY_TWO_MEANS}")
    logger.info(f"Total clustering calls: {total}")
    logger.info(f"Time taken: {toc - tic:.4f} seconds")
