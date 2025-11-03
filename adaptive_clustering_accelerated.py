"""Adaptive clustering with binary tree structure."""

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from src.two_means import two_means_pca_threshold

try:
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans

    _HAS_GPU = cp.is_available()
except Exception:
    _HAS_GPU = False
    warnings.warn("CuPy or cuML not available; GPU acceleration will be disabled.")

CP_DTYPE = cp.float64 if _HAS_GPU else None
NP_DTYPE = np.float64
print(f"CUML available: {_HAS_GPU}")

# Global counters for clustering method usage
NUM_CLUSTERED_BY_CUML = 0
NUM_CLUSTERED_BY_SKLEARN = 0
NUM_CLUSTERED_BY_MINIBATCH = 0
NUM_CLUSTERED_BY_TWO_MEANS = 0


@dataclass
class Cluster:
    """Leaf cluster with computed surrogate model parameters."""

    index: int
    centroid: np.ndarray
    value: np.ndarray
    gradient: Optional[np.ndarray] = None
    hessian: float = 0.0
    radius: float = 0.0


@dataclass
class TreeNode:
    """Binary tree node: either leaf (cluster) or internal (left/right children)."""

    cluster: Optional[Cluster] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    def leaves(self) -> list[Cluster]:
        """Collect all leaf clusters from subtree."""
        if self.cluster:
            return [self.cluster]
        leaves = []
        if self.left:
            leaves.extend(self.left.leaves())
        if self.right:
            leaves.extend(self.right.leaves())
        return leaves


class AdaptiveClusterer:
    """
    Adaptive clustering with binary tree splitting.

    Shape Conventions:
        - x.shape: (n, d) where n=number of points, d=input dimensions
        - f.shape: (n, fd) where fd=output dimensions (function values)
        - grad.shape: (n, fd, d) where gradients are Jacobian matrices

    Note: If fd=1 (scalar function), f.shape=(n, 1) and grad.shape=(n, 1, d)
    """

    def __init__(
        self,
        atol: float = 0.01,
        rtol: float = 0.01,
        n_clusters: int = 10,
        min_points: int = 1,
        use_gradient: bool = False,
        seed: int = 0,
        *,
        # Keyword arguments
        use_gpu: bool = True,
        gpu_len_threshold: int = 10_000,
        mini_batch_len_threshold: int = 50_000,
        # Fast k=2 split settings
        use_fast_two_means: bool = True,
        fast_two_means_len_threshold: int | str = "auto",
    ):
        """
        Initialize the adaptive clusterer.

        Args:
            atol: Absolute tolerance for cluster acceptance.
            rtol: Relative tolerance for cluster acceptance.
            n_clusters: Number of initial clusters.
            min_points: Minimum points per cluster.
            use_gradient: Whether to use gradient information.
            seed: Random seed for reproducibility.
            use_gpu: Whether to use GPU acceleration if available.
            gpu_len_threshold: Minimum data length to use GPU clustering. The value for this was decided based on empirical benchmarks from: research/benchmark_kmeans_cpu_vs_gpu.py
            mini_batch_len_threshold: Minimum data length to use MiniBatchKMeans. The value for this was decided based on empirical benchmarks from: research/benchmark_kmeans_vs_minibatch.py
            use_fast_two_means: Whether to use fast 2-means splitting.
            fast_two_means_len_threshold: Maximum data length for fast 2-means splitting. If "auto", then, based on the dimensionality, the model uses thresholds tuned on TACC-Vista system.
        """
        self.atol = atol
        self.rtol = rtol
        self.n_clusters = n_clusters
        self.min_points = min_points
        self.use_gradient = use_gradient
        self.seed = seed

        self.use_gpu = use_gpu and _HAS_GPU
        self.gpu_len_threshold = gpu_len_threshold
        self.mini_batch_len_threshold = mini_batch_len_threshold
        self.use_fast_two_means = use_fast_two_means

        if not (
            isinstance(fast_two_means_len_threshold, int)
            or (
                isinstance(fast_two_means_len_threshold, str)
                and fast_two_means_len_threshold == "auto"
            )
        ):
            raise ValueError(
                "fast_two_means_len_threshold must be 'auto' or an integer."
            )
        self.ftm_len_threshold_property = fast_two_means_len_threshold

        self._cluster_id = 0

    def _get_fast_two_means_len_threshold(self, dim: int) -> int:
        """Determine fast two-means length threshold based on dimensionality, as per tuned values."""
        if isinstance(self.ftm_len_threshold_property, int):
            return self.ftm_len_threshold_property

        if dim < 10:
            return 8_192
        elif dim < 50:
            return 1_024
        else:
            return 512

    def fit(self, X: np.ndarray, f: Callable) -> list[TreeNode]:
        """
        Main clustering algorithm.

        Args:
            X: Input data points with shape (n, d) where n=number of points, d=input dimensions.
            f: Function to evaluate. Should return values with shape (n, fd) where fd=output dimensions.
               If use_gradient=True, should return tuple (values, gradients) where gradients have shape (n, fd, d).

        Returns:
            List of tree nodes representing the clustered data.
        """
        # Prepare data - ensure proper shape without forcing 2D
        X = X.astype(NP_DTYPE)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D array with shape (n, d), got shape {X.shape}"
            )

        F = self._get_function_values(X, f)

        # Get dimensionality and set fast two-means threshold
        dim = X.shape[1]
        self.fast_two_means_len_threshold = self._get_fast_two_means_len_threshold(dim)
        print(
            f"Using fast two-means length threshold (dim={dim}): {self.fast_two_means_len_threshold}"
        )

        # Initial clustering
        labels, centroids = self._initial_cluster(X)

        # Process each initial cluster
        trees = []
        for i in range(self.n_clusters):
            mask = labels == i
            if not np.any(mask):
                # No points in this cluster
                continue

            cluster_X, cluster_F = X[mask], F[mask]

            # Check if initial cluster meets tolerance
            if self._meets_tolerance(cluster_X, cluster_F, centroids[i], f):
                # Create leaf directly
                trees.append(
                    TreeNode(
                        cluster=self._create_cluster(
                            cluster_X, cluster_F, centroids[i], f
                        )
                    )
                )
            else:
                # Split recursively
                tree = self._split_recursive(cluster_X, cluster_F, f)
                if tree:
                    trees.append(tree)

        return trees

    def _get_function_values(self, X: np.ndarray, f: Callable) -> np.ndarray:
        """
        Compute function values.

        Args:
            X: Input points with shape (n, d)
            f: Function that returns values with shape (n, fd) or tuple (values, gradients)

        Returns:
            Function values with shape (n, fd)
        """
        if self.use_gradient:
            val, _ = f(X)
        else:
            val = f(X)
        return np.array(val, dtype=NP_DTYPE)

    def _normalize_data(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Z-score normalize the given data.

        Args:
            X: Data to normalize.

        Returns:
            Normalized data, mean, and std deviation.
        """
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std == 0, 1, std)  # Prevent division by zero
        return (X - mean) / std, mean, std

    def _initial_cluster(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Perform initial clustering."""
        global NUM_CLUSTERED_BY_CUML, NUM_CLUSTERED_BY_SKLEARN, NUM_CLUSTERED_BY_MINIBATCH

        X_norm, mean, std = self._normalize_data(X)

        n = len(X)
        if self.use_gpu and n >= self.gpu_len_threshold:
            Xg = cp.asarray(X_norm, dtype=CP_DTYPE)
            km = cuKMeans(n_clusters=self.n_clusters, random_state=self.seed)
            labels = km.fit_predict(Xg).get()
            centroids = cp.asnumpy(km.cluster_centers_) * std + mean
            NUM_CLUSTERED_BY_CUML += 1
            return labels, centroids

        if n >= self.mini_batch_len_threshold:
            # Calculate batch size based on n
            batch = int(min(8192, max(256, np.sqrt(n))))
            km = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.seed,
                n_init=1,
                # max_iter=50,
                batch_size=batch,
                init="k-means++",
            )
            NUM_CLUSTERED_BY_MINIBATCH += 1
        else:
            km = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.seed,
                n_init=1,
                # max_iter=50,
                # algorithm="elkan",
                init="k-means++",
            )
            NUM_CLUSTERED_BY_SKLEARN += 1

        labels = km.fit_predict(X_norm)
        centroids = km.cluster_centers_ * std + mean
        return labels, centroids

    def _compute_max_error(
        self, X: np.ndarray, F: np.ndarray, centroid: np.ndarray, f: Callable
    ) -> float:
        """
        Compute maximum approximation error in cluster.

        Args:
            X: Points in cluster with shape (n, d)
            F: Function values with shape (n, fd)
            centroid: Cluster centroid with shape (d,)
            f: Function that returns values with shape (n, fd) or tuple (values, gradients)
        """
        # Evaluate function at centroid and compute the errors
        if self.use_gradient:
            f_c, grad_f = f(np.atleast_2d(centroid))  # Ensure centroid is (1, d)

            # Squeeze to get single values: f_c shape (fd,), grad_f shape (fd, d)
            f_c = f_c.squeeze(0)
            grad_f = grad_f.squeeze(0)

            # Taylor model error
            dx = X - centroid  # Shape (n, d)
            f_taylor = f_c + (dx @ grad_f.T)  # Shape (n, fd)
            errors = np.linalg.norm(F - f_taylor, axis=1)
        else:
            f_c = f(np.atleast_2d(centroid))[
                0
            ]  # Ensure centroid is (1, d), extract single value
            errors = np.linalg.norm(F - f_c, axis=1)

        return float(np.max(errors))

    def _meets_tolerance(
        self, X: np.ndarray, F: np.ndarray, centroid: np.ndarray, f: Callable
    ) -> bool:
        """Check if cluster meets the absolute/relative tolerance criteria."""
        error_max = self._compute_max_error(X, F, centroid, f)

        # Absolute tolerance check
        if error_max <= self.atol:
            return True

        # Relative tolerance check
        if len(F) > 0:
            min_norm = np.min(np.linalg.norm(F, axis=1))
            if error_max <= self.rtol * min_norm:
                return True

        return False

    def _create_cluster(
        self, X: np.ndarray, F: np.ndarray, centroid: np.ndarray, f: Callable
    ) -> Cluster:
        """
        Create cluster with computed parameters.

        Args:
            X: Points in cluster with shape (n, d)
            F: Function values with shape (n, fd)
            centroid: Cluster centroid with shape (d,)
            f: Function that returns values with shape (n, fd) or tuple (values, gradients)
        """
        # Evaluate function at centroid
        if self.use_gradient:
            f_c, grad_f = f(np.atleast_2d(centroid))  # Ensure centroid is (1, d)
        else:
            f_c = f(np.atleast_2d(centroid))  # Ensure centroid is (1, d)
            grad_f = None

        # Compute radius and hessian proxy
        radius = np.max(np.linalg.norm(X - centroid, axis=1)) if len(X) > 0 else 0.0
        error_max = self._compute_max_error(X, F, centroid, f)
        hessian = 2 * error_max / (radius**2) if radius > 0 else 0.0

        cluster = Cluster(
            index=self._cluster_id,
            centroid=centroid,
            value=f_c,
            gradient=grad_f,
            hessian=hessian,
            radius=radius,
        )
        self._cluster_id += 1
        return cluster

    def _split_recursive(
        self, X: np.ndarray, F: np.ndarray, f: Callable
    ) -> Optional[TreeNode]:
        """
        Recursively split cluster until tolerance is met.

        Args:
            X: Data points in the cluster.
            F: Function values at data points.
            f: Function to evaluate.

        Returns:
            TreeNode representing the cluster or None if not splittable.
        """
        if len(X) < 2:
            # Base case: not enough points to split
            return None

        # Split into two subclusters
        labels, centroids = self._split_binary(X)

        # Process each subcluster
        left_tree = self._process_subcluster(X, F, labels, 0, centroids[0], f)
        right_tree = self._process_subcluster(X, F, labels, 1, centroids[1], f)

        if left_tree is None and right_tree is None:
            return None

        return TreeNode(left=left_tree, right=right_tree)

    def _split_binary(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Split cluster into two using K-means.

        Args:
            X: Data points in the cluster.

        Returns:
            Cluster labels and centroids.
        """
        global NUM_CLUSTERED_BY_CUML, NUM_CLUSTERED_BY_SKLEARN, NUM_CLUSTERED_BY_MINIBATCH, NUM_CLUSTERED_BY_TWO_MEANS

        n = len(X)
        X_norm, mean, std = self._normalize_data(X)

        # For small clusters, use a rigorous fast 2-means: PCA projection + exact threshold scan
        if self.use_fast_two_means and n <= self.fast_two_means_len_threshold:
            labels, centroids = two_means_pca_threshold(X_norm, mean, std)
            NUM_CLUSTERED_BY_TWO_MEANS += 1
            return labels, centroids

        if self.use_gpu and n >= self.gpu_len_threshold:
            Xg = cp.asarray(X_norm, dtype=CP_DTYPE)
            km = cuKMeans(n_clusters=2, random_state=self.seed)
            labels = km.fit_predict(Xg).get()
            centroids = cp.asnumpy(km.cluster_centers_) * std + mean
            NUM_CLUSTERED_BY_CUML += 1
            return labels, centroids

        if n >= self.mini_batch_len_threshold:
            batch = int(min(8192, max(256, np.sqrt(n))))
            km = MiniBatchKMeans(
                n_clusters=2,
                random_state=self.seed,
                n_init=1,
                # max_iter=50,
                batch_size=batch,
            )
            NUM_CLUSTERED_BY_MINIBATCH += 1
        else:
            km = KMeans(
                n_clusters=2,
                random_state=self.seed,
                n_init=1,
                # max_iter=50,
                # algorithm="elkan",
            )
            NUM_CLUSTERED_BY_SKLEARN += 1

        labels = km.fit_predict(X_norm)
        centroids = km.cluster_centers_ * std + mean
        return labels, centroids

    def _process_subcluster(
        self,
        X: np.ndarray,
        F: np.ndarray,
        labels: np.ndarray,
        label: int,
        centroid: np.ndarray,
        f: Callable,
    ) -> Optional[TreeNode]:
        """
        Process a single subcluster, which involves first adding extra points if needed,
        then checking for dead branches and tolerance, and finally either creating a leaf
        cluster or further splitting recursively.

        Args:
            X: Data points in the cluster.
            F: Function values at data points.
            labels: Cluster labels for data points.
            label: Label of the subcluster to process.
            centroid: Centroid of the subcluster.
            f: Function to evaluate.
        """
        mask = labels == label
        sub_X, sub_F = X[mask], F[mask]

        if not np.any(mask) or len(sub_X) <= 1:
            # Not enough points to form a subcluster
            return None

        # Add extra points if needed
        if len(sub_X) <= self.min_points:
            extra_X, extra_F = self._generate_extra_points(sub_X, f)
            if len(extra_X) > 0:
                sub_X = np.vstack([sub_X, extra_X])
                sub_F = np.vstack([sub_F, extra_F])

        # Check for dead branch (zero radius)
        radius = (
            np.max(np.linalg.norm(sub_X - centroid, axis=1)) if len(sub_X) > 0 else 0.0
        )
        if np.isclose(radius, 0.0):
            return None

        # Check tolerance
        if self._meets_tolerance(sub_X, sub_F, centroid, f):
            return TreeNode(cluster=self._create_cluster(sub_X, sub_F, centroid, f))
        else:
            return self._split_recursive(sub_X, sub_F, f)

    def _generate_extra_points(
        self, sub_X: np.ndarray, f: Callable
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate extra points via convex combinations."""
        Q = len(sub_X)
        if Q < 2:
            raise ValueError(
                "At least two points are required to generate extra points."
            )

        n_extra = self.min_points - len(sub_X) + 1

        rng = np.random.default_rng(self.seed)
        lambdas = rng.random(n_extra)  # in [0,1)

        # Pick any base index for each column
        idx0 = rng.integers(0, Q, size=n_extra)

        # Pick a non-zero offset for each column, then wrap with modulo
        # This will ensure idx1 != idx0 for every column
        offs = rng.integers(1, Q, size=n_extra)
        idx1 = (idx0 + offs) % Q

        # vectorized convex combination to generate new points
        new_X = lambdas[:, None] * sub_X[idx0] + (1.0 - lambdas)[:, None] * sub_X[idx1]

        return new_X, self._get_function_values(new_X, f)


def adaptive_cluster_algorithm(
    X_train,
    f,
    atol=0.01,
    rtol=0.01,
    N_init_clusters=10,
    min_ppc=1,
    use_gradient=False,
    seed=0,
) -> tuple[int, list[TreeNode]]:
    """API compatible adaptive clustering function."""
    clusterer = AdaptiveClusterer(
        atol=atol,
        rtol=rtol,
        n_clusters=N_init_clusters,
        min_points=min_ppc,
        use_gradient=use_gradient,
        seed=seed,
    )
    trees = clusterer.fit(X_train, f)
    total_clusters = sum(len(tree.leaves()) for tree in trees)
    return total_clusters, trees


def flatten_cluster_centers_with_gradient(nc: int, n: int, m: int, tree):
    """
    Flatten leaf cluster data into contiguous NumPy arrays.

    Shape Convention:
        Following the convention x.shape=(n_pts, d), f.shape=(n_pts, fd), grad.shape=(n_pts, fd, d):
        - centroids: (nc, n) where nc=number of clusters, n=input dimensions
        - QoI_values: (nc, m) where m=output dimensions
        - gradients: (nc, m, n) where gradients are Jacobian matrices
        - hessians: (nc,) scalar Hessian proxies per cluster
        - dims: (nc,) cluster radii (name kept for API compatibility)

    Args:
        nc: Expected number of leaf clusters (rows in outputs).
        n: Input dimension (centroid length).
        m: Output dimension (value length).
        tree: A TreeNode or an iterable of TreeNode roots.

    Returns:
        centroids: (nc, n) float64
        QoI_values: (nc, m) float64
        gradients: (nc, m, n) float64 (zeros if gradient is None)
        hessians: (nc,) float64
        dims: (nc,) float64 (cluster radius; kept name for API compatibility)

    Notes:
        - Traversal is iterative (stack-based) to avoid recursion overhead and
          extra allocations (e.g., not building a leaves() list).
        - Gradients are coerced to shape (m, n); if None, filled with zeros.
        - Raises if encountered shapes are incompatible with (m, n) or if the
          actual leaf count differs from nc (to avoid silent shape mismatches).
    """
    centroids = np.empty((nc, n), dtype=np.float64)
    QoI_values = np.empty((nc, m), dtype=np.float64)
    gradients = np.empty((nc, m, n), dtype=np.float64)
    hessians = np.empty((nc,), dtype=np.float64)
    dims = np.empty((nc,), dtype=np.float64)

    # Initialize stack of nodes (supports a single TreeNode or an iterable of roots).
    stack = []
    try:
        # If 'tree' is iterable of roots
        for node in tree:
            if node is not None:
                stack.append(node)
    except TypeError:
        # Single root case
        if tree is not None:
            stack.append(tree)

    k = 0  # row index into output arrays

    while stack:
        node = stack.pop()
        cl = node.cluster

        if cl is not None:
            # Centroid
            c = np.asarray(cl.centroid, dtype=np.float64).flatten()
            if c.size != n:
                raise ValueError(f"Centroid size {c.size} != n={n}")
            centroids[k] = c

            # QoI/value
            v = np.asarray(cl.value, dtype=np.float64).flatten()
            if v.size != m:
                raise ValueError(f"Value size {v.size} != m={m}")
            QoI_values[k] = v

            # Gradient (coerce to (m, n); fill zeros if None)
            g = cl.gradient
            if g is None:
                gradients[k].fill(0.0)
            else:
                g_arr = np.asarray(g, dtype=np.float64)
                if g_arr.shape == (m, n):
                    gradients[k] = g_arr
                elif g_arr.ndim == 1 and m == 1 and g_arr.size == n:
                    gradients[k, 0, :] = g_arr
                elif g_arr.ndim == 2 and g_arr.shape == (n, m):
                    gradients[k] = g_arr.T
                elif g_arr.size == m * n:
                    gradients[k] = g_arr.reshape(m, n)
                else:
                    raise ValueError(
                        f"Gradient shape {g_arr.shape} not compatible with (m, n)=({m}, {n})"
                    )

            # Scalars
            hessians[k] = float(cl.hessian)
            dims[k] = float(cl.radius)

            k += 1
            if k > nc:
                raise ValueError(f"Encountered more leaves ({k}) than nc={nc}")
        else:
            # Internal node: push children (right first so left is processed first)
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

    if k != nc:
        # Be strict to avoid returning partially filled arrays with undefined rows.
        raise ValueError(f"Expected nc={nc} leaves, found {k}")

    return centroids, QoI_values, gradients, hessians, dims


def plot_clusters(trees: list[TreeNode], fname: str = "adaptive_clusters.png"):
    """Plot 2D clusters from the tree nodes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    plt.figure(figsize=(8, 8))
    for tree in trees:
        leaves = tree.leaves()
        for cluster in leaves:
            x, y = float(cluster.centroid[0]), float(cluster.centroid[1])
            circle = Circle(
                (x, y),
                cluster.radius,
                color=np.random.rand(3),
                alpha=0.3,
                label=f"Cluster {cluster.index}",
            )
            plt.gca().add_artist(circle)
            plt.plot(x, y, "o", color="black", markersize=5)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Adaptive Clustering Visualization")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(fname)


if __name__ == "__main__":
    import cProfile
    import time
    from pathlib import Path

    FILE_NAME = Path(__file__).resolve().stem

    def sample_function(Xb) -> tuple[np.ndarray, np.ndarray]:
        """
        f(x) = [x1^2 + x2^2, x1 - x2 + x3]
        Jacobian/Gradient:
        [[2*x1, 2*x2, 0],
         [1,    -1,  1]]
        """
        x1, x2, x3 = Xb[:, 0], Xb[:, 1], Xb[:, 2]
        val0 = x1**2 + x2**2
        val1 = x1 - x2 + x3
        values = np.stack([val0, val1], axis=1)
        g0 = np.stack([2 * x1, 2 * x2, np.zeros_like(x1)], axis=1)
        g1 = np.stack([np.ones_like(x1), -np.ones_like(x1), np.ones_like(x1)], axis=1)
        grads = np.stack([g0, g1], axis=1)  # (B, 2, 3)
        return values, grads

    # Generate sample data
    seed = 0
    np.random.seed(seed)
    N_samples = 80_000
    X_samples = np.random.uniform(-10, 10, size=(N_samples, 3))
    print(f"Generated {N_samples} sample data points.")

    cprofile = cProfile.Profile()
    cprofile.enable()

    # Run adaptive clustering
    tic = time.perf_counter()
    total_clusters, clusters = adaptive_cluster_algorithm(
        X_samples,
        sample_function,
        atol=1.0,
        rtol=0.01,
        N_init_clusters=5,
        min_ppc=5,
        use_gradient=True,
        seed=seed,
    )
    toc = time.perf_counter()
    print(f"Total clusters formed: {total_clusters}")
    total = (
        NUM_CLUSTERED_BY_CUML
        + NUM_CLUSTERED_BY_SKLEARN
        + NUM_CLUSTERED_BY_MINIBATCH
        + NUM_CLUSTERED_BY_TWO_MEANS
    )
    print(f"Clusters by CUML: {NUM_CLUSTERED_BY_CUML}")
    print(f"Clusters by SKLearn KMeans: {NUM_CLUSTERED_BY_SKLEARN}")
    print(f"Clusters by SKLearn MiniBatchKMeans: {NUM_CLUSTERED_BY_MINIBATCH}")
    print(f"Clusters by TwoMeans: {NUM_CLUSTERED_BY_TWO_MEANS}")
    print(f"Total clustering calls: {total}")
    print(f"Time taken: {toc - tic:.4f} seconds")
    if N_samples <= 5_000:
        # Plot only for small datasets
        plot_clusters(clusters, fname=f"{FILE_NAME}_clusters.png")

    cprofile.disable()
    cprofile.dump_stats(f"{FILE_NAME}_profile.prof")
    # cprofile.print_stats(sort="time")
