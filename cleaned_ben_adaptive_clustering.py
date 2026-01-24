import logging
from typing import Callable, Optional

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

TINY_EPS = 1e-10


class GlobalData:
    """
    Manages the global dataset to avoid copying large arrays during BFS.
    """

    def __init__(
        self,
        X: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        f_fn: Callable,
        g_fn: Callable,
    ):
        self.X = X
        self.f = f
        self.g = g
        self.f_fn = f_fn
        self.g_fn = g_fn
        self.N_original = X.shape[0]

    def add_synthetic_points(self, X_synth: np.ndarray) -> np.ndarray:
        """Adds new points to the global store and returns their indices."""
        n_start = self.X.shape[0]
        n_new = X_synth.shape[0]

        # Evaluate oracle
        f_synth = self.flatten_f(self.f_fn(X_synth))
        g_synth = self.flatten_g(self.g_fn(X_synth))

        # Append to storage
        self.X = np.vstack([self.X, X_synth])
        self.f = np.concatenate([self.f, f_synth])
        self.g = np.vstack([self.g, g_synth])

        return np.arange(n_start, n_start + n_new)

    @staticmethod
    def flatten_f(F):
        F = np.asarray(F)
        if F.ndim == 2 and F.shape[1] == 1:
            return F[:, 0]
        return F.reshape((-1,))

    @staticmethod
    def flatten_g(G):
        G = np.asarray(G)
        if G.ndim == 3 and G.shape[1] == 1:
            return G[:, 0, :]
        return G


def fit_ellipse_pca_scaled(
    points: np.ndarray, axis_ratio_limit: float = 100.0
) -> np.ndarray:
    """
    Fits an ellipse with regularization to prevent infinite stretching.
    axis_ratio_limit: Max allowed ratio between Major and Minor axis.
    """
    N, D = points.shape
    if N < D:
        return np.eye(D)

    U, _, _ = np.linalg.svd(points.T)

    # Project points onto eigenvectors to get actual extents
    projections = points @ U
    radii = np.max(np.abs(projections), axis=0)

    # Avoid zero radius
    radii = np.maximum(radii, TINY_EPS)

    # --- REGULARIZATION ---
    # Prevent the ellipse from being too thin.
    # If max_radius / min_radius > limit, clamp min_radius.
    max_r = np.max(radii)
    min_allowed_r = max_r / axis_ratio_limit
    radii = np.maximum(radii, min_allowed_r)

    return (U * (radii**-2)) @ U.T


def fit_ellipse_pca_metric(points: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    N, D = points.shape
    if N < D:
        return np.eye(D)

    U, Sigma, _ = np.linalg.svd(points.T)

    # --- FIX: Regularization ---
    # Prevent singular values from being too small compared to the largest one
    # or just add a flat constant.
    max_sigma = np.max(Sigma)
    Sigma_clean = np.maximum(Sigma, max_sigma * reg)

    # Now invert
    return (U * (Sigma_clean**-2)) @ U.T


def generate_padding(
    indices: np.ndarray, data: GlobalData, min_pts: int, seed: int
) -> Optional[np.ndarray]:
    """Generates synthetic points via convex combinations if cluster is too small."""
    count = len(indices)
    if count >= min_pts:
        return None

    needed = min_pts - count
    # Cannot interpolate with < 2 points, but we handle count<=1 in main loop logic
    if count < 2:
        return None

    rng = np.random.default_rng(seed)
    X_k = data.X[indices]

    # Random pairs
    i_idx = rng.integers(0, count, size=needed)
    j_raw = rng.integers(0, count - 1, size=needed)
    # Ensure j != i
    j_idx = j_raw + (j_raw >= i_idx).astype(int)

    alpha = rng.random((needed, 1))
    X_synth = alpha * X_k[i_idx] + (1.0 - alpha) * X_k[j_idx]

    return X_synth


def adaptive_clustering(
    X: np.ndarray,
    f_fn: Callable,
    g_fn: Callable,
    TOL: float = 0.1,
    NUM_INITIAL_CLUSTERS: int = 10,
    MIN_PTS_PER_CLUSTER: int = 50,
    MAX_LEVELS: int = 50,
    SEED: int = 0,
) -> dict:
    """
    BFS implementation of Metric-Adaptive Clustering.
    """

    # 1. Initialize Global Data
    f0 = GlobalData.flatten_f(f_fn(X))
    g0 = GlobalData.flatten_g(g_fn(X))
    data = GlobalData(X, f0, g0, f_fn, g_fn)

    total_f_norm = np.linalg.norm(data.f)  # Used for relative error check

    # 2. Initial KMeans
    logger.info("Running Initial KMeans...")
    kmeans = KMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
    labels = kmeans.fit_predict(X)

    # Queue holds tuples: (indices_array, depth_level, current_total_cluster_count_estimate)
    # The estimate is needed because the error threshold scales with 1/sqrt(N_clusters).
    # In BFS, we estimate total clusters roughly as n_init * 2^depth.
    queue = []

    for k in range(NUM_INITIAL_CLUSTERS):
        idxs = np.where(labels == k)[0]
        if len(idxs) > 0:
            queue.append((idxs, 0))

    finalized_results = {
        "centroids": [],
        "f": [],
        "g": [],
        "Q": [],  # Trust Region Ellipse
        "counts": [],
        "depth": [],
        "radii": [],
    }

    # BFS Loop
    while len(queue) > 0:
        print(f"Queue Size: {len(queue)}")

        # Pop next cluster
        indices, depth = queue.pop(0)

        # ---------------------------
        # A. Dead Branch Check
        # ---------------------------
        if len(indices) <= 1:
            continue  # Dead branch, not enough variance

        # ---------------------------
        # B. Padding (Enrichment)
        # ---------------------------
        # If we pad, we update global data and local indices
        X_synth = generate_padding(
            indices, data, MIN_PTS_PER_CLUSTER, seed=SEED + depth
        )
        if X_synth is not None:
            new_idxs = data.add_synthetic_points(X_synth)
            indices = np.concatenate([indices, new_idxs])

        # ---------------------------
        # C. Compute Centroid & Error
        # ---------------------------
        X_k = data.X[indices]
        f_k = data.f[indices]

        # Centroid
        mu = np.mean(X_k, axis=0)

        # Fresh eval at centroid
        f_mu = data.flatten_f(data.f_fn(mu[None, :]))[0]
        g_mu = data.flatten_g(data.g_fn(mu[None, :]))[0]

        # Linear Model Deviation
        dx = X_k - mu
        # f_approx = f(c) + g(c)^T * (x - c)
        f_lin = f_mu + (dx @ g_mu)

        # Error vector (L2 norm per point if f is scalar, but here f is scalar per point)
        # e_i = |f_true - f_approx|
        err_vec = np.abs(f_k - f_lin)

        # Aggregate Errors (Logic from original helper)
        abs_error_norm = np.linalg.norm(err_vec, 2)
        rel_error_norm = abs_error_norm / total_f_norm

        # Estimate total clusters at this depth for ATOL scaling
        estimated_total_clusters = NUM_INITIAL_CLUSTERS * (2**depth)

        # Convergence Condition
        is_converged = rel_error_norm <= TOL or (depth >= MAX_LEVELS)

        # Identify "Good Points" (used for Trust Region OR Splitting Metric)
        # Condition: e_i <= tol OR e_i <= rtol * |f(c)|
        # Note: We use the raw tol here, not the scaled one, for point filtering
        good_mask_abs = err_vec <= TOL
        good_mask_rel = err_vec <= (TOL * np.abs(f_mu))
        good_mask = np.logical_or(good_mask_abs, good_mask_rel)

        dx_good = dx[good_mask]

        # ---------------------------
        # D. Decision: Finalize or Split
        # ---------------------------
        if is_converged:
            # --- LEAF NODE ---
            # Fit Trust Region Ellipse (Using Scaled PCA on good points)
            if len(dx_good) > dx.shape[1]:
                Q_trust = fit_ellipse_pca_scaled(dx_good)
            else:
                Q_trust = np.eye(dx.shape[1])

            # Radii is the distance from the centroid to the farthest point which had good error
            if len(dx_good) > 0:
                dists_good = np.linalg.norm(dx_good, axis=1)
                max_radius = np.max(dists_good)
            else:
                max_radius = 0.0

            finalized_results["centroids"].append(mu)
            finalized_results["f"].append(f_mu)
            finalized_results["g"].append(g_mu)
            finalized_results["Q"].append(Q_trust)
            finalized_results["radii"].append(max_radius)
            finalized_results["counts"].append(len(indices))
            finalized_results["depth"].append(depth)

        else:
            # --- SPLIT NODE ---
            # 1. Compute Metric Q from Good Points
            dim = dx.shape[1]
            if len(dx_good) > dim:
                Q_metric = fit_ellipse_pca_metric(dx_good)
            else:
                Q_metric = np.eye(dim)

            # 2. Metric Adaptive Transformation
            # Decompose Q = U * D * U.T
            # Transform B = U * sqrt(D) * U.T
            # We want to cluster in space x' = x @ B

            # Eigen decomposition
            D_eig, U_eig = np.linalg.eigh(Q_metric)
            D_eig = np.maximum(D_eig, TINY_EPS)

            # B = U * D^0.5 * U.T
            B = U_eig * (D_eig**0.5) @ U_eig.T

            # Rotate points: dx is centered data
            dx_rotated = dx @ B

            # 3. KMeans Split (k=2)
            # Use fixed seed for reproducibility within the BFS
            km_split = KMeans(n_clusters=2, random_state=SEED + depth)
            split_labels = km_split.fit_predict(dx_rotated)

            idx0 = indices[split_labels == 0]
            idx1 = indices[split_labels == 1]

            # 4. Add children to queue
            # Note: If KMeans creates an empty cluster (rare but possible), skip it
            if len(idx0) > 0:
                queue.append((idx0, depth + 1))
            if len(idx1) > 0:
                queue.append((idx1, depth + 1))

    # Convert lists to arrays for final output
    return {k: np.array(v) for k, v in finalized_results.items()}


if __name__ == "__main__":
    import argparse
    from functools import partial

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    from src.logging_config import setup_logging
    from src.problems.quadratic import quadratic, quadratic_grad
    from src.problems.shifted_gaussians import (
        CENTERS_2D,
        shifted_gaussians,
        shifted_gaussians_grad,
    )

    setup_logging(console_level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tol", type=float, default=0.1, help="Tolerance for adaptive clustering"
    )
    args = parser.parse_args()
    TOL = args.tol

    shif_g_fn = partial(shifted_gaussians, centers=CENTERS_2D)
    shif_g_grad_fn = partial(shifted_gaussians_grad, centers=CENTERS_2D)

    # Generate Data
    N = 50_000
    D = 2
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(N, D))
    results = adaptive_clustering(
        X,
        f_fn=shif_g_fn,
        g_fn=shif_g_grad_fn,
        TOL=TOL,
        NUM_INITIAL_CLUSTERS=1,
        MIN_PTS_PER_CLUSTER=50,
        MAX_LEVELS=20,
        SEED=0,
    )

    # Plot centroids with ellipses on contourf
    x1 = np.linspace(-2, 2, 200)
    x2 = np.linspace(-2, 2, 200)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T
    F_grid = shifted_gaussians(X_grid, centers=CENTERS_2D).reshape(X1.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(X1, X2, F_grid, levels=50, cmap="viridis")
    plt.colorbar(label="Function Value")

    for i in range(results["centroids"].shape[0]):
        mu = results["centroids"][i]
        Q = results["Q"][i]

        # Compute eigenvalues and eigenvectors of Q
        w, V = np.linalg.eigh(Q)
        order = np.argsort(w)[::-1]
        w = w[order]
        V = V[:, order]
        a = 1.0 / np.sqrt(w[0])
        b = 1.0 / np.sqrt(w[1])
        angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))

        ellipse = Ellipse(
            xy=mu,
            width=2 * a,
            height=2 * b,
            angle=angle,
            edgecolor="red",
            facecolor="none",
            lw=2,
            alpha=0.7,
        )
        plt.gca().add_patch(ellipse)
    plt.scatter(results["centroids"][:, 0], results["centroids"][:, 1], c="red", s=50)
    plt.title(f"Adaptive Clustering with Ellipsoids on Shifted Gaussians (TOL={TOL})")
    fname = "new_ac.png"
    plt.savefig(fname)
    plt.close()

    # Plot a similar figure - contourf, centroids, but circles with radius instead of ellipses
    plt.figure(figsize=(10, 8))
    plt.contourf(X1, X2, F_grid, levels=50, cmap="viridis")
    plt.colorbar(label="Function Value")

    for i in range(results["centroids"].shape[0]):
        mu = results["centroids"][i]
        radius = results["radii"][i]

        circle = Circle(
            xy=mu,
            radius=radius,
            edgecolor="blue",
            facecolor="none",
            lw=2,
            alpha=0.7,
        )
        plt.gca().add_patch(circle)
    plt.scatter(results["centroids"][:, 0], results["centroids"][:, 1], c="blue", s=50)
    plt.title(f"Adaptive Clustering with Circles on Shifted Gaussians (TOL={TOL})")
    fname = "new_ac_circles.png"
    plt.savefig(fname)
    plt.close()

    # Do a similar run on Quadratic Function
    N = 20_000
    D = 2
    rng = np.random.default_rng(1)
    X = rng.uniform(-5, 5, size=(N, D))
    results = adaptive_clustering(
        X,
        f_fn=quadratic,
        g_fn=quadratic_grad,
        TOL=TOL,
        NUM_INITIAL_CLUSTERS=1,
        MIN_PTS_PER_CLUSTER=50,
        MAX_LEVELS=20,
        SEED=0,
    )

    # Plot centroids with ellipses on contourf
    x1 = np.linspace(-5, 5, 200)
    x2 = np.linspace(-5, 5, 200)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T
    F_grid = quadratic(X_grid).reshape(X1.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(X1, X2, F_grid, levels=50, cmap="viridis")
    plt.colorbar(label="Function Value")
    for i in range(results["centroids"].shape[0]):
        mu = results["centroids"][i]
        Q = results["Q"][i]

        # Compute eigenvalues and eigenvectors of Q
        w, V = np.linalg.eigh(Q)
        order = np.argsort(w)[::-1]
        w = w[order]
        V = V[:, order]
        a = 1.0 / np.sqrt(w[0])
        b = 1.0 / np.sqrt(w[1])
        angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))

        ellipse = Ellipse(
            xy=mu,
            width=2 * a,
            height=2 * b,
            angle=angle,
            edgecolor="red",
            facecolor="none",
            lw=2,
            alpha=0.7,
        )
        plt.gca().add_patch(ellipse)
    plt.scatter(results["centroids"][:, 0], results["centroids"][:, 1], c="red", s=50)
    plt.title(f"Adaptive Clustering with Ellipsoids on Quadratic Function (TOL={TOL})")
    fname = "new_ac_quadratic.png"
    plt.savefig(fname)
    plt.close()

    # Plot a similar figure - contourf, centroids, but circles with radius instead of ellipses
    plt.figure(figsize=(10, 8))
    plt.contourf(X1, X2, F_grid, levels=50, cmap="viridis")
    plt.colorbar(label="Function Value")

    for i in range(results["centroids"].shape[0]):
        mu = results["centroids"][i]
        radius = results["radii"][i]

        circle = Circle(
            xy=mu,
            radius=radius,
            edgecolor="blue",
            facecolor="none",
            lw=2,
            alpha=0.7,
        )
        plt.gca().add_patch(circle)
    plt.scatter(results["centroids"][:, 0], results["centroids"][:, 1], c="blue", s=50)
    plt.title(f"Adaptive Clustering with Circles on Quadratic Function (TOL={TOL})")
    fname = "new_ac_quadratic_circles.png"
    plt.savefig(fname)
    plt.close()
