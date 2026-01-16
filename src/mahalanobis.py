import logging
import time
from typing import Tuple

import cupy as cp

logger = logging.getLogger(__name__)

# TODO: To speed things up, we could find the out the nearest neighours first in L2
# space through cupy kNN, and then compute Mahalanobis distances only for those. This
# would be particularly useful when K (# of clusters), d (dimensionality), M (# of
# query points) are all large.


def knn_mahalanobis_clusters(
    query_points: cp.ndarray,
    centroids: cp.ndarray,
    Q: cp.ndarray,
    k: int,
    *,
    return_sqrt: bool = False,
    eps: float = 0.0,
    trust_region_containment: bool = False,
    containment_level: float = 1.0,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    GPU-only kNN over cluster centroids using the cluster-specific Mahalanobis distance.
    For query point x (row of `query_points`) and cluster k with centroid μ_k and
    trust-region matrix Q_k, we define the squared Mahalanobis distance:

        dist2_k(x) = (x - μ_k)^T Q_k (x - μ_k).

    Optionally enforce trust-region containment:
        eligible if dist2_k(x) <= containment_level,
    where containment_level=1.0 corresponds to the ellipse boundary
        (x - μ_k)^T Q_k (x - μ_k) <= 1.

    In containment mode, clusters that do not contain a query are returned as missing
    results (idx=-1, dist=+inf) if fewer than k eligible clusters exist.

    Note: This version computes the full (M, K) distance matrix on the GPU. It is fast
    for moderate M*K, but can be memory-heavy for very large workloads.

    Args:
        query_points: (M, d) Query points.
        centroids: (K, d) Cluster centroids.
        Q: (K, d, d) Per-cluster trust-region / metric matrices (symmetric PSD).
        k: Number of nearest clusters to return per query.
        return_sqrt: If True, return sqrt(dist2) instead of dist2.
        eps: Nonnegative guard used only when return_sqrt=True:
            sqrt(max(dist2 + eps, 0)).
        trust_region_containment: If True, only clusters satisfying
            dist2 <= containment_level are considered eligible.
        containment_level: Trust-region level defining containment (1.0 = boundary).

    Returns:
        nn_idx: (M, k) int32 indices of nearest clusters (sorted by increasing distance).
            If trust_region_containment=True and a query has fewer than k eligible
            clusters, remaining entries are -1.
        nn_dist: (M, k) distances corresponding to nn_idx (squared unless return_sqrt).
            Missing entries are +inf.
    """
    # Ensure GPU arrays.
    query_points = cp.asarray(query_points)
    centroids = cp.asarray(centroids)
    Q = cp.asarray(Q)

    M, d = query_points.shape
    K = int(centroids.shape[0])

    # Validate inputs.
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > K:
        raise ValueError(f"k={k} cannot exceed number of clusters K={K}.")
    if centroids.shape[1] != d:
        raise ValueError("centroids must have shape (K, d) matching query_points.")
    if Q.shape != (K, d, d):
        raise ValueError("Q must have shape (K, d, d).")
    if containment_level <= 0:
        raise ValueError("containment_level must be positive.")
    if eps < 0:
        raise ValueError("eps must be nonnegative.")

    logger.debug(
        "kNN Mahalanobis Clusters Search: k=%d, return_sqrt=%s, eps=%g, "
        "trust_region_containment=%s, containment_level=%g",
        k,
        return_sqrt,
        eps,
        trust_region_containment,
        containment_level,
    )
    _start_time = time.perf_counter()

    # Compute full squared Mahalanobis distance matrix dist2 (M, K):
    # dist2[m, k] = dx[m,k,:]^T Q[k,:,:] dx[m,k,:]
    dx = query_points[:, None, :] - centroids[None, :, :]  # (M, K, d)
    dist2 = cp.einsum("mkd,kde,mke->mk", dx, Q, dx, optimize=True)  # (M, K)

    if trust_region_containment:
        # Mask out clusters whose trust region does not contain the query.
        dist2 = cp.where(dist2 <= containment_level, dist2, cp.inf)

    # Select k smallest per query using argpartition (for fast selection),
    # then sort those k by distance for nicer output ordering.
    part = cp.argpartition(dist2, kth=k - 1, axis=1)[:, :k]  # (M, k)
    part_dist2 = cp.take_along_axis(dist2, part, axis=1)  # (M, k)

    order = cp.argsort(part_dist2, axis=1)  # (M, k)
    nn_idx = cp.take_along_axis(part, order, axis=1).astype(cp.int32)  # (M, k)
    nn_dist2 = cp.take_along_axis(part_dist2, order, axis=1)  # (M, k)

    if trust_region_containment:
        # Convert masked-out (+inf) entries into "missing" indices (-1).
        nn_idx = cp.where(cp.isfinite(nn_dist2), nn_idx, -1).astype(cp.int32)

    logger.debug(
        "Runtime - knn_mahalanobis_clusters: %.3fs", time.perf_counter() - _start_time
    )

    if return_sqrt:
        nn_dist2 = cp.sqrt(cp.maximum(nn_dist2 + eps, 0))

    return nn_idx, nn_dist2


if __name__ == "__main__":
    from src.gpu_adaptive_clustering_metric import adaptive_clustering
    from src.logging_config import setup_logging
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging(console_level=logging.DEBUG)

    cp.random.seed(0)
    _N = 100_000
    d = 2
    print(f"Generating {_N:,} points (d={d})...")
    PROB_LIMS = (-1.0, 3)
    X = cp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(_N, d))

    TOL = 1e-3
    N_INIT_CLUSTERS = 100
    N_MIN_PER_CLUSTER = 1000

    print("Starting Adaptive GPU Clustering...")
    results = adaptive_clustering(
        X,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        ATOL=TOL,
        RTOL=TOL,
        NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
        MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
        MAX_CLUSTERS_EST=100_000,
    )

    print("-" * 40)
    print(f"Final Cluster Count: {results['centroids'].shape[0]:,}")

    print("Testing knn_mahalanobis_clusters...")
    M = 30
    k = 5
    query_points = cp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(M, d))
    centroids = results["centroids"]
    Q = results["Q"]
    print(f"# of Clusters: {centroids.shape[0]:,}, Query Points: {M:,}, k={k}")

    for trust_region_containment in (False, True):
        _start_time = time.time()
        nn_idx, nn_dist2 = knn_mahalanobis_clusters(
            query_points,
            centroids,
            Q,
            k,
            return_sqrt=False,
            trust_region_containment=trust_region_containment,
        )
        _end_time = time.time()
        mode = "CONTAINED" if trust_region_containment else "UNCONSTRAINED"
        print(f"[{mode}] Runtime: {_end_time - _start_time:.3f}s")

    trust_region_containment = False
    nn_idx_vis, nn_dist2_vis = knn_mahalanobis_clusters(
        query_points,
        centroids,
        Q,
        k,
        return_sqrt=False,
        trust_region_containment=trust_region_containment,
        containment_level=1.0,
    )

    # Optional visualization (small sizes only)
    if d == 2 and len(results["centroids"]) < 3000 and M < 100:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Ellipse

        x = cp.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        y = cp.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        X_grid, Y_grid = cp.meshgrid(x, y)
        Z_grid = rosenbrock(cp.vstack([X_grid.ravel(), Y_grid.ravel()]).T).reshape(
            X_grid.shape
        )

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Subplot 1: centroids + ellipses
        cs = axs[0].contourf(
            X_grid.get(),
            Y_grid.get(),
            Z_grid.get(),
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(cs, ax=axs[0])
        axs[0].scatter(
            centroids[:, 0].get(),
            centroids[:, 1].get(),
            c="red",
            s=10,
            label="Centroids",
        )

        # Draw trust ellipses (level=1)
        for i in range(centroids.shape[0]):
            mu = centroids[i].get()
            Qi = Q[i].get()
            w, V = np.linalg.eigh(Qi)
            w = np.maximum(w, 1e-12)
            order = np.argsort(w)[::-1]
            w = w[order]
            V = V[:, order]

            a = np.sqrt(1.0 / w[0])
            b = np.sqrt(1.0 / w[1])
            angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
            ellipse = Ellipse(
                xy=(mu[0], mu[1]),
                width=2 * a,
                height=2 * b,
                angle=angle,
                edgecolor="red",
                facecolor="none",
                lw=1,
                alpha=0.5,
            )
            axs[0].add_patch(ellipse)

        axs[0].legend()
        axs[0].set_title("Cluster Centroids with Trust Ellipses")
        axs[0].set_xlabel("x1")
        axs[0].set_ylabel("x2")
        axs[0].set_aspect("equal")

        # Subplot 2: queries + (contained) ellipses from neighbors
        cs2 = axs[1].contourf(
            X_grid.get(),
            Y_grid.get(),
            Z_grid.get(),
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(cs2, ax=axs[1])

        dist0 = nn_dist2_vis[:, 0]
        finite = cp.isfinite(dist0)
        max_finite = cp.max(dist0[finite]).get() if bool(cp.any(finite)) else 1.0

        for m in range(M):
            d0 = float(dist0[m].get())
            color = plt.cm.viridis(d0 / max_finite) if np.isfinite(d0) else (0, 0, 0, 1)  # type: ignore

            axs[1].scatter(
                query_points[m, 0].get(),
                query_points[m, 1].get(),
                c=[color],
                s=30,
                edgecolor="black",
                zorder=5,
            )

            for kk in range(k):
                cid = int(nn_idx_vis[m, kk].get())
                if cid < 0:
                    continue

                mu = centroids[cid].get()
                Qi = Q[cid].get()

                # Determine if actually contained (for styling)
                dx = query_points[m].get() - mu
                d2 = float(dx @ Qi @ dx)
                contained = d2 <= 1.0 + 1e-10

                w, V = np.linalg.eigh(Qi)
                w = np.maximum(w, 1e-12)
                order = np.argsort(w)[::-1]
                w = w[order]
                V = V[:, order]

                a = np.sqrt(1.0 / w[0])
                b = np.sqrt(1.0 / w[1])
                angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))

                ellipse = Ellipse(
                    xy=(mu[0], mu[1]),
                    width=2 * a,
                    height=2 * b,
                    angle=angle,
                    edgecolor=color,
                    facecolor="none",
                    lw=2.0 if contained else 1.0,
                    alpha=0.9 if contained else 0.35,
                    linestyle="-" if contained else "--",
                )
                axs[1].add_patch(ellipse)

        axs[1].set_title(
            f"Query Points with Nearest Containing Clusters (k={k}, containment={trust_region_containment})"
        )

        axs[1].set_xlabel("x1")
        axs[1].set_ylabel("x2")
        axs[1].set_aspect("equal")

        plt.tight_layout()
        fname = f"knn_mahalanobis_clusters_demo_{'cont' if trust_region_containment else 'unconstr'}.png"
        plt.savefig(fname, dpi=300)
        print(f"Saved plot to {fname}")
    else:
        print("Skipping plot due to size constraints.")
