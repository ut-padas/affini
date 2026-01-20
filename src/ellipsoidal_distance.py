import logging
import time
from typing import Tuple

from src import IS_CUPY_BACKEND, Array, xp
from src.logging_config import log_mem_stats

logger = logging.getLogger(__name__)

# TODO: To speed things up, we could find the out the nearest neighours first in L2
# space through cupy kNN, and then compute Ellipsoidal distances only for those. This
# would be particularly useful when K (# of clusters), d (dimensionality), M (# of
# query points) are all large.


def knn_ellipsoidal_distance(
    query_points: Array,
    centroids: Array,
    Q: Array,
    k: int,
    *,
    return_sqrt: bool = False,
    eps: float = 0.0,
    enforce_containment: bool = False,
    containment_level: float = 1.0,
    batch_size: int = 1024,
    cluster_batch_size: int = 10_000,
    LOG_MEM_STATS: bool = True,
) -> Tuple[Array, Array]:
    """
    GPU-only kNN over cluster centroids using the cluster-specific Ellipsoidal trust
    distance metric defined by per-cluster matrices Q.
    For query point x (row of `query_points`) and cluster k with centroid μ_k and
    trust-region matrix Q_k, we define the squared Ellipsoidal distance as

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
        enforce_containment: If True, only clusters satisfying
            dist2 <= containment_level are considered eligible.
        containment_level: Trust-region level defining containment (1.0 = boundary). This
            is ignored if enforce_containment=False.
        batch_size: Number of query points to process at once. Lower this if OOM persists.
        cluster_batch_size: Number of clusters to process at once. Lower this if OOM persists.
        LOG_MEM_STATS: If True, log GPU memory stats.

    Returns:
        nn_idx: (M, k) int32 indices of nearest clusters (sorted by increasing distance).
            If enforce_containment=True and a query has fewer than k eligible
            clusters, remaining entries are -1.
        nn_dist: (M, k) distances corresponding to nn_idx (squared unless return_sqrt).
            Missing entries are +inf.
    """
    if LOG_MEM_STATS:
        log_mem_stats("knn_ellipsoidal_distance: start")

    # Ensure GPU arrays.
    query_points = xp.asarray(query_points)
    centroids = xp.asarray(centroids)
    Q = xp.asarray(Q)

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

    logger.debug(
        "kNN Ellipsoidal Clusters Search: k=%d, return_sqrt=%s, eps=%g, "
        "enforce_containment=%s, containment_level=%g, batch_size=%d, "
        "cluster_batch_size=%d",
        k,
        return_sqrt,
        eps,
        enforce_containment,
        containment_level,
        batch_size,
        cluster_batch_size,
    )
    _start_time = time.perf_counter()

    all_indices = []
    all_dists = []

    # Outer Loop: Iterate over Query Batches (B rows)
    for start_idx in range(0, M, batch_size):
        end_idx = min(start_idx + batch_size, M)
        batch_queries = query_points[start_idx:end_idx]  # (B, d)
        current_B = batch_queries.shape[0]

        # Allocate result buffer for this batch: (B, K)
        # For B=1024, K=1.4M, float64 -> ~11 GB. This fits in 100GB VRAM.
        # If K grows to >10M, we would need a running top-k reduction, but this is faster for now.
        dist2_batch = xp.empty((current_B, K), dtype=query_points.dtype)

        # Inner Loop: Iterate over Cluster Batches (K_chunk clusters)
        # This prevents allocating the massive (B, K, d) difference tensor (~115 GB)
        for c_start in range(0, K, cluster_batch_size):
            c_end = min(c_start + cluster_batch_size, K)

            # Slice geometry for this chunk
            Q_chunk = Q[c_start:c_end]  # (K_c, d, d)
            mu_chunk = centroids[c_start:c_end]  # (K_c, d)

            # 1. Compute dx for this chunk: (B, K_c, d)
            # Memory: 1024 * 10000 * 10 * 8 bytes ~= 800 MB (Safe)
            dx = batch_queries[:, None, :] - mu_chunk[None, :, :]

            # 2. Compute (Q * dx^T)^T
            # dx shape: (B, K_c, d)
            # dx_T shape: (K_c, B, d) -> transpose(0,2,1) -> (K_c, d, B) matches Q's last dim
            dx_T = dx.transpose(1, 0, 2)

            # Q_chunk (K_c, d, d) @ (K_c, d, B) -> (K_c, d, B)
            temp = xp.matmul(Q_chunk, dx_T.transpose(0, 2, 1))

            # Transpose back to align with dx: (B, K_c, d)
            temp = temp.transpose(2, 0, 1)

            # 3. Final dot product (B, K_c) and store in main buffer
            # sum over 'd' dimension
            dist2_batch[:, c_start:c_end] = xp.sum(dx * temp, axis=-1)

            # Free chunk memory immediately
            del dx, dx_T, temp, Q_chunk, mu_chunk

        # --- Post-Processing for the Batch ---
        if enforce_containment:
            dist2_batch = xp.where(
                dist2_batch <= containment_level, dist2_batch, xp.inf
            )

        # Select k smallest
        # argpartition is faster than sort for finding top-k
        part = xp.argpartition(dist2_batch, kth=k - 1, axis=1)[:, :k]  # (B, k)
        part_dist2 = xp.take_along_axis(dist2_batch, part, axis=1)  # (B, k)

        # Sort the top k for clean output
        order = xp.argsort(part_dist2, axis=1)  # (B, k)
        batch_nn_idx = xp.take_along_axis(part, order, axis=1).astype(xp.int32)
        batch_nn_dist2 = xp.take_along_axis(part_dist2, order, axis=1)  # (B, k)

        if enforce_containment:
            # Convert masked-out (+inf) entries into "missing" indices (-1).
            batch_nn_idx = xp.where(
                xp.isfinite(batch_nn_dist2), batch_nn_idx, -1
            ).astype(xp.int32)

        if return_sqrt:
            batch_nn_dist2 = xp.sqrt(xp.maximum(batch_nn_dist2 + eps, 0))

        # Append to results
        all_indices.append(batch_nn_idx)
        all_dists.append(batch_nn_dist2)

        # Explicitly free large temp arrays
        del dist2_batch, part, part_dist2

        if IS_CUPY_BACKEND:
            xp.get_default_memory_pool().free_all_blocks()

    nn_idx = xp.concatenate(all_indices, axis=0)
    nn_dist = xp.concatenate(all_dists, axis=0)

    logger.debug("Runtime - knn_elli: %.3fs", time.perf_counter() - _start_time)

    return nn_idx, nn_dist


if __name__ == "__main__":
    import argparse

    from src import FIGURES_DIR
    from src.adaptive_clustering import adaptive_clustering
    from src.logging_config import setup_logging
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging(console_level=logging.DEBUG)

    MD_DIR = FIGURES_DIR / "ellipsoidal_knn"
    MD_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Demo of knn_ellipsoidal_distance with Adaptive GPU Clustering"
    )
    parser.add_argument(
        "--enforce-containment",
        action="store_true",
        help="Enable trust-region containment mode in kNN search",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest clusters to find per query point",
    )
    parser.add_argument(
        "--containment-level",
        type=float,
        default=1.0,
        help="Trust-region containment level (default: 1.0 = boundary)",
    )
    args = parser.parse_args()
    ENFORCE_CONTAINMENT = args.enforce_containment
    K = args.k
    CONTAINMENT_LEVEL = args.containment_level

    img_fpath = (
        MD_DIR
        / f"knn_ellipsoidal_distance_k{K}_{'constrained' if ENFORCE_CONTAINMENT else 'unconstrained'}_containment{CONTAINMENT_LEVEL}.png"
    )
    if img_fpath.exists():
        print(f"Image {img_fpath} already exists, skipping demo.")
        exit(0)

    xp.random.seed(0)
    _N = 100_000
    d = 2
    print(f"Generating {_N:,} points (d={d})...")
    PROB_LIMS = (-1.0, 3)
    X = xp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(_N, d))

    TOL = 1e-3
    N_INIT_CLUSTERS = d * 2
    N_MIN_PER_CLUSTER = 1000

    print("Starting Adaptive GPU Clustering...")
    results = adaptive_clustering(
        X,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        TOL=TOL,
        NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
        MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
        SYNTH_BUDGET_FACTOR=5,
    )

    print("-" * 40)
    print(f"Final Cluster Count: {results['centroids'].shape[0]:,}")

    print("Testing kNN Ellipsoidal Distance Search...")
    M = 30
    query_points = xp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(M, d))
    centroids = results["centroids"]
    Q = results["Q"]
    print(f"# of Clusters: {centroids.shape[0]:,}, Query Points: {M:,}, k={K}")

    for _t in (False, True):
        _start_time = time.time()
        nn_idx, nn_dist2 = knn_ellipsoidal_distance(
            query_points,
            centroids,
            Q,
            K,
            return_sqrt=False,
            enforce_containment=_t,
        )
        _end_time = time.time()
        mode = "CONTAINED" if ENFORCE_CONTAINMENT else "UNCONSTRAINED"
        print(f"[{mode}] Runtime: {_end_time - _start_time:.3f}s")

    nn_idx_vis, nn_dist2_vis = knn_ellipsoidal_distance(
        query_points,
        centroids,
        Q,
        K,
        return_sqrt=False,
        enforce_containment=ENFORCE_CONTAINMENT,
        containment_level=CONTAINMENT_LEVEL,
    )

    # Optional visualization (small sizes only)
    if d == 2 and len(results["centroids"]) < 3000 and M < 100:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Ellipse

        if IS_CUPY_BACKEND:
            centroids = xp.asnumpy(centroids)
            Q = xp.asnumpy(Q)
            query_points = xp.asnumpy(query_points)
            nn_idx_vis = xp.asnumpy(nn_idx_vis)
            nn_dist2_vis = xp.asnumpy(nn_dist2_vis)

        x = np.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        y = np.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        X_grid, Y_grid = np.meshgrid(x, y)
        Z_grid = rosenbrock(np.vstack([X_grid.ravel(), Y_grid.ravel()]).T).reshape(
            X_grid.shape
        )

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Subplot 1: centroids + ellipses
        cs = axs[0].contourf(
            X_grid,
            Y_grid,
            Z_grid,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(cs, ax=axs[0])
        axs[0].scatter(
            centroids[:, 0],
            centroids[:, 1],
            c="red",
            s=10,
            label="Centroids",
        )

        # Draw trust ellipses (level=1)
        for i in range(centroids.shape[0]):
            mu = centroids[i]
            Qi = Q[i]
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
            X_grid,
            Y_grid,
            Z_grid,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(cs2, ax=axs[1])

        dist0 = nn_dist2_vis[:, 0]
        finite = np.isfinite(dist0)
        max_finite = np.max(dist0[finite]) if bool(np.any(finite)) else 1.0
        for m in range(M):
            d0 = float(dist0[m])
            color = plt.cm.viridis(d0 / max_finite) if np.isfinite(d0) else (0, 0, 0, 1)  # type: ignore

            axs[1].scatter(
                query_points[m, 0],
                query_points[m, 1],
                c=[color],
                s=30,
                edgecolor="black",
                zorder=5,
            )

            for kk in range(K):
                cid = int(nn_idx_vis[m, kk])
                if cid < 0:
                    continue

                mu = centroids[cid]
                Qi = Q[cid]

                # Determine if actually contained (for styling)
                dx = query_points[m] - mu
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
            f"Query Points with Nearest Containing Clusters (k={K}, containment={CONTAINMENT_LEVEL}, level={CONTAINMENT_LEVEL})",
            fontsize=6,
        )

        axs[1].set_xlabel("x1")
        axs[1].set_ylabel("x2")
        axs[1].set_aspect("equal")

        plt.tight_layout()
        plt.savefig(img_fpath, dpi=300)
        print(f"Saved plot to {img_fpath}")
    else:
        print("Skipping plot due to size constraints.")
