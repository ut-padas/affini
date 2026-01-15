import cupy as cp


def knn_mahalanobis_clusters(
    X_new: cp.ndarray,  # (M, d)
    centroids: cp.ndarray,  # (K, d)
    Q: cp.ndarray,  # (K, d, d) PSD / symmetric
    k: int,
    *,
    chunk_size: int = 2048,
    return_sqrt: bool = False,
    eps: float = 0.0,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Compute k-nearest clusters for each query x in X_new using cluster-wise
    Mahalanobis distance induced by Q_k:

        dist2[m, k] = (x_m - mu_k)^T Q_k (x_m - mu_k)

    This is GPU-only and uses streaming top-k over cluster chunks to limit memory.

    Args:
        X_new: (M, d) query points.
        centroids: (K, d) cluster centroids.
        Q: (K, d, d) cluster trust/metric matrices (assumed symmetric PSD-ish).
        k: number of nearest clusters to return per query.
        chunk_size: number of clusters to process per chunk (memory knob).
        return_sqrt: if True, return sqrt(dist2) instead of dist2.
        eps: optional ridge added to dist2 before sqrt; use >0 if you worry about
             tiny negative numerical noise in PSD computations.

    Returns:
        nn_idx: (M, k) int32 indices of nearest clusters.
        nn_dist: (M, k) distances (squared unless return_sqrt=True).
    """
    X_new = cp.asarray(X_new)
    centroids = cp.asarray(centroids)
    Q = cp.asarray(Q)

    M, d = X_new.shape
    K = centroids.shape[0]
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > K:
        raise ValueError(f"k={k} cannot exceed number of clusters K={K}.")
    if centroids.shape[1] != d:
        raise ValueError("centroids must have shape (K, d) matching X_new.")
    if Q.shape != (K, d, d):
        raise ValueError("Q must have shape (K, d, d).")

    # Running best top-k per query
    best_dist2 = cp.full((M, k), cp.inf, dtype=X_new.dtype)
    best_idx = cp.full((M, k), -1, dtype=cp.int32)

    # Process clusters in chunks to avoid (M, K, d) allocation
    for start in range(0, K, chunk_size):
        end = min(K, start + chunk_size)
        C = end - start

        mu_c = centroids[start:end]  # (C, d)
        Q_c = Q[start:end]  # (C, d, d)

        # dx: (M, C, d) = X_new[:, None, :] - mu_c[None, :, :]
        dx = X_new[:, None, :] - mu_c[None, :, :]

        # dist2: (M, C) where dist2[m, c] = dx[m,c,:]^T Q_c[c,:,:] dx[m,c,:]
        # einsum avoids forming intermediate (M,C,d) @ (C,d,d) in a weird broadcasted way
        dist2 = cp.einsum("mcd,cde,mce->mc", dx, Q_c, dx, optimize=True)

        # Local top-k within this chunk (per row)
        # argpartition is O(C) and faster than full sort.
        part = cp.argpartition(dist2, kth=k - 1, axis=1)[:, :k]  # (M, k)
        part_dist2 = cp.take_along_axis(dist2, part, axis=1)  # (M, k)
        part_idx = (part + start).astype(cp.int32)  # global cluster ids

        # Merge current best with new candidates: shape (M, 2k)
        merged_dist2 = cp.concatenate([best_dist2, part_dist2], axis=1)
        merged_idx = cp.concatenate([best_idx, part_idx], axis=1)

        # Re-select top-k from merged (per row)
        sel = cp.argpartition(merged_dist2, kth=k - 1, axis=1)[:, :k]
        best_dist2 = cp.take_along_axis(merged_dist2, sel, axis=1)
        best_idx = cp.take_along_axis(merged_idx, sel, axis=1)

    # Optional: sort the final k neighbors by distance (ascending)
    order = cp.argsort(best_dist2, axis=1)
    best_dist2 = cp.take_along_axis(best_dist2, order, axis=1)
    best_idx = cp.take_along_axis(best_idx, order, axis=1)

    if return_sqrt:
        # Guard against tiny negative due to numerical noise
        best_dist2 = cp.sqrt(cp.maximum(best_dist2 + eps, 0))

    return best_idx, best_dist2


if __name__ == "__main__":
    from src.gpu_adaptive_clustering_metric import adaptive_clustering
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

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
        N_INIT_CLUSTERS=N_INIT_CLUSTERS,
        N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
        MAX_EST_CLUSTERS=100_000,
    )

    print("-" * 40)
    print(f"Final Cluster Count: {results['centroids'].shape[0]:,}")

    # Test knn_mahalanobis_clusters
    print("Testing knn_mahalanobis_clusters...")
    M = 5000
    k = 5
    X_new = cp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(M, d))
    centroids = results["centroids"]
    Q = results["Q_matrices"]
    print(f"Finding {k} nearest clusters for {M:,} new points...")
    nn_idx, nn_dist = knn_mahalanobis_clusters(
        X_new, centroids, Q, k, chunk_size=5000, return_sqrt=False
    )

    # Print some results
    for i in range(5):
        print(f"Query Point {i}: {X_new[i].get()}")
        print(f"  Nearest Clusters Indices: {nn_idx[i].get()}")
        print(f"  Nearest Clusters Distances^2: {nn_dist[i].get()}")
