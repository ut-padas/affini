import time

import numpy as np

# import cupy as cp
cp = np
from sklearn.cluster import KMeans


def batched_estimate_hessian_symmetric(DeltaX, DeltaG, starts, counts):
    """
    Computes H for K clusters simultaneously using segmented aggregation.
    """
    d = int(DeltaX.shape[1])
    # 1. Compute outer products for every point: (N, d, d)
    # This might use a lot of memory. If N is huge, we might need to chunk this.
    # For N=100k, d=10 -> 100k * 100 * 8 bytes = 80MB. Totally fine.

    # XX_i = x_i * x_i.T
    XX_all = DeltaX[:, :, None] * DeltaX[:, None, :]

    # XG_i = x_i * g_i.T
    XG_all = DeltaX[:, :, None] * DeltaG[:, None, :]

    # 2. Aggregate by cluster using reduceat (requires data to be sorted by cluster!)
    # We assume DeltaX is already sorted by cluster_id.

    # XX_sum: (K, d, d)
    XX_sum = cp.add.reduceat(XX_all, starts)
    XG_sum = cp.add.reduceat(XG_all, starts)

    C = XX_sum

    # Symmetrize RHS: S = E + E.T
    S = XG_sum + XG_sum.transpose(0, 2, 1)

    # 4. Solve Sylvester-like system per cluster.
    # H C + C H = S  (Lyapunov equation)
    # Since C is symmetric positive definite (mostly), we can use eigendecomposition.

    # Batched Eigh: (K, d) eigenvalues, (K, d, d) eigenvectors
    # C = U D U^T
    dvals, U = cp.linalg.eigh((C + C.transpose(0, 2, 1)) * 0.5)
    dvals = cp.maximum(dvals, 1e-18)

    # Transform RHS: Gtilde = U^T S U
    # Matmul: (K, d, d) @ (K, d, d) -> (K, d, d)
    Gtilde = U.transpose(0, 2, 1) @ S @ U

    # Solve elementwise: F_ij = Gtilde_ij / (d_i + d_j)
    denom = dvals[:, :, None] + dvals[:, None, :]
    F = Gtilde / denom

    # Back transform: H = U F U^T
    H = U @ F @ U.transpose(0, 2, 1)
    return (H + H.transpose(0, 2, 1)) * 0.5


def batched_sqrt_abs_psd(H):
    """
    Computes S = |H|^0.5 for a batch of symmetric matrices H.
    H: (K, d, d)
    Returns S: (K, d, d)
    """
    # 1. Eigendecomposition
    w, U = cp.linalg.eigh((H + H.transpose(0, 2, 1)) * 0.5)

    # 2. Abs and Sqrt
    w_abs_sqrt = cp.sqrt(cp.abs(w))

    # 3. Reconstruct: U diag(w) U^T
    # Broadcast multiply: U * w[:, None, :] scales columns of U
    S = (U * w_abs_sqrt[:, None, :]) @ U.transpose(0, 2, 1)
    return S


def batched_top_eigvec(Cov):
    """
    Returns top eigenvector v for batch of covariance matrices.
    Cov: (K, d, d)
    Returns: v (K, d)
    """
    w, V = cp.linalg.eigh((Cov + Cov.transpose(0, 2, 1)) * 0.5)
    # eigh returns eigenvalues in ascending order, so last one is max
    return V[:, :, -1]


def batched_calculate_Q(DeltaX, starts, counts, K, ridge=1e-12):
    """
    Batched version of PCA ellipse parameters.
    """
    # This is tricky because SVD on ragged arrays isn't standard batched op.
    # However, we can compute the covariance matrix (d,d) per cluster easily,
    # and get the PCA directions from that, which is equivalent to SVD on centered data.

    # 1. Compute Covariances: (K, d, d)
    XX_all = DeltaX[:, :, None] * DeltaX[:, None, :]
    Cov = cp.add.reduceat(XX_all, starts)  # This is Y^T Y roughly

    # 2. Eigen vectors of Cov give us V (which is U in the SVD of Y^T)
    # Y = U S V^T -> Y^T Y = V S^2 V^T.
    # The U in the original code comes from SVD of Y.T (d, m).
    # original: U, _, _ = svd(Y.T). U is (d,d).
    # This U is actually the eigenvectors of Y.T @ Y if m > d.
    # Wait, Y.T is (d, m). Y.T @ Y is (d, d).
    # So we just need eigendecomposition of the scatter matrix.

    _, U = cp.linalg.eigh(Cov)
    # U columns are principal components.
    # Note: eigh sorts ascending, we usually want descending for PCA,
    # but the formula uses all axes so order doesn't strictly matter
    # as long as we align d_axes correctly.

    # 3. Project points: proj = U^T @ y_i
    # We need to do this efficiently.
    # DeltaX: (N, d). U: (K, d, d).
    # We need to match each point to its cluster's U.
    # Expand U to (N, d, d) via repeat/indexing

    # We need an array 'map_n_to_k' of size N
    map_n_to_k = cp.zeros(DeltaX.shape[0], dtype=cp.int32)
    # Fill this map. A cumsum approach or repeat count
    # Since we have counts, we can repeat indices
    cluster_indices = cp.arange(K)
    map_n_to_k = cp.repeat(cluster_indices, counts)

    U_expanded = U[map_n_to_k]  # (N, d, d)

    # proj_i = U_k.T @ x_i  -> (d, d) @ (d, 1)
    # (N, d, d) transpose is (N, d, d) swapping last two dims
    # (N, d, 1) input

    # Einsum is cleaner: P_ij = sum_k U_kji * X_ik (vector per point)
    # U_expanded: (N, rows, cols). We want U.T @ x.
    # U_expanded.transpose(0, 2, 1) @ DeltaX[:, :, None]

    proj = U_expanded.transpose(0, 2, 1) @ DeltaX[:, :, None]  # (N, d, 1)
    proj = proj[:, :, 0]  # (N, d)

    # 4. Max projection per axis per cluster
    proj_abs = cp.abs(proj)
    # reduceat max
    d_axes = cp.maximum.reduceat(proj_abs, starts)  # (K, d)

    d_axes = cp.maximum(d_axes, ridge)
    Dinv2 = 1.0 / (d_axes**2)  # (K, d)

    # 5. Q = U Dinv2 U^T
    # Broadcast Dinv2 into diagonal matrix
    # U is (K, d, d).
    # Scale U columns by sqrt(Dinv2) or similar?
    # Q = (U * Dinv2[:, None, :]) @ U.T ??
    # Dinv2 is per column of U.

    Q_inner = U * Dinv2[:, None, :]  # Scale columns of U by weights
    Q = Q_inner @ U.transpose(0, 2, 1)

    return (Q + Q.transpose(0, 2, 1)) * 0.5


def _init_output_buffers(max_estimated_clusters, d, dtype):
    cap = max(1, int(max_estimated_clusters))
    buffers = {
        "centroids": cp.empty((cap, d), dtype=dtype),
        "f": cp.empty((cap,), dtype=dtype),
        "g": cp.empty((cap, d), dtype=dtype),
        "Q": cp.empty((cap, d, d), dtype=dtype),
        "E_abs": cp.empty((cap,), dtype=dtype),
        "E_rel": cp.empty((cap,), dtype=dtype),
        "counts": cp.empty((cap,), dtype=cp.int64),
    }
    return buffers, cap


def _ensure_capacity(buffers, out_n, cap, required_total):
    required_total = int(required_total)
    if required_total <= cap:
        return buffers, cap

    # Grow geometrically to reduce future reallocations.
    new_cap = max(required_total, max(1, cap) * 2)

    d = int(buffers["centroids"].shape[1])
    dtype = buffers["centroids"].dtype

    new_buffers = {
        "centroids": cp.empty((new_cap, d), dtype=dtype),
        "f": cp.empty((new_cap,), dtype=dtype),
        "g": cp.empty((new_cap, d), dtype=dtype),
        "Q": cp.empty((new_cap, d, d), dtype=dtype),
        "E_abs": cp.empty((new_cap,), dtype=dtype),
        "E_rel": cp.empty((new_cap,), dtype=dtype),
        "counts": cp.empty((new_cap,), dtype=buffers["counts"].dtype),
    }

    n = int(out_n)
    if n > 0:
        for k in new_buffers:
            new_buffers[k][:n] = buffers[k][:n]

    return new_buffers, int(new_cap)


def adaptive_clustering_fit(
    X_cpu,
    labels,
    *,
    f_fn,
    g_fn,
    ATOL=1e-2,
    RTOL=1e-2,
    N_min=100,
    max_estimated_clusters=100_000,
    verbose=True,
):
    # 1. Transfer to GPU
    X = cp.asarray(X_cpu)
    labels = cp.asarray(labels).copy()
    N = int(X.shape[0])
    d = int(X.shape[1])

    # Initialize output buffers now that we know dimension + dtype.
    buffers, cap = _init_output_buffers(max_estimated_clusters, d=d, dtype=X.dtype)
    out_n = 0

    # Calculate F_inf for relative error
    F_all = f_fn(X).ravel()
    G_all = g_fn(X)[:, 0, :]
    F_inf = float(cp.max(cp.abs(F_all))) + 1e-30

    # Frontier State
    active_mask = cp.ones(N, dtype=bool)  # All points active initially

    level = 0
    while True:
        n_active = int(cp.sum(active_mask))
        if n_active == 0:
            break

        t_start = time.perf_counter()

        # --- 1. Compact & Sort Frontier ---
        active_indices = cp.nonzero(active_mask)[0]

        # Sort active points by their label to enable reduceat
        active_labels = labels[active_indices]
        sort_order = cp.argsort(active_labels)

        sorted_indices = active_indices[sort_order]
        sorted_labels = active_labels[sort_order]
        sorted_X = X[sorted_indices]
        sorted_F = F_all[sorted_indices]
        sorted_G = G_all[sorted_indices]

        # Get segment info
        unique_cluster_ids, starts, counts = cp.unique(
            sorted_labels, return_index=True, return_counts=True
        )
        K_batch = int(unique_cluster_ids.shape[0])

        # --- 2. Batched Centroids ---
        sums = cp.add.reduceat(sorted_X, starts)
        mus = sums / counts[:, None]

        # --- 3. Batched Oracle ---
        f_mus = f_fn(mus)[:, 0]  # (K,)
        g_mus = g_fn(mus)[:, 0, :]  # (K, d)

        # --- 4. Batched Error Check ---
        map_n_to_k = cp.repeat(cp.arange(K_batch), counts)
        DeltaX = sorted_X - mus[map_n_to_k]
        f_linear = f_mus[map_n_to_k] + cp.sum(DeltaX * g_mus[map_n_to_k], axis=1)
        err_point = cp.abs(sorted_F - f_linear)

        # Max error per cluster
        E_abs = cp.maximum.reduceat(err_point, starts)
        E_rel = E_abs / F_inf

        # --- 5. Decision ---
        is_good = (E_abs < ATOL) | (E_rel < RTOL)

        # Extract Good Clusters
        good_indices = cp.nonzero(is_good)[0]
        if good_indices.size > 0:
            cluster_good_mask = is_good
            point_good_mask = cluster_good_mask[map_n_to_k]

            good_DeltaX = DeltaX[point_good_mask]
            good_counts = counts[good_indices]
            good_starts = cp.r_[0, cp.cumsum(good_counts)[:-1]]

            Qs = batched_calculate_Q(
                good_DeltaX, good_starts, good_counts, int(good_indices.size)
            )

            # Write into preallocated buffers.
            n_good = int(good_indices.size)
            start_out = int(out_n)
            end_out = start_out + n_good
            buffers, cap = _ensure_capacity(
                buffers, out_n=out_n, cap=cap, required_total=end_out
            )

            buffers["centroids"][start_out:end_out] = mus[good_indices]
            buffers["f"][start_out:end_out] = f_mus[good_indices]
            buffers["g"][start_out:end_out] = g_mus[good_indices]
            buffers["Q"][start_out:end_out] = Qs
            buffers["E_abs"][start_out:end_out] = E_abs[good_indices]
            buffers["E_rel"][start_out:end_out] = E_rel[good_indices]
            buffers["counts"][start_out:end_out] = good_counts
            out_n = int(end_out)

            # Mark points as inactive
            global_indices_good = sorted_indices[point_good_mask]
            active_mask[global_indices_good] = False

        # Extract Bad Clusters for Splitting
        bad_indices = cp.nonzero(~is_good)[0]
        if bad_indices.size == 0:
            break  # Done

        # --- 6. Batched Splitting Logic ---
        cluster_bad_mask = ~is_good
        point_bad_mask = cluster_bad_mask[map_n_to_k]

        bad_DeltaX = DeltaX[point_bad_mask]

        # For gradients, we need original G and mu
        bad_G_points = sorted_G[point_bad_mask]
        bad_g_mus = g_mus[map_n_to_k][point_bad_mask]
        bad_DeltaG = bad_G_points - bad_g_mus

        bad_counts = counts[bad_indices]
        bad_starts = cp.r_[0, cp.cumsum(bad_counts)[:-1]]
        K_bad = int(bad_indices.size)

        # 6a. Estimate Hessian
        H = batched_estimate_hessian_symmetric(
            bad_DeltaX, bad_DeltaG, bad_starts, bad_counts
        )

        # 6b. Transform Geometry
        M = batched_sqrt_abs_psd(H)  # (K_bad, d, d)

        # Map bad clusters to points
        map_bad_n_to_k = cp.repeat(cp.arange(K_bad), bad_counts)
        M_expanded = M[map_bad_n_to_k]
        Z = (M_expanded @ bad_DeltaX[:, :, None])[:, :, 0]

        # 6c. PCA on Z
        Z_starts = bad_starts
        ZZ_all = Z[:, :, None] * Z[:, None, :]
        Cov_Z = cp.add.reduceat(ZZ_all, Z_starts)
        v = batched_top_eigvec(Cov_Z)  # (K_bad, d)

        # 6d. Scores and Split
        v_expanded = v[map_bad_n_to_k]
        scores = cp.sum(Z * v_expanded, axis=1)
        split_decision = scores >= 0

        # Update Labels
        global_indices_bad = sorted_indices[point_bad_mask]
        old_ids = sorted_labels[point_bad_mask]
        new_ids = old_ids * 2 + split_decision.astype(cp.int32)
        labels[global_indices_bad] = new_ids

        level += 1
        t_end = time.perf_counter()
        if verbose:
            print(
                f"Level {level}: Active {n_active:,} -> {K_batch} clusters (Good: {good_indices.size}, Bad: {bad_indices.size}) | {t_end - t_start:.4f}s"
            )

    n_final = int(out_n)
    results = {k: buffers[k][:n_final] for k in buffers}
    return results


ROSENBROCK_SCALE = 100.0


def rosenbrock(X):
    x_i = X[:, :-1]
    x_next = X[:, 1:]
    F = cp.sum(
        ROSENBROCK_SCALE * (x_next - x_i**2.0) ** 2.0 + (1.0 - x_i) ** 2.0,
        axis=1,
        keepdims=True,
    )
    return F


def rosenbrock_grad(X):
    n_pts, n_dim = X.shape
    J = cp.zeros((n_pts, 1, n_dim), dtype=X.dtype)
    J[:, 0, :-1] += 2 * (X[:, :-1] - 1)
    J[:, 0, :-1] -= 4 * ROSENBROCK_SCALE * X[:, :-1] * (X[:, 1:] - X[:, :-1] ** 2)
    J[:, 0, 1:] += 2 * ROSENBROCK_SCALE * (X[:, 1:] - X[:, :-1] ** 2)
    return J


if __name__ == "__main__":
    cp.random.seed(0)

    _N = 100_000
    d = 2
    print(f"Generating {_N:,} points (d={d})...")

    X_list = []
    PROB_LIMS = [(-2.0, 3.0) for _ in range(d)]
    for i in range(d):
        x_min, x_max = PROB_LIMS[i]
        Xi = cp.random.uniform(x_min, x_max, size=(_N, 1))
        X_list.append(Xi)
    X_gpu = cp.hstack(X_list)

    print("Running Init KMeans (CPU on sample)...")
    idx_sample = np.random.choice(_N, min(10000, _N), replace=False)
    X_sample_cpu = X_gpu[idx_sample]

    K0 = 10
    kmeans = KMeans(n_clusters=K0, n_init=1, random_state=0)
    kmeans.fit(X_sample_cpu)

    centroids_gpu = cp.asarray(kmeans.cluster_centers_)
    dists = cp.linalg.norm(X_gpu[:, None, :] - centroids_gpu[None, :, :], axis=2)
    initial_labels = cp.argmin(dists, axis=1)
    initial_labels0 = initial_labels.copy()

    print("Starting Adaptive GPU Clustering...")
    start_total = time.perf_counter()

    ATOL = 1e-3
    RTOL = 1e-3
    results = adaptive_clustering_fit(
        X_gpu,
        initial_labels,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        ATOL=ATOL,
        RTOL=RTOL,
        max_estimated_clusters=100_000,
        verbose=True,
    )
    end_total = time.perf_counter()

    print("-" * 40)
    print(f"Total Time: {end_total - start_total:.4f} s")
    print(f"Final Cluster Count: {results['centroids'].shape[0]:,}")
    print(f"Points Processed: {cp.sum(results['counts']):,}")

    sizes = results["counts"]
    print(
        f"Size Stats: Min {cp.min(sizes)} | Max {cp.max(sizes)} | Mean {cp.mean(sizes):.1f}"
    )

    E_abs_vals = results["E_abs"]
    E_rel_vals = results["E_rel"]
    print(
        f"E_abs Stats: Min {cp.min(E_abs_vals):.3g} | Max {cp.max(E_abs_vals):.3g} | Mean {cp.mean(E_abs_vals):.3g}"
    )
    print(
        f"E_rel Stats: Min {cp.min(E_rel_vals):.3g} | Max {cp.max(E_rel_vals):.3g} | Mean {cp.mean(E_rel_vals):.3g}"
    )

    if d == 2:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
        except Exception as e:
            print(f"Skipping 2D plot (matplotlib import failed): {e}")
        else:
            X = np.asarray(X_gpu)
            initial_labels_np = np.asarray(initial_labels0)
            counts0 = np.bincount(initial_labels_np, minlength=K0).astype(np.float64)
            counts0 = np.maximum(counts0, 1.0)
            sums0 = np.stack(
                [
                    np.bincount(initial_labels_np, weights=X[:, j], minlength=K0)
                    for j in range(d)
                ],
                axis=1,
            )
            centroids0 = sums0 / counts0[:, None]
            final_centroids = np.asarray(results["centroids"])
            final_Q = np.asarray(results["Q"])

            x1 = np.linspace(PROB_LIMS[0][0], PROB_LIMS[0][1], 300)
            x2 = np.linspace(PROB_LIMS[1][0], PROB_LIMS[1][1], 300)
            X1, X2 = np.meshgrid(x1, x2)
            X_grid = np.stack([X1.ravel(), X2.ravel()], axis=1)
            F_grid = rosenbrock(X_grid).reshape(X1.shape)

            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            cs = axs[0, 0].contourf(
                X1, X2, np.log10(F_grid + 1e-8), levels=50, cmap="viridis"
            )
            fig.colorbar(cs, ax=axs[0, 0])
            axs[0, 0].set_title("Rosenbrock function (log10 scale)")
            axs[0, 0].set_xlabel("x1")
            axs[0, 0].set_ylabel("x2")
            axs[0, 0].set_aspect("equal")

            axs[0, 1].contourf(
                X1, X2, np.log10(F_grid + 1e-8), levels=50, cmap="viridis"
            )
            axs[0, 1].scatter(centroids0[:, 0], centroids0[:, 1], c="k", s=40)
            fig.colorbar(cs, ax=axs[0, 1])
            axs[0, 1].set_title("Initial cluster centroids")
            axs[0, 1].set_xlabel("x1")
            axs[0, 1].set_ylabel("x2")
            axs[0, 1].set_aspect("equal")

            axs[1, 0].contourf(
                X1, X2, np.log10(F_grid + 1e-8), levels=50, cmap="viridis"
            )
            axs[1, 0].scatter(final_centroids[:, 0], final_centroids[:, 1], c="k", s=40)
            fig.colorbar(cs, ax=axs[1, 0])
            axs[1, 0].set_title("Final cluster centroids")
            axs[1, 0].set_xlabel("x1")
            axs[1, 0].set_ylabel("x2")
            axs[1, 0].set_aspect("equal")

            axs[1, 1].set_title("Final clusters with trust ellipses")
            axs[1, 1].set_xlabel("x1")
            axs[1, 1].set_ylabel("x2")
            axs[1, 1].set_aspect("equal")

            cs4 = axs[1, 1].contourf(
                X1, X2, np.log10(F_grid + 1e-8), levels=50, cmap="viridis"
            )
            fig.colorbar(cs4, ax=axs[1, 1])

            level = 1.0  # ellipse: (x-mu)^T Q (x-mu) <= level
            for mu, Q in zip(final_centroids, final_Q):
                w, V = np.linalg.eigh((Q + Q.T) * 0.5)
                w = np.maximum(w, 1e-15)

                order = np.argsort(w)[::-1]
                w = w[order]
                V = V[:, order]

                a = np.sqrt(level) / np.sqrt(w[0])
                b = np.sqrt(level) / np.sqrt(w[1])
                angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))

                ellipse = Ellipse(
                    xy=(mu[0], mu[1]),
                    width=2.0 * a,
                    height=2.0 * b,
                    angle=angle,
                    edgecolor="red",
                    facecolor="none",
                    lw=1,
                )
                axs[1, 1].add_patch(ellipse)
                axs[1, 1].scatter(mu[0], mu[1], c="k", s=20)

            plt.tight_layout()
            fig.suptitle(
                f"Adaptive Clustering on Rosenbrock: N={_N}, d={d}, ATOL={ATOL}, RTOL={RTOL}, Final Clusters={final_centroids.shape[0]}",
                fontsize=16,
            )
            plt.subplots_adjust(top=0.92)
            fname = (
                f"temp_rosenbrock_adaptive_clusters_N{_N}_ATOL{ATOL:g}_RTOL{RTOL:g}.png"
            )
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"Saved 2D cluster plot to: {fname}")
    else:
        print("Skipping 2D plot since d != 2.")
