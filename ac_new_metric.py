import time
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


# ============================================================
# Rosenbrock (scalar) + gradient (batched)
# ============================================================
def rosenbrock(X: np.ndarray) -> np.ndarray:
    """
    Rosenbrock function.
    X: (n, d)
    returns: F: (n, 1)
    """
    x_i = X[:, :-1]
    x_next = X[:, 1:]
    F = np.sum(
        100.0 * (x_next - x_i**2.0) ** 2.0 + (1.0 - x_i) ** 2.0, axis=1, keepdims=True
    )
    return F


def rosenbrock_grad(X: np.ndarray) -> np.ndarray:
    """
    Gradient of Rosenbrock.
    X: (n, d)
    returns: J: (n, 1, d)
    """
    n_pts, n_dim = X.shape
    J = np.zeros((n_pts, 1, n_dim), dtype=X.dtype)

    J[:, 0, :-1] += 2 * (X[:, :-1] - 1)
    J[:, 0, :-1] -= 400 * X[:, :-1] * (X[:, 1:] - X[:, :-1] ** 2)
    J[:, 0, 1:] += 200 * (X[:, 1:] - X[:, :-1] ** 2)
    return J


# ============================================================
# Cluster record
# ============================================================
@dataclass
class ClusterRecord:
    idx: np.ndarray  # (m,) indices into global X
    mu: np.ndarray  # (d,)
    f_mu: float  # scalar
    g_mu: np.ndarray  # (d,)
    Q: np.ndarray  # (d,d) PSD trust ellipse matrix
    E_abs: float
    E_rel: float


# ============================================================
# Linear algebra helpers (PSD abs + sqrt)
# ============================================================
def abs_psd_sym(H: np.ndarray) -> np.ndarray:
    """Return |H| = U |Λ| U^T for symmetric H."""
    w, U = np.linalg.eigh((H + H.T) * 0.5)
    w_abs = np.abs(w)
    return (U * w_abs) @ U.T


def sqrt_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return M^{1/2} for PSD M."""
    w, U = np.linalg.eigh((M + M.T) * 0.5)
    w_clip = np.clip(w, eps, None)
    return (U * np.sqrt(w_clip)) @ U.T


def top_eigvec_sym(A: np.ndarray) -> np.ndarray:
    """Top eigenvector of symmetric A."""
    w, V = np.linalg.eigh((A + A.T) * 0.5)
    return V[:, np.argmax(w)]


# ============================================================
# Geometry-based trust ellipse: PCA-aligned, ∞-projection scaling
# Q = U diag(d_j^{-2}) U^T where d_j = max_i |u_j^T y_i|
# ============================================================
def calculate_pca_ellipse_parameters(Y: np.ndarray, ridge: float = 1e-12) -> np.ndarray:
    """
    Y: (m, d) centered points (x_i - mu)
    returns Q: (d, d) PSD
    """
    # SVD on Y^T -> U gives PCA directions of columns
    U, _, _ = np.linalg.svd(Y.T, full_matrices=True)  # U: (d,d)

    # Projections: proj[j, i] = u_j^T y_i
    proj = U.T @ Y.T  # (d, m)
    d_axes = np.max(np.abs(proj), axis=1)  # (d,)

    # avoid divide-by-zero if degenerate
    d_axes = np.maximum(d_axes, ridge)
    Dinv2 = 1.0 / (d_axes * d_axes)  # (d,)
    Q = (U * Dinv2) @ U.T
    return (Q + Q.T) * 0.5


# ============================================================
# Hessian estimation by least squares from gradients:
#   ∇f(x_i) - ∇f(mu) ≈ H (x_i - mu)
# ============================================================
def estimate_hessian_ls_symmetric(
    DeltaX: np.ndarray, DeltaG: np.ndarray, lam: float = 1e-6
) -> np.ndarray:
    """
    Solve: min_H || H X - G ||_F^2  s.t. H = H^T
    with ridge: C = X X^T + lam I (Tikhonov Regularization)
    Returns symmetric H.

    DeltaX: (m, d)
    DeltaG: (m, d)
    """
    X = DeltaX.T  # (d, m)
    G = DeltaG.T  # (d, m)

    C = X @ X.T
    if lam > 0:
        C = C + lam * np.eye(C.shape[0], dtype=C.dtype)

    E = X @ G.T  # (d, d)
    S = E + E.T  # symmetric RHS

    # C = U D U^T (symmetric eigendecomp)
    dvals, U = np.linalg.eigh((C + C.T) * 0.5)
    dvals = np.maximum(dvals, 1e-18)

    # transform RHS: Gtilde = U^T S U
    Gtilde = U.T @ S @ U

    # Solve DF + FD = Gtilde elementwise
    denom = dvals[:, None] + dvals[None, :]  # (d, d)
    F = Gtilde / denom

    # back-transform: H = U F U^T
    H = U @ F @ U.T
    return (H + H.T) * 0.5


def _evaluate_frontier_batch(
    X: np.ndarray,
    frontier: list[np.ndarray],
    F_all: np.ndarray,
    F_inf: float,
    ATOL: float,
    RTOL: float,
    f_fn,
    g_fn,
    N_min: int = 0,
):
    """Evaluate all clusters in `frontier` in a mostly-batched way.

    Notes
    -----
    Full vectorization is hard because clusters are ragged (variable m), but we can:
    - compute all centroids in one pass over flattened points (O(N*d))
    - call `f_fn`/`g_fn` once on stacked centroids
    - compute per-point linearization errors and per-cluster max via reduceat
    - handle optional synthetic augmentation with a small loop over only the small clusters
    """
    K = len(frontier)
    if K == 0:
        return (
            np.empty((0, X.shape[1]), dtype=X.dtype),
            np.empty((0,), dtype=float),
            np.empty((0, X.shape[1]), dtype=X.dtype),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=bool),
        )

    sizes = np.fromiter((idx.size for idx in frontier), dtype=np.int64, count=K)
    if np.any(sizes <= 0):
        raise ValueError("frontier contains an empty cluster")

    # Flatten the ragged clusters into a single contiguous view.
    # cluster_id is contiguous by construction, so reduceat works without sorting.
    flat_idx = np.concatenate(frontier)
    cluster_id = np.repeat(np.arange(K, dtype=np.int64), sizes)

    X_flat = X[flat_idx]  # (sum_m, d)
    F_flat = F_all[flat_idx]  # (sum_m,)

    # Compute all centroids via bincount (loops over d, not over clusters)
    d = X.shape[1]
    sums = np.empty((K, d), dtype=X.dtype)
    for j in range(d):
        sums[:, j] = np.bincount(cluster_id, weights=X_flat[:, j], minlength=K).astype(
            X.dtype, copy=False
        )
    mu = sums / sizes[:, None]

    # Batch oracle calls for all centroids
    f_mu = f_fn(mu)[:, 0].astype(float, copy=False)  # (K,)
    g_mu = g_fn(mu)[:, 0, :]  # (K, d)

    # Per-point linearization error, then per-cluster max error
    DeltaX_flat = X_flat - mu[cluster_id]
    fhat_flat = f_mu[cluster_id] + np.sum(DeltaX_flat * g_mu[cluster_id], axis=1)
    err_flat = np.abs(F_flat - fhat_flat)

    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    E_abs = np.maximum.reduceat(err_flat, starts).astype(float, copy=False)

    # Optional augmentation for small clusters: avoid concatenation; just update the max.
    if N_min > 0:
        small = np.where((sizes < N_min) & (sizes > 1))[0]
        for k in small:
            idx = frontier[k]
            Xc = X[idx]
            m = Xc.shape[0]
            n_synth = int(N_min - m)

            idx0 = np.random.randint(0, m, size=n_synth)
            offset = np.random.randint(1, m, size=n_synth)
            idx1 = (idx0 + offset) % m

            alpha = np.random.uniform(0.0, 1.0, size=(n_synth, 1))
            X_synth = (1.0 - alpha) * Xc[idx0] + alpha * Xc[idx1]
            F_synth = f_fn(X_synth).reshape(-1)

            DeltaX_synth = X_synth - mu[k][None, :]
            fhat_synth = f_mu[k] + DeltaX_synth @ g_mu[k]
            e_synth = np.abs(F_synth - fhat_synth)
            E_abs[k] = float(max(E_abs[k], np.max(e_synth)))

    E_rel = E_abs / float(F_inf)
    good_mask = (E_abs < ATOL) | (E_rel < RTOL)
    return mu, f_mu, g_mu, E_abs, E_rel, good_mask


# ============================================================
# Core BFS adaptive clustering (vectorized within clusters)
# ============================================================
def adaptive_bfs_clusters(
    X: np.ndarray,
    initial_clusters: list[np.ndarray],
    ATOL: float,
    RTOL: float,
    f_fn,
    g_fn,
    lam_hess: float = 1e-6,
    N_min: int = 0,
    debug: bool = False,
) -> list[ClusterRecord]:
    """
    Scalar f assumed. Uses BFS: at each level, split all bad clusters, keep good clusters.
    Stops when no bad clusters remain.
    Returns list of ClusterRecord with requested fields.
    """
    F_all = f_fn(X).reshape(-1)  # (N,)
    G_all = g_fn(X)[:, 0, :]  # (N, d)

    F_inf = np.max(np.abs(F_all)) + 1e-30  # scalar

    final_records: list[ClusterRecord] = []
    frontier: list[np.ndarray] = initial_clusters

    time_level_start = time.perf_counter()
    num_level = 0
    while len(frontier) > 0:
        if debug:
            time_level_end = time.perf_counter()
            msg = f"Level {num_level:02}: Frontier = {len(frontier):,} | Finalized = {len(final_records):,}"
            if num_level > 0:
                msg += f" | Prev level: {time_level_end - time_level_start:.3f} s"
            print(msg)
            time_level_start = time.perf_counter()
        # ---- Evaluate all clusters in this BFS level ----
        good_clusters: list[np.ndarray] = []
        bad_clusters: list[np.ndarray] = []
        accepted_cluster_data: list[
            tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]
        ] = []
        failed_cluster_data: list[
            tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]
        ] = []

        mu_arr, f_mu_arr, g_mu_arr, E_abs_arr, E_rel_arr, good_mask = (
            _evaluate_frontier_batch(
                X=X,
                frontier=frontier,
                F_all=F_all,
                F_inf=F_inf,
                ATOL=ATOL,
                RTOL=RTOL,
                f_fn=f_fn,
                g_fn=g_fn,
                N_min=N_min,
            )
        )

        for k, idx in enumerate(frontier):
            mu = mu_arr[k]
            f_mu = float(f_mu_arr[k])
            g_mu = g_mu_arr[k]
            E_abs = float(E_abs_arr[k])
            E_rel = float(E_rel_arr[k])

            if bool(good_mask[k]):
                good_clusters.append(idx)
                accepted_cluster_data.append((idx, mu, f_mu, g_mu, E_abs, E_rel))
            else:
                bad_clusters.append(idx)
                failed_cluster_data.append((idx, mu, f_mu, g_mu, E_abs, E_rel))

        # ---- Finalize GOOD clusters with trust ellipse Q (batched per cluster) ----
        for idx, mu, f_mu, g_mu, E_abs, E_rel in accepted_cluster_data:
            Y = X[idx] - mu[None, :]  # (m, d)
            Q = calculate_pca_ellipse_parameters(Y)
            final_records.append(
                ClusterRecord(
                    idx=idx,
                    mu=mu,
                    f_mu=f_mu,
                    g_mu=g_mu,
                    Q=Q,
                    E_abs=E_abs,
                    E_rel=E_rel,
                )
            )

        # ---- Split BAD clusters using Hessian-weighted PCA ----
        next_frontier: list[np.ndarray] = []

        for idx, mu, f_mu, g_mu, E_abs, E_rel in failed_cluster_data:
            Xc = X[idx]  # (m, d)
            DeltaX = Xc - mu[None, :]  # (m, d)

            # Gradient diffs using cached gradients at points
            DeltaG = G_all[idx] - g_mu[None, :]  # (m, d)

            H = estimate_hessian_ls_symmetric(DeltaX, DeltaG, lam=lam_hess)
            M = abs_psd_sym(H)  # |H| (PSD)
            S = sqrt_psd(M)  # |H|^{1/2}

            # Transform points: z_i = S (x_i - mu)
            Z = (S @ DeltaX.T).T  # (m, d)

            # Top PC direction in Z-space
            Cov = Z.T @ Z  # (d, d)
            v = top_eigvec_sym(Cov)  # (d,)

            scores = Z @ v  # (m,)
            left_mask = scores >= 0
            right_mask = ~left_mask

            # Avoid degenerate split (all on one side)
            if left_mask.all() or right_mask.all():
                # fallback: split by farthest-from-centroid in Euclidean
                r2 = np.sum(DeltaX * DeltaX, axis=1)
                pivot = int(np.argmax(r2))
                # split by sign of projection on (x_pivot - mu)
                dir_vec = DeltaX[pivot]
                if np.linalg.norm(dir_vec) < 1e-12:
                    # final fallback: random half split
                    perm = np.random.permutation(len(idx))
                    mid = len(idx) // 2
                    left_idx = idx[perm[:mid]]
                    right_idx = idx[perm[mid:]]
                else:
                    scores2 = DeltaX @ dir_vec
                    left_idx = idx[scores2 >= 0]
                    right_idx = idx[scores2 < 0]
            else:
                left_idx = idx[left_mask]
                right_idx = idx[right_mask]

            next_frontier.append(left_idx)
            next_frontier.append(right_idx)

        # Stop if there are no bad clusters (i.e., next_frontier empty)
        frontier = next_frontier
        num_level += 1

    return final_records


# ============================================================
# Demo / test on Rosenbrock
# ============================================================
if __name__ == "__main__":
    np.random.seed(0)

    # Generate test points
    d = 10
    PROB_LIMS = [(-2.0, 3.0) for _ in range(d)]
    assert d == len(PROB_LIMS), "Adjust PROB_LIMS for desired d."
    _N = 400_000
    X_list = []
    for i in range(d):
        x_min, x_max = PROB_LIMS[i]
        Xi = np.random.uniform(x_min, x_max, size=(_N, 1))
        X_list.append(Xi)
    X = np.hstack(X_list)  # (_N, d)
    N = X.shape[0]
    print(f"Test points: N={N:,}, d={d}")

    start_time = time.perf_counter()
    # Initial clusters via KMeans
    tic_init_clust = time.perf_counter()
    K0 = 10
    kmeans = KMeans(n_clusters=K0, n_init="auto", random_state=0)
    labels = kmeans.fit_predict(X)
    print(f"Initial KMeans clusters: K0={K0}")
    toc_init_clust = time.perf_counter()
    print(
        f"    Time for initial KMeans clustering: {toc_init_clust - tic_init_clust:.3f} s"
    )

    # Remove empty clusters
    initial_clusters = [np.where(labels == k)[0] for k in range(K0)]
    initial_clusters = [idx for idx in initial_clusters if idx.size > 0]

    # Tolerances
    ATOL = 1e-2
    RTOL = 1e-2
    N_min = 100
    print(f"Clustering tolerances: ATOL={ATOL}, RTOL={RTOL}, N_min={N_min}")

    tic_ac = time.perf_counter()
    final_clusters = adaptive_bfs_clusters(
        X=X,
        initial_clusters=initial_clusters,
        ATOL=ATOL,
        RTOL=RTOL,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        lam_hess=1e-4,
        N_min=N_min,
        debug=True,
    )
    toc_ac = time.perf_counter()
    print(
        f"    Adaptive clustering time: {toc_ac - tic_ac:.3f} s, Total time: {toc_ac - start_time:.3f} s"
    )

    # Print summary
    E_abs_vals = np.array([c.E_abs for c in final_clusters])
    E_rel_vals = np.array([c.E_rel for c in final_clusters])
    sizes = np.array([c.idx.size for c in final_clusters])

    print(f"Final clusters: {len(final_clusters):,}")
    print(
        "Size stats: min/mean/max =",
        round(sizes.min(), 3),
        round(sizes.mean(), 3),
        round(sizes.max(), 3),
    )
    print(
        "E_abs stats: min/mean/max =",
        round(E_abs_vals.min(), 3),
        round(E_abs_vals.mean(), 3),
        round(E_abs_vals.max(), 3),
    )
    print(
        "E_rel stats: min/mean/max =",
        round(E_rel_vals.min(), 3),
        round(E_rel_vals.mean(), 3),
        round(E_rel_vals.max(), 3),
    )

    if d == 2:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        # 1 plot 4 subplots (2X2)
        # Subplot 1: coloured contourf of Rosenbrock
        # Subplot 2: scatter of initial cluster centroids on contourf
        # Subplot 3: scatter of final cluster centroids on contourf
        # Subplot 4: scatter of final cluster centroids with trust ellipses on contourf

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

        # Subplot 2: initial cluster centroids
        axs[0, 1].contourf(X1, X2, np.log10(F_grid + 1e-8), levels=50, cmap="viridis")
        for idx in initial_clusters:
            Xc = X[idx]
            mu = Xc.mean(axis=0)
            axs[0, 1].scatter(mu[0], mu[1], c="k", s=40)
        fig.colorbar(cs, ax=axs[0, 1])
        axs[0, 1].set_title("Initial cluster centroids")
        axs[0, 1].set_xlabel("x1")
        axs[0, 1].set_ylabel("x2")
        axs[0, 1].set_aspect("equal")

        # Subplot 3: final cluster centroids
        axs[1, 0].contourf(X1, X2, np.log10(F_grid + 1e-8), levels=50, cmap="viridis")
        for c in final_clusters:
            mu = c.mu
            axs[1, 0].scatter(mu[0], mu[1], c="k", s=40)
        fig.colorbar(cs, ax=axs[1, 0])
        axs[1, 0].set_title("Final cluster centroids")
        axs[1, 0].set_xlabel("x1")
        axs[1, 0].set_ylabel("x2")
        axs[1, 0].set_aspect("equal")

        # Subplot 4: final cluster centroids with trust ellipses
        axs[1, 1].set_title("Final clusters with trust ellipses")
        axs[1, 1].set_xlabel("x1")
        axs[1, 1].set_ylabel("x2")
        axs[1, 1].set_aspect("equal")

        cs4 = axs[1, 1].contourf(
            X1, X2, np.log10(F_grid + 1e-8), levels=50, cmap="viridis"
        )
        level = 1.0  # ellipse corresponds to (x-mu)^T Q (x-mu) <= level
        for c in final_clusters:
            mu = c.mu
            Q = c.Q

            # Eigen-decomposition of Q (symmetric PSD)
            w, V = np.linalg.eigh((Q + Q.T) * 0.5)
            w = np.maximum(w, 1e-15)

            # Sort eigenpairs so column 0 is the major axis (largest eigenvalue)
            order = np.argsort(w)[::-1]
            w = w[order]
            V = V[:, order]

            # Semi-axis lengths for (x-mu)^T Q (x-mu) <= level:
            # a = sqrt(level)/sqrt(w0), b = sqrt(level)/sqrt(w1)
            a = np.sqrt(level) / np.sqrt(w[0])
            b = np.sqrt(level) / np.sqrt(w[1])

            # Angle of major axis (eigenvector corresponding to w[0])
            angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))

            ellipse = Ellipse(
                xy=(mu[0], mu[1]),
                width=2.0 * a,  # diameter
                height=2.0 * b,  # diameter
                angle=angle,
                edgecolor="red",
                facecolor="none",
                lw=1,
            )
            axs[1, 1].add_patch(ellipse)
            axs[1, 1].scatter(mu[0], mu[1], c="k", s=20)

        fig.colorbar(cs4, ax=axs[1, 1])

        plt.tight_layout()
        # Add title to entire figure
        fig.suptitle(
            f"Adaptive Clustering on Rosenbrock: N={N}, d={d}, ATOL={ATOL}, RTOL={RTOL}, Final Clusters={len(final_clusters)}",
            fontsize=16,
        )
        plt.subplots_adjust(top=0.92)
        fname = f"rosenbrock_adaptive_clusters_N{N}_ATOL{ATOL}_RTOL{RTOL}.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved 2D cluster plot to: {fname}")
    else:
        print("Skipping 2D plot since d != 2.")
