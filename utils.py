import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

# Adjust imports based on your actual source tree
from src import IS_CUPY_BACKEND, Array, xp

logger = logging.getLogger(__name__)

TINY_EPS = 1e-12

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & MODES
# -----------------------------------------------------------------------------


class SurrogateMode(Enum):
    AFFINE_COV = "affine_cov"  # 1. No grads, fit A/b, split via covariance
    TAYLOR_HESSIAN = "taylor_hessian"  # 2. Grads everywhere, split via Hessian LSQ
    TAYLOR_COV_CENTROID = (
        "taylor_cov_centroid"  # 3. Grads at centroids only, split via covariance
    )
    NEAREST_NEIGHBOR = "nearest_neighbor"  # 4. No grads, f(mu) only, split via PCA/XX^T


@dataclass
class ClusteringConfig:
    mode: SurrogateMode
    error_metric: str  # "L2", "Linf-global", "Linf-local"
    store_hess: bool = False
    store_affine_fits: bool = False

    @property
    def requires_grad_everywhere(self) -> bool:
        return self.mode == SurrogateMode.TAYLOR_HESSIAN

    @property
    def requires_grad_centroids(self) -> bool:
        return self.mode in (
            SurrogateMode.TAYLOR_HESSIAN,
            SurrogateMode.TAYLOR_COV_CENTROID,
        )

    @property
    def requires_no_grads(self) -> bool:
        return self.mode in (
            SurrogateMode.AFFINE_COV,
            SurrogateMode.NEAREST_NEIGHBOR,
        )


# -----------------------------------------------------------------------------
# 2. MATH HELPERS (Vectorized xp)
# -----------------------------------------------------------------------------


def repeats(arr: Array, counts: Array) -> Array:
    """Repeats elements of an array given a counts array."""
    if IS_CUPY_BACKEND:
        return xp.repeat(arr, counts)
    else:
        return np.repeat(arr, counts)


def _reduce_max_1d(vals: Array, starts: Array) -> Array:
    """Segmented maximum reduction for contiguous blocks."""
    return xp.maximum.reduceat(vals, starts)


def _reduce_sum_1d(vals: Array, starts: Array) -> Array:
    """Segmented sum reduction for contiguous blocks."""
    return xp.add.reduceat(vals, starts)


def get_n_smallest_per_block(values: Array, blocks: Array, n: int) -> Array:
    """Returns a boolean mask for the n smallest values in each block."""
    sort_idx = xp.lexsort(xp.stack([values, blocks]))
    block_sizes = xp.bincount(blocks)
    block_starts = xp.concatenate((xp.array([0]), xp.cumsum(block_sizes[:-1])))

    offsets = xp.zeros(len(blocks), dtype=xp.int64)
    diffs = xp.diff(block_starts, prepend=0)
    offsets[block_starts] = diffs
    local_indices = xp.cumsum(offsets)

    inv_sort_idx = xp.empty_like(sort_idx)
    inv_sort_idx[sort_idx] = xp.arange(len(sort_idx), dtype=sort_idx.dtype)
    sorted_local_indices = inv_sort_idx - local_indices
    return sorted_local_indices < n


# -----------------------------------------------------------------------------
# 3. PIPELINE STAGE 1: EVALUATION
# -----------------------------------------------------------------------------


def evaluate_cluster_errors(
    config: ClusteringConfig,
    X_batch: Array,
    F_batch: Array,
    mus: Array,
    f_mus: Array,
    g_mus: Optional[Array],
    aug_starts: Array,
    aug_sizes: Array,
    aug_ids: Array,
    F_max_global: Array,
    TOL: float,
) -> tuple[Array, Array, dict[str, Any]]:
    """
    Fits the appropriate surrogate and computes the error metric.
    Returns: (is_good_mask, E_cluster, surrogate_params_dict)
    """
    N, d = X_batch.shape
    fd = F_batch.shape[1]
    K_batch = len(aug_sizes)
    surrogate_params = {}

    dx = X_batch - mus[aug_ids]

    # --- 1. Compute Differences based on Mode ---
    if config.mode == SurrogateMode.NEAREST_NEIGHBOR:
        diff = F_batch - f_mus[aug_ids]

    elif config.mode in (
        SurrogateMode.TAYLOR_HESSIAN,
        SurrogateMode.TAYLOR_COV_CENTROID,
    ):
        assert g_mus is not None
        lin = xp.einsum("nmd,nd->nm", g_mus[aug_ids], dx)
        diff = F_batch - (f_mus[aug_ids] + lin)
        surrogate_params = {"g_mus": g_mus}

    elif config.mode == SurrogateMode.AFFINE_COV:
        # Adaptive Tikhonov fitting
        mean_F = _reduce_sum_1d(F_batch, aug_starts) / aug_sizes[:, None]
        Fc = F_batch - mean_F[aug_ids]

        auto_cov = _reduce_sum_1d(dx[:, :, None] * dx[:, None, :], aug_starts)
        cross_cov = _reduce_sum_1d(dx[:, :, None] * Fc[:, None, :], aug_starts)

        tr = xp.trace(auto_cov, axis1=1, axis2=2)
        lam = xp.maximum(1e-8 * tr, TINY_EPS)
        reg = lam[:, None, None] * xp.eye(d, dtype=xp.float64)[None, :, :]

        A_T = xp.linalg.solve(auto_cov + reg, cross_cov)
        A = A_T.transpose(0, 2, 1)
        b = mean_F - xp.einsum("kfd,kd->kf", A, mus)

        lin = xp.einsum("nfd,nd->nf", A[aug_ids], X_batch)
        diff = F_batch - (b[aug_ids] + lin)
        surrogate_params = {"A": A, "b": b}

    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    # --- 2. Evaluate Error Metrics ---
    if config.error_metric == "L2":
        err_pts = xp.sum(diff**2, axis=1)
        E_cluster = xp.sqrt(_reduce_sum_1d(err_pts, aug_starts))
    elif config.error_metric == "Linf-global":
        err_pts = xp.max(xp.abs(diff) / (F_max_global[None, :] + TINY_EPS), axis=1)
        E_cluster = _reduce_max_1d(err_pts, aug_starts)
    elif config.error_metric == "Linf-local":
        E_cluster_rel = xp.zeros(K_batch, dtype=xp.float64)
        E_cluster_abs = xp.zeros(K_batch, dtype=xp.float64)
        for j in range(fd):
            abs_diff_j = xp.abs(diff[:, j])
            local_max_F_j = _reduce_max_1d(xp.abs(F_batch[:, j]), aug_starts)
            max_err_j = _reduce_max_1d(abs_diff_j, aug_starts)

            E_cluster_abs = xp.maximum(E_cluster_abs, max_err_j)
            E_cluster_rel = xp.maximum(
                E_cluster_rel, max_err_j / xp.maximum(local_max_F_j, TINY_EPS)
            )

        is_good = (E_cluster_rel < TOL) | (E_cluster_abs < TOL)
        return is_good, E_cluster_rel, surrogate_params

    is_good = E_cluster < TOL
    return is_good, E_cluster, surrogate_params


# -----------------------------------------------------------------------------
# 4. PIPELINE STAGE 2: ACCEPTED STATS (Q, H, radii)
# -----------------------------------------------------------------------------


def compute_accepted_stats(
    config: ClusteringConfig,
    X_good: Array,
    G_good: Optional[Array],
    mus_good: Array,
    g_mus_good: Optional[Array],
    dense_labels_good: Array,
    counts_good: Array,
    Q_SHRINK: float,
    surrogate_params: dict,
) -> tuple[Array, Optional[Array], Array, Optional[Array]]:
    """
    Computes Q (Trust Region shape), H (Hessian), radii, and L_obs for accepted clusters.
    """
    K_good = len(counts_good)
    N_good, d = X_good.shape
    good_starts = xp.r_[0, xp.cumsum(counts_good)[:-1]]

    dx = X_good - mus_good[dense_labels_good]

    # --- 1. Compute Q (Inverse Covariance) ---
    Cov = _reduce_sum_1d(dx[:, :, None] * dx[:, None, :], good_starts)
    Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

    if Q_SHRINK > 0.0:
        tr = xp.trace(Cov, axis1=1, axis2=2)
        scale_tr = tr / max(d, 1)
        I = xp.eye(d, dtype=xp.float64)[None, :, :]
        Cov = (1.0 - Q_SHRINK) * Cov + Q_SHRINK * (scale_tr[:, None, None] * I)
        Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

    Lambda, U = xp.linalg.eigh(Cov)
    Lambda_inv = 1.0 / xp.maximum(Lambda, TINY_EPS)

    proj = xp.einsum("ni,nij->nj", dx, U[dense_labels_good])
    dist_sq_mahalanobis = xp.sum(proj**2 * Lambda_inv[dense_labels_good], axis=1)
    max_dist = _reduce_max_1d(dist_sq_mahalanobis, good_starts)

    scale_fac = 1.0 / xp.maximum(max_dist, TINY_EPS)
    Qs = U * (scale_fac[:, None] * Lambda_inv)[:, None, :] @ U.transpose(0, 2, 1)

    # --- 2. Compute Radii & L_obs ---
    dist_euclidean = xp.sqrt(xp.sum(dx**2, axis=1))
    radii = _reduce_max_1d(dist_euclidean, good_starts)

    L_obs = None
    if (
        config.requires_grad_everywhere
        and G_good is not None
        and g_mus_good is not None
    ):
        dG = G_good - g_mus_good[dense_labels_good]
        dg_norms = xp.sqrt(xp.sum(dG**2, axis=2))
        safe_dist = xp.maximum(dist_euclidean, TINY_EPS)[:, None]
        ratio_pts = xp.max(dg_norms / safe_dist, axis=1)
        ratio_pts = xp.where(
            dist_euclidean > TINY_EPS, ratio_pts, xp.zeros_like(ratio_pts)
        )
        L_obs = _reduce_max_1d(ratio_pts, good_starts)

    # --- 3. Compute Hessian (If requested) ---
    Hs = None
    if config.store_hess:
        if config.mode == SurrogateMode.TAYLOR_HESSIAN and G_good and g_mus_good:
            g_mus_exp = g_mus_good[dense_labels_good]
            dG = G_good - g_mus_exp
            E_mat = xp.einsum("ni,nj->nij", dx, xp.sum(dG, axis=1))
            E_k = _reduce_sum_1d(E_mat, good_starts)
            S_h = E_k + E_k.transpose(0, 2, 1)

            dvals, U_h = xp.linalg.eigh(Cov)
            dvals = xp.maximum(dvals, TINY_EPS)
            Gtilde = U_h.transpose(0, 2, 1) @ S_h @ U_h
            F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])
            Hs = U_h @ F_mat @ U_h.transpose(0, 2, 1)
            Hs = 0.5 * (Hs + Hs.transpose(0, 2, 1))

        elif config.mode == SurrogateMode.AFFINE_COV:
            Hs = xp.linalg.inv(Cov + TINY_EPS * xp.eye(d)[None, :, :])

    return Qs, Hs, radii, L_obs


# -----------------------------------------------------------------------------
# 5. PIPELINE STAGE 3: SPLIT REJECTED
# -----------------------------------------------------------------------------


def split_rejected_clusters(
    config: ClusteringConfig,
    X_bad: Array,
    F_bad: Array,
    G_bad: Optional[Array],
    mus_bad: Array,
    f_mus_bad: Array,
    dense_labels_bad: Array,
    counts_bad: Array,
    MIN_PTS_PER_CLUSTER: int,
    surrogate_params: dict,
) -> Array:
    """
    Computes the splitting direction and returns the sort_order for the next level.
    """
    K_bad = len(counts_bad)
    N_bad, d = X_bad.shape
    bad_starts = xp.r_[0, xp.cumsum(counts_bad)[:-1]]
    dx = X_bad - mus_bad[dense_labels_bad]

    # --- Mode 2: Hessian LSQ Split ---
    if config.mode == SurrogateMode.TAYLOR_HESSIAN and G_bad is not None:
        g_mus_exp = surrogate_params["g_mus"][dense_labels_bad]
        f_mus_exp = f_mus_bad[dense_labels_bad]

        lin = xp.einsum("nmd,nd->nm", g_mus_exp, dx)
        diff_bad = F_bad - (f_mus_exp + lin)

        sq_err = _reduce_sum_1d(diff_bad**2, bad_starts)
        sq_val = _reduce_sum_1d(F_bad**2, bad_starts)
        scores = xp.sqrt(sq_err) / (xp.sqrt(sq_val) + TINY_EPS)
        active_comps = xp.argmax(scores, axis=1)  # (K,)

        active_j = active_comps[dense_labels_bad]
        n_idx = xp.arange(N_bad)
        dG_active = G_bad[n_idx, active_j, :] - g_mus_exp[n_idx, active_j, :]

        C = _reduce_sum_1d(dx[:, :, None] * dx[:, None, :], bad_starts)
        E = _reduce_sum_1d(dx[:, :, None] * dG_active[:, None, :], bad_starts)

        C = 0.5 * (C + C.transpose(0, 2, 1))
        S = E + E.transpose(0, 2, 1)

        dvals, U = xp.linalg.eigh(C)
        dvals = xp.maximum(dvals, TINY_EPS)
        Gtilde = U.transpose(0, 2, 1) @ S @ U
        F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])
        _, V = xp.linalg.eigh(F_mat)

        Z = U @ V
        M_split = (Z * xp.sqrt(xp.abs(_))[:, None, :]) @ Z.transpose(0, 2, 1)
        M_split = 0.5 * (M_split + M_split.transpose(0, 2, 1))

        CovZ = M_split @ C @ M_split.transpose(0, 2, 1)

    # --- Mode 1 & 3: Covariance Level Curves Split ---
    elif config.mode in (SurrogateMode.AFFINE_COV, SurrogateMode.TAYLOR_COV_CENTROID):
        if config.mode == SurrogateMode.TAYLOR_COV_CENTROID:
            g_mus_exp = surrogate_params["g_mus"][dense_labels_bad]
            lin_bad = xp.einsum("nmd,nd->nm", g_mus_exp, dx)
            diff_bad = F_bad - (f_mus_bad[dense_labels_bad] + lin_bad)
            E_bad = xp.linalg.norm(diff_bad, axis=1)
        else:  # AFFINE_COV - RESTORE TWO-PASS LOCAL FIT
            dist_to_mu = xp.linalg.norm(dx, axis=1)
            n_tangent = max(d * 2, MIN_PTS_PER_CLUSTER // 4)
            tangent_mask = get_n_smallest_per_block(
                dist_to_mu, dense_labels_bad, n_tangent
            )

            X_core = X_bad[tangent_mask]
            F_core = F_bad[tangent_mask]
            labels_core = dense_labels_bad[tangent_mask]

            core_counts = xp.bincount(labels_core, minlength=K_bad)
            core_starts = xp.r_[0, xp.cumsum(core_counts)[:-1]]

            dX_core = X_core - mus_bad[labels_core]
            dF_core = F_core - f_mus_bad[labels_core]

            M_XX = _reduce_sum_1d(
                dX_core[:, :, None] * dX_core[:, None, :], core_starts
            )
            M_XF = _reduce_sum_1d(
                dX_core[:, :, None] * dF_core[:, None, :], core_starts
            )

            max_eigvals_proxy = xp.trace(M_XX, axis1=1, axis2=2)
            _lam = xp.maximum(1e-8 * max_eigvals_proxy, TINY_EPS)
            reg = _lam[:, None, None] * xp.eye(d, dtype=xp.float64)[None, :, :]

            A_T_tangent = xp.linalg.solve(M_XX + reg, M_XF)
            A_tangent = A_T_tangent.transpose(0, 2, 1)

            lin_bad = xp.einsum("nmd,nd->nm", A_tangent[dense_labels_bad], dx)
            diff_bad = F_bad - (f_mus_bad[dense_labels_bad] + lin_bad)
            E_bad = xp.linalg.norm(diff_bad, axis=1)

        # CRITICAL FIX: Find SMALLEST errors (where surrogate == true func)
        n_est_cov = max(1, MIN_PTS_PER_CLUSTER // 3)
        cov_est_mask = get_n_smallest_per_block(E_bad, dense_labels_bad, n_est_cov)

        X_hard = X_bad[cov_est_mask]
        dx_hard = X_hard - mus_bad[dense_labels_bad[cov_est_mask]]
        hard_counts = xp.bincount(dense_labels_bad[cov_est_mask], minlength=K_bad)
        hard_starts = xp.r_[0, xp.cumsum(hard_counts)[:-1]]

        H_inv = _reduce_sum_1d(dx_hard[:, :, None] * dx_hard[:, None, :], hard_starts)
        H_inv = 0.5 * (H_inv + H_inv.transpose(0, 2, 1))

        dvals, U = xp.linalg.eigh(H_inv)
        dvals_sqrt_inv = 1.0 / xp.sqrt(xp.maximum(xp.abs(dvals), TINY_EPS))
        M_split = (U * dvals_sqrt_inv[:, None]) @ U.transpose(0, 2, 1)
        M_split = 0.5 * (M_split + M_split.transpose(0, 2, 1))

        C = _reduce_sum_1d(dx[:, :, None] * dx[:, None, :], bad_starts)
        CovZ = M_split @ C @ M_split.transpose(0, 2, 1)

    # --- Mode 4: Nearest Neighbor / PCA Split ---
    elif config.mode == SurrogateMode.NEAREST_NEIGHBOR:
        C = _reduce_sum_1d(dx[:, :, None] * dx[:, None, :], bad_starts)
        CovZ = 0.5 * (C + C.transpose(0, 2, 1))
        M_split = xp.eye(d, dtype=xp.float64)[None, :, :].repeat(K_bad, axis=0)

    else:
        raise ValueError(f"Unknown split mode: {config.mode}")

    # --- Apply Split Projection ---
    _, V_z = xp.linalg.eigh(CovZ)
    v = V_z[:, :, -1]  # Highest variance eigenvector

    z = xp.einsum("nij,nj->ni", M_split[dense_labels_bad], dx)
    scores_proj = xp.sum(z * v[dense_labels_bad], axis=1)

    sort_order = xp.lexsort(xp.stack([scores_proj, dense_labels_bad]))
    return sort_order


# -----------------------------------------------------------------------------
# 6. MISC UTILS
# -----------------------------------------------------------------------------


class CudaTimer:
    """Helper context manager for timing GPU operations with synchronization."""

    def __init__(self, key: str, stats: dict, enabled: bool):
        self.key = key
        self.stats = stats
        self.enabled = enabled
        self.t0 = None

    def __enter__(self):
        if self.enabled:
            if IS_CUPY_BACKEND:
                xp.cuda.Device().synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.enabled:
            if IS_CUPY_BACKEND:
                xp.cuda.Device().synchronize()
            dt = time.perf_counter() - self.t0  # type: ignore
            self.stats[self.key] += dt


def convert_to_2d_array(F: Array) -> Array:
    F = xp.asarray(F)
    if F.ndim == 1:
        return F[:, None]
    return F


def convert_to_3d_array(G: Array) -> Array:
    G = xp.asarray(G)
    if G.ndim == 2:
        return G[:, None, :]
    return G


def allocate_output_buffers_host(
    MAX_CLUSTERS_EST: int,
    d: int,
    fd: int,
    store_hess: bool,
    store_affine_fit: bool,
    no_gradient: bool,
) -> tuple[dict, int]:
    cap = max(1, int(MAX_CLUSTERS_EST))
    buffers = {
        "centroids": np.empty((cap, d), dtype=np.float64),
        "f": np.empty((cap, fd), dtype=np.float64),
        "Q": np.empty((cap, d, d), dtype=np.float64),
        "E_cluster": np.empty((cap,), dtype=np.float64),
        "radii": np.empty((cap,), dtype=np.float64),
        "counts": np.empty((cap,), dtype=np.int64),
    }
    if not no_gradient:
        buffers["g"] = np.empty((cap, fd, d), dtype=np.float64)
        buffers["L_obs"] = np.empty((cap,), dtype=np.float64)
    if store_hess:
        buffers["H"] = np.empty((cap, d, d), dtype=np.float64)
    if store_affine_fit:
        buffers["A_fit"] = np.empty((cap, fd, d), dtype=np.float64)
        buffers["b_fit"] = np.empty((cap, fd), dtype=np.float64)
    return buffers, cap


def adjust_buffer_capacity_host(
    buffers: dict,
    current_count: int,
    current_cap: int,
    required_total: int,
    store_hess: bool,
    store_affine_fit: bool,
    no_gradient: bool,
) -> tuple[dict, int]:
    if required_total <= current_cap:
        return buffers, current_cap
    new_cap = max(required_total, max(1, current_cap) * 2)
    d = int(buffers["centroids"].shape[1])
    fd = int(buffers["f"].shape[1])
    new_buffers = {
        "centroids": np.empty((new_cap, d), dtype=np.float64),
        "f": np.empty((new_cap, fd), dtype=np.float64),
        "Q": np.empty((new_cap, d, d), dtype=np.float64),
        "E_cluster": np.empty((new_cap,), dtype=np.float64),
        "radii": np.empty((new_cap,), dtype=np.float64),
        "counts": np.empty((new_cap,), dtype=np.int64),
    }
    if not no_gradient:
        new_buffers["g"] = np.empty((new_cap, fd, d), dtype=np.float64)
        new_buffers["L_obs"] = np.empty((new_cap,), dtype=np.float64)
    if store_hess:
        new_buffers["H"] = np.empty((new_cap, d, d), dtype=np.float64)
    if store_affine_fit:
        new_buffers["A_fit"] = np.empty((new_cap, fd, d), dtype=np.float64)
        new_buffers["b_fit"] = np.empty((new_cap, fd), dtype=np.float64)

    if current_count > 0:
        for k in new_buffers:
            new_buffers[k][:current_count] = buffers[k][:current_count]
    del buffers
    return new_buffers, int(new_cap)


def get_prime_factors(n: int) -> list[int]:
    if n < 2:
        return []
    factors: list[int] = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 2
    if n > 1:
        factors.append(n)
    return factors


def uniform_grid_cluster_labels(
    X_host: np.ndarray, num_initial_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    if num_initial_clusters < 1:
        raise ValueError("num_initial_clusters must be >= 1")
    _, d = X_host.shape
    lower, upper = np.min(X_host, axis=0), np.max(X_host, axis=0)
    extents = np.maximum(upper - lower, TINY_EPS)
    bins = np.ones(d, dtype=np.int64)
    for p in sorted(get_prime_factors(num_initial_clusters), reverse=True):
        split_scores = extents / bins
        split_dim = int(np.argmax(split_scores))
        bins[split_dim] *= p
    rel = (X_host - lower[None, :]) / extents
    rel = np.clip(rel, 0.0, 1.0 - np.finfo(np.float64).eps)
    coords = np.floor(rel * bins[None, :]).astype(np.int64)
    coords = np.minimum(coords, bins[None, :] - 1)
    labels = np.ravel_multi_index(coords.T, dims=tuple(int(b) for b in bins))
    return labels.astype(np.int64), bins


def log_mem_stats(prefix: str):
    if IS_CUPY_BACKEND:
        import cupy as cp

        mempool = cp.get_default_memory_pool()
        used = mempool.used_bytes() / (1024**3)
        total = mempool.total_bytes() / (1024**3)
        logger.info(f"{prefix} | Mem: {used:.2f}GB / {total:.2f}GB")
