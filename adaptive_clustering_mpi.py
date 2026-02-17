"""
MPI-aware Adaptive Clustering.

New features vs adaptive_clustering_mpi_exp.py:
  1. MPI parallelism: Rank 0 inits/clusters/scatters; all ranks refine locally;
     Rank 0 gathers/merges.
  2. Dynamic pool resizing: no pre-committed SYNTH_BUDGET cap; pool grows
     on demand via ensure_pool_capacity.

Non-rank-0 ranks return None from adaptive_clustering.
"""

import logging
import os
import time
from typing import Any, Callable, Literal, Optional

import numpy as np
from mpi4py import MPI
from sklearn.cluster import KMeans

from src import IS_CUPY_BACKEND, Array, xp
from src.logging_utils import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)

TINY_EPS = 1e-12


def _to_host(arr: Array) -> Array:
    """Move array to host (NumPy) if it's on device (CuPy)."""
    if hasattr(arr, "get"):  # CuPy array
        return xp.asnumpy(arr)
    return arr


def _to_device(arr: np.ndarray) -> Array:
    """Move array to device (CuPy) if available, else keep as NumPy."""
    return xp.asarray(arr)


def scatter_maximum_1d(out: Array, idx: Array, vals: Array) -> None:
    """
    1D segmented max scatter: ``out[idx[i]] = max(out[idx[i]], vals[i])``.
    Uses `cupyx.scatter_max` on GPU and `numpy.maximum.at` semantics on CPU.
    Eg:
        out = [0, 0, 0]
        idx = [0, 1, 0, 2, 1]
        vals = [1, 2, 3, 4, 0]
        scatter_maximum_1d(out, idx, vals)
        # out is now [3, 2, 4]

    Args:
        out: (K,) output array updated in-place.
        idx: (N,) integer indices in [0, K).
        vals: (N,) values to reduce via max per index.
    """
    if IS_CUPY_BACKEND:
        from cupyx import scatter_max as cupy_scatter_max  # type: ignore[import]

        cupy_scatter_max(out, idx, vals)
    else:
        xp.maximum.at(out, idx, vals)


def compute_label_segments(sorted_labels: Array) -> tuple[Array, Array, Array]:
    """Compute segmentation metadata for a label-sorted 1D label array.

    Args:
        sorted_labels: (N,) labels sorted in nondecreasing order.

    Returns:
        uniq: (K,) unique label values (ascending).
        starts: (K,) start indices of each label-run in `sorted_labels`.
        counts: (K,) run lengths.
    """
    uniq, starts, counts = xp.unique(
        sorted_labels, return_index=True, return_counts=True
    )
    return uniq, starts, counts


def ensure_pool_capacity(
    pool: dict[str, Array],
    n_total: int,
    needed: int,
    capacity: int,
) -> tuple[int, dict[str, Array]]:
    """Grow pool dict if n_total + needed > capacity. Returns (new_cap, new_pool)."""
    required = n_total + needed
    if required <= capacity:
        return capacity, pool
    new_cap = max(required, capacity * 2)
    logger.info(f"Pool resize: {capacity:,} -> {new_cap:,} (need {required:,})")
    new_pool: dict[str, Array] = {}
    for name, arr in pool.items():
        shape = list(arr.shape)
        shape[0] = new_cap
        new_pool[name] = xp.empty(shape, dtype=arr.dtype)
        new_pool[name][:n_total] = arr[:n_total]
    del pool
    if IS_CUPY_BACKEND:
        xp.get_default_memory_pool().free_all_blocks()
    return new_cap, new_pool


def generate_sythetic_points(
    X_sorted: Array,
    starts: Array,
    counts: Array,
    uniq_labels: Array,
    MIN_PTS_PER_CLUSTER: int,
    *,
    SEED: int,
    level: int,
) -> tuple[Optional[Array], Optional[Array]]:
    """
    Batch generate synthetic point generation via convex combinations within clusters.

    For each cluster whose active count is below `MIN_PTS_PER_CLUSTER`, generate the
    required number of synthetic points via convex combinations of *two distinct*
    points from the same cluster:

        x_synth = a * x_i + (1 - a) * x_j,    a ~ Uniform(0, 1)

    This stabilizes covariance / Hessian-estimation steps in later geometry, while the
    pool design makes synthetic points persist across levels.

    Args:
        X_sorted: (N_active, d) active points sorted by label.
        starts: (K,) cluster segment starts within `X_sorted`.
        counts: (K,) active counts per cluster.
        uniq_labels: (K,) label values for each segment.
        MIN_PTS_PER_CLUSTER: minimum desired points per cluster.
        SEED: base RNG seed.
        level: current refinement level; used to derive deterministic RNG stream.

    Returns:
        X_synth: (M, d) synthetic points, or None if no padding needed.
        labels_synth: (M,) *global* label values associated to each synthetic point,
            or None if no padding needed.
    """
    # If any cluster has <3 points, convex-combo padding is ill-defined (degenerate).
    if xp.any(counts < 3):
        bad = uniq_labels[xp.nonzero(counts < 3)[0][:10]]
        raise RuntimeError(
            f"Level {level}: cluster(s) with <3 points detected. "
            f"Example labels: {bad.tolist()} (showing up to 10)."
        )

    # How many synthetic points we need per cluster.
    need = xp.maximum(0, MIN_PTS_PER_CLUSTER - counts)  # (K,)
    total_synth = int(xp.sum(need))
    if total_synth == 0:
        return None, None

    rng = xp.random.default_rng(SEED + level)
    rand = rng.random
    k_synth = repeats(xp.arange(int(need.shape[0])), need)
    labels_synth = uniq_labels[k_synth]
    m = counts[k_synth]
    s0 = starts[k_synth]
    i = (xp.floor(rand(total_synth) * m)).astype(xp.int64)
    j = (xp.floor(rand(total_synth) * (m - 1))).astype(xp.int64)
    j = j + (j >= i)
    idx_i = s0 + i
    idx_j = s0 + j
    a = rand((total_synth, 1)).astype(X_sorted.dtype)
    X_synth = a * X_sorted[idx_i] + (1.0 - a) * X_sorted[idx_j]
    return X_synth, labels_synth.astype(xp.int64)


def calculate_scalar_metrics(
    DeltaX: Array,
    G_points: Array,
    G_centroids: Array,
    map_n_to_k: Array,
    K: int,
) -> tuple[Array, Array]:
    dist_x = xp.linalg.norm(DeltaX, axis=1)
    radii = xp.zeros((K,), dtype=DeltaX.dtype)
    scatter_maximum_1d(radii, map_n_to_k, dist_x)
    g_center_exp = G_centroids[map_n_to_k]
    diff_g_norms = xp.linalg.norm(G_points - g_center_exp, axis=2)
    valid_mask = dist_x > TINY_EPS
    ratios = xp.zeros_like(diff_g_norms)
    if xp.any(valid_mask):
        ratios[valid_mask] = diff_g_norms[valid_mask] / dist_x[valid_mask][:, None]
    max_ratio_per_point = xp.max(ratios, axis=1)
    L_obs = xp.zeros((K,), dtype=DeltaX.dtype)
    scatter_maximum_1d(L_obs, map_n_to_k, max_ratio_per_point)
    return radii, L_obs


def calculate_Q(
    DeltaX: Array,
    starts: Array,
    counts: Array,
    K: int,
    *,
    Q_SHRINK: float = 0.0,
) -> tuple[Array, Array]:
    XX_all = DeltaX[:, :, None] * DeltaX[:, None, :]
    Cov = xp.add.reduceat(XX_all, starts)
    Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

    if Q_SHRINK > 0.0:
        d = int(Cov.shape[1])
        tr = xp.trace(Cov, axis1=1, axis2=2)
        scale = tr / max(d, 1)
        I = xp.eye(d, dtype=Cov.dtype)[None, :, :]
        Cov = (1.0 - Q_SHRINK) * Cov + Q_SHRINK * (scale[:, None, None] * I)
        Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))
        Lambda, U = xp.linalg.eigh(Cov)
        Lambda_inv = 1.0 / xp.maximum(Lambda, TINY_EPS)
        map_n_to_k = repeats(xp.arange(K), counts)
        U_exp = U[map_n_to_k]
        L_inv_exp = Lambda_inv[map_n_to_k]
        proj = (U_exp.transpose(0, 2, 1) @ DeltaX[:, :, None])[:, :, 0]
        dist_sq = xp.sum((proj * proj) * L_inv_exp, axis=1)
        max_dist_sq = xp.zeros((K,), dtype=proj.dtype)
        scatter_maximum_1d(max_dist_sq, map_n_to_k, dist_sq)
        max_dist_sq = xp.maximum(max_dist_sq, TINY_EPS)
        final_D = (1.0 / max_dist_sq)[:, None] * Lambda_inv
        Q = (U * final_D[:, None, :]) @ U.transpose(0, 2, 1)
    else:
        _, U = xp.linalg.eigh(Cov)
        map_n_to_k = repeats(xp.arange(K), counts)
        U_exp = U[map_n_to_k]
        proj = (U_exp.transpose(0, 2, 1) @ DeltaX[:, :, None])[:, :, 0]
        proj_abs = xp.abs(proj)
        d_axes = xp.zeros((K, proj_abs.shape[1]), dtype=proj_abs.dtype)
        scatter_maximum_1d(d_axes, map_n_to_k, proj_abs)
        d_axes = xp.maximum(d_axes, TINY_EPS)
        Dinv2 = 1.0 / (d_axes * d_axes)
        Q = (U * Dinv2[:, None, :]) @ U.transpose(0, 2, 1)

    return 0.5 * (Q + Q.transpose(0, 2, 1)), Cov


def compute_good_covariance_hessian(
    DeltaX: Array,
    Fvals_linear: Array,
    Fvals_exact: Array,
    starts: Array,
) -> Array:
    points_in_level_curve = xp.full(len(DeltaX), False)
    Ndx, d = DeltaX.shape
    errors = Fvals_linear - Fvals_exact
    for j in range(len(starts)):
        eps = 1e-3
        errors_current = errors[starts[j] : starts[j + 1]]
        within_tol = errors_current < eps
        if len(within_tol) <= 2 * d * xp.log(d) + 1:
            within_tol = errors_current > eps
        else:
            while xp.sum(within_tol) < xp.floor(2 * d * xp.log(d)):
                eps *= 2
                within_tol = errors_current < eps
        points_in_level_curve[starts[j] : starts[j + 1]] = within_tol.flatten()
    DeltaX_within_tol = DeltaX[points_in_level_curve]
    XX_all = DeltaX_within_tol[:, :, None] * DeltaX_within_tol[:, None, :]
    C = xp.add.reduceat(XX_all, starts)
    dvals, U = xp.linalg.eigh(C)
    abs_dvals = np.abs(_to_host(dvals))
    inv_abs_sqrt_dvals = _to_device(abs_dvals ** (-0.5))
    M = (U * inv_abs_sqrt_dvals[:, None, :]) @ U.transpose(0, 2, 1)
    return 0.5 * (M + M.transpose(0, 2, 1))


def estimate_hessian_sqrt_symmetric(
    DeltaX: Array,
    DeltaG: Array,
    starts: Array,
) -> Array:
    XX_all = DeltaX[:, :, None] * DeltaX[:, None, :]
    XG_all = DeltaX[:, :, None] * DeltaG[:, None, :]
    C = xp.add.reduceat(XX_all, starts)
    E = xp.add.reduceat(XG_all, starts)
    C = 0.5 * (C + C.transpose(0, 2, 1))
    S = E + E.transpose(0, 2, 1)
    dvals, U = xp.linalg.eigh(C)
    dvals = xp.maximum(dvals, TINY_EPS)
    Gtilde = U.transpose(0, 2, 1) @ S @ U
    denom = dvals[:, :, None] + dvals[:, None, :]
    F_mat = Gtilde / denom
    h_eigvals, V = xp.linalg.eigh(F_mat)
    Z = U @ V
    w_abs_sqrt = xp.sqrt(xp.abs(h_eigvals))
    M = (Z * w_abs_sqrt[:, None, :]) @ Z.transpose(0, 2, 1)
    return 0.5 * (M + M.transpose(0, 2, 1))


def get_dominant_eigenvector(Cov: Array) -> Array:
    w, V = xp.linalg.eigh(Cov)
    return V[:, :, -1]


def compute_median_split_decision(
    scores: Array,
    cluster_id: Array,
    starts: Array,
    counts: Array,
) -> Array:
    N = int(scores.shape[0])
    if N == 0:
        return scores >= 0
    idx_by_score = xp.argsort(scores)
    cid_by_score = cluster_id[idx_by_score]
    idx_group = xp.argsort(cid_by_score)
    order = idx_by_score[idx_group]
    scores_sorted = scores[order]
    i0 = starts + (counts - 1) // 2
    i1 = starts + counts // 2
    median = 0.5 * (scores_sorted[i0] + scores_sorted[i1])
    return scores >= median[cluster_id]


# ---------------------------------------------------------------------------
# Output buffer helpers (unchanged from original)
# ---------------------------------------------------------------------------


def allocate_output_buffers(
    MAX_CLUSTERS_EST: int,
    d: int,
    fd: int,
    dtype: Any,
    STORE_HESSIAN_SQRT_PER_CLUSTER: bool,
    STORE_TREE_DATA: bool,
) -> tuple[dict[str, Array], int]:
    cap = max(1, int(MAX_CLUSTERS_EST))
    buffers: dict[str, Any] = {
        "centroids": xp.empty((cap, d), dtype=dtype),
        "f": xp.empty((cap, fd), dtype=dtype),
        "g": xp.empty((cap, fd, d), dtype=dtype),
        "radii": xp.empty((cap,), dtype=dtype),
        "L_obs": xp.empty((cap,), dtype=dtype),
        "E_cluster": xp.empty((cap,), dtype=dtype),
        "counts": xp.empty((cap,), dtype=xp.int64),
        "Q": xp.empty((cap, d, d), dtype=dtype),
    }
    if STORE_HESSIAN_SQRT_PER_CLUSTER:
        buffers["H_sqrt"] = xp.empty((cap, d, d), dtype=dtype)
    if STORE_TREE_DATA:
        buffers["tree_data"] = {}
    return buffers, cap


def adjust_buffer_capacity(
    buffers: dict[str, Any],
    current_count: int,
    current_cap: int,
    required_total: int,
    STORE_HESSIAN_SQRT_PER_CLUSTER: bool,
    STORE_TREE_DATA: bool,
) -> tuple[dict[str, Any], int]:
    if required_total <= current_cap:
        return buffers, current_cap
    d = int(buffers["centroids"].shape[1])
    fd = int(buffers["f"].shape[1])
    new_cap = max(required_total, max(1, current_cap) * 2)
    dtype = buffers["centroids"].dtype
    logger.warning(f"Output buffers resize: {current_cap:,} -> {new_cap:,}")
    new_buffers: dict[str, Any] = {
        "centroids": xp.empty((new_cap, d), dtype=dtype),
        "f": xp.empty((new_cap, fd), dtype=dtype),
        "g": xp.empty((new_cap, fd, d), dtype=dtype),
        "radii": xp.empty((new_cap,), dtype=dtype),
        "L_obs": xp.empty((new_cap,), dtype=dtype),
        "E_cluster": xp.empty((new_cap,), dtype=dtype),
        "counts": xp.empty((new_cap,), dtype=xp.int64),
        "Q": xp.empty((new_cap, d, d), dtype=dtype),
    }
    if current_count > 0:
        for k in new_buffers:
            new_buffers[k][:current_count] = buffers[k][:current_count]
    if STORE_HESSIAN_SQRT_PER_CLUSTER:
        new_buffers["H_sqrt"] = xp.empty((new_cap, d, d), dtype=dtype)
        if current_count > 0:
            new_buffers["H_sqrt"][:current_count] = buffers["H_sqrt"][:current_count]
    if STORE_TREE_DATA:
        new_buffers["tree_data"] = dict(buffers.get("tree_data", {}))
    del buffers
    return new_buffers, int(new_cap)


# ---------------------------------------------------------------------------
# Local refinement loop (shared by each MPI rank)
# ---------------------------------------------------------------------------
def _run_local_refinement(
    # Pool (dict of arrays, all of length >= n_total)
    pool: dict[str, Array],
    n_total: int,
    capacity: int,
    # Oracle
    f_fn,
    g_fn,
    oracle_fn,
    # Config
    TOL: float,
    ERROR_METRIC: str,
    MIN_PTS_PER_CLUSTER: int,
    MAX_CLUSTERS_EST: int,
    Q_SHRINK: float,
    SEED: int,
    LOG_MEM_STATS: bool,
    MAX_LEVELS: int,
    USE_HESSIAN_INFORMED_SPLIT: bool,
    USE_COVARIANCE_HESSIAN: bool,
    IGNORE_GRADIENT: bool,
    STORE_HESSIAN_SQRT_PER_CLUSTER: bool,
    STORE_TREE_DATA: bool,
    F_max_global: Array,  # frozen global scale (broadcast from rank 0)
    rank: int,
) -> tuple[dict[str, Any], int, int]:
    """Run the adaptive refinement loop on the local data subset."""
    d = int(pool["X"].shape[1])
    fd = int(pool["F"].shape[1])
    dtype = pool["X"].dtype
    total_f_evals = 0

    out_buffers, out_cap = allocate_output_buffers(
        MAX_CLUSTERS_EST,
        d,
        fd,
        dtype,
        STORE_HESSIAN_SQRT_PER_CLUSTER,
        STORE_TREE_DATA,
    )
    out_count = 0

    level = 1
    while level <= MAX_LEVELS:
        t_level = time.perf_counter()
        if LOG_MEM_STATS:
            log_mem_stats(f"R{rank} Start: L{level:02}")

        idx_active = xp.nonzero(pool["active"][:n_total])[0]
        n_active = int(idx_active.shape[0])
        if n_active == 0:
            logger.info(f"R{rank} L{level:03}: no active points, done.")
            break

        # Sort & segment
        label_active = pool["labels"][idx_active]
        perm = xp.argsort(label_active)
        idx_sorted = idx_active[perm]
        label_sorted = label_active[perm]
        uniq_labels, starts, counts = compute_label_segments(label_sorted)
        K_batch = int(uniq_labels.shape[0])

        # --- Dynamic-pool padding ---
        need = xp.maximum(0, MIN_PTS_PER_CLUSTER - counts)
        total_synth_needed = int(xp.sum(need))
        capacity, pool = ensure_pool_capacity(
            pool, n_total, total_synth_needed, capacity
        )

        X_sorted = pool["X"][idx_sorted]
        X_synth, labels_synth = generate_sythetic_points(
            X_sorted,
            starts,
            counts,
            uniq_labels,
            MIN_PTS_PER_CLUSTER,
            SEED=SEED + rank,
            level=level,
        )

        if X_synth is not None:
            M_synth = int(X_synth.shape[0])
            end_new = n_total + M_synth

            if oracle_fn is not None:
                F_new, G_new = oracle_fn(X_synth)
                total_f_evals += M_synth
            else:
                F_new = f_fn(X_synth)
                total_f_evals += M_synth
                G_new = (
                    xp.zeros((M_synth, fd, d), dtype=dtype)
                    if IGNORE_GRADIENT
                    else g_fn(X_synth)
                )

            pool["X"][n_total:end_new] = X_synth
            pool["F"][n_total:end_new] = F_new
            pool["G"][n_total:end_new] = G_new
            pool["labels"][n_total:end_new] = labels_synth
            pool["active"][n_total:end_new] = True
            n_total = end_new

            # Re-sort after padding
            idx_active = xp.nonzero(pool["active"][:n_total])[0]
            label_active = pool["labels"][idx_active]
            perm = xp.argsort(label_active)
            idx_sorted = idx_active[perm]
            label_sorted = label_active[perm]
            uniq_labels, starts, counts = compute_label_segments(label_sorted)
            K_batch = int(uniq_labels.shape[0])
            X_sorted = pool["X"][idx_sorted]

        if LOG_MEM_STATS:
            log_mem_stats(f"R{rank} After: L{level:02} Padding")

        # --- Error computation ---
        F_sorted = pool["F"][idx_sorted]
        G_sorted = pool["G"][idx_sorted]

        mus = xp.add.reduceat(X_sorted, starts) / counts[:, None]

        if oracle_fn is not None:
            f_mus, g_mus = oracle_fn(mus)
            total_f_evals += mus.shape[0]
        else:
            f_mus = f_fn(mus)
            total_f_evals += mus.shape[0]
            g_mus = (
                xp.zeros((K_batch, fd, d), dtype=dtype)
                if IGNORE_GRADIENT
                else g_fn(mus)
            )

        map_n_to_k = repeats(xp.arange(K_batch), counts)
        dx = X_sorted - mus[map_n_to_k]
        g_mus_exp = g_mus[map_n_to_k]
        linear_term = (g_mus_exp @ dx[:, :, None])[:, :, 0]
        f_lin = f_mus[map_n_to_k] + linear_term
        diff = F_sorted - f_lin
        abs_diff = xp.abs(diff)

        if ERROR_METRIC == "L2":
            sq_err = diff**2
            E_cluster = xp.sqrt(xp.sum(xp.add.reduceat(sq_err, starts), axis=1))
            is_good = E_cluster < TOL

        elif ERROR_METRIC == "Linf-global":
            E_rel_max = xp.zeros((K_batch,), dtype=diff.dtype)
            for j in range(fd):
                max_diff_j = xp.zeros((K_batch,), dtype=diff.dtype)
                scatter_maximum_1d(max_diff_j, map_n_to_k, abs_diff[:, j])
                E_rel_max = xp.maximum(
                    E_rel_max, max_diff_j / (F_max_global[j] + TINY_EPS)
                )
            E_cluster = E_rel_max
            is_good = E_cluster < TOL

        elif ERROR_METRIC == "Linf-local":
            E_rel_max = xp.zeros((K_batch,), dtype=diff.dtype)
            E_abs_max = xp.zeros((K_batch,), dtype=diff.dtype)
            abs_F = xp.abs(F_sorted)
            for j in range(fd):
                max_diff_j = xp.zeros((K_batch,), dtype=diff.dtype)
                scatter_maximum_1d(max_diff_j, map_n_to_k, abs_diff[:, j])
                E_abs_max = xp.maximum(E_abs_max, max_diff_j)
                local_max_F = xp.zeros((K_batch,), dtype=diff.dtype)
                scatter_maximum_1d(local_max_F, map_n_to_k, abs_F[:, j])
                E_rel_max = xp.maximum(E_rel_max, max_diff_j / (local_max_F + TINY_EPS))
            E_cluster = E_rel_max
            is_good = (E_cluster < TOL) | (E_abs_max < TOL)
        else:
            raise ValueError(f"Unknown ERROR_METRIC: {ERROR_METRIC}")

        if LOG_MEM_STATS:
            log_mem_stats(f"R{rank} After: L{level:02} Error")

        # --- Finalize good clusters ---
        good_idx = xp.nonzero(is_good)[0]
        n_good = int(good_idx.shape[0])

        if n_good > 0:
            all_radii, all_L_obs = calculate_scalar_metrics(
                dx, G_sorted, g_mus, map_n_to_k, K_batch
            )
            mask_good_pts = is_good[map_n_to_k]
            dx_good = dx[mask_good_pts]
            good_counts = counts[good_idx]
            good_starts = xp.r_[0, xp.cumsum(good_counts)[:-1]]

            Qs, _ = calculate_Q(
                dx_good, good_starts, good_counts, n_good, Q_SHRINK=Q_SHRINK
            )

            end_out = out_count + n_good
            out_buffers, out_cap = adjust_buffer_capacity(
                out_buffers,
                out_count,
                out_cap,
                end_out,
                STORE_HESSIAN_SQRT_PER_CLUSTER,
                STORE_TREE_DATA,
            )

            if STORE_HESSIAN_SQRT_PER_CLUSTER:
                diff_good = diff[mask_good_pts]
                F_good = F_sorted[mask_good_pts]
                sq_err_g = xp.add.reduceat(diff_good**2, good_starts)
                sq_norm_g = xp.add.reduceat(F_good**2, good_starts)
                rel_l2 = xp.sqrt(sq_err_g) / (xp.sqrt(sq_norm_g) + TINY_EPS)
                active_comps = xp.argmax(rel_l2, axis=1)
                G_good_all = G_sorted[mask_good_pts]
                g_mu_good_all = g_mus[good_idx]
                map_good_n_to_k = repeats(xp.arange(n_good), good_counts)
                pt_active = active_comps[map_good_n_to_k]
                idx_n = xp.arange(int(diff_good.shape[0]))
                G_sel = G_good_all[idx_n, pt_active, :]
                g_mu_sel = g_mu_good_all[map_good_n_to_k, pt_active, :]
                H_sqrt = estimate_hessian_sqrt_symmetric(
                    dx_good, G_sel - g_mu_sel, good_starts
                )
                out_buffers["H_sqrt"][out_count:end_out] = H_sqrt

            out_buffers["centroids"][out_count:end_out] = mus[good_idx]
            out_buffers["f"][out_count:end_out] = f_mus[good_idx]
            out_buffers["g"][out_count:end_out] = g_mus[good_idx]
            out_buffers["radii"][out_count:end_out] = all_radii[good_idx]
            out_buffers["L_obs"][out_count:end_out] = all_L_obs[good_idx]
            out_buffers["Q"][out_count:end_out] = Qs
            out_buffers["E_cluster"][out_count:end_out] = E_cluster[good_idx]
            out_buffers["counts"][out_count:end_out] = counts[good_idx]
            out_count = end_out

            if STORE_TREE_DATA:
                points_in_clusters = [X_sorted[map_n_to_k == gi] for gi in good_idx]
                out_buffers["tree_data"][level] = {
                    "centroids": mus[good_idx],
                    "f": f_mus[good_idx],
                    "g": g_mus[good_idx],
                    "radii": all_radii[good_idx],
                    "L_obs": all_L_obs[good_idx],
                    "E_cluster": E_cluster[good_idx],
                    "counts": counts[good_idx],
                    "Q": Qs,
                    "points_in_clusters": points_in_clusters,
                }

            pool["active"][idx_sorted[mask_good_pts]] = False

        # --- Split bad clusters ---
        bad_idx = xp.nonzero(~is_good)[0]
        n_bad_clusters = int(bad_idx.shape[0])

        if n_bad_clusters == 0:
            logger.info(f"R{rank} L{level:03}: no bad clusters, done.")
            break

        if LOG_MEM_STATS:
            log_mem_stats(f"R{rank} After: L{level:02} Finalize good")

        mask_bad_pts = (~is_good)[map_n_to_k]
        dx_bad = dx[mask_bad_pts]
        bad_counts = counts[bad_idx]
        bad_starts = xp.r_[0, xp.cumsum(bad_counts)[:-1]]

        if IGNORE_GRADIENT or not USE_HESSIAN_INFORMED_SPLIT:
            M_split = xp.eye(d)[None, :, :].repeat(n_bad_clusters, axis=0)
        else:
            diff_bad = diff[mask_bad_pts]
            F_bad_pts = F_sorted[mask_bad_pts]
            sum_sq_err = xp.add.reduceat(diff_bad**2, bad_starts)
            sum_sq_val = xp.add.reduceat(F_bad_pts**2, bad_starts)
            scores = xp.sqrt(sum_sq_err) / (xp.sqrt(sum_sq_val) + TINY_EPS)
            active_comps = xp.argmax(scores, axis=1)

            if not USE_COVARIANCE_HESSIAN:
                G_bad_all = G_sorted[mask_bad_pts]
                g_mu_bad_all = g_mus[bad_idx]
                map_bad_n_to_k = repeats(xp.arange(n_bad_clusters), bad_counts)
                pt_active_comp = active_comps[map_bad_n_to_k]
                range_N = xp.arange(int(dx_bad.shape[0]))
                G_sel = G_bad_all[range_N, pt_active_comp, :]
                g_mu_sel = g_mu_bad_all[map_bad_n_to_k, pt_active_comp, :]
                dG_bad = G_sel - g_mu_sel
                M_split = estimate_hessian_sqrt_symmetric(dx_bad, dG_bad, bad_starts)
            else:
                F_bad_all = F_sorted[mask_bad_pts]
                M_split = compute_good_covariance_hessian(
                    dx_bad, F_bad_all, f_lin[mask_bad_pts], bad_starts
                )

        map_bad_n_to_k = repeats(xp.arange(n_bad_clusters), bad_counts)
        M_exp = M_split[map_bad_n_to_k]
        Z = (M_exp @ dx_bad[:, :, None])[:, :, 0]
        ZZ_all = Z[:, :, None] * Z[:, None, :]
        CovZ = xp.add.reduceat(ZZ_all, bad_starts)
        CovZ = 0.5 * (CovZ + CovZ.transpose(0, 2, 1))
        v = get_dominant_eigenvector(CovZ)
        v_exp = v[map_bad_n_to_k]
        proj_scores = xp.sum(Z * v_exp, axis=1)
        split_right = compute_median_split_decision(
            proj_scores, map_bad_n_to_k, bad_starts, bad_counts
        )

        idx_bad_pool = idx_sorted[mask_bad_pts]
        old_labels = uniq_labels[bad_idx]
        old_per_point = old_labels[map_bad_n_to_k]
        new_labels = old_per_point * 2 + split_right.astype(xp.int64)
        pool["labels"][idx_bad_pool] = new_labels

        logger.info(
            f"R{rank} L{level:03}: K={K_batch}, active={n_active}, "
            f"good={n_good}, bad={n_bad_clusters}, "
            f"t={time.perf_counter()-t_level:.2f}s"
        )
        level += 1

    return out_buffers, out_count, total_f_evals


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def adaptive_clustering(
    X: Array,
    *,
    f_fn: Optional[Callable[[Array], Array]] = None,
    g_fn: Optional[Callable[[Array], Array]] = None,
    oracle_fn: Optional[Callable[[Array], tuple[Array, Array]]] = None,
    TOL: float,
    ERROR_METRIC: Literal["L2", "Linf-global", "Linf-local"] = "Linf-global",
    INITIAL_CLUSTERING_METHOD: Literal["kmeans", "knndbscan"] = "kmeans",
    NUM_INITIAL_CLUSTERS: int = 100,
    MIN_PTS_PER_CLUSTER: int = 100,
    MAX_CLUSTERS_EST: int = 100_000,
    Q_SHRINK: float = 0.05,
    SEED: int = 0,
    LOG_MEM_STATS: bool = False,
    # Pool sizing (SYNTH_BUDGET is an initial-capacity hint; pool grows dynamically)
    SYNTH_BUDGET: Optional[int] = None,
    SYNTH_BUDGET_FACTOR: float = 50.0,
    MAX_LEVELS: int = 50,
    # Hessian informed splitting
    USE_HESSIAN_INFORMED_SPLIT: bool = True,
    USE_COVARIANCE_HESSIAN: bool = False,
    # Testing flags
    IGNORE_GRADIENT: bool = False,
    STORE_HESSIAN_SQRT_PER_CLUSTER: bool = False,
    STORE_TREE_DATA: bool = False,
) -> Optional[dict[str, Array]]:
    """
    MPI-aware adaptive clustering for F: R^d -> R^m.

    Architecture:
      - Rank 0: evaluates F/G on X, runs initial clustering, partitions by cluster
        label, scatters subsets to all ranks.
      - All ranks: independently run adaptive refinement on their local subset with
        dynamic pool resizing (no hard SYNTH_BUDGET cap).
      - Rank 0: gathers local results, concatenates, and returns merged dict.
      - Non-zero ranks return None.

    Return keys (rank 0 only): centroids, f, g, radii, L_obs, E_cluster, counts, Q,
    [H_sqrt], [tree_data], num_levels, num_generated_pts, total_f_evals.
    """
    if oracle_fn is not None:
        if f_fn is not None or g_fn is not None:
            raise ValueError("Specify oracle_fn OR (f_fn + g_fn), not both.")
    else:
        if f_fn is None or g_fn is None:
            raise ValueError("Must specify oracle_fn OR both f_fn and g_fn.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    _start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Rank 0: evaluate, cluster, partition, prepare scatter payloads
    # ------------------------------------------------------------------
    F_max_global_cpu: Optional[np.ndarray] = None
    scatter_data: Optional[list] = None

    if rank == 0:
        X0 = xp.asarray(X)
        N0, d = X0.shape
        logger.info(f"Rank 0: N0={N0:,}, d={d}")

        _t0 = time.perf_counter()
        if oracle_fn is not None:
            F0_val, G0_val = oracle_fn(X0)
        else:
            F0_val = f_fn(X0)  # type: ignore
            G0_val = (
                xp.zeros((N0, F0_val.shape[1], d), dtype=X0.dtype)
                if IGNORE_GRADIENT
                else g_fn(X0)  # type: ignore
            )
        logger.info(f"Rank 0: initial eval {time.perf_counter()-_t0:.2f}s")

        F_max_global_cpu = _to_host(xp.max(xp.abs(F0_val), axis=0))

        # Initial clustering
        _t1 = time.perf_counter()
        if INITIAL_CLUSTERING_METHOD == "kmeans":
            if IS_CUPY_BACKEND:
                from cuml.cluster import KMeans as cuKMeans  # type: ignore[import]

                km = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
                labels0 = km.fit_predict(X0)
            else:
                km = KMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
                labels0 = km.fit_predict(np.asarray(_to_host(X0)))
            labels0 = (
                labels0.astype(np.int64)
                if isinstance(labels0, np.ndarray)
                else _to_host(labels0).astype(np.int64)
            )
        elif INITIAL_CLUSTERING_METHOD == "knndbscan":
            from knndbscan import (
                knn_dbscan,
            )  # lazy: avoids UCX atexit registration at module load

            _device = "gpu" if IS_CUPY_BACKEND else "cpu"
            labels0 = _to_host(
                knn_dbscan(
                    X0,
                    eps=0.1,
                    min_samples=MIN_PTS_PER_CLUSTER,
                    n_jobs=-1,
                    device=_device,
                )
            ).astype(np.int64)
        else:
            raise ValueError(
                f"Invalid INITIAL_CLUSTERING_METHOD: {INITIAL_CLUSTERING_METHOD}"
            )
        logger.info(
            f"Rank 0: clustering {time.perf_counter()-_t1:.2f}s, "
            f"{len(np.unique(labels0))} clusters -> {size} ranks"
        )

        # Move data to CPU for safe MPI scattering
        X0_cpu = _to_host(X0)
        F0_cpu = _to_host(F0_val)
        G0_cpu = _to_host(G0_val)

        # Partition by label chunks so full clusters stay together
        unique_lbls = np.unique(labels0)
        label_chunks = np.array_split(unique_lbls, size)

        scatter_data = []
        for chunk in label_chunks:
            mask = np.isin(labels0, chunk)
            scatter_data.append(
                {
                    "X": X0_cpu[mask],
                    "F": F0_cpu[mask],
                    "G": G0_cpu[mask],
                    "labels": labels0[mask],
                }
            )
    else:
        scatter_data = None

    # ------------------------------------------------------------------
    # 2. Broadcast F_max_global; scatter data
    # ------------------------------------------------------------------
    F_max_global_cpu = comm.bcast(F_max_global_cpu, root=0)
    F_max_global = _to_device(F_max_global_cpu)  # type: ignore

    local_dict = comm.scatter(scatter_data, root=0)

    local_X = _to_device(local_dict["X"])
    local_F = _to_device(local_dict["F"])
    local_G = _to_device(local_dict["G"])
    local_labels = _to_device(local_dict["labels"])

    N_local = int(local_X.shape[0])
    d = int(local_X.shape[1])
    fd = int(local_F.shape[1])
    dtype = local_X.dtype

    logger.info(f"Rank {rank}: received {N_local:,} pts, starting refinement.")

    # ------------------------------------------------------------------
    # 3. Build local pool (dynamic; starts at SYNTH_BUDGET_FACTOR * N_local)
    # ------------------------------------------------------------------
    if SYNTH_BUDGET is None:
        init_cap = int(N_local + max(0, SYNTH_BUDGET_FACTOR * N_local))
    else:
        init_cap = int(N_local + SYNTH_BUDGET // max(1, comm.Get_size()))

    capacity = init_cap
    pool: dict[str, Array] = {
        "X": xp.empty((capacity, d), dtype=dtype),
        "F": xp.empty((capacity, fd), dtype=dtype),
        "G": xp.empty((capacity, fd, d), dtype=dtype),
        "labels": xp.empty((capacity,), dtype=xp.int64),
        "active": xp.zeros((capacity,), dtype=bool),
    }
    pool["X"][:N_local] = local_X
    pool["F"][:N_local] = local_F
    pool["G"][:N_local] = local_G
    pool["labels"][:N_local] = local_labels
    pool["active"][:N_local] = True

    # ------------------------------------------------------------------
    # 4. Local refinement
    # ------------------------------------------------------------------
    out_buffers, out_count, local_f_evals = _run_local_refinement(
        pool=pool,
        n_total=N_local,
        capacity=capacity,
        f_fn=f_fn,
        g_fn=g_fn,
        oracle_fn=oracle_fn,
        TOL=TOL,
        ERROR_METRIC=ERROR_METRIC,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        MAX_CLUSTERS_EST=max(1, MAX_CLUSTERS_EST // max(1, size)),
        Q_SHRINK=Q_SHRINK,
        SEED=SEED,
        LOG_MEM_STATS=LOG_MEM_STATS,
        MAX_LEVELS=MAX_LEVELS,
        USE_HESSIAN_INFORMED_SPLIT=USE_HESSIAN_INFORMED_SPLIT,
        USE_COVARIANCE_HESSIAN=USE_COVARIANCE_HESSIAN,
        IGNORE_GRADIENT=IGNORE_GRADIENT,
        STORE_HESSIAN_SQRT_PER_CLUSTER=STORE_HESSIAN_SQRT_PER_CLUSTER,
        STORE_TREE_DATA=STORE_TREE_DATA,
        F_max_global=F_max_global,
        rank=rank,
    )

    # ------------------------------------------------------------------
    # 5. Pack local results to CPU dict
    # ------------------------------------------------------------------
    local_result_cpu = {
        k: _to_host(out_buffers[k][:out_count]) for k in out_buffers if k != "tree_data"
    }
    if STORE_TREE_DATA:
        local_result_cpu["tree_data"] = out_buffers.get("tree_data", {})

    # Reduce total_f_evals across all ranks
    total_f_evals = comm.reduce(local_f_evals, op=MPI.SUM, root=0)

    # ------------------------------------------------------------------
    # 6. Gather & merge on rank 0
    # ------------------------------------------------------------------
    all_results = comm.gather(local_result_cpu, root=0)

    if rank == 0:
        logger.info("Rank 0: merging results...")
        array_keys = [k for k in all_results[0] if k != "tree_data"]  # type: ignore
        merged: dict[str, Any] = {}

        for k in array_keys:
            merged[k] = np.concatenate([r[k] for r in all_results], axis=0)  # type: ignore
            if IS_CUPY_BACKEND:
                merged[k] = _to_device(merged[k])

        if STORE_TREE_DATA:
            merged["tree_data"] = {}
            for r in all_results:  # type: ignore
                for lvl, data in r.get("tree_data", {}).items():
                    if lvl not in merged["tree_data"]:
                        merged["tree_data"][lvl] = data
                    else:
                        # Merge same-level tree data across ranks by concatenating
                        existing = merged["tree_data"][lvl]
                        for dk in (
                            "centroids",
                            "f",
                            "g",
                            "radii",
                            "L_obs",
                            "E_cluster",
                            "counts",
                            "Q",
                        ):
                            if dk in existing and dk in data:
                                existing[dk] = np.concatenate(
                                    [_to_host(existing[dk]), _to_host(data[dk])], axis=0
                                )
                        existing.setdefault("points_in_clusters", []).extend(
                            data.get("points_in_clusters", [])
                        )

        total_clusters = int(merged["centroids"].shape[0])
        merged["num_levels"] = MAX_LEVELS  # approximation
        merged["num_generated_pts"] = total_clusters  # rough proxy
        merged["total_f_evals"] = int(total_f_evals)  # type: ignore
        logger.info(
            f"Rank 0: final clusters={total_clusters:,}, "
            f"total time={time.perf_counter()-_start:.2f}s"
        )
        return merged
    else:
        return None


if __name__ == "__main__":
    import argparse
    from functools import partial

    from src import FIGURES_DIR
    from src.logging_utils import setup_logging
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging(console_level=logging.DEBUG)

    AC_PLOTS_DIR = FIGURES_DIR / "adaptive_clustering_rosenbrock"

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Adaptive GPU Clustering Example with Rosenbrock Function"
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Dimensionality of the problem"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-2, help="Tolerance for training and evaluation"
    )
    parser.add_argument(
        "--n-train", type=float, default=100_000, help="Number of training samples"
    )
    parser.add_argument(
        "--n-init-clusters",
        type=int,
        default=25,
        help="Number of initial clusters for adaptive clustering",
    )
    parser.add_argument(
        "--n-min-per-cluster",
        type=int,
        default=500,
        help="Minimum points per cluster for adaptive clustering",
    )
    parser.add_argument(
        "--error-metric",
        type=str,
        default="Linf-global",
        choices=["L2", "Linf-global", "Linf-local"],
        help="Error metric for adaptive clustering",
    )
    parser.add_argument(
        "--initial-clustering-method",
        type=str,
        default="kmeans",
        choices=["kmeans", "knndbscan"],
        help="Method for initial clustering",
    )
    parser.add_argument(
        "--rosenbrock-a",
        type=float,
        default=100.0,
        help="Rosenbrock function parameter 'a'",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="",
        help="Subdirectory for output files",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-execution even if output file exists",
    )

    args = parser.parse_args()
    DIM = args.dim
    TOL = args.tol
    N_TRAIN = int(args.n_train)
    INITIAL_CLUSTERING_METHOD = args.initial_clustering_method
    NUM_INITIAL_CLUSTERS = args.n_init_clusters
    MIN_PTS_PER_CLUSTER = args.n_min_per_cluster
    ERROR_METRIC = args.error_metric
    A_ROSENBROCK = args.rosenbrock_a
    OUTPUT_SUBDIR = args.output_subdir
    FORCE_REEXEC = args.force

    # Check if file already exists
    AC_PLOTS_DIR = AC_PLOTS_DIR / OUTPUT_SUBDIR if OUTPUT_SUBDIR else AC_PLOTS_DIR
    AC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    img_fpath = (
        AC_PLOTS_DIR
        / f"adaptive_clustering_rosenbrock_{A_ROSENBROCK}_tol{TOL}_error_metric_{ERROR_METRIC}_init{NUM_INITIAL_CLUSTERS}_min{MIN_PTS_PER_CLUSTER}_initmethod_{INITIAL_CLUSTERING_METHOD}.png"
    )
    if DIM == 2 and img_fpath.exists() and not FORCE_REEXEC:
        print(f"Plot file {img_fpath} already exists. Skipping execution.")
        exit(0)

    # Creat partial Rosenbrock functions
    rosenbrock = partial(rosenbrock, a=A_ROSENBROCK)
    rosenbrock_grad = partial(rosenbrock_grad, a=A_ROSENBROCK)

    xp.random.seed(0)
    print(f"Generating {N_TRAIN:,} points (d={DIM})...")
    PROB_LIMS = (-1.0, 3)
    X = xp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(N_TRAIN, DIM))

    print("Starting Adaptive Clustering...")
    results = adaptive_clustering(
        X,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        TOL=TOL,
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        ERROR_METRIC=ERROR_METRIC,
        SYNTH_BUDGET_FACTOR=1000,
    )

    # Only rank 0 gets results; other ranks return None.
    if results is None:
        MPI.Finalize()
        os._exit(0)

    print("-" * 40)
    print(f"Final Cluster Count: {results['centroids'].shape[0]:,}")

    E_cluster_vals = results["E_cluster"]
    counts = results["counts"]
    print(
        "E_cluster stats: min/mean/max =",
        xp.round(E_cluster_vals.min(), 6),
        xp.round(E_cluster_vals.mean(), 6),
        xp.round(E_cluster_vals.max(), 6),
    )
    print(
        "Counts stats: min/mean/max =",
        int(counts.min()),
        f"{counts.mean():.1f}",
        int(counts.max()),
    )

    if DIM == 2 and len(results["centroids"]) <= 10_000:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Ellipse

        if IS_CUPY_BACKEND:
            results = {k: xp.asnumpy(v) for k, v in results.items()}

        # 1 plot 2 subplots (1X2)
        # Subplot 1: scatter of final cluster centroids on contourf
        # Subplot 2: scatter of final cluster centroids with trust ellipsoids on contourf as well as circles corresponding to radii

        x = np.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        y = np.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        X_grid, Y_grid = np.meshgrid(x, y)

        # Ensure input to rosenbrock is (N, 2)
        grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
        Z_grid = rosenbrock(grid_pts)
        if hasattr(Z_grid, "get"):
            Z_grid = Z_grid.get()  # Handle cupy return if needed
        Z_grid = Z_grid.reshape(X_grid.shape)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # -- Subplot 1: Contours + Centroids --
        cs = axs[0].contourf(
            X_grid,
            Y_grid,
            Z_grid,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(cs, ax=axs[0])
        axs[0].set_title("Rosenbrock Function Contour")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].set_aspect("equal")
        axs[0].scatter(
            results["centroids"][:, 0],
            results["centroids"][:, 1],
            c="red",
            s=5,
            label="Centroids",
        )

        # -- Subplot 2: Contours + OBBs --
        cs2 = axs[1].contourf(
            X_grid,
            Y_grid,
            Z_grid,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(cs2, ax=axs[1])
        axs[1].set_title("Adaptive Clustering with Oriented Bounding Boxes")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].set_aspect("equal")

        # Plot Centroids
        axs[1].scatter(
            results["centroids"][:, 0],
            results["centroids"][:, 1],
            c="red",
            s=5,
            zorder=10,
        )

        # Draw Ellipsoids
        centroids = results["centroids"]
        Qs = results["Q"]
        radii = results["radii"]

        for i in range(centroids.shape[0]):
            center = centroids[i]
            Q = Qs[i]
            radius = radii[i]

            # Eigen-decomposition of Q
            eigvals, eigvecs = np.linalg.eigh(Q)
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            a, b = 2.0 * np.sqrt(1.0 / eigvals)  # 2-sigma lengths
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ellip = Ellipse(
                xy=center,
                width=a,
                height=b,
                angle=angle,
                edgecolor="blue",
                facecolor="none",
                lw=0.5,
                alpha=0.7,
            )
            axs[1].add_patch(ellip)

            # Draw radius circle with dashed line
            circ = Circle(
                xy=center,
                radius=radius,
                edgecolor="black",
                facecolor="none",
                lw=0.5,
                alpha=0.7,
                linestyle="dashed",
            )
            axs[1].add_patch(circ)

        # Update legend to include ellipses and circles
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Centroids",
                markerfacecolor="red",
                markersize=5,
            ),
            Line2D([0], [0], color="blue", lw=0.5, label="Trust Ellipsoid"),
            Line2D(
                [0],
                [0],
                color="black",
                lw=0.5,
                linestyle="dashed",
                label="Trust Radius",
            ),
        ]
        axs[1].legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        fig.suptitle(
            f"Rosenbrock (a={A_ROSENBROCK}) (TOL={TOL}, ERROR_METRIC={ERROR_METRIC}), N_CLUSTERS={len(centroids):,}, INITIAL_CLUSTERING_METHOD={INITIAL_CLUSTERING_METHOD}",
            fontsize=11,
        )

        plt.subplots_adjust(top=0.88)
        plt.savefig(img_fpath, dpi=300)
        print(f"Saved figure to {img_fpath}")
    else:
        print("Skipping plot (d != 2 or too many clusters).")

    # Save npz file of results
    npz_fpath = img_fpath.with_suffix(".npz")
    if IS_CUPY_BACKEND:
        results = {k: xp.asnumpy(v) for k, v in results.items()}
    xp.savez_compressed(npz_fpath, **results)

    # Finalize MPI cleanly, then use os._exit to bypass Python atexit/GC
    # shutdown which triggers a UCX/knndbscan libucp destructor segfault.
    MPI.Finalize()
    os._exit(0)
