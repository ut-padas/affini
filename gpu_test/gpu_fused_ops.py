import gc
import logging
import time
from typing import Any, Callable, Literal, Optional

import numpy as np
from cuml.cluster import KMeans as cuKMeans
from cupyx import scatter_max as cupy_scatter_max
from sklearn.cluster import KMeans

from src import IS_CUPY_BACKEND, Array, xp
from src.logging_config import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)

TINY_EPS = 1e-12


_taylor_kernel_code = r"""
extern "C" __global__
void fused_taylor_error(
    const double* X, const double* mus, const long long* labels,
    const double* F_vals, const double* f_mus, const double* g_mus,
    double* E_out, int N, int D) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    long long k = labels[idx];
    int mu_offset = k * D;
    int x_offset = idx * D;

    double dot = 0.0;
    for (int j = 0; j < D; j++) {
        double diff = X[x_offset + j] - mus[mu_offset + j];
        dot += diff * g_mus[mu_offset + j];
    }

    double f_lin = f_mus[k] + dot;
    // CRITICAL FIX: fabs() ensures floating point absolute value (not integer abs)
    E_out[idx] = fabs(F_vals[idx] - f_lin);
}
"""
_mod_taylor = xp.RawModule(code=_taylor_kernel_code)
_kern_taylor = _mod_taylor.get_function("fused_taylor_error")


_stats_kernel_code = r"""
extern "C" __global__
void fused_hessian_stats(
    const double* X, const double* G, 
    const double* mus, const double* g_mus, 
    const long long* labels,
    double* C_out, double* E_out,
    int N, int D, int K, bool calc_E) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    long long k = labels[idx];
    int mu_offset = k * D;
    int x_offset = idx * D;

    for (int r = 0; r < D; r++) {
        double dx_r = X[x_offset + r] - mus[mu_offset + r];

        // Covariance C (Symmetric): dx * dx.T
        for (int c = r; c < D; c++) {
            double dx_c = X[x_offset + c] - mus[mu_offset + c];
            double val_c = dx_r * dx_c;
            
            // CRITICAL FIX: Direct double atomicAdd
            atomicAdd(&C_out[k*D*D + r*D + c], val_c);
            
            if (r != c) {
                atomicAdd(&C_out[k*D*D + c*D + r], val_c);
            }
        }

        if (calc_E) {
            for (int c = 0; c < D; c++) {
                double dg_c = G[x_offset + c] - g_mus[mu_offset + c];
                double val_e = dx_r * dg_c;
                atomicAdd(&E_out[k*D*D + r*D + c], val_e);
            }
        }
    }
}
"""
_mod_stats = xp.RawModule(code=_stats_kernel_code)
_kern_stats = _mod_stats.get_function("fused_hessian_stats")


_proj_score_kernel_code = r"""
extern "C" __global__
void fused_proj_score(
    const double* X, const double* mus, const long long* labels,
    const double* M, const double* v,
    double* scores_out, int N, int D) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    long long k = labels[idx];
    int mu_offset = k * D;
    int x_offset = idx * D;
    
    int M_start = k * D * D;
    int v_start = k * D;

    double score = 0.0;
    
    for (int r = 0; r < D; r++) {
        double z_r = 0.0;
        for (int c = 0; c < D; c++) {
            double dx_c = X[x_offset + c] - mus[mu_offset + c];
            z_r += M[M_start + r * D + c] * dx_c;
        }
        score += z_r * v[v_start + r];
    }
    
    scores_out[idx] = score;
}
"""
_mod_proj = xp.RawModule(code=_proj_score_kernel_code)
_kern_proj = _mod_proj.get_function("fused_proj_score")


# Kernel: Q-Scaling (Mahalanobis Distance)
# Fuses: dx = X - mu; proj = U.T @ dx; dist = sum(proj^2 * L_inv)
# Memory: O(1) per point (No N*D*D allocations)
_q_kernel_code = r"""
extern "C" __global__
void fused_q_dist(
    const double* X, const double* mus, const long long* labels,
    const double* U, const double* L_inv,
    double* dist_out, int N, int D) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    long long k = labels[idx];
    int mu_offset = k * D;
    int x_offset = idx * D;
    
    // U is (K, D, D). Row-major.
    // U[k] is the eigenvector matrix. 
    // We want z = U^T * dx.
    // z_j = (Col j of U) dot dx.
    // U indices: k*D*D + row*D + col.
    
    int u_start = k * D * D;
    int l_start = k * D;
    
    double dist_sq = 0.0;

    for (int j = 0; j < D; j++) {
        // Compute projection onto j-th eigenvector
        double proj_j = 0.0;
        for (int r = 0; r < D; r++) {
            double dx_r = X[x_offset + r] - mus[mu_offset + r];
            // U[r, j] is the r-th component of j-th eigenvector
            proj_j += dx_r * U[u_start + r * D + j];
        }
        
        // dist += (proj_j^2) * lambda_inv_j
        dist_sq += (proj_j * proj_j) * L_inv[l_start + j];
    }
    
    dist_out[idx] = dist_sq;
}
"""
_mod_q = xp.RawModule(code=_q_kernel_code)
_kern_q = _mod_q.get_function("fused_q_dist")


def run_fused_q_dist(X, mus, labels, U, L_inv):
    N, D = X.shape
    dist_out = xp.empty((N,), dtype=xp.float64)
    labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)

    block = 256
    grid = (N + block - 1) // block
    _kern_q(
        (grid,),
        (block,),
        (
            xp.ascontiguousarray(X),
            xp.ascontiguousarray(mus),
            labels_64,
            xp.ascontiguousarray(U),
            xp.ascontiguousarray(L_inv),
            dist_out,
            N,
            D,
        ),
    )
    return dist_out


# -----------------------------------------------------------------------------
# 2. TYPE-SAFE WRAPPERS
# -----------------------------------------------------------------------------


def run_fused_taylor(X, mus, labels, F_vals, f_mus, g_mus):
    N, D = X.shape
    E_out = xp.empty((N,), dtype=xp.float64)
    # FIX: Explicit int64 cast prevents garbage reads
    labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)

    block = 256
    grid = (N + block - 1) // block
    _kern_taylor(
        (grid,),
        (block,),
        (
            xp.ascontiguousarray(X),
            xp.ascontiguousarray(mus),
            labels_64,
            xp.ascontiguousarray(F_vals),
            xp.ascontiguousarray(f_mus),
            xp.ascontiguousarray(g_mus),
            E_out,
            N,
            D,
        ),
    )
    return E_out


def run_fused_stats(X, mus, labels, K, D, G=None, g_mus=None, calc_E=False):
    N = X.shape[0]
    C_out = xp.zeros((K, D, D), dtype=xp.float64)
    E_out = (
        xp.zeros((K, D, D), dtype=xp.float64)
        if calc_E
        else xp.array([], dtype=xp.float64)
    )

    G_ptr = G if G is not None else X
    g_mus_ptr = g_mus if g_mus is not None else mus
    labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)

    block = 256
    grid = (N + block - 1) // block
    _kern_stats(
        (grid,),
        (block,),
        (
            xp.ascontiguousarray(X),
            xp.ascontiguousarray(G_ptr),
            xp.ascontiguousarray(mus),
            xp.ascontiguousarray(g_mus_ptr),
            labels_64,
            C_out,
            E_out,
            N,
            D,
            K,
            calc_E,
        ),
    )
    return C_out, E_out


def run_fused_score(X, mus, labels, M, v):
    N, D = X.shape
    scores = xp.empty((N,), dtype=xp.float64)
    labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)

    block = 256
    grid = (N + block - 1) // block
    _kern_proj(
        (grid,),
        (block,),
        (
            xp.ascontiguousarray(X),
            xp.ascontiguousarray(mus),
            labels_64,
            xp.ascontiguousarray(M),
            xp.ascontiguousarray(v),
            scores,
            N,
            D,
        ),
    )
    return scores


def scatter_maximum_1d(out: Array, idx: Array, vals: Array) -> None:
    if IS_CUPY_BACKEND:
        cupy_scatter_max(out, idx, vals)
    else:
        xp.maximum.at(out, idx, vals)


def flatten_function_array(F: Array) -> Array:
    F = xp.asarray(F)
    if getattr(F, "ndim", 1) == 2 and F.shape[1] == 1:
        return F[:, 0]
    return F.reshape((-1,))


def flatten_gradient_array(G: Array) -> Array:
    G = xp.asarray(G)
    if getattr(G, "ndim", 1) == 3 and G.shape[1] == 1:
        return G[:, 0, :]
    if getattr(G, "ndim", 1) == 2:
        return G
    raise ValueError(f"Unexpected gradient shape {G.shape}; expected (N,d) or (N,1,d).")


def compute_label_segments(sorted_labels: Array) -> tuple[Array, Array, Array]:
    uniq, starts, counts = xp.unique(
        sorted_labels, return_index=True, return_counts=True
    )
    return uniq, starts, counts


def compute_median_split_decision(
    scores: Array, cluster_id: Array, starts: Array, counts: Array
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


def allocate_output_buffers(
    MAX_CLUSTERS_EST: int, d: int, dtype: Any
) -> tuple[dict, int]:
    cap = max(1, int(MAX_CLUSTERS_EST))
    buffers = {
        "centroids": xp.empty((cap, d), dtype=dtype),
        "f": xp.empty((cap,), dtype=dtype),
        "g": xp.empty((cap, d), dtype=dtype),
        "Q": xp.empty((cap, d, d), dtype=dtype),
        "E_rel": xp.empty((cap,), dtype=dtype),
        "counts": xp.empty((cap,), dtype=xp.int64),
    }
    return buffers, cap


def adjust_buffer_capacity(
    buffers: dict, current_count: int, current_cap: int, required_total: int
) -> tuple[dict, int]:
    if required_total <= current_cap:
        return buffers, current_cap

    new_cap = max(required_total, max(1, current_cap) * 2)
    logger.warning(f"Resizing output buffers: {current_cap} -> {new_cap} clusters")
    d = int(buffers["centroids"].shape[1])
    dtype = buffers["centroids"].dtype
    new_buffers = {
        "centroids": xp.empty((new_cap, d), dtype=dtype),
        "f": xp.empty((new_cap,), dtype=dtype),
        "g": xp.empty((new_cap, d), dtype=dtype),
        "Q": xp.empty((new_cap, d, d), dtype=dtype),
        "E_rel": xp.empty((new_cap,), dtype=dtype),
        "counts": xp.empty((new_cap,), dtype=xp.int64),
    }
    if current_count > 0:
        for k in new_buffers:
            new_buffers[k][:current_count] = buffers[k][:current_count]
    del buffers
    return new_buffers, int(new_cap)


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
    need = xp.maximum(0, MIN_PTS_PER_CLUSTER - counts)
    total_synth = int(xp.sum(need))
    if total_synth == 0:
        return None, None

    rng = xp.random.default_rng(SEED + level)
    k_synth = repeats(xp.arange(int(need.shape[0])), need)
    labels_synth = uniq_labels[k_synth]

    m = counts[k_synth]
    s0 = starts[k_synth]

    rand = rng.random
    i = (xp.floor(rand(total_synth) * m)).astype(xp.int64)
    j = (xp.floor(rand(total_synth) * (m - 1))).astype(xp.int64)
    j = j + (j >= i)

    idx_i = s0 + i
    idx_j = s0 + j

    a = rand((total_synth, 1)).astype(X_sorted.dtype)
    X_synth = a * X_sorted[idx_i] + (1.0 - a) * X_sorted[idx_j]
    return X_synth, labels_synth.astype(xp.int64)


# -----------------------------------------------------------------------------
# 3. MAIN ADAPTIVE CLUSTERING ROUTINE
# -----------------------------------------------------------------------------


def adaptive_clustering(
    X: Array,
    *,
    f_fn: Callable[[Array], Array],
    g_fn: Callable[[Array], Array],
    TOL: float,
    ERROR_SCALING_METHOD: Literal["global", "local"] = "global",
    NUM_INITIAL_CLUSTERS: int = 100,
    MIN_PTS_PER_CLUSTER: int = 100,
    MAX_CLUSTERS_EST: int = 100_000,
    SEED: int = 0,
    LOG_MEM_STATS: bool = False,
    SYNTH_BUDGET: Optional[int] = None,
    SYNTH_BUDGET_FACTOR: float = 1.0,
    Q_EPS_AXIS: float = 1e-8,
    Q_SHRINK: float = 0.05,
    H_RIDGE: float = 1e-18,
    MAX_LEVELS: int = 50,
) -> dict[str, Array]:
    _start_time = time.perf_counter()
    X0 = xp.asarray(X, dtype=xp.float64)
    N0, d = X0.shape

    if SYNTH_BUDGET is None:
        SYNTH_BUDGET = int(max(0, SYNTH_BUDGET_FACTOR * N0))

    capacity = int(N0 + SYNTH_BUDGET)
    X_pool = xp.empty((capacity, d), dtype=X0.dtype)
    X_pool[:N0] = X0
    n_total = int(N0)

    _t0 = time.perf_counter()
    F_pool = xp.empty((capacity,), dtype=X0.dtype)
    G_pool = xp.empty((capacity, d), dtype=X0.dtype)
    F_pool[:N0] = flatten_function_array(f_fn(X_pool[:N0]))
    G_pool[:N0] = flatten_gradient_array(g_fn(X_pool[:N0]))
    logger.info(f"Runtime - Initial f/g eval: {time.perf_counter()-_t0:.3f}s")

    F_inf = xp.max(xp.abs(F_pool[:N0]))
    synth_added_prev_level = False

    _t1 = time.perf_counter()
    if IS_CUPY_BACKEND:
        km = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
        labels0 = km.fit_predict(X_pool[:N0])
    else:
        km = KMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
        labels0 = km.fit_predict(xp.asarray(X_pool[:N0]))
    labels_pool = xp.empty((capacity,), dtype=xp.int64)
    labels_pool[:N0] = labels0.astype(xp.int64)
    logger.info(f"Runtime - Initial KMeans: {time.perf_counter()-_t1:.3f}s")

    active = xp.zeros((capacity,), dtype=bool)
    active[:N0] = True

    buffers, out_cap = allocate_output_buffers(MAX_CLUSTERS_EST, d, X0.dtype)
    out_count = 0

    level = 0
    while level <= MAX_LEVELS:
        t_level = time.perf_counter()
        if LOG_MEM_STATS:
            log_mem_stats(f"Start: L{level:02}")

        if synth_added_prev_level:
            F_inf = xp.max(xp.abs(F_pool[:n_total]))

        idx_active = xp.nonzero(active[:n_total])[0]
        n_active = int(idx_active.shape[0])
        if n_active == 0:
            break

        synth_added_this_level = False
        label_active = labels_pool[idx_active]
        perm = xp.argsort(label_active)
        idx_sorted = idx_active[perm]
        label_sorted = label_active[perm]
        uniq_labels, starts, counts = compute_label_segments(label_sorted)
        K_batch = int(uniq_labels.shape[0])

        # ---- Synthetic Points ----
        X_sorted = X_pool[idx_sorted]
        X_synth, labels_synth = generate_sythetic_points(
            X_sorted,
            starts,
            counts,
            uniq_labels,
            MIN_PTS_PER_CLUSTER,
            SEED=SEED,
            level=level,
        )
        if X_synth is not None:
            M = int(X_synth.shape[0])
            n_plus_M = n_total + M

            # [FIX: Dynamic Resizing - Sequential Strategy]
            if n_plus_M > capacity:
                new_capacity = max(n_plus_M, int(capacity * 1.5))
                logger.warning(
                    f"Pool capacity exceeded. Resizing: {capacity} -> {new_capacity}"
                )

                # 1. Resize X (Largest)
                new_X_pool = xp.empty((new_capacity, d), dtype=X_pool.dtype)
                new_X_pool[:n_total] = X_pool[:n_total]
                del X_pool  # <--- FREE OLD X IMMEDIATELY
                gc.collect()
                X_pool = new_X_pool  # Rebind

                # 2. Resize G (Largest)
                new_G_pool = xp.empty((new_capacity, d), dtype=G_pool.dtype)
                new_G_pool[:n_total] = G_pool[:n_total]
                del G_pool  # <--- FREE OLD G IMMEDIATELY
                gc.collect()
                G_pool = new_G_pool

                # 3. Resize F (Smaller)
                new_F_pool = xp.empty((new_capacity,), dtype=F_pool.dtype)
                new_F_pool[:n_total] = F_pool[:n_total]
                del F_pool
                F_pool = new_F_pool

                # 4. Resize Labels
                new_labels_pool = xp.empty((new_capacity,), dtype=labels_pool.dtype)
                new_labels_pool[:n_total] = labels_pool[:n_total]
                del labels_pool
                labels_pool = new_labels_pool

                # 5. Resize Active
                new_active = xp.zeros((new_capacity,), dtype=active.dtype)
                new_active[:n_total] = active[:n_total]
                del active
                active = new_active

                # Update capacity tracker
                capacity = new_capacity
                gc.collect()

            X_pool[n_total:n_plus_M] = X_synth
            labels_pool[n_total:n_plus_M] = labels_synth
            active[n_total:n_plus_M] = True
            F_pool[n_total:n_plus_M] = flatten_function_array(f_fn(X_synth))
            G_pool[n_total:n_plus_M] = flatten_gradient_array(g_fn(X_synth))
            n_total += M
            synth_added_this_level = True

            # Re-sort
            idx_active = xp.nonzero(active[:n_total])[0]
            label_active = labels_pool[idx_active]
            perm = xp.argsort(label_active)
            idx_sorted = idx_active[perm]
            label_sorted = label_active[perm]
            uniq_labels, starts, counts = compute_label_segments(label_sorted)
            K_batch = int(uniq_labels.shape[0])

        # ---- Centroids ----
        X_sorted = X_pool[idx_sorted]
        sums = xp.add.reduceat(X_sorted, starts)
        mus = sums / counts[:, None]
        f_mus = flatten_function_array(f_fn(mus))
        g_mus = flatten_gradient_array(g_fn(mus))

        # ---- Fused Taylor Error (SAFE) ----
        err_point = run_fused_taylor(
            X_sorted,
            mus,
            xp.searchsorted(uniq_labels, label_sorted),
            F_pool[idx_sorted],
            f_mus,
            g_mus,
        )

        E_abs = xp.zeros((K_batch,), dtype=err_point.dtype)
        map_n_to_k = repeats(xp.arange(K_batch), counts)
        scatter_maximum_1d(E_abs, map_n_to_k, err_point)

        if ERROR_SCALING_METHOD == "global":
            E_rel = E_abs / F_inf
            is_good = E_rel < TOL
        else:
            F_max_cluster = xp.zeros((K_batch,), dtype=F_pool.dtype)
            scatter_maximum_1d(F_max_cluster, map_n_to_k, xp.abs(F_pool[idx_sorted]))
            E_rel = E_abs / xp.maximum(F_max_cluster, TINY_EPS)
            is_good = (E_rel < TOL) | (E_abs < TOL)

        # ---- Finalize Good Clusters ----
        good_idx = xp.nonzero(is_good)[0]
        n_good = int(good_idx.shape[0])
        if n_good > 0:
            point_good_mask = is_good[map_n_to_k]
            X_good = X_sorted[point_good_mask]

            good_counts = counts[good_idx]
            dense_labels_good = repeats(xp.arange(n_good), good_counts)

            # Fused Covariance
            Cov, _ = run_fused_stats(
                X_good, mus[good_idx], dense_labels_good, n_good, d, calc_E=False
            )
            Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

            if Q_SHRINK > 0.0:
                tr = xp.trace(Cov, axis1=1, axis2=2)
                scale = tr / max(d, 1)
                I = xp.eye(d, dtype=Cov.dtype)[None, :, :]
                Cov = (1.0 - Q_SHRINK) * Cov + Q_SHRINK * (scale[:, None, None] * I)
                Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

            Lambda, U = xp.linalg.eigh(Cov)
            Lambda_inv = 1.0 / xp.maximum(Lambda, Q_EPS_AXIS)

            # --- REPLACED: Fused Q-Scaling ---
            # Old broadcasting code removed.
            # No more U_T, dx_good, or proj allocations.

            # We calculate dist_sq directly from X_good and small U/L matrices
            dist_sq = run_fused_q_dist(
                X_good, mus[good_idx], dense_labels_good, U, Lambda_inv
            )
            # ---------------------------------

            max_dist_sq = xp.zeros((n_good,), dtype=dist_sq.dtype)
            scatter_maximum_1d(max_dist_sq, dense_labels_good, dist_sq)
            scale_fac = 1.0 / xp.maximum(max_dist_sq, Q_EPS_AXIS)

            # Reconstruct Qs for output (This is small, K_good * D^2, safe to keep)
            U_T = U.transpose(0, 2, 1)
            final_D = scale_fac[:, None] * Lambda_inv
            Qs = (U * final_D[:, None, :]) @ U_T

            end_out = out_count + n_good
            buffers, out_cap = adjust_buffer_capacity(
                buffers, out_count, out_cap, end_out
            )
            buffers["centroids"][out_count:end_out] = mus[good_idx]
            buffers["f"][out_count:end_out] = f_mus[good_idx]
            buffers["g"][out_count:end_out] = g_mus[good_idx]
            buffers["Q"][out_count:end_out] = Qs
            buffers["E_rel"][out_count:end_out] = E_rel[good_idx]
            buffers["counts"][out_count:end_out] = counts[good_idx].astype(xp.int64)
            out_count = end_out

            global_ids_good = idx_sorted[point_good_mask]
            active[global_ids_good] = False
            del X_good

        # ---- Split Bad Clusters ----
        bad_idx = xp.nonzero(~is_good)[0]
        K_bad = int(bad_idx.shape[0])
        if K_bad == 0:
            logger.info(f"Level {level:03}: no bad clusters remain.")
            break

        point_bad_mask = (~is_good)[map_n_to_k]
        X_bad = X_sorted[point_bad_mask]
        G_bad = G_pool[idx_sorted][point_bad_mask]

        bad_counts = counts[bad_idx]
        dense_labels_bad = repeats(xp.arange(K_bad), bad_counts)

        # Fused Hessian Stats
        C, E = run_fused_stats(
            X_bad,
            mus[bad_idx],
            dense_labels_bad,
            K_bad,
            d,
            G=G_bad,
            g_mus=g_mus[bad_idx],
            calc_E=True,
        )
        C = 0.5 * (C + C.transpose(0, 2, 1))
        S = E + E.transpose(0, 2, 1)

        dvals, U = xp.linalg.eigh(C)
        dvals = xp.maximum(dvals, H_RIDGE)
        Gtilde = U.transpose(0, 2, 1) @ S @ U
        denom = dvals[:, :, None] + dvals[:, None, :]
        F_mat = Gtilde / denom

        h_eigvals, V = xp.linalg.eigh(F_mat)
        Z_rot = U @ V
        w_abs_sqrt = xp.sqrt(xp.abs(h_eigvals))
        M = (Z_rot * w_abs_sqrt[:, None, :]) @ Z_rot.transpose(0, 2, 1)
        M = 0.5 * (M + M.transpose(0, 2, 1))

        CovZ = M @ C @ M.transpose(0, 2, 1)
        w_z, V_z = xp.linalg.eigh(CovZ)
        v = V_z[:, :, -1]

        # Fused Scoring
        scores = run_fused_score(X_bad, mus[bad_idx], dense_labels_bad, M, v)

        split_right = compute_median_split_decision(
            scores, dense_labels_bad, xp.r_[0, xp.cumsum(bad_counts)[:-1]], bad_counts
        )

        global_ids_bad = idx_sorted[point_bad_mask]
        old_label_values = uniq_labels[bad_idx]
        old_per_point = old_label_values[dense_labels_bad]
        labels_pool[global_ids_bad] = old_per_point * 2 + split_right.astype(xp.int64)

        synth_added_prev_level = synth_added_this_level
        logger.info(
            f"Level {level:03}: Clusters={K_batch:,}, active={n_active:,}, pool_used={n_total/capacity:.1%}"
        )
        level += 1

    final_count = int(out_count)
    out_dict = {k: buffers[k][:final_count] for k in buffers}
    out_dict["num_levels"] = level
    return out_dict


if __name__ == "__main__":
    import argparse
    from functools import partial

    from src import FIGURES_DIR
    from src.logging_config import setup_logging
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
        default=20,
        help="Number of initial clusters for adaptive clustering",
    )
    parser.add_argument(
        "--n-min-per-cluster",
        type=int,
        default=500,
        help="Minimum points per cluster for adaptive clustering",
    )
    parser.add_argument(
        "--error-scaling-method",
        type=str,
        default="global",
        choices=["local", "global"],
        help="Type of error scaling for relative error computation",
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
    NUM_INITIAL_CLUSTERS = args.n_init_clusters
    MIN_PTS_PER_CLUSTER = args.n_min_per_cluster
    ERROR_SCALING_METHOD = args.error_scaling_method
    A_ROSENBROCK = args.rosenbrock_a
    OUTPUT_SUBDIR = args.output_subdir
    FORCE_REEXEC = args.force

    # Check if file already exists
    AC_PLOTS_DIR = AC_PLOTS_DIR / OUTPUT_SUBDIR if OUTPUT_SUBDIR else AC_PLOTS_DIR
    AC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    img_fpath = (
        AC_PLOTS_DIR
        / f"ac_fused_rosenbrock_{A_ROSENBROCK}_tol{TOL}_init{NUM_INITIAL_CLUSTERS}_min{MIN_PTS_PER_CLUSTER}_{ERROR_SCALING_METHOD}.png"
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
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        # SYNTH_BUDGET_FACTOR=100,
        ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
        LOG_MEM_STATS=True,
    )

    print("-" * 40)
    print(f"Final Cluster Count: {results['centroids'].shape[0]:,}")

    E_rel_vals = results["E_rel"]
    counts = results["counts"]
    print(
        "E_rel stats: min/mean/max =",
        xp.round(E_rel_vals.min(), 6),
        xp.round(E_rel_vals.mean(), 6),
        xp.round(E_rel_vals.max(), 6),
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
        from matplotlib.patches import Ellipse

        if IS_CUPY_BACKEND:
            results = {k: xp.asnumpy(v) for k, v in results.items()}

        # 1 plot 2 subplots (1X2)
        # Subplot 1: scatter of final cluster centroids on contourf
        # Subplot 2: scatter of final cluster centroids with trust ellipses on contourf

        x = np.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        y = np.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        X_grid, Y_grid = np.meshgrid(x, y)
        Z_grid = rosenbrock(np.vstack([X_grid.ravel(), Y_grid.ravel()]).T).reshape(
            X_grid.shape
        )

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
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

        # Subplot 2: with ellipses
        cs2 = axs[1].contourf(
            X_grid,
            Y_grid,
            Z_grid,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(cs2, ax=axs[1])
        axs[1].set_title("Adaptive Clustering with Trust Ellipses")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].set_aspect("equal")
        axs[1].scatter(
            results["centroids"][:, 0],
            results["centroids"][:, 1],
            c="red",
            s=10,
            label="Centroids",
        )

        # Draw ellipses
        level = 1.0
        for i in range(results["centroids"].shape[0]):
            mu = results["centroids"][i]
            Q = results["Q"][i]

            # Eigen-decomposition of Q
            w, V = np.linalg.eigh(Q)
            w = np.maximum(w, 1e-12)

            order = np.argsort(w)[::-1]
            w = w[order]
            V = V[:, order]

            a = np.sqrt(level / w[0])
            b = np.sqrt(level / w[1])

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
            axs[1].add_patch(ellipse)
        axs[1].legend()
        plt.tight_layout()
        fig.suptitle(
            f"Fused AC: Rosenbrock (a={A_ROSENBROCK}) (TOL={TOL} ({ERROR_SCALING_METHOD}), N_INIT={NUM_INITIAL_CLUSTERS}, N_MIN={MIN_PTS_PER_CLUSTER}), Final Clusters: {results['centroids'].shape[0]:,}",
            fontsize=16,
        )

        plt.subplots_adjust(top=0.88)
        plt.savefig(img_fpath, dpi=300)
        print(f"Saved figure to {img_fpath}")
    else:
        print("Skipping plot (d != 2 or too many clusters).")
