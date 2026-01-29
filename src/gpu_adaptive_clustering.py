import gc
import logging
import time
from typing import Any, Callable, Literal, Optional

import cupy as cp
import numpy as np
from cuml.cluster import KMeans as cuKMeans
from cupyx import scatter_max as cupy_scatter_max

from src import IS_CUPY_BACKEND, Array, xp
from src.logging_utils import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)

TINY_EPS = 1e-12

# -----------------------------------------------------------------------------
# 1. CUDA KERNELS
# -----------------------------------------------------------------------------

# Kernel to compute aggregated error per point (Sum-Sq for L2, Max-Rel for Linf-Global)
_taylor_metric_kernel_code = r"""
extern "C" __global__
void fused_taylor_metric(
    const double* X, const double* mus, const long long* labels,
    const double* F_vals, const double* f_mus, const double* g_mus,
    const double* scales, 
    double* E_out, 
    int N, int D, int FD, int METRIC_MODE) 
{
    // METRIC_MODE: 0 = L2 (Sum Squares), 1 = Linf (Max Relative)
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    
    long long k = labels[idx];
    int x_offset = idx * D;
    int mu_offset = k * D;
    int f_idx_start = idx * FD;
    int f_mu_start = k * FD;
    int g_mu_start = k * FD * D;

    double val_acc = 0.0; 

    for (int j = 0; j < FD; j++) {
        double dot = 0.0;
        int g_row_start = g_mu_start + j * D;
        for (int r = 0; r < D; r++) {
            double diff = X[x_offset + r] - mus[mu_offset + r];
            dot += diff * g_mus[g_row_start + r];
        }
        
        double f_lin = f_mus[f_mu_start + j] + dot;
        double diff = F_vals[f_idx_start + j] - f_lin;
        double abs_diff = fabs(diff);

        if (METRIC_MODE == 0) {
            val_acc += diff * diff;
        } else {
            double s = scales[j]; 
            double rel = abs_diff / (s + 1e-12);
            if (rel > val_acc) val_acc = rel;
        }
    }
    E_out[idx] = val_acc;
}
"""

# Kernel to compute Absolute Error for a SINGLE output component (For Linf-Local loop)
_taylor_comp_abs_kernel_code = r"""
extern "C" __global__
void fused_taylor_component_abs(
    const double* X, const double* mus, const long long* labels,
    const double* F_vals, const double* f_mus, const double* g_mus,
    double* E_out, 
    int N, int D, int FD, int COMP_IDX) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    
    long long k = labels[idx];
    int x_offset = idx * D;
    int mu_offset = k * D;
    
    // Access specific component indices
    int f_val_idx = idx * FD + COMP_IDX;
    int f_mu_idx = k * FD + COMP_IDX;
    int g_mu_idx = k * FD * D + COMP_IDX * D;

    double dot = 0.0;
    for (int r = 0; r < D; r++) {
        double diff = X[x_offset + r] - mus[mu_offset + r];
        dot += diff * g_mus[g_mu_idx + r];
    }
    
    double f_lin = f_mus[f_mu_idx] + dot;
    E_out[idx] = fabs(F_vals[f_val_idx] - f_lin);
}
"""

# Pointwise metrics (Radius, Lipschitz)
_metrics_kernel_code = r"""
extern "C" __global__
void fused_pointwise_metrics(
    const double* X, const double* G,
    const double* mus, const double* g_mus,
    const long long* labels,
    double* dist_out, double* ratio_out,
    int N, int D, int FD) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    
    long long k = labels[idx];
    int x_offset = idx * D;
    int mu_offset = k * D;

    double dist_sq = 0.0;
    for (int r = 0; r < D; r++) {
        double dx = X[x_offset + r] - mus[mu_offset + r];
        dist_sq += dx * dx;
    }
    double dist = sqrt(dist_sq);
    dist_out[idx] = dist;

    double max_ratio = 0.0;
    if (dist > 1e-12) {
        int g_idx_start = idx * FD * D;
        int g_mu_start = k * FD * D;

        for (int j = 0; j < FD; j++) {
            double dg_sq = 0.0;
            int row_offset_x = g_idx_start + j * D;
            int row_offset_mu = g_mu_start + j * D;
            
            for (int r = 0; r < D; r++) {
                double dg = G[row_offset_x + r] - g_mus[row_offset_mu + r];
                dg_sq += dg * dg;
            }
            double ratio = sqrt(dg_sq) / dist;
            if (ratio > max_ratio) max_ratio = ratio;
        }
    }
    ratio_out[idx] = max_ratio;
}
"""

# Hessian Statistics (Covariance C and Gradient Interaction E)
# Supports selecting a specific active component for E.
_stats_kernel_code = r"""
extern "C" __global__
void fused_hessian_stats(
    const double* X, const double* G, 
    const double* mus, const double* g_mus, 
    const long long* labels,
    const int* active_comps, 
    double* C_out, double* E_out,
    int N, int D, int FD, int K, bool calc_E) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    
    long long k = labels[idx];
    int mu_offset = k * D;
    int x_offset = idx * D;

    // 1. Covariance (Active Component independent)
    for (int r = 0; r < D; r++) {
        double dx_r = X[x_offset + r] - mus[mu_offset + r];
        for (int c = r; c < D; c++) {
            double dx_c = X[x_offset + c] - mus[mu_offset + c];
            double val_c = dx_r * dx_c;
            atomicAdd(&C_out[k*D*D + r*D + c], val_c);
            if (r != c) {
                atomicAdd(&C_out[k*D*D + c*D + r], val_c);
            }
        }
    }

    // 2. Gradient Interaction (E = sum dx dg^T) for the active component
    if (calc_E) {
        int active_j = active_comps[k]; 
        int g_row_x = (idx * FD * D) + (active_j * D);
        int g_row_mu = (k * FD * D) + (active_j * D);

        for (int r = 0; r < D; r++) {
            double dx_r = X[x_offset + r] - mus[mu_offset + r];
            for (int c = 0; c < D; c++) {
                double dg_c = G[g_row_x + c] - g_mus[g_row_mu + c];
                double val_e = dx_r * dg_c;
                atomicAdd(&E_out[k*D*D + r*D + c], val_e);
            }
        }
    }
}
"""

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
    int u_start = k * D * D;
    int l_start = k * D;
    double dist_sq = 0.0;
    for (int j = 0; j < D; j++) {
        double proj_j = 0.0;
        for (int r = 0; r < D; r++) {
            double dx_r = X[x_offset + r] - mus[mu_offset + r];
            proj_j += dx_r * U[u_start + r * D + j];
        }
        dist_sq += (proj_j * proj_j) * L_inv[l_start + j];
    }
    dist_out[idx] = dist_sq;
}
"""

# Compilation
_mod_taylor = xp.RawModule(code=_taylor_metric_kernel_code)
_kern_taylor = _mod_taylor.get_function("fused_taylor_metric")

_mod_taylor_comp = xp.RawModule(code=_taylor_comp_abs_kernel_code)
_kern_taylor_comp = _mod_taylor_comp.get_function("fused_taylor_component_abs")

_mod_metrics = xp.RawModule(code=_metrics_kernel_code)
_kern_metrics = _mod_metrics.get_function("fused_pointwise_metrics")

_mod_stats = xp.RawModule(code=_stats_kernel_code)
_kern_stats = _mod_stats.get_function("fused_hessian_stats")

_mod_proj = xp.RawModule(code=_proj_score_kernel_code)
_kern_proj = _mod_proj.get_function("fused_proj_score")

_mod_q = xp.RawModule(code=_q_kernel_code)
_kern_q = _mod_q.get_function("fused_q_dist")


# -----------------------------------------------------------------------------
# 2. WRAPPERS
# -----------------------------------------------------------------------------


def run_fused_taylor_metric(X, mus, labels, F_vals, f_mus, g_mus, scales, mode):
    """mode: 0=L2, 1=Linf"""
    N, D = X.shape
    FD = F_vals.shape[1]
    E_out = xp.empty((N,), dtype=xp.float64)
    labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)
    block = 256
    grid = (N + block - 1) // block
    scales_ptr = xp.ascontiguousarray(scales, dtype=xp.float64)
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
            scales_ptr,
            E_out,
            N,
            D,
            FD,
            int(mode),
        ),
    )
    return E_out


def run_fused_taylor_comp_abs(X, mus, labels, F_vals, f_mus, g_mus, comp_idx):
    N, D = X.shape
    FD = F_vals.shape[1]
    E_out = xp.empty((N,), dtype=xp.float64)
    labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)
    block = 256
    grid = (N + block - 1) // block
    _kern_taylor_comp(
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
            FD,
            int(comp_idx),
        ),
    )
    return E_out


def run_fused_metrics(X, G, mus, g_mus, labels):
    N, D = X.shape
    FD = G.shape[1]
    dist_out = xp.empty((N,), dtype=xp.float64)
    ratio_out = xp.empty((N,), dtype=xp.float64)
    labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)
    block = 256
    grid = (N + block - 1) // block
    _kern_metrics(
        (grid,),
        (block,),
        (
            xp.ascontiguousarray(X),
            xp.ascontiguousarray(G),
            xp.ascontiguousarray(mus),
            xp.ascontiguousarray(g_mus),
            labels_64,
            dist_out,
            ratio_out,
            N,
            D,
            FD,
        ),
    )
    return dist_out, ratio_out


def run_fused_stats(
    X, mus, labels, K, D, FD, active_comps, G=None, g_mus=None, calc_E=False
):
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
    ac_32 = xp.ascontiguousarray(active_comps, dtype=xp.int32)
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
            ac_32,
            C_out,
            E_out,
            N,
            D,
            FD,
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
# 3. UTILS
# -----------------------------------------------------------------------------


def flatten_function_array(F: Array) -> Array:
    F = xp.asarray(F)
    if F.ndim == 1:
        return F[:, None]
    return F


def flatten_gradient_array(G: Array) -> Array:
    G = xp.asarray(G)
    if G.ndim == 2:
        return G[:, None, :]
    return G


def allocate_output_buffers_cpu(
    MAX_CLUSTERS_EST: int, d: int, fd: int, store_hess: bool
) -> tuple[dict, int]:
    cap = max(1, int(MAX_CLUSTERS_EST))
    buffers = {
        "centroids": np.empty((cap, d), dtype=np.float64),
        "f": np.empty((cap, fd), dtype=np.float64),
        "g": np.empty((cap, fd, d), dtype=np.float64),
        "Q": np.empty((cap, d, d), dtype=np.float64),
        "E_cluster": np.empty((cap,), dtype=np.float64),
        "radii": np.empty((cap,), dtype=np.float64),
        "L_obs": np.empty((cap,), dtype=np.float64),
        "counts": np.empty((cap,), dtype=np.int64),
    }
    if store_hess:
        buffers["H_sqrt"] = np.empty((cap, d, d), dtype=np.float64)
    return buffers, cap


def adjust_buffer_capacity_cpu(
    buffers: dict,
    current_count: int,
    current_cap: int,
    required_total: int,
    store_hess: bool,
) -> tuple[dict, int]:
    if required_total <= current_cap:
        return buffers, current_cap
    new_cap = max(required_total, max(1, current_cap) * 2)
    d = int(buffers["centroids"].shape[1])
    fd = int(buffers["f"].shape[1])
    new_buffers = {
        "centroids": np.empty((new_cap, d), dtype=np.float64),
        "f": np.empty((new_cap, fd), dtype=np.float64),
        "g": np.empty((new_cap, fd, d), dtype=np.float64),
        "Q": np.empty((new_cap, d, d), dtype=np.float64),
        "E_cluster": np.empty((new_cap,), dtype=np.float64),
        "radii": np.empty((new_cap,), dtype=np.float64),
        "L_obs": np.empty((new_cap,), dtype=np.float64),
        "counts": np.empty((new_cap,), dtype=np.int64),
    }
    if store_hess:
        new_buffers["H_sqrt"] = np.empty((new_cap, d, d), dtype=np.float64)
    if current_count > 0:
        for k in new_buffers:
            new_buffers[k][:current_count] = buffers[k][:current_count]
    del buffers
    return new_buffers, int(new_cap)


# -----------------------------------------------------------------------------
# 4. MAIN ALGORITHM
# -----------------------------------------------------------------------------


def adaptive_clustering(
    X: Array,
    *,
    f_fn: Callable[[Array], Array],
    g_fn: Callable[[Array], Array],
    TOL: float,
    ERROR_METRIC: Literal["L2", "Linf-global", "Linf-local"] = "Linf-global",
    NUM_INITIAL_CLUSTERS: int = 100,
    MIN_PTS_PER_CLUSTER: int = 100,
    MAX_CLUSTERS_EST: int = 100_000,
    SEED: int = 0,
    Q_EPS_AXIS: float = 1e-8,
    Q_SHRINK: float = 0.05,
    H_RIDGE: float = 1e-18,
    MAX_LEVELS: int = 50,
    BATCH_SIZE: int = 10_000_000,
    LOG_MEM_STATS: bool = False,
    # --- New Flags ---
    USE_HESSIAN_INFORMED_SPLIT: bool = True,
    IGNORE_GRADIENT: bool = False,
    STORE_HESSIAN_SQRT_PER_CLUSTER: bool = False,
    SYNTH_BUDGET: Optional[int] = None,
    SYNTH_BUDGET_FACTOR: float = 50.0,
) -> dict[str, Any]:

    _start_time = time.perf_counter()
    mempool = cp.get_default_memory_pool()

    # --- 1. Init ---
    X0_gpu = xp.asarray(X, dtype=xp.float64)
    d = X0_gpu.shape[1]

    F0_gpu = flatten_function_array(f_fn(X0_gpu))  # (N, fd)
    fd = F0_gpu.shape[1]

    if IGNORE_GRADIENT:
        G0_gpu = xp.zeros((X0_gpu.shape[0], fd, d), dtype=xp.float64)
    else:
        G0_gpu = flatten_gradient_array(g_fn(X0_gpu))

    if LOG_MEM_STATS:
        log_mem_stats("After Init F/G")

    F_max_global = xp.max(xp.abs(F0_gpu), axis=0)  # (fd,)

    km = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
    labels0 = km.fit_predict(X0_gpu)

    sort_idx = xp.argsort(labels0)
    CLUSTER_PTS = X0_gpu[sort_idx]
    CLUSTER_F = F0_gpu[sort_idx]
    CLUSTER_G = G0_gpu[sort_idx]

    uniq_labels, CLUSTER_SIZES = xp.unique(labels0[sort_idx], return_counts=True)
    CLUSTER_STARTS = xp.r_[0, xp.cumsum(CLUSTER_SIZES)[:-1]]

    del X0_gpu, F0_gpu, G0_gpu, labels0, sort_idx, uniq_labels
    mempool.free_all_blocks()

    buffers, out_cap = allocate_output_buffers_cpu(
        MAX_CLUSTERS_EST, d, fd, STORE_HESSIAN_SQRT_PER_CLUSTER
    )
    out_count = 0

    level = 0
    while level <= MAX_LEVELS:
        if LOG_MEM_STATS:
            log_mem_stats(f"Start L{level}")
        num_clusters = len(CLUSTER_SIZES)
        if num_clusters == 0:
            break

        level_augmented_sizes = xp.maximum(CLUSTER_SIZES, MIN_PTS_PER_CLUSTER)
        level_total_aug_pts = int(xp.sum(level_augmented_sizes))
        aug_sizes_cpu = xp.asnumpy(level_augmented_sizes)
        aug_cumsum_cpu = np.cumsum(aug_sizes_cpu)

        next_level_X_list = []
        next_level_F_list = []
        next_level_G_list = []
        next_level_sizes_list = []

        num_batches = (level_total_aug_pts + BATCH_SIZE - 1) // BATCH_SIZE
        start_cluster_idx = 0

        for b in range(num_batches):
            target_aug_count = (b + 1) * BATCH_SIZE
            end_cluster_idx = int(np.searchsorted(aug_cumsum_cpu, target_aug_count))
            end_cluster_idx = min(end_cluster_idx + 1, num_clusters)
            if start_cluster_idx >= end_cluster_idx:
                break

            # Slice
            sizes = CLUSTER_SIZES[start_cluster_idx:end_cluster_idx]
            starts_local = CLUSTER_STARTS[start_cluster_idx:end_cluster_idx]
            slice_start = int(starts_local[0])
            slice_end = int(starts_local[-1] + sizes[-1])

            pts = CLUSTER_PTS[slice_start:slice_end]
            f_vals = CLUSTER_F[slice_start:slice_end]
            g_vals = CLUSTER_G[slice_start:slice_end]
            starts_local_shifted = starts_local - slice_start

            # Augment
            aug_sizes = level_augmented_sizes[start_cluster_idx:end_cluster_idx]
            aug_total = int(xp.sum(aug_sizes))

            X_batch = xp.empty((aug_total, d), dtype=xp.float64)
            F_batch = xp.empty((aug_total, fd), dtype=xp.float64)
            G_batch = xp.empty((aug_total, fd, d), dtype=xp.float64)

            aug_starts = xp.r_[0, xp.cumsum(aug_sizes)[:-1]]
            aug_ids = repeats(xp.arange(len(aug_sizes)), aug_sizes)

            pt_idxs_in_cluster = (
                xp.arange(aug_total, dtype=xp.int32) - aug_starts[aug_ids]
            )
            is_real = pt_idxs_in_cluster < sizes[aug_ids]
            X_batch[is_real] = pts
            F_batch[is_real] = f_vals
            G_batch[is_real] = g_vals

            is_synth = ~is_real
            num_synth = int(xp.sum(is_synth))
            if num_synth > 0:
                synth_c_ids = aug_ids[is_synth]
                rng = xp.random.default_rng(SEED + level + b)
                c_starts = starts_local_shifted[synth_c_ids]
                c_counts = sizes[synth_c_ids]

                rand1 = (rng.random(num_synth) * c_counts).astype(xp.int64)
                rand2 = (rng.random(num_synth) * (c_counts - 1)).astype(xp.int64) + 1
                idx1 = c_starts + rand1
                idx2 = c_starts + (rand1 + rand2) % c_counts

                p1 = pts[idx1]
                p2 = pts[idx2]
                alpha = rng.random((num_synth, 1)).astype(xp.float64)
                X_synth_vals = p1 * alpha + p2 * (1.0 - alpha)
                X_batch[is_synth] = X_synth_vals

                F_synth = flatten_function_array(f_fn(X_synth_vals))
                F_batch[is_synth] = F_synth
                if IGNORE_GRADIENT:
                    G_batch[is_synth] = 0.0
                else:
                    G_synth = flatten_gradient_array(g_fn(X_synth_vals))
                    G_batch[is_synth] = G_synth

                del p1, p2, alpha, X_synth_vals

            # --- Centroids ---
            sums = xp.add.reduceat(X_batch, aug_starts)
            mus = sums / aug_sizes[:, None]
            f_mus = flatten_function_array(f_fn(mus))
            if IGNORE_GRADIENT:
                g_mus = xp.zeros((len(aug_sizes), fd, d), dtype=xp.float64)
            else:
                g_mus = flatten_gradient_array(g_fn(mus))

            # --- Error Metrics ---
            K_batch = len(aug_sizes)
            dense_labels = aug_ids

            if ERROR_METRIC == "Linf-local":
                # Hybrid Acceptance: Relative < TOL OR Absolute < TOL
                # Iterate dimensions to compute cluster stats
                E_cluster_accum = xp.zeros((K_batch,), dtype=xp.float64)
                E_cluster_abs_max = xp.zeros((K_batch,), dtype=xp.float64)

                # Precompute Abs F Max per cluster for all dimensions
                # Since scatter_max works on vectors, we can't do it at once easily.
                # Loop is fine since fd is small (usually 1-100).

                for j in range(fd):
                    # 1. Abs F Max per cluster
                    F_max_j = xp.zeros((K_batch,), dtype=xp.float64)
                    cupy_scatter_max(F_max_j, dense_labels, xp.abs(F_batch[:, j]))

                    # 2. Absolute Taylor Error per point for component j
                    diff_pts_j = run_fused_taylor_comp_abs(
                        X_batch, mus, dense_labels, F_batch, f_mus, g_mus, j
                    )

                    # 3. Max Abs Error per cluster
                    E_abs_j = xp.zeros((K_batch,), dtype=xp.float64)
                    cupy_scatter_max(E_abs_j, dense_labels, diff_pts_j)

                    # 4. Relative Error per cluster
                    scale = xp.maximum(F_max_j, TINY_EPS)
                    E_rel_j = E_abs_j / scale

                    # 5. Accumulate Max across components
                    E_cluster_accum = xp.maximum(E_cluster_accum, E_rel_j)
                    E_cluster_abs_max = xp.maximum(E_cluster_abs_max, E_abs_j)

                is_good = (E_cluster_accum < TOL) | (E_cluster_abs_max < TOL)
                E_cluster = E_cluster_accum  # Store Rel error for output

            else:
                # L2 or Linf-global
                if ERROR_METRIC == "L2":
                    scales_ker = xp.ones((fd,), dtype=xp.float64)
                    metric_mode = 0
                else:
                    scales_ker = F_max_global
                    metric_mode = 1

                # Kernel returns:
                # Mode 0 (L2): Sum of squares of errors (over components) per point
                # Mode 1 (Linf): Max relative error (over components) per point
                err_pts = run_fused_taylor_metric(
                    X_batch,
                    mus,
                    dense_labels,
                    F_batch,
                    f_mus,
                    g_mus,
                    scales_ker,
                    metric_mode,
                )

                if ERROR_METRIC == "L2":
                    # Sum over points
                    sum_sq = xp.add.reduceat(err_pts, aug_starts)
                    # Correct L2 metric: Total Frobenius Norm
                    E_cluster = xp.sqrt(sum_sq)
                else:
                    # Max over points
                    E_cluster = xp.zeros((K_batch,), dtype=xp.float64)
                    cupy_scatter_max(E_cluster, dense_labels, err_pts)

                is_good = E_cluster < TOL

            # --- Good Clusters ---
            good_idx_local = xp.nonzero(is_good)[0]
            n_good = int(good_idx_local.shape[0])

            if n_good > 0:
                good_mask = is_good[dense_labels]
                X_good = X_batch[good_mask]
                G_good = G_batch[good_mask]
                good_counts = aug_sizes[good_idx_local]
                dense_labels_good = repeats(xp.arange(n_good), good_counts)

                radii_pt, ratio_pt = run_fused_metrics(
                    X_good,
                    G_good,
                    mus[good_idx_local],
                    g_mus[good_idx_local],
                    dense_labels_good,
                )
                radii = xp.zeros((n_good,), dtype=radii_pt.dtype)
                L_obs = xp.zeros((n_good,), dtype=ratio_pt.dtype)
                cupy_scatter_max(radii, dense_labels_good, radii_pt)
                cupy_scatter_max(L_obs, dense_labels_good, ratio_pt)

                # Q Matrix
                Cov, _ = run_fused_stats(
                    X_good,
                    mus[good_idx_local],
                    dense_labels_good,
                    n_good,
                    d,
                    fd,
                    xp.zeros(n_good, dtype=xp.int32),
                    calc_E=False,
                )
                Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))
                if Q_SHRINK > 0.0:
                    tr = xp.trace(Cov, axis1=1, axis2=2)
                    scale_tr = tr / max(d, 1)
                    I = xp.eye(d, dtype=Cov.dtype)[None, :, :]
                    Cov = (1.0 - Q_SHRINK) * Cov + Q_SHRINK * (
                        scale_tr[:, None, None] * I
                    )
                    Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

                Lambda, U = xp.linalg.eigh(Cov)
                Lambda_inv = 1.0 / xp.maximum(Lambda, Q_EPS_AXIS)
                dist_sq = run_fused_q_dist(
                    X_good, mus[good_idx_local], dense_labels_good, U, Lambda_inv
                )
                max_dist = xp.zeros((n_good,), dtype=dist_sq.dtype)
                cupy_scatter_max(max_dist, dense_labels_good, dist_sq)
                scale_fac = 1.0 / xp.maximum(max_dist, Q_EPS_AXIS)
                Qs = (U * (scale_fac[:, None] * Lambda_inv)[:, None, :]) @ U.transpose(
                    0, 2, 1
                )

                H_sqrt = None
                if STORE_HESSIAN_SQRT_PER_CLUSTER:
                    # Calc active component for good clusters (for storage)
                    # Reuse same logic as bad clusters (L2 Rel)
                    g_mus_exp = g_mus[good_idx_local][dense_labels_good]
                    dx = X_good - mus[good_idx_local][dense_labels_good]
                    lin = xp.einsum("nmd,nd->nm", g_mus_exp, dx)
                    diff_vals = F_batch[good_mask] - (
                        f_mus[good_idx_local][dense_labels_good] + lin
                    )

                    good_starts = xp.r_[0, xp.cumsum(good_counts)[:-1]]
                    sq_err = xp.add.reduceat(diff_vals**2, good_starts)
                    sq_val = xp.add.reduceat(F_batch[good_mask] ** 2, good_starts)
                    scores = xp.sqrt(sq_err) / (xp.sqrt(sq_val) + TINY_EPS)
                    active_comps_good = xp.argmax(scores, axis=1).astype(xp.int32)

                    C_h, E_h = run_fused_stats(
                        X_good,
                        mus[good_idx_local],
                        dense_labels_good,
                        n_good,
                        d,
                        fd,
                        active_comps_good,
                        G=G_good,
                        g_mus=g_mus[good_idx_local],
                        calc_E=True,
                    )
                    C_h = 0.5 * (C_h + C_h.transpose(0, 2, 1))
                    S_h = E_h + E_h.transpose(0, 2, 1)
                    dvals, U_h = xp.linalg.eigh(C_h)
                    dvals = xp.maximum(dvals, H_RIDGE)
                    Gtilde = U_h.transpose(0, 2, 1) @ S_h @ U_h
                    F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])
                    _, V_h = xp.linalg.eigh(F_mat)
                    Z = U_h @ V_h
                    M = (Z * xp.sqrt(xp.abs(_))[:, None, :]) @ Z.transpose(0, 2, 1)
                    H_sqrt = 0.5 * (M + M.transpose(0, 2, 1))

                end_out = out_count + n_good
                buffers, out_cap = adjust_buffer_capacity_cpu(
                    buffers, out_count, out_cap, end_out, STORE_HESSIAN_SQRT_PER_CLUSTER
                )
                buffers["centroids"][out_count:end_out] = xp.asnumpy(
                    mus[good_idx_local]
                )
                buffers["f"][out_count:end_out] = xp.asnumpy(f_mus[good_idx_local])
                buffers["g"][out_count:end_out] = xp.asnumpy(g_mus[good_idx_local])
                buffers["Q"][out_count:end_out] = xp.asnumpy(Qs)
                if H_sqrt is not None:
                    buffers["H_sqrt"][out_count:end_out] = xp.asnumpy(H_sqrt)
                buffers["E_cluster"][out_count:end_out] = xp.asnumpy(
                    E_cluster[good_idx_local]
                )
                buffers["radii"][out_count:end_out] = xp.asnumpy(radii)
                buffers["L_obs"][out_count:end_out] = xp.asnumpy(L_obs)
                buffers["counts"][out_count:end_out] = xp.asnumpy(
                    good_counts.astype(xp.int64)
                )
                out_count = end_out

            # --- Bad Clusters ---
            bad_idx_local = xp.nonzero(~is_good)[0]
            n_bad = int(bad_idx_local.shape[0])

            if n_bad > 0:
                bad_mask = (~is_good)[dense_labels]
                X_bad = X_batch[bad_mask]
                F_bad = F_batch[bad_mask]
                G_bad = G_batch[bad_mask]
                bad_counts = aug_sizes[bad_idx_local]
                dense_labels_bad = repeats(xp.arange(n_bad), bad_counts)

                if IGNORE_GRADIENT or not USE_HESSIAN_INFORMED_SPLIT:
                    M_split = xp.eye(d, dtype=xp.float64)[None, :, :].repeat(
                        n_bad, axis=0
                    )
                    C, _ = run_fused_stats(
                        X_bad,
                        mus[bad_idx_local],
                        dense_labels_bad,
                        n_bad,
                        d,
                        fd,
                        xp.zeros(n_bad, dtype=xp.int32),
                        calc_E=False,
                    )
                    C = 0.5 * (C + C.transpose(0, 2, 1))
                    CovZ = C
                else:
                    # Determine active comp for split (L2 Rel)
                    g_mus_exp_bad = g_mus[bad_idx_local][dense_labels_bad]
                    dx_bad = X_bad - mus[bad_idx_local][dense_labels_bad]
                    lin_bad = xp.einsum("nmd,nd->nm", g_mus_exp_bad, dx_bad)
                    diff_bad = F_bad - (
                        f_mus[bad_idx_local][dense_labels_bad] + lin_bad
                    )

                    bad_starts = xp.r_[0, xp.cumsum(bad_counts)[:-1]]
                    sq_err = xp.add.reduceat(diff_bad**2, bad_starts)
                    sq_val = xp.add.reduceat(F_bad**2, bad_starts)
                    scores = xp.sqrt(sq_err) / (xp.sqrt(sq_val) + TINY_EPS)
                    active_comps = xp.argmax(scores, axis=1).astype(xp.int32)

                    C, E = run_fused_stats(
                        X_bad,
                        mus[bad_idx_local],
                        dense_labels_bad,
                        n_bad,
                        d,
                        fd,
                        active_comps,
                        G=G_bad,
                        g_mus=g_mus[bad_idx_local],
                        calc_E=True,
                    )
                    C = 0.5 * (C + C.transpose(0, 2, 1))
                    S = E + E.transpose(0, 2, 1)

                    dvals, U = xp.linalg.eigh(C)
                    dvals = xp.maximum(dvals, H_RIDGE)
                    Gtilde = U.transpose(0, 2, 1) @ S @ U
                    F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])
                    _, V = xp.linalg.eigh(F_mat)
                    Z = U @ V
                    M_split = (Z * xp.sqrt(xp.abs(_))[:, None, :]) @ Z.transpose(
                        0, 2, 1
                    )
                    M_split = 0.5 * (M_split + M_split.transpose(0, 2, 1))
                    CovZ = M_split @ C @ M_split.transpose(0, 2, 1)

                _, V_z = xp.linalg.eigh(CovZ)
                v = V_z[:, :, -1]

                scores_proj = run_fused_score(
                    X_bad, mus[bad_idx_local], dense_labels_bad, M_split, v
                )
                sort_order = xp.lexsort(xp.stack([scores_proj, dense_labels_bad]))

                next_level_X_list.append(xp.asnumpy(X_bad[sort_order]))
                next_level_F_list.append(xp.asnumpy(F_bad[sort_order]))
                next_level_G_list.append(xp.asnumpy(G_bad[sort_order]))

                half = bad_counts // 2
                other = bad_counts - half
                new_sz = xp.zeros(2 * n_bad, dtype=xp.int64)
                new_sz[0::2] = half
                new_sz[1::2] = other
                next_level_sizes_list.append(xp.asnumpy(new_sz))

            start_cluster_idx = end_cluster_idx
            del X_batch, F_batch, G_batch
            mempool.free_all_blocks()

        if len(next_level_sizes_list) == 0:
            break

        next_X = np.concatenate(next_level_X_list)
        next_F = np.concatenate(next_level_F_list)
        next_G = np.concatenate(next_level_G_list)
        next_sizes = np.concatenate(next_level_sizes_list)

        valid = next_sizes > 0
        if not np.all(valid):
            keep = np.repeat(valid, next_sizes)
            CLUSTER_PTS = xp.asarray(next_X[keep])
            CLUSTER_F = xp.asarray(next_F[keep])
            CLUSTER_G = xp.asarray(next_G[keep])
            CLUSTER_SIZES = xp.asarray(next_sizes[valid])
        else:
            CLUSTER_PTS = xp.asarray(next_X)
            CLUSTER_F = xp.asarray(next_F)
            CLUSTER_G = xp.asarray(next_G)
            CLUSTER_SIZES = xp.asarray(next_sizes)

        CLUSTER_STARTS = xp.r_[0, xp.cumsum(CLUSTER_SIZES)[:-1]]

        del (
            next_level_X_list,
            next_level_F_list,
            next_level_G_list,
            next_level_sizes_list,
        )
        del next_X, next_F, next_G, next_sizes
        gc.collect()
        level += 1

    final_count = int(out_count)
    out_dict = {k: buffers[k][:final_count] for k in buffers}
    out_dict["num_levels"] = level
    logger.info(
        f"Complete: {final_count} clusters. Time: {time.perf_counter()-_start_time:.2f}s"
    )
    return out_dict


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
    ERROR_METRIC = args.error_metric
    A_ROSENBROCK = args.rosenbrock_a
    OUTPUT_SUBDIR = args.output_subdir
    FORCE_REEXEC = args.force

    # Check if file already exists
    AC_PLOTS_DIR = AC_PLOTS_DIR / OUTPUT_SUBDIR if OUTPUT_SUBDIR else AC_PLOTS_DIR
    AC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    img_fpath = (
        AC_PLOTS_DIR
        / f"adaptive_clustering_rosenbrock_{A_ROSENBROCK}_tol{TOL}_error_metric_{ERROR_METRIC}_init{NUM_INITIAL_CLUSTERS}_min{MIN_PTS_PER_CLUSTER}.png"
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
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS * DIM,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        ERROR_METRIC=ERROR_METRIC,
        BATCH_SIZE=100_000_000,
    )

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
            f"Rosenbrock (a={A_ROSENBROCK}) (TOL={TOL}, ERROR_METRIC={ERROR_METRIC}), N_CLUSTERS={len(centroids):,}",
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
    xp.savez_compressed(npz_fpath, **results)
