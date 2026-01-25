import gc
import logging
import time
from typing import Any, Callable, Literal, Optional

import cupy as cp
import numpy as np
from cuml.cluster import KMeans as cuKMeans
from cupyx import scatter_max as cupy_scatter_max

from src import IS_CUPY_BACKEND, Array, xp
from src.logging_config import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)

TINY_EPS = 1e-12

# -----------------------------------------------------------------------------
# 1. CUDA KERNELS (Unchanged)
# -----------------------------------------------------------------------------
# [Keeping your optimized dense kernels]

_taylor_kernel_sq_code = r"""
extern "C" __global__
void fused_taylor_sq(
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
    double diff = F_vals[idx] - f_lin;
    E_out[idx] = diff * diff; 
}
"""

_taylor_kernel_abs_code = r"""
extern "C" __global__
void fused_taylor_abs(
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
    E_out[idx] = fabs(F_vals[idx] - f_lin);
}
"""

_metrics_kernel_code = r"""
extern "C" __global__
void fused_pointwise_metrics(
    const double* X, const double* G,
    const double* mus, const double* g_mus,
    const long long* labels,
    double* dist_out, double* ratio_out,
    int N, int D) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    long long k = labels[idx];
    int mu_offset = k * D;
    int x_offset = idx * D;
    double dist_sq = 0.0;
    double diff_g_sq = 0.0;
    for (int j = 0; j < D; j++) {
        double dx = X[x_offset + j] - mus[mu_offset + j];
        double dg = G[x_offset + j] - g_mus[mu_offset + j];
        dist_sq += dx * dx;
        diff_g_sq += dg * dg;
    }
    double dist = sqrt(dist_sq);
    dist_out[idx] = dist;
    if (dist > 1e-12) {
        ratio_out[idx] = sqrt(diff_g_sq) / dist;
    } else {
        ratio_out[idx] = 0.0;
    }
}
"""

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
        for (int c = r; c < D; c++) {
            double dx_c = X[x_offset + c] - mus[mu_offset + c];
            double val_c = dx_r * dx_c;
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

_mod_taylor_sq = xp.RawModule(code=_taylor_kernel_sq_code)
_kern_taylor_sq = _mod_taylor_sq.get_function("fused_taylor_sq")
_mod_taylor_abs = xp.RawModule(code=_taylor_kernel_abs_code)
_kern_taylor_abs = _mod_taylor_abs.get_function("fused_taylor_abs")
_mod_metrics = xp.RawModule(code=_metrics_kernel_code)
_kern_metrics = _mod_metrics.get_function("fused_pointwise_metrics")
_mod_stats = xp.RawModule(code=_stats_kernel_code)
_kern_stats = _mod_stats.get_function("fused_hessian_stats")
_mod_proj = xp.RawModule(code=_proj_score_kernel_code)
_kern_proj = _mod_proj.get_function("fused_proj_score")
_mod_q = xp.RawModule(code=_q_kernel_code)
_kern_q = _mod_q.get_function("fused_q_dist")

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------


def run_fused_taylor_error(X, mus, labels, F_vals, f_mus, g_mus, mode="abs"):
    N, D = X.shape
    E_out = xp.empty((N,), dtype=xp.float64)
    labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)
    block = 256
    grid = (N + block - 1) // block
    kernel = _kern_taylor_sq if mode == "sq" else _kern_taylor_abs
    kernel(
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


def run_fused_metrics(X, G, mus, g_mus, labels):
    N, D = X.shape
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
        ),
    )
    return dist_out, ratio_out


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


def allocate_output_buffers_cpu(MAX_CLUSTERS_EST: int, d: int) -> tuple[dict, int]:
    cap = max(1, int(MAX_CLUSTERS_EST))
    buffers = {
        "centroids": np.empty((cap, d), dtype=np.float64),
        "f": np.empty((cap,), dtype=np.float64),
        "g": np.empty((cap, d), dtype=np.float64),
        "Q": np.empty((cap, d, d), dtype=np.float64),
        "E_cluster": np.empty((cap,), dtype=np.float64),
        "radii": np.empty((cap,), dtype=np.float64),
        "L_obs": np.empty((cap,), dtype=np.float64),
        "counts": np.empty((cap,), dtype=np.int64),
    }
    return buffers, cap


def adjust_buffer_capacity_cpu(
    buffers: dict, current_count: int, current_cap: int, required_total: int
) -> tuple[dict, int]:
    if required_total <= current_cap:
        return buffers, current_cap
    new_cap = max(required_total, max(1, current_cap) * 2)
    logger.warning(
        f"Resizing CPU output buffers: {current_cap:,} -> {new_cap:,} clusters"
    )
    d = int(buffers["centroids"].shape[1])
    new_buffers = {
        "centroids": np.empty((new_cap, d), dtype=np.float64),
        "f": np.empty((new_cap,), dtype=np.float64),
        "g": np.empty((new_cap, d), dtype=np.float64),
        "Q": np.empty((new_cap, d, d), dtype=np.float64),
        "E_cluster": np.empty((new_cap,), dtype=np.float64),
        "radii": np.empty((new_cap,), dtype=np.float64),
        "L_obs": np.empty((new_cap,), dtype=np.float64),
        "counts": np.empty((new_cap,), dtype=np.int64),
    }
    if current_count > 0:
        for k in new_buffers:
            new_buffers[k][:current_count] = buffers[k][:current_count]
    del buffers
    return new_buffers, int(new_cap)


# -----------------------------------------------------------------------------
# 3. MAIN ALGORITHM (Using Sort-Based Offloading)
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
    SYNTH_BUDGET: Optional[int] = None,
    SYNTH_BUDGET_FACTOR: float = 1.0,
    Q_EPS_AXIS: float = 1e-8,
    Q_SHRINK: float = 0.05,
    H_RIDGE: float = 1e-18,
    MAX_LEVELS: int = 50,
    BATCH_SIZE: int = 5_000_000,
) -> dict[str, Any]:

    _start_time = time.perf_counter()
    mempool = cp.get_default_memory_pool()

    # 1. Initialize First Level on GPU (Assuming Level 0 fits in VRAM)
    # If Level 0 is too big for VRAM, we can't start easily anyway.
    X0_gpu = xp.asarray(X, dtype=xp.float64)
    N0, d = X0_gpu.shape

    # Eval f/g for initial points
    F0_gpu = flatten_function_array(f_fn(X0_gpu))
    G0_gpu = flatten_gradient_array(g_fn(X0_gpu))

    F_inf = float(xp.max(xp.abs(F0_gpu)))

    # KMeans
    km = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
    labels0 = km.fit_predict(X0_gpu)

    # --- PHYSICAL SORTING (The key from the old code) ---
    # We group points by cluster physically in memory
    sort_idx = xp.argsort(labels0)

    # Primary GPU Buffers (These will hold ONE level's data)
    CLUSTER_PTS = X0_gpu[sort_idx]
    CLUSTER_F = F0_gpu[sort_idx]
    CLUSTER_G = G0_gpu[sort_idx]

    # Calculate initial sizes/starts
    uniq_labels, CLUSTER_SIZES = xp.unique(labels0[sort_idx], return_counts=True)
    CLUSTER_STARTS = xp.r_[0, xp.cumsum(CLUSTER_SIZES)[:-1]]

    del X0_gpu, F0_gpu, G0_gpu, labels0, sort_idx, uniq_labels
    mempool.free_all_blocks()

    # Output Buffers (CPU)
    buffers, out_cap = allocate_output_buffers_cpu(MAX_CLUSTERS_EST, d)
    out_count = 0

    level = 0
    while level <= MAX_LEVELS:
        log_mem_stats(f"Start L{level}")

        num_clusters = len(CLUSTER_SIZES)
        if num_clusters == 0:
            break

        total_pts_level = int(xp.sum(CLUSTER_SIZES))

        # --- Augmentation (Synthetic Points) ---
        # We calculate augmentation requirements on GPU, but handle memory carefully
        level_augmented_sizes = xp.maximum(CLUSTER_SIZES, MIN_PTS_PER_CLUSTER)
        level_total_aug_pts = int(xp.sum(level_augmented_sizes))

        # Helper: CPU cumulative sums for batching
        aug_sizes_cpu = xp.asnumpy(level_augmented_sizes)
        aug_cumsum_cpu = np.cumsum(aug_sizes_cpu)

        # Host buffers for next level (The "Offload Strategy")
        next_level_X_list = []
        next_level_F_list = []
        next_level_G_list = []
        next_level_sizes_list = (
            []
        )  # Store sizes on CPU or GPU? GPU is fine for small array.

        # --- BATCH LOOP ---
        num_batches = (level_total_aug_pts + BATCH_SIZE - 1) // BATCH_SIZE
        start_cluster_idx = 0

        for b in range(num_batches):
            target_aug_count = (b + 1) * BATCH_SIZE
            end_cluster_idx = int(np.searchsorted(aug_cumsum_cpu, target_aug_count))
            end_cluster_idx = min(end_cluster_idx + 1, num_clusters)

            if start_cluster_idx >= end_cluster_idx:
                break

            # Slice Clusters
            sizes = CLUSTER_SIZES[start_cluster_idx:end_cluster_idx]
            starts_local = CLUSTER_STARTS[start_cluster_idx:end_cluster_idx]

            # Map to points in CLUSTER_PTS
            slice_start = int(starts_local[0])
            slice_end = int(starts_local[-1] + sizes[-1])

            # Zero-copy slice (Points are already contiguous!)
            pts = CLUSTER_PTS[slice_start:slice_end]
            f_vals = CLUSTER_F[slice_start:slice_end]
            g_vals = CLUSTER_G[slice_start:slice_end]

            # Adjust starts to be local to this batch (0-based)
            starts_local_shifted = starts_local - slice_start

            # --- AUGMENTATION ---
            # We construct a temporary 'aug_pts' buffer for this batch only
            aug_sizes = level_augmented_sizes[start_cluster_idx:end_cluster_idx]
            aug_total = int(xp.sum(aug_sizes))

            # Allocate Batch Buffers
            X_batch = xp.empty((aug_total, d), dtype=xp.float64)
            F_batch = xp.empty((aug_total,), dtype=xp.float64)
            G_batch = xp.empty((aug_total, d), dtype=xp.float64)

            # Helper indices
            aug_starts = xp.r_[0, xp.cumsum(aug_sizes)[:-1]]
            aug_ids = repeats(xp.arange(len(aug_sizes)), aug_sizes)

            # Fill Real Data
            # Vectorized fill:
            # We need to map 'pts' (packed real) to 'X_batch' (packed augmented)
            # This requires a scatter or kernel because 'gaps' are created for synth points

            # Create boolean mask for real points in X_batch
            pt_idxs_in_cluster = (
                xp.arange(aug_total, dtype=xp.int32) - aug_starts[aug_ids]
            )
            is_real = pt_idxs_in_cluster < sizes[aug_ids]

            X_batch[is_real] = pts
            F_batch[is_real] = f_vals
            G_batch[is_real] = g_vals

            # Fill Synthetic
            is_synth = ~is_real
            num_synth = int(xp.sum(is_synth))

            if num_synth > 0:
                synth_c_ids = aug_ids[is_synth]
                # To generate synth points, we need random pairs from real points
                # s_starts = starts_local_shifted[synth_c_ids]
                # s_sizes = sizes[synth_c_ids]

                # Using simple logic from your old code:
                # Pick 2 random indices within the cluster limits
                rng = xp.random.default_rng(SEED + level + b)

                # Pointers relative to 'pts' array
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

                # Eval F/G for synth
                F_synth_vals = flatten_function_array(f_fn(X_synth_vals))
                G_synth_vals = flatten_gradient_array(g_fn(X_synth_vals))
                F_batch[is_synth] = F_synth_vals
                G_batch[is_synth] = G_synth_vals

                del p1, p2, alpha, X_synth_vals, F_synth_vals, G_synth_vals

            # --- PROCESS BATCH ---
            # Now we have X_batch, F_batch, G_batch fully populated and grouped by cluster

            # 1. Centroids
            sums = xp.add.reduceat(X_batch, aug_starts)
            mus = sums / aug_sizes[:, None]
            f_mus = flatten_function_array(f_fn(mus))
            g_mus = flatten_gradient_array(g_fn(mus))

            # 2. Error
            K_batch = len(aug_sizes)
            dense_labels = aug_ids  # Already computed

            if ERROR_METRIC == "L2":
                err_sq = run_fused_taylor_error(
                    X_batch, mus, dense_labels, F_batch, f_mus, g_mus, mode="sq"
                )
                # reduceat is safe for contiguous segments
                sum_sq = xp.add.reduceat(err_sq, aug_starts)
                E_cluster = xp.sqrt(sum_sq)
                is_good = E_cluster < TOL
            elif ERROR_METRIC == "Linf-global":
                err_abs = run_fused_taylor_error(
                    X_batch, mus, dense_labels, F_batch, f_mus, g_mus, mode="abs"
                )
                # Fallback to scatter for max (reduceat max not reliable) OR use simple loop if fast enough
                # Actually, using cupyx.scatter_max is reliable
                E_abs = xp.zeros((K_batch,), dtype=err_abs.dtype)
                cupy_scatter_max(E_abs, dense_labels, err_abs)
                E_cluster = E_abs / F_inf
                is_good = E_cluster < TOL
            elif ERROR_METRIC == "Linf-local":
                err_abs = run_fused_taylor_error(
                    X_batch, mus, dense_labels, F_batch, f_mus, g_mus, mode="abs"
                )
                E_abs = xp.zeros((K_batch,), dtype=err_abs.dtype)
                cupy_scatter_max(E_abs, dense_labels, err_abs)

                F_max = xp.zeros((K_batch,), dtype=F_batch.dtype)
                cupy_scatter_max(F_max, dense_labels, xp.abs(F_batch))
                E_cluster = E_abs / xp.maximum(F_max, TINY_EPS)
                is_good = (E_cluster < TOL) | (E_abs < TOL)

            # 3. Handle Good (Download)
            good_idx_local = xp.nonzero(is_good)[0]
            n_good = int(good_idx_local.shape[0])

            if n_good > 0:
                good_mask = is_good[dense_labels]
                X_good = X_batch[good_mask]
                G_good = G_batch[good_mask]

                good_counts = aug_sizes[good_idx_local]
                # Need dense labels for metrics relative to X_good
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

                Cov, _ = run_fused_stats(
                    X_good,
                    mus[good_idx_local],
                    dense_labels_good,
                    n_good,
                    d,
                    calc_E=False,
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
                dist_sq = run_fused_q_dist(
                    X_good, mus[good_idx_local], dense_labels_good, U, Lambda_inv
                )
                max_dist = xp.zeros((n_good,), dtype=dist_sq.dtype)
                cupy_scatter_max(max_dist, dense_labels_good, dist_sq)
                scale_fac = 1.0 / xp.maximum(max_dist, Q_EPS_AXIS)
                final_D = scale_fac[:, None] * Lambda_inv
                Qs = (U * final_D[:, None, :]) @ U.transpose(0, 2, 1)

                # STORE ON CPU
                end_out = out_count + n_good
                buffers, out_cap = adjust_buffer_capacity_cpu(
                    buffers, out_count, out_cap, end_out
                )

                buffers["centroids"][out_count:end_out] = xp.asnumpy(
                    mus[good_idx_local]
                )
                buffers["f"][out_count:end_out] = xp.asnumpy(f_mus[good_idx_local])
                buffers["g"][out_count:end_out] = xp.asnumpy(g_mus[good_idx_local])
                buffers["Q"][out_count:end_out] = xp.asnumpy(Qs)
                buffers["E_cluster"][out_count:end_out] = xp.asnumpy(
                    E_cluster[good_idx_local]
                )
                buffers["radii"][out_count:end_out] = xp.asnumpy(radii)
                buffers["L_obs"][out_count:end_out] = xp.asnumpy(L_obs)
                buffers["counts"][out_count:end_out] = xp.asnumpy(
                    good_counts.astype(xp.int64)
                )
                out_count = end_out

            # 4. Handle Bad (Split & Stream to CPU)
            bad_idx_local = xp.nonzero(~is_good)[0]
            n_bad = int(bad_idx_local.shape[0])

            if n_bad > 0:
                bad_mask = (~is_good)[dense_labels]
                X_bad = X_batch[bad_mask]
                F_bad = F_batch[bad_mask]
                G_bad = G_batch[bad_mask]

                bad_counts = aug_sizes[bad_idx_local]
                dense_labels_bad = repeats(xp.arange(n_bad), bad_counts)

                # Hessian Split Logic
                C, E = run_fused_stats(
                    X_bad,
                    mus[bad_idx_local],
                    dense_labels_bad,
                    n_bad,
                    d,
                    G=G_bad,
                    g_mus=g_mus[bad_idx_local],
                    calc_E=True,
                )
                C = 0.5 * (C + C.transpose(0, 2, 1))
                S = E + E.transpose(0, 2, 1)
                dvals, U = xp.linalg.eigh(C)
                dvals = xp.maximum(dvals, H_RIDGE)
                Gtilde = U.transpose(0, 2, 1) @ S @ U
                denom = dvals[:, :, None] + dvals[:, None, :]
                F_mat = Gtilde / denom
                _, V = xp.linalg.eigh(F_mat)
                Z_rot = U @ V
                w_abs_sqrt = xp.sqrt(xp.abs(_))
                M = (Z_rot * w_abs_sqrt[:, None, :]) @ Z_rot.transpose(0, 2, 1)
                M = 0.5 * (M + M.transpose(0, 2, 1))
                CovZ = M @ C @ M.transpose(0, 2, 1)
                _, V_z = xp.linalg.eigh(CovZ)
                v = V_z[:, :, -1]

                scores = run_fused_score(
                    X_bad, mus[bad_idx_local], dense_labels_bad, M, v
                )

                # We need to sort X_bad by (Cluster, Score) to effectively split
                # But physically sorting inside batch is simpler:
                # We can just use the boolean mask logic from median split to send points to Left/Right queues

                # Re-calculate starts for X_bad to compute medians
                bad_starts = xp.r_[0, xp.cumsum(bad_counts)[:-1]]

                # Need per-cluster median
                # scores are not sorted by score, they are grouped by cluster.
                # We need to compute median per cluster.
                # Just use quickselect or sort per cluster?
                # Faster: Just global sort by (Cluster, Score)
                # Since X_bad is already grouped by cluster, we just need to sort by score STABLE-ly
                # But cp.argsort is stable? No.
                # Simple hack: Sort using (ClusterID * LargeNumber + Score)
                # But scores are float.

                # Standard trick: Lexsort
                # keys = (dense_labels_bad, scores)
                # But cp.lexsort keys order is reversed: (scores, dense_labels_bad)
                # This sorts primarily by cluster, secondarily by score

                sort_order = xp.lexsort(xp.stack([scores, dense_labels_bad]))

                X_bad_sorted = X_bad[sort_order]
                F_bad_sorted = F_bad[sort_order]
                G_bad_sorted = G_bad[sort_order]

                # Split counts
                half_counts = bad_counts // 2
                other_counts = bad_counts - half_counts

                # We need to interleave these for the next level: [L0, R0, L1, R1...]
                # This ensures the physical layout remains (Cluster 0 Children, Cluster 1 Children...)
                # Actually, simpler: Just dump all Left children then all Right children?
                # No, we want DFS/BFS order.
                # The easiest way to preserve "Group by Cluster" for next level is to just
                # dump X_bad_sorted into the next level list.
                # AND record the new sizes: [half_0, other_0, half_1, other_1...]

                # Construct new sizes array
                new_sizes = xp.zeros(2 * n_bad, dtype=xp.int64)
                new_sizes[0::2] = half_counts
                new_sizes[1::2] = other_counts

                # OFFLOAD TO HOST
                next_level_X_list.append(xp.asnumpy(X_bad_sorted))
                next_level_F_list.append(xp.asnumpy(F_bad_sorted))
                next_level_G_list.append(xp.asnumpy(G_bad_sorted))
                next_level_sizes_list.append(xp.asnumpy(new_sizes))

            # Prepare next batch
            start_cluster_idx = end_cluster_idx

            # Cleanup GPU Batch
            del X_batch, F_batch, G_batch
            mempool.free_all_blocks()

        # --- END OF LEVEL ---
        if len(next_level_sizes_list) == 0:
            break

        # Prepare Next Level Data
        # Concatenate on Host
        next_X_host = np.concatenate(next_level_X_list)
        next_F_host = np.concatenate(next_level_F_list)
        next_G_host = np.concatenate(next_level_G_list)
        next_sizes_host = np.concatenate(next_level_sizes_list)

        # Filter empty
        valid = next_sizes_host > 0
        if not np.all(valid):
            # This is tricky because points correspond to sizes implicitly by order
            # We must filter points too.
            # But wait, did we generate 0-size clusters?
            # half_counts = size // 2. If size=1, half=0, other=1.
            # So yes, 0 sizes possible.
            # Reconstruct keep mask for points

            # Expansion of 'valid' to points
            # repeat 'valid' by 'next_sizes_host'
            keep_points = np.repeat(valid, next_sizes_host)

            CLUSTER_SIZES = xp.asarray(next_sizes_host[valid])

            # Upload Filtered Points
            # If too big, this might fail. But we just cleared VRAM.
            # Assuming active set fits in VRAM.
            # If not, we'd need to keep this on Host and stream slices.
            # Given previous error, let's assume it might be tight but usually active set shrinks?
            # Actually, active set grows (augmentation).
            # So we MUST check capacity.

            req_bytes = next_X_host.nbytes + next_F_host.nbytes + next_G_host.nbytes
            # Quick check: 60M points * 10 dim * 8 bytes ~ 4.8GB.
            # 60M * 100 dim => 48GB.
            # With 10D, we are fine uploading all to GPU.

            CLUSTER_PTS = xp.asarray(next_X_host[keep_points])
            CLUSTER_F = xp.asarray(next_F_host[keep_points])
            CLUSTER_G = xp.asarray(next_G_host[keep_points])
        else:
            CLUSTER_SIZES = xp.asarray(next_sizes_host)
            CLUSTER_PTS = xp.asarray(next_X_host)
            CLUSTER_F = xp.asarray(next_F_host)
            CLUSTER_G = xp.asarray(next_G_host)

        CLUSTER_STARTS = xp.r_[0, xp.cumsum(CLUSTER_SIZES)[:-1]]

        # Cleanup Host
        del (
            next_level_X_list,
            next_level_F_list,
            next_level_G_list,
            next_level_sizes_list,
        )
        del next_X_host, next_F_host, next_G_host, next_sizes_host
        gc.collect()

        level += 1

    final_count = int(out_count)
    out_dict = {k: buffers[k][:final_count] for k in buffers}
    out_dict["num_levels"] = level

    logger.info(f"Adaptive clustering complete: {final_count:,} final clusters.")
    logger.info(f"Runtime - Total: {time.perf_counter() -_start_time:.3f}s")

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
        SYNTH_BUDGET_FACTOR=10,
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
