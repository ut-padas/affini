import gc
import logging
import time
from typing import Any, Callable, Literal, Optional

import numpy as np
from cuml.cluster import KMeans as cuKMeans
from cupyx import scatter_add as cupy_scatter_add
from cupyx import scatter_max as cupy_scatter_max
from sklearn.cluster import KMeans

from src import IS_CUPY_BACKEND, Array, xp
from src.logging_utils import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)

TINY_EPS = 1e-12

# -----------------------------------------------------------------------------
# 1. CUDA KERNELS (Your Custom Kernels)
# -----------------------------------------------------------------------------

# Kernel for L2 Error: computes (f - f_lin)^2
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

# Kernel for Linf Error: computes |f - f_lin|
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

# Kernel for Metrics: computes |dx| and |dg|/|dx|
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

    // Avoid division by zero
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

        // Covariance C (Symmetric): dx * dx.T
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

# Compilation
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
# 2. TYPE-SAFE WRAPPERS
# -----------------------------------------------------------------------------


def run_fused_taylor_error(X, mus, labels, F_vals, f_mus, g_mus, mode="abs"):
    """Computes per-point Taylor error. mode='abs' -> |f - f_lin|, mode='sq' -> (f - f_lin)^2"""
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
    """Computes per-point dist=|x-mu| and ratio=|g-gmu|/|x-mu|."""
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


# -----------------------------------------------------------------------------
# 3. UTILS
# -----------------------------------------------------------------------------


def scatter_maximum_1d(out: Array, idx: Array, vals: Array) -> None:
    if IS_CUPY_BACKEND:
        cupy_scatter_max(out, idx, vals)
    else:
        xp.maximum.at(out, idx, vals)


def scatter_add_1d(out: Array, idx: Array, vals: Array) -> None:
    if IS_CUPY_BACKEND:
        cupy_scatter_add(out, idx, vals)
    else:
        xp.add.at(out, idx, vals)


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
        "E_cluster": xp.empty((cap,), dtype=dtype),
        "radii": xp.empty((cap,), dtype=dtype),
        "L_obs": xp.empty((cap,), dtype=dtype),
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
        "E_cluster": xp.empty((new_cap,), dtype=dtype),
        "radii": xp.empty((new_cap,), dtype=dtype),
        "L_obs": xp.empty((new_cap,), dtype=dtype),
        "counts": xp.empty((new_cap,), dtype=xp.int64),
    }
    if current_count > 0:
        for k in new_buffers:
            new_buffers[k][:current_count] = buffers[k][:current_count]
    del buffers
    return new_buffers, int(new_cap)


def generate_sythetic_points_batched(
    X_pool: Array,
    indices: Array,
    starts: Array,
    counts: Array,
    uniq_labels: Array,
    MIN_PTS_PER_CLUSTER: int,
    *,
    SEED: int,
    level: int,
) -> tuple[Optional[Array], Optional[Array]]:
    """Generates synthetic points by only materializing required clusters."""
    # Only process clusters that need padding
    need = xp.maximum(0, MIN_PTS_PER_CLUSTER - counts)
    total_synth = int(xp.sum(need))
    if total_synth == 0:
        return None, None

    # Identify which clusters need padding
    pad_mask = need > 0
    if not xp.any(pad_mask):
        return None, None

    # Extract just the relevant indices to minimize memory
    # We construct a mask of points belonging to clusters that need padding
    # This might still be large if many clusters need padding, but better than full sort

    rng = xp.random.default_rng(SEED + level)
    k_synth = repeats(xp.arange(int(need.shape[0])), need)  # Indices into uniq_labels
    labels_synth = uniq_labels[k_synth]

    # For pair selection:
    # m = count of points in cluster k
    # s0 = start index in the *global sorted* list

    # We need to fetch X values.
    # To avoid materializing X_sorted for ALL points, we can do it just for the pairs.
    # But picking random pairs efficiently without sorting is hard.
    # Strategy: Since we passed `indices` which are pointers to X_pool sorted by label,
    # we can pick indices directly from that array.

    m = counts[k_synth]
    s0 = starts[k_synth]

    rand = rng.random
    i = (xp.floor(rand(total_synth) * m)).astype(xp.int64)
    j = (xp.floor(rand(total_synth) * (m - 1))).astype(xp.int64)
    j = j + (j >= i)

    # Indices into the `indices` array (which maps to X_pool)
    ptr_i = s0 + i
    ptr_j = s0 + j

    # Global indices in X_pool
    idx_i = indices[ptr_i]
    idx_j = indices[ptr_j]

    # Now we only materialize the pairs!
    X_i = X_pool[idx_i]
    X_j = X_pool[idx_j]

    a = rand((total_synth, 1)).astype(X_pool.dtype)
    X_synth = a * X_i + (1.0 - a) * X_j

    return X_synth, labels_synth.astype(xp.int64)


# -----------------------------------------------------------------------------
# 4. MAIN ROUTINE WITH BATCHING
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
    BATCH_SIZE: int = 10_000_000,  # Max points to materialize in VRAM at once
) -> dict[str, Array]:

    X0 = xp.asarray(X, dtype=xp.float64)
    N0, d = X0.shape

    if SYNTH_BUDGET is None:
        SYNTH_BUDGET = int(max(0, SYNTH_BUDGET_FACTOR * N0))

    capacity = int(N0 + SYNTH_BUDGET)
    X_pool = xp.empty((capacity, d), dtype=X0.dtype)
    X_pool[:N0] = X0
    n_total = int(N0)
    log_mem_stats("After pool allocation")

    _t0 = time.perf_counter()
    F_pool = xp.empty((capacity,), dtype=X0.dtype)
    G_pool = xp.empty((capacity, d), dtype=X0.dtype)
    F_pool[:N0] = flatten_function_array(f_fn(X_pool[:N0]))
    G_pool[:N0] = flatten_gradient_array(g_fn(X_pool[:N0]))
    logger.info(f"Runtime - Initial f/g eval: {time.perf_counter()-_t0:.3f}s")
    log_mem_stats("After f/g eval")

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
        log_mem_stats(f"Start Level {level:03}")
        if synth_added_prev_level:
            F_inf = xp.max(xp.abs(F_pool[:n_total]))

        idx_active = xp.nonzero(active[:n_total])[0]
        n_active = int(idx_active.shape[0])
        if n_active == 0:
            break

        synth_added_this_level = False

        # We need to group points by label to identify clusters
        # NOTE: Sorting indices is cheap (integer array). Materializing X is expensive.
        label_active = labels_pool[idx_active]
        perm = xp.argsort(label_active)
        idx_sorted = idx_active[perm]  # Indices into X_pool, grouped by cluster
        label_sorted = label_active[perm]

        uniq_labels, starts, counts = compute_label_segments(label_sorted)
        K_total = int(uniq_labels.shape[0])

        # ---- Synthetic Points (Optimized: No full materialization) ----
        # Checks counts to see if any padding is needed
        if xp.any(counts < MIN_PTS_PER_CLUSTER):
            X_synth, labels_synth = generate_sythetic_points_batched(
                X_pool,
                idx_sorted,
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

                if n_plus_M > capacity:
                    new_capacity = max(n_plus_M, int(capacity * 1.5))
                    logger.warning(
                        f"Pool capacity exceeded. Resizing: {capacity} -> {new_capacity}"
                    )
                    # Standard Resize logic
                    new_X = xp.empty((new_capacity, d), dtype=X_pool.dtype)
                    new_X[:n_total] = X_pool[:n_total]
                    del X_pool
                    gc.collect()
                    X_pool = new_X
                    new_G = xp.empty((new_capacity, d), dtype=G_pool.dtype)
                    new_G[:n_total] = G_pool[:n_total]
                    del G_pool
                    gc.collect()
                    G_pool = new_G
                    new_F = xp.empty((new_capacity,), dtype=F_pool.dtype)
                    new_F[:n_total] = F_pool[:n_total]
                    del F_pool
                    F_pool = new_F
                    new_L = xp.empty((new_capacity,), dtype=labels_pool.dtype)
                    new_L[:n_total] = labels_pool[:n_total]
                    del labels_pool
                    labels_pool = new_L
                    new_A = xp.zeros((new_capacity,), dtype=active.dtype)
                    new_A[:n_total] = active[:n_total]
                    del active
                    active = new_A
                    capacity = new_capacity
                    gc.collect()

                X_pool[n_total:n_plus_M] = X_synth
                labels_pool[n_total:n_plus_M] = labels_synth
                active[n_total:n_plus_M] = True
                F_pool[n_total:n_plus_M] = flatten_function_array(f_fn(X_synth))
                G_pool[n_total:n_plus_M] = flatten_gradient_array(g_fn(X_synth))
                n_total += M
                synth_added_this_level = True
                log_mem_stats("After synthetic addition")

                # Re-sort indices after addition
                idx_active = xp.nonzero(active[:n_total])[0]
                label_active = labels_pool[idx_active]
                perm = xp.argsort(label_active)
                idx_sorted = idx_active[perm]
                label_sorted = label_active[perm]
                uniq_labels, starts, counts = compute_label_segments(label_sorted)
                K_total = int(uniq_labels.shape[0])

        # ---------------------------------------------------------------------
        # MAIN BATCH LOOP
        # Processing clusters in chunks to fit in VRAM
        # ---------------------------------------------------------------------

        current_k = 0
        while current_k < K_total:
            # 1. Determine Batch Size
            # Accumulate counts until we hit BATCH_SIZE points
            # This logic avoids splitting a single cluster across batches
            next_k = current_k
            points_in_batch = 0

            # 1. Determine Batch Size
            remaining_counts = counts[current_k:]
            cum_counts = xp.cumsum(remaining_counts)

            # FIX: Explicitly cast BATCH_SIZE to a cupy array
            batch_target = xp.array(BATCH_SIZE, dtype=xp.int64)
            split_loc = xp.searchsorted(cum_counts, batch_target, side="right")

            if split_loc == 0:
                # Even the first cluster is bigger than batch size, must take at least one
                batch_k_count = 1
            else:
                batch_k_count = int(split_loc)

            end_k = current_k + batch_k_count

            # 2. Extract Batch Data
            batch_counts = counts[current_k:end_k]
            batch_start_idx = starts[current_k]
            batch_end_idx = starts[current_k] + xp.sum(
                batch_counts
            )  # Indices in idx_sorted

            # Pointers to X_pool
            indices_batch = idx_sorted[batch_start_idx:batch_end_idx]

            # MATERIALIZE BATCH (This is the only large allocation)
            X_batch = X_pool[indices_batch]

            # Prepare metadata for batch (local clustering)
            K_batch = end_k - current_k
            # dense_labels for scatter operations within batch (0..K_batch-1)
            dense_labels = repeats(xp.arange(K_batch), batch_counts)
            # Recompute starts relative to batch
            batch_starts_local = xp.r_[0, xp.cumsum(batch_counts)[:-1]]

            # 3. Compute Centroids
            sums = xp.add.reduceat(X_batch, batch_starts_local)
            mus = sums / batch_counts[:, None]
            f_mus = flatten_function_array(f_fn(mus))
            g_mus = flatten_gradient_array(g_fn(mus))

            # 4. Error Calculation
            if ERROR_METRIC == "L2":
                err_sq_point = run_fused_taylor_error(
                    X_batch,
                    mus,
                    dense_labels,
                    F_pool[indices_batch],
                    f_mus,
                    g_mus,
                    mode="sq",
                )
                sum_sq = xp.zeros((K_batch,), dtype=err_sq_point.dtype)
                scatter_add_1d(sum_sq, dense_labels, err_sq_point)
                E_cluster = xp.sqrt(sum_sq)
                is_good = E_cluster < TOL

            elif ERROR_METRIC == "Linf-global":
                err_abs_point = run_fused_taylor_error(
                    X_batch,
                    mus,
                    dense_labels,
                    F_pool[indices_batch],
                    f_mus,
                    g_mus,
                    mode="abs",
                )
                E_abs = xp.zeros((K_batch,), dtype=err_abs_point.dtype)
                scatter_maximum_1d(E_abs, dense_labels, err_abs_point)
                E_cluster = E_abs / F_inf
                is_good = E_cluster < TOL

            elif ERROR_METRIC == "Linf-local":
                err_abs_point = run_fused_taylor_error(
                    X_batch,
                    mus,
                    dense_labels,
                    F_pool[indices_batch],
                    f_mus,
                    g_mus,
                    mode="abs",
                )
                E_abs = xp.zeros((K_batch,), dtype=err_abs_point.dtype)
                scatter_maximum_1d(E_abs, dense_labels, err_abs_point)

                # Fetch F locally
                F_batch_vals = F_pool[indices_batch]
                F_max_cluster = xp.zeros((K_batch,), dtype=F_batch_vals.dtype)
                scatter_maximum_1d(F_max_cluster, dense_labels, xp.abs(F_batch_vals))

                normalization = xp.maximum(F_max_cluster, TINY_EPS)
                E_cluster = E_abs / normalization
                is_good = (E_cluster < TOL) | (E_abs < TOL)
            else:
                raise ValueError(f"Unknown metric {ERROR_METRIC}")

            # 5. Handle Good Clusters
            good_mask_batch = is_good
            good_idx_local = xp.nonzero(good_mask_batch)[0]
            n_good = int(good_idx_local.shape[0])

            if n_good > 0:
                # Mask relative to batch
                points_good_mask = good_mask_batch[dense_labels]

                # Subset X_batch for Q/Metrics calc
                X_good = X_batch[points_good_mask]
                G_good = G_pool[indices_batch][points_good_mask]

                good_counts_local = batch_counts[good_idx_local]
                dense_labels_good = repeats(xp.arange(n_good), good_counts_local)

                # Metrics
                radii_point, ratio_point = run_fused_metrics(
                    X_good,
                    G_good,
                    mus[good_idx_local],
                    g_mus[good_idx_local],
                    dense_labels_good,
                )
                radii = xp.zeros((n_good,), dtype=radii_point.dtype)
                L_obs = xp.zeros((n_good,), dtype=ratio_point.dtype)
                scatter_maximum_1d(radii, dense_labels_good, radii_point)
                scatter_maximum_1d(L_obs, dense_labels_good, ratio_point)

                # Q Calc
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
                max_dist_sq = xp.zeros((n_good,), dtype=dist_sq.dtype)
                scatter_maximum_1d(max_dist_sq, dense_labels_good, dist_sq)
                scale_fac = 1.0 / xp.maximum(max_dist_sq, Q_EPS_AXIS)
                final_D = scale_fac[:, None] * Lambda_inv
                Qs = (U * final_D[:, None, :]) @ U.transpose(0, 2, 1)

                # Write to global buffers
                end_out = out_count + n_good
                buffers, out_cap = adjust_buffer_capacity(
                    buffers, out_count, out_cap, end_out
                )
                buffers["centroids"][out_count:end_out] = mus[good_idx_local]
                buffers["f"][out_count:end_out] = f_mus[good_idx_local]
                buffers["g"][out_count:end_out] = g_mus[good_idx_local]
                buffers["Q"][out_count:end_out] = Qs
                buffers["E_cluster"][out_count:end_out] = E_cluster[good_idx_local]
                buffers["radii"][out_count:end_out] = radii
                buffers["L_obs"][out_count:end_out] = L_obs
                buffers["counts"][out_count:end_out] = good_counts_local.astype(
                    xp.int64
                )
                out_count = end_out

                # Deactivate good points globally
                indices_good_global = indices_batch[points_good_mask]
                active[indices_good_global] = False

                del X_good, G_good

            # 6. Handle Bad Clusters (Split)
            bad_mask_batch = ~good_mask_batch
            bad_idx_local = xp.nonzero(bad_mask_batch)[0]
            n_bad = int(bad_idx_local.shape[0])

            if n_bad > 0:
                points_bad_mask = bad_mask_batch[dense_labels]
                X_bad = X_batch[points_bad_mask]
                # Need G_batch for splitting
                G_bad = G_pool[indices_batch][points_bad_mask]

                bad_counts_local = batch_counts[bad_idx_local]
                dense_labels_bad = repeats(xp.arange(n_bad), bad_counts_local)

                # Stats
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

                h_eigvals, V = xp.linalg.eigh(F_mat)
                Z_rot = U @ V
                w_abs_sqrt = xp.sqrt(xp.abs(h_eigvals))
                M = (Z_rot * w_abs_sqrt[:, None, :]) @ Z_rot.transpose(0, 2, 1)
                M = 0.5 * (M + M.transpose(0, 2, 1))
                CovZ = M @ C @ M.transpose(0, 2, 1)
                _, V_z = xp.linalg.eigh(CovZ)
                v = V_z[:, :, -1]

                scores = run_fused_score(
                    X_bad, mus[bad_idx_local], dense_labels_bad, M, v
                )

                split_right = compute_median_split_decision(
                    scores,
                    dense_labels_bad,
                    xp.r_[0, xp.cumsum(bad_counts_local)[:-1]],
                    bad_counts_local,
                )

                # Update Labels Locally then Write Global
                # get original labels from pool for these clusters
                # uniq_labels is the list of ALL active labels.
                # We need the labels corresponding to the bad clusters in this batch
                # batch_labels = uniq_labels[current_k:end_k]
                bad_global_labels = uniq_labels[current_k:end_k][bad_idx_local]

                old_per_point = bad_global_labels[dense_labels_bad]
                new_labels = old_per_point * 2 + split_right.astype(xp.int64)

                indices_bad_global = indices_batch[points_bad_mask]
                labels_pool[indices_bad_global] = new_labels

                del X_bad, G_bad

            # Cleanup Batch
            del X_batch
            # gc.collect() # Optional: might slow down loop, usually del is enough for CuPy memory pool

            # Advance loop
            current_k = end_k

        # End of Level
        synth_added_prev_level = synth_added_this_level
        logger.info(
            f"Level {level:03}: active={n_active:,}, pool_used={n_total/capacity:.1%}"
        )
        level += 1
        log_mem_stats(f"End Level {level:03}")

    final_count = int(out_count)
    out_dict = {k: buffers[k][:final_count] for k in buffers}
    out_dict["num_levels"] = level
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
        SYNTH_BUDGET_FACTOR=10,
        BATCH_SIZE=1_000_000,
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
