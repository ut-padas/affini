import logging
import time

import cupy as cp
import numpy as np

from src import IS_CUPY_BACKEND, Array, xp

logger = logging.getLogger(__name__)

TINY_EPS = 1e-12

# -----------------------------------------------------------------------------
# 1. CUDA KERNELS
# -----------------------------------------------------------------------------
# Kernel: fused_taylor_metric
# -----------------------------------------------------------------------------
# Computes the error between the true function value and the first-order
# Taylor approximation at the cluster centroid.
#
# Math:
#   f_lin_j(x) = f_j(mu) + dot(x - mu, g_j(mu))
#   diff_j     = F_vals[j] - f_lin_j
#
# Modes:
#   METRIC_MODE == 0 (L2): Returns sum of squared errors across components.
#       Out[i] = sum_j (diff_j)^2
#   METRIC_MODE == 1 (Linf): Returns max relative error across components.
#       Out[i] = max_j ( |diff_j| / scale[j] )
#
# Dimensions:
#   N: Number of points
#   D: Input dimension
#   FD: Output dimension (Function Dimension)
# -----------------------------------------------------------------------------
_taylor_metric_kernel_code = r"""
extern "C" __global__
void fused_taylor_metric(
    const double* X,        // (N, D) Input points
    const double* mus,      // (K, D) Centroids
    const long long* labels,// (N,)   Cluster assignment indices
    const double* F_vals,   // (N, FD) True function values
    const double* f_mus,    // (K, FD) Function values at centroids
    const double* g_mus,    // (K, FD, D) Gradients at centroids
    const double* scales,   // (FD,) Normalization scales (for Linf)
    double* E_out,          // (N,) Output error metric
    int N, int D, int FD, int METRIC_MODE) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    
    long long k = labels[idx];
    
    // Offset calculations
    int x_offset = idx * D;
    int mu_offset = k * D;
    int f_idx_start = idx * FD;
    int f_mu_start = k * FD;
    int g_mu_start = k * FD * D;

    double val_acc = 0.0; 

    for (int j = 0; j < FD; j++) {
        // 1. Compute dot product: (x - mu)^T * g_j(mu)
        double dot = 0.0;
        int g_row_start = g_mu_start + j * D;
        for (int r = 0; r < D; r++) {
            double diff = X[x_offset + r] - mus[mu_offset + r];
            dot += diff * g_mus[g_row_start + r];
        }
        
        // 2. Linear prediction
        double f_lin = f_mus[f_mu_start + j] + dot;
        double diff = F_vals[f_idx_start + j] - f_lin;
        double abs_diff = fabs(diff);

        // 3. Accumulate based on metric mode
        if (METRIC_MODE == 0) {
            // L2: Sum of squares
            val_acc += diff * diff;
        } else {
            // Linf: Max Relative Error
            double s = scales[j]; 
            double rel = abs_diff / (s + 1e-12);
            if (rel > val_acc) val_acc = rel;
        }
    }
    E_out[idx] = val_acc;
}
"""

# Kernel: fused_taylor_component_abs
# -----------------------------------------------------------------------------
# Computes the absolute Taylor error for a SINGLE specific output component.
# Used for Linf-local metric where normalization logic is complex and per-cluster.
#
# Math: E_out[i] = | f_c(x) - (f_c(mu) + g_c(mu)^T (x-mu)) |
# -----------------------------------------------------------------------------
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
    
    // Offsets for the specific component COMP_IDX
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

# Kernel: fused_pointwise_metrics
# -----------------------------------------------------------------------------
# Computes geometry metrics for the trust region.
# 1. Euclidean distance from centroid: |x - mu|
# 2. Local Lipschitz ratio estimate: max_j ( |g_j(x) - g_j(mu)| / |x - mu| )
# -----------------------------------------------------------------------------
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

    // 1. Distance
    double dist_sq = 0.0;
    for (int r = 0; r < D; r++) {
        double dx = X[x_offset + r] - mus[mu_offset + r];
        dist_sq += dx * dx;
    }
    double dist = sqrt(dist_sq);
    dist_out[idx] = dist;

    // 2. Lipschitz Ratio (Max over FD components)
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

# Kernel: fused_hessian_stats
# -----------------------------------------------------------------------------
# Accumulates the sufficient statistics matrices C and E for Hessian estimation.
# Uses atomicAdd to sum contributions from all points in a cluster.
#
# Math:
#   C_k = sum_{x in C_k} (x - mu)(x - mu)^T
#   E_k = sum_{x in C_k} (x - mu)(g_active(x) - g_active(mu))^T
#
# Supports "Active Component" selection: For vector functions, we estimate the
# Hessian of the specific output component that dominates the local error.
# -----------------------------------------------------------------------------
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

# Kernel: fused_proj_score
# -----------------------------------------------------------------------------
# Projects points onto the splitting direction vector v in the transformed space M.
# Used to determine cluster splits (median split).
#
# Math: Score = v^T * M * (x - mu)
# -----------------------------------------------------------------------------
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
    
    // Matrix-vector multiplication: z = M * (x - mu)
    // Then dot product: score = v^T * z
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

# Kernel: fused_q_dist
# -----------------------------------------------------------------------------
# Computes the Ellipsoidal distance for the Trust Ellipsoid Q.
#
# Math: dist^2 = (x - mu)^T Q (x - mu)
#       where Q = U * diag(L_inv) * U^T
# -----------------------------------------------------------------------------
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
        // Project x onto j-th eigenvector
        double proj_j = 0.0;
        for (int r = 0; r < D; r++) {
            double dx_r = X[x_offset + r] - mus[mu_offset + r];
            proj_j += dx_r * U[u_start + r * D + j];
        }
        // Scale by inverse square of extent along this axis
        dist_sq += (proj_j * proj_j) * L_inv[l_start + j];
    }
    dist_out[idx] = dist_sq;
}
"""

# Compilation of RawModules
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
# 2. PYTHON WRAPPERS
# -----------------------------------------------------------------------------
def run_fused_taylor_metric(X, mus, labels, F_vals, f_mus, g_mus, scales, mode):
    """
    Calculates aggregated Taylor error per point (Sum-sq or Max-Rel).
    Mode: 0 = L2 (Sum of Squares), 1 = Linf (Max Relative)
    """
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
    """Calculates Absolute Taylor error for a single specific output component."""
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
    """Calculates point radii and Lipschitz ratio estimates."""
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
    """Accumulates Covariance (C) and Gradient Interaction (E) matrices."""
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
    """Projects points onto split direction."""
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
    """Calculates Ellipsoidal distance for trust ellipsoids."""
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
class CudaTimer:
    """Helper context manager for timing GPU operations with synchronization."""

    def __init__(self, key: str, stats: dict, enabled: bool):
        self.key = key
        self.stats = stats
        self.enabled = enabled
        self.t0 = None

    def __enter__(self):
        if self.enabled:
            cp.cuda.Device().synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.enabled:
            cp.cuda.Device().synchronize()
            dt = time.perf_counter() - self.t0  # type: ignore
            self.stats[self.key] += dt


def convert_to_2d_array(F: Array) -> Array:
    """Ensures function values are (N, fd) shape."""
    F = xp.asarray(F)
    if F.ndim == 1:
        return F[:, None]
    return F


def convert_to_3d_array(G: Array) -> Array:
    """Ensures gradients are (N, fd, d) shape."""
    G = xp.asarray(G)
    if G.ndim == 2:
        return G[:, None, :]
    return G


def allocate_output_buffers_cpu(
    MAX_CLUSTERS_EST: int, d: int, fd: int, store_hess: bool
) -> tuple[dict, int]:
    """Allocates host (CPU) memory for final clusters."""
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
    """Doubles the capacity of output buffers if required."""
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


def get_n_smallest_per_block(values, start_indices, n):
    # 1. Create an array of block IDs for every element
    # Example: starts [0, 3, 7] -> block_ids [0, 0, 0, 1, 1, 1, 1, 2, 2...]
    block_ids = xp.zeros(values.size, dtype=xp.int32)
    # Mark the start of each block with a 1
    block_ids[start_indices] = 1
    # Cumulative sum turns markers into IDs: [1, 1, 1, 2, 2, 2, 2, 3, 3...]
    block_ids = xp.cumsum(block_ids) - 1

    # 2. Sort indices: primary key = block_ids, secondary key = values
    # This groups blocks together and sorts values within them
    sort_idx = xp.lexsort((values, block_ids))

    sorted_values = values[sort_idx]
    sorted_blocks = block_ids[sort_idx]

    # 3. Pick the first 'n' elements of each block
    # Generate local indices (0, 1, 2... for each block)
    # We do this by finding where the block ID changes
    block_change = xp.diff(sorted_blocks, prepend=-1)

    # Use a clever trick: the index within the block is
    # the distance from the last 'change' point
    # For speed, we can use a scan or just compute local counts
    # A simple way: find the rank of each element within its block
    local_idx = xp.arange(len(sorted_values))
    # To get local index 0, 1, 2... within blocks:
    # We subtract the global index of the first element of each block
    first_elem_indices = xp.where(block_change != 0)[0]
    # Expand the start indices to match the full array size
    offsets = xp.zeros_like(sorted_values)
    offsets[first_elem_indices] = first_elem_indices
    offsets = xp.maximum.accumulate(offsets)

    intra_block_idx = local_idx - offsets

    # 4. Mask and extract
    mask = intra_block_idx < n
    return mask
