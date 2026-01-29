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
# Computes the Mahalanobis-like distance for the Trust Ellipsoid Q.
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
    Q_SHRINK: float = 0.05,
    MAX_LEVELS: int = 50,
    BATCH_SIZE: int = 10_000_000,
    LOG_MEM_STATS: bool = False,
    USE_HESSIAN_INFORMED_SPLIT: bool = True,
    IGNORE_GRADIENT: bool = False,
    STORE_HESSIAN_SQRT_PER_CLUSTER: bool = False,
) -> dict[str, Any]:
    """
    Adaptive Clustering for Vector-Valued Functions (GPU Accelerated).

    Constructs a piecewise-local linear surrogate model for a function F: R^d \to R^fd.
    The algorithm recursively splits clusters where the local first-order Taylor approximation fails to meet the specified `TOL`.

    Architecture (Sort-Streaming):
    To handle datasets larger than GPU memory, this implementation uses a "Sort-Streaming" approach:
    1.  **Physical Sort:** Points are sorted by cluster label in VRAM, allowing zero-copy slicing.
    2.  **Streaming:** The current level is processed in batches on the GPU.
    3.  **Offloading:** Accepted ("Good") clusters are immediately moved to CPU host memory.
    4.  **Refinement:** Rejected ("Bad") clusters are split and streamed to a CPU staging buffer
        for the next level.

    **Error Metrics:**
    Let E(x) = F(x) - (F(μ_k) + ∇F(μ_k)(x - μ_k)).
    -   L2: RMSE-style check. √(∑_{x ∈ C_k} ‖E(x)‖_F^2) < TOL.
        (Strict total energy check).
    -   Linf-global: Max relative error normalized by global range.
        max_{x ∈ C_k} max_j |E_j(x)| / max_y |F_j(y)| < TOL.
    -   Linf-local: Max relative error normalized by local cluster range.
        max_{x ∈ C_k} max_j |E_j(x)| / max_{y ∈ C_k} |F_j(y)| < TOL.
        Includes hybrid acceptance (Absolute < TOL) to handle zero-crossings.

    Args:
        X: Input training points. Shape (N, d).
        f_fn: Callable returning function values. Output shape (N, fd).
        g_fn: Callable returning gradients/Jacobians. Output shape (N, fd, d).
        TOL: Error tolerance for cluster acceptance.
        ERROR_METRIC: Metric for acceptance ('L2', 'Linf-global', 'Linf-local').
        NUM_INITIAL_CLUSTERS: Number of seeds for initial K-Means.
        MIN_PTS_PER_CLUSTER: Minimum points per cluster. If a cluster has fewer,
            synthetic points are generated via convex combinations to stabilize geometry.
        MAX_CLUSTERS_EST: Initial capacity for CPU output buffers (auto-expands).
        SEED: Random seed for initialization and synthetic generation.
        Q_SHRINK: Shrinkage factor a for covariance: Σ ← (1-a)Σ + a (Tr(Σ)/d)I.
        MAX_LEVELS: Maximum recursion depth for splitting.
        BATCH_SIZE: Maximum number of points processed in a single GPU kernel launch.
        LOG_MEM_STATS: If True, prints VRAM usage at key steps.
        USE_HESSIAN_INFORMED_SPLIT: If True, splits clusters along the dominant
        eigenvector of the local Hessian estimate. If False, splits along the principal
        axis (PCA) of points.
        IGNORE_GRADIENT: If True, assumes ∇F = 0 (Voronoi/Piecewise-Constant
        approximation).
        STORE_HESSIAN_SQRT_PER_CLUSTER: If True, computes and stores |H|^{1/2} for every
        accepted cluster (useful for anisotropic trust regions downstream).

    Returns:
        A dictionary containing the surrogate model parameters:
        -   `centroids`: (K, d) Cluster centers.
        -   `f`: (K, fd) Function values at centroids.
        -   `g`: (K, fd, d) Gradients/Jacobians at centroids.
        -   `Q`: (K, d, d) Trust region shape matrices (inverse covariance).
        -   `radii`: (K,) Trust region radii.
        -   `L_obs`: (K,) Observed Lipschitz constants.
        -   `E_cluster`: (K,) Max observed error per cluster.
        -   `counts`: (K,) Number of points support per cluster.
        -   `H_sqrt`: (K, d, d) (Optional) Symmetric Hessian square roots.
        -   `num_levels`: (int) Total levels refined.
    """

    _start_time = time.perf_counter()
    mempool = cp.get_default_memory_pool()

    # --- 1. Initialization Phase ---
    # Upload initial data to GPU
    X0_gpu = xp.asarray(X, dtype=xp.float64)  # Shape: (N, d)
    d = X0_gpu.shape[1]

    # Evaluate physics
    F0_gpu = convert_to_2d_array(f_fn(X0_gpu))  # Shape: (N, fd)
    fd = F0_gpu.shape[1]

    if IGNORE_GRADIENT:
        G0_gpu = xp.zeros((X0_gpu.shape[0], fd, d), dtype=xp.float64)
    else:
        G0_gpu = convert_to_3d_array(g_fn(X0_gpu))  # Shape: (N, fd, d)

    if LOG_MEM_STATS:
        log_mem_stats("After Init F/G")

    # Compute Global Scale for Linf-global metric: S_j = max_x |F_j(x)|
    F_max_global = xp.max(xp.abs(F0_gpu), axis=0)  # Shape: (fd,)

    # Initial K-Means Clustering
    km = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
    labels0 = km.fit_predict(X0_gpu)

    # --- Physical Sort ---
    # Group data by cluster in VRAM to enable contiguous slicing later
    sort_idx = xp.argsort(labels0)
    CLUSTER_PTS = X0_gpu[sort_idx]  # (N, d)
    CLUSTER_F = F0_gpu[sort_idx]  # (N, fd)
    CLUSTER_G = G0_gpu[sort_idx]  # (N, fd, d)

    uniq_labels, CLUSTER_SIZES = xp.unique(labels0[sort_idx], return_counts=True)
    CLUSTER_STARTS = xp.r_[0, xp.cumsum(CLUSTER_SIZES)[:-1]]

    # Free temp buffers
    del X0_gpu, F0_gpu, G0_gpu, labels0, sort_idx, uniq_labels
    mempool.free_all_blocks()

    # Allocate Output Buffers on CPU (Host) to save VRAM
    buffers, out_cap = allocate_output_buffers_cpu(
        MAX_CLUSTERS_EST, d, fd, STORE_HESSIAN_SQRT_PER_CLUSTER
    )
    out_count = 0

    # --- 2. Main Refinement Loop (BFS) ---
    level = 0
    while level <= MAX_LEVELS:
        if LOG_MEM_STATS:
            log_mem_stats(f"Start L{level}")
        num_clusters = len(CLUSTER_SIZES)
        if num_clusters == 0:
            break

        # Calculate augmentation needs (how many synthetic points per cluster?)
        level_augmented_sizes = xp.maximum(CLUSTER_SIZES, MIN_PTS_PER_CLUSTER)
        level_total_aug_pts = int(xp.sum(level_augmented_sizes))

        # CPU view for batch scheduling
        aug_sizes_cpu = xp.asnumpy(level_augmented_sizes)
        aug_cumsum_cpu = np.cumsum(aug_sizes_cpu)

        # Host Staging Buffers for the NEXT level (Streaming architecture)
        next_level_X_list = []
        next_level_F_list = []
        next_level_G_list = []
        next_level_sizes_list = []

        # --- Batch Processing Loop ---
        # Process clusters in chunks that fit in GPU memory
        num_batches = (level_total_aug_pts + BATCH_SIZE - 1) // BATCH_SIZE
        start_cluster_idx = 0

        for b in range(num_batches):
            # Determine cluster range for this batch
            target_aug_count = (b + 1) * BATCH_SIZE
            end_cluster_idx = int(np.searchsorted(aug_cumsum_cpu, target_aug_count))
            end_cluster_idx = min(end_cluster_idx + 1, num_clusters)
            if start_cluster_idx >= end_cluster_idx:
                break

            # 1. Zero-Copy Slicing from GPU Arrays
            sizes = CLUSTER_SIZES[start_cluster_idx:end_cluster_idx]
            starts_local = CLUSTER_STARTS[start_cluster_idx:end_cluster_idx]
            slice_start = int(starts_local[0])
            slice_end = int(starts_local[-1] + sizes[-1])

            pts = CLUSTER_PTS[slice_start:slice_end]
            f_vals = CLUSTER_F[slice_start:slice_end]
            g_vals = CLUSTER_G[slice_start:slice_end]
            starts_local_shifted = starts_local - slice_start

            # 2. Augmentation (Synthetic Points via Convex Combinations)
            aug_sizes = level_augmented_sizes[start_cluster_idx:end_cluster_idx]
            aug_total = int(xp.sum(aug_sizes))

            X_batch = xp.empty((aug_total, d), dtype=xp.float64)
            F_batch = xp.empty((aug_total, fd), dtype=xp.float64)
            G_batch = xp.empty((aug_total, fd, d), dtype=xp.float64)

            # Map augmented indices to local cluster IDs
            aug_starts = xp.r_[0, xp.cumsum(aug_sizes)[:-1]]
            aug_ids = repeats(xp.arange(len(aug_sizes)), aug_sizes)

            # Fill Real Data
            pt_idxs_in_cluster = (
                xp.arange(aug_total, dtype=xp.int32) - aug_starts[aug_ids]
            )
            is_real = pt_idxs_in_cluster < sizes[aug_ids]
            X_batch[is_real] = pts
            F_batch[is_real] = f_vals
            G_batch[is_real] = g_vals

            # Fill Synthetic Data
            is_synth = ~is_real
            num_synth = int(xp.sum(is_synth))
            if num_synth > 0:
                synth_c_ids = aug_ids[is_synth]
                rng = xp.random.default_rng(SEED + level + b)

                # Vectorized random pair selection within clusters
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

                # Evaluate Physics for Synth Points
                F_synth = convert_to_2d_array(f_fn(X_synth_vals))
                F_batch[is_synth] = F_synth
                if IGNORE_GRADIENT:
                    G_batch[is_synth] = 0.0
                else:
                    G_synth = convert_to_3d_array(g_fn(X_synth_vals))
                    G_batch[is_synth] = G_synth

                del p1, p2, alpha, X_synth_vals

            # 3. Compute Centroids & Reference Gradients
            sums = xp.add.reduceat(X_batch, aug_starts)
            mus = sums / aug_sizes[:, None]  # (K_batch, d)
            f_mus = convert_to_2d_array(f_fn(mus))  # (K_batch, fd)
            if IGNORE_GRADIENT:
                g_mus = xp.zeros((len(aug_sizes), fd, d), dtype=xp.float64)
            else:
                g_mus = convert_to_3d_array(g_fn(mus))  # (K_batch, fd, d)

            # 4. Error Metrics & Acceptance Logic
            K_batch = len(aug_sizes)
            dense_labels = aug_ids

            if ERROR_METRIC == "Linf-local":
                # Special Case: Hybrid Acceptance (Rel < TOL || Abs < TOL)
                E_cluster_accum = xp.zeros((K_batch,), dtype=xp.float64)
                E_cluster_abs_max = xp.zeros((K_batch,), dtype=xp.float64)

                # Loop over dimensions to avoid materializing (N, FD) diff tensor
                for j in range(fd):
                    # Local Scale: max |f_j| in cluster
                    F_max_j = xp.zeros((K_batch,), dtype=xp.float64)
                    cupy_scatter_max(F_max_j, dense_labels, xp.abs(F_batch[:, j]))

                    # Taylor Error (Absolute)
                    diff_pts_j = run_fused_taylor_comp_abs(
                        X_batch, mus, dense_labels, F_batch, f_mus, g_mus, j
                    )

                    # Max Error per cluster
                    E_abs_j = xp.zeros((K_batch,), dtype=xp.float64)
                    cupy_scatter_max(E_abs_j, dense_labels, diff_pts_j)

                    # Normalize
                    scale = xp.maximum(F_max_j, TINY_EPS)
                    E_rel_j = E_abs_j / scale

                    # Accumulate Max
                    E_cluster_accum = xp.maximum(E_cluster_accum, E_rel_j)
                    E_cluster_abs_max = xp.maximum(E_cluster_abs_max, E_abs_j)

                is_good = (E_cluster_accum < TOL) | (E_cluster_abs_max < TOL)
                E_cluster = E_cluster_accum  # Log relative error

            else:
                # Standard Metrics (L2 or Linf-global)
                if ERROR_METRIC == "L2":
                    scales_ker = xp.ones((fd,), dtype=xp.float64)
                    metric_mode = 0  # Sum Squares
                else:
                    scales_ker = F_max_global
                    metric_mode = 1  # Max Relative

                # err_pts: (N,) aggregated error per point
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
                    # Sum squared errors over points -> sqrt (Frobenius Norm)
                    sum_sq = xp.add.reduceat(err_pts, aug_starts)
                    E_cluster = xp.sqrt(sum_sq)
                else:
                    # Max relative error over points
                    E_cluster = xp.zeros((K_batch,), dtype=xp.float64)
                    cupy_scatter_max(E_cluster, dense_labels, err_pts)

                is_good = E_cluster < TOL

            # 5. Handle Accepted ("Good") Clusters
            good_idx_local = xp.nonzero(is_good)[0]
            n_good = int(good_idx_local.shape[0])

            if n_good > 0:
                good_mask = is_good[dense_labels]
                X_good = X_batch[good_mask]
                G_good = G_batch[good_mask]
                good_counts = aug_sizes[good_idx_local]
                dense_labels_good = repeats(xp.arange(n_good), good_counts)

                # Geometry Metrics (Radii, Lipschitz)
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

                # Compute Trust Ellipsoid (Q) via Covariance
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

                # Covariance Shrinkage: C <- (1-a)C + a*tr(C)/d*I
                if Q_SHRINK > 0.0:
                    tr = xp.trace(Cov, axis1=1, axis2=2)
                    scale_tr = tr / max(d, 1)
                    I = xp.eye(d, dtype=Cov.dtype)[None, :, :]
                    Cov = (1.0 - Q_SHRINK) * Cov + Q_SHRINK * (
                        scale_tr[:, None, None] * I
                    )
                    Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

                # Eigendecomposition for Q
                Lambda, U = xp.linalg.eigh(Cov)
                Lambda_inv = 1.0 / xp.maximum(Lambda, TINY_EPS)

                # Scale Q by max distance in eigen-basis
                dist_sq = run_fused_q_dist(
                    X_good, mus[good_idx_local], dense_labels_good, U, Lambda_inv
                )
                max_dist = xp.zeros((n_good,), dtype=dist_sq.dtype)
                cupy_scatter_max(max_dist, dense_labels_good, dist_sq)
                scale_fac = 1.0 / xp.maximum(max_dist, TINY_EPS)

                # Q = U * (scale * L_inv) * U^T
                Qs = (U * (scale_fac[:, None] * Lambda_inv)[:, None, :]) @ U.transpose(
                    0, 2, 1
                )

                # Optional: Hessian Storage
                H_sqrt = None
                if STORE_HESSIAN_SQRT_PER_CLUSTER:
                    # Identify active component for storage (based on L2 relative error)
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

                    # Estimate Symmetric Hessian
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

                    # Solve Sylvester Eq via Eigendecomposition
                    dvals, U_h = xp.linalg.eigh(C_h)
                    dvals = xp.maximum(dvals, TINY_EPS)
                    Gtilde = U_h.transpose(0, 2, 1) @ S_h @ U_h
                    F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])
                    _, V_h = xp.linalg.eigh(F_mat)

                    # Reconstruct |H|^1/2
                    Z = U_h @ V_h
                    M = (Z * xp.sqrt(xp.abs(_))[:, None, :]) @ Z.transpose(0, 2, 1)
                    H_sqrt = 0.5 * (M + M.transpose(0, 2, 1))

                # Download Results to CPU
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

            # 6. Handle Rejected ("Bad") Clusters -> Split
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
                    # Fallback: PCA Split (Split along variance of X)
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
                    # Hessian Split: Split along eigenvector of Hessian estimate
                    # 1. Determine "Active Component" (Output dim with highest error)
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

                    # 2. Accumulate Stats and Solve for Hessian
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
                    dvals = xp.maximum(dvals, TINY_EPS)
                    Gtilde = U.transpose(0, 2, 1) @ S @ U
                    F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])
                    _, V = xp.linalg.eigh(F_mat)

                    # M = |H|^1/2
                    Z = U @ V
                    M_split = (Z * xp.sqrt(xp.abs(_))[:, None, :]) @ Z.transpose(
                        0, 2, 1
                    )
                    M_split = 0.5 * (M_split + M_split.transpose(0, 2, 1))

                    # Covariance in transformed space Z = M * dx
                    CovZ = M_split @ C @ M_split.transpose(0, 2, 1)

                # Get Split Direction (Dominant Eigenvector)
                _, V_z = xp.linalg.eigh(CovZ)
                v = V_z[:, :, -1]

                # Project Points & Sort
                scores_proj = run_fused_score(
                    X_bad, mus[bad_idx_local], dense_labels_bad, M_split, v
                )
                sort_order = xp.lexsort(xp.stack([scores_proj, dense_labels_bad]))

                # Stream to Host Buffers
                next_level_X_list.append(xp.asnumpy(X_bad[sort_order]))
                next_level_F_list.append(xp.asnumpy(F_bad[sort_order]))
                next_level_G_list.append(xp.asnumpy(G_bad[sort_order]))

                # Calculate Child Sizes (Binary Split)
                half = bad_counts // 2
                other = bad_counts - half
                new_sz = xp.zeros(2 * n_bad, dtype=xp.int64)
                new_sz[0::2] = half
                new_sz[1::2] = other
                next_level_sizes_list.append(xp.asnumpy(new_sz))

            start_cluster_idx = end_cluster_idx
            del X_batch, F_batch, G_batch
            mempool.free_all_blocks()

        # --- 3. Level Transition ---
        if len(next_level_sizes_list) == 0:
            break

        # Consolidate Next Level Data
        next_X = np.concatenate(next_level_X_list)
        next_F = np.concatenate(next_level_F_list)
        next_G = np.concatenate(next_level_G_list)
        next_sizes = np.concatenate(next_level_sizes_list)

        # Upload to GPU (Handling potential zero-size clusters from splits)
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

        # Cleanup Host Buffers
        del (
            next_level_X_list,
            next_level_F_list,
            next_level_G_list,
            next_level_sizes_list,
        )
        del next_X, next_F, next_G, next_sizes
        gc.collect()
        level += 1

    # --- 4. Finalization ---
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
