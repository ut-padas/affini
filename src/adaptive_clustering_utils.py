import logging
import time

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

# Compilation of RawModules (only when CuPy backend is available)
if IS_CUPY_BACKEND:
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
# Each function dispatches to the raw CUDA kernel (IS_CUPY_BACKEND=True)
# or a fully-vectorized xp fallback (IS_CUPY_BACKEND=False).
# -----------------------------------------------------------------------------
def run_fused_taylor_metric(X, mus, labels, F_vals, f_mus, g_mus, scales, mode):
    """
    Calculates aggregated Taylor error per point (Sum-sq or Max-Rel).
    Mode: 0 = L2 (Sum of Squares), 1 = Linf (Max Relative)
    """
    if IS_CUPY_BACKEND:
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
    else:
        # dx[i] = X[i] - mus[labels[i]]              shape: (N, D)
        # lin[i,j] = g_mus[labels[i], j, :] . dx[i]  shape: (N, FD)
        # diff[i,j] = F_vals[i,j] - (f_mus[labels[i],j] + lin[i,j])
        dx = X - mus[labels]  # (N, D)
        lin = xp.einsum("nfd,nd->nf", g_mus[labels], dx)  # (N, FD)
        diff = F_vals - (f_mus[labels] + lin)  # (N, FD)
        if mode == 0:
            # L2: sum of squared errors across output dims
            E_out = xp.sum(diff**2, axis=1)  # (N,)
        else:
            # Linf: max relative error across output dims
            E_out = xp.max(xp.abs(diff) / (scales[None, :] + TINY_EPS), axis=1)  # (N,)
        return E_out


def run_fused_taylor_comp_abs(X, mus, labels, F_vals, f_mus, g_mus, comp_idx):
    """Calculates Absolute Taylor error for a single specific output component."""
    if IS_CUPY_BACKEND:
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
    else:
        # E_out[i] = | F_vals[i, comp_idx] - (f_mus[labels[i], comp_idx]
        #                                      + g_mus[labels[i], comp_idx, :] . dx[i]) |
        dx = X - mus[labels]  # (N, D)
        g_comp = g_mus[labels, comp_idx, :]  # (N, D)
        dot = xp.sum(dx * g_comp, axis=1)  # (N,)
        f_lin = f_mus[labels, comp_idx] + dot  # (N,)
        E_out = xp.abs(F_vals[:, comp_idx] - f_lin)  # (N,)
        return E_out


def run_fused_metrics(X, G, mus, g_mus, labels):
    """Calculates point radii and Lipschitz ratio estimates."""
    if IS_CUPY_BACKEND:
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
    else:
        dx = X - mus[labels]  # (N, D)
        dist = xp.sqrt(xp.sum(dx**2, axis=1))  # (N,)

        # dG[i,j,:] = G[i,j,:] - g_mus[labels[i],j,:]   shape: (N, FD, D)
        dG = G - g_mus[labels]  # (N, FD, D)
        # ||dG[i,j,:]||_2 / dist[i]  →  max over j
        dg_norms = xp.sqrt(xp.sum(dG**2, axis=2))  # (N, FD)
        safe_dist = xp.maximum(dist, TINY_EPS)
        ratio = xp.max(dg_norms / safe_dist[:, None], axis=1)  # (N,)
        # Zero out ratio where point coincides with centroid
        ratio = xp.where(dist > TINY_EPS, ratio, xp.zeros_like(ratio))
        return dist, ratio


def run_fused_stats(
    X, mus, labels, K, D, FD, active_comps, G=None, g_mus=None, calc_E=False
):
    """Accumulates Covariance (C) and Gradient Interaction (E) matrices."""
    if IS_CUPY_BACKEND:
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
    else:
        # C_k = sum_{i: label=k} (x_i - mu_k)(x_i - mu_k)^T
        # E_k = sum_{i: label=k} (x_i - mu_k)(g_active(x_i) - g_active(mu_k))^T
        N = X.shape[0]
        dx = X - mus[labels]  # (N, D)
        C_out = xp.zeros((K, D, D), dtype=xp.float64)
        # Outer product per point: (N, D, D)
        outer_C = dx[:, :, None] * dx[:, None, :]  # (N, D, D)
        xp.add.at(C_out, labels, outer_C)
        C_out = xp.asarray(C_out)

        if calc_E:
            assert G is not None and g_mus is not None
            # active component for each point: active_comps[labels[i]]
            active_j = active_comps[labels]  # (N,)
            n_idx = xp.arange(N)
            # g_x[i,:] = G[i, active_j[i], :]      shape: (N, D)
            # g_mu[i,:] = g_mus[labels[i], active_j[i], :]  shape: (N, D)
            g_x = G[n_idx, active_j, :]  # (N, D)
            g_mu = g_mus[
                labels, active_j, :
            ]  # (N, D) — fancy index: (labels[i], active_j[i])
            # Note: g_mus[labels, active_j] uses paired fancy indexing correctly here
            dg = g_x - g_mu  # (N, D)
            # E contribution: dx_r * dg_c  →  outer product (N, D, D)
            outer_E = dx[:, :, None] * dg[:, None, :]  # (N, D, D)
            E_out = xp.zeros((K, D, D), dtype=xp.float64)
            xp.add.at(E_out, labels, outer_E)
            E_out = xp.asarray(E_out)
        else:
            E_out = xp.array([], dtype=xp.float64)

        return C_out, E_out


def run_fused_score(X, mus, labels, M, v):
    """Projects points onto split direction: score[i] = v[k]^T M[k] (x[i] - mu[k])."""
    if IS_CUPY_BACKEND:
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
    else:
        # z[i] = M[labels[i]] @ (x[i] - mu[labels[i]])  shape: (N, D)
        # score[i] = v[labels[i]] . z[i]
        dx = X - mus[labels]  # (N, D)
        z = xp.einsum("nij,nj->ni", M[labels], dx)  # (N, D)
        return xp.sum(z * v[labels], axis=1)  # (N,)


def run_fused_q_dist(X, mus, labels, U, L_inv):
    """Calculates Ellipsoidal distance for trust ellipsoids: dist^2 = (x-mu)^T Q (x-mu)."""
    if IS_CUPY_BACKEND:
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
    else:
        # Q = U diag(L_inv) U^T,  so (x-mu)^T Q (x-mu) = sum_j proj_j^2 * L_inv_j
        # where proj_j = (x - mu) . U[:,j]
        dx = X - mus[labels]  # (N, D)
        # proj[i,j] = dx[i,:] . U[labels[i],:,j]  →  dx @ U  (batch matmul)
        # U[labels] shape: (N, D, D);  proj = einsum("ni,nij->nj", dx, U[labels])
        proj = xp.einsum("ni,nij->nj", dx, U[labels])  # (N, D)
        return xp.sum(proj**2 * L_inv[labels], axis=1)  # (N,)


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
        from cupyx import scatter_max as cupy_scatter_max

        cupy_scatter_max(out, idx, vals)
    else:
        xp.maximum.at(out, idx, vals)


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


def allocate_output_buffers_host(
    MAX_CLUSTERS_EST: int, d: int,
) -> tuple[dict, int]:
    """Allocates host (CPU) memory for final clusters."""
    cap = max(1, int(MAX_CLUSTERS_EST))
    buffers = {
        "centroids": np.empty((cap, d), dtype=np.float64),
        "f": np.empty((cap), dtype=np.float64),
        "Q": np.empty((cap, d, d), dtype=np.float64),
        "E_cluster": np.empty((cap,), dtype=np.float64),
        "radii": np.empty((cap,), dtype=np.float64),
        "L_obs": np.empty((cap,), dtype=np.float64),
        "counts": np.empty((cap,), dtype=np.int64),
    }
    return buffers, cap


def adjust_buffer_capacity_host(
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
        new_buffers["H"] = np.empty((new_cap, d, d), dtype=np.float64)
    if current_count > 0:
        for k in new_buffers:
            new_buffers[k][:current_count] = buffers[k][:current_count]
    del buffers
    return new_buffers, int(new_cap)


def get_n_smallest_per_block(values, blocks, n):
    """
    Returns a boolean mask for the n smallest values in each block.
    """
    # 1. Get indices that would sort by block, then by value
    # lexsort uses keys from last to first
    sort_idx = xp.lexsort(xp.stack([values, blocks]))

    # 2. Compute the sizes of each block
    block_sizes = xp.bincount(blocks)
    block_starts = xp.concatenate((xp.array([0]), xp.cumsum(block_sizes[:-1])))

    # 3. Create array like blocks, but with the start indices instead of the original ids
    offsets = xp.zeros(len(blocks), dtype=xp.int64)
    diffs = xp.diff(block_starts, prepend=0)
    offsets[block_starts] = diffs
    local_indices = xp.cumsum(offsets)

    # 4. Sort these and create a mask for the n smallest in each block
    inv_sort_idx = xp.empty_like(sort_idx)
    inv_sort_idx[sort_idx] = xp.arange(len(sort_idx), dtype=sort_idx.dtype)
    sorted_local_indices = inv_sort_idx - local_indices

    # 5. Return the mask for the n smallest in each block
    pts_mask = sorted_local_indices < n
    return pts_mask


def get_prime_factors(n: int) -> list[int]:
    """Return prime factors of an integer in non-decreasing order.

    Args:
        n: Integer to factorize.

    Returns:
        A list containing the prime factors of ``n`` in non-decreasing order.
        Returns an empty list when ``n < 2``.

    Example:
        get_prime_factors(60)  # Returns [2, 2, 3, 5]
    """
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
        # Any leftover > 1 must be prime once trial divisors exceeded sqrt(n).
        factors.append(n)
    return factors


def uniform_grid_cluster_labels(
    X_host: np.ndarray,
    num_initial_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign labels using a uniform axis-aligned grid.

    The grid's total number of cells is exactly ``num_initial_clusters``. This is
    achieved by prime-factorizing ``num_initial_clusters`` and assigning each
    factor to the currently widest normalized dimension.

    Args:
        X_host: Input points with shape ``(N, d)``.
        num_initial_clusters: Target number of grid cells/clusters.

    Returns:
        A tuple ``(labels, bins)`` where:
        - ``labels`` has shape ``(N,)`` and contains flat grid-cell indices.
        - ``bins`` has shape ``(d,)`` and stores the number of bins per
          dimension.

    Raises:
        ValueError: If ``num_initial_clusters < 1``.
        ValueError: If ``X_host`` is not a 2D array.
    """
    if num_initial_clusters < 1:
        raise ValueError("num_initial_clusters must be >= 1 for uniform_grid")
    if X_host.ndim != 2:
        raise ValueError(f"X_host must be 2D, got shape={X_host.shape}")

    _, d = X_host.shape
    lower = np.min(X_host, axis=0)
    upper = np.max(X_host, axis=0)
    extents = np.maximum(upper - lower, TINY_EPS)

    bins = np.ones(d, dtype=np.int64)
    for p in sorted(get_prime_factors(num_initial_clusters), reverse=True):
        # Prefer splitting the dimension with the currently largest cell width.
        split_scores = extents / bins
        split_dim = int(np.argmax(split_scores))
        bins[split_dim] *= p

    extents_safe = np.maximum(upper - lower, TINY_EPS)
    rel = (X_host - lower[None, :]) / extents_safe[None, :]

    # Keep right-edge points in-range so floor(...) never produces out-of-bounds.
    rel = np.clip(rel, 0.0, 1.0 - np.finfo(np.float64).eps)
    coords = np.floor(rel * bins[None, :]).astype(np.int64)
    coords = np.minimum(coords, bins[None, :] - 1)

    # Flatten multi-dimensional bin coordinates into a single cluster id.
    labels = np.ravel_multi_index(coords.T, dims=tuple(int(b) for b in bins))

    return labels.astype(np.int64), bins
