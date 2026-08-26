import glob
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Adjust imports based on your actual source tree
from src import IS_CUPY_BACKEND, Array, xp

if IS_CUPY_BACKEND:
    import cupy as cp

logger = logging.getLogger(__name__)

TINY_EPS = 1e-12


# -----------------------------------------------------------------------------
# 1. CONFIGURATION & MODES
# -----------------------------------------------------------------------------
class SurrogateMode(Enum):
    NEAREST_NEIGHBOR = "nearest_neighbor"  #  No grads, f(mu) only, split via X^T X
    AFFINE_COV_NO_HESS = "affine_cov_no_hess"  # No grads, fit A/b, split via X^T X
    TAYLOR_COV_NO_HESS = (
        "taylor_cov_no_hess"  # Grads at centroids only, split via X^T X
    )
    AFFINE_COV_HESS = "affine_cov_hess"  # No grads, fit A/b, split via X^T H X, where H is estimated from covariance
    TAYLOR_COV_HESS = "taylor_cov_hess"  # Grads at centroids only, split via X^T H X, where H is estimated from covariance
    TAYLOR_LSQ_HESS = "taylor_lsq_hess"  # Grads everywhere, split via X^T H X, where H is estimated from LSQ fit to gradient residuals
    QUADRATIC = "quadratic"  # No grads, save as affine_cov_hess but with quadratic fit.
    # TODO: need to make it so that the Hessian of the quadratic is the Hessian we use for clustering.


@dataclass
class ClusteringConfig:
    mode: SurrogateMode
    error_metric: str  # "L2", "Linf-global", "Linf-local"
    store_hess: bool = False
    store_affine_fits: bool = False

    @property
    def requires_grad_everywhere(self) -> bool:
        """
        Returns True if the surrogate mode requires gradients at every point.
        """
        return self.mode == SurrogateMode.TAYLOR_LSQ_HESS

    @property
    def requires_grad_centroids(self) -> bool:
        """
        Returns True if the surrogate mode requires gradients only at the centroids.
        """
        return self.mode in (
            SurrogateMode.TAYLOR_LSQ_HESS,
            SurrogateMode.TAYLOR_COV_HESS,
            SurrogateMode.TAYLOR_COV_NO_HESS,
        )

    @property
    def requires_no_grads(self) -> bool:
        """
        Returns True if the surrogate mode does not require any gradients.
        """
        return self.mode in (
            SurrogateMode.NEAREST_NEIGHBOR,
            SurrogateMode.AFFINE_COV_NO_HESS,
            SurrogateMode.AFFINE_COV_HESS,
        )


# -----------------------------------------------------------------------------
# 2. CUDA KERNELS
# -----------------------------------------------------------------------------
_segmented_max_kernel_code = r"""
extern "C" __global__
void segmented_max_float64(const double* vals, const long long* starts, double* out, int num_starts, int num_vals) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_starts) {
        long long start = starts[tid];
        long long end = (tid + 1 < num_starts) ? starts[tid + 1] : num_vals;
        
        double max_val = vals[start]; 
        for (long long i = start + 1; i < end; ++i) {
            if (vals[i] > max_val) {
                max_val = vals[i];
            }
        }
        out[tid] = max_val;
    }
}
"""

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

_cross_cov_kernel_code = r"""
extern "C" __global__
void fused_cross_cov(
    const double* X, const double* F,
    const double* mus, const double* f_mus,
    const long long* labels,
    double* cross_out,
    int N, int D, int FD) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    
    long long k = labels[idx];
    int mu_offset = k * D;
    int x_offset = idx * D;
    int f_offset = idx * FD;
    int f_mu_offset = k * FD;

    for (int r = 0; r < D; r++) {
        double dx_r = X[x_offset + r] - mus[mu_offset + r];
        for (int c = 0; c < FD; c++) {
            double df_c = F[f_offset + c] - f_mus[f_mu_offset + c];
            atomicAdd(&cross_out[k*D*FD + r*FD + c], dx_r * df_c);
        }
    }
}
"""

_affine_fit_inplace_kernel_code = r"""
extern "C" __global__
void fused_affine_fit_regularize(
    double* Cov,
    const double* traces,
    double tiny_eps,
    int D,
    int K)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= K) return;

    double lam = 1e-8 * traces[k];
    if (lam < tiny_eps) lam = tiny_eps;

    int matrix_offset = k * D * D;
    for (int r = 0; r < D; r++) {
        Cov[matrix_offset + r * D + r] += lam;
    }
}
"""

_inplace_cholesky_sub_kernel_code = r"""
extern "C" __global__
void batched_inplace_cholesky_solve(
    double* A,
    double* B,
    int D,
    int FD,
    int K)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= K) return;

    int a_start = k * D * D;
    int b_start = k * D * FD;

    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int p = 0; p < j; p++) {
                sum += A[a_start + i * D + p] * A[a_start + j * D + p];
            }
            if (i == j) {
                double val = A[a_start + i * D + i] - sum;
                A[a_start + i * D + i] = sqrt(val > 0.0 ? val : 1e-12);
            } else {
                A[a_start + i * D + j] =
                    (A[a_start + i * D + j] - sum) / A[a_start + j * D + j];
            }
        }
    }

    for (int j = 0; j < FD; j++) {
        for (int i = 0; i < D; i++) {
            double sum = 0.0;
            for (int p = 0; p < i; p++) {
                sum += A[a_start + i * D + p] * B[b_start + p * FD + j];
            }
            B[b_start + i * FD + j] =
                (B[b_start + i * FD + j] - sum) / A[a_start + i * D + i];
        }
    }

    for (int j = 0; j < FD; j++) {
        for (int i = D - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int p = i + 1; p < D; p++) {
                sum += A[a_start + p * D + i] * B[b_start + p * FD + j];
            }
            B[b_start + i * FD + j] =
                (B[b_start + i * FD + j] - sum) / A[a_start + i * D + i];
        }
    }
}
"""

_fused_split_projection_kernel_code = r"""
extern "C" __global__
void fused_split_projection(
    const double* X,
    const double* mus,
    const long long* labels,
    const double* M_split,
    const double* v,
    double* scores_out,
    int N,
    int D)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    long long k = labels[idx];
    int mu_offset = k * D;
    int x_offset = idx * D;
    int m_offset = k * D * D;
    int v_offset = k * D;

    double final_score = 0.0;
    for (int i = 0; i < D; i++) {
        double z_i = 0.0;
        for (int j = 0; j < D; j++) {
            double dx_j = X[x_offset + j] - mus[mu_offset + j];
            z_i += M_split[m_offset + i * D + j] * dx_j;
        }
        final_score += z_i * v[v_offset + i];
    }
    scores_out[idx] = final_score;
}
"""

_fused_mahalanobis_projection_kernel_code = r"""
extern "C" __global__
void fused_mahalanobis_projection(
    const double* dx,
    const long long* labels,
    const double* U,
    double* proj_out,
    int N,
    int D)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    long long k = labels[idx];
    int dx_offset = idx * D;
    int u_offset = k * D * D;
    int proj_offset = idx * D;

    // Match xp.einsum("ni,nij->nj", dx, U[labels]):
    // proj[idx, j] = sum_i dx[idx, i] * U[k, i, j]
    for (int j = 0; j < D; j++) {
        double sum = 0.0;
        for (int i = 0; i < D; i++) {
            sum += dx[dx_offset + i] * U[u_offset + i * D + j];
        }
        proj_out[proj_offset + j] = sum;
    }
}
"""

if IS_CUPY_BACKEND:
    _mod_seg_max = cp.RawModule(code=_segmented_max_kernel_code)
    _kern_seg_max = _mod_seg_max.get_function("segmented_max_float64")

    _mod_stats = cp.RawModule(code=_stats_kernel_code)
    _kern_stats = _mod_stats.get_function("fused_hessian_stats")

    _mod_cross = cp.RawModule(code=_cross_cov_kernel_code)
    _kern_cross = _mod_cross.get_function("fused_cross_cov")

    _mod_affine_reg = cp.RawModule(code=_affine_fit_inplace_kernel_code)
    _kern_affine_reg = _mod_affine_reg.get_function("fused_affine_fit_regularize")

    _mod_cholesky = cp.RawModule(code=_inplace_cholesky_sub_kernel_code)
    _kern_cholesky = _mod_cholesky.get_function("batched_inplace_cholesky_solve")

    _mod_split_proj = cp.RawModule(code=_fused_split_projection_kernel_code)
    _kern_fused_split_proj = _mod_split_proj.get_function("fused_split_projection")

    _mod_mah_proj = cp.RawModule(code=_fused_mahalanobis_projection_kernel_code)
    _kern_fused_mah_proj = _mod_mah_proj.get_function("fused_mahalanobis_projection")


# -----------------------------------------------------------------------------
# 3. MATH HELPERS & WRAPPERS
# -----------------------------------------------------------------------------
def repeats(arr: Array, counts: Array) -> Array:
    """Repeats elements of an array given a counts array."""
    return xp.repeat(arr, counts)


def _reduce_max_1d(vals: Array, starts: Array) -> Array:
    """Segmented maximum reduction for contiguous blocks."""
    if IS_CUPY_BACKEND:
        K = starts.shape[0]
        N = vals.shape[0]

        # Ensure proper types for the C-kernel (double, long long) and memory layout
        vals_c = cp.ascontiguousarray(vals, dtype=cp.float64)
        starts_c = cp.ascontiguousarray(starts, dtype=cp.int64)
        out = cp.empty(K, dtype=cp.float64)

        # Standard CUDA block/grid sizing
        threads_per_block = 256
        blocks_per_grid = (K + threads_per_block - 1) // threads_per_block
        _kern_seg_max(
            (blocks_per_grid,), (threads_per_block,), (vals_c, starts_c, out, K, N)
        )

        # Recast if the original array wasn't float64 (e.g., float32)
        if vals.dtype != cp.float64:
            out = out.astype(vals.dtype)
        return out
    else:
        # Standard NumPy fallback
        return xp.maximum.reduceat(vals, starts)


def _reduce_sum_1d(vals: Array, starts: Array) -> Array:
    """Segmented sum reduction for contiguous blocks."""
    return xp.add.reduceat(vals, starts)


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
        ac_ptr = (
            active_comps if active_comps is not None else xp.zeros(K, dtype=xp.int32)
        )

        labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)
        ac_32 = xp.ascontiguousarray(ac_ptr, dtype=xp.int32)
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
        dx = X - mus[labels]
        outer_C = dx[:, :, None] * dx[:, None, :]
        C_out = xp.zeros((K, D, D), dtype=xp.float64)
        xp.add.at(C_out, labels, outer_C)

        if calc_E:
            assert G is not None and g_mus is not None
            active_j = active_comps[labels]
            n_idx = xp.arange(X.shape[0])
            g_x = G[n_idx, active_j, :]
            g_mu = g_mus[labels, active_j, :]
            dg = g_x - g_mu
            outer_E = dx[:, :, None] * dg[:, None, :]
            E_out = xp.zeros((K, D, D), dtype=xp.float64)
            xp.add.at(E_out, labels, outer_E)
        else:
            E_out = xp.array([], dtype=xp.float64)
        return C_out, E_out


def run_fused_cross_cov(X, F, mus, f_mus, labels, K, D, FD):
    """Accumulates Cross-Covariance matrix (dx * df^T)."""

    # TODO: The raw kernel below seems to be broken
    #       for functions with multi-dimensional outputs.
    '''
    if IS_CUPY_BACKEND:
        N = X.shape[0]
        cross_out = xp.zeros((K, D, FD), dtype=xp.float64)
        labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)
        block = 256
        grid = (N + block - 1) // block
        _kern_cross(
            (grid,),
            (block,),
            (
                xp.ascontiguousarray(X),
                xp.ascontiguousarray(F),
                xp.ascontiguousarray(mus),
                xp.ascontiguousarray(f_mus),
                labels_64,
                cross_out,
                N,
                D,
                FD,
            ),
        )
        return cross_out
    else:
    '''
    dx = X - mus[labels]
    df = F - f_mus[labels]
    outer = dx[:, :, None] * df[:, None, :]
    cross_out = xp.zeros((K, D, FD), dtype=xp.float64)
    xp.add.at(cross_out, labels, outer)
    return cross_out


def solve_regularized_covariance_inplace(auto_cov, cross_cov, D, FD):
    """
    Solves ``(auto_cov + lambda I) X = cross_cov`` while minimizing peak GPU
    allocations. On CuPy, both inputs are overwritten in-place and ``cross_cov``
    becomes the solution. NumPy keeps the original vectorized fallback.
    """
    if IS_CUPY_BACKEND:
        K = int(auto_cov.shape[0])
        traces = xp.trace(auto_cov, axis1=1, axis2=2)
        block = 256
        grid = (K + block - 1) // block
        _kern_affine_reg((grid,), (block,), (auto_cov, traces, TINY_EPS, D, K))
        _kern_cholesky((grid,), (block,), (auto_cov, cross_cov, D, FD, K))
        return cross_cov

    tr = xp.trace(auto_cov, axis1=1, axis2=2)
    lam = xp.maximum(1e-8 * tr, TINY_EPS)
    reg = lam[:, None, None] * xp.eye(D, dtype=xp.float64)[None, :, :]
    return xp.linalg.solve(auto_cov + reg, cross_cov)


def run_fused_split_projection(X, mus, labels, M_split, v):
    """
    Computes ``dot(M_split[label] @ (x - mu[label]), v[label])`` per point
    without materializing an ``(N, d, d)`` indexed matrix tensor on the GPU.
    """
    if IS_CUPY_BACKEND:
        N, D = X.shape
        scores_out = xp.empty((N,), dtype=xp.float64)
        labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)
        block = 256
        grid = (N + block - 1) // block
        _kern_fused_split_proj(
            (grid,),
            (block,),
            (
                xp.ascontiguousarray(X),
                xp.ascontiguousarray(mus),
                labels_64,
                xp.ascontiguousarray(M_split),
                xp.ascontiguousarray(v),
                scores_out,
                N,
                D,
            ),
        )
        return scores_out

    dx = X - mus[labels]
    z = xp.einsum("nij,nj->ni", M_split[labels], dx)
    return xp.sum(z * v[labels], axis=1)


def run_fused_mahalanobis_projection(dx, labels, U):
    """
    Computes ``dx @ U[label]`` per point without materializing
    ``U[labels]`` as an ``(N, d, d)`` tensor.

    This is used in accepted-cluster trust ellipsoid construction, where
    materializing the indexed eigenvector tensor can be O(N*d^2) memory.
    """
    if IS_CUPY_BACKEND:
        N, D = dx.shape
        proj_out = xp.empty((N, D), dtype=xp.float64)
        labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)
        block = 256
        grid = (N + block - 1) // block
        _kern_fused_mah_proj(
            (grid,),
            (block,),
            (
                xp.ascontiguousarray(dx),
                labels_64,
                xp.ascontiguousarray(U),
                proj_out,
                N,
                D,
            ),
        )
        return proj_out

    return xp.einsum("ni,nij->nj", dx, U[labels])


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
# 4. PIPELINE STAGE 1: EVALUATION
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
        SurrogateMode.TAYLOR_LSQ_HESS,
        SurrogateMode.TAYLOR_COV_HESS,
        SurrogateMode.TAYLOR_COV_NO_HESS,
    ):
        assert g_mus is not None
        lin = xp.einsum("nmd,nd->nm", g_mus[aug_ids], dx)  # (N, fd)
        diff = F_batch - (f_mus[aug_ids] + lin)
        surrogate_params = {"g_mus": g_mus}

    elif config.mode in (
        SurrogateMode.AFFINE_COV_HESS,
        SurrogateMode.AFFINE_COV_NO_HESS,
    ):
        # Adaptive Tikhonov fitting
        mean_F = _reduce_sum_1d(F_batch, aug_starts) / aug_sizes[:, None]  # (K, fd)

        # Computation of Covariance and Cross-Covariance
        auto_cov, _ = run_fused_stats(
            X_batch, mus, aug_ids, K_batch, d, fd, None, calc_E=False
        )  # (K, d, d)
        cross_cov = run_fused_cross_cov(
            X_batch, F_batch, mus, mean_F, aug_ids, K_batch, d, fd
        )  # (K, d, fd)

        A_T = solve_regularized_covariance_inplace(auto_cov, cross_cov, d, fd)
        del auto_cov
        A = A_T.transpose(0, 2, 1)  # (K, fd, d)
        b = mean_F - xp.einsum("kfd,kd->kf", A, mus)  # (K, fd)

        lin = xp.einsum("nfd,nd->nf", A[aug_ids], X_batch)  # (N, fd)
        diff = F_batch - (b[aug_ids] + lin)
        surrogate_params = {"A": A, "b": b}

    elif config.mode == SurrogateMode.QUADRATIC:
        """
        Fit model 0.5 * x^T * H * x + w^T * x + b, for H, w, b.
        Requires at least D = d(d+1) + 1 points, probably more for decent conditioning.
        """

        # Build the quadratic part
        D = d**2 + d + 1
        quadratic_lhs_matrix = xp.empty(N, D)
        for i in range(d):
            start = i * d
            end = (i + 1) * d
            quadratic_lhs_matrix[:, start:end] = 0.5 * dx[:, i] * dx

        # Build the linear part
        quadratic_lhs_matrix[:, (d**2) : -1] = dx

        # Build the constant part
        quadratic_lhs_matrix[:, -1] = xp.ones(N)

        # Compute covariance matrix
        # I think we can just do this with run_fused_stats (WHY IS IT CALLED THIS?)
        # Replacing mus with zeros since we include it directly into the covariance.
        lhsmat, _ = run_fused_stats(
            quadratic_lhs_matrix,
            xp.zeros(K_batch),
            aug_ids,
            K_batch,
            D,
            fd,
            None,
            calc_E=False,
        )

        # Adaptive Tikhonov fitting
        mean_F = _reduce_sum_1d(F_batch, aug_starts) / aug_sizes[:, None]  # (K, fd)

        rhsvec = run_fused_cross_cov(
            quadratic_lhs_matrix,
            F_batch,
            xp.zeros(K_batch),
            mean_F,
            aug_ids,
            K_batch,
            D,
            fd,
        )

        # Solve linear system
        tr = xp.trace(lhsmat, axis1=1, axis2=2)  # (K,)
        lam = xp.maximum(1e-8 * tr, TINY_EPS)
        reg = lam[:, None, None] * xp.eye(d, dtype=xp.float64)[None, :, :]  # (K, D, D)
        weights = xp.linalg.solve(lhsmat + reg, rhsvec).T  # (K, fd, D)

        # First d**2 dimensions define the Hessians.
        H = weights[:, :, : (d**2)].reshape(K_batch, fd, d, d)  # (K, fd, d, d)
        # Next d dimensions define the gradient.
        w = weights[:, :, (d**2) : (d**2 + d)]  # (K, fd, d)
        # Last dimension defines the linear term.
        b = weights[:, :, -1]  # (K, fd)

        # Compute the surrogate at the training data to measure errors
        quadratic_term = xp.einsum(
            "nfd,nfde,nfe->nf", X_batch, H[aug_ids], X_batch
        )  # (N, fd)
        linear_term = xp.einsum("nfd,nd->nf", w[aug_ids], X_batch)  # (N, fd)
        surrogate_val = quadratic_term + linear_term + b[aug_ids]
        diff = F_batch - surrogate_val
        surrogate_params = {"H": H, "w": w, "b": b}

    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    # --- 2. Evaluate Error Metrics ---
    if config.error_metric == "L2":
        # Calculate sum of squared errors
        err_pts = xp.sum(diff**2, axis=1)  # (N,)
        sq_err_cluster = _reduce_sum_1d(err_pts, aug_starts)  # (K,)

        # Calculate sum of squared true values for relative scaling
        val_pts = xp.sum(F_batch**2, axis=1)  # (N,)
        sq_val_cluster = _reduce_sum_1d(val_pts, aug_starts)  # (K,)

        # Relative L2 Error
        E_cluster_rel = xp.sqrt(sq_err_cluster) / (xp.sqrt(sq_val_cluster) + TINY_EPS)

        # Absolute RMS Error (Average absolute error per point)
        E_cluster_abs = xp.sqrt(sq_err_cluster / aug_sizes)

        # Pass if relative error is good, OR if absolute error is already negligible
        is_good = (E_cluster_rel < TOL) | (E_cluster_abs < TOL)
        return is_good, E_cluster_rel, surrogate_params
    elif config.error_metric == "Linf-global":
        err_pts = xp.max(
            xp.abs(diff) / (F_max_global[None, :] + TINY_EPS), axis=1
        )  # (N,)
        E_cluster = _reduce_max_1d(err_pts, aug_starts)  # (K,)
    elif config.error_metric == "Linf-local":
        E_cluster_rel = xp.zeros(K_batch, dtype=xp.float64)
        E_cluster_abs = xp.zeros(K_batch, dtype=xp.float64)
        for j in range(fd):
            abs_diff_j = xp.abs(diff[:, j])  # (N,)
            local_max_F_j = _reduce_max_1d(xp.abs(F_batch[:, j]), aug_starts)  # (K,)
            max_err_j = _reduce_max_1d(abs_diff_j, aug_starts)  # (K,)

            E_cluster_abs = xp.maximum(E_cluster_abs, max_err_j)  # (K,)
            E_cluster_rel = xp.maximum(
                E_cluster_rel, max_err_j / xp.maximum(local_max_F_j, TINY_EPS)
            )  # (K,)

        # Pass if relative error is good, OR if absolute error is already negligible
        is_good = (E_cluster_rel < TOL) | (E_cluster_abs < TOL)
        return is_good, E_cluster_rel, surrogate_params

    is_good = E_cluster < TOL  # (K,)
    return is_good, E_cluster, surrogate_params


# -----------------------------------------------------------------------------
# 5. PIPELINE STAGE 2: ACCEPTED STATS (Q, H, radii)
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
) -> tuple[Array, Optional[Array], Array, Optional[Array]]:
    """
    Computes Q (Trust Region shape), H (Hessian), radii, and L_obs for accepted clusters.
    """
    K_good = len(counts_good)
    N_good, d = X_good.shape
    good_starts = xp.r_[0, xp.cumsum(counts_good)[:-1]]  # (K,)

    dx = X_good - mus_good[dense_labels_good]  # (N_good, d)

    # --- 1. Compute Q (Inverse Covariance) ---
    Cov, _ = run_fused_stats(
        X_good, mus_good, dense_labels_good, K_good, d, 0, None, calc_E=False
    )
    Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))  # (K, d, d)

    if Q_SHRINK > 0.0:
        tr = xp.trace(Cov, axis1=1, axis2=2)  # (K,)
        scale_tr = tr / max(d, 1)
        I = xp.eye(d, dtype=xp.float64)[None, :, :]
        Cov = (1.0 - Q_SHRINK) * Cov + Q_SHRINK * (
            scale_tr[:, None, None] * I
        )  # (K, d, d)
        Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

    Lambda, U = xp.linalg.eigh(Cov)
    Lambda_inv = 1.0 / xp.maximum(Lambda, TINY_EPS)  # (K, d)

    proj = run_fused_mahalanobis_projection(dx, dense_labels_good, U)  # (N_good, d)
    dist_sq_mahalanobis = xp.sum(
        proj**2 * Lambda_inv[dense_labels_good], axis=1
    )  # (N_good,)
    max_dist = _reduce_max_1d(dist_sq_mahalanobis, good_starts)  # (K,)

    scale_fac = 1.0 / xp.maximum(max_dist, TINY_EPS)
    Qs = (
        U * (scale_fac[:, None] * Lambda_inv)[:, None, :] @ U.transpose(0, 2, 1)
    )  # (K, d, d)

    # --- 2. Compute Radii & L_obs ---
    dist_euclidean = xp.sqrt(xp.sum(dx**2, axis=1))
    radii = _reduce_max_1d(dist_euclidean, good_starts)  # (K,)

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
        L_obs = _reduce_max_1d(ratio_pts, good_starts)  # (K,)

    # --- 3. Compute Hessian (If requested) ---
    Hs = None
    if config.store_hess:
        if (
            config.mode == SurrogateMode.TAYLOR_LSQ_HESS
            and G_good is not None
            and g_mus_good is not None
        ):
            # Accumulate Gradient interaction matrix over all components iteratively to avoid OOM
            fd = G_good.shape[1]
            E_k = xp.zeros((K_good, d, d), dtype=xp.float64)
            for j in range(fd):
                ac = xp.full(K_good, j, dtype=xp.int32)
                _, E_j = run_fused_stats(
                    X_good,
                    mus_good,
                    dense_labels_good,
                    K_good,
                    d,
                    fd,
                    ac,
                    G=G_good,
                    g_mus=g_mus_good,
                    calc_E=True,
                )
                E_k += E_j

            S_h = E_k + E_k.transpose(0, 2, 1)

            dvals, U_h = xp.linalg.eigh(Cov)
            dvals = xp.maximum(dvals, TINY_EPS)
            Gtilde = U_h.transpose(0, 2, 1) @ S_h @ U_h  # (K, d, d)
            F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])  # (K, d, d)
            Hs = U_h @ F_mat @ U_h.transpose(0, 2, 1)
            Hs = 0.5 * (Hs + Hs.transpose(0, 2, 1))

        elif config.mode == SurrogateMode.AFFINE_COV_HESS:
            Hs = xp.linalg.inv(Cov + TINY_EPS * xp.eye(d)[None, :, :])

    return Qs, Hs, radii, L_obs


# -----------------------------------------------------------------------------
# 6. PIPELINE STAGE 3: SPLIT REJECTED
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
    fd = F_bad.shape[1]
    bad_starts = xp.r_[0, xp.cumsum(counts_bad)[:-1]]
    dx = X_bad - mus_bad[dense_labels_bad]

    if config.mode == SurrogateMode.QUADRATIC:
        """
        Use the Hessian matrix of the least squares fit for the splitting direction.
        TODO: what to do about the Hessian?
        """
        print(
            "NOTE: specific split_rejected_clusters for SurrogateMode.QUADRATIC not implemented yet.\n\
              For now, doing the same thing as the SurrogateMode.*_COV_HESS versions."
        )

    if config.mode == SurrogateMode.TAYLOR_LSQ_HESS and G_bad is not None:
        g_mus_exp = surrogate_params["g_mus"][dense_labels_bad]  # (N_bad, fd, d)
        f_mus_exp = f_mus_bad[dense_labels_bad]  # (N_bad, fd)

        lin = xp.einsum("nmd,nd->nm", g_mus_exp, dx)
        diff_bad = F_bad - (f_mus_exp + lin)  # (N_bad, fd)

        sq_err = _reduce_sum_1d(diff_bad**2, bad_starts)  # (K,)
        sq_val = _reduce_sum_1d(F_bad**2, bad_starts)  # (K,)
        scores = xp.sqrt(sq_err) / (xp.sqrt(sq_val) + TINY_EPS)
        active_comps = xp.argmax(scores, axis=1).astype(xp.int32)  # (K,)

        # Covariance and Active-Component Gradient Interaction
        C, E = run_fused_stats(
            X_bad,
            mus_bad,
            dense_labels_bad,
            K_bad,
            d,
            fd,
            active_comps,
            G=G_bad,
            g_mus=surrogate_params["g_mus"],
            calc_E=True,
        )

        C = 0.5 * (C + C.transpose(0, 2, 1))
        S = E + E.transpose(0, 2, 1)

        dvals, U = xp.linalg.eigh(C)
        dvals = xp.maximum(dvals, TINY_EPS)  # (K, d)
        Gtilde = U.transpose(0, 2, 1) @ S @ U
        F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])  # (K, d, d)
        _, V = xp.linalg.eigh(F_mat)

        Z = U @ V  # (K, d, d)
        M_split = (Z * xp.sqrt(xp.abs(_))[:, None, :]) @ Z.transpose(0, 2, 1)
        M_split = 0.5 * (M_split + M_split.transpose(0, 2, 1))  # (K, d, d)

        CovZ = M_split @ C @ M_split.transpose(0, 2, 1)  # (K, d, d)

    # --- Mode 1 & 3: Covariance Level Curves Split ---
    elif config.mode in (
        SurrogateMode.AFFINE_COV_HESS,
        SurrogateMode.TAYLOR_COV_HESS,
        SurrogateMode.QUADRATIC,
    ):
        if config.mode == SurrogateMode.TAYLOR_COV_HESS:
            g_mus_exp = surrogate_params["g_mus"][dense_labels_bad]  # (N_bad, fd, d)
            lin_bad = xp.einsum("nmd,nd->nm", g_mus_exp, dx)  # (N_bad, fd)
            diff_bad = F_bad - (f_mus_bad[dense_labels_bad] + lin_bad)  # (N_bad, fd)
            E_bad = xp.linalg.norm(diff_bad, axis=1)  # (N_bad,)
        else:  # AFFINE_COV - TWO-PASS LOCAL FIT
            dist_to_mu = xp.linalg.norm(dx, axis=1)  # (N_bad,)
            n_tangent = max(d * 2, MIN_PTS_PER_CLUSTER // 4)
            tangent_mask = get_n_smallest_per_block(
                dist_to_mu, dense_labels_bad, n_tangent
            )  # (N_bad,)

            X_core = X_bad[tangent_mask]  # (N_core, d)
            F_core = F_bad[tangent_mask]  # (N_core, fd)
            labels_core = dense_labels_bad[tangent_mask]  # (N_core,)

            # Tangent Fitting
            M_XX, _ = run_fused_stats(
                X_core, mus_bad, labels_core, K_bad, d, 0, None, calc_E=False
            )  # (K, d, d)
            M_XF = run_fused_cross_cov(
                X_core, F_core, mus_bad, f_mus_bad, labels_core, K_bad, d, fd
            )  # (K, d, fd)

            A_T_tangent = solve_regularized_covariance_inplace(M_XX, M_XF, d, fd)
            A_tangent = A_T_tangent.transpose(0, 2, 1)  # (K, fd, d)

            lin_bad = xp.einsum(
                "nmd,nd->nm", A_tangent[dense_labels_bad], dx
            )  # (N_bad, fd)
            diff_bad = F_bad - (f_mus_bad[dense_labels_bad] + lin_bad)  # (N_bad, fd)
            E_bad = xp.linalg.norm(diff_bad, axis=1)  # (N_bad,)

        # Find SMALLEST errors (where surrogate == true func)
        n_est_cov = max(1, MIN_PTS_PER_CLUSTER // 3)
        cov_est_mask = get_n_smallest_per_block(E_bad, dense_labels_bad, n_est_cov)

        X_hard = X_bad[cov_est_mask]  # (N_hard, d)

        # Hessian Inverse Estimate
        H_inv, _ = run_fused_stats(
            X_hard,
            mus_bad,
            dense_labels_bad[cov_est_mask],
            K_bad,
            d,
            0,
            None,
            calc_E=False,
        )  # (K, d, d)
        H_inv = 0.5 * (H_inv + H_inv.transpose(0, 2, 1))

        dvals, U = xp.linalg.eigh(H_inv)
        dvals_sqrt_inv = 1.0 / xp.sqrt(xp.maximum(xp.abs(dvals), TINY_EPS))
        M_split = (U * dvals_sqrt_inv[:, None]) @ U.transpose(0, 2, 1)  # (K, d, d)
        M_split = 0.5 * (M_split + M_split.transpose(0, 2, 1))

        C, _ = run_fused_stats(
            X_bad, mus_bad, dense_labels_bad, K_bad, d, 0, None, calc_E=False
        )  # (K, d, d)
        CovZ = M_split @ C @ M_split.transpose(0, 2, 1)

    elif config.mode in (
        SurrogateMode.NEAREST_NEIGHBOR,
        SurrogateMode.AFFINE_COV_NO_HESS,
        SurrogateMode.TAYLOR_COV_NO_HESS,
    ):
        C, _ = run_fused_stats(
            X_bad, mus_bad, dense_labels_bad, K_bad, d, 0, None, calc_E=False
        )  # (K, d, d)
        CovZ = 0.5 * (C + C.transpose(0, 2, 1))
        M_split = xp.eye(d, dtype=xp.float64)[None, :, :].repeat(
            K_bad, axis=0
        )  # (K, d, d)

    else:
        raise ValueError(f"Unknown split mode: {config.mode}")

    # --- Apply Split Projection ---
    _, V_z = xp.linalg.eigh(CovZ)
    v = V_z[:, :, -1]  # Highest variance eigenvector # (K, d)

    scores_proj = run_fused_split_projection(
        X_bad, mus_bad, dense_labels_bad, M_split, v
    )

    if IS_CUPY_BACKEND:
        del CovZ, V_z, v, M_split
        cp.get_default_memory_pool().free_all_blocks()

    sort_order = xp.lexsort(xp.stack([scores_proj, dense_labels_bad]))  # (N_bad,)
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
    """
    Ensures that F is a 2D array of shape (N, fd). If F is 1D, it converts it to (N, 1).
    """
    F = xp.asarray(F)
    if F.ndim == 1:
        return F[:, None]  # Convert (N,) to (N, 1)
    return F


def convert_to_3d_array(G: Array) -> Array:
    """
    Ensures that G is a 3D array of shape (N, 1, d). If G is 2D, it converts it to (N, 1, d).
    """
    G = xp.asarray(G)
    if G.ndim == 2:
        return G[:, None, :]  # Convert (N, d) to (N, 1, d)
    return G


def allocate_output_buffers_host(
    MAX_CLUSTERS_EST: int,
    d: int,
    fd: int,
    store_hess: bool,
    store_affine_fit: bool,
    no_gradient: bool,
) -> tuple[dict, int]:
    """
    Allocates output buffers for centroids, surrogate params, Q, E_cluster, radii, counts, and optionally gradients and Hessians.

    Returns the buffers and the initial capacity.
    """
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
    """
    Checks if the current buffers have enough capacity for required_total clusters. If not, it doubles the capacity until it does, copying existing data.

    Returns the (possibly new) buffers and the new capacity.
    """
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
    """
    Returns the prime factors of n as a list.
    For example, get_prime_factors(12) returns [2, 2, 3].
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
        factors.append(n)
    return factors


def uniform_grid_cluster_labels(
    X_host: np.ndarray, num_initial_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assigns each point in X_host to a cluster based on a uniform grid.

    The grid is constructed by dividing the bounding box of X_host into bins along each dimension. The number of bins in each dimension is determined by factorizing num_initial_clusters into prime factors and assigning them to dimensions in a way that balances the bin sizes.

    Returns:
    - labels: An array of shape (N,) containing the cluster label for each point.
    - bins: An array of shape (d,) containing the number of bins along each dimension. The total number of clusters is the product of the values in bins.
    """
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


"""
A new function to save the data to a single file.
I imagine this will cause problems with MPI, since two processes could read/write at the same time.
But for now I will run this without work stealing and it will be fine.
I just need the training data to do the testing with FALKON.
This will also be nice to have for later if, say, Skandan needs to compare training data.
"""


def save_generated_data(X, filename):
    path = Path("saved_training_data/")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    xp.save(f"saved_training_data/{filename}_{uuid.uuid4()}.npy", X)


def combine_generated_data(filename):
    path = Path("saved_training_data/")
    input_dir = "saved_training_data"

    # Grab all .npy files in the folder with the correct beginning
    files = glob.glob(os.path.join(input_dir, f"{filename}_*.npy"))

    # Load all files into a list
    data_list = [np.load(f) for f in files]

    if data_list:
        # Stack them into one large matrix and save
        final_matrix = np.vstack(data_list)
        if path.exists():
            shutil.rmtree(input_dir)
        np.save(f"{filename}.npy", final_matrix)
    else:
        print("ERROR: No files found to stack.")
