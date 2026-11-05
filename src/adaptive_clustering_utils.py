import glob
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Optional

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
    """
    Named presets, each a pair (local model on the leaves, split strategy for
    rejected clusters): see ``src.local_models.local_model_for_mode`` and
    ``src.splitting.split_strategy_for_mode``.
    """

    NEAREST_NEIGHBOR = "nearest_neighbor"  #  No grads, f(mu) only, split via X^T X
    AFFINE_COV_NO_HESS = "affine_cov_no_hess"  # No grads, fit A/b, split via X^T X
    TAYLOR_COV_NO_HESS = (
        "taylor_cov_no_hess"  # Grads at centroids only, split via X^T X
    )
    AFFINE_COV_HESS = "affine_cov_hess"  # No grads, fit A/b, split via X^T H X, where H is estimated from covariance
    AFFINE_COV_HESS_EXACT_GRAD = "affine_cov_hess_exact_grad" # Same as above, but with Hessian estimated with exact gradient.
    AFFINE_EXACT_HESS = "affine_exact_hess" # No grads, fit A/b, split via X^T H X, where H is given by an exact formula
    TAYLOR_COV_HESS = "taylor_cov_hess"  # Grads at centroids only, split via X^T H X, where H is estimated from covariance
    TAYLOR_EXACT_HESS = "taylor_exact_hess" # Grads at centroids only, split via X^T H X, where H is given by an exact formula
    TAYLOR_LSQ_HESS = "taylor_lsq_hess"  # Grads everywhere, split via X^T H X, where H is estimated from LSQ fit to gradient residuals


@dataclass(kw_only=True)
class ClusteringConfig:
    """Settings for adaptive_clustering(). All fields are keyword-only."""

    # --- Surrogate and acceptance test ---
    mode: SurrogateMode
    atol: float  # absolute error tolerance for accepting a cluster
    rtol: float  # relative error tolerance (w.r.t. max |F| over the initial data)
    error_metric: Literal["L2", "Linf-global", "Linf-local"] = "Linf-global"

    # --- Initial clustering ---
    initial_clustering_method: Literal["kmeans", "knndbscan", "uniform_grid"] = "kmeans"
    num_initial_clusters: int = 100
    save_initial_clustering: bool = False  # save labels to INITIAL_CLUSTERING_FILE
    use_existing_initial_clustering: bool = False  # load labels from that file if valid

    # --- Refinement ---
    min_pts_per_cluster: int = 100
    sampling_method: str = "convex_combination"  # see src.sampling.SAMPLERS
    q_reg: float = 1e-6  # regularization added when computing the trust-region shape Q
    max_levels: int = 50  # maximum number of refinement levels
    batch_size: int = 10_000_000  # max points moved to the device per batch
    seed: int = 0
    max_clusters_est: int = 100_000  # initial capacity of the output buffers (all ranks)
    use_work_stealing: bool = True

    # --- Outputs ---
    # Store a Hessian per accepted cluster and use it in a second-order Taylor
    # prediction. Only TAYLOR_LSQ_HESS estimates one; elsewhere it is ignored.
    store_hess: bool = False
    # Deprecated and ignored: what is stored is decided by the local model (affine
    # modes always store A_fit/b_fit). Kept so existing callers still work.
    store_affine_fits: bool = False

    # --- Diagnostics ---
    log_mem_stats: bool = False
    profile_timing: bool = False
    save_all_gen_training_data: bool = False

    def __post_init__(self) -> None:
        # Imported here: src.sampling imports Evaluator from this module.
        from src.sampling import SAMPLERS

        if self.sampling_method not in SAMPLERS:
            raise ValueError(
                f"Unknown sampling method {self.sampling_method!r}; "
                f"choose one of {sorted(SAMPLERS)}."
            )
        if self.store_hess and self.mode != SurrogateMode.TAYLOR_LSQ_HESS:
            self.store_hess = False
            logger.warning(
                f"store_hess is only supported by {SurrogateMode.TAYLOR_LSQ_HESS.value}; "
                f"ignoring it for mode {self.mode.value}."
            )

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
            SurrogateMode.TAYLOR_EXACT_HESS,
            SurrogateMode.AFFINE_COV_HESS_EXACT_GRAD
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
            SurrogateMode.AFFINE_EXACT_HESS
        )


@dataclass
class Evaluator:
    """
    Evaluates the target function, and optionally its Jacobian, and counts evaluations.

    Provide ``f_fn`` (with ``g_fn`` if gradients are needed), or ``oracle_fn``, which
    returns ``(F, G)`` in one call. When gradients are not needed and ``f_fn`` is
    available, only ``f_fn`` is called.

    ``hessian_fn`` (optional) returns the exact Hessian, shape (N, d, d); it is
    required by the *_EXACT_HESS modes and is not included in ``n_evals``.
    """

    f_fn: Optional[Callable[[Array], Array]] = None
    g_fn: Optional[Callable[[Array], Array]] = None
    oracle_fn: Optional[Callable[[Array], tuple[Array, Array]]] = None
    hessian_fn: Optional[Callable[[Array], Array]] = None
    n_evals: int = 0  # number of points evaluated so far

    def __post_init__(self) -> None:
        if self.f_fn is None and self.oracle_fn is None:
            raise ValueError("Provide f_fn or oracle_fn.")

    @property
    def has_grad(self) -> bool:
        """True if gradients can be computed (g_fn or oracle_fn was given)."""
        return self.oracle_fn is not None or self.g_fn is not None

    def __call__(
        self, X: Array, need_grad: bool = False
    ) -> tuple[Array, Optional[Array]]:
        """
        Return ``F`` with shape (N, fd) and, if ``need_grad``, ``G`` with shape
        (N, fd, d); otherwise ``G`` is None.
        """
        if need_grad and not self.has_grad:
            raise ValueError("Gradients requested but no g_fn or oracle_fn was given.")
        self.n_evals += int(X.shape[0])

        if self.f_fn is not None and not need_grad:
            return convert_to_2d_array(self.f_fn(X)), None
        if self.oracle_fn is not None:
            F, G = self.oracle_fn(X)
            return convert_to_2d_array(F), (convert_to_3d_array(G) if need_grad else None)
        assert self.f_fn is not None and self.g_fn is not None
        return convert_to_2d_array(self.f_fn(X)), convert_to_3d_array(self.g_fn(X))


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
# 4. CLUSTER ERROR METRICS
# -----------------------------------------------------------------------------
def cluster_errors(
    config: ClusteringConfig,
    F_batch: Array,
    F_hat: Array,
    aug_starts: Array,
    aug_sizes: Array,
    F_max_global: Array,
) -> tuple[Array, Array]:
    """
    Error of the fitted values ``F_hat`` against the true values ``F_batch`` on
    each cluster (rows ``aug_starts[k] : aug_starts[k] + aug_sizes[k]``), using
    ``config.error_metric``, and whether each cluster is within tolerance.
    Returns: (is_good_mask (K,), E_cluster (K,))
    """
    fd = F_batch.shape[1]
    K_batch = len(aug_sizes)
    diff = F_batch - F_hat

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
        is_good = (E_cluster_rel < config.rtol) | (E_cluster_abs < config.atol)
        return is_good, E_cluster_rel

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
        is_good = (E_cluster_rel < config.rtol) | (E_cluster_abs < config.atol)
        return is_good, E_cluster_rel

    is_good = E_cluster < config.atol  # (K,)
    return is_good, E_cluster


# -----------------------------------------------------------------------------
# 5. ACCEPTED CLUSTERS: TRUST-REGION SHAPE (Q) AND LSQ HESSIAN (H)
# -----------------------------------------------------------------------------
def lsq_hessian_in_eigenbasis(C: Array, S: Array) -> tuple[Array, Array]:
    """
    Least-squares Hessian from gradient residuals, in the eigenbasis of ``C``.

    Solves ``C H + H C = S`` for symmetric ``H`` (``C``: (K, d, d) point covariance,
    ``S``: (K, d, d) symmetrized gradient-residual moments). Returns ``(U, F)``
    with ``H = U F U^T``, where ``U`` holds the eigenvectors of ``C``.
    """
    dvals, U = xp.linalg.eigh(C)
    dvals = xp.maximum(dvals, TINY_EPS)
    Gtilde = U.transpose(0, 2, 1) @ S @ U  # (K, d, d)
    F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])  # (K, d, d)
    return U, F_mat


def _symmetric_point_covariance(
    X: Array, centroids: Array, labels: Array, K: int
) -> Array:
    """(K, d, d) covariance of each cluster's points about its centroid."""
    Cov, _ = run_fused_stats(
        X, centroids, labels, K, X.shape[1], 0, None, calc_E=False
    )
    return 0.5 * (Cov + Cov.transpose(0, 2, 1))


def trust_region_shapes(
    X: Array, centroids: Array, labels: Array, counts: Array, q_reg: float
) -> Array:
    """
    Shape matrix Q (K, d, d) of each cluster's trust region.

    Q is the inverse covariance of the cluster's points (eigenvalues clamped
    below at ``q_reg``), scaled so that the farthest point lies on the boundary:
    max over the cluster's points of dx^T Q dx is 1. Points are stored
    contiguously by cluster; ``labels`` gives each point's cluster (0..K-1).
    """
    K = len(counts)
    starts = xp.r_[0, xp.cumsum(counts)[:-1]]  # (K,)
    dx = X - centroids[labels]  # (N, d)

    Cov = _symmetric_point_covariance(X, centroids, labels, K)
    Lambda, U = xp.linalg.eigh(Cov)
    if q_reg > 0.0:
        Lambda = xp.maximum(Lambda, q_reg)
    Lambda_inv = 1.0 / xp.maximum(Lambda, TINY_EPS)  # (K, d)

    proj = run_fused_mahalanobis_projection(dx, labels, U)  # (N, d)
    dist_sq_mahalanobis = xp.sum(proj**2 * Lambda_inv[labels], axis=1)  # (N,)
    max_dist = _reduce_max_1d(dist_sq_mahalanobis, starts)  # (K,)

    scale_fac = 1.0 / xp.maximum(max_dist, TINY_EPS)
    return (
        U * (scale_fac[:, None] * Lambda_inv)[:, None, :] @ U.transpose(0, 2, 1)
    )  # (K, d, d)


def lsq_hessians(
    X: Array, G: Array, centroids: Array, g_c: Array, labels: Array, counts: Array
) -> Array:
    """
    Symmetric Hessian H (K, d, d) of each cluster, fitted by least squares to the
    gradient residuals ``G - g_c`` (summed over the output components); see
    ``lsq_hessian_in_eigenbasis``. Used as the second-order Taylor term.
    """
    K = len(counts)
    d, fd = X.shape[1], G.shape[1]
    Cov = _symmetric_point_covariance(X, centroids, labels, K)

    # Accumulate the gradient interaction matrix one component at a time to avoid OOM
    E_k = xp.zeros((K, d, d), dtype=xp.float64)
    for j in range(fd):
        ac = xp.full(K, j, dtype=xp.int32)
        _, E_j = run_fused_stats(
            X, centroids, labels, K, d, fd, ac, G=G, g_mus=g_c, calc_E=True
        )
        E_k += E_j
    S_h = E_k + E_k.transpose(0, 2, 1)

    U_h, F_mat = lsq_hessian_in_eigenbasis(Cov, S_h)
    Hs = U_h @ F_mat @ U_h.transpose(0, 2, 1)
    return 0.5 * (Hs + Hs.transpose(0, 2, 1))


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
