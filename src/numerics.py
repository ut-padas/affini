"""
Batched numerical building blocks shared by the local models, split strategies,
acceptance criteria and trust regions.

Every function works on a batch of clusters stored contiguously (``labels`` gives
each row's cluster, ``starts`` the first row of each cluster) and runs as a few
vectorized passes, using custom CUDA kernels on the CuPy backend and NumPy
fallbacks otherwise. Nothing here knows about the refinement loop.
"""

from src import IS_CUPY_BACKEND, Array, xp

if IS_CUPY_BACKEND:
    import cupy as cp

TINY_EPS = 1e-12


# -----------------------------------------------------------------------------
# 1. CUDA KERNELS
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
# 2. MATH HELPERS & WRAPPERS
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
# 3. COVARIANCE AND HESSIAN HELPERS
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


# -----------------------------------------------------------------------------
# 4. ARRAY SHAPES
# -----------------------------------------------------------------------------
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
