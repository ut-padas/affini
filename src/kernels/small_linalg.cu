// Small batched dense algebra for src/numerics/linalg.py: SPD solves, products with a fixed inner order, the
// affine-fit Cholesky.

// Cholesky A = L L^T in place, column by column (lower triangle only). One thread per system
// suits the p <= ~30 systems of training. Returns false for a non-SPD (or NaN) pivot.
__device__ bool chol_inplace(double* a, int p) {
    for (int j = 0; j < p; ++j) {
        double s = a[j * p + j];
        for (int k = 0; k < j; ++k) s -= a[j * p + k] * a[j * p + k];
        if (!(s > 0.0)) return false;
        double d = sqrt(s);
        a[j * p + j] = d;
        for (int i = j + 1; i < p; ++i) {
            double t = a[i * p + j];
            for (int k = 0; k < j; ++k) t -= a[i * p + k] * a[j * p + k];
            a[i * p + j] = t / d;
        }
    }
    return true;
}

// Right-hand side c of x (p, q): forward substitution with L, then back substitution with L^T.
// Columns never read each other, so solving them in any order or in parallel gives the same bits.
__device__ void chol_solve_column(const double* a, double* x, int p, int q, int c) {
    for (int i = 0; i < p; ++i) {
        double t = x[i * q + c];
        for (int k = 0; k < i; ++k) t -= a[i * p + k] * x[k * q + c];
        x[i * q + c] = t / a[i * p + i];
    }
    for (int i = p - 1; i >= 0; --i) {
        double t = x[i * q + c];
        for (int k = i + 1; k < p; ++k) t -= a[k * p + i] * x[k * q + c];
        x[i * q + c] = t / a[i * p + i];
    }
}

__device__ void chol_solve(const double* a, double* x, int p, int q) {
    for (int c = 0; c < q; ++c) chol_solve_column(a, x, p, q, c);
}

// One thread per system: factor and solve (see ``spd_solve``); a non-SPD system gets an
// all-NaN solution.
extern "C" __global__ void spd_solve(double* L, double* X, long long nb, int p, int q) {
    long long b = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= nb) return;
    double* a = L + b * p * p;
    double* x = X + b * p * q;
    if (!chol_inplace(a, p)) {
        for (int t = 0; t < p * q; ++t) x[t] = nan("");
        return;
    }
    chol_solve(a, x, p, q);
}

// One thread per system: the factor alone (``spd_factor``). A non-SPD system's factor is all
// NaN, so every later solve with it is all NaN, as spd_solve would return.
extern "C" __global__ void spd_factor(double* L, long long nb, int p) {
    long long b = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= nb) return;
    double* a = L + b * p * p;
    if (!chol_inplace(a, p))
        for (int t = 0; t < p * p; ++t) a[t] = nan("");
}

// One block per system: the factorization of chol_inplace with each column's below-diagonal
// entries computed by different threads. Every entry keeps its own sequential inner sum, in the
// same order, so the factor is bit-identical to chol_inplace; only independent entries run in
// parallel. For large p, where one thread per system leaves the GPU nearly idle.
extern "C" __global__ void spd_factor_coop(double* L, long long nb, int p) {
    __shared__ double diag;
    __shared__ int ok;
    long long b = blockIdx.x;
    if (b >= nb) return;
    double* a = L + b * p * p;
    if (threadIdx.x == 0) ok = 1;
    for (int j = 0; j < p; ++j) {
        // 1. The pivot, by one thread (a sequential sum, as chol_inplace).
        if (threadIdx.x == 0) {
            double s = a[j * p + j];
            for (int k = 0; k < j; ++k) s -= a[j * p + k] * a[j * p + k];
            if (!(s > 0.0)) ok = 0;
            diag = sqrt(s);
            if (ok) a[j * p + j] = diag;
        }
        __syncthreads();
        if (!ok) break;
        // 2. Column j below the pivot: independent entries, one per thread.
        for (int i = j + 1 + threadIdx.x; i < p; i += blockDim.x) {
            double t = a[i * p + j];
            for (int k = 0; k < j; ++k) t -= a[i * p + k] * a[j * p + k];
            a[i * p + j] = t / diag;
        }
        __syncthreads();
    }
    if (!ok)  // a non-SPD system: an all-NaN factor, as spd_factor
        for (int t = threadIdx.x; t < p * p; t += blockDim.x) a[t] = nan("");
}

// Packed lower triangle in shared memory: row i starts at i (i + 1) / 2.
__device__ __forceinline__ int tri(int i) { return i * (i + 1) / 2; }

// Cholesky of the packed lower triangle sh (in shared memory) by the whole block; col (p doubles
// of shared memory) is workspace. Right-looking, so once column j is final every warp updates the
// trailing triangle (one warp per row). Entry (i, m) still receives a_im - L_i0 L_m0 - L_i1 L_m1 -
// ... in k order with the same operands as chol_inplace's inner sum, and is divided by its pivot
// last, so the factor is bit-identical to chol_inplace and spd_factor_coop; only the schedule
// differs. Keep the row operand first in each product (L_ik * L_mk), as chol_inplace does.
// Returns false (to every thread) for a non-SPD or NaN pivot, leaving sh partly factored.
__device__ bool chol_packed_block(double* sh, double* col, int p, int* ok) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n_warps = blockDim.x >> 5;
    if (threadIdx.x == 0) *ok = 1;
    __syncthreads();
    for (int j = 0; j < p; ++j) {
        // 1. The pivot: a_jj already holds a_jj - sum_k<j L_jk^2, accumulated in k order.
        if (threadIdx.x == 0) {
            double s = sh[tri(j) + j];
            if (!(s > 0.0)) *ok = 0;
            sh[tri(j) + j] = sqrt(s);
        }
        __syncthreads();
        if (!*ok) return false;
        // 2. Column j below the pivot, also copied to col[] so the update reads it without bank conflicts.
        double d = sh[tri(j) + j];
        for (int i = j + 1 + threadIdx.x; i < p; i += blockDim.x) {
            double v = sh[tri(i) + j] / d;
            sh[tri(i) + j] = v;
            col[i] = v;
        }
        __syncthreads();
        // 3. Trailing update: a_im -= L_ij L_mj for j < m <= i.
        for (int i = j + 1 + warp; i < p; i += n_warps) {
            double li = col[i];
            double* row = sh + tri(i);
            for (int m = j + 1 + lane; m <= i; m += 32) row[m] -= li * col[m];
        }
        __syncthreads();
    }
    return true;
}

// One block per system, its packed lower triangle in shared memory (p (p + 1) / 2 + p doubles,
// so p <= 239 on a 227 KB card): the factor of chol_packed_block, bit-identical to spd_factor.
extern "C" __global__ void spd_factor_shared(double* L, long long nb, int p) {
    extern __shared__ double sh[];  // packed lower triangle, then column workspace
    __shared__ int ok;
    long long b = blockIdx.x;
    if (b >= nb) return;
    double* a = L + b * p * p;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n_warps = blockDim.x >> 5;
    // 1. Load the lower triangle; the upper triangle is never read.
    for (int i = warp; i < p; i += n_warps)
        for (int m = lane; m <= i; m += 32) sh[tri(i) + m] = a[i * p + m];
    // 2. Factor; 3. write the lower triangle back (the upper one keeps its input, as chol_inplace leaves it).
    if (!chol_packed_block(sh, sh + tri(p), p, &ok)) {
        for (int t = threadIdx.x; t < p * p; t += blockDim.x) a[t] = nan("");
        return;
    }
    for (int i = warp; i < p; i += n_warps)
        for (int m = lane; m <= i; m += 32) a[i * p + m] = sh[tri(i) + m];
}

// Solve the q right-hand sides x (p, q, in shared memory) with the packed factor sh, by the whole
// block. The forward substitution is right-looking with chol_solve's order (x_i - L_i0 x_0 - L_i1
// x_1 - ... then / L_ii), so its bits are chol_solve's. The back substitution is right-looking
// too, which subtracts L_ki x_k in decreasing k where chol_solve subtracts in increasing k: it
// rounds differently from chol_solve (a fixed order per system, so still reproducible and
// independent of the batch), in exchange for p parallel steps instead of a serial chain of
// p^2 / 2 dependent products.
__device__ void chol_packed_solve_block(const double* sh, double* x, int p, int q) {
    // 1. Forward: x_j final once divided; then every later row subtracts its L_ij x_j.
    for (int j = 0; j < p; ++j) {
        for (int c = threadIdx.x; c < q; c += blockDim.x) x[j * q + c] /= sh[tri(j) + j];
        __syncthreads();
        for (int t = threadIdx.x; t < (p - j - 1) * q; t += blockDim.x) {
            int i = j + 1 + t / q, c = t % q;
            x[i * q + c] -= sh[tri(i) + j] * x[j * q + c];
        }
        __syncthreads();
    }
    // 2. Back: x_k final once divided; then every earlier row subtracts L_ki x_k (row k of L: contiguous).
    for (int k = p - 1; k >= 0; --k) {
        for (int c = threadIdx.x; c < q; c += blockDim.x) x[k * q + c] /= sh[tri(k) + k];
        __syncthreads();
        for (int t = threadIdx.x; t < k * q; t += blockDim.x) {
            int i = t / q, c = t % q;
            x[i * q + c] -= sh[tri(k) + i] * x[k * q + c];
        }
        __syncthreads();
    }
}

// One block per system: factor (as spd_factor_shared) and solve its q right-hand sides with the
// factor still in shared memory (p (p + 1) / 2 + p + p q doubles), so L never goes to global
// memory; the solve is chol_packed_solve_block's, which rounds differently from spd_solve. A
// non-SPD system gets an all-NaN solution, as spd_solve.
extern "C" __global__ void spd_solve_shared(const double* A, double* X, long long nb, int p, int q) {
    extern __shared__ double sh[];  // packed lower triangle, column workspace, then x (p, q)
    __shared__ int ok;
    long long b = blockIdx.x;
    if (b >= nb) return;
    const double* a = A + b * p * p;
    double* xg = X + b * p * q;
    double* x = sh + tri(p) + p;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n_warps = blockDim.x >> 5;
    // 1. Load the lower triangle and the right-hand sides; 2. factor; 3. solve; 4. write x.
    for (int i = warp; i < p; i += n_warps)
        for (int m = lane; m <= i; m += 32) sh[tri(i) + m] = a[i * p + m];
    for (int t = threadIdx.x; t < p * q; t += blockDim.x) x[t] = xg[t];
    if (!chol_packed_block(sh, sh + tri(p), p, &ok)) {
        for (int t = threadIdx.x; t < p * q; t += blockDim.x) xg[t] = nan("");
        return;
    }
    chol_packed_solve_block(sh, x, p, q);
    for (int t = threadIdx.x; t < p * q; t += blockDim.x) xg[t] = x[t];
}

// One block per system: solves with a factor from spd_factor, staged into shared memory as a
// packed triangle (p (p + 1) / 2 + p q doubles) and solved as spd_solve_shared does, so its bits
// are spd_solve_shared's. A factor that failed (all NaN) gives all-NaN solutions.
extern "C" __global__ void spd_solve_factored_shared(const double* L, double* X, long long nb, int p, int q) {
    extern __shared__ double sh[];  // packed lower triangle, then x (p, q)
    long long b = blockIdx.x;
    if (b >= nb) return;
    const double* a = L + b * p * p;
    double* xg = X + b * p * q;
    double* x = sh + tri(p);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n_warps = blockDim.x >> 5;
    for (int i = warp; i < p; i += n_warps)
        for (int m = lane; m <= i; m += 32) sh[tri(i) + m] = a[i * p + m];
    for (int t = threadIdx.x; t < p * q; t += blockDim.x) x[t] = xg[t];
    __syncthreads();
    chol_packed_solve_block(sh, x, p, q);
    for (int t = threadIdx.x; t < p * q; t += blockDim.x) xg[t] = x[t];
}

// One thread per (system, right-hand side): solves with a factor from spd_factor. Each column
// keeps the solve half of spd_solve's arithmetic, so factor-then-solve gives spd_solve's bits,
// and a system with many columns (an inverse, q = p) is no longer one thread's serial p^3 work.
// Threads of one system are adjacent, so their x reads coalesce and their L reads are shared.
extern "C" __global__ void spd_solve_factored(const double* L, double* X, long long nb, int p, int q) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= nb * q) return;
    long long b = t / q;
    chol_solve_column(L + b * p * p, X + b * p * q, p, q, (int)(t - b * q));
}

// Batched small products with a fixed sequential inner sum (see ``det_matmul``).
extern "C" __global__ void det_matmul(const double* A, const double* B, double* out, long long total, int m,
                                      int n, int q) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= total) return;
    long long b = t / ((long long)m * q);
    int i = (int)((t / q) % m), j = (int)(t % q);
    const double* a = A + b * m * n + (long long)i * n;
    const double* bb = B + b * n * q + j;
    double s = 0.0;
    for (int k = 0; k < n; ++k) s += a[k] * bb[(long long)k * q];
    out[t] = s;
}

// One thread per cluster: the affine fit's ridge, max(1e-8 tr(Cov_k), tiny_eps), added to the
// diagonal in place.
extern "C" __global__ void fused_affine_fit_regularize(double* Cov, const double* traces, double tiny_eps,
                                                       int D, int K) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= K) return;

    double lam = 1e-8 * traces[k];
    if (lam < tiny_eps) lam = tiny_eps;

    int matrix_offset = k * D * D;
    for (int r = 0; r < D; r++) { Cov[matrix_offset + r * D + r] += lam; }
}

// One thread per cluster: Cholesky of the regularized D x D matrix in A, in place (a
// non-positive pivot becomes sqrt(1e-12) instead of failing), then forward and back
// substitution of the FD right-hand sides in B, which becomes the solution.
extern "C" __global__ void batched_inplace_cholesky_solve(double* A, double* B, int D, int FD, int K) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= K) return;

    int a_start = k * D * D;
    int b_start = k * D * FD;

    // 1. A = L L^T, row by row, L overwriting the lower triangle (the upper one is never read).
    for (int i = 0; i < D; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int p = 0; p < j; p++) { sum += A[a_start + i * D + p] * A[a_start + j * D + p]; }
            if (i == j) {
                double val = A[a_start + i * D + i] - sum;
                A[a_start + i * D + i] = sqrt(val > 0.0 ? val : 1e-12);
            } else {
                A[a_start + i * D + j] = (A[a_start + i * D + j] - sum) / A[a_start + j * D + j];
            }
        }
    }

    // 2. Forward substitution L y = b, then 3. back substitution L^T x = y, per right-hand side.
    for (int j = 0; j < FD; j++) {
        for (int i = 0; i < D; i++) {
            double sum = 0.0;
            for (int p = 0; p < i; p++) { sum += A[a_start + i * D + p] * B[b_start + p * FD + j]; }
            B[b_start + i * FD + j] = (B[b_start + i * FD + j] - sum) / A[a_start + i * D + i];
        }
    }

    for (int j = 0; j < FD; j++) {
        for (int i = D - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int p = i + 1; p < D; p++) { sum += A[a_start + p * D + i] * B[b_start + p * FD + j]; }
            B[b_start + i * FD + j] = (B[b_start + i * FD + j] - sum) / A[a_start + i * D + i];
        }
    }
}
