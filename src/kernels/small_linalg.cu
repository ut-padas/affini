// Small batched dense algebra for src/numerics: SPD solves, products with a fixed inner order, the affine-fit
// Cholesky.

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

// Each right-hand side: forward substitution with L, then back substitution with L^T.
__device__ void chol_solve(const double* a, double* x, int p, int q) {
    for (int c = 0; c < q; ++c) {
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

// One thread per system: solves with a factor from spd_factor; the same arithmetic as the
// solve half of spd_solve, so factor-then-solve gives spd_solve's bits.
extern "C" __global__ void spd_solve_factored(const double* L, double* X, long long nb, int p, int q) {
    long long b = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= nb) return;
    chol_solve(L + b * p * p, X + b * p * q, p, q);
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
