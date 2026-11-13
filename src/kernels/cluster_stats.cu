// Per-cluster helpers of src/numerics.py: segmented max, split and trust-region projections, point norms.

// One thread per segment: the maximum of vals[starts[k] : starts[k + 1]] (the last segment
// ends at num_vals), scanned in order. Segments must be non-empty.
extern "C" __global__ void segmented_max_float64(const double* vals, const long long* starts, double* out,
                                                 int num_starts, int num_vals) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_starts) {
        long long start = starts[tid];
        long long end = (tid + 1 < num_starts) ? starts[tid + 1] : num_vals;

        double max_val = vals[start];
        for (long long i = start + 1; i < end; ++i) {
            if (vals[i] > max_val) { max_val = vals[i]; }
        }
        out[tid] = max_val;
    }
}

// One thread per row: (M_k (x - mu_k)) . v_k for the row's cluster k, without gathering an
// (N, d, d) tensor of the M_k.
extern "C" __global__ void fused_split_projection(const double* X, const double* mus, const long long* labels,
                                                  const double* M_split, const double* v, double* scores_out,
                                                  int N, int D) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    long long k = labels[idx];
    long long mu_offset = k * D;
    long long x_offset = (long long)idx * D;
    long long m_offset = k * D * D;
    long long v_offset = k * D;

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

// One thread per row: the row vector dx U_k of its cluster's eigenvector matrix.
extern "C" __global__ void fused_mahalanobis_projection(const double* dx, const long long* labels,
                                                        const double* U, double* proj_out, int N, int D) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    long long k = labels[idx];
    long long dx_offset = (long long)idx * D;
    long long u_offset = k * D * D;
    long long proj_offset = (long long)idx * D;

    // Match xp.einsum("ni,nij->nj", dx, U[labels]):
    // proj[idx, j] = sum_i dx[idx, i] * U[k, i, j]
    for (int j = 0; j < D; j++) {
        double sum = 0.0;
        for (int i = 0; i < D; i++) { sum += dx[dx_offset + i] * U[u_offset + i * D + j]; }
        proj_out[proj_offset + j] = sum;
    }
}

// Per-point norms that would otherwise need (N, d) / (N, fd, d) temporaries:
// |x - mu[l]| and |F - f_c[l] - A[l] (x - mu[l])| for A (K, fd, d). The offset
// x - mu is loaded into registers once (up to MAXD dims; beyond that it is re-read).
#define MAXD 128
// One thread per row: |x - mu_k|.
extern "C" __global__ void centroid_distance(const double* X, const double* mus, const long long* labels,
                                             double* out, long long N, int D) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    const double* x = X + i * D;
    const double* mu = mus + labels[i] * D;
    double s = 0.0;
    for (int j = 0; j < D; ++j) {
        double t = x[j] - mu[j];
        s += t * t;
    }
    out[i] = sqrt(s);
}

// One thread per row: |F - f_c,k - A_k (x - mu_k)| over the fd outputs, with x - mu_k held
// in registers when d <= MAXD.
extern "C" __global__ void taylor_residual_norm(const double* X, const double* F, const double* mus,
                                                const double* f_c, const long long* labels, const double* A,
                                                double* out, long long N, int D, int FD) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    const long long k = labels[i];
    const double* x = X + i * D;
    const double* mu = mus + k * D;
    double dx[MAXD];
    const bool cached = D <= MAXD;
    if (cached)
        for (int j = 0; j < D; ++j) dx[j] = x[j] - mu[j];
    double s = 0.0;
    for (int m = 0; m < FD; ++m) {
        const double* a = A + (k * FD + m) * D;
        double lin = 0.0;
        for (int j = 0; j < D; ++j) lin += a[j] * (cached ? dx[j] : x[j] - mu[j]);
        double r = F[i * FD + m] - (f_c[k * FD + m] + lin);
        s += r * r;
    }
    out[i] = sqrt(s);
}
