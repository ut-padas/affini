// src/neighbors/containment.py (TrustRegionContainment.containing): which leaves' trust
// ellipsoids contain each query.

// One thread per (leaf j, query q): blockIdx.y is the query, the x dimension runs over leaves.
// mask[q * n + j] is 1 when dx^T T_j dx <= 1 with dx = query - c_j, else 0; every entry is
// written, so the caller need not clear it. Decisions are exact float64 tests, with two cheap
// rejections first that can only discard pairs whose distance is above 1:
//   (1) a_j |dx|^2 > lim, and (2) a_j |dx|^2 + g_j (u_j . dx)^2 > lim,
// where a_j <= lambda_min(T_j) and g_j = lambda_max - lambda_min with u_j its eigenvector, both
// lowered by their rounding error in Python, so a_j |dx|^2 + g_j (u_j . dx)^2 <= dx^T T_j dx;
// lim = 1 + slack covers the rounding of these d-term sums. The full d^2 product runs only
// for pairs that pass both. Centres and directions are stored transposed, (d, n), so that a
// warp's 32 leaves read 32 consecutive doubles per coordinate; the query sits in shared
// memory, and T_j (n, d, d) is read only by surviving pairs.
extern "C" __global__ void contained_mask(const double* __restrict__ Q, const double* __restrict__ Ct,
                                          const double* __restrict__ Ut, const double* __restrict__ a,
                                          const double* __restrict__ g, const double* __restrict__ T,
                                          long long n, int d, double lim, unsigned char* __restrict__ mask) {
    extern __shared__ double q[];
    const long long row = blockIdx.y;
    for (int k = threadIdx.x; k < d; k += blockDim.x) q[k] = Q[row * d + k];
    __syncthreads();
    const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    // 1. |dx|^2 and the ball test.
    double d2 = 0.0;
    for (int k = 0; k < d; ++k) {
        const double t = q[k] - Ct[(long long)k * n + j];
        d2 += t * t;
    }
    unsigned char inside = 0;
    if (a[j] * d2 <= lim) {
        // 2. The leading direction's term.
        double p = 0.0;
        for (int k = 0; k < d; ++k) p += Ut[(long long)k * n + j] * (q[k] - Ct[(long long)k * n + j]);
        if (a[j] * d2 + g[j] * p * p <= lim) {
            // 3. The full quadratic form, row by row of T_j.
            const double* Tj = T + j * (long long)d * d;
            double s = 0.0;
            for (int k = 0; k < d; ++k) {
                const double xk = q[k] - Ct[(long long)k * n + j];
                double r = 0.0;
                for (int l = 0; l < d; ++l) r += Tj[(long long)k * d + l] * (q[l] - Ct[(long long)l * n + j]);
                s += xk * r;
            }
            inside = s <= 1.0;
        }
    }
    mask[row * n + j] = inside;
}
