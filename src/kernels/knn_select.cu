// src/neighbors/knn.py helpers: per-row radix selection of the k smallest distances and candidate distances
// in float64.

// Radix selection of each row's k smallest values (the idea of cuVS's radix top-k):
// one block per row. Floats are mapped to order-preserving unsigned keys; four passes
// over 8-bit digits, each building a 256-bin histogram of the keys that match the
// prefix found so far, pin down the exact k-th smallest key. A final pass writes every
// index with a smaller key, plus the first ties (in index order, via a block-wide scan
// over contiguous per-thread chunks) until k are taken. Output order is unsorted.
__device__ __forceinline__ unsigned int fkey(float v) {
    unsigned int u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One block per row of D (n columns): k column indices into out, unsorted (see above).
extern "C" __global__ void __launch_bounds__(256)
    radix_select_k(const float* __restrict__ D, int n, int k, long long* __restrict__ out) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix, s_mask;
    __shared__ int s_remaining;
    __shared__ int scan_less[256], scan_eq[256];
    const float* row = D + (long long)blockIdx.x * n;
    long long* o = out + (long long)blockIdx.x * k;
    const int tid = threadIdx.x;
    if (tid == 0) {
        s_prefix = 0u;
        s_mask = 0u;
        s_remaining = k;
    }
    __syncthreads();

    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0u;
        __syncthreads();
        const unsigned int prefix = s_prefix, mask = s_mask;
        for (int c = tid; c < n; c += 256) {
            const unsigned int key = fkey(row[c]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int rem = s_remaining;
            unsigned int b = 0;
            for (; b < 256; ++b) {
                if ((int)hist[b] >= rem) break;
                rem -= (int)hist[b];
            }
            s_remaining = rem;  // how many keys equal to the k-th to take
            s_prefix = prefix | (b << shift);
            s_mask = mask | (255u << shift);
        }
        __syncthreads();
    }

    // Contiguous chunk per thread, so a scan gives index-ordered output positions.
    const unsigned int kth = s_prefix;
    const int chunk = (n + 255) / 256, c0 = tid * chunk, c1 = min(n, c0 + chunk);
    int nl = 0, ne = 0;
    for (int c = c0; c < c1; ++c) {
        const unsigned int key = fkey(row[c]);
        nl += key < kth;
        ne += key == kth;
    }
    scan_less[tid] = nl;
    scan_eq[tid] = ne;
    __syncthreads();
    if (tid == 0) {
        int a = 0, b = 0;
        for (int t = 0; t < 256; ++t) {
            int x = scan_less[t], y = scan_eq[t];
            scan_less[t] = a;
            scan_eq[t] = b;
            a += x;
            b += y;
        }
    }
    __syncthreads();
    const int n_less = k - s_remaining;
    int pl = scan_less[tid], pe = scan_eq[tid];
    for (int c = c0; c < c1; ++c) {
        const unsigned int key = fkey(row[c]);
        if (key < kth)
            o[pl++] = c;
        else if (key == kth) {
            if (pe < s_remaining) o[n_less + pe] = c;
            ++pe;
        }
    }
}

// One thread per (query, candidate): the exact float64 squared distance. The _rn intrinsics
// forbid fused multiply-adds, so the sum rounds as the sequential CPU reference does.
extern "C" __global__ void candidate_d2(const double* Q, const double* P, const long long* cand, long long m,
                                        int kk, int d, double* out) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= m * kk) return;
    const long long r = e / kk;
    const double* q = Q + r * d;
    const double* p = P + cand[e] * d;
    double s = 0.0;
    for (int t = 0; t < d; ++t) {
        const double df = __dsub_rn(q[t], p[t]);
        s = __dadd_rn(s, __dmul_rn(df, df));
    }
    out[e] = s;  // strict sequential IEEE sum, as the CPU reference
}
