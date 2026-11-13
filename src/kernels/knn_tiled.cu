// Exact kNN candidate kernel for src/neighbors/knn.py; K, BM and QT are #defined by the loader before this
// text.

// Compile-time parameters (prepended): K (neighbours), BM (queries per block),
// QT (query element type: float, or double read and rounded to float on load, so
// float64 queries need no float32 copy).
#define BN 64              // leaves per tile
#define BK 16              // dimensions per tile step
#define TM (BM / 16)       // queries per thread (16 x 16 thread grid)
#define TN 4               // leaves per thread
#define BIG 3.0e38f        // "no candidate yet"
#define J ((K + 31) / 32)  // sorted-list slots owned by each lane

// One block of 16 x 16 threads per BM queries: every BN-leaf tile of distances is formed in
// shared memory from BK-dimension steps, then merged into each query's sorted K best, kept
// across tiles in shared memory and spread over a warp's lanes (J slots each).
extern "C" __global__ void __launch_bounds__(256)
    knn_tiled(const QT* __restrict__ Q,        // (m, d) queries, row-major
              const double* __restrict__ ctr,  // (d,) centre subtracted from queries on load
              const float* __restrict__ X,     // (n, d) centred leaf centroids, row-major
              const float* __restrict__ xn,    // (n,) |x|^2
              int m, int n, int d,
              float* out_d,           // (m, K) distances, nearest first
              long long* out_i) {     // (m, K) leaf indices
    __shared__ float Qs[BK][BM];      // query tile, transposed for conflict-free reads
    __shared__ float Xs[BK][BN];      // leaf tile
    __shared__ float Dt[BM][BN + 1];  // distance tile (+1 avoids bank conflicts)
    __shared__ float topd[BM][K];     // per-query sorted best distances (without |q|^2)
    __shared__ int topi[BM][K];       // and their leaf indices

    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int lane = tid % 32, warp = tid / 32;
    const int m0 = blockIdx.x * BM;

    for (int e = tid; e < BM * K; e += 256) {
        topd[e / K][e % K] = BIG;
        topi[e / K][e % K] = 0x7fffffff;
    }
    __syncthreads();

    for (int n0 = 0; n0 < n; n0 += BN) {
        // ---- 1. dot products for a BM x BN tile, accumulated in registers ----
        float acc[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

        for (int k0 = 0; k0 < d; k0 += BK) {
            for (int e = tid; e < BM * BK; e += 256) {
                int r = e / BK, c = e % BK, gq = m0 + r, gk = k0 + c;
                Qs[c][r] = (gq < m && gk < d) ? (float)((double)Q[(long long)gq * d + gk] - ctr[gk]) : 0.f;
            }
            for (int e = tid; e < BN * BK; e += 256) {
                int r = e / BK, c = e % BK, gx = n0 + r, gk = k0 + c;
                Xs[c][r] = (gx < n && gk < d) ? X[(long long)gx * d + gk] : 0.f;
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                float a[TM], b[TN];
#pragma unroll
                for (int i = 0; i < TM; ++i) a[i] = Qs[kk][ty * TM + i];
#pragma unroll
                for (int j = 0; j < TN; ++j) b[j] = Xs[kk][tx * TN + j];
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
            }
            __syncthreads();
        }

// ---- 2. partial squared distances |x|^2 - 2 q.x into the shared tile ----
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int col = n0 + tx * TN + j;
                Dt[ty * TM + i][tx * TN + j] = col < n ? xn[col] - 2.f * acc[i][j] : BIG;
            }
        __syncthreads();

        // ---- 3. selection: each warp handles BM / 8 rows of the tile ----
        for (int rr = 0; rr < BM / 8; ++rr) {
            const int r = warp * (BM / 8) + rr;
            float thr = topd[r][K - 1];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float v = Dt[r][lane + 32 * h];
                const int c = n0 + lane + 32 * h;
                // Lanes whose value beats the current k-th best; usually none.
                unsigned ball = __ballot_sync(0xffffffff, v < thr);
                while (ball) {
                    const int src = __ffs(ball) - 1;
                    const float vs = __shfl_sync(0xffffffff, v, src);
                    const int cs = __shfl_sync(0xffffffff, c, src);
                    if (vs < topd[r][K - 1]) {
                        // Warp-parallel insertion: lane l owns slots l, l + 32, ... (J of
                        // them); the insert position is the number of slots <= vs (ties
                        // keep order). All slots are read before any is written.
                        float a[J];
                        int ai[J];
                        int pos = 0;
#pragma unroll
                        for (int j = 0; j < J; ++j) {
                            const int slot = lane + 32 * j;
                            a[j] = slot < K ? topd[r][slot] : BIG;
                            ai[j] = slot < K ? topi[r][slot] : 0;
                            pos += __popc(__ballot_sync(0xffffffff, a[j] <= vs));
                        }
                        __syncwarp();
#pragma unroll
                        for (int j = 0; j < J; ++j) {
                            const int slot = lane + 32 * j;
                            if (slot >= pos && slot + 1 < K) {
                                topd[r][slot + 1] = a[j];
                                topi[r][slot + 1] = ai[j];
                            }
                        }
                        if (lane == 0) {
                            topd[r][pos] = vs;
                            topi[r][pos] = cs;
                        }
                    }
                    __syncwarp();
                    thr = topd[r][K - 1];
                    ball &= ~(1u << src);
                    ball &= __ballot_sync(0xffffffff, v < thr);  // drop lanes no longer good enough
                }
            }
        }
        __syncthreads();
    }

    // ---- 4. output: add |q|^2 back and take the square root ----
    for (int e = tid; e < BM * K; e += 256) {
        const int r = e / K, j = e % K, gq = m0 + r;
        if (gq < m) {
            float qn = 0.f;
            for (int t = 0; t < d; ++t) {
                const float q = (float)((double)Q[(long long)gq * d + t] - ctr[t]);
                qn = fmaf(q, q, qn);
            }
            out_d[(long long)gq * K + j] = sqrtf(fmaxf(topd[r][j] + qn, 0.f));
            out_i[(long long)gq * K + j] = topi[r][j];
        }
    }
}
