"""
Exact k-nearest-neighbour search over the leaf centroids, used by
``AFINNISurrogate.predict``: ``GpuExactKNN`` (CUDA, replaces cuML's NearestNeighbors)
and ``CpuExactKNN`` (Numba + BLAS, replaces FAISS).

Contract of ``kneighbors``: neighbours are chosen by squared distances of the
*original* float64 points (direct differences, strict sequential IEEE sum); equal
distances keep the smaller leaf index; distances are float64. Strategies:

- CPU, d <= CPU_TREE_MAX_D: SciPy cKDTree on the original points (built with the
  index), its k+1 results re-scored with the shared strict reduction; rows with a
  near-tie at the k-th go to the reference scan.
- CPU above that, and GPU: a centred float32 search proposes k' = k + max(8, k)
  candidates, re-ranked from the original coordinates; a conservative error bound
  (centred magnitudes, so a common translation costs nothing) certifies that no
  omitted point is nearer; uncertified rows, and rows with non-finite, out-of-range or
  duplicate fast results, go to a float64 direct scan with the same reduction (on the
  device for GPU indexes). If k' covers every point, the scan is used directly.

``kneighbors_unverified`` returns the float32 candidates as found (k-means assignment,
which tolerates float32 ties; see ``src/kmeans.py``). ``last_search_stats`` holds the
last call's queries, candidate count, fallback rows and strategy. Queries must have
shape (m, d); wrong shapes are rejected before any native kernel. Range policy: inputs
are finite and every requested neighbour's squared distance must be representable in
float64; otherwise ``kneighbors`` raises ValueError instead of returning placeholders.

Memory: each index keeps the original float64 points (shared with the caller when
already contiguous float64; they must not be mutated while indexed). The float32 search
adds a centred float32 copy and norms (4 n (d + 1) bytes; on the CPU tree path only
when ``kneighbors_unverified`` is called). Verification and fallback temporaries are
bounded per block of ``_VERIFY_BLOCK_BYTES`` (a few such arrays), not overall; the GPU
fallback sorts each block's full distance rows.

GPU
---
Design (the ideas behind cuVS ``fusedL2Knn`` and FAISS ``WarpSelect``): one CUDA kernel
computes squared distances tile by tile and keeps each query's ``k`` best in shared
memory, so the (queries x leaves) distance matrix is never stored.

- Precision: float32, like cuML (it stores the index and returns distances in float32).
  ``predict`` re-ranks the returned candidates with float64 trust-region distances.
- Distances use the expanded form ``|q|^2 - 2 q.x + |x|^2``; ``|x|^2`` is precomputed
  once per index, ``|q|^2`` is constant per query and only added to the final ``k``.
- Each block owns ``BM`` queries and streams all leaves through shared memory in
  ``BN x BK`` tiles; each thread accumulates a ``TM x TN`` block of dot products in
  registers (a small hand-written matrix product).
- Selection: per row, a warp checks 64 tile entries against the row's current k-th
  best with ``__ballot_sync``; only the few that beat it are inserted, and the 32 lanes
  shift the sorted list together. Equal distances keep the earlier (smaller) index.
- Queries per block shrink as ``k`` grows (64 / 32 / 16 for k <= 32 / 128 / 256) so the
  per-query lists fit in shared memory; each lane owns ``ceil(k / 32)`` list slots.
  Above ``MAX_K`` an exact blocked path is used: float32 distance blocks from one
  matrix product each, then per-row radix selection of the k smallest (cuVS "radix
  top-k" idea) and a sort of only those k.

Measured on an RTX 2000 Ada laptop GPU against cuML 26.06 (float32 both, identical
recall): 1.3-2.1x faster for d = 5-100, k = 10-64, up to 1.1M leaves, with equal or
slightly lower memory; 1.1-1.8x for k = 100-200, parity at k = 256 (fused kernel), and
1.5-2.8x for k = 300-1000 (radix-select path).

This is an independent, simplified reimplementation of ideas from the following
sources (no code copied); cuVS is Apache-2.0, FAISS is MIT. Permalinks are to the
commits that were read:

- cuVS fused distance + warp-select kNN (k <= 64 dispatch, register tiles, norms in
  the epilogue, per-row warp queues):
  https://github.com/NVIDIA/cuvs/blob/920fc837e370b76345ab91dcbd699da40525a51b/cpp/src/neighbors/detail/fused_l2_knn.cuh
- cuVS brute-force dispatch and tiled fallback (expanded L2 from precomputed norms):
  https://github.com/NVIDIA/cuvs/blob/920fc837e370b76345ab91dcbd699da40525a51b/cpp/src/neighbors/detail/knn_brute_force.cuh
- FAISS tile sizing for a fixed memory budget (``chooseTileSize``), as vendored in cuVS:
  https://github.com/NVIDIA/cuvs/blob/920fc837e370b76345ab91dcbd699da40525a51b/cpp/src/neighbors/detail/faiss_select/DistanceUtils.h
- FAISS warp/block selection (threshold filtering against the queue's k-th value,
  warp-wide merges):
  https://github.com/facebookresearch/faiss/blob/277b3c76b7a1507915eb8f5010dcb367cbc95494/faiss/gpu/utils/Select.cuh
  https://github.com/facebookresearch/faiss/blob/277b3c76b7a1507915eb8f5010dcb367cbc95494/faiss/gpu/utils/WarpSelectKernel.cuh
  https://github.com/facebookresearch/faiss/blob/277b3c76b7a1507915eb8f5010dcb367cbc95494/faiss/gpu/utils/MergeNetworkWarp.cuh
- FAISS L2 selection over GEMM output with norms added while selecting:
  https://github.com/facebookresearch/faiss/blob/277b3c76b7a1507915eb8f5010dcb367cbc95494/faiss/gpu/impl/L2Select.cu
"""

import os

import numpy as np

from src import Array

MAX_K = 256  # largest k served by the fused kernel; above it, the blocked exact path

# Queries per block for a given k: the per-query lists (BM x K x 8 bytes) plus the
# distance and input tiles must fit in the 48 KB of static shared memory.
_BM_FOR_K = ((32, 64), (128, 32), (256, 16))

# Blocked exact path for k > MAX_K: float32 distance blocks of at most this size,
# then radix selection per row (bounded memory for any k).
_BLOCK_BYTES = 256 * 2**20

_KERNEL_SRC = r"""
// Compile-time parameters (prepended): K (neighbours), BM (queries per block),
// QT (query element type: float, or double read and rounded to float on load, so
// float64 queries need no float32 copy).
#define BN 64              // leaves per tile
#define BK 16              // dimensions per tile step
#define TM (BM / 16)       // queries per thread (16 x 16 thread grid)
#define TN 4               // leaves per thread
#define BIG 3.0e38f        // "no candidate yet"
#define J ((K + 31) / 32)  // sorted-list slots owned by each lane

extern "C" __global__ void __launch_bounds__(256)
knn_tiled(const QT* __restrict__ Q,      // (m, d) queries, row-major
          const double* __restrict__ ctr,// (d,) centre subtracted from queries on load
          const float* __restrict__ X,   // (n, d) centred leaf centroids, row-major
          const float* __restrict__ xn,  // (n,) |x|^2
          int m, int n, int d,
          float* out_d,                  // (m, K) distances, nearest first
          long long* out_i) {            // (m, K) leaf indices
    __shared__ float Qs[BK][BM];         // query tile, transposed for conflict-free reads
    __shared__ float Xs[BK][BN];         // leaf tile
    __shared__ float Dt[BM][BN + 1];     // distance tile (+1 avoids bank conflicts)
    __shared__ float topd[BM][K];        // per-query sorted best distances (without |q|^2)
    __shared__ int   topi[BM][K];        // and their leaf indices

    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int lane = tid % 32, warp = tid / 32;
    const int m0 = blockIdx.x * BM;

    for (int e = tid; e < BM * K; e += 256) { topd[e / K][e % K] = BIG; topi[e / K][e % K] = 0x7fffffff; }
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
                        float a[J]; int ai[J]; int pos = 0;
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
                            if (slot >= pos && slot + 1 < K) { topd[r][slot + 1] = a[j]; topi[r][slot + 1] = ai[j]; }
                        }
                        if (lane == 0) { topd[r][pos] = vs; topi[r][pos] = cs; }
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
            for (int t = 0; t < d; ++t) { const float q = (float)((double)Q[(long long)gq * d + t] - ctr[t]); qn = fmaf(q, q, qn); }
            out_d[(long long)gq * K + j] = sqrtf(fmaxf(topd[r][j] + qn, 0.f));
            out_i[(long long)gq * K + j] = topi[r][j];
        }
    }
}
"""

_KERNELS: dict = {}


def _kernel(k: int, bm: int, query_type: str):
    """Compile (once per combination) the kernel for ``k``, ``bm`` queries per block and the query dtype."""
    import cupy as cp

    key = (k, bm, query_type)
    if key not in _KERNELS:
        src = f"#define K {k}\n#define BM {bm}\n#define QT {query_type}\n" + _KERNEL_SRC
        _KERNELS[key] = cp.RawKernel(src, "knn_tiled")
    return _KERNELS[key]


class GpuExactKNN:
    """
    Exact Euclidean kNN index over ``points`` (n, d) on the GPU; see the module
    docstring for the precision contract and retained memory.
    """

    def __init__(self, points: Array):
        import cupy as cp

        # Centre everything on the centroids' mean: distances are unchanged, but the
        # norms in |q|^2 - 2 q.x + |x|^2 stay small, so float32 does not lose the
        # distance to cancellation when the data sit far from the origin.
        # points64: the original float64 coordinates (shared, not copied, when the input
        # already is contiguous float64; the caller must not mutate it while indexed),
        # used for exact re-ranking and the fallback. points: centred float32 copy for
        # the fast candidate search.
        self.points64 = cp.ascontiguousarray(cp.asarray(points, dtype=cp.float64))
        self.center = cp.mean(self.points64, axis=0)
        self.points = cp.ascontiguousarray(self.points64 - self.center, dtype=cp.float32)
        self.sq_norms = cp.einsum("ij,ij->i", self.points, self.points)

    def kneighbors(self, queries: Array, k: int) -> tuple[Array, Array]:
        """Exact ``(distances, indices)``, each (n_queries, min(k, n)), nearest first (module contract)."""
        import cupy as cp

        _check_queries(queries, self.points64.shape[1])
        return _verified(self, cp, queries, k)

    def kneighbors_unverified(self, queries: Array, k: int) -> tuple[Array, Array]:
        """The float32 candidates and distances as found: nearest up to float32 resolution, not certified."""
        _check_queries(queries, self.points64.shape[1])
        self.last_search_stats = _stats(queries.shape[0], min(int(k), self.points64.shape[0]), 0, "unverified")
        return self._search32(queries, k)

    def _search32(self, queries: Array, k: int) -> tuple[Array, Array]:
        """The float32 search: fused kernel for ``k <= MAX_K``, else the radix-select path."""
        import cupy as cp

        n = int(self.points.shape[0])
        k = min(int(k), n)
        if k < 1:
            raise ValueError(f"k must be at least 1; got {k}.")
        if k > MAX_K:
            return self._kneighbors_blocked(queries, k)
        double = queries.dtype == cp.float64
        Q = cp.ascontiguousarray(queries, dtype=cp.float64 if double else cp.float32)
        m, d = int(Q.shape[0]), int(self.points.shape[1])
        out_d = cp.empty((m, k), dtype=cp.float32)
        out_i = cp.empty((m, k), dtype=cp.int64)
        if m == 0:
            return out_d, out_i
        bm = next(bm for k_max, bm in _BM_FOR_K if k <= k_max)
        _kernel(k, bm, "double" if double else "float")(
            ((m + bm - 1) // bm,), (256,),
            (Q, self.center, self.points, self.sq_norms, cp.int32(m), cp.int32(n), cp.int32(d), out_d, out_i),
        )  # fmt: skip
        return out_d, out_i

    def _kneighbors_blocked(self, queries: Array, k: int) -> tuple[Array, Array]:
        """
        Exact kNN for ``k > MAX_K``: float32 distance blocks (one GEMM each), radix
        selection of each row's k smallest (``_RADIX_SELECT_SRC``), then a sort of just
        those k. Ties at the k-th distance keep the smallest indices.
        """
        import cupy as cp

        global _SELECT_KERNEL
        if _SELECT_KERNEL is None:
            _SELECT_KERNEL = cp.RawKernel(_RADIX_SELECT_SRC, "radix_select_k")
        n = int(self.points.shape[0])
        m = int(queries.shape[0])
        out_d = cp.empty((m, k), dtype=cp.float32)
        out_i = cp.empty((m, k), dtype=cp.int64)
        rows = max(1, _BLOCK_BYTES // (4 * n))
        for s in range(0, m, rows):
            q = (cp.asarray(queries[s : s + rows], dtype=cp.float64) - self.center).astype(cp.float32)
            r = int(q.shape[0])
            # |x|^2 - 2 q.x (|q|^2 is constant per row and added after selection).
            d2 = cp.ascontiguousarray(self.sq_norms[None, :] - 2.0 * (q @ self.points.T))
            cand = cp.empty((r, k), dtype=cp.int64)
            _SELECT_KERNEL((r,), (256,), (d2, cp.int32(n), cp.int32(k), cand))
            cand_d2 = cp.take_along_axis(d2, cand, axis=1)
            del d2
            # Nearest first, ties by index: stable sort by index, then by distance.
            by_idx = cp.argsort(cand, axis=1)
            cand, cand_d2 = cp.take_along_axis(cand, by_idx, axis=1), cp.take_along_axis(cand_d2, by_idx, axis=1)
            order = cp.argsort(cand_d2, axis=1)
            out_i[s : s + r] = cp.take_along_axis(cand, order, axis=1)
            q_sq = cp.einsum("ij,ij->i", q, q)[:, None]
            out_d[s : s + r] = cp.sqrt(cp.maximum(cp.take_along_axis(cand_d2, order, axis=1) + q_sq, 0.0))
        return out_d, out_i


# Radix selection of each row's k smallest values (the idea of cuVS's radix top-k):
# one block per row. Floats are mapped to order-preserving unsigned keys; four passes
# over 8-bit digits, each building a 256-bin histogram of the keys that match the
# prefix found so far, pin down the exact k-th smallest key. A final pass writes every
# index with a smaller key, plus the first ties (in index order, via a block-wide scan
# over contiguous per-thread chunks) until k are taken. Output order is unsorted.
_RADIX_SELECT_SRC = r"""
__device__ __forceinline__ unsigned int fkey(float v) {
    unsigned int u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

extern "C" __global__ void __launch_bounds__(256)
radix_select_k(const float* __restrict__ D, int n, int k, long long* __restrict__ out) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix, s_mask;
    __shared__ int s_remaining;
    __shared__ int scan_less[256], scan_eq[256];
    const float* row = D + (long long)blockIdx.x * n;
    long long* o = out + (long long)blockIdx.x * k;
    const int tid = threadIdx.x;
    if (tid == 0) { s_prefix = 0u; s_mask = 0u; s_remaining = k; }
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
            int rem = s_remaining; unsigned int b = 0;
            for (; b < 256; ++b) { if ((int)hist[b] >= rem) break; rem -= (int)hist[b]; }
            s_remaining = rem;                       // how many keys equal to the k-th to take
            s_prefix = prefix | (b << shift);
            s_mask = mask | (255u << shift);
        }
        __syncthreads();
    }

    // Contiguous chunk per thread, so a scan gives index-ordered output positions.
    const unsigned int kth = s_prefix;
    const int chunk = (n + 255) / 256, c0 = tid * chunk, c1 = min(n, c0 + chunk);
    int nl = 0, ne = 0;
    for (int c = c0; c < c1; ++c) { const unsigned int key = fkey(row[c]); nl += key < kth; ne += key == kth; }
    scan_less[tid] = nl; scan_eq[tid] = ne;
    __syncthreads();
    if (tid == 0) {
        int a = 0, b = 0;
        for (int t = 0; t < 256; ++t) { int x = scan_less[t], y = scan_eq[t]; scan_less[t] = a; scan_eq[t] = b; a += x; b += y; }
    }
    __syncthreads();
    const int n_less = k - s_remaining;
    int pl = scan_less[tid], pe = scan_eq[tid];
    for (int c = c0; c < c1; ++c) {
        const unsigned int key = fkey(row[c]);
        if (key < kth) o[pl++] = c;
        else if (key == kth) { if (pe < s_remaining) o[n_less + pe] = c; ++pe; }
    }
}
"""
_SELECT_KERNEL = None


# --------------------------------------------------------------------------- CPU
# FAISS IndexFlatL2 on the CPU (faiss/utils/distances.cpp) uses SIMD direct distances
# for small problems and BLAS sgemm tiles (4,096 queries x 1,024 database rows) with
# the |q|^2 + |x|^2 - 2 q.x epilogue and per-query heaps otherwise. Same plan here:
# - d <= CPU_DIRECT_MAX_D: each thread scans the database for its queries computing
#   differences directly (no cancellation, vectorizable inner loop over d).
# - larger d: float32 BLAS tiles of dot products, then a parallel per-query update
#   of a sorted top-k list that rejects almost every value with one comparison.
# FAISS's approximate HNSW index, previously used above 5,000 leaves, lost recall at
# high dimension (0.56 at d=100), so the CPU search is now always exact.
# Permalinks to the FAISS sources that were read:
#   https://github.com/facebookresearch/faiss/blob/277b3c76b7a1507915eb8f5010dcb367cbc95494/faiss/utils/distances.cpp
#   https://github.com/facebookresearch/faiss/blob/277b3c76b7a1507915eb8f5010dcb367cbc95494/faiss/utils/Heap.h
CPU_DIRECT_MAX_D = 16  # brute-force float32 search: direct scan up to here, BLAS tiles above
# Exact KD-tree search up to this d: 5x faster than verified brute force at d=5, a wash at
# d=10 and 2x slower at d=16 (12k leaves, k=10, 8 threads), so d > 10 uses brute force.
CPU_TREE_MAX_D = 10
_CPU_ROW_BLOCK = 1024  # queries per BLAS tile
_CPU_COL_BLOCK = 4096  # database rows per BLAS tile
_CPU_KERNELS = None


def _cpu_kernels():
    """Compile the Numba kernels on first use (cached on disk afterwards)."""
    global _CPU_KERNELS
    if _CPU_KERNELS is not None:
        return _CPU_KERNELS
    import numba as nb

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def direct_topk(Q, X, topd, topi):
        m, d = Q.shape
        n = X.shape[0]
        k = topd.shape[1]
        zero = Q.dtype.type(0.0)  # accumulate in the input precision (float32 or float64)
        for r in nb.prange(m):
            thr = topd[r, k - 1]
            for j in range(n):
                s = zero
                for t in range(d):
                    diff = Q[r, t] - X[j, t]
                    s += diff * diff
                if s < thr:
                    # Sorted insertion; equal distances keep the earlier index.
                    p = k - 1
                    while p > 0 and topd[r, p - 1] > s:
                        topd[r, p] = topd[r, p - 1]
                        topi[r, p] = topi[r, p - 1]
                        p -= 1
                    topd[r, p] = s
                    topi[r, p] = j
                    thr = topd[r, k - 1]

    @nb.njit(parallel=True, fastmath=False, cache=True)
    def update_topk(ip, xn, col0, topd, topi):
        rows, cols = ip.shape
        k = topd.shape[1]
        for r in nb.prange(rows):
            thr = topd[r, k - 1]
            for j in range(cols):
                v = xn[col0 + j] - np.float32(2.0) * ip[r, j]  # |q|^2 is added at the end
                if v < thr:
                    p = k - 1
                    while p > 0 and topd[r, p - 1] > v:
                        topd[r, p] = topd[r, p - 1]
                        topi[r, p] = topi[r, p - 1]
                        p -= 1
                    topd[r, p] = v
                    topi[r, p] = col0 + j
                    thr = topd[r, k - 1]

    # The float64 reference scan: same loop, strict IEEE arithmetic (no reassociation).
    direct_topk_ref = nb.njit(parallel=True, fastmath=False, cache=True)(direct_topk.py_func)

    @nb.njit(parallel=True, fastmath=False, cache=True)
    def candidate_d2_ref(Q, P, cand, out):
        # Same strict sequential reduction as direct_topk_ref, for the selected candidates.
        m, kk = cand.shape
        d = Q.shape[1]
        for r in nb.prange(m):
            for j in range(kk):
                s = 0.0
                for t in range(d):
                    diff = Q[r, t] - P[cand[r, j], t]
                    s += diff * diff
                out[r, j] = s

    _CPU_KERNELS = (direct_topk, update_topk, direct_topk_ref, candidate_d2_ref)
    return _CPU_KERNELS


def _stats(queries, candidates, fallback_rows, strategy: str) -> dict:
    """``last_search_stats`` record (summed per predict call in ``AFINNISurrogate.search_stats``)."""
    return {"queries": int(queries), "candidates": int(candidates), "fallback_rows": int(fallback_rows), "strategy": strategy}


def _require_finite(ok) -> None:
    """Range policy: every requested neighbour's squared distance must be finite in float64 (else ValueError)."""
    if not bool(ok.all()):
        raise ValueError(
            "kNN distances exceed the float64 range for some queries (squared distance "
            "overflow); rescale the inputs or reject such queries."
        )


def _check_queries(queries, d: int) -> None:
    """Native kernels have no bounds checks and use d as the row stride: reject wrong shapes first."""
    shape = getattr(queries, "shape", None)
    if shape is None or len(shape) != 2 or shape[1] != d:
        raise ValueError(f"queries must have shape (m, {d}); got {shape}.")


def _cpu_threads() -> int:
    """Threads for the CPU search: ``OMP_NUM_THREADS`` if set (one per MPI rank in our launchers), else all cores."""
    import numba as nb

    requested = os.environ.get("OMP_NUM_THREADS")
    limit = nb.config.NUMBA_NUM_THREADS
    return max(1, min(limit, int(requested))) if requested and requested.isdigit() else limit


class CpuExactKNN:
    """
    Exact Euclidean kNN index over ``points`` (n, d) on the CPU. Fitting keeps a
    centred float32 copy and the squared norms; queries are float64 or float32.
    """

    def __init__(self, points: np.ndarray):
        # points64: original float64 coordinates (shared when already contiguous float64;
        # must not be mutated while indexed) for exact re-ranking and the fallback.
        self.points64 = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
        self.center = self.points64.mean(axis=0)
        self._points32 = self._sq_norms32 = None
        if self.points64.shape[1] <= CPU_TREE_MAX_D:
            from scipy.spatial import cKDTree

            # Built here, not on first query, so retraining fails before installing a model;
            # the float32 copy is only created if kneighbors_unverified is called.
            self._kdtree = cKDTree(self.points64, leafsize=32, balanced_tree=False, compact_nodes=False)
        else:
            _ = self.points  # the verified search needs it: build it with the index

    @property
    def points(self) -> np.ndarray:
        """Centred float32 copy for the float32 search (created on first use)."""
        if self._points32 is None:
            self._points32 = np.ascontiguousarray(self.points64 - self.center, dtype=np.float32)
            self._sq_norms32 = np.einsum("ij,ij->i", self._points32, self._points32)
        return self._points32

    @property
    def sq_norms(self) -> np.ndarray:
        """float32 squared norms of ``points``."""
        _ = self.points
        return self._sq_norms32

    def kneighbors(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact ``(distances, indices)``, each (n_queries, min(k, n)), nearest first (module contract)."""
        _check_queries(queries, self.points64.shape[1])
        if self.points64.shape[1] <= CPU_TREE_MAX_D:
            return self._kneighbors_kdtree(queries, k)
        return _verified(self, np, queries, k)

    def kneighbors_unverified(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The float32 candidates and distances as found: nearest up to float32 resolution, not certified."""
        _check_queries(queries, self.points64.shape[1])
        self.last_search_stats = _stats(np.shape(queries)[0], min(int(k), self.points64.shape[0]), 0, "unverified")
        return self._search32(queries, k)

    def _kneighbors_kdtree(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Low d (<= CPU_TREE_MAX_D): exact search with SciPy's cKDTree on the original
        float64 points, multithreaded; no float32 stage. The k + 1 returned neighbours are
        re-scored with the shared strict reduction and ordered by (squared distance,
        index); rows whose (k+1)-th is within rounding of the k-th go to the reference scan.
        """
        n = int(self.points64.shape[0])
        k = min(int(k), n)
        if k < 1:
            raise ValueError(f"k must be at least 1; got {k}.")
        q = np.ascontiguousarray(np.asarray(queries, dtype=np.float64))
        m = q.shape[0]
        self.last_search_stats = _stats(m, min(k + 1, n), 0, "kdtree")
        if m == 0:
            return np.empty((0, k)), np.empty((0, k), dtype=np.int64)
        kk = min(k + 1, n)
        dist, idx = self._kdtree.query(q, k=kk, workers=_cpu_threads())
        dist, idx = dist.reshape(m, kk), idx.reshape(m, kk).astype(np.int64)
        # SciPy marks "no neighbour" (here: distance overflow) with index n and inf; such
        # entries must never reach the unchecked scorer. Requested neighbours must be
        # valid (range policy below); an invalid extra boundary candidate only means
        # there is no tie to check, so it is scored as +inf.
        invalid = (idx >= n) | (idx < 0) | ~np.isfinite(dist)
        _require_finite(~invalid[:, :k])
        idx = np.where(invalid, 0, idx)
        # Order by the shared strict squared distances (rounded roots can merge unequal
        # squared distances); only rows not already in (d2, index) order are re-sorted.
        dist = _candidate_d2(np, q, self.points64, idx)
        dist[invalid] = np.inf
        _require_finite(np.isfinite(dist[:, :k]))
        ties = np.nonzero(np.any(dist[:, 1:] <= dist[:, :-1], axis=1))[0]
        if ties.size:
            ti, td = idx[ties], dist[ties]
            order = np.argsort(ti, axis=1, kind="stable")
            ti, td = np.take_along_axis(ti, order, axis=1), np.take_along_axis(td, order, axis=1)
            order = np.argsort(td, axis=1, kind="stable")
            idx[ties], dist[ties] = np.take_along_axis(ti, order, axis=1), np.take_along_axis(td, order, axis=1)
        flagged = ~np.isfinite(dist[:, k - 1])
        if kk > k:
            # The tree's own reduction may differ from the reference at rounding level, so
            # a (k+1)-th within a relative 1e-12 of the k-th sends the row to the reference.
            flagged |= dist[:, k] - dist[:, k - 1] <= 1e-12 * np.abs(dist[:, k - 1])
        out_d, out_i = np.sqrt(dist[:, :k]), idx[:, :k].copy()
        _fallback(self, np, q, k, flagged, out_d, out_i)
        self.last_search_stats["fallback_rows"] = int(flagged.sum())
        return out_d, out_i

    def _search32(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The float32 search: direct scan for low d, BLAS tiles otherwise."""
        import numba as nb

        n, d = self.points.shape
        k = min(int(k), n)
        if k < 1:
            raise ValueError(f"k must be at least 1; got {k}.")
        Q = np.ascontiguousarray(np.asarray(queries, dtype=np.float64) - self.center, dtype=np.float32)
        m = Q.shape[0]
        out_i = np.zeros((m, k), dtype=np.int64)
        if m == 0:
            return np.empty((0, k), dtype=np.float32), out_i
        direct_topk, update_topk = _cpu_kernels()[:2]
        previous = nb.get_num_threads()
        nb.set_num_threads(_cpu_threads())
        try:
            if d <= CPU_DIRECT_MAX_D:
                topd = np.full((m, k), np.inf, dtype=np.float32)
                direct_topk(Q, self.points, topd, out_i)
                return np.sqrt(topd), out_i
            out_d = np.empty((m, k), dtype=np.float32)
            for r0 in range(0, m, _CPU_ROW_BLOCK):
                q = Q[r0 : r0 + _CPU_ROW_BLOCK]
                topd = np.full((q.shape[0], k), np.inf, dtype=np.float32)
                topi = out_i[r0 : r0 + _CPU_ROW_BLOCK]
                for c0 in range(0, n, _CPU_COL_BLOCK):
                    ip = q @ self.points[c0 : c0 + _CPU_COL_BLOCK].T  # BLAS sgemm
                    update_topk(ip, self.sq_norms, c0, topd, topi)
                q_sq = np.einsum("ij,ij->i", q, q)[:, None]
                out_d[r0 : r0 + _CPU_ROW_BLOCK] = np.sqrt(np.maximum(topd + q_sq, 0.0))
            return out_d, out_i
        finally:
            nb.set_num_threads(previous)



# --------------------------------------------------------------- exact verification
# float32 search can be wrong when coordinates are large relative to the gaps between
# points: centring and rounding can merge distinct points (e.g. 0 and 1 next to 1e9),
# and the expanded form cancels. So the float32 search returns k' = k + slack
# candidates; their distances are recomputed in float64 from direct differences, and a
# rigorous bound on the float32 error decides whether any point outside the candidates
# could still be nearer than the k-th. Queries where it could are recomputed by a
# float64 direct scan on the CPU (cancellation-free, slower, rare for normal data).
_VERIFY_SLACK = 8  # extra float32 candidates beyond k
_U32 = 2.0**-24  # float32 unit roundoff
_VERIFY_BLOCK_BYTES = 64 * 2**20


def _error_bound(xp, q_sq, x_sq_max, d, q_c_sq, x_c_sq_max):
    """
    Conservative bound on |float32 squared distance - squared distance of the original
    float64 points|, for one query, assuming finite inputs and d below ~1e5:
    float32 coordinate rounding (~3u (|q|+|x|)^2) and expanded-form arithmetic
    (~(d+2) u (|q|^2+|x|^2+2|q||x|)) on the centred values, plus the float64 centring:
    IEEE subtraction rounds relative to its result, |fl(a-c) - (a-c)| <= u64 |a-c|, so
    that term scales with the centred norms, not the absolute coordinates (a common
    translation costs nothing); and the square root/square round
    trip, all with a safety factor of two. Non-finite results disable certification.
    """
    return 2.0 * (d + 8) * _U32 * 2.0 * (q_sq + x_sq_max) + 16.0 * 2.0**-53 * (q_c_sq + x_c_sq_max)


def _verified(index, xp, queries, k):
    """
    Exact kNN on the original float64 coordinates: float32 candidates (centred), exact
    re-rank from direct differences of the original points, a bound check that no
    omitted point can be nearer, and a strict-IEEE float64 direct scan of the original
    points for every query the bound cannot certify (or with invalid fast results).
    """
    n, d = index.points64.shape
    k = min(int(k), n)
    if k < 1:
        raise ValueError(f"k must be at least 1; got {k}.")
    kk = min(n, k + max(_VERIFY_SLACK, k))
    q64 = xp.ascontiguousarray(xp.asarray(queries, dtype=xp.float64))  # original coordinates
    m = int(q64.shape[0])
    out_d = xp.empty((m, k), dtype=xp.float64)
    out_i = xp.empty((m, k), dtype=xp.int64)
    index.last_search_stats = _stats(m, kk, 0, "verified")
    if m == 0:
        return out_d, out_i
    if kk >= n:
        # Every point is needed: skip the fast search (its result would have to be a
        # valid permutation of the database) and scan the original points directly.
        flagged = xp.ones(m, dtype=bool)
        _fallback(index, xp, q64, k, flagged, out_d, out_i)
        _require_finite(xp.isfinite(out_d))
        index.last_search_stats = _stats(m, n, m, "full-scan")
        return out_d, out_i
    d32, cand = index._search32(queries, kk)
    if not hasattr(index, "_bound_consts"):  # computed once per index, no (n, d) temporary
        # Centred float64 norms exceed the float32 ones by at most ~d u32 relative, and
        # their term carries 16 u64, so twice the float32 maximum is a safe stand-in.
        x_sq = float(index.sq_norms.astype(xp.float64).max())
        index._bound_consts = (x_sq, 2.0 * x_sq)
    x_sq_max, x_orig_sq_max = index._bound_consts
    flagged = xp.zeros(m, dtype=bool)
    rows = max(1, _VERIFY_BLOCK_BYTES // (8 * kk * max(d, 1)))
    for s in range(0, m, rows):
        q = q64[s : s + rows]
        c = cand[s : s + rows]
        # An invalid fast search is never trusted: out-of-range or duplicate candidates,
        # or non-finite fast distances (e.g. float32 overflow), send the row to fallback.
        valid = xp.all((c >= 0) & (c < n), axis=1) & xp.all(xp.isfinite(d32[s : s + rows]), axis=1)
        c = xp.clip(c, 0, n - 1)
        stable = {} if xp is not np else {"kind": "stable"}
        c = xp.take_along_axis(c, xp.argsort(c, axis=1, **stable), axis=1)
        valid &= xp.all(c[:, 1:] != c[:, :-1], axis=1)
        d64 = _candidate_d2(xp, q, index.points64, c)  # same reduction as the fallback
        order = xp.argsort(d64, axis=1, **stable)[:, :k]
        out_i[s : s + rows] = xp.take_along_axis(c, order, axis=1)
        dk = xp.take_along_axis(d64, order, axis=1)
        out_d[s : s + rows] = xp.sqrt(dk)
        ok = valid & xp.all(xp.isfinite(dk), axis=1)
        if kk < n:
            # Every omitted point has float32 squared distance >= the k'-th candidate's,
            # hence original squared distance >= that minus the bound; equality stays
            # ambiguous (a tied omitted point with a smaller index must be considered).
            qc = (q - index.center).astype(xp.float32).astype(xp.float64)
            qc64 = q - index.center
            bound = _error_bound(xp, (qc * qc).sum(axis=1), x_sq_max, d, (qc64 * qc64).sum(axis=1), x_orig_sq_max)
            outside_min = d32[s : s + rows, kk - 1].astype(xp.float64) ** 2 - bound
            ok &= xp.isfinite(bound) & xp.isfinite(outside_min) & (outside_min > dk[:, -1])
        flagged[s : s + rows] = ~ok
    _fallback(index, xp, q64, k, flagged, out_d, out_i)
    _require_finite(xp.isfinite(out_d))
    index.last_search_stats = _stats(m, kk, flagged.sum(), "verified")
    return out_d, out_i


def _fallback(index, xp, q64, k, flagged, out_d, out_i):
    """Recompute the flagged rows by the reference scan of the original points (strict float64; on the device for GPU)."""
    if not bool(flagged.any()):
        return
    fi = xp.nonzero(flagged)[0]
    if xp is np:
        fd, fidx = _direct_float64(q64[fi], index.points64, k)
        out_d[fi], out_i[fi] = fd, fidx
        return
    # GPU: the same strict sequential reduction on the device, in query blocks of at
    # most _VERIFY_BLOCK_BYTES of distances; a stable sort keeps the smaller index on ties.
    n = int(index.points64.shape[0])
    rows = max(1, _VERIFY_BLOCK_BYTES // (8 * n))
    all_idx = xp.arange(n, dtype=xp.int64)
    for s in range(0, int(fi.shape[0]), rows):
        f = fi[s : s + rows]
        cand = xp.broadcast_to(all_idx, (int(f.shape[0]), n))
        d2 = _candidate_d2(xp, q64[f], index.points64, cand)
        order = xp.argsort(d2, axis=1)[:, :k]  # CuPy's argsort is stable
        out_i[f] = order
        out_d[f] = xp.sqrt(xp.take_along_axis(d2, order, axis=1))


_CAND_D2_SRC = r"""
extern "C" __global__ void candidate_d2(const double* Q, const double* P, const long long* cand,
                                        long long m, int kk, int d, double* out) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= m * kk) return;
    const long long r = e / kk;
    const double* q = Q + r * d;
    const double* p = P + cand[e] * d;
    double s = 0.0;
    for (int t = 0; t < d; ++t) { const double df = __dsub_rn(q[t], p[t]); s = __dadd_rn(s, __dmul_rn(df, df)); }
    out[e] = s;  // strict sequential IEEE sum, as the CPU reference
}
"""
_CAND_D2_KERNEL = None


def _candidate_d2(xp, q, P, cand):
    """Squared distances of the candidates, with the reference's strict sequential reduction."""
    global _CAND_D2_KERNEL
    m, kk = cand.shape
    out = xp.empty((m, kk), dtype=xp.float64)
    if xp is np:
        import numba as nb

        previous = nb.get_num_threads()
        nb.set_num_threads(_cpu_threads())
        try:
            _cpu_kernels()[3](np.ascontiguousarray(q), P, np.ascontiguousarray(cand), out)
        finally:
            nb.set_num_threads(previous)
        return out
    if _CAND_D2_KERNEL is None:
        _CAND_D2_KERNEL = xp.RawKernel(_CAND_D2_SRC, "candidate_d2")
    _CAND_D2_KERNEL(((m * kk + 255) // 256,), (256,),
                    (xp.ascontiguousarray(q), P, xp.ascontiguousarray(cand), xp.int64(m), xp.int32(kk),
                     xp.int32(q.shape[1]), out))  # fmt: skip
    return out


def _direct_float64(Q, P, k):
    """Reference scan: strict float64 direct differences of the original points; ties keep the smaller index."""
    import numba as nb

    direct_ref = _cpu_kernels()[2]
    topd = np.full((Q.shape[0], k), np.inf)
    topi = np.zeros((Q.shape[0], k), dtype=np.int64)
    previous = nb.get_num_threads()
    nb.set_num_threads(_cpu_threads())
    try:
        direct_ref(np.ascontiguousarray(Q), np.ascontiguousarray(P), topd, topi)
    finally:
        nb.set_num_threads(previous)
    return np.sqrt(topd), topi
