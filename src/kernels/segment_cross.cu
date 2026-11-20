// Deterministic segmented sums and cross products for segment_cross in src/numerics/segments.py
// (untiled, shared-memory tiled, chunk reduction).

// Deterministic segmented cross products (see ``segment_cross``). Output entry (r, c) of
// segment k is a sequential sum over fixed SEGMENT_CHUNK-row chunks counted from the
// segment's own start, then a sequential sum over those chunks, so its bits depend only
// on the segment's rows: identical between runs and for any batch around the segment.
extern "C" __global__ void segment_cross_partial(const double* A, const double* B, const double* w,
                                                 const double* a_off, const double* b_off,
                                                 const long long* cum, const long long* starts,
                                                 const long long* counts, const int* er, const int* ec,
                                                 double* partial, long long n_chunks, int K, int E, int P,
                                                 int Q, int CH, int flags, int S, long long w_stride,
                                                 long long b_stride) {
    // 1. Thread -> (weight set sw, chunk j, entry e). The launch covers an upper bound of
    //    chunks (no host sync to count them), so threads past the last real chunk exit.
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= S * n_chunks * E) return;
    long long sj = t / E;  // weight set and chunk: S weight vectors share A, and B unless b_stride > 0
    int e = (int)(t - sj * E);
    long long sw = sj / n_chunks, j = sj - sw * n_chunks;
    w += sw * w_stride;
    B += sw * b_stride;
    if (j >= cum[K - 1]) return;
    int lo = 0, hi = K - 1;  // the segment owning chunk j: first k with cum[k] > j
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cum[mid] > j)
            hi = mid;
        else
            lo = mid + 1;
    }
    // 2. Chunk j's rows: chunks are counted from the segment's own start, never from the
    //    batch's, which is what makes a segment's bits independent of its neighbours.
    long long k = lo;
    long long first = cum[k] - (counts[k] + CH - 1) / CH;
    long long r0 = starts[k] + (j - first) * CH;
    long long r1 = min(r0 + CH, starts[k] + counts[k]);
    // 3. Sequential sum of this entry over the chunk. Offsets are subtracted per row (not
    //    folded into a sum - n * mean correction), which would cancel for tight clusters.
    int r = er[e], c = ec[e];
    double ao = (flags & 2) ? a_off[k * P + r] : 0.0;
    double bo = (flags & 4) ? b_off[k * Q + c] : 0.0;
    double s = 0.0;
    if (flags & 8) {  // B is all ones: plain segmented sums of A's columns
        for (long long i = r0; i < r1; ++i) s += A[i * P + r] - ao;
    } else if (flags & 1) {
        for (long long i = r0; i < r1; ++i) s += ((A[i * P + r] - ao) * w[i]) * (B[i * Q + c] - bo);
    } else {
        for (long long i = r0; i < r1; ++i) s += (A[i * P + r] - ao) * (B[i * Q + c] - bo);
    }
    partial[t] = s;
}

// The same partial sums as segment_cross_partial (same rows, same order, same arithmetic,
// so the same bits), one block per (weight set, chunk): the chunk's rows are loaded once
// into shared memory, offset- and weight-adjusted, and every entry's thread reads them
// there instead of re-reading and re-weighting global rows per entry. Used from
// SEGMENT_TILED_MIN_E entries (numerics.py); each thread owns up to 8 entries (acc[]) of its
// block's entry group.
//
// Bit-identity with the untiled kernel holds because, per entry, the row order and the
// expression ((A - a_off) * w) * (B - b_off) are the same; only where the operands are
// read from changes. Keep it that way: reassociating would change every GPU tree.
extern "C" __global__ void segment_cross_tiled(const double* A, const double* B, const double* w,
                                               const double* a_off, const double* b_off, const long long* cum,
                                               const long long* starts, const long long* counts,
                                               const int* er, const int* ec, double* partial,
                                               long long n_chunks, int K, int E, int P, int Q, int CH,
                                               int flags, int S, long long w_stride, long long b_stride,
                                               int TR, int same) {
    extern __shared__ double sh[];  // TR rows of A (and of B unless it is the same tile)
    // 1. Block -> (weight set, chunk); find the chunk's segment and rows as in the untiled kernel.
    long long sj = blockIdx.x;
    // Entry group: blockIdx.y covers entries e_base .. e_base + 8 blockDim.x - 1, so any E fits
    // (the tile is reloaded per group; each entry still sums its rows in the same order).
    const int e_base = blockIdx.y * 8 * blockDim.x;
    long long sw = sj / n_chunks, j = sj - sw * n_chunks;
    if (j >= cum[K - 1]) return;
    int lo = 0, hi = K - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cum[mid] > j)
            hi = mid;
        else
            lo = mid + 1;
    }
    long long k = lo;
    long long first = cum[k] - (counts[k] + CH - 1) / CH;
    long long r0 = starts[k] + (j - first) * CH;
    long long r1 = min(r0 + CH, starts[k] + counts[k]);
    w += sw * w_stride;
    B += sw * b_stride;
    double* sA = sh;
    double* sB = same ? sh : sh + TR * P;
    const bool ones = flags & 8;
    double acc[8];
    for (int m = 0; m < 8; ++m) acc[m] = 0.0;
    // 2. Stream the chunk through shared memory TR rows at a time: the block loads the
    //    tile once (offset- and weight-adjusted, coalesced over columns), then every
    //    entry's thread accumulates over the tile's rows in row order.
    for (long long base = r0; base < r1; base += TR) {
        int nrow = (int)min((long long)TR, r1 - base);
        for (int t = threadIdx.x; t < nrow * P; t += blockDim.x) {
            int row = t / P, col = t - row * P;
            double v = A[(base + row) * P + col] - ((flags & 2) ? a_off[k * P + col] : 0.0);
            if ((flags & 1) && !ones) v = v * w[base + row];
            sA[row * P + col] = v;
        }
        if (!same && !ones) {
            for (int t = threadIdx.x; t < nrow * Q; t += blockDim.x) {
                int row = t / Q, col = t - row * Q;
                sB[row * Q + col] = B[(base + row) * Q + col] - ((flags & 4) ? b_off[k * Q + col] : 0.0);
            }
        }
        __syncthreads();
        for (int m = 0; m < 8; ++m) {
            int e = e_base + threadIdx.x + m * blockDim.x;
            if (e >= E) break;
            int r = er[e], c = ec[e];
            double s = acc[m];
            if (ones) {
                for (int row = 0; row < nrow; ++row) s += sA[row * P + r];
            } else {
                for (int row = 0; row < nrow; ++row) s += sA[row * P + r] * sB[row * Q + c];
            }
            acc[m] = s;
        }
        __syncthreads();
    }
    // 3. One partial per (weight set, chunk, entry), laid out as the untiled kernel's.
    for (int m = 0; m < 8; ++m) {
        int e = e_base + threadIdx.x + m * blockDim.x;
        if (e >= E) break;
        partial[sj * E + e] = acc[m];
    }
}

// The same partial sums as segment_cross_partial for large P x Q outputs (Grams of many
// features), one block per (weight set, chunk, 64 x 64 output tile), 16 x 16 threads each
// owning a 4 x 4 register tile. Per staged row a thread reads 4 values of A and 4 of B from
// shared memory for 16 products, where segment_cross_tiled reads 2 per product, which made it
// shared-memory-bound. Entry (r, c) still sums ((A - a_off) w)_r (B - b_off)_c over the chunk's
// rows in row order, carried across the staged row tiles, so its bits are those of the other
// two kernels. Only entries with eidx[r * Q + c] >= 0 are written (the upper triangle if
// symmetric); the partials keep the untiled kernel's (weight set, chunk, entry) layout.
extern "C" __global__ void segment_cross_tiled2d(const double* A, const double* B, const double* w,
                                                 const double* a_off, const double* b_off, const long long* cum,
                                                 const long long* starts, const long long* counts,
                                                 const int* tile_r, const int* tile_c, const int* eidx,
                                                 double* partial, long long n_chunks, int K, int E, int P, int Q,
                                                 int CH, int flags, long long w_stride, long long b_stride) {
    const int TR = 32;  // rows staged per pass: 2 x 32 x 64 doubles = 32 KB of shared memory
    __shared__ double sA[TR][64], sB[TR][64];
    // 1. Block -> (weight set, chunk) and output tile; the chunk's rows as in the untiled kernel.
    long long sj = blockIdx.x;
    long long sw = sj / n_chunks, j = sj - sw * n_chunks;
    if (j >= cum[K - 1]) return;
    int lo = 0, hi = K - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cum[mid] > j)
            hi = mid;
        else
            lo = mid + 1;
    }
    long long k = lo;
    long long first = cum[k] - (counts[k] + CH - 1) / CH;
    long long r0 = starts[k] + (j - first) * CH;
    long long r1 = min(r0 + CH, starts[k] + counts[k]);
    w += sw * w_stride;
    B += sw * b_stride;
    const int R0 = tile_r[blockIdx.y] * 64, C0 = tile_c[blockIdx.y] * 64;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    double acc[4][4];
    for (int u = 0; u < 4; ++u)
        for (int v = 0; v < 4; ++v) acc[u][v] = 0.0;
    // 2. Stage TR rows of the tile's A and B columns (offset- and weight-adjusted as in the other
    //    kernels; columns past P or Q as 0, never written), then accumulate in row order.
    for (long long base = r0; base < r1; base += TR) {
        int nrow = (int)min((long long)TR, r1 - base);
        for (int t = threadIdx.x; t < TR * 64; t += blockDim.x) {
            int row = t >> 6, col = t & 63;
            double a = 0.0, b = 0.0;
            if (row < nrow && R0 + col < P) {
                a = A[(base + row) * P + R0 + col] - ((flags & 2) ? a_off[k * P + R0 + col] : 0.0);
                if (flags & 1) a = a * w[base + row];
            }
            if (row < nrow && C0 + col < Q)
                b = B[(base + row) * Q + C0 + col] - ((flags & 4) ? b_off[k * Q + C0 + col] : 0.0);
            sA[row][col] = a;
            sB[row][col] = b;
        }
        __syncthreads();
        for (int row = 0; row < nrow; ++row) {
            double a[4], b[4];
            for (int u = 0; u < 4; ++u) a[u] = sA[row][ty + 16 * u];
            for (int v = 0; v < 4; ++v) b[v] = sB[row][tx + 16 * v];
            for (int u = 0; u < 4; ++u)
                for (int v = 0; v < 4; ++v) acc[u][v] += a[u] * b[v];
        }
        __syncthreads();
    }
    // 3. One partial per computed entry, at its index in the entry list.
    for (int u = 0; u < 4; ++u)
        for (int v = 0; v < 4; ++v) {
            int r = R0 + ty + 16 * u, c = C0 + tx + 16 * v;
            if (r < P && c < Q) {
                int e = eidx[r * Q + c];
                if (e >= 0) partial[sj * E + e] = acc[u][v];
            }
        }
}

// One thread per (weight set, segment, entry): the segment's chunk partials summed in chunk
// order; with sym, also written to the mirrored entry.
extern "C" __global__ void segment_cross_reduce(const double* partial, const long long* cum,
                                                const long long* counts, const int* er, const int* ec,
                                                double* out, int K, int E, int P, int Q, int CH, int sym,
                                                int S, long long n_chunks) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= (long long)S * K * E) return;
    long long sk = t / E;
    int e = (int)(t - sk * E);
    long long sw = sk / K, k = sk - sw * K;
    long long nch = (counts[k] + CH - 1) / CH, first = sw * n_chunks + cum[k] - nch;
    double s = 0.0;
    for (long long j = 0; j < nch; ++j) s += partial[(first + j) * E + e];
    int r = er[e], c = ec[e];
    out[sk * P * Q + r * Q + c] = s;
    if (sym && r != c) out[sk * P * Q + c * Q + r] = s;
}
