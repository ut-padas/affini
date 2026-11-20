// ART routing and plane scores on the GPU (src/art/routing.py: RoutingTable.prepare_queries and
// RoutingTable.walk; src/art/geometry.py: plane_scores).

// NumPy's row sum of u_j w_j, term for term, so a plane score has the same bits on the CPU
// (numpy.sum in plane_scores, the Numba walk) and here, and a split, the training points it cut
// and every query routed through it agree on each side:
// - left to right below 8 terms, starting from 0.0;
// - up to 128 terms, eight partial sums r0..r7 seeded with the first eight products, extended in
//   steps of 8, combined as ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), then the tail
//   left to right;
// - beyond 128, the two halves (the first rounded down to a multiple of 8) summed alike.
// The _rn intrinsics round every product and sum on its own: NVRTC would otherwise contract a
// product and a sum into a fused multiply-add, which rounds once and changes the bits.
__device__ double pairwise_dot(const double* u, const double* w, int n) {
    if (n < 8) {
        double res = 0.0;
        for (int t = 0; t < n; ++t) res = __dadd_rn(res, __dmul_rn(u[t], w[t]));
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = __dmul_rn(u[j], w[j]);
        int i = 8;
        const int stop = n - n % 8;
        for (; i < stop; i += 8)
            for (int j = 0; j < 8; ++j) r[j] = __dadd_rn(r[j], __dmul_rn(u[i + j], w[i + j]));
        double res = __dadd_rn(__dadd_rn(__dadd_rn(r[0], r[1]), __dadd_rn(r[2], r[3])),
                               __dadd_rn(__dadd_rn(r[4], r[5]), __dadd_rn(r[6], r[7])));
        for (; i < n; ++i) res = __dadd_rn(res, __dmul_rn(u[i], w[i]));
        return res;
    }
    int n2 = n / 2;
    n2 -= n2 % 8;
    // Recursion depth is log2(n / 128) + 1, a few frames for any realistic dimension.
    return __dadd_rn(pairwise_dot(u, w, n2), pairwise_dot(u + n2, w + n2, n - n2));
}

// One thread per row: out[i] = w_i^T u_i for C-contiguous (n, d) U and W, in NumPy's order.
extern "C" __global__ void plane_scores_rows(const double* U, const double* W, double* out, long long n,
                                             int d) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pairwise_dot(U + i * d, W + i * d, d);
}

// One thread per query: u = (x - lower) / width per coordinate, as Domain.normalize computes it
// in two rounded steps; outside[i] if any u lies outside [0, 1] before the clip; U the clipped
// coordinates. A NaN or infinite coordinate sets *nonfinite (every writer stores 1, so the race
// is benign); the caller raises before using any output.
extern "C" __global__ void route_prepare(const double* X, const double* lower, const double* width, double* U,
                                         bool* outside, int* nonfinite, long long n, int d) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    bool out = false;
    for (int j = 0; j < d; ++j) {
        const double x = X[i * d + j];
        if (!isfinite(x)) *nonfinite = 1;
        const double u = __ddiv_rn(__dsub_rn(x, lower[j]), width[j]);
        if (u < 0.0 || u > 1.0) out = true;
        U[i * d + j] = u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u);
    }
    outside[i] = out;
}

// One thread per query: from node start[i] down to its leaf, out[i] the leaf's node. A split's
// right child is child[2 node + 1], taken when w^T u - z >= 0 (the subtraction rounded on its
// own, as NumPy computes plane_scores(u, w) - z); a leaf is its own child, so stopping at it
// gives the node the level-by-level definition reaches. Queries of a warp diverge where their
// paths differ in length; every node read is the query's own path.
extern "C" __global__ void route_walk(const double* U, const long long* start, const bool* is_leaf,
                                      const double* normal, const double* offset, const long long* child,
                                      long long* out, long long n, int d) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const double* u = U + i * d;
    long long node = start[i];
    while (!is_leaf[node]) {
        const double s = pairwise_dot(u, normal + node * d, d);
        node = child[2 * node + (__dsub_rn(s, offset[node]) >= 0.0 ? 1 : 0)];
    }
    out[i] = node;
}
