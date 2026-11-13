// Per-cluster coordinate sums for src/neighbors/kmeans.py.

// One block per cluster; threads stride over dimensions and add the cluster's rows
// (``order[starts[c]:starts[c+1]]``) in a fixed order, so sums are reproducible.
extern "C" __global__ void centroid_sums(const double* __restrict__ X, const long long* __restrict__ order,
                                         const long long* __restrict__ starts, int d,
                                         double* __restrict__ sums) {
    const int c = blockIdx.x;
    const long long b = starts[c], e = starts[c + 1];
    for (int t = threadIdx.x; t < d; t += blockDim.x) {
        double s = 0.0;
        for (long long r = b; r < e; ++r) s += X[order[r] * d + t];
        sums[(long long)c * d + t] = s;
    }
}
