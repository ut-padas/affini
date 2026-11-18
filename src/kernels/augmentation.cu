// AFINNI's augmentation of rejected clusters (src/components/sampling.py): convex combinations.

// One thread per output element (r, c) of the (n, d) row-major result: X[i1[r], c] * a[r] +
// X[i2[r], c] * (1 - a[r]). The _rn intrinsics forbid fused multiply-adds, so every element
// rounds as the separate NumPy and CuPy operations do and the output is bit-identical to them.
extern "C" __global__ void convex_combination(const double* X, const long long* i1, const long long* i2,
                                              const double* a, long long d, long long num_elements,
                                              double* out) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_elements) return;
    const long long r = i / d, c = i % d;
    const double w = a[r];
    out[i] = __dadd_rn(__dmul_rn(X[i1[r] * d + c], w), __dmul_rn(X[i2[r] * d + c], __dsub_rn(1.0, w)));
}
