// The kernel ridge correction of KRR-enriched leaves at prediction time (src/components/local_models.py:
// KRREnrichedModel.correction on the GPU; _cpu_krr_correction is the same loop in Numba).

// The kernel of KRR_KERNELS (src/components/local_models.py) with id kid, at q = |z - c|^2, the
// squared distance in the leaf's trust metric: 0 gaussian, 1 matern32, 2 matern52. A new kernel is
// added here, in the Numba loop and in KRR_KERNELS, with the same id.
__device__ double krr_kernel(double q, int kid) {
    if (kid == 0) return exp(-0.5 * q);
    const double r = sqrt(q);
    if (kid == 1) {
        const double s = 1.7320508075688772 * r;  // sqrt(3) r
        return (1.0 + s) * exp(-s);
    }
    const double s = 2.23606797749979 * r;  // sqrt(5) r
    return (1.0 + s + (5.0 / 3.0) * q) * exp(-s);
}

// One thread per query i, leaf k = leaf[i]: z = W_k x_i (written to Z[i], the thread's own row), then
// out[i] = sum_j alpha_kj kernel(|z - c_kj|^2) over the leaf's p centres, every sum in index order.
// Each output's bits depend on the query and its leaf only, not on the batch. Not bit-identical to the
// NumPy reference (KRR_KERNELS on whole arrays): the compiler may fuse products and sums, which
// training does not depend on, since finalize fits from its own kernel matrix.
extern "C" __global__ void krr_correction(const double* X, const long long* leaf, const double* W,
                                          const double* C, const double* alpha, double* Z, double* out,
                                          long long n, int d, int p, int fd, int kid) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const long long k = leaf[i];
    const double* x = X + i * d;
    const double* Wk = W + k * d * d;
    double* z = Z + i * d;
    // 1. Whiten the query: z = W_k x, so q = (x - y)^T T_k (x - y) = |z - W_k y|^2.
    for (int a = 0; a < d; ++a) {
        double s = 0.0;
        for (int c = 0; c < d; ++c) s += Wk[a * d + c] * x[c];
        z[a] = s;
    }
    // 2. Accumulate alpha_j phi_j per output over the centres (stored whitened, W_k c_j).
    double* o = out + i * fd;
    for (int f = 0; f < fd; ++f) o[f] = 0.0;
    const double* Ck = C + k * (long long)p * d;
    const double* Ak = alpha + k * (long long)p * fd;
    for (int j = 0; j < p; ++j) {
        double q = 0.0;
        for (int c = 0; c < d; ++c) {
            const double t = z[c] - Ck[j * d + c];
            q += t * t;
        }
        const double phi = krr_kernel(q, kid);
        for (int f = 0; f < fd; ++f) o[f] += phi * Ak[j * fd + f];
    }
}
