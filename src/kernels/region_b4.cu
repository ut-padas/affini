// Region mode B4 (src/region.py): soft gates, line-search trials and node selection, residuals, objective
// terms, B3 bin sides.
//
// One B4 iteration on the GPU (_soft_objective_fused, then _soft_step), all node state in place:
//   soft_gates -> segment_cross (both children's Grams) -> spd_solve -> soft_residuals
//   -> soft_row_terms -> segmented sum (J, gradient)            [objective]
//   soft_node_prep -> (soft_trial_terms -> segmented sum -> soft_node_select) x 2 trial groups
// The sequence has no host reads; one synchronization per iteration checks for active nodes.

// GPU forms of the B4 gate and trial loss: the score w^T u, the stable logistic of
// _sigmoid and the floors evaluated in one pass per row (and per line-search trial).
__device__ void gates(const double* u, const double* w, double z, double T, double delta, int d, double* WL,
                      double* WR, double* sp_sm) {
    double s = 0.0;
    for (int j = 0; j < d; ++j) s += u[j] * w[j];
    s = (s - z) / T;
    // Stable logistic: exp(-|s|) never overflows, and sigma(s), sigma(-s) come from the same
    // e. The sign tests mirror _sigmoid(s) and _sigmoid(-s) in NumPy (at s = 0 both are 0.5).
    double e = exp(-fabs(s));
    double big = 1.0 / (1.0 + e), small = e / (1.0 + e);
    double sp = s >= 0 ? big : small, sm = -s >= 0 ? big : small;
    // Floored weights W = delta + (1 - 2 delta) sigma, the same in every fit and loss term
    *WL = delta + (1 - 2 * delta) * sm;
    *WR = delta + (1 - 2 * delta) * sp;
    *sp_sm = sp * sm;
}

// One thread per row: W_L, W_R and sigma sigma' at its node's plane (_soft_gates in region.py).
extern "C" __global__ void soft_gates(const double* U, const long long* lab, const double* w, const double* z,
                                      const double* T, double delta, double* WL, double* WR, double* sp_sm,
                                      long long N, int d) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    long long k = lab[i];
    gates(U + i * d, w + k * d, z[k], T[k], delta, d, WL + i, WR + i, sp_sm + i);
}

// Trial m of a node: alpha 2^-m (exact), w_t = unit(w - a g_w) (NaN if the norm is 0 or not
// finite, as _unit), z_t = z - a g_z. Returns 1 / |w - a g_w| and z_t; w_t_j = (w_j - a g_j) * inv.
__device__ double trial_plane(const double* w, const double* g, double z, double gz, double alpha, int m,
                              int d, double* zt) {
    double a = ldexp(alpha, -m), s = 0.0;  // ldexp: alpha * 2^-m exactly, as repeated halving
    for (int j = 0; j < d; ++j) {
        double v = w[j] - a * g[j];
        s += v * v;
    }
    double norm = sqrt(s);
    *zt = z - a * gz;
    return (isfinite(norm) && norm > 0) ? 1.0 / norm : nan("");
}

// out[i, m] = W_L r_L + W_R r_R at trial m's plane for row i; M trials. One thread per row:
// w_t^T u = inv_m (w^T u - a_m g^T u), so the two dot products are formed once, not per trial.
extern "C" __global__ void soft_trial_terms(const double* U, const long long* lab, const double* w,
                                            const double* g_w, const double* inv_t, const double* z_t,
                                            const double* alpha, const double* T, double delta,
                                            const double* rL, const double* rR, const unsigned char* active,
                                            const unsigned char* finite, const unsigned char* stepped,
                                            double* out, long long N, int d, int M, int m0, int Mt) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    long long k = lab[i];
    double* o = out + i * Mt;
    // Trials m0 .. m0 + Mt - 1 only, and none for a node that cannot use them (inactive, not
    // finite, or already stepped at an earlier trial): its sums are never read.
    if (!active[k] || !finite[k] || stepped[k]) {
        for (int mm = 0; mm < Mt; ++mm) o[mm] = 0.0;
        return;
    }
    const double *wk = w + k * d, *gk = g_w + k * d, *u = U + i * d;
    // 1. The row's two projections, shared by all trials (each trial plane is w - a g, rescaled).
    double uw = 0.0, ug = 0.0;
    for (int j = 0; j < d; ++j) {
        uw += u[j] * wk[j];
        ug += u[j] * gk[j];
    }
    // 2. Frozen-model loss integrand of every trial: the children's residuals rL, rR stay fixed
    //    (majorization argument, design Section 20), only the gates move.
    double Tk = T[k], l = rL[i], r = rR[i];
    for (int mm = 0; mm < Mt; ++mm) {
        int m = m0 + mm;
        double a = ldexp(alpha[k], -m);
        double s = ((uw - a * ug) * inv_t[k * M + m] - z_t[k * M + m]) / Tk;
        double e = exp(-fabs(s)), big = 1.0 / (1.0 + e), small = e / (1.0 + e);
        double sp = s >= 0 ? big : small, sm = -s >= 0 ? big : small;
        o[mm] = (delta + (1 - 2 * delta) * sm) * l + (delta + (1 - 2 * delta) * sp) * r;
    }
}

// Per node before the trials: project g_w to the tangent of |w| = 1 (in place), the step
// alpha = T / |g|, the descent margin and whether the node may step at all.
extern "C" __global__ void soft_node_prep(const double* w, double* g_w, const double* z, const double* g_z,
                                          const double* J, const double* gap, const double* T,
                                          const unsigned char* active, double* gn2, double* alpha,
                                          double* margin, unsigned char* finite, double* inv_t, double* z_t,
                                          int K, int d, int M, double eps) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= K) return;
    // 1. Riemannian gradient on |w| = 1: remove the radial component g.w w.
    double* g = g_w + (long long)k * d;
    const double* wk = w + (long long)k * d;
    double dot = 0.0;
    for (int j = 0; j < d; ++j) dot += g[j] * wk[j];
    double s = 0.0;
    for (int j = 0; j < d; ++j) {
        g[j] -= dot * wk[j];
        s += g[j] * g[j];
    }
    // 2. First trial moves the gate about one width T; the margin is the inner solves' estimated
    //    suboptimality plus rounding of J, which a step must beat to count as descent.
    s += g_z[k] * g_z[k];
    gn2[k] = s;
    alpha[k] = T[k] / sqrt(fmax(s, 1e-300));
    margin[k] = gap[k] + 64 * eps * (fabs(J[k]) + 1e-300);
    finite[k] = isfinite(J[k]) && isfinite(gap[k]) && isfinite(s);
    for (int m = 0; m < M; ++m)  // every trial's plane, once per node: shared by the terms and the selection
        inv_t[(long long)k * M + m] =
            trial_plane(wk, g, z[k], g_z[k], alpha[k], m, d, z_t + (long long)k * M + m);
}

// Per node after a group of trials m0 .. m0 + Mt - 1 (S holds their sums): a node that has not
// stepped takes its first trial that passes the Armijo and margin tests. Trials run in groups
// so that later halvings are evaluated only for nodes still waiting (98% of steps are taken by
// trial 3); the chosen step is the sequential halving loop's. After the last group (final),
// nodes that never stepped are classified and active is updated. stops (int64, 4) counts
// nonfinite / saturated / converged / stalled.
extern "C" __global__ void soft_node_select(double* w, double* z, const double* g_w, const double* inv_t,
                                            const double* z_t, const double* S, const double* pen,
                                            const double* n, const double* J, const double* gn2,
                                            const double* alpha, const double* margin,
                                            const unsigned char* finite, const double* sp_sum,
                                            const double* T, unsigned char* active, unsigned char* stepped,
                                            unsigned long long* stops, int K, int d, int M, int m0, int Mt,
                                            int final) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= K) return;
    bool act = active[k], fin = finite[k], taken = stepped[k];
    // 1. Largest remaining step first.
    if (act && fin && !taken) {
        for (int mm = 0; mm < Mt && !taken; ++mm) {
            int m = m0 + mm;
            double Ft = (S[(long long)k * Mt + mm] + pen[k]) / n[k];
            double a = ldexp(alpha[k], -m);
            if (Ft <= J[k] - 1e-4 * a * gn2[k] && J[k] - Ft > margin[k]) {
                double* wk = w + (long long)k * d;
                const double* gk = g_w + (long long)k * d;
                double inv = inv_t[(long long)k * M + m];
                for (int j = 0; j < d; ++j) wk[j] = (wk[j] - a * gk[j]) * inv;
                z[k] = z_t[(long long)k * M + m];
                taken = true;
            }
        }
        stepped[k] = taken;
    }
    if (!final) return;
    // 2. A node that could not step stops for good; record why, in the CPU path's precedence.
    if (act && !taken) {
        bool saturated = fin && (sp_sum[k] / n[k] < 1e-6);
        bool flat = fin && !saturated && (T[k] * sqrt(gn2[k]) <= margin[k]);
        int c = !fin ? 0 : saturated ? 1 : flat ? 2 : 3;
        atomicAdd(stops + c, 1ULL);  // integer counts: the order cannot change them
    }
    active[k] = act && taken;
}

// Prefix and suffix sums over the m + 1 bins of every (node, entry): L[:, i] = S[:, 0..i] and
// R[:, i] = S[:, i+1..m] (summed from the last bin), i < m; written directly, no scan temporaries.
extern "C" __global__ void bin_sides(const double* S, double* L, double* R, long long K, int m, long long X) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= K * X) return;
    long long k = t / X, x = t - k * X;
    const double* s = S + k * (m + 1) * X + x;
    double acc = 0.0;
    for (int i = 0; i < m; ++i) {
        acc += s[(long long)i * X];
        L[(k * m + i) * X + x] = acc;
    }
    acc = 0.0;
    for (int i = m; i >= 1; --i) {
        acc += s[(long long)i * X];
        R[(k * m + i - 1) * X + x] = acc;
    }
}

// Residuals of both children's fits: E[s, i, :] = Y[i, :] - Phi[i, :] beta[s, lab[i]], r[s, i] = |E[s, i]|^2.
extern "C" __global__ void soft_residuals(const double* Phi, const double* Y, const long long* lab,
                                          const double* beta, double* E, double* r, long long N, int p,
                                          int fd, int K) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= 2 * N) return;
    long long sd = t / N, i = t - sd * N;  // side (0 = left, 1 = right) and row
    // beta is read in place by node: no (N, p, fd) gathered copy of the coefficients
    const double* b = beta + (sd * K + lab[i]) * p * fd;
    double acc = 0.0;
    for (int f = 0; f < fd; ++f) {
        double v = 0.0;
        for (int c = 0; c < p; ++c) v += Phi[i * p + c] * b[c * fd + f];
        double e = Y[i * fd + f] - v;
        E[(sd * N + i) * fd + f] = e;
        acc += e * e;
    }
    r[t] = acc;
}

// Per-row terms summed per node for the objective and its gradient (one segmented sum of
// all d + 3 columns instead of four reductions). By the envelope theorem only the gates
// carry the gradient: dJ/ds_i = (r_R - r_L)(1 - 2 delta) sigma sigma' / (T n).
// [W_L r_L + W_R r_R, a u_1 .. a u_d, a, sigma sigma'] with a = (r_R - r_L)(1 - 2 delta) sigma sigma' / (T
// n).
extern "C" __global__ void soft_row_terms(const double* U, const long long* lab, const double* WL,
                                          const double* WR, const double* sp_sm, const double* r,
                                          const double* T, const double* n, double delta, double* out,
                                          long long N, int d) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    long long k = lab[i];
    double rL = r[i], rR = r[N + i];
    double a = (rR - rL) * (1 - 2 * delta) * sp_sm[i] / T[k] / n[k];
    double* o = out + i * (d + 3);
    o[0] = WL[i] * rL + WR[i] * rR;
    for (int j = 0; j < d; ++j) o[1 + j] = a * U[i * d + j];
    o[d + 1] = a;
    o[d + 2] = sp_sm[i];
}
