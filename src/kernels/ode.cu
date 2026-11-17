// Batched ensemble integration of autonomous ODEs for src/numerics/ode.py (integrate_ensemble):
// explicit embedded Runge-Kutta pairs with SciPy's RK45 step-size control.

#define ODE_MAX_STAGES 7
#define ODE_MAX_COMPONENTS_PER_LANE 32  // components one lane carries: ceil(d / L)
#define ODE_MAX_STEPS 1000000           // a trajectory that needs more steps returns NaN

// One right-hand side per problem: component j of dx/dt at the state x (all d components, in shared
// memory). Adding a problem = one such function and one pair of entry points at the end of the file.
__device__ __forceinline__ double lorenz96_rhs(const double* x, int j, int d, const double* params) {
    int jp1 = j + 1 == d ? 0 : j + 1, jm1 = j == 0 ? d - 1 : j - 1, jm2 = j >= 2 ? j - 2 : j - 2 + d;
    return (x[jp1] - x[jm2]) * x[jm1] - x[j] + params[0];
}

// Sum over the L lanes of a group with a fixed butterfly, so every lane holds the same bits.
template <int L>
__device__ __forceinline__ double group_sum(double v) {
#pragma unroll
    for (int off = L / 2; off > 0; off /= 2) v += __shfl_xor_sync(0xffffffffu, v, off, L);
    return v;
}

template <int L>
__device__ __forceinline__ void group_sync() {
    if (L > 1) __syncwarp();
}

// One trajectory per group of L consecutive threads (L = 1: one per thread; L = 32: one per warp, for
// large d). Lane l owns components j = l, l + L, ...; the stage inputs go through the group's d doubles
// of shared memory, because a component's derivative reads its neighbours. Every control decision
// (accept, step size, the last step) is taken from group-reduced values, so all lanes of a group follow
// the same path and the group never diverges. A trajectory's bits do not depend on the other
// trajectories of the launch. Butcher tableau: c (s), A (s x s, row-major), b (s), e = b - b_hat (s);
// fsal: the last stage is evaluated at the new solution and serves as the next step's first stage.
template <int L, typename RHS>
__device__ void ensemble_rk(RHS rhs, const double* __restrict__ X0, double* __restrict__ X1, long long n,
                            int d, double t1, double rtol, double atol, const double* __restrict__ A,
                            const double* __restrict__ b, const double* __restrict__ e, int s, int fsal,
                            int order, const double* __restrict__ params) {
    extern __shared__ double shared[];
    const int lane = threadIdx.x % L, group = threadIdx.x / L;
    const long long traj = (long long)blockIdx.x * (blockDim.x / L) + group;
    const bool active = traj < n;
    double* x = shared + (long long)group * d;
    const int m_count = (d + L - 1) / L;
    double y[ODE_MAX_COMPONENTS_PER_LANE], yn[ODE_MAX_COMPONENTS_PER_LANE];
    double k[ODE_MAX_STAGES][ODE_MAX_COMPONENTS_PER_LANE];

    // 1. Load the state and f(y0).
    for (int m = 0; m < m_count; ++m) {
        int j = lane + L * m;
        y[m] = (active && j < d) ? X0[traj * d + j] : 0.0;
        if (j < d) x[j] = y[m];
    }
    group_sync<L>();
    for (int m = 0; m < m_count; ++m) {
        int j = lane + L * m;
        k[0][m] = j < d ? rhs(x, j, d, params) : 0.0;
    }
    group_sync<L>();

    // 2. Initial step by Hairer's rule (Solving ODEs I, II.4): from |y0|, |f0| and |f1 - f0| in the
    // scaled norm, with f1 at an explicit Euler step of 0.01 |y0| / |f0|.
    double sy = 0.0, sf = 0.0;
    for (int m = 0; m < m_count; ++m) {
        int j = lane + L * m;
        if (j < d) {
            double sc = atol + rtol * fabs(y[m]);
            sy += (y[m] / sc) * (y[m] / sc);
            sf += (k[0][m] / sc) * (k[0][m] / sc);
        }
    }
    double d0 = sqrt(group_sum<L>(sy) / d), d1 = sqrt(group_sum<L>(sf) / d);
    double h0 = fmin((d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1, t1);
    for (int m = 0; m < m_count; ++m) {
        int j = lane + L * m;
        if (j < d) x[j] = y[m] + h0 * k[0][m];
    }
    group_sync<L>();
    double sd = 0.0;
    for (int m = 0; m < m_count; ++m) {
        int j = lane + L * m;
        if (j < d) {
            double sc = atol + rtol * fabs(y[m]);
            double df = (rhs(x, j, d, params) - k[0][m]) / sc;
            sd += df * df;
        }
    }
    group_sync<L>();
    double d2 = sqrt(group_sum<L>(sd) / d) / h0, dm = fmax(d1, d2);
    double h1 = dm <= 1e-15 ? fmax(1e-6, h0 * 1e-3) : pow(0.01 / dm, 1.0 / order);
    double h = fmin(fmin(100.0 * h0, h1), t1);

    // 3. Steps: stages, the embedded error, and the step-size update, each rule as in SciPy's RK45 (safety
    // 0.9, factor in [0.2, 10], no growth right after a rejection, failure below ten ulps of t), so a
    // trajectory takes SciPy's steps up to rounding.
    double t = 0.0;
    long long steps = 0;
    bool failed = false, rejected = false;
    while (t < t1) {
        const double min_step = 10.0 * (nextafter(t, INFINITY) - t);
        if (h < min_step) {
            if (rejected) {
                failed = true;
                break;
            }
            h = min_step;
        }
        bool last = t + h >= t1;
        if (last) h = t1 - t;  // the last step lands exactly on t1
        for (int i = 1; i < s; ++i) {
            for (int m = 0; m < m_count; ++m) {
                int j = lane + L * m;
                if (j < d) {
                    double acc = 0.0;
                    for (int q = 0; q < i; ++q) acc += A[i * s + q] * k[q][m];
                    x[j] = y[m] + h * acc;
                }
            }
            group_sync<L>();
            for (int m = 0; m < m_count; ++m) {
                int j = lane + L * m;
                k[i][m] = j < d ? rhs(x, j, d, params) : 0.0;
            }
            group_sync<L>();
        }
        double se = 0.0;
        for (int m = 0; m < m_count; ++m) {
            int j = lane + L * m;
            double accb = 0.0, acce = 0.0;
            for (int q = 0; q < s; ++q) {
                accb += b[q] * k[q][m];
                acce += e[q] * k[q][m];
            }
            yn[m] = y[m] + h * accb;
            if (j < d) {
                double r = h * acce / (atol + rtol * fmax(fabs(y[m]), fabs(yn[m])));
                se += r * r;
            }
        }
        double err = sqrt(group_sum<L>(se) / d);
        if (!(err == err) || ++steps > ODE_MAX_STEPS) {  // a NaN error or no progress: give up
            failed = true;
            break;
        }
        if (err < 1.0) {
            t = last ? t1 : t + h;
            for (int m = 0; m < m_count; ++m) {
                y[m] = yn[m];
                if (fsal) k[0][m] = k[s - 1][m];
            }
            if (!fsal) {  // a method without FSAL needs f at the new state
                for (int m = 0; m < m_count; ++m) {
                    int j = lane + L * m;
                    if (j < d) x[j] = y[m];
                }
                group_sync<L>();
                for (int m = 0; m < m_count; ++m) {
                    int j = lane + L * m;
                    k[0][m] = j < d ? rhs(x, j, d, params) : 0.0;
                }
                group_sync<L>();
            }
            const double factor = err == 0.0 ? 10.0 : fmin(10.0, 0.9 * pow(err, -1.0 / order));
            h *= rejected ? fmin(1.0, factor) : factor;
            rejected = false;
        } else {
            h *= fmax(0.2, 0.9 * pow(err, -1.0 / order));
            rejected = true;
        }
    }

    // 4. Store; a failed trajectory is all NaN.
    if (active) {
        for (int m = 0; m < m_count; ++m) {
            int j = lane + L * m;
            if (j < d) X1[traj * d + j] = failed ? nan("") : y[m];
        }
    }
}

struct Lorenz96 {
    __device__ __forceinline__ double operator()(const double* x, int j, int d, const double* p) const {
        return lorenz96_rhs(x, j, d, p);
    }
};

// Entry points: one trajectory per thread (d <= ODE_MAX_COMPONENTS_PER_LANE) and one per warp.
extern "C" __global__ void ode_lorenz96_thread(const double* X0, double* X1, long long n, int d, double t1,
                                               double rtol, double atol, const double* A, const double* b,
                                               const double* e, int s, int fsal, int order,
                                               const double* params) {
    ensemble_rk<1>(Lorenz96(), X0, X1, n, d, t1, rtol, atol, A, b, e, s, fsal, order, params);
}

extern "C" __global__ void ode_lorenz96_warp(const double* X0, double* X1, long long n, int d, double t1,
                                             double rtol, double atol, const double* A, const double* b,
                                             const double* e, int s, int fsal, int order,
                                             const double* params) {
    ensemble_rk<32>(Lorenz96(), X0, X1, n, d, t1, rtol, atol, A, b, e, s, fsal, order, params);
}
