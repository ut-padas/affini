// ART sampling (src/art/sampling.py, HitAndRunSampler._walk_fused): whole hit-and-run walks, one
// thread per chain.

// Whole hit-and-run walks, one thread per chain (``_walk_fused``), mirroring ``_steps``: the direction
// from the same uniforms, the box chord, the carried-constraint and Voronoi-bisector chords
// (slacks clamped at 0, faces with a zero rate skipped), the feasibility checks and the
// move. Reductions propagate NaN like np.max/np.min, so a NaN chord end flags the chain.
#define MAXD 256
#define INFINITY __longlong_as_double(0x7ff0000000000000LL)
__device__ inline double nanmax(double a, double b) { return (isnan(a) || isnan(b)) ? nan("") : fmax(a, b); }
__device__ inline double nanmin(double a, double b) { return (isnan(a) || isnan(b)) ? nan("") : fmin(a, b); }
// splitmix64 finalizer: must stay identical to src/art/rng.py _mix64, since every uniform of the
// walk is a hash of (node key, purpose tag, counter) shared with the NumPy sampler.
__device__ inline unsigned long long mix64(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Slack checks at u: any constraint or bisector violated beyond rounding.
__device__ bool infeasible(const double* u, const double* cons, int h, int d, const double* C,
                           const double* c_r, int R, long long vroot, double tol) {
    for (int k = 0; k < h; ++k) {
        const double* a = cons + (long long)k * (d + 1);
        double s = 0.0;
        for (int j = 0; j < d; ++j) s += a[j] * u[j];
        if (a[d] - s < -tol * (1.0 + fabs(a[d]))) return true;
    }
    for (int rr = 0; rr < R; ++rr) {
        if (rr == vroot) continue;  // the owning centre's own bisector does not exist
        // Bisector of centres c_r, c_j in distance units: n.(u - c_r) <= |c_j - c_r| / 2 with
        // n = (c_j - c_r) / |c_j - c_r|. Formed from the centre difference, not from dot products
        // and squared norms, which cancel for close centres (design 42.7).
        const double* cj = C + (long long)rr * d;
        double nrm = 0.0, s = 0.0;
        for (int j = 0; j < d; ++j) {
            double D = cj[j] - c_r[j];
            nrm += D * D;
        }
        nrm = sqrt(nrm);
        for (int j = 0; j < d; ++j) s += ((cj[j] - c_r[j]) / nrm) * (u[j] - c_r[j]);
        if (0.5 * nrm - s < -tol) return true;
    }
    return false;
}

// One thread per chain: steps t_from .. t_to - 1, then the endpoint check. bad[i] is set when
// a step starts on, or the walk ends on, a violated face (beyond tol), or a chord is empty or
// not finite; a bad chain stops where it is. Uniforms of step t are hashes q + 1 .. of the
// chain's counter block, exactly as HitAndRunSampler._uniforms draws them.
extern "C" __global__ void hr_walk(double* U, bool* bad, const long long* owner, const long long* draw,
                                   const unsigned long long* keys, const double* cons_all,
                                   const long long* root, const double* lo_all, const double* hi_all,
                                   const double* C, long long n, int d, int h, int R, int q, int coordinate,
                                   int t_from, int t_to, long long block, long long base,
                                   unsigned long long tag, double tol) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double* u = U + i * d;
    const long long k = owner[i], rt = root[i];
    const double* cons = cons_all + k * (long long)h * (d + 1);
    const double* lo = lo_all + rt * d;
    const double* hi = hi_all + rt * d;
    const double* c_r = R ? C + rt * d : C;
    const unsigned long long key = keys[k];
    // Node constraints, root bounds and Voronoi centres are read in place (shared by the node's
    // chains, so cached): no per-chain copies, which at high dimension dominated memory.
    bool bd = false;
    double g[MAXD], r[MAXD + 2];
    for (int t = t_from; t < t_to; ++t) {
        // 1. The step starts where the last one ended; a start outside a face flags the chain. The
        //    slacks for that check are the chord's own (step 4), computed once with the same expressions.
        // 2. The 2q + 1 uniforms of step t: q + 1 hashes, high then low 32 bits (counter_uniform_pairs)
        long long ctr0 = draw[i] * block + base + (long long)t * (q + 1);
        for (int p = 0; p <= q; ++p) {
            unsigned long long z = mix64(key ^ mix64((unsigned long long)(ctr0 + p) + tag));
            r[2 * p] = ((double)(z >> 32) + 0.5) * 2.3283064365386963e-10;
            r[2 * p + 1] = ((double)(z & 0xFFFFFFFFULL) + 0.5) * 2.3283064365386963e-10;
        }
        // 3. Direction: systematic axis sweeps on the recovery stream (every d steps touch every axis),
        //    else an isotropic unit vector from Box-Muller pairs (cosine then sine halves, as NumPy).
        if (coordinate) {
            for (int j = 0; j < d; ++j) g[j] = j == t % d ? 1.0 : 0.0;
        } else {
            // Each pair's radius and angle once: both halves of a pair use the same values.
            for (int p = 0; p < q; ++p) {
                double radius = sqrt(-2.0 * log(r[p])), angle = 2.0 * 3.141592653589793 * r[q + p];
                g[p] = radius * cos(angle);
                if (q + p < d) g[q + p] = radius * sin(angle);
            }
            double nn = 0.0;  // summed in coordinate order, as before
            for (int j = 0; j < d; ++j) nn += g[j] * g[j];
            nn = sqrt(nn);
            for (int j = 0; j < d; ++j) g[j] /= nn;
        }
        // 4. Chord {t : u + t g feasible}: the box gives [max fmin, min fmax] of the face crossings
        //    (fmin/fmax ignore one NaN, like np.fmin/fmax); then every half-space with a nonzero
        //    rate ag along g cuts it at slack / ag. Slacks are clamped at 0 so rounding just
        //    outside a face still leaves u inside its own chord (feasibility is checked on the same slacks).
        double t_min = -INFINITY, t_max = INFINITY;
        for (int j = 0; j < d; ++j) {
            double a = (lo[j] - u[j]) / g[j], c = (hi[j] - u[j]) / g[j];
            if (coordinate && g[j] == 0.0) {
                a = -INFINITY;
                c = INFINITY;
            }
            t_min = j ? nanmax(t_min, fmin(a, c)) : fmin(a, c);
            t_max = j ? nanmin(t_max, fmax(a, c)) : fmax(a, c);
        }
        if (h > 0) {
            double up = INFINITY, dn = -INFINITY;
            for (int kk = 0; kk < h; ++kk) {
                const double* a = cons + (long long)kk * (d + 1);
                double s = 0.0, ag = 0.0;
                for (int j = 0; j < d; ++j) {
                    s += a[j] * u[j];
                    ag += a[j] * g[j];
                }
                double slack = a[d] - s;
                if (slack < -tol * (1.0 + fabs(a[d]))) bd = true;  // step 1's feasibility check
                double ratio = isnan(slack) ? nan("") : fmax(slack, 0.0) / ag;
                if (ag > 0)
                    up = nanmin(up, ratio);
                else if (ag < 0)
                    dn = nanmax(dn, ratio);
            }
            t_max = nanmin(t_max, up);
            t_min = nanmax(t_min, dn);
        }
        if (R > 0) {
            double up = INFINITY, dn = -INFINITY;
            for (int rr = 0; rr < R; ++rr) {
                if (rr == rt) continue;
                const double* cj = C + (long long)rr * d;
                double nrm = 0.0, s = 0.0, ag = 0.0;
                for (int j = 0; j < d; ++j) {
                    double D = cj[j] - c_r[j];
                    nrm += D * D;
                }
                nrm = sqrt(nrm);
                for (int j = 0; j < d; ++j) {
                    double nj = (cj[j] - c_r[j]) / nrm;
                    s += nj * (u[j] - c_r[j]);
                    ag += nj * g[j];
                }
                double slack = 0.5 * nrm - s;
                if (slack < -tol) bd = true;  // step 1's feasibility check
                double ratio = isnan(slack) ? nan("") : fmax(slack, 0.0) / ag;
                if (ag > 0)
                    up = nanmin(up, ratio);
                else if (ag < 0)
                    dn = nanmax(dn, ratio);
            }
            t_max = nanmin(t_max, up);
            t_min = nanmax(t_min, dn);
        }
        // 5. Move to a uniform point of the chord; a chain with an empty or non-finite chord stops.
        if (!(isfinite(t_min) && isfinite(t_max) && t_min <= t_max)) bd = true;
        if (!bd) {
            double step = t_min + (t_max - t_min) * r[2 * q];
            for (int j = 0; j < d; ++j) u[j] += step * g[j];
        }
    }
    if (infeasible(u, cons, h, d, C, c_r, R, rt, tol)) bd = true;  // the returned endpoint too
    bad[i] = bd;
}
