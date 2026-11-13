"""Uniform sampling inside a node's polytope by hit-and-run (CUDA and Numba walks, NumPy fallback)."""
import math
from dataclasses import dataclass

import numpy as np

from src import IS_CUPY_BACKEND, Array, kernels, xp
from src.art.geometry import Domain, Roots
from src.art.rng import _GOLDEN, _MIX1, _MIX2, counter_uniform_pairs
from src.art.split_search import segment_sum
from src.components.sampling import ClusterPoints, Sampler
from src.numerics import (
    det_einsum,
    repeats,
)


# --- Sampling inside polytopes -------------------------------------------------------
@dataclass
class NodePolytopes:
    """Per-node geometry and random stream of a batch, for ``HitAndRunSampler``."""

    root: Array  # (K,) int64
    depth: Array  # (K,) int64
    keys: Array  # (K,) uint64 stream keys
    constraints: Array  # (K, h, d + 1) half-spaces [a | b], a^T u <= b, zero-padded
    purpose: int
    insufficient: Array | None = None  # (K,) set by the sampler: chains failed the drift check


class HitAndRunSampler(Sampler):
    """
    Points inside each node's polytope (root cell plus ancestor half-spaces).

    A box or grid root is sampled exactly (uniform). Otherwise each new point is the
    endpoint of its own hit-and-run chain of ``walk_steps`` steps, started at a uniformly
    chosen inherited point (a Voronoi root without points starts at its centre):
    endpoints are conditionally independent but share their starts' bias, so they are
    approximately, not exactly, uniform. Voronoi bisectors are evaluated against every
    centre at each step, so the chord is exact without a stored face list. Not registered
    in ``SAMPLERS``: it needs ``ClusterPoints.polytopes``, which only ART provides.
    Randomness comes from the node keys, not from ``rng``.
    """

    name = "hit_and_run"
    # Bound the (rows, h + R, d + 1) constraint temporaries of one chunk of chains.
    CHUNK_ELEMENTS = 1 << 23

    def __init__(self, domain: Domain, roots: Roots, walk_steps: int, max_walk_steps: int | None = None,
                 mixing_z: float = 4.0):  # fmt: skip
        self.domain, self.roots, self.walk_steps = domain, roots, int(walk_steps)
        self.max_walk_steps = int(max_walk_steps or 4 * walk_steps)
        self.mixing_z = float(mixing_z)
        self._lo, self._hi = roots.bounds()
        if roots.is_voronoi:
            self._C = xp.asarray(roots.centres)

    def settings(self) -> dict:
        return {"walk_steps": self.walk_steps, "max_walk_steps": self.max_walk_steps, "mixing_z": self.mixing_z}

    def sample(self, clusters: ClusterPoints, n_new: Array, rng=None) -> Array:
        """
        New points. Every chain runs ``walk_steps`` steps; a node whose endpoints at the
        halfway step and at the end differ (any coordinate or the squared radius,
        standardized mean difference above ``mixing_z``) has its chains continued,
        doubling their length, up to ``max_walk_steps``. The returned points are those same
        chains' endpoints: this is a drift check, not a separate pilot and production run,
        and not a proof of mixing (design Section 33.4).

        Two failures the drift check cannot see are handled explicitly. Every chain that did
        not move (e.g. from a seed in a corner, where an isotropic direction escapes with
        probability ``2^(1-d)``) is rerun with coordinate directions, which escape box
        corners and leave the uniform law invariant, on a separate random stream; a node
        with any chain still unmoved is flagged. A start, step or returned endpoint that
        violates a face by more than rounding is never silently relaxed. Nodes with
        either problem remaining are flagged in ``polytopes.insufficient``. The recovery
        depends on the first walk's outcome, so it is an adaptive repair, not an
        independent production run.
        """
        geo: NodePolytopes = clusters.polytopes
        d, K = clusters.d, clusters.n_clusters
        geo.insufficient = xp.zeros(K, dtype=bool)
        n_total = int(xp.sum(n_new))
        if n_total == 0:
            return xp.empty((0, d), dtype=xp.float64)
        owner = repeats(xp.arange(K), n_new)
        draw = xp.arange(n_total, dtype=xp.int64) - (xp.cumsum(n_new) - n_new)[owner]
        U, walk = self._starts(clusters, geo, owner, draw)
        if bool(xp.any(walk)):
            U0 = U
            bad = xp.zeros(n_total, dtype=bool)

            def run(U, rows, bad, coordinate):
                """Walk ``rows``, doubling while their node drifts; returns (U, bad, drift)."""
                # A coordinate recovery runs at least two full sweeps when the budget allows.
                t_end = min(max(self.walk_steps, 2 * d), self.max_walk_steps) if coordinate else self.walk_steps
                U_mid, bad = self._walk(clusters, geo, owner, draw, U, rows, 0, t_end // 2, bad, coordinate)
                U, bad = self._walk(clusters, geo, owner, draw, U_mid, rows, t_end // 2, t_end, bad, coordinate)
                while True:
                    drift = self._drifting(U_mid, U, owner, rows, n_new)
                    if not bool(xp.any(drift)) or 2 * t_end > self.max_walk_steps:
                        return U, bad, drift
                    U_mid = U
                    U, bad = self._walk(clusters, geo, owner, draw, U, rows & drift[owner], t_end, 2 * t_end, bad, coordinate)
                    t_end *= 2

            U, bad, drift = run(U, walk, bad, coordinate=False)
            # A chain that left any coordinate at its start value has not left the start's
            # face: a full-dimensional move changes every coordinate with probability 1, so
            # "moved at all" would accept points still on a zero-volume face of the cell.
            unmoved = walk & xp.any(U == U0, axis=1)
            if bool(xp.any(unmoved)):
                # Every such chain is rerun, not only nodes where most are stuck: a node with
                # 40% of its points on a corner (a zero-volume atom) is as wrong. The recovery
                # walk must pass the same drift check, on its own chains: moving is not mixing.
                touched = _per_node_sum(unmoved.astype(xp.float64), n_new) > 0
                U, bad, recovery_drift = run(U, unmoved, bad, coordinate=True)
                drift = xp.where(touched, recovery_drift | drift, drift)
            stuck = _per_node_sum((walk & xp.any(U == U0, axis=1)).astype(xp.float64), n_new) > 0
            infeasible = _per_node_sum(bad.astype(xp.float64), n_new) > 0
            geo.insufficient = drift | stuck | infeasible
        return self.domain.denormalize(U)

    def _starts(self, clusters: ClusterPoints, geo: NodePolytopes, owner: Array, draw: Array) -> tuple[Array, Array]:
        """
        Normalized start points, and which of them walk: exact uniform points of a box
        root cell (final as they are), otherwise a uniformly chosen inherited point (a
        Voronoi root without points starts at its centre, which lies in its own cell).
        """
        d = clusters.d
        root = geo.root[owner]
        lo, hi = self._lo[root], self._hi[root]
        U = lo + (hi - lo) * self._uniforms(geo, owner, draw, 0, d)  # hashes [0, ceil(d/2))
        walk = (geo.depth[owner] > 0) | self.roots.is_voronoi
        if not bool(xp.any(walk)):
            return U, walk
        counts = clusters.counts[owner]
        pick = xp.minimum((self._uniforms(geo, owner, draw, self._start_hashes, 1)[:, 0] * counts).astype(xp.int64),
                          xp.maximum(counts - 1, 0))  # fmt: skip
        has_seed = counts > 0
        seeds = self.domain.normalize(clusters.X[xp.where(has_seed, clusters.starts[owner] + pick, 0)]) \
            if clusters.X.shape[0] else U
        if self.roots.is_voronoi:
            seeds = xp.where(has_seed[:, None], seeds, self._C[root])
        return xp.where(walk[:, None], seeds, U), walk

    @property
    def _start_hashes(self) -> int:
        return (self.domain.d + 1) // 2  # the start point's d uniforms

    @property
    def _step_hashes(self) -> int:
        return (self.domain.d + 1) // 2 + 1  # ceil(d/2) Box-Muller pairs, then the chord position

    def _uniforms(self, geo: NodePolytopes, owner: Array, draw: Array, offset: int, count: int) -> Array:
        """
        (n, count) uniforms of each new point's stream, from ``ceil(count / 2)`` hashes at
        hash ``offset``; the per-point layout (start, pick, then each step) is fixed by
        ``max_walk_steps``, so continuing a chain reuses no number.
        """
        block = self._start_hashes + 1 + self.max_walk_steps * self._step_hashes
        counter = (draw * block)[:, None] + offset + xp.arange((count + 1) // 2)
        return counter_uniform_pairs(geo.keys[owner][:, None], geo.purpose, counter).reshape(len(owner), -1)[:, :count]

    def _walk(self, clusters, geo, owner, draw, U, rows, t_from: int, t_to: int, bad: Array,
              coordinate: bool = False) -> tuple[Array, Array]:  # fmt: skip
        """
        ``U`` after hit-and-run steps ``t_from .. t_to - 1`` of the chains in ``rows``, in
        bounded chunks, and ``bad`` with the chains that met an infeasible face added.
        ``coordinate`` uses coordinate directions on the separate recovery stream.
        """
        idx = xp.nonzero(rows)[0]
        if int(idx.shape[0]) == 0 or t_to <= t_from:
            return U, bad
        d = clusters.d
        if d <= 256 or not IS_CUPY_BACKEND:  # one fused walk per chain (CUDA or Numba); memory O(d), no chunking
            U, bad = U.copy(), bad.copy()
            U_sel, bad_sel = self._walk_fused(geo, owner[idx], draw[idx], U[idx], t_from, t_to, coordinate)
            U[idx], bad[idx] = U_sel, bad[idx] | bad_sel
            return U, bad
        width = int(geo.constraints.shape[1]) * (d + 1) + (self.roots.n_roots * (2 * d + 4) if self.roots.is_voronoi else 0)
        chunk = max(1, self.CHUNK_ELEMENTS // max(1, width + 4 * d))
        U, bad = U.copy(), bad.copy()
        for s in range(0, int(idx.shape[0]), chunk):
            sel = idx[s : s + chunk]
            U[sel], bad_sel = self._steps(geo, owner[sel], draw[sel], U[sel], t_from, t_to, coordinate)
            bad[sel] = bad[sel] | bad_sel
        return U, bad

    # Recovery walks draw from purpose + this, so ordinary streams are unchanged.
    RECOVERY_PURPOSE_OFFSET = 16
    # A face may be violated by rounding only: relative to the face offset's scale.
    FEASIBILITY_TOL = 1e-9

    def _steps(self, geo: NodePolytopes, owner: Array, draw: Array, U: Array, t_from: int, t_to: int,
               coordinate: bool = False) -> tuple[Array, Array]:  # fmt: skip
        d = U.shape[1]
        root = geo.root[owner]
        lo, hi = self._lo[root], self._hi[root]
        A = geo.constraints[owner, :, :d]  # (n, h, d)
        b = geo.constraints[owner, :, d]  # (n, h)
        bad = xp.zeros(int(U.shape[0]), dtype=bool)
        stream = geo
        if coordinate:
            stream = NodePolytopes(geo.root, geo.depth, geo.keys, geo.constraints, geo.purpose + self.RECOVERY_PURPOSE_OFFSET)
        if self.roots.is_voronoi:
            # Bisector j of the owning centre c_r, from the centre difference D = c_j - c_r:
            # n^T (u - c_r) <= |D| / 2 with n = D / |D|. Expanding it as differences of dot
            # products and squared norms of the centres cancels catastrophically for close
            # centres (a cell of width 1e-12 came out 1e-5 wide); this form keeps the slack
            # in distance units, so FEASIBILITY_TOL means the same thing for every face.
            # The owning centre's own row (D = 0) is removed by identity.
            c_r = self._C[root]
            own = xp.arange(self.roots.n_roots)[None, :] == root[:, None]
            D = self._C[None, :, :] - c_r[:, None, :]  # (n, R, d), chunked by _walk
            norm = xp.linalg.norm(D, axis=2)
            normal = xp.where(own[:, :, None], 0.0, D / xp.where(own, 1.0, norm)[:, :, None])
            half = xp.where(own, xp.inf, 0.5 * norm)

        def slacks(U):
            """Cut and bisector slacks of ``U`` (distance units), and which chains violate one beyond rounding."""
            cut = b - det_einsum("nhd,nd->nh", A, U)
            out = xp.any(cut < -self.FEASIBILITY_TOL * (1.0 + xp.abs(b)), axis=1)
            vor = None
            if self.roots.is_voronoi:
                vor = half - det_einsum("nrd,nd->nr", normal, U - c_r)
                out |= xp.any(vor < -self.FEASIBILITY_TOL, axis=1)
            return cut, vor, out

        for t in range(t_from, t_to):
            q = (d + 1) // 2
            r = self._uniforms(stream, owner, draw, self._start_hashes + 1 + t * self._step_hashes, 2 * q + 1)
            if coordinate:
                # Systematic sweeps: every d steps touch every axis, so a chain cannot keep an
                # untouched corner coordinate (random axes miss one with prob. ~d (1-1/d)^t).
                g = xp.zeros((int(U.shape[0]), d))
                g[:, t % d] = 1.0
            else:
                # Box-Muller gives two independent normals per pair (cosine and sine).
                radius, angle = xp.sqrt(-2.0 * xp.log(r[:, :q])), 2.0 * math.pi * r[:, q : 2 * q]
                g = xp.concatenate([radius * xp.cos(angle), radius * xp.sin(angle)], axis=1)[:, :d]
                g /= xp.linalg.norm(g, axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                # The box faces bound t between (lo - U)/g and (hi - U)/g in either order:
                # fmin/fmax pick the same values as selecting on the sign of g, in fewer passes.
                t_lo_face, t_hi_face = (lo - U) / g, (hi - U) / g
                if coordinate:
                    # Axes the move does not touch bound nothing; on a face their 0/0 would
                    # be NaN, which fmin/fmax would turn into an infinite chord end.
                    t_lo_face, t_hi_face = xp.where(g == 0, -xp.inf, t_lo_face), xp.where(g == 0, xp.inf, t_hi_face)
                t_min = xp.max(xp.fmin(t_lo_face, t_hi_face), axis=1)
                t_max = xp.min(xp.fmax(t_lo_face, t_hi_face), axis=1)
                cut, vor, infeasible = slacks(U)
                bad |= infeasible
                if b.shape[1] > 0:
                    t_min, t_max = _chord(det_einsum("nhd,nd->nh", A, g), cut, t_min, t_max)
                if vor is not None:
                    t_min, t_max = _chord(det_einsum("nrd,nd->nr", normal, g), vor, t_min, t_max)
                bad |= ~(xp.isfinite(t_min) & xp.isfinite(t_max) & (t_min <= t_max))
            step = t_min + (t_max - t_min) * r[:, 2 * q]
            U = U + xp.where(bad, 0.0, step)[:, None] * g  # a failed chain stops where it is
        with np.errstate(invalid="ignore"):
            bad |= slacks(U)[2]  # the returned endpoint too, not only each step's start
        return U, bad

    def _walk_fused(self, geo: NodePolytopes, owner: Array, draw: Array, U: Array, t_from: int, t_to: int,
                    coordinate: bool) -> tuple[Array, Array]:  # fmt: skip
        """
        ``_steps`` on the GPU for whole walks: one thread per chain runs steps ``t_from ..
        t_to - 1`` and the endpoint check in a single ``hr_walk`` launch, drawing its
        uniforms from the same counter stream as ``_uniforms`` (bit-identical) and reading
        its node's constraints and its root's bounds in place instead of per-chain copies.
        The moves equal the NumPy form's up to fused-multiply-add rounding.
        """
        n, d = (int(x) for x in U.shape)
        purpose = geo.purpose + (self.RECOVERY_PURPOSE_OFFSET if coordinate else 0)
        cons = xp.ascontiguousarray(geo.constraints, dtype=xp.float64)  # (K, h, d + 1)
        root = geo.root[owner].astype(xp.int64)
        C = xp.ascontiguousarray(self._C, dtype=xp.float64) if self.roots.is_voronoi else xp.zeros((1, d))  # unused if R = 0
        U = xp.ascontiguousarray(U, dtype=xp.float64).copy()
        bad = xp.zeros(n, dtype=bool)
        if n and not IS_CUPY_BACKEND:  # the same walk as a serial Numba loop
            _cpu_hr_walk()(U, bad, owner.astype(np.int64), draw.astype(np.int64), geo.keys.astype(np.uint64), cons, root,
                           np.ascontiguousarray(self._lo, dtype=np.float64), np.ascontiguousarray(self._hi, dtype=np.float64),
                           C, d, int(cons.shape[1]), self.roots.n_roots if self.roots.is_voronoi else 0, (d + 1) // 2,
                           int(coordinate), t_from, t_to, self._start_hashes + 1 + self.max_walk_steps * self._step_hashes,
                           self._start_hashes + 1, np.uint64((purpose << 56) + _GOLDEN % 2**56), self.FEASIBILITY_TOL)  # fmt: skip
        elif n:
            _hr_kernel()(((n + 127) // 128,), (128,), (
                U, bad, owner.astype(xp.int64), draw.astype(xp.int64), geo.keys.astype(xp.uint64), cons, root,
                xp.ascontiguousarray(self._lo, dtype=xp.float64), xp.ascontiguousarray(self._hi, dtype=xp.float64), C,
                np.int64(n), np.int32(d), np.int32(cons.shape[1]), np.int32(self.roots.n_roots if self.roots.is_voronoi else 0),
                np.int32((d + 1) // 2), np.int32(int(coordinate)), np.int32(t_from), np.int32(t_to),
                np.int64(self._start_hashes + 1 + self.max_walk_steps * self._step_hashes), np.int64(self._start_hashes + 1),
                np.uint64((purpose << 56) + _GOLDEN % 2**56), np.float64(self.FEASIBILITY_TOL)))  # fmt: skip
        return U, bad

    def _drifting(self, U_mid: Array, U_end: Array, owner: Array, walk: Array, n_new: Array) -> Array:
        """
        (K,) nodes whose ``walk`` chains moved visibly between the halfway and final steps,
        from those chains' own count, means and centred variances (at least 4 chains).
        """
        K = int(n_new.shape[0])
        out = xp.zeros(K, dtype=bool)
        nz = xp.nonzero(n_new > 0)[0]
        if int(nz.shape[0]) == 0:
            return out
        starts = (xp.cumsum(n_new) - n_new)[nz]
        # Statistics of the chains being diagnosed only: a recovery walks a few chains of a
        # node, and the node's fixed rows would otherwise dilute their movement away.
        active = walk.astype(xp.float64)[:, None]
        count = segment_sum(active[:, 0], starts)
        m = xp.maximum(count, 1.0)[:, None]
        slot = xp.zeros(K, dtype=xp.int64)
        slot[nz] = xp.arange(int(nz.shape[0]))
        row_node = slot[owner]

        def mean_of(f):
            return segment_sum(f * active, starts) / m

        # The squared radius is taken about the node's halfway mean, not a fixed point: far
        # from that point a narrow node's radii would all agree to rounding.
        centre = mean_of(U_mid)[row_node]

        def moments(V):
            f = xp.concatenate([V, xp.sum((V - centre) ** 2, axis=1, keepdims=True)], axis=1)
            mean = mean_of(f)
            # Centred second moments: mean(f^2) - mean^2 cancels for a narrow node far from
            # the origin and turned rounding-sized shifts into "drift".
            return mean, mean_of((f - mean[row_node]) ** 2)

        (m1, v1), (m2, v2) = moments(U_mid), moments(U_end)
        # Mean differences below the coordinates' rounding resolution are not drift.
        resolution = 64 * np.finfo(np.float64).eps * (xp.abs(m1) + xp.abs(m2))
        z = xp.abs(m2 - m1) / xp.sqrt(xp.maximum((v1 + v2) / m, resolution**2 + 1e-300))
        out[nz] = (count >= 4) & (xp.max(z, axis=1) > self.mixing_z)
        return out


_HR_KERNEL = None


def _hr_kernel():
    """The compiled ``hr_walk`` kernel (compiled on first use)."""
    global _HR_KERNEL
    if _HR_KERNEL is None:
        _HR_KERNEL = kernels.module("hit_and_run").get_function("hr_walk")
    return _HR_KERNEL


_CPU_HR_WALK = None


def _cpu_hr_walk():
    """
    ``hr_walk`` (kernels/hit_and_run.cu) as a serial Numba loop over chains, compiled on first use:
    the same direction, chords, feasibility checks and counter-stream uniforms, statement for
    statement, so CPU and GPU walks agree up to rounding. Serial for the reason given in
    ``_cpu_soft_gates``.
    """
    global _CPU_HR_WALK
    if _CPU_HR_WALK is None:
        import numba as nb

        m1, m2 = np.uint64(_MIX1), np.uint64(_MIX2)

        @nb.njit(cache=True, error_model="numpy")  # IEEE inf/NaN on division by zero, as NumPy and CUDA
        def mix(z):  # pragma: no cover - compiled
            z = (z ^ (z >> np.uint64(30))) * m1
            z = (z ^ (z >> np.uint64(27))) * m2
            return z ^ (z >> np.uint64(31))

        @nb.njit(cache=True, error_model="numpy")  # IEEE inf/NaN on division by zero, as NumPy and CUDA
        def walk(U, bad, owner, draw, keys, cons_all, root, lo_all, hi_all, C, d, h, R, q, coordinate, t_from, t_to,
                 block, base, tag, tol):  # pragma: no cover - compiled  # fmt: skip
            g = np.empty(d)
            r = np.empty(2 * q + 2)
            for i in range(U.shape[0]):
                u = U[i]
                k, rt = owner[i], root[i]
                cons, lo, hi = cons_all[k], lo_all[rt], hi_all[rt]
                bd = False
                for t in range(t_from, t_to):
                    ctr0 = draw[i] * block + base + t * (q + 1)
                    for p in range(q + 1):
                        z = mix(keys[k] ^ mix(np.uint64(ctr0 + p) + tag))
                        r[2 * p] = (np.float64(z >> np.uint64(32)) + 0.5) * 2.3283064365386963e-10
                        r[2 * p + 1] = (np.float64(z & np.uint64(0xFFFFFFFF)) + 0.5) * 2.3283064365386963e-10
                    if coordinate:
                        for j in range(d):
                            g[j] = 1.0 if j == t % d else 0.0
                    else:
                        for p in range(q):
                            radius, angle = np.sqrt(-2.0 * np.log(r[p])), 2.0 * np.pi * r[q + p]
                            g[p] = radius * np.cos(angle)
                            if q + p < d:
                                g[q + p] = radius * np.sin(angle)
                        nn = 0.0
                        for j in range(d):
                            nn += g[j] * g[j]
                        nn = np.sqrt(nn)
                        for j in range(d):
                            g[j] /= nn
                    t_min, t_max = -np.inf, np.inf
                    for j in range(d):
                        a, c = (lo[j] - u[j]) / g[j], (hi[j] - u[j]) / g[j]
                        if coordinate and g[j] == 0.0:
                            a, c = -np.inf, np.inf
                        lo_t, hi_t = np.fmin(a, c), np.fmax(a, c)
                        t_min = lo_t if j == 0 else (np.nan if np.isnan(t_min) or np.isnan(lo_t) else max(t_min, lo_t))
                        t_max = hi_t if j == 0 else (np.nan if np.isnan(t_max) or np.isnan(hi_t) else min(t_max, hi_t))
                    up, dn = np.inf, -np.inf
                    for kk in range(h):
                        a = cons[kk]
                        sdot, ag = 0.0, 0.0
                        for j in range(d):
                            sdot += a[j] * u[j]
                            ag += a[j] * g[j]
                        slack = a[d] - sdot
                        if slack < -tol * (1.0 + abs(a[d])):
                            bd = True
                        ratio = np.nan if np.isnan(slack) else max(slack, 0.0) / ag
                        if ag > 0:
                            up = np.nan if np.isnan(up) or np.isnan(ratio) else min(up, ratio)
                        elif ag < 0:
                            dn = np.nan if np.isnan(dn) or np.isnan(ratio) else max(dn, ratio)
                    for rr in range(R):
                        if rr == rt:
                            continue
                        nrm, sdot, ag = 0.0, 0.0, 0.0
                        for j in range(d):
                            nrm += (C[rr, j] - C[rt, j]) ** 2
                        nrm = np.sqrt(nrm)
                        for j in range(d):
                            nj = (C[rr, j] - C[rt, j]) / nrm
                            sdot += nj * (u[j] - C[rt, j])
                            ag += nj * g[j]
                        slack = 0.5 * nrm - sdot
                        if slack < -tol:
                            bd = True
                        ratio = np.nan if np.isnan(slack) else max(slack, 0.0) / ag
                        if ag > 0:
                            up = np.nan if np.isnan(up) or np.isnan(ratio) else min(up, ratio)
                        elif ag < 0:
                            dn = np.nan if np.isnan(dn) or np.isnan(ratio) else max(dn, ratio)
                    if h > 0 or R > 0:
                        t_max = np.nan if np.isnan(t_max) or np.isnan(up) else min(t_max, up)
                        t_min = np.nan if np.isnan(t_min) or np.isnan(dn) else max(t_min, dn)
                    if not (np.isfinite(t_min) and np.isfinite(t_max) and t_min <= t_max):
                        bd = True
                    if not bd:
                        step = t_min + (t_max - t_min) * r[2 * q]
                        for j in range(d):
                            u[j] += step * g[j]
                for kk in range(h):  # the returned endpoint too
                    a = cons[kk]
                    sdot = 0.0
                    for j in range(d):
                        sdot += a[j] * u[j]
                    if a[d] - sdot < -tol * (1.0 + abs(a[d])):
                        bd = True
                for rr in range(R):
                    if rr == rt:
                        continue
                    nrm, sdot = 0.0, 0.0
                    for j in range(d):
                        nrm += (C[rr, j] - C[rt, j]) ** 2
                    nrm = np.sqrt(nrm)
                    for j in range(d):
                        sdot += ((C[rr, j] - C[rt, j]) / nrm) * (u[j] - C[rt, j])
                    if 0.5 * nrm - sdot < -tol:
                        bd = True
                bad[i] = bd

        _CPU_HR_WALK = walk
    return _CPU_HR_WALK


def _per_node_sum(values: Array, n_new: Array) -> Array:
    """(K,) sums of each node's rows of ``values`` (rows grouped by node, ``n_new`` per node; empty nodes give 0)."""
    out = xp.zeros(int(n_new.shape[0]))
    nz = xp.nonzero(n_new > 0)[0]
    if int(nz.shape[0]):
        out[nz] = segment_sum(values, (xp.cumsum(n_new) - n_new)[nz])
    return out


def _chord(Ag: Array, slack: Array, t_min: Array, t_max: Array) -> tuple[Array, Array]:
    """
    Intersect the chord ``[t_min, t_max]`` with half-spaces whose slack along the
    direction is ``Ag``. Rounding can leave a point a hair outside a face, so the slack
    is clamped at 0 and the chord still contains the current point; the caller checks
    that no slack is negative beyond rounding (``FEASIBILITY_TOL``) and flags the chain
    otherwise. Rows with ``Ag == 0`` (padding, or the owning Voronoi centre) are skipped.
    """
    ratio = xp.maximum(slack, 0.0) / Ag
    t_max = xp.minimum(t_max, xp.min(xp.where(Ag > 0, ratio, xp.inf), axis=1))
    t_min = xp.maximum(t_min, xp.max(xp.where(Ag < 0, ratio, -xp.inf), axis=1))
    return t_min, t_max
