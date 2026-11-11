"""
Region mode (``ClusteringConfig(algorithm="region")``): an ART-style region tree on
AFINNI's batched refinement loop.

Every node of the tree is a convex polytope of the declared box domain: a root cell
(the whole box, a uniform-grid cell, or the Voronoi cell of a k-means centre) cut by its
ancestors' hyperplanes. Per batch of nodes, ``RegionPlan.process_batch``:

1. tops every node up to ``ceil(sample_multiplier * (d + 1))`` training points inside
   its polytope (``HitAndRunSampler``) and draws ``validation_samples`` fresh points;
   the B1 variant (``sampling="convex_combination"``) instead keeps AFINNI's
   convex-combination augmentation of rejected nodes;
2. fits the local model on the training points and screens it with the acceptance test;
3. deploys the screened models (``Q`` if stored, finalize, re-test) and checks them on
   the validation points;
4. splits every rejected node by a hyperplane (``split_search``: ``"gap"`` along the
   split strategy's direction, ``"bins"`` a threshold search over a bank of directions,
   ``"soft"`` that search polished by the consistent soft-oblique optimizer), cut so the
   stored predicate reproduces the children exactly; a node that cannot be split, or is
   at ``max_levels``, keeps its fitted model as a terminal leaf with a status.

Geometry lives in normalized coordinates ``u = (x - lower) / (upper - lower)`` of the
domain; every predicate is ``w^T u - z >= 0`` (right child) with ``|w| = 1``, evaluated
by ``plane_scores`` both in training and in ``route``. Children carry their ancestors'
half-spaces (``constraints``), their node key ``(root, path_code)`` and depth, so a
stolen node needs nothing from the rank that created it.

Random numbers come from a counter-based generator keyed by ``(seed, root, path_code,
purpose, draw index)``: a node's samples and random directions do not depend on the
batch it is processed in, the number of ranks or the steal schedule, so region-mode
results are the same for any batch size and rank count (except the B1 variant, whose
augmentation uses a per-batch stream, as in AFINNI mode). A node draws
each purpose once (attempt 0); a second draw would need an attempt index.
"""

import contextlib
import math
from dataclasses import dataclass, field

import numpy as np

from src import Array, _to_host, xp
from src.acceptance import tolerance_vector
from src.adaptive_clustering_utils import uniform_grid_bins
from src.clusters import ParamSpec
from src.local_models import ClusterFitData, deploy_leaves, output_columns
from src.numerics import _reduce_max_1d, _reduce_sum_1d, sort_by_block_then_value
from src.sampling import ClusterPoints, Sampler, augment_clusters, augmentation_targets
from src.splitting import SplitData
from src.utils import repeats

# Leaf statuses, stored as int8 codes (index into these tuples).
TERMINAL_REASONS = ("accepted", "max_depth", "split_failed")
VALIDATION_STATUSES = ("not_performed", "passed", "failed", "insufficient_sampling")
ACCEPTED, MAX_DEPTH, SPLIT_FAILED = 0, 1, 2
NOT_PERFORMED, PASSED, FAILED, INSUFFICIENT_SAMPLING = 0, 1, 2, 3

PURPOSE_TRAIN, PURPOSE_VALIDATION, PURPOSE_DIRECTION = 1, 2, 3
MAX_DEPTH_LIMIT = 62  # path codes are positive int64 with a leading sentinel bit

# --- Counter-based random numbers --------------------------------------------------

_GOLDEN = 0x9E3779B97F4A7C15
_MIX1, _MIX2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def _mix64(z: Array) -> Array:
    """The splitmix64 finalizer on uint64 arrays (wrapping arithmetic, same on NumPy and CuPy)."""
    z = (z ^ (z >> xp.uint64(30))) * xp.uint64(_MIX1)
    z = (z ^ (z >> xp.uint64(27))) * xp.uint64(_MIX2)
    return z ^ (z >> xp.uint64(31))


def node_keys(seed: int, root: Array, path_code: Array) -> Array:
    """(K,) uint64 stream keys, one per node; children's keys are unrelated to their parent's."""
    base = xp.uint64((int(seed) * _GOLDEN + 1) % 2**64)
    k = _mix64(base + xp.asarray(root).astype(xp.uint64) * xp.uint64(_GOLDEN))
    return _mix64(k ^ xp.asarray(path_code).astype(xp.uint64))


def counter_uniform(keys: Array, purpose: int, counter: Array) -> Array:
    """Uniform draws in the open interval (0, 1): one per ``(keys[i], purpose, counter[i])``, broadcast."""
    tag = xp.uint64((purpose << 56) + _GOLDEN % 2**56)
    z = _mix64(keys ^ _mix64(xp.asarray(counter).astype(xp.uint64) + tag))
    return ((z >> xp.uint64(11)).astype(xp.float64) + 0.5) * 2.0**-53


def counter_uniform_pairs(keys: Array, purpose: int, counter: Array) -> Array:
    """
    Two uniforms in (0, 1) per ``(keys[i], purpose, counter[i])``, from the high and low 32
    bits of one hash: shape ``counter.shape + (2,)``. Half the hashing of ``counter_uniform``,
    at 2^-32 resolution, which hit-and-run directions and chord positions do not resolve.
    """
    tag = xp.uint64((purpose << 56) + _GOLDEN % 2**56)
    z = _mix64(keys ^ _mix64(xp.asarray(counter).astype(xp.uint64) + tag))
    hi = (z >> xp.uint64(32)).astype(xp.float64)
    lo = (z & xp.uint64(0xFFFFFFFF)).astype(xp.float64)
    return (xp.stack([hi, lo], axis=-1) + 0.5) * 2.0**-32


def _normals(u1: Array, u2: Array) -> Array:
    """Standard normals from two arrays of open-interval uniforms (Box-Muller)."""
    return xp.sqrt(-2.0 * xp.log(u1)) * xp.cos(2.0 * math.pi * u2)


# --- Domain and roots ----------------------------------------------------------------


@dataclass(frozen=True)
class Domain:
    """The box ``[lower, upper]`` that routing covers and the audit samples (host arrays)."""

    lower: np.ndarray  # (d,)
    upper: np.ndarray  # (d,)

    @property
    def d(self) -> int:
        return int(self.lower.shape[0])

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def normalize(self, X: Array) -> Array:
        return (X - xp.asarray(self.lower)) / xp.asarray(self.width)

    def denormalize(self, U: Array) -> Array:
        return xp.asarray(self.lower) + U * xp.asarray(self.width)

    @classmethod
    def resolve(cls, spec, X_host: np.ndarray) -> "Domain":
        """The domain of a config: its ``(lower, upper)``, or the points' bounding box for ``"data_bbox"``."""
        if isinstance(spec, str):  # "data_bbox", validated by the config
            lower, upper = X_host.min(axis=0), X_host.max(axis=0)
            if not np.all(lower < upper):
                raise ValueError("domain='data_bbox' needs training points that span every dimension.")
            return cls(lower.astype(np.float64), upper.astype(np.float64))
        lower, upper = (np.asarray(b, dtype=np.float64) for b in spec)
        if lower.shape != (X_host.shape[1],):
            raise ValueError(f"domain has dimension {lower.shape[0]}, but the points have {X_host.shape[1]}.")
        outside = np.any((X_host < lower) | (X_host > upper), axis=1)
        if outside.any():
            raise ValueError(f"{int(outside.sum())} training points lie outside the domain {spec}.")
        return cls(lower, upper)


@dataclass(frozen=True)
class Roots:
    """
    The root cells, in normalized coordinates: one box (``bins`` all ones), a uniform grid
    of ``prod(bins)`` boxes, or the Voronoi cells of ``centres`` clipped to the domain.
    Box and grid cells are sampled exactly; Voronoi cells need hit-and-run with their
    bisectors, evaluated against every centre in tiles (no face list is stored).
    """

    bins: np.ndarray  # (d,) int64
    centres: np.ndarray | None = None  # (R, d) Voronoi centres in u, or None
    TILE_ELEMENTS = 1 << 22  # bound on (queries x centres x d) temporaries in root_of

    @classmethod
    def build(cls, kind: str, d: int, num_cells: int, centres: np.ndarray | None = None) -> "Roots":
        if kind == "box":
            return cls(np.ones(d, dtype=np.int64))
        if kind == "grid":
            return cls(uniform_grid_bins(np.ones(d), num_cells))
        if centres is None or len(centres) == 0:
            raise ValueError("Voronoi roots need centres.")
        # Duplicate centres would own empty cells under lowest-index ownership while their
        # closed inequalities coincide; keep the first of each.
        centres = np.asarray(centres, dtype=np.float64)
        _, first = np.unique(centres, axis=0, return_index=True)
        return cls(np.ones(d, dtype=np.int64), centres[np.sort(first)])

    @property
    def n_roots(self) -> int:
        return int(np.prod(self.bins)) if self.centres is None else int(self.centres.shape[0])

    @property
    def is_voronoi(self) -> bool:
        return self.centres is not None

    def bounds(self) -> tuple[Array, Array]:
        """(R, d) lower and upper corners of every root's bounding box in normalized coordinates."""
        if self.is_voronoi:
            R, d = self.centres.shape
            return xp.zeros((R, d)), xp.ones((R, d))
        coords = np.stack(np.unravel_index(np.arange(self.n_roots), tuple(int(b) for b in self.bins)), axis=1)
        return xp.asarray(coords / self.bins), xp.asarray((coords + 1) / self.bins)

    def root_of(self, U: Array) -> Array:
        """
        (n,) int64 root of every normalized point: grid points on a cell face go to the
        upper cell; Voronoi points to the nearest centre by exact squared distance (ties
        to the lowest index). Training and routing both assign roots through this.
        """
        if self.is_voronoi:
            C = xp.asarray(self.centres)
            rows = max(1, self.TILE_ELEMENTS // max(1, C.size))
            out = [
                xp.argmin(xp.sum((U[s : s + rows, None, :] - C[None]) ** 2, axis=2), axis=1)
                for s in range(0, int(U.shape[0]), rows)
            ]
            return xp.concatenate(out).astype(xp.int64) if out else xp.zeros(0, dtype=xp.int64)
        bins = xp.asarray(self.bins)
        coords = xp.minimum(xp.floor(U * bins).astype(xp.int64), bins - 1)
        coords = xp.maximum(coords, 0)
        strides = xp.asarray(np.r_[np.cumprod(self.bins[::-1])[::-1][1:], 1].astype(np.int64))
        return xp.sum(coords * strides, axis=1)


# --- Sampling inside polytopes -------------------------------------------------------


@dataclass
class RegionPoints:
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
    in ``SAMPLERS``: it needs ``ClusterPoints.region``, which only region mode provides.
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
        either problem remaining are flagged in ``region.insufficient``. The recovery
        depends on the first walk's outcome, so it is an adaptive repair, not an
        independent production run.
        """
        geo: RegionPoints = clusters.region
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
                t_end = self.walk_steps
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
            unmoved = walk & ~xp.any(U != U0, axis=1)
            if bool(xp.any(unmoved)):
                # Every unmoved chain is rerun, not only nodes where most are stuck: a node
                # with 40% of its points left on a corner (a zero-volume atom) is as wrong.
                # The recovery walk must pass the same drift check: moving is not mixing.
                touched = _per_node_sum(unmoved.astype(xp.float64), n_new) > 0
                U, bad, recovery_drift = run(U, unmoved, bad, coordinate=True)
                drift = xp.where(touched, recovery_drift | drift, drift)
            stuck = _per_node_sum((walk & ~xp.any(U != U0, axis=1)).astype(xp.float64), n_new) > 0
            infeasible = _per_node_sum(bad.astype(xp.float64), n_new) > 0
            geo.insufficient = drift | stuck | infeasible
        return self.domain.denormalize(U)

    def _starts(self, clusters: ClusterPoints, geo: RegionPoints, owner: Array, draw: Array) -> tuple[Array, Array]:
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

    def _uniforms(self, geo: RegionPoints, owner: Array, draw: Array, offset: int, count: int) -> Array:
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

    def _steps(self, geo: RegionPoints, owner: Array, draw: Array, U: Array, t_from: int, t_to: int,
               coordinate: bool = False) -> tuple[Array, Array]:  # fmt: skip
        d = U.shape[1]
        root = geo.root[owner]
        lo, hi = self._lo[root], self._hi[root]
        A = geo.constraints[owner, :, :d]  # (n, h, d)
        b = geo.constraints[owner, :, d]  # (n, h)
        bad = xp.zeros(int(U.shape[0]), dtype=bool)
        stream = geo
        if coordinate:
            stream = RegionPoints(geo.root, geo.depth, geo.keys, geo.constraints, geo.purpose + self.RECOVERY_PURPOSE_OFFSET)
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
            cut = b - xp.einsum("nhd,nd->nh", A, U)
            out = xp.any(cut < -self.FEASIBILITY_TOL * (1.0 + xp.abs(b)), axis=1)
            vor = None
            if self.roots.is_voronoi:
                vor = half - xp.einsum("nrd,nd->nr", normal, U - c_r)
                out |= xp.any(vor < -self.FEASIBILITY_TOL, axis=1)
            return cut, vor, out

        for t in range(t_from, t_to):
            q = (d + 1) // 2
            r = self._uniforms(stream, owner, draw, self._start_hashes + 1 + t * self._step_hashes, 2 * q + 1)
            if coordinate:
                axis = xp.minimum((r[:, 0] * d).astype(xp.int64), d - 1)
                g = (xp.arange(d)[None, :] == axis[:, None]).astype(xp.float64)
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
                    t_min, t_max = _chord(xp.einsum("nhd,nd->nh", A, g), cut, t_min, t_max)
                if vor is not None:
                    t_min, t_max = _chord(xp.einsum("nrd,nd->nr", normal, g), vor, t_min, t_max)
                bad |= ~(xp.isfinite(t_min) & xp.isfinite(t_max) & (t_min <= t_max))
            step = t_min + (t_max - t_min) * r[:, 2 * q]
            U = U + xp.where(bad, 0.0, step)[:, None] * g  # a failed chain stops where it is
        with np.errstate(invalid="ignore"):
            bad |= slacks(U)[2]  # the returned endpoint too, not only each step's start
        return U, bad

    def _drifting(self, U_mid: Array, U_end: Array, owner: Array, walk: Array, n_new: Array) -> Array:
        """(K,) nodes whose walking chains moved visibly between the halfway and final steps."""
        K = int(n_new.shape[0])
        out = xp.zeros(K, dtype=bool)
        nz = xp.nonzero(n_new > 0)[0]
        if int(nz.shape[0]) == 0:
            return out
        starts = (xp.cumsum(n_new) - n_new)[nz]
        m = n_new[nz].astype(xp.float64)[:, None]
        slot = xp.zeros(K, dtype=xp.int64)
        slot[nz] = xp.arange(int(nz.shape[0]))
        row_node = slot[owner]

        def mean_of(f):
            return segment_sum(f, starts) / m

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
        walking = segment_sum(walk.astype(xp.float64), starts) > 0
        out[nz] = walking & (n_new[nz] >= 4) & (xp.max(z, axis=1) > self.mixing_z)
        return out


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


# --- Hyperplane predicates -----------------------------------------------------------


def plane_scores(U: Array, W: Array) -> Array:
    """``w_i^T u_i`` row by row; training and routing both call this, so they classify identically."""
    return xp.sum(U * W, axis=1)


def gap_cuts(sorted_scores: Array, labels: Array, counts: Array, min_side: int) -> tuple[Array, Array, Array]:
    """
    Per cluster, the cut position nearest the median such that both sides keep
    ``min_side`` points and the cut falls between two *distinct* scores, so the
    predicate ``score >= z`` with ``z`` the first right-side score reproduces the cut
    exactly. ``sorted_scores`` and ``labels`` are in split order (ascending within each
    cluster). Returns ``(valid, position, z)``, each (K,).
    """
    starts = xp.r_[0, xp.cumsum(counts)[:-1]]
    pos = xp.arange(int(labels.shape[0]), dtype=xp.int64) - starts[labels]
    n = counts[labels]
    prev = xp.r_[xp.asarray([-xp.inf]), sorted_scores[:-1]]
    valid = (pos >= min_side) & (n - pos >= min_side) & (sorted_scores > prev)
    # Distance from the median, ties broken towards the lower position: unique per cluster.
    cost = (xp.abs(2 * pos - n) * 2 + (2 * pos > n)).astype(xp.float64)
    cost = xp.where(valid, cost, xp.inf)
    first = sort_by_block_then_value(cost, labels)[starts]
    return valid[first], pos[first], sorted_scores[first]


@dataclass
class _SplitRows:
    """
    The rows of the nodes being split, with their affine split features (training views
    only). Weighted Grams and cross products are batched matrix products over the nodes,
    grouped by point count (``cross``): no per-point outer product is ever formed, which
    would cost ``N p^2`` memory traffic per Gram and dominate at high dimension. Grouping by
    exact count needs no padding, so a node's arithmetic does not depend on the batch.
    """

    U: Array  # (N, d) normalized points
    Y: Array  # (N, fd) outputs scaled by each node's tolerance scale
    Phi: Array  # (N, p) features [u, 1]
    labels: Array  # (N,)
    counts: Array  # (k,)
    starts: Array = field(init=False)
    lam: Array = field(init=False)  # (k,) fixed ridge per node
    groups: list = field(init=False)  # (node ids (g,), row ids (g, c)) per distinct count c

    def __post_init__(self) -> None:
        self.starts = xp.r_[0, xp.cumsum(self.counts)[:-1]]
        tr = segment_sum(xp.sum(self.Phi**2, axis=1), self.starts)
        self.lam = 1e-8 * tr / self.Phi.shape[1] + 1e-300
        counts = np.asarray(_to_host(self.counts))
        self.groups = []
        for c in np.unique(counts):
            nodes = xp.asarray(np.nonzero(counts == c)[0])
            self.groups.append((nodes, self.starts[nodes][:, None] + xp.arange(int(c))[None, :]))
        self._Phi_g = self.gather(self.Phi)
        self._Y_g = self.gather(self.Y)

    def reduce(self, a: Array) -> Array:
        return segment_sum(a, self.starts)

    def gather(self, a: Array) -> list[Array]:
        """``a`` (N, ...) as one (g, c, ...) block per count group."""
        return [a[rows] for _, rows in self.groups]

    def cross(self, A: list[Array], B: list[Array], w: Array | None = None) -> Array:
        """(k, a, b) per-node products ``A_k^T diag(w_k) B_k`` of gathered blocks (``gather``)."""
        k = int(self.counts.shape[0])
        out = xp.empty((k, A[0].shape[-1], B[0].shape[-1]))
        for (nodes, rows), Ag, Bg in zip(self.groups, A, B, strict=True):
            Aw = Ag if w is None else Ag * w[rows][..., None]
            out[nodes] = xp.swapaxes(Aw, 1, 2) @ Bg
        return out

    def gram(self, w: Array | None = None) -> tuple[Array, Array]:
        """The weighted Gram ``Phi^T W Phi`` (k, p, p) and right-hand side ``Phi^T W Y`` (k, p, fd)."""
        return self.cross(self._Phi_g, self._Phi_g, w), self.cross(self._Phi_g, self._Y_g, w)

    def solve(self, G: Array, B: Array) -> Array:
        """Ridge solutions ``(G + lam I)^-1 B`` for (k, ..., p, p) Grams (lam fixed per node)."""
        p = G.shape[-1]
        lam = self.lam.reshape((-1,) + (1,) * (G.ndim - 1))
        return xp.linalg.solve(G + lam * xp.eye(p), B)


def _hard_sse(rows: _SplitRows, w: Array, z: Array, min_side: int) -> tuple[Array, Array]:
    """
    Direct residual sum of squares of the two hard children ``w^T u < z`` / ``>= z``,
    each refit on its own rows (both Grams formed directly, no complementary
    subtraction), and whether both sides keep ``min_side`` points. Used to rank final
    candidates: the prefix-statistic SSE is only a screen (it cancels badly near a good fit).
    """
    side = plane_scores(rows.U, w[rows.labels]) >= z[rows.labels]
    right = side.astype(xp.float64)
    n_right = rows.reduce(right)
    ok = (n_right >= min_side) & (rows.counts - n_right >= min_side) & xp.isfinite(z)
    sse = xp.zeros(int(rows.counts.shape[0]))
    for mask in (1.0 - right, right):
        G, B = rows.gram(mask)
        beta = rows.solve(G, B)  # (k, p, fd)
        res = rows.Y - xp.einsum("np,npf->nf", rows.Phi, beta[rows.labels])
        sse = sse + rows.reduce(mask * xp.sum(res**2, axis=1))
    return xp.where(ok, sse, xp.inf), ok


# --- The plan ------------------------------------------------------------------------


class RegionPlan:
    """
    The region algorithm's plan methods over the shared executor
    (``src.adaptive_clustering._run_local_refinement``): batch sizing, steal settings,
    output columns and ``process_batch``. ``counters`` accumulates this rank's split
    diagnostics (reported as ``profiling_stats["region"]``).
    """

    def __init__(self, config, d: int, fd: int, domain: Domain, roots: Roots):
        self.config, self.d, self.fd = config, d, fd
        self.domain, self.roots = domain, roots
        s = config.region
        self.settings = s
        self.p_eff = d + 1
        self.convex_sampling = s.sampling == "convex_combination"
        self.n_target = 0 if self.convex_sampling else math.ceil(s.sample_multiplier * self.p_eff)
        self.n_val = int(s.validation_samples)
        self.min_side = self.p_eff + int(s.min_side_margin)
        if not self.convex_sampling and self.n_target < 2 * self.min_side:
            raise ValueError(
                f"Region nodes get {self.n_target} training points, fewer than twice the "
                f"minimum side size {self.min_side}: no split could ever be made. Raise "
                f"region.sample_multiplier or lower region.min_side_margin."
            )
        if config.max_levels > MAX_DEPTH_LIMIT:
            raise ValueError(f"Region mode supports max_levels <= {MAX_DEPTH_LIMIT}; got {config.max_levels}.")
        self.sampler = HitAndRunSampler(domain, roots, s.walk_steps, s.max_walk_steps, s.mixing_z)
        node_pts = max(self.n_target, 2 * config.min_pts_per_cluster) + self.n_val
        self.steal_settings = {
            "min_load": s.steal_min_load or 4 * node_pts,
            "steal_size": s.steal_size or 2 * node_pts,
            "lease_received": True,
            "work_sizes": self.work_sizes,  # one work unit for loads, eligibility and grants
        }
        self.counters = {
            "splits": 0, "split_failed": 0, "rank_deficient": 0, "insufficient_sampling": 0,
            "soft_improved": 0, "soft_gradient_converged": 0, "soft_line_search_stalled": 0,
            "soft_gate_saturated": 0, "soft_nonfinite": 0, "soft_iteration_limit": 0,
        }  # fmt: skip

    def work_sizes(self, sizes: np.ndarray) -> np.ndarray:
        """Points each queued node will occupy on the device: after top-up (or augmentation) and validation."""
        if self.convex_sampling:
            return np.asarray(_to_host(augmentation_targets(sizes, self.config.min_pts_per_cluster)))
        return np.maximum(sizes, self.n_target) + self.n_val

    def bytes_per_point(self) -> int:
        """
        Peak device bytes per processed point (training, validation and split work arrays).
        B3/B4's per-node candidate arrays (``n_bins - 1`` Grams per side, the screen) are
        charged to the fewest points a node can hold, so small nodes do not undercount.
        """
        d, fd, h = self.d, self.fd, self.config.max_levels + 1
        p = d + 1
        g = fd * d if self.config.requires_grad_everywhere else 0
        search = 0
        if self.settings.split_search != "gap":
            # Gathered feature and output blocks (one copy of each) for the batched Grams.
            search = 2 * (p + fd) + 4
            nb, D = self.settings.n_bins, self.settings.n_directions
            per_node = 3 * (nb - 1) * (p * p + p * fd + 1) + 2 * D * (nb - 1)
            search += -(-per_node // max(1, int(self.work_sizes(np.zeros(1, np.int64))[0])))
        return 8 * (4 * d + 3 * fd + 2 * g + 2 * h * (d + 1) + 12 + search)

    def leaf_columns(self, model) -> dict[str, ParamSpec]:
        cols = output_columns(model, self.d, self.fd)
        if not self.settings.store_Q:
            del cols["Q"]
        return {
            **cols,
            "root": ParamSpec((), np.int64),
            "path_code": ParamSpec((), np.int64),
            "terminal_reason": ParamSpec((), np.int8),
            "validation_status": ParamSpec((), np.int8),
        }

    def split_columns(self) -> dict[str, ParamSpec]:
        return {
            "root": ParamSpec((), np.int64),
            "path_code": ParamSpec((), np.int64),
            "w": ParamSpec((self.d,)),
            "z": ParamSpec(()),
        }

    def initial_queue(self, X: np.ndarray, F: np.ndarray, G: np.ndarray | None):
        """Every root as a queued node (possibly with no points), points sorted by root."""
        from src.clusters import ClusterBatch

        root = _to_host(self.roots.root_of(self.domain.normalize(xp.asarray(X)))).astype(np.int64)
        order = np.argsort(root, kind="stable")
        R = self.roots.n_roots
        return ClusterBatch(
            X[order], F[order], G[order] if G is not None else None,
            np.bincount(root, minlength=R).astype(np.int64),
            root=np.arange(R, dtype=np.int64), path_code=np.ones(R, dtype=np.int64),
            constraints=np.zeros((R, 0, self.d + 1)),
        )  # fmt: skip

    def process_batch(self, *, X, F, G, sizes, depth, root, path_code, constraints, components, evaluator,
                      accepted, splits, next_level, progress=None, timer=None) -> dict[str, int]:  # fmt: skip
        """
        Top up, fit, deploy, validate and split one batch of nodes (see the module
        docstring), appending leaves to ``accepted``, split records to ``splits`` and
        children to ``next_level``. ``progress()``, if given, is called between bounded
        chunks of long split work so steal requests are answered meanwhile. ``timer(key)``
        (the executor's profiler phases) times each stage under the shared phase keys:
        sampling as ``time_augment``, statistics and fit as ``time_error_calc``, deploying
        leaves as ``time_good_Q``, splitting as ``time_bad_split``. Returns counts for the
        executor's statistics.
        """
        timer = timer or (lambda key: contextlib.nullcontext())
        cfg = self.config
        model = components.local_model
        acceptance, trust_region = components.acceptance, components.trust_region
        K = int(sizes.shape[0])
        keys = node_keys(cfg.seed, root, path_code)
        geo = RegionPoints(root, depth, keys, constraints, PURPOSE_TRAIN)
        stats = {"generated": 0, "fallback_clusters": 0, "fallback_pts": 0}

        # 1. Training points: inherited ones, then new ones inside the polytope.
        with timer("time_augment"):
            # B1 keeps each node's points, except that an empty root (possible for grid or
            # Voronoi cells) is sampled up to 2 * p_eff points so it can be fitted.
            n_train = (
                xp.where(sizes == 0, 2 * self.p_eff, sizes) if self.convex_sampling else xp.maximum(sizes, self.n_target)
            )
            aug = augment_clusters(X, F, G, sizes, n_train, self.sampler, evaluator, None, region=geo)
            Xt, Ft, Gt, labels = aug.X, aug.F, aug.G, aug.labels
            stats["generated"] += 0 if aug.X_new is None else int(aug.X_new.shape[0])
            starts = xp.r_[0, xp.cumsum(n_train)[:-1]]
            del aug
            insufficient = geo.insufficient if geo.insufficient is not None else xp.zeros(K, dtype=bool)
            Xv = Fv = Gv = None
            if self.n_val:
                geo_v = RegionPoints(root, depth, keys, constraints, PURPOSE_VALIDATION)
                Xv = self.sampler.sample(ClusterPoints(X, sizes, region=geo_v), xp.full(K, self.n_val))
                Fv, Gv = evaluator(Xv, need_grad=Gt is not None)
                stats["generated"] += int(Xv.shape[0])
                insufficient = insufficient | geo_v.insufficient
            self.counters["insufficient_sampling"] += int(xp.sum(insufficient))

        # 2. Statistics and fit on the training points only.
        with timer("time_error_calc"):
            mus = xp.add.reduceat(Xt, starts) / n_train[:, None]
            if cfg.requires_f_centroids or cfg.requires_grad_centroids:
                f_mus, g_mus = evaluator(mus, need_grad=cfg.requires_grad_centroids)
            else:  # no component reads f(c): column "f" is NaN rather than an oracle call
                f_mus, g_mus = xp.full((K, self.fd), xp.nan), None
            fit = ClusterFitData(X=Xt, F=Ft, G=Gt, labels=labels, starts=starts, sizes=n_train,
                                 centroids=mus, f_c=f_mus, g_c=g_mus)  # fmt: skip
            params = model.fit(fit)
            F_hat = model.evaluate({**params, "centroids": mus, "f": f_mus}, Xt, labels)
            good, E_abs, E_rel = acceptance(Ft, F_hat, starts, n_train)
            del F_hat

        # 3. Deploy the screened models, then check them on the validation points.
        with timer("time_good_Q"):
            store_Q = self.settings.store_Q
            is_leaf = xp.zeros(K, dtype=bool)
            status_v = xp.full(K, NOT_PERFORMED, dtype=xp.int8)
            good_idx = xp.nonzero(good)[0]
            # Set after validation below: a node whose chains drifted is reported as such
            # whatever its validation said, since its points need not represent its cell.
            if int(good_idx.shape[0]) > 0:
                leaves = deploy_leaves(model, trust_region, acceptance, fit, good_idx, params, E_abs, E_rel, store_Q=store_Q)
                if self.n_val and int(leaves.idx.shape[0]) > 0:
                    ok = self._validate(model, leaves, Xv, Fv, components)
                    status_v[leaves.idx] = xp.where(ok, PASSED, FAILED).astype(xp.int8)
                    keep = xp.nonzero(ok)[0]
                    leaves.idx, leaves.rows = leaves.idx[keep], {k: v[keep] for k, v in leaves.rows.items()}
                status_v = xp.where(insufficient, INSUFFICIENT_SAMPLING, status_v).astype(xp.int8)
                self._append_leaves(accepted, leaves, root, path_code, ACCEPTED, status_v)
                is_leaf[leaves.idx] = True

        # 4. Split the rejected nodes; those that cannot split keep their model as a leaf.
        rejected = ~is_leaf
        fallback = rejected & (depth >= cfg.max_levels)
        to_split = xp.nonzero(rejected & ~fallback)[0]
        failed_split = to_split[:0]
        if int(to_split.shape[0]) > 0:
            with timer("time_bad_split"):
                failed_split, n_gen = self._split(
                    to_split, fit, Xv, Fv, Gv, components, evaluator, keys, depth, root, path_code,
                    constraints, splits, next_level, progress,
                )  # fmt: skip
            stats["generated"] += n_gen
        status_v = xp.where(insufficient, INSUFFICIENT_SAMPLING, status_v).astype(xp.int8)
        for idx, reason in ((xp.nonzero(fallback)[0], MAX_DEPTH), (failed_split, SPLIT_FAILED)):
            if int(idx.shape[0]) > 0:
                with timer("time_good_Q"):
                    leaves = deploy_leaves(model, trust_region, acceptance, fit, idx, params, E_abs, E_rel,
                                           retest=False, store_Q=store_Q)  # fmt: skip
                self._append_leaves(accepted, leaves, root, path_code, reason, status_v)
                stats["fallback_clusters"] += int(idx.shape[0])
                stats["fallback_pts"] += int(xp.sum(n_train[idx]))
        stats.update(
            accepted=int(xp.sum(is_leaf)), rejected=K - int(xp.sum(is_leaf)), n_pts=int(Xt.shape[0])
        )
        return stats

    def _validate(self, model, leaves, Xv, Fv, components) -> Array:
        """Whether each deployed leaf passes the acceptance test on its node's validation points."""
        L = int(leaves.idx.shape[0])
        rows = (leaves.idx[:, None] * self.n_val + xp.arange(self.n_val)).ravel()
        labels = repeats(xp.arange(L), xp.full(L, self.n_val))
        F_hat = model.evaluate(leaves.rows, Xv[rows], labels)
        sizes = xp.full(L, self.n_val, dtype=xp.int64)
        ok, _, _ = components.acceptance(Fv[rows], F_hat, xp.arange(L) * self.n_val, sizes)
        return ok

    def _append_leaves(self, accepted, leaves, root, path_code, reason: int, status_v) -> None:
        idx = leaves.idx
        accepted.append({
            **leaves.rows,
            "root": root[idx], "path_code": path_code[idx],
            "terminal_reason": xp.full(int(idx.shape[0]), reason, dtype=xp.int8),
            "validation_status": status_v[idx],
        })  # fmt: skip

    # --- Splitting -------------------------------------------------------------------

    def _split(self, idx, fit, Xv, Fv, Gv, components, evaluator, keys, depth, root, path_code, constraints,
               splits, next_level, progress) -> tuple[Array, int]:  # fmt: skip
        """
        Split nodes ``idx`` on their training points plus their (now spent) validation
        points; return the nodes whose split was invalid and the number of new oracle
        points (B1's augmentation). Validation points of a failed node may train its
        children, but never validate them.
        """
        d = self.d
        k = int(idx.shape[0])
        sel = xp.zeros(int(fit.sizes.shape[0]), dtype=bool)
        sel[idx] = True
        train_rows = sel[fit.labels]
        counts = fit.sizes[idx]
        X, F = fit.X[train_rows], fit.F[train_rows]
        G = fit.G[train_rows] if fit.G is not None else None
        n_generated = 0
        if self.n_val:
            # Interleave per node: its training points, then its validation points.
            v_rows = (idx[:, None] * self.n_val + xp.arange(self.n_val)).ravel()
            total = counts + self.n_val
            labels = repeats(xp.arange(k), total)
            pos = xp.arange(int(xp.sum(total))) - (xp.cumsum(total) - total)[labels]
            is_train = pos < counts[labels]
            X = _interleave(X, Xv[v_rows], is_train)
            F = _interleave(F, Fv[v_rows], is_train)
            G = _interleave(G, Gv[v_rows], is_train) if G is not None else None
            counts = total
        if self.convex_sampling:
            # B1: AFINNI's augmentation (convex combinations, inside the parent's hull and
            # so inside its polytope), with a per-batch stream as in AFINNI mode.
            targets = augmentation_targets(counts, self.config.min_pts_per_cluster)
            rng = xp.random.default_rng(int(_to_host(keys[idx[0]])) % 2**63)
            aug = augment_clusters(X, F, G, counts, targets, self.config.components().sampler, evaluator, rng,
                                   centroids=fit.centroids[idx])  # fmt: skip
            X, F, G, labels, counts = aug.X, aug.F, aug.G, aug.labels, targets
            n_generated = 0 if aug.X_new is None else int(aug.X_new.shape[0])
        elif not self.n_val:
            labels = repeats(xp.arange(k), counts)
        data = SplitData(X=X, F=F, G=G, labels=labels, counts=counts, centroids=fit.centroids[idx],
                         f_c=fit.f_c[idx], g_c=fit.g_c[idx] if fit.g_c is not None else None,
                         hessian_fn=evaluator.hessian_fn)  # fmt: skip
        U = self.domain.normalize(X)
        w, z = self._propose(data, U, components, keys[idx], progress)

        # Verify: the stored predicate defines the children; both keep min_side points.
        scores = plane_scores(U, w[labels])
        order = sort_by_block_then_value(scores, labels)
        right = (scores >= z[labels]).astype(xp.int64)
        starts = xp.r_[0, xp.cumsum(counts)[:-1]]
        n_right = xp.rint(segment_sum(right.astype(xp.float64), starts)).astype(xp.int64)
        cut = counts - n_right
        valid = xp.all(xp.isfinite(w), axis=1) & xp.isfinite(z) & (cut >= self.min_side) & (n_right >= self.min_side)
        side = plane_scores(U[order], w[labels]) - z[labels] >= 0
        pos = xp.arange(int(labels.shape[0])) - starts[labels]
        if not bool(xp.all((side == (pos >= cut[labels])) | ~valid[labels])):
            raise RuntimeError("A region split predicate does not reproduce its children.")
        good = xp.nonzero(valid)[0]
        self.counters["splits"] += int(good.shape[0])
        self.counters["split_failed"] += k - int(good.shape[0])
        if int(good.shape[0]) > 0:
            keep = valid[labels][order]
            child_sizes = xp.stack([cut[good], counts[good] - cut[good]], axis=1).ravel()
            node, pc = idx[good], path_code[idx[good]]
            row = xp.concatenate([w[good], z[good][:, None]], axis=1)  # left: w^T u <= z
            parent_c = constraints[node]
            children_c = xp.stack([
                xp.concatenate([parent_c, row[:, None, :]], axis=1),
                xp.concatenate([parent_c, -row[:, None, :]], axis=1),
            ], axis=1).reshape(2 * int(good.shape[0]), -1, d + 1)  # fmt: skip
            rows_sorted = order[keep]
            next_level.append(
                X[rows_sorted], F[rows_sorted], G[rows_sorted] if G is not None else None, child_sizes,
                depth=xp.repeat(depth[node] + 1, 2), root=xp.repeat(root[node], 2),
                path_code=xp.stack([2 * pc, 2 * pc + 1], axis=1).ravel(), constraints=children_c,
            )  # fmt: skip
            splits.append({"root": root[node], "path_code": pc, "w": w[good], "z": z[good]})
        return idx[~valid], n_generated

    def _propose(self, data: SplitData, U: Array, components, keys: Array, progress) -> tuple[Array, Array]:
        """Hyperplanes ``(w, z)`` in ``u`` for every node of ``data`` (invalid ones are caught by the caller)."""
        mode = self.settings.split_search
        V, M = self._metric_basis(data, components.split_strategy)
        width = xp.asarray(self.domain.width)
        lead = _unit(xp.einsum("kij,kj->ki", M, V[:, :, -1]) * width)  # scores (x - c)^T M v, in u
        if mode == "gap":
            scores = plane_scores(U, lead[data.labels])
            order = sort_by_block_then_value(scores, data.labels)
            ok, _, z = gap_cuts(scores[order], data.labels, data.counts, self.min_side)
            return lead, xp.where(ok, z, xp.nan)
        rows = self._split_rows(data, U)
        bank = self._direction_bank(rows, lead, V, M, width, keys)
        w, z, sse = self._threshold_search(rows, bank)
        if mode == "soft":
            if progress is not None:
                progress()
            w, z = self._soft_refine(rows, w, z, sse, progress)
        return w, z

    def _metric_basis(self, data: SplitData, splitter) -> tuple[Array, Array]:
        """The split strategy's metric ``M`` and the eigenvectors ``V`` of ``M C M`` (ascending)."""
        M, CovZ = splitter.metric(data)
        return xp.linalg.eigh(CovZ)[1], M

    def _split_rows(self, data: SplitData, U: Array) -> _SplitRows:
        """Affine split features and tolerance-scaled outputs, so components combine by their tolerances."""
        fd = data.F.shape[1]
        acc = self.config.components().acceptance
        atol = xp.maximum(tolerance_vector("atol", acc.atol, fd), 0.0)
        rtol = xp.maximum(tolerance_vector("rtol", acc.rtol, fd), 0.0)
        starts = xp.r_[0, xp.cumsum(data.counts)[:-1]]
        Fmax = xp.stack([_reduce_max_1d(xp.abs(data.F[:, j]), starts) for j in range(fd)], axis=1)
        scale = xp.maximum(xp.maximum(atol[None, :], rtol[None, :] * Fmax), 1e-300)
        Y = data.F / scale[data.labels]
        Phi = xp.concatenate([U, xp.ones((int(U.shape[0]), 1))], axis=1)
        return _SplitRows(U, Y, Phi, data.labels, data.counts)

    def _direction_bank(self, rows: _SplitRows, lead, V, M, width, keys) -> Array:
        """
        (k, D, d) unit candidate normals in ``u``: the split strategy's leading direction,
        its next eigenvectors, the leading covariance direction of the points, and
        node-keyed random directions to fill ``n_directions``.
        """
        D, d = self.settings.n_directions, self.d
        dirs = [lead]
        for j in range(2, min(d, 3) + 1):  # further metric eigenvectors span near-tied eigenspaces
            dirs.append(_unit(xp.einsum("kij,kj->ki", M, V[:, :, -j]) * width))
        mean = rows.reduce(rows.U) / rows.counts[:, None]
        dU = rows.U - mean[rows.labels]
        dU_g = rows.gather(dU)
        cov = rows.cross(dU_g, dU_g)
        evals, evecs = xp.linalg.eigh(cov)
        # Points on a lower-dimensional set (e.g. unmixed samples): the fit is then ridge-
        # regularized along the missing directions, which is worth seeing in the counters.
        self.counters["rank_deficient"] += int(xp.sum(evals[:, 0] <= 1e-12 * xp.maximum(evals[:, -1], 1e-300)))
        dirs.append(evecs[:, :, -1])
        n_rand = max(0, D - len(dirs))
        if n_rand:
            r = counter_uniform(keys[:, None], PURPOSE_DIRECTION, xp.arange(2 * n_rand * d)[None, :])
            g = _normals(r[:, : n_rand * d], r[:, n_rand * d :]).reshape(-1, n_rand, d)
            dirs.extend(_unit(g[:, i]) for i in range(n_rand))
        return xp.stack(dirs[:D], axis=1)

    def _threshold_search(self, rows: _SplitRows, bank: Array) -> tuple[Array, Array, Array]:
        """
        B3: along every bank direction, evaluate ``n_bins - 1`` quantile cut positions
        with the left side's Gram formed directly for each cut (batched products, no
        prefix sums), the right side's by subtraction from the node total (a screen), keep
        the ``n_screened`` best per node, then recompute their SSE directly (``_hard_sse``)
        and return the best ``(w, z)`` and its SSE. Only cuts between distinct scores that
        leave ``min_side`` points per side are candidates.
        """
        k, D, _ = bank.shape
        nb = self.settings.n_bins
        b = xp.arange(1, nb)
        n = rows.counts
        # Only nodes that can keep min_side points on both sides have candidates (sparse B1
        # nodes may not); every index below stays inside its own node's block.
        eligible = n >= 2 * self.min_side
        pos = xp.clip(xp.round(b[None, :] * n[:, None] / nb).astype(xp.int64), self.min_side, xp.maximum(n - self.min_side, self.min_side)[:, None])
        pos = xp.clip(pos, 1, xp.maximum(n - 1, 1)[:, None])
        G_tot, B_tot = rows.gram()
        C_tot = rows.reduce(xp.sum(rows.Y**2, axis=1))
        screen = xp.full((k, D, nb - 1), xp.inf)
        z_all = xp.full((k, D, nb - 1), xp.nan)
        for j in range(D):
            s = plane_scores(rows.U, bank[rows.labels, j])
            order = sort_by_block_then_value(s, rows.labels)
            ss = s[order]
            at = rows.starts[:, None] + pos  # first right-side row of each candidate
            ok = eligible[:, None] & (pos >= self.min_side) & (n[:, None] - pos >= self.min_side) & (ss[at] > ss[at - 1])
            # Each row's rank along this direction within its node; the left side of cut i
            # is the rows ranked below pos[:, i]. Each Gram is a batched product over the
            # node's own rows, so nothing depends on the other nodes of the batch.
            rank = xp.empty(int(order.shape[0]), dtype=xp.int64)
            rank[order] = xp.arange(int(order.shape[0])) - rows.starts[rows.labels[order]]
            y2 = xp.sum(rows.Y**2, axis=1)
            GL, BL, CL = xp.empty((k, nb - 1) + G_tot.shape[1:]), xp.empty((k, nb - 1) + B_tot.shape[1:]), xp.empty((k, nb - 1))
            for i in range(nb - 1):
                left = (rank < pos[rows.labels, i]).astype(xp.float64)
                GL[:, i], BL[:, i] = rows.gram(left)
                CL[:, i] = rows.reduce(left * y2)
            GR, BR, CR = G_tot[:, None] - GL, B_tot[:, None] - BL, C_tot[:, None] - CL
            sse = xp.zeros((k, nb - 1))
            for Gs, Bs, Cs in ((GL, BL, CL), (GR, BR, CR)):
                beta = rows.solve(Gs, Bs)
                sse = sse + Cs - 2 * xp.sum(beta * Bs, axis=(2, 3)) + xp.sum(beta * (Gs @ beta), axis=(2, 3))
            screen[:, j] = xp.where(ok, sse, xp.inf)
            z_all[:, j] = ss[at]
        flat = screen.reshape(k, -1)
        n_keep = min(self.settings.n_screened, flat.shape[1])
        best_w, best_z = bank[:, 0], xp.full(k, xp.nan)
        best_sse = xp.full(k, xp.inf)
        cand = xp.argsort(flat, axis=1)[:, :n_keep]
        for t in range(n_keep):
            c = cand[:, t]
            j, i = c // (nb - 1), c % (nb - 1)
            w = bank[xp.arange(k), j]
            z = z_all[xp.arange(k), j, i]
            z = xp.where(xp.isfinite(flat[xp.arange(k), c]), z, xp.nan)
            sse, _ = _hard_sse(rows, w, xp.where(xp.isfinite(z), z, xp.inf), self.min_side)
            better = sse < best_sse
            best_w = xp.where(better[:, None], w, best_w)
            best_z = xp.where(better, z, best_z)
            best_sse = xp.minimum(sse, best_sse)
        return best_w, best_z, best_sse

    def _soft_refine(self, rows: _SplitRows, w0: Array, z0: Array, sse0: Array, progress) -> tuple[Array, Array]:
        """
        B4: polish each B3 plane by projected gradient descent on the consistent soft
        objective ``J(w, z) = min_B F`` (Section 33.3 of the design), with a fixed ridge, a
        fixed temperature ``T = soft_temperature * std(w^T u)``, floored weights
        ``W_R = delta + (1 - 2 delta) sigma(s)``, ``W_L = delta + (1 - 2 delta) sigma(-s)``,
        and frozen-model Armijo steps. A step is accepted only if its decrease exceeds the
        inner solves' estimated suboptimality ``tr(R^T H^-1 R) / n`` (``R`` from the data
        residual) plus rounding. This is numerically monitored descent, not a certified
        optimum. Each node's termination is counted once: ``soft_gate_saturated`` (gate
        derivative vanished), ``soft_gradient_converged`` (even a first-order step of
        one gate width could not beat the margin), ``soft_line_search_stalled`` (no trial
        qualified otherwise), ``soft_nonfinite`` (objective or gap not finite) or
        ``soft_iteration_limit``. The result is hardened (``z`` moved to the first
        right-side score) and kept only if its directly refit hard SSE beats the B3
        incumbent's; otherwise the incumbent is returned.
        """
        s_cfg = self.settings
        delta = s_cfg.soft_weight_floor
        n = rows.counts.astype(xp.float64)
        lab = rows.labels
        ok0 = xp.isfinite(z0)
        w, z = w0.copy(), xp.where(ok0, z0, 0.0)
        eps = float(np.finfo(np.float64).eps)
        s0 = plane_scores(rows.U, w[lab])
        # Centred variance (the raw-moment form cancels to zero or below for a narrow node
        # far from the origin, which made T ~ 1e-151 and the gates hard), floored at the
        # scores' rounding resolution.
        mean = rows.reduce(s0) / n
        var = rows.reduce((s0 - mean[lab]) ** 2) / n
        resolution = 64 * eps * xp.maximum(xp.abs(mean), 1e-300)
        T = s_cfg.soft_temperature * xp.sqrt(xp.maximum(var, resolution**2))
        active = ok0.copy()

        def gate(w_, z_):
            return _soft_gates(rows, w_, z_, T, delta)

        for it in range(s_cfg.soft_iters):
            if not bool(xp.any(active)):
                break
            if progress is not None and it % 4 == 3:
                progress()
            J, g_w, g_z, rL, rR, pen, gap, sp_sm = _soft_objective(rows, w, z, T, delta)
            g_w = g_w - xp.sum(g_w * w, axis=1, keepdims=True) * w  # tangent to |w| = 1
            gn2 = xp.sum(g_w**2, axis=1) + g_z**2
            alpha = T / xp.sqrt(xp.maximum(gn2, 1e-300))  # a first trial moves the gate about one width
            margin = gap + 64 * eps * (xp.abs(J) + 1e-300)
            finite = xp.isfinite(J) & xp.isfinite(gap) & xp.isfinite(gn2)
            taken = xp.zeros_like(active)
            for _ in range(12):
                if not bool(xp.any(active & finite & ~taken)):
                    break  # every node has stepped; further halvings could not accept anything
                w_t = _unit(w - alpha[:, None] * g_w)
                z_t = z - alpha * g_z
                WLt, WRt, _ = gate(w_t, z_t)
                Ft = (rows.reduce(WLt * rL + WRt * rR) + pen) / n
                accept = active & finite & ~taken & (Ft <= J - 1e-4 * alpha * gn2) & (J - Ft > margin)
                w = xp.where(accept[:, None], w_t, w)
                z = xp.where(accept, z_t, z)
                taken = taken | accept
                alpha = alpha * 0.5
            stopped = active & ~taken
            saturated = finite & (rows.reduce(sp_sm) / n < 1e-6)
            flat = finite & ~saturated & (T * xp.sqrt(gn2) <= margin)
            self.counters["soft_nonfinite"] += int(xp.sum(stopped & ~finite))
            self.counters["soft_gate_saturated"] += int(xp.sum(stopped & saturated))
            self.counters["soft_gradient_converged"] += int(xp.sum(stopped & flat))
            self.counters["soft_line_search_stalled"] += int(xp.sum(stopped & finite & ~saturated & ~flat))
            active = active & taken
        self.counters["soft_iteration_limit"] += int(xp.sum(active))

        # Harden: move z to the first right-side score so the predicate reproduces the cut.
        s = plane_scores(rows.U, w[lab])
        first_right = sort_by_block_then_value(xp.where(s >= z[lab], s, xp.inf), lab)[rows.starts]
        z_h = s[first_right]
        z_h = xp.where(s[first_right] >= z, z_h, xp.nan)
        sse, _ = _hard_sse(rows, w, xp.where(xp.isfinite(z_h), z_h, xp.inf), self.min_side)
        better = ok0 & (sse < sse0)
        self.counters["soft_improved"] += int(xp.sum(better))
        return xp.where(better[:, None], w, w0), xp.where(better, z_h, z0)


def segment_sum(a: Array, starts: Array) -> Array:
    """
    Sums of the row blocks ``a[starts[i]:starts[i + 1]]`` (the last runs to the end), for any
    number of trailing axes. Uses only the forms AFINNI mode's loop already runs on both
    backends: ``_reduce_sum_1d`` for 1-D input, 2-D ``add.reduceat`` otherwise.
    """
    if a.ndim == 1:
        return _reduce_sum_1d(a.astype(xp.float64), starts)
    return xp.add.reduceat(a.reshape(a.shape[0], -1), starts, axis=0).reshape(-1, *a.shape[1:])


def _soft_gates(rows: _SplitRows, w: Array, z: Array, T: Array, delta: float) -> tuple[Array, Array, Array]:
    """Floored soft memberships ``(W_L, W_R)`` and ``sigma(s) sigma(-s)``, with ``s = (w^T u - z) / T``."""
    s = (plane_scores(rows.U, w[rows.labels]) - z[rows.labels]) / T[rows.labels]
    sig_p, sig_m = _sigmoid(s), _sigmoid(-s)
    return delta + (1 - 2 * delta) * sig_m, delta + (1 - 2 * delta) * sig_p, sig_p * sig_m


def _soft_objective(rows: _SplitRows, w: Array, z: Array, T: Array, delta: float):
    """
    The consistent soft objective ``J = min_B F`` per node and its gradient in ``(w, z)``
    (not yet projected to ``|w| = 1``). With ``B`` at its minimizer, the envelope theorem
    leaves only the gate derivative: ``dJ/ds_i = (r_R,i - r_L,i) (1 - 2 delta) sigma sigma' / n``.
    Also returns the frozen residuals and penalty (for line-search trials), the inner
    solves' suboptimality estimate ``sum_s tr(R^T H^-1 R) / n`` and ``sigma(s) sigma(-s)``.
    ``R = Phi^T W (Y - Phi B) - lam B`` is formed from the data residual, not as
    ``Phi^T W Y - (H + lam I) B`` from the sums, which cancels to rounding noise near a fit.
    """
    n = rows.counts.astype(xp.float64)
    lab = rows.labels
    WL, WR, sp_sm = _soft_gates(rows, w, z, T, delta)
    res, pen, gap = [], 0.0, 0.0
    for Wt in (WL, WR):
        H, Bs = rows.gram(Wt)
        beta = rows.solve(H, Bs)
        E = rows.Y - xp.einsum("np,npf->nf", rows.Phi, beta[lab])
        res.append(xp.sum(E**2, axis=1))
        pen = pen + rows.lam * xp.sum(beta**2, axis=(1, 2))
        R = rows.cross(rows._Phi_g, rows.gather(E), Wt) - rows.lam[:, None, None] * beta
        gap = gap + xp.sum(R * rows.solve(H, R), axis=(1, 2)) / n
    rL, rR = res
    J = (rows.reduce(WL * rL + WR * rR) + pen) / n
    a = (rR - rL) * (1 - 2 * delta) * sp_sm / T[lab] / n[lab]
    return J, rows.reduce(a[:, None] * rows.U), -rows.reduce(a), rL, rR, pen, gap, sp_sm


def _unit(a: Array) -> Array:
    """Rows of ``a`` scaled to unit length; zero or non-finite rows become NaN (an invalid split)."""
    norm = xp.linalg.norm(a, axis=-1, keepdims=True)
    ok = xp.isfinite(norm) & (norm > 0)
    return xp.where(ok, a / xp.where(ok, norm, 1.0), xp.nan)


def _sigmoid(s: Array) -> Array:
    """The logistic function without overflow for large ``|s|`` and without clipping."""
    e = xp.exp(-xp.abs(s))
    return xp.where(s >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _interleave(a: Array, b: Array, first: Array) -> Array:
    """Rows of ``a`` where ``first`` is True and rows of ``b`` elsewhere, in order."""
    out = xp.empty((int(first.shape[0]), *a.shape[1:]), dtype=a.dtype)
    out[first] = a
    out[~first] = b
    return out


# --- Routing -------------------------------------------------------------------------


@dataclass
class RoutingTable:
    """
    The explicit tree, validated and stored as compact node arrays: node ``i`` is a split
    (children ``child[i]``, plane ``w[i]``, ``z[i]``) or a leaf (row ``node_leaf[i]`` of the
    leaf arrays), and ``root_node[r]`` is root ``r``'s node. Routing follows child indices,
    so no logical key is packed into an integer and every depth the path codes allow
    (``MAX_DEPTH_LIMIT``) routes. The logical identity ``(root, path_code)`` of every node
    is kept as provenance (``node_root``, ``node_path``), which saving and loading rebuild
    from, so a loaded table passes the same checks as a trained one.
    """

    domain: Domain
    roots: Roots
    is_leaf: Array  # (M,) bool
    child: Array  # (M, 2) int64 node indices, -1 for leaves
    w: Array  # (M, d), zero rows for leaves
    z: Array  # (M,)
    node_leaf: Array  # (M,) leaf row, -1 for splits
    root_node: Array  # (R,)
    max_depth: int
    node_root: np.ndarray  # (M,) provenance
    node_path: np.ndarray  # (M,) provenance

    @property
    def n_leaves(self) -> int:
        return int(np.sum(_to_host(self.is_leaf)))

    @classmethod
    def build(cls, domain: Domain, roots: Roots, leaf_root, leaf_path, split_root, split_path, w, z) -> "RoutingTable":
        """
        Assemble and check the tree: leaf ``i`` (in leaf-array order) is ``(leaf_root[i],
        leaf_path[i])``; split ``j`` is ``(split_root[j], split_path[j])`` with plane
        ``(w[j], z[j])``. Checks: a finite domain of positive width, roots of its dimension
        (positive grid bins; finite, distinct Voronoi centres), valid roots and positive
        path codes, finite planes of the domain's dimension, unique nodes, both children of
        every split, a split parent for every non-root node, and a node for every root.
        Loading goes through here too, so a malformed file fails like a malformed tree.
        Raises ``ValueError`` otherwise.
        """
        lo, hi = np.asarray(domain.lower), np.asarray(domain.upper)
        if lo.ndim != 1 or lo.shape != hi.shape or lo.size == 0 or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))) \
                or not np.all(hi > lo):  # fmt: skip
            raise ValueError("Region domain must be finite (d,) bounds with upper > lower.")
        bins = np.asarray(roots.bins)
        if bins.shape != lo.shape or not np.issubdtype(bins.dtype, np.integer) or np.any(bins < 1):
            raise ValueError("Region root bins must be positive integers, one per dimension.")
        if roots.centres is not None:
            C = np.asarray(roots.centres)
            if C.ndim != 2 or C.shape[1] != lo.size or C.shape[0] == 0 or not np.all(np.isfinite(C)) \
                    or len(np.unique(C, axis=0)) != len(C):  # fmt: skip
                raise ValueError("Voronoi root centres must be finite, distinct (R, d) points.")
        lr, lp, sr, sp = (np.asarray(_to_host(a), dtype=np.int64).reshape(-1) for a in (leaf_root, leaf_path, split_root, split_path))
        W, Z = np.asarray(_to_host(w), dtype=np.float64), np.asarray(_to_host(z), dtype=np.float64).reshape(-1)
        d, R = domain.d, roots.n_roots
        if len(lr) != len(lp) or len(sr) != len(sp) or W.shape != (len(sr), d) or Z.shape != (len(sr),):
            raise ValueError("Region tree arrays have inconsistent shapes.")
        roots_all, paths_all = np.r_[sr, lr], np.r_[sp, lp]
        if np.any(paths_all < 1) or np.any(roots_all < 0) or np.any(roots_all >= R):
            raise ValueError("Region tree has a non-positive path code or an out-of-range root.")
        if not (np.all(np.isfinite(W)) and np.all(np.isfinite(Z))):
            raise ValueError("Region tree has a non-finite split plane.")
        index = {}
        for i, key in enumerate(zip(roots_all.tolist(), paths_all.tolist(), strict=True)):
            if key in index:
                raise ValueError("Region tree has duplicate node keys.")
            index[key] = i
        S, M = len(sr), len(sr) + len(lr)
        child = np.full((M, 2), -1, dtype=np.int64)
        for j, (r, p) in enumerate(zip(sr.tolist(), sp.tolist(), strict=True)):
            left, right = index.get((r, 2 * p)), index.get((r, 2 * p + 1))
            if left is None or right is None:
                raise ValueError("A region split is missing one of its children.")
            child[j] = (left, right)
        splits = set(zip(sr.tolist(), sp.tolist(), strict=True))
        if any(p > 1 and (r, p >> 1) not in splits for r, p in zip(roots_all.tolist(), paths_all.tolist(), strict=True)):
            raise ValueError("A region node's parent is not a split node.")
        root_node = np.array([index.get((r, 1), -1) for r in range(R)], dtype=np.int64)
        if np.any(root_node < 0):
            raise ValueError("A region root has no node.")
        is_leaf = np.r_[np.zeros(S, dtype=bool), np.ones(len(lr), dtype=bool)]
        node_leaf = np.r_[np.full(S, -1, dtype=np.int64), np.arange(len(lr), dtype=np.int64)]
        node_w = np.vstack([W, np.zeros((len(lr), d))]) if M else np.zeros((0, d))
        max_depth = int(max((int(p).bit_length() - 1 for p in paths_all.tolist()), default=0))
        return cls(domain, roots, xp.asarray(is_leaf), xp.asarray(child), xp.asarray(node_w),
                   xp.asarray(np.r_[Z, np.zeros(len(lr))]), xp.asarray(node_leaf), xp.asarray(root_node),
                   max_depth, roots_all, paths_all)  # fmt: skip

    def route(self, X: Array) -> tuple[Array, Array]:
        """
        ``(leaf, outside)``: each query's leaf row and whether it lies outside the domain.
        Outside queries are routed from their nearest domain point (clipped coordinates)
        and evaluated at their own coordinates, i.e. extrapolated by that leaf; the audit
        says nothing about them. Non-finite queries raise ``ValueError``.
        """
        X = xp.asarray(X, dtype=xp.float64)
        if X.ndim != 2 or X.shape[1] != self.domain.d:
            raise ValueError(f"Queries must have shape (n, {self.domain.d}); got {tuple(X.shape)}.")
        if not bool(xp.all(xp.isfinite(X))):
            raise ValueError("Queries contain NaN or infinity.")
        U = self.domain.normalize(X)
        outside = xp.any((U < 0) | (U > 1), axis=1)
        U = xp.clip(U, 0.0, 1.0)
        node = self.root_node[self.roots.root_of(U)]
        for _ in range(self.max_depth + 1):
            act = xp.nonzero(~self.is_leaf[node])[0]
            if int(act.shape[0]) == 0:
                break
            na = node[act]
            side = (plane_scores(U[act], self.w[na]) - self.z[na] >= 0).astype(xp.int64)
            node[act] = self.child[na, side]
        return self.node_leaf[node], outside

    def arrays(self) -> dict[str, np.ndarray]:
        """The host arrays that ``from_arrays`` rebuilds the table from (for saving): domain, roots and provenance."""
        split = ~np.asarray(_to_host(self.is_leaf))
        leaf_rows = np.asarray(_to_host(self.node_leaf))[~split]
        order = np.argsort(leaf_rows)  # leaves in leaf-array order
        out = {
            "domain_lower": self.domain.lower, "domain_upper": self.domain.upper, "root_bins": self.roots.bins,
            "split_root": self.node_root[split], "split_path": self.node_path[split],
            "w": np.asarray(_to_host(self.w))[split], "z": np.asarray(_to_host(self.z))[split],
            "leaf_root": self.node_root[~split][order], "leaf_path": self.node_path[~split][order],
        }  # fmt: skip
        if self.roots.is_voronoi:
            out["root_centres"] = self.roots.centres
        return out

    @classmethod
    def from_arrays(cls, a: dict[str, np.ndarray]) -> "RoutingTable":
        """Rebuild (and so re-validate) a table from ``arrays()``; also reads the earlier packed-key format."""
        centres = np.asarray(a["root_centres"]) if "root_centres" in a else None
        domain = Domain(np.asarray(a["domain_lower"], dtype=np.float64), np.asarray(a["domain_upper"], dtype=np.float64))
        roots = Roots(np.asarray(a["root_bins"]), centres)
        if "split_root" in a:
            return cls.build(domain, roots, a["leaf_root"], a["leaf_path"], a["split_root"], a["split_path"], a["w"], a["z"])
        # Format written before compact indices: keys root << shift | path_code, sorted.
        shift = int(a["shift"])
        mask = (np.int64(1) << shift) - 1
        sk, lk = np.asarray(a["split_keys"], dtype=np.int64), np.asarray(a["leaf_keys"], dtype=np.int64)
        leaf_row = np.asarray(a["leaf_index"], dtype=np.int64)
        lr, lp = np.empty_like(lk), np.empty_like(lk)
        lr[leaf_row], lp[leaf_row] = lk >> shift, lk & mask
        return cls.build(domain, roots, lr, lp, sk >> shift, sk & mask, a["w"], a["z"])
