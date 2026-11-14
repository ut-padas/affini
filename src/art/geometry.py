"""The routing domain, the root cells (box, grid, Voronoi) and the plane predicates training and routing share."""

from dataclasses import dataclass

import numpy as np

from src import Array, xp
from src.numerics import sort_by_block_then_value, uniform_grid_bins


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
        """The coordinates ``u`` in ``[0, 1]^d`` of the domain, in which every ART plane and region is stated."""
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
        upper cell; Voronoi points to the nearest centre (ties to the lowest index).
        Training, sampling and routing all assign roots through this.

        Voronoi ownership is decided relative to a candidate ``r`` (the smallest squared
        distance): ``|u - c_j|^2 - |u - c_r|^2 = D_j^T (D_j - 2 (u - c_r))`` with
        ``D_j = c_j - c_r``, the sampler's bisector test. Full squared distances share
        the tangential part, which can absorb the normal difference of close centres
        (2-D centres 1e-8 apart misassigned 21 of 200 in-cell points).
        """
        if self.is_voronoi:
            C = xp.asarray(self.centres)
            rows = max(1, self.TILE_ELEMENTS // max(1, 3 * C.size))
            out = []
            for s in range(0, int(U.shape[0]), rows):
                u = U[s : s + rows]
                cand = xp.argmin(xp.sum((u[:, None, :] - C[None]) ** 2, axis=2), axis=1)
                D = C[None] - C[cand][:, None, :]
                rel = xp.sum(D * (D - 2.0 * (u - C[cand])[:, None, :]), axis=2)  # 0 at the candidate
                out.append(xp.argmin(rel, axis=1))
            return xp.concatenate(out).astype(xp.int64) if out else xp.zeros(0, dtype=xp.int64)
        bins = xp.asarray(self.bins)
        coords = xp.minimum(xp.floor(U * bins).astype(xp.int64), bins - 1)
        coords = xp.maximum(coords, 0)
        strides = xp.asarray(np.r_[np.cumprod(self.bins[::-1])[::-1][1:], 1].astype(np.int64))
        return xp.sum(coords * strides, axis=1)


# --- Plane predicates ----------------------------------------------------------------
def plane_scores(U: Array, normals: Array) -> Array:
    """
    ``a_i^T u_i`` row by row, with ``a_i`` the unit normal of point ``i``'s plane (``a`` in the
    companion); training and routing both call this, so they classify identically.
    """
    return xp.sum(U * normals, axis=1)


def gap_cuts(sorted_scores: Array, labels: Array, counts: Array, min_side: int) -> tuple[Array, Array, Array]:
    """
    Per node, the cut position nearest the median such that both sides keep
    ``min_side`` points and the cut falls between two *distinct* scores, so the
    predicate ``score >= z`` with the offset ``z`` the first right-side score reproduces the
    cut exactly. ``sorted_scores`` and ``labels`` are in split order (ascending within each
    node). Returns ``(valid, position, z)``, each (K,).
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
