"""``RoutingTable``: the trained tree's split planes, used to route query points to their leaf."""
import math
from dataclasses import dataclass

import numpy as np

from src import IS_CUPY_BACKEND, Array, to_host, xp
from src.art.geometry import Domain, Roots

_CPU_KERNELS = None
_GPU_KERNELS = None
_THREADS = 128  # threads per block of the GPU routing kernels (one thread per query)


def _gpu_kernels():
    """``(route_prepare, route_walk)`` of ``kernels/art_routing.cu`` (compiled on first use)."""
    global _GPU_KERNELS
    if _GPU_KERNELS is None:
        from src import kernels

        module = kernels.module("art_routing")
        _GPU_KERNELS = (module.get_function("route_prepare"), module.get_function("route_walk"))
    return _GPU_KERNELS


def _cpu_kernels():
    """
    ``(walk, pairwise_dot, prepare)``: the Numba forms of ``kernels/art_routing.cu``'s
    ``route_walk`` and ``route_prepare`` and their dot product, compiled on first use (cached on
    disk), statement for statement, so both backends route every query to the same leaf.

    The dot product adds ``u_j a_j`` in the order of NumPy's row sum, which ``plane_scores`` (and
    with it every training split) uses: left to right below 8 terms, eight partial sums combined
    as ``((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))`` up to 128, and halves (rounded to a
    multiple of 8) beyond. ``fastmath=False`` keeps every product and sum rounded on its own (no
    fused multiply-add). ``tests/unit/test_art.py`` checks both against the level-by-level
    definition.
    """
    global _CPU_KERNELS
    if _CPU_KERNELS is not None:
        return _CPU_KERNELS
    import numba as nb

    @nb.njit(cache=True, fastmath=False)
    def pairwise_dot(u, w, lo, n):  # pragma: no cover - compiled
        if n < 8:
            res = 0.0
            for t in range(lo, lo + n):
                res += u[t] * w[t]
            return res
        if n <= 128:
            r0, r1, r2, r3 = u[lo] * w[lo], u[lo + 1] * w[lo + 1], u[lo + 2] * w[lo + 2], u[lo + 3] * w[lo + 3]
            r4, r5, r6, r7 = u[lo + 4] * w[lo + 4], u[lo + 5] * w[lo + 5], u[lo + 6] * w[lo + 6], u[lo + 7] * w[lo + 7]
            i, stop = 8, n - n % 8
            while i < stop:
                t = lo + i
                r0 += u[t] * w[t]
                r1 += u[t + 1] * w[t + 1]
                r2 += u[t + 2] * w[t + 2]
                r3 += u[t + 3] * w[t + 3]
                r4 += u[t + 4] * w[t + 4]
                r5 += u[t + 5] * w[t + 5]
                r6 += u[t + 6] * w[t + 6]
                r7 += u[t + 7] * w[t + 7]
                i += 8
            res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < n:
                res += u[lo + i] * w[lo + i]
                i += 1
            return res
        n2 = n // 2
        n2 -= n2 % 8
        return pairwise_dot(u, w, lo, n2) + pairwise_dot(u, w, lo + n2, n - n2)

    @nb.njit(parallel=True, cache=True, fastmath=False)
    def walk(U, start, is_leaf, normal, offset, child, out):  # pragma: no cover - compiled
        d = U.shape[1]
        for i in nb.prange(U.shape[0]):
            node = start[i]
            # A query stops at its own leaf: no later step could move it (leaves are their own children).
            while not is_leaf[node]:
                s = pairwise_dot(U[i], normal[node], 0, d)
                node = child[node, 1 if s - offset[node] >= 0 else 0]
            out[i] = node

    @nb.njit(parallel=True, cache=True, fastmath=False)
    def prepare(X, lower, width, U, outside):  # pragma: no cover - compiled
        # One pass for the NumPy prologue's five: the same (x - lower) / width per coordinate as
        # Domain.normalize, the outside flag before clipping, and the clip. Returns the rows
        # holding a NaN or infinity.
        bad = 0
        for i in nb.prange(X.shape[0]):
            out, nonfinite = False, 0
            for j in range(X.shape[1]):
                x = X[i, j]
                if not np.isfinite(x):
                    nonfinite = 1
                u = (x - lower[j]) / width[j]
                if u < 0.0 or u > 1.0:
                    out = True
                U[i, j] = min(max(u, 0.0), 1.0)
            outside[i] = out
            bad += nonfinite
        return bad

    _CPU_KERNELS = (walk, pairwise_dot, prepare)
    return _CPU_KERNELS


# --- Routing -------------------------------------------------------------------------
@dataclass
class RoutingTable:
    """
    The explicit tree, validated and stored as compact node arrays: node ``i`` is a split
    (children ``child[i]`` = (left, right), plane ``normal[i]``, ``offset[i]``: the plane
    ``(a, z)`` of the companion, whose side ``normal^T u >= offset`` is the right child) or a
    leaf (row ``node_leaf[i]`` of the leaf arrays), and ``root_node[r]`` is root ``r``'s node.
    A leaf is its own child on both sides, with a zero plane (``0 >= 0``: right), so a query
    that reaches it stays there: ``max_depth`` level-by-level steps take every query to its leaf.
    Routing follows child indices, so no logical key is packed into an integer and every
    depth the locational codes allow (``MAX_DEPTH_LIMIT``) routes. The logical identity
    ``(root, locational_code)`` of every node is kept as provenance (``node_root``,
    ``node_locational_code``), which saving and loading rebuild from, so a loaded table
    passes the same checks as a trained one.
    """

    domain: Domain
    roots: Roots
    is_leaf: Array  # (M,) bool
    child: Array  # (M, 2) int64 node indices; a leaf's are its own index
    normal: Array  # (M, d) unit plane normals, zero rows for leaves
    offset: Array  # (M,)
    node_leaf: Array  # (M,) leaf row, -1 for splits
    root_node: Array  # (R,)
    max_depth: int
    node_root: np.ndarray  # (M,) provenance
    node_locational_code: np.ndarray  # (M,) provenance

    @property
    def n_leaves(self) -> int:
        return int(np.sum(to_host(self.is_leaf)))

    @classmethod
    def build(cls, domain: Domain, roots: Roots, leaf_root, leaf_locational_code, split_root, split_locational_code,
              normal, offset) -> "RoutingTable":
        """
        Assemble and check the tree: leaf ``i`` (in leaf-array order) is ``(leaf_root[i],
        leaf_locational_code[i])``; split ``j`` is ``(split_root[j], split_locational_code[j])``
        with plane ``(normal[j], offset[j])``. Checks: a finite domain of positive width, roots
        of its dimension (positive grid bins; finite, distinct Voronoi centres), valid roots and
        positive locational codes, finite planes of the domain's dimension, unique nodes, both
        children of every split, a split parent for every non-root node, and a node for every root.
        Loading goes through here too, so a malformed file fails like a malformed tree.
        Raises ``ValueError`` otherwise.
        """
        lo, hi = np.asarray(domain.lower), np.asarray(domain.upper)
        if lo.ndim != 1 or lo.shape != hi.shape or lo.size == 0 or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))) \
                or not np.all(hi > lo):  # fmt: skip
            raise ValueError("ART domain must be finite (d,) bounds with upper > lower.")
        bins = np.asarray(roots.bins)
        if bins.shape != lo.shape or not np.issubdtype(bins.dtype, np.integer) or np.any(bins < 1):
            raise ValueError("ART root bins must be positive integers, one per dimension.")
        lr, lc, sr, sc = (np.asarray(to_host(a), dtype=np.int64).reshape(-1)
                          for a in (leaf_root, leaf_locational_code, split_root, split_locational_code))
        # Every root must have a node, so a valid table has at most as many roots as nodes;
        # checked in Python integers, before n_roots' fixed-width product could wrap.
        n_nodes = len(lr) + len(sr)
        n_root_cells = math.prod(int(b) for b in bins) if roots.centres is None else len(roots.centres)
        if n_root_cells > n_nodes:
            raise ValueError(f"ART roots ({n_root_cells}) outnumber the tree's nodes ({n_nodes}).")
        if roots.centres is not None:
            C = np.asarray(roots.centres)
            if C.ndim != 2 or C.shape[1] != lo.size or C.shape[0] == 0 or not np.all(np.isfinite(C)) \
                    or len(np.unique(C, axis=0)) != len(C):  # fmt: skip
                raise ValueError("Voronoi root centres must be finite, distinct (R, d) points.")
        W, Z = np.asarray(to_host(normal), dtype=np.float64), np.asarray(to_host(offset), dtype=np.float64).reshape(-1)
        d, R = domain.d, roots.n_roots
        if len(lr) != len(lc) or len(sr) != len(sc) or W.shape != (len(sr), d) or Z.shape != (len(sr),):
            raise ValueError("ART tree arrays have inconsistent shapes.")
        roots_all, codes_all = np.r_[sr, lr], np.r_[sc, lc]
        if np.any(codes_all < 1) or np.any(roots_all < 0) or np.any(roots_all >= R):
            raise ValueError("ART tree has a non-positive locational code or an out-of-range root.")
        if not (np.all(np.isfinite(W)) and np.all(np.isfinite(Z))):
            raise ValueError("ART tree has a non-finite split plane.")
        index = {}
        for i, key in enumerate(zip(roots_all.tolist(), codes_all.tolist(), strict=True)):
            if key in index:
                raise ValueError("ART tree has duplicate node keys.")
            index[key] = i
        S, M = len(sr), len(sr) + len(lr)
        child = np.full((M, 2), -1, dtype=np.int64)
        for j, (r, p) in enumerate(zip(sr.tolist(), sc.tolist(), strict=True)):
            left, right = index.get((r, 2 * p)), index.get((r, 2 * p + 1))
            if left is None or right is None:
                raise ValueError("An ART split is missing one of its children.")
            child[j] = (left, right)
        child[S:] = np.arange(S, M, dtype=np.int64)[:, None]  # leaves (zero plane, see the class docstring)
        splits = set(zip(sr.tolist(), sc.tolist(), strict=True))
        if any(p > 1 and (r, p >> 1) not in splits for r, p in zip(roots_all.tolist(), codes_all.tolist(), strict=True)):
            raise ValueError("A ART node's parent is not a split node.")
        root_node = np.array([index.get((r, 1), -1) for r in range(R)], dtype=np.int64)
        if np.any(root_node < 0):
            raise ValueError("An ART root has no node.")
        is_leaf = np.r_[np.zeros(S, dtype=bool), np.ones(len(lr), dtype=bool)]
        node_leaf = np.r_[np.full(S, -1, dtype=np.int64), np.arange(len(lr), dtype=np.int64)]
        node_w = np.vstack([W, np.zeros((len(lr), d))]) if M else np.zeros((0, d))
        max_depth = int(max((int(p).bit_length() - 1 for p in codes_all.tolist()), default=0))
        return cls(domain, roots, xp.asarray(is_leaf), xp.asarray(child), xp.asarray(node_w),
                   xp.asarray(np.r_[Z, np.zeros(len(lr))]), xp.asarray(node_leaf), xp.asarray(root_node),
                   max_depth, roots_all, codes_all)  # fmt: skip

    def route(self, X: Array) -> tuple[Array, Array]:
        """
        ``(leaf, outside)``: each query's leaf row and whether it lies outside the domain.
        Outside queries are routed from their nearest domain point (clipped coordinates)
        and evaluated at their own coordinates, i.e. extrapolated by that leaf; the audit
        says nothing about them. Non-finite queries raise ``ValueError``.
        """
        U, outside = self.prepare_queries(X)
        return self.node_leaf[self.walk(U, self.root_node[self.roots.root_of(U)])], outside

    def prepare_queries(self, X: Array) -> tuple[Array, Array]:
        """
        ``(U, outside)``: the queries in the domain's unit box, clipped to it, and whether each lay
        outside before the clip, in one compiled pass with ``Domain.normalize``'s arithmetic.
        Raises ``ValueError`` for a misshaped query array or a NaN or infinite coordinate.
        """
        X = xp.ascontiguousarray(xp.asarray(X, dtype=xp.float64))
        if X.ndim != 2 or X.shape[1] != self.domain.d:
            raise ValueError(f"Queries must have shape (n, {self.domain.d}); got {tuple(X.shape)}.")
        n, d = X.shape
        U, outside = xp.empty_like(X), xp.empty(n, dtype=bool)
        lower = xp.asarray(self.domain.lower, dtype=xp.float64)
        width = xp.asarray(self.domain.width, dtype=xp.float64)
        if IS_CUPY_BACKEND:
            nonfinite = xp.zeros(1, dtype=xp.int32)
            if n:
                _gpu_kernels()[0](((n + _THREADS - 1) // _THREADS,), (_THREADS,),
                                  (X, lower, width, U, outside, nonfinite, np.int64(n), np.int32(d)))  # fmt: skip
            bad = int(nonfinite[0])
        else:
            bad = _cpu_kernels()[2](X, lower, width, U, outside)
        if bad:
            raise ValueError("Queries contain NaN or infinity.")
        return U, outside

    def walk(self, U: Array, start: Array) -> Array:
        """
        The leaf node of every query of ``prepare_queries``' ``U``, walking from node ``start``:
        one compiled thread per query, reading only the planes on its own path and stopping at
        its leaf. Every plane score adds its terms in the order ``plane_scores`` uses (NumPy's
        row sum), so a query lands where the level-by-level definition
        ``node = child[node, plane_scores(u, normal[node]) - offset[node] >= 0]`` sends it, and
        each training point in its own leaf.
        """
        start = xp.ascontiguousarray(start, dtype=xp.int64)
        out = xp.empty_like(start)
        n, d = int(U.shape[0]), int(U.shape[1])
        if n == 0:
            return out
        if IS_CUPY_BACKEND:
            _gpu_kernels()[1](((n + _THREADS - 1) // _THREADS,), (_THREADS,),
                              (U, start, self.is_leaf, self.normal, self.offset, self.child, out, np.int64(n), np.int32(d)))  # fmt: skip
        else:
            _cpu_kernels()[0](U, start, self.is_leaf, self.normal, self.offset, self.child, out)
        return out

    def arrays(self) -> dict[str, np.ndarray]:
        """
        The host arrays that ``from_arrays`` rebuilds the table from (for saving): domain, roots,
        the split planes and every node's root and locational code.
        """
        split = ~np.asarray(to_host(self.is_leaf))
        leaf_rows = np.asarray(to_host(self.node_leaf))[~split]
        order = np.argsort(leaf_rows)  # leaves in leaf-array order
        out = {
            "domain_lower": self.domain.lower, "domain_upper": self.domain.upper, "root_bins": self.roots.bins,
            "split_root": self.node_root[split], "split_locational_code": self.node_locational_code[split],
            "normal": np.asarray(to_host(self.normal))[split], "offset": np.asarray(to_host(self.offset))[split],
            "leaf_root": self.node_root[~split][order],
            "leaf_locational_code": self.node_locational_code[~split][order],
        }  # fmt: skip
        if self.roots.is_voronoi:
            out["root_centres"] = self.roots.centres
        return out

    @classmethod
    def from_arrays(cls, a: dict[str, np.ndarray]) -> "RoutingTable":
        """Rebuild (and so re-validate) a table from ``arrays()``."""
        centres = np.asarray(a["root_centres"]) if "root_centres" in a else None
        domain = Domain(np.asarray(a["domain_lower"], dtype=np.float64), np.asarray(a["domain_upper"], dtype=np.float64))
        roots = Roots(np.asarray(a["root_bins"]), centres)
        return cls.build(domain, roots, a["leaf_root"], a["leaf_locational_code"], a["split_root"],
                         a["split_locational_code"], a["normal"], a["offset"])
