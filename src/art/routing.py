"""``RoutingTable``: the trained tree's split planes, used to route query points to their leaf."""
import math
from dataclasses import dataclass

import numpy as np

from src import Array, to_host, xp
from src.art.geometry import Domain, Roots, plane_scores


# --- Routing -------------------------------------------------------------------------
@dataclass
class RoutingTable:
    """
    The explicit tree, validated and stored as compact node arrays: node ``i`` is a split
    (children ``child[i]`` = (left, right), plane ``normal[i]``, ``offset[i]``: the plane
    ``(a, z)`` of the companion, whose side ``normal^T u >= offset`` is the right child) or a
    leaf (row ``node_leaf[i]`` of the leaf arrays), and ``root_node[r]`` is root ``r``'s node.
    Routing follows child indices,
    so no logical key is packed into an integer and every depth the path codes allow
    (``MAX_DEPTH_LIMIT``) routes. The logical identity ``(root, path_code)`` of every node
    is kept as provenance (``node_root``, ``node_path``), which saving and loading rebuild
    from, so a loaded table passes the same checks as a trained one.
    """

    domain: Domain
    roots: Roots
    is_leaf: Array  # (M,) bool
    child: Array  # (M, 2) int64 node indices, -1 for leaves
    normal: Array  # (M, d) unit plane normals, zero rows for leaves
    offset: Array  # (M,)
    node_leaf: Array  # (M,) leaf row, -1 for splits
    root_node: Array  # (R,)
    max_depth: int
    node_root: np.ndarray  # (M,) provenance
    node_path: np.ndarray  # (M,) provenance

    @property
    def n_leaves(self) -> int:
        return int(np.sum(to_host(self.is_leaf)))

    @classmethod
    def build(cls, domain: Domain, roots: Roots, leaf_root, leaf_path, split_root, split_path, normal, offset) -> "RoutingTable":
        """
        Assemble and check the tree: leaf ``i`` (in leaf-array order) is ``(leaf_root[i],
        leaf_path[i])``; split ``j`` is ``(split_root[j], split_path[j])`` with plane
        ``(normal[j], offset[j])``. Checks: a finite domain of positive width, roots of its dimension
        (positive grid bins; finite, distinct Voronoi centres), valid roots and positive
        path codes, finite planes of the domain's dimension, unique nodes, both children of
        every split, a split parent for every non-root node, and a node for every root.
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
        lr, lp, sr, sp = (np.asarray(to_host(a), dtype=np.int64).reshape(-1) for a in (leaf_root, leaf_path, split_root, split_path))
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
        if len(lr) != len(lp) or len(sr) != len(sp) or W.shape != (len(sr), d) or Z.shape != (len(sr),):
            raise ValueError("ART tree arrays have inconsistent shapes.")
        roots_all, paths_all = np.r_[sr, lr], np.r_[sp, lp]
        if np.any(paths_all < 1) or np.any(roots_all < 0) or np.any(roots_all >= R):
            raise ValueError("ART tree has a non-positive path code or an out-of-range root.")
        if not (np.all(np.isfinite(W)) and np.all(np.isfinite(Z))):
            raise ValueError("ART tree has a non-finite split plane.")
        index = {}
        for i, key in enumerate(zip(roots_all.tolist(), paths_all.tolist(), strict=True)):
            if key in index:
                raise ValueError("ART tree has duplicate node keys.")
            index[key] = i
        S, M = len(sr), len(sr) + len(lr)
        child = np.full((M, 2), -1, dtype=np.int64)
        for j, (r, p) in enumerate(zip(sr.tolist(), sp.tolist(), strict=True)):
            left, right = index.get((r, 2 * p)), index.get((r, 2 * p + 1))
            if left is None or right is None:
                raise ValueError("An ART split is missing one of its children.")
            child[j] = (left, right)
        splits = set(zip(sr.tolist(), sp.tolist(), strict=True))
        if any(p > 1 and (r, p >> 1) not in splits for r, p in zip(roots_all.tolist(), paths_all.tolist(), strict=True)):
            raise ValueError("A ART node's parent is not a split node.")
        root_node = np.array([index.get((r, 1), -1) for r in range(R)], dtype=np.int64)
        if np.any(root_node < 0):
            raise ValueError("An ART root has no node.")
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
            side = (plane_scores(U[act], self.normal[na]) - self.offset[na] >= 0).astype(xp.int64)
            node[act] = self.child[na, side]
        return self.node_leaf[node], outside

    def arrays(self) -> dict[str, np.ndarray]:
        """
        The host arrays that ``from_arrays`` rebuilds the table from (for saving): domain, roots and
        provenance. The split planes are stored under the keys ``"w"`` (normals) and ``"z"``
        (offsets), the model file's format.
        """
        split = ~np.asarray(to_host(self.is_leaf))
        leaf_rows = np.asarray(to_host(self.node_leaf))[~split]
        order = np.argsort(leaf_rows)  # leaves in leaf-array order
        out = {
            "domain_lower": self.domain.lower, "domain_upper": self.domain.upper, "root_bins": self.roots.bins,
            "split_root": self.node_root[split], "split_path": self.node_path[split],
            "w": np.asarray(to_host(self.normal))[split], "z": np.asarray(to_host(self.offset))[split],
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
