"""
Binary trees recorded during refinement (``ClusteringConfig.track_trees``).

Every initial cluster roots one binary tree. Each rejected cluster becomes an
internal node holding the hyperplane ``w^T (x - c) + b = 0`` that split it, with
``c`` the cluster's centroid; its children lie on side 0 (``w^T (x - c) + b < 0``)
and side 1 (``> 0``). Every other
cluster becomes a leaf node: an accepted cluster, or an unresolved one (dropped
at ``max_levels``, with no leaf data). So every child slot of an internal node
holds exactly one node.

Nodes are identified within their tree by a Morton ID (int64): the path from the
root, one bit per level, left-aligned in bits 62..DEPTH_BITS, with the depth in
the low DEPTH_BITS bits. The root is 0, and the last path bit is the side of the
parent a node lies on (``side_of``). Sorting by Morton ID is preorder: a node
precedes its descendants, its side-0 subtree precedes its side-1 subtree, and
every subtree is a contiguous range.

``BinaryTreeForest`` holds every tree of one rank (or, after merging, of the whole
run) as two tables of contiguous device arrays, ``internal_nodes`` and
``leaf_nodes``, in arrival order while training; ``sort`` reorders both by
(root, Morton ID), so each tree is one contiguous block of each, which ``tree``
returns as a ``BinaryTree``. ``find_leaves`` routes points top-down to their leaf,
locating each node by its Morton ID (``find_rows``), so no child pointers are stored.
``ancestor_planes`` walks the other way, following ``parent`` rows up from a batch
of clusters to collect the half-spaces that bound each one's region; it works on
the unsorted forest, so samplers can use it during training.
"""

from dataclasses import dataclass

import numpy as np

from typing import Literal

from src import IS_CUPY_BACKEND, Array, _to_device, _to_host, xp
from src.clusters import ColumnBuffer, ParamSpec
from src.knn import make_knn_index
from src.numerics import sort_by_block_then_value

DEPTH_BITS = 6
DEPTH_MASK = (1 << DEPTH_BITS) - 1
# Deepest node a Morton ID can address: one path bit per level in bits 62..DEPTH_BITS.
MAX_DEPTH = 63 - DEPTH_BITS

LEAF_COLUMNS = {
    "parent": ParamSpec((), np.int64),  # row in internal_nodes; -1 for an unsplit root
    "root": ParamSpec((), np.int64),
    "morton": ParamSpec((), np.int64),
    # Row in the trained leaf arrays (centroids, local models, ...); -1 if unresolved.
    "leaf": ParamSpec((), np.int64),
}


def internal_columns(d: int) -> dict[str, ParamSpec]:
    """Columns of ``internal_nodes``: the hyperplane, the cluster's centroid, then the node's place in its tree."""
    return {
        "W": ParamSpec((d,)),
        "b": ParamSpec(()),
        # Mean of the cluster's points when it was split (before augmentation); a
        # root's split sits at its root centroid. The hyperplane is measured from
        # it, w^T (x - centroid) + b. Not strictly required (the absolute form
        # w^T x + (b - w^T centroid) holds the same plane), but it keeps the side
        # test on the cluster's scale (numerical stability) and places nodes for
        # debugging and plotting.
        "centroid": ParamSpec((d,)),
        "parent": ParamSpec((), np.int64),  # row in internal_nodes; -1 for a root's split
        "root": ParamSpec((), np.int64),
        "morton": ParamSpec((), np.int64),
    }


# Both functions use only operators, no array-module calls, so they work on host
# and device arrays alike.


def child_morton(morton: Array, side: Array) -> Array:
    """Morton IDs of the children on ``side`` (0 or 1) of nodes ``morton``; callers keep depth < MAX_DEPTH."""
    depth = morton & DEPTH_MASK
    return (morton & ~DEPTH_MASK) | (side.astype(np.int64) << (62 - depth)) | (depth + 1)


def side_of(morton: Array) -> Array:
    """The side (0 or 1) of its parent's hyperplane each node lies on: its last path bit (0 for a root)."""
    return (morton >> (63 - (morton & DEPTH_MASK))) & 1


@dataclass(frozen=True)
class BinaryTree:
    """
    One tree of a sorted forest, both tables in preorder (internal row 0 is the
    root's split, if the root was split at all). ``parent`` columns hold rows of
    this tree's ``internal_nodes`` (-1 for the root); other columns are views
    into the forest.
    """

    root: int
    internal_nodes: dict[str, Array]
    leaf_nodes: dict[str, Array]

    @property
    def n_internal(self) -> int:
        return int(self.internal_nodes["b"].shape[0])

    @property
    def n_leaves(self) -> int:
        return int(self.leaf_nodes["leaf"].shape[0])


class BinaryTreeForest:
    """
    Every binary tree of one rank, as two device tables (``internal_columns``,
    ``LEAF_COLUMNS``) whose ``parent`` columns refer to rows of ``internal_nodes``.

    While unsorted, rows are in arrival order and ``add_splits`` / ``add_leaves``
    write into spare capacity (grown by at least doubling). ``sort`` reorders both
    tables by (root, Morton ID), remaps the parents and freezes the forest, after
    which ``tree`` slices out single trees. ``root_centroids`` (n_roots, d), the
    means of the initial clusters indexed by root, are set on the merged forest:
    traversal starts by choosing a tree with them.
    """

    def __init__(self, d: int, capacity: int = 1024):
        self.d = d
        self._internal = ColumnBuffer(internal_columns(d), capacity, on_device=True)
        self._leaves = ColumnBuffer(LEAF_COLUMNS, capacity, on_device=True)
        self._sorted: tuple[dict[str, Array], dict[str, Array]] | None = None
        self.root_centroids: Array | None = None
        # Set by sort(): the roots present (ascending) and where each tree's rows
        # start in each table (n_roots + 1 entries each).
        self.roots: np.ndarray | None = None
        self.internal_starts: np.ndarray | None = None
        self.leaf_starts: np.ndarray | None = None
        # Device copies for find_rows, and its binary-search length per table.
        self._search: dict[str, tuple[Array, Array, int]] = {}
        self._root_index = None  # exact kNN index over root_centroids, built on first use

    @property
    def is_sorted(self) -> bool:
        return self._sorted is not None

    @property
    def internal_nodes(self) -> dict[str, Array]:
        return self._sorted[0] if self.is_sorted else self._internal.result()

    @property
    def leaf_nodes(self) -> dict[str, Array]:
        return self._sorted[1] if self.is_sorted else self._leaves.result()

    @property
    def n_internal(self) -> int:
        return int(self.internal_nodes["b"].shape[0])

    @property
    def n_leaves(self) -> int:
        return int(self.leaf_nodes["leaf"].shape[0])

    def add_splits(
        self, W: Array, b: Array, centroid: Array, parent: Array, root: Array, morton: Array
    ) -> Array:
        """
        Record one internal node per split cluster and return their rows (device
        int64), which become the ``parent`` of the clusters' children.
        """
        self._check_unsorted()
        start = self._internal.n_rows
        self._internal.append(
            {"W": W, "b": b, "centroid": centroid, "parent": parent, "root": root, "morton": morton}
        )
        return xp.arange(start, self._internal.n_rows, dtype=xp.int64)

    def add_leaves(self, parent: Array, root: Array, morton: Array, leaf: Array) -> None:
        """Record leaf nodes; ``leaf`` is each one's row in the leaf arrays, or -1 if unresolved."""
        self._check_unsorted()
        self._leaves.append({"parent": parent, "root": root, "morton": morton, "leaf": leaf})

    def _check_unsorted(self) -> None:
        if self.is_sorted:
            raise RuntimeError("The forest is sorted (frozen); add nodes before sort().")

    def to_host(self) -> dict:
        """Both tables and ``root_centroids`` as host arrays, for sending between ranks."""
        tables = {"internal_nodes": self.internal_nodes, "leaf_nodes": self.leaf_nodes}
        out = {kind: {k: np.asarray(_to_host(v)) for k, v in t.items()} for kind, t in tables.items()}
        out["root_centroids"] = None if self.root_centroids is None else np.asarray(_to_host(self.root_centroids))
        return out

    @classmethod
    def sorted_from(cls, arrays: dict, d: int) -> "BinaryTreeForest":
        """
        A sorted forest from ``to_host``-style ``arrays`` (``parent`` referring to
        rows of these arrays). Arrays of an already sorted forest keep their order:
        (root, morton) identifies a node uniquely within each table.
        """
        internal, leaves = arrays["internal_nodes"], arrays["leaf_nodes"]
        forest = cls(d, capacity=max(len(internal["b"]), len(leaves["leaf"])))
        forest.add_splits(**internal)
        forest.add_leaves(**leaves)
        if arrays.get("root_centroids") is not None:
            forest.root_centroids = xp.asarray(arrays["root_centroids"])
        forest.sort()
        return forest

    def sort(self) -> None:
        """Reorder both tables by (root, Morton ID), remap their parents, and freeze the forest."""
        self._check_unsorted()
        internal, leaves = self._internal.result(), self._leaves.result()
        n = int(internal["b"].shape[0])
        order = sort_by_block_then_value(internal["morton"], internal["root"])
        new_row = xp.empty((n,), dtype=xp.int64)
        new_row[order] = xp.arange(n, dtype=xp.int64)

        def remapped(table, table_order):
            out = {k: v[table_order] for k, v in table.items()}
            if n:  # with no internal rows every parent is -1 (and new_row is empty)
                out["parent"] = xp.where(out["parent"] >= 0, new_row[out["parent"]], -1)
            return out

        internal = remapped(internal, order)
        leaves = remapped(leaves, sort_by_block_then_value(leaves["morton"], leaves["root"]))
        self._sorted = (internal, leaves)
        self._internal = self._leaves = None  # the unsorted buffers are no longer used

        internal_roots, leaf_roots = _to_host(internal["root"]), _to_host(leaves["root"])
        self.roots = np.union1d(internal_roots, leaf_roots).astype(np.int64)
        self.internal_starts = np.searchsorted(internal_roots, np.r_[self.roots, np.iinfo(np.int64).max])
        self.leaf_starts = np.searchsorted(leaf_roots, np.r_[self.roots, np.iinfo(np.int64).max])
        for kind, starts in (("internal_nodes", self.internal_starts), ("leaf_nodes", self.leaf_starts)):
            largest_tree = int(np.diff(starts).max(initial=0))
            # A lower-bound search over m rows settles in bit_length(m) halvings.
            self._search[kind] = (xp.asarray(self.roots), xp.asarray(starts), largest_tree.bit_length())

    def tree(self, root: int) -> BinaryTree:
        """The tree grown from initial cluster ``root``; empty if this forest has none."""
        if not self.is_sorted:
            raise RuntimeError("Call sort() before taking single trees.")
        i = int(np.searchsorted(self.roots, root))
        found = i < len(self.roots) and self.roots[i] == root
        internal = slice(*self.internal_starts[i : i + 2]) if found else slice(0, 0)
        leaves = slice(*self.leaf_starts[i : i + 2]) if found else slice(0, 0)

        def local(table, rows):
            out = {k: v[rows] for k, v in table.items()}
            out["parent"] = xp.where(out["parent"] >= 0, out["parent"] - internal.start, -1)
            return out

        return BinaryTree(
            root=int(root),
            internal_nodes=local(self.internal_nodes, internal),
            leaf_nodes=local(self.leaf_nodes, leaves),
        )

    def find_rows(
        self, kind: Literal["internal_nodes", "leaf_nodes"], root: Array, morton: Array
    ) -> Array:
        """
        Row of each node (``root``, ``morton``) in the sorted table ``kind``, or -1
        where the table has no such node: a binary search inside each node's tree
        block, vectorized over the nodes (the same number of halvings for all).
        """
        if not self.is_sorted:
            raise RuntimeError("Call sort() before looking up nodes.")
        mortons = getattr(self, kind)["morton"]
        roots, starts, steps = self._search[kind]
        if mortons.shape[0] == 0 or roots.shape[0] == 0:
            return xp.full(root.shape, -1, dtype=xp.int64)
        tree = xp.minimum(xp.searchsorted(roots, root), roots.shape[0] - 1)
        known = roots[tree] == root
        lo = xp.where(known, starts[tree], 0)
        end = xp.where(known, starts[tree + 1], 0)  # unknown roots search an empty range
        hi = end
        last = mortons.shape[0] - 1
        for _ in range(steps):
            mid = (lo + hi) // 2
            right = (mid < hi) & (mortons[xp.minimum(mid, last)] < morton)
            lo = xp.where(right, mid + 1, lo)
            hi = xp.where(right, hi, mid)
        found = (lo < end) & (mortons[xp.minimum(lo, last)] == morton)
        return xp.where(found, lo, -1)

    def ancestor_planes(self, parent: Array, morton: Array, origins: Array) -> tuple[Array, Array]:
        """
        The half-spaces that bound K clusters, gathered from their ancestors' splits.

        A cluster's region is the domain (its root's cell) intersected with one
        half-space per ancestor: the side of that ancestor's hyperplane it descends
        from. ``parent``/``morton`` (K,) are the clusters' tree fields
        (``ClusterBatch.tree_fields``) and ``origins`` (K, d) their centroids.
        Returns ``planes`` (K, D, d+1) float64 on the device, D the deepest of the
        clusters, and ``depth`` (K,) int64, the number of real rows of each:

        - Row ``planes[k, j] = (a, beta)`` is the constraint
          ``a @ (x - origins[k]) + beta <= 0``. ``a`` is the ancestor's normal,
          scaled to unit length and flipped for side 1, so ``beta`` is the signed
          distance from ``origins[k]`` to that face (negative inside).
        - Rows are ordered nearest ancestor first (j = 0 is the split that made the
          cluster), usually the tightest faces, so a rejection test can stop early.
        - Rows ``j >= depth[k]`` are padding, ``(0, ..., 0, -1)``, which every
          point satisfies, so a kernel may loop over all D rows without branching.

        Measuring from each cluster's own centroid keeps ``beta`` on the cluster's
        scale; the stored form is measured from the ancestor's centroid (see
        ``internal_columns``). The planes are exact up to rounding: a point on a
        cut, or within rounding of it, can evaluate slightly positive, so checks
        of existing points need a tolerance (``src.sampling.ClusterRegions.contains``).

        Walks ``parent`` pointers, so it works while the forest is still unsorted
        (during training) and after sorting; every ancestor must be in this
        forest, which holds without work stealing. One gather per depth level,
        vectorized over the clusters.
        """
        parent = xp.asarray(parent, dtype=xp.int64)
        morton = xp.asarray(morton, dtype=xp.int64)
        origins = xp.asarray(origins, dtype=xp.float64)
        K, d = int(parent.shape[0]), self.d
        depth = morton & DEPTH_MASK
        D = int(depth.max()) if K else 0
        planes = xp.zeros((K, D, d + 1), dtype=xp.float64)
        planes[:, :, d] = -1.0

        nodes = self.internal_nodes
        W, b, c = nodes["W"], nodes["b"], nodes["centroid"]
        node, path = parent.copy(), morton.copy()
        steps = xp.zeros(K, dtype=xp.int64)
        for j in range(D):
            live = xp.nonzero(node >= 0)[0]
            steps[live] += 1
            p = node[live]
            sign = 1.0 - 2.0 * side_of(path[live]).astype(xp.float64)  # side 1 flips to "<= 0"
            norm = xp.linalg.norm(W[p], axis=1)
            # A zero normal (a degenerate split) keeps the stored predicate b < 0 / > 0.
            scale = sign / xp.where(norm > 0, norm, 1.0)
            planes[live, j, :d] = W[p] * scale[:, None]
            planes[live, j, d] = (b[p] + xp.sum(W[p] * (origins[live] - c[p]), axis=1)) * scale
            path[live], node[live] = nodes["morton"][p], nodes["parent"][p]
        # Each cluster crossed one ancestor per level of its depth and then reached
        # its root; anything else means the tree fields and the forest disagree.
        if K and not (bool(xp.all(node < 0)) and bool(xp.all(steps == depth))):
            raise RuntimeError("Ancestor chains do not match the depths in the Morton IDs.")
        return planes, depth

    def find_leaves(self, X: Array) -> Array:
        """
        For each point of ``X`` (n, d), the row in the trained leaf arrays of the leaf
        it falls in (``leaf_nodes["leaf"]``), or -1 where it reaches an unresolved
        cluster. The tree is the one whose root centroid is nearest (exact kNN, ties
        to the smaller root); each step goes to side 1 where ``w^T (x - c) + b > 0``
        and to side 0 otherwise. One step per level, vectorized over the points still at an
        internal node; a point within rounding of a hyperplane may go either way.
        """
        if not self.is_sorted or self.root_centroids is None:
            raise RuntimeError("find_leaves needs a sorted forest with root_centroids (a trained one).")
        X = xp.asarray(X, dtype=xp.float64)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"Expected points of shape (n, {self.d}); got {tuple(X.shape)}.")
        if self._root_index is None:
            self._root_index = make_knn_index(self.root_centroids)
        queries = _to_device(X) if IS_CUPY_BACKEND else _to_host(X)
        _, nearest = self._root_index.kneighbors(queries, 1)
        root = xp.asarray(nearest[:, 0]).astype(xp.int64)  # root ids are root_centroids rows

        W, b, c = (self.internal_nodes[k] for k in ("W", "b", "centroid"))
        morton = xp.zeros(X.shape[0], dtype=xp.int64)
        node = self.find_rows("internal_nodes", root, morton)  # -1: the root was never split
        active = xp.nonzero(node >= 0)[0]
        while active.shape[0]:
            i = node[active]
            side = (xp.sum(W[i] * (X[active] - c[i]), axis=1) + b[i] > 0).astype(xp.int64)
            morton[active] = child_morton(morton[active], side)
            node[active] = self.find_rows("internal_nodes", root[active], morton[active])
            active = active[node[active] >= 0]  # -1: the child is a leaf node

        rows = self.find_rows("leaf_nodes", root, morton)  # every child slot holds a node
        return xp.where(rows >= 0, self.leaf_nodes["leaf"][xp.maximum(rows, 0)], -1)
