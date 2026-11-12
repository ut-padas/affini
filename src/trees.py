"""
Binary split trees recorded during refinement (``ClusteringConfig.track_trees``).

Every initial cluster roots one binary tree. Each rejected cluster becomes an
internal node holding the hyperplane ``w^T x + b = 0`` that split it; its
children lie on side 0 (``w^T x + b < 0``) and side 1 (``> 0``). Every other
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
returns as a ``BinaryTree``.
"""

from dataclasses import dataclass

import numpy as np

from src import Array, _to_host, xp
from src.clusters import ColumnBuffer, ParamSpec
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
    """Columns of ``internal_nodes``: the hyperplane, then the node's place in its tree."""
    return {
        "W": ParamSpec((d,)),
        "b": ParamSpec(()),
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
    Every split tree of one rank, as two device tables (``internal_columns``,
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

    def add_splits(self, W: Array, b: Array, parent: Array, root: Array, morton: Array) -> Array:
        """
        Record one internal node per split cluster and return their rows (device
        int64), which become the ``parent`` of the clusters' children.
        """
        self._check_unsorted()
        start = self._internal.n_rows
        self._internal.append({"W": W, "b": b, "parent": parent, "root": root, "morton": morton})
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
