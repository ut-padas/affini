"""
Binary split trees recorded during refinement (``ClusteringConfig.track_trees``).

Every initial cluster roots one binary tree. Each rejected cluster becomes a node
holding the hyperplane ``w^T x + b = 0`` that split it; its children lie on side 0
(``w^T x + b < 0``) and side 1 (``> 0``). Accepted clusters are the leaves and are
stored with the accepted output columns, not here.

Nodes and leaves are identified within their tree by a Morton ID (int64): the path
from the root, one bit per level, left-aligned in bits 62..DEPTH_BITS, with the
depth in the low DEPTH_BITS bits. The root is 0. Sorting by Morton ID is preorder:
a node precedes its descendants, its side-0 subtree precedes its side-1 subtree,
and every subtree is a contiguous range.

``BinaryTreeForest`` collects every tree of one rank in contiguous device arrays,
in arrival order (level by level, trees interleaved) while training; ``sort``
reorders it by (root, Morton ID) so each tree is a contiguous block, which
``tree`` returns as a ``BinaryTree``.
"""

from dataclasses import dataclass

import numpy as np

from src import Array, _to_host, xp
from src.clusters import _grown
from src.numerics import sort_by_block_then_value

DEPTH_BITS = 6
DEPTH_MASK = (1 << DEPTH_BITS) - 1
# Deepest node a Morton ID can address: one path bit per level in bits 62..DEPTH_BITS.
MAX_DEPTH = 63 - DEPTH_BITS


def child_morton(morton: Array, side: Array) -> Array:
    """Morton IDs of the children on ``side`` (0 or 1) of nodes ``morton``; callers keep depth < MAX_DEPTH."""
    depth = morton & DEPTH_MASK
    # Works on host and device arrays alike: only operators, no array-module calls.
    return (morton & ~DEPTH_MASK) | (side.astype(np.int64) << (62 - depth)) | (depth + 1)


@dataclass(frozen=True)
class BinaryTree:
    """
    One tree of a sorted forest, rows in preorder (row 0 is the root's split, if
    the root was split at all). ``parent`` holds rows of this tree (-1 for the
    root's split); the arrays are views into the forest except ``parent``.
    """

    root: int
    W: Array  # (n, d) hyperplane normals
    b: Array  # (n,) hyperplane offsets
    parent: Array  # (n,) int64
    side: Array  # (n,) int8
    morton: Array  # (n,) int64

    @property
    def n_nodes(self) -> int:
        return int(self.b.shape[0])


class BinaryTreeForest:
    """
    The split hyperplanes of every tree on one rank, in contiguous device arrays.

    While unsorted, rows are in arrival order and ``append`` writes into spare
    capacity (grown by at least doubling), so recording a level costs one copy of
    its hyperplanes. ``parent`` always refers to rows of this forest. After
    ``sort`` the forest is frozen: rows are ordered by (root, Morton ID) and
    ``tree`` slices out single trees.
    """

    def __init__(self, d: int, capacity: int = 1024):
        cap = max(1, int(capacity))
        self.d = d
        self._W = xp.empty((cap, d), dtype=xp.float64)
        self._b = xp.empty((cap,), dtype=xp.float64)
        self._parent = xp.empty((cap,), dtype=xp.int64)
        self._side = xp.empty((cap,), dtype=xp.int8)
        self._root = xp.empty((cap,), dtype=xp.int64)
        self._morton = xp.empty((cap,), dtype=xp.int64)
        self.n_nodes = 0
        self.is_sorted = False
        # Set by sort(): the root ids present (ascending) and where each tree starts.
        self.roots: np.ndarray | None = None
        self.tree_starts: np.ndarray | None = None

    @property
    def W(self) -> Array:
        return self._W[: self.n_nodes]

    @property
    def b(self) -> Array:
        return self._b[: self.n_nodes]

    @property
    def parent(self) -> Array:
        return self._parent[: self.n_nodes]

    @property
    def side(self) -> Array:
        return self._side[: self.n_nodes]

    @property
    def root(self) -> Array:
        return self._root[: self.n_nodes]

    @property
    def morton(self) -> Array:
        return self._morton[: self.n_nodes]

    def append(
        self, W: Array, b: Array, parent: Array, side: Array, root: Array, morton: Array
    ) -> Array:
        """
        Append one node per split cluster and return their rows (device int64),
        which become the ``parent`` of the clusters' children. Accepts host or
        device arrays; ``parent`` must be -1 or an existing row.
        """
        if self.is_sorted:
            raise RuntimeError("The forest is sorted (frozen); append before sort().")
        k = int(b.shape[0])
        if W.shape != (k, self.d) or any(a.shape != (k,) for a in (parent, side, root, morton)):
            raise ValueError(
                f"Expected W of shape ({k}, {self.d}) and parent/side/root/morton of "
                f"shape ({k},); got W {W.shape}, parent {parent.shape}, side {side.shape}, "
                f"root {root.shape}, morton {morton.shape}."
            )
        end = self.n_nodes + k
        if end > self._b.shape[0]:
            cap = max(end, 2 * self._b.shape[0])
            for name in ("_W", "_b", "_parent", "_side", "_root", "_morton"):
                setattr(self, name, _grown(getattr(self, name), cap, self.n_nodes))
        rows = slice(self.n_nodes, end)
        self._W[rows] = xp.asarray(W)
        self._b[rows] = xp.asarray(b)
        self._parent[rows] = xp.asarray(parent)
        self._side[rows] = xp.asarray(side)
        self._root[rows] = xp.asarray(root)
        self._morton[rows] = xp.asarray(morton)
        self.n_nodes = end
        return xp.arange(rows.start, end, dtype=xp.int64)

    def sort(self) -> Array:
        """
        Reorder rows by (root, Morton ID), remap ``parent``, and freeze the forest.
        Returns ``new_row`` (n,) with ``new_row[old] = new``, for remapping any other
        references to rows (e.g. the leaves' parents).
        """
        if self.is_sorted:
            raise RuntimeError("The forest is already sorted.")
        n = self.n_nodes
        order = sort_by_block_then_value(self.morton, self.root)
        new_row = xp.empty((n,), dtype=xp.int64)
        new_row[order] = xp.arange(n, dtype=xp.int64)
        old_parent = self.parent[order]
        parent = xp.where(old_parent >= 0, new_row[old_parent], -1) if n else old_parent

        self._W, self._b, self._side, self._root, self._morton = (
            self.W[order], self.b[order], self.side[order], self.root[order], self.morton[order],
        )  # fmt: skip
        self._parent = parent.astype(xp.int64)
        self.roots, self.tree_starts = np.unique(_to_host(self._root), return_index=True)
        self.tree_starts = np.r_[self.tree_starts, n].astype(np.int64)
        self.is_sorted = True
        return new_row

    def tree(self, root: int) -> BinaryTree:
        """The tree grown from initial cluster ``root``; empty if that root was never split."""
        if not self.is_sorted:
            raise RuntimeError("Call sort() before taking single trees.")
        i = int(np.searchsorted(self.roots, root))
        if i < len(self.roots) and self.roots[i] == root:
            start, end = int(self.tree_starts[i]), int(self.tree_starts[i + 1])
        else:
            start = end = 0
        parent = self._parent[start:end]
        return BinaryTree(
            root=int(root),
            W=self._W[start:end],
            b=self._b[start:end],
            parent=xp.where(parent >= 0, parent - start, -1),
            side=self._side[start:end],
            morton=self._morton[start:end],
        )
