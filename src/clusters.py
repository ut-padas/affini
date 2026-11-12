"""
Host-side containers for queues of clusters.

``ClusterBatch`` is a rank's queue of clusters stored contiguously in host memory;
it is what the refinement loop processes, what work stealing sends between ranks,
and what each level hands to the next. ``ClusterQueueBuilder`` accumulates the
children produced during one level into the next level's ``ClusterBatch``.
``ColumnBuffer`` accumulates rows of named columns (one per accepted cluster, or
per split-tree node on the device).

Queues can outgrow host RAM: every rejected cluster is augmented to at least
``2 * min_pts_per_cluster`` points, so a deep level in high dimension holds tens of
millions of points. Queue buffers therefore check the available host memory before
they grow and raise ``MemoryError`` with advice, rather than letting the operating
system kill the job.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil

from src import Array, _to_host, xp


# Per-cluster positions in the split trees, carried only with track_trees (the
# side of the parent's hyperplane is the last path bit of the Morton ID).
TREE_FIELDS = {"parent": np.int64, "root": np.int64, "morton": np.int64}


@dataclass
class ClusterBatch:
    """
    Clusters stored contiguously in host memory.

    Cluster ``k`` owns rows ``starts[k] : starts[k] + sizes[k]`` of ``X``, ``F``
    and (if present) ``G``. ``depth[k]`` is the cluster's depth in its tree (roots
    are 0); it travels with the cluster when work is stolen, so the depth limit
    does not depend on which rank processes it.

    With ``track_trees`` the clusters also carry their place in the split trees
    (``TREE_FIELDS``, see ``src.trees``); otherwise those fields are None. An
    initial cluster has ``parent`` -1 and ``morton`` 0.
    """

    X: np.ndarray  # (n_pts, d)
    F: np.ndarray  # (n_pts, fd)
    G: np.ndarray | None  # (n_pts, fd, d), or None when gradients are not carried
    sizes: np.ndarray  # (n_clusters,) int64
    depth: np.ndarray | None = None  # (n_clusters,) int64; None means all roots
    parent: np.ndarray | None = None  # (n_clusters,) int64 forest row of the split that made it
    root: np.ndarray | None = None  # (n_clusters,) int64 global index of its initial cluster
    morton: np.ndarray | None = None  # (n_clusters,) int64 Morton ID within its tree
    starts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.starts = np.r_[0, np.cumsum(self.sizes)[:-1]]
        if self.depth is None:
            self.depth = np.zeros(len(self.sizes), dtype=np.int64)
        given = [getattr(self, name) is not None for name in TREE_FIELDS]
        if any(given) and not all(given):
            raise ValueError(f"Give all of the tree fields {list(TREE_FIELDS)} or none.")

    @property
    def has_tree(self) -> bool:
        return self.parent is not None

    def tree_fields(self, index: Any = slice(None)) -> dict[str, np.ndarray]:
        """``TREE_FIELDS`` indexed by cluster (a slice, mask or indices); empty without tree tracking."""
        if not self.has_tree:
            return {}
        return {name: getattr(self, name)[index] for name in TREE_FIELDS}

    @classmethod
    def empty(cls, d: int, fd: int) -> "ClusterBatch":
        return cls(
            np.empty((0, d), dtype=np.float64),
            np.empty((0, fd), dtype=np.float64),
            None,
            np.empty(0, dtype=np.int64),
        )

    @property
    def n_clusters(self) -> int:
        return len(self.sizes)

    @property
    def n_pts(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def fd(self) -> int:
        return int(self.F.shape[1])

    def split(self, cluster_idx: int) -> tuple["ClusterBatch", "ClusterBatch"]:
        """Return (clusters before ``cluster_idx``, clusters from ``cluster_idx`` on)."""
        pt = int(self.starts[cluster_idx])
        G_head = self.G[:pt] if self.G is not None else None
        G_tail = self.G[pt:] if self.G is not None else None
        head, tail = slice(None, cluster_idx), slice(cluster_idx, None)
        return (
            ClusterBatch(self.X[:pt], self.F[:pt], G_head, self.sizes[head], self.depth[head],
                         **self.tree_fields(head)),
            ClusterBatch(self.X[pt:], self.F[pt:], G_tail, self.sizes[tail], self.depth[tail],
                         **self.tree_fields(tail)),
        )  # fmt: skip

    def select(self, keep: np.ndarray) -> "ClusterBatch":
        """The clusters where ``keep`` (n_clusters,) is True, in order."""
        rows = np.repeat(keep, self.sizes)
        G = self.G[rows] if self.G is not None else None
        return ClusterBatch(
            self.X[rows], self.F[rows], G, self.sizes[keep], self.depth[keep],
            **self.tree_fields(keep),
        )  # fmt: skip


# Host RAM kept free for the OS, the GPU driver and other processes: the larger of
# this share of total RAM and HOST_MEMORY_RESERVE_MIN bytes.
HOST_MEMORY_RESERVE_SHARE = 0.1
HOST_MEMORY_RESERVE_MIN = 2 * 2**30


def check_host_memory(n_bytes: int, what: str) -> None:
    """Raise ``MemoryError`` if ``n_bytes`` more would eat into the RAM reserve (see above)."""
    mem = psutil.virtual_memory()
    reserve = max(HOST_MEMORY_RESERVE_SHARE * mem.total, HOST_MEMORY_RESERVE_MIN)
    if n_bytes > mem.available - reserve:
        raise MemoryError(
            f"{what} needs {n_bytes / 2**30:.2f} GiB more host memory, but only "
            f"{mem.available / 2**30:.2f} GiB is available and {reserve / 2**30:.2f} GiB "
            f"is kept free for the system. Rejected clusters are augmented "
            f"to 2 * min_pts_per_cluster points before splitting, so the queue grows with "
            f"the number of rejected clusters: loosen atol/rtol, lower min_pts_per_cluster, "
            f"cap max_levels, or spread the work over more MPI ranks on more nodes."
        )


class ClusterQueueBuilder:
    """
    Growable host-side buffers that collect clusters (one level's children) and
    turn them into a ``ClusterBatch``.

    The number of points only becomes known batch by batch, so the buffers are
    written in place instead of concatenated at the end. Capacity beyond what is
    written is never touched, so a generous initial capacity costs address space,
    not RAM, and avoids the grow-and-copy that would hold two buffers at once.
    """

    def __init__(
        self, d: int, fd: int, has_g: bool, capacity_pts: int, capacity_clusters: int,
        track_trees: bool = False,
    ):
        """``track_trees`` makes every ``append`` require the ``TREE_FIELDS``."""
        self.d, self.fd = d, fd
        cap_pts, cap_clusters = max(1, capacity_pts), max(1, capacity_clusters)
        self._X = np.empty((cap_pts, d), dtype=np.float64)
        self._F = np.empty((cap_pts, fd), dtype=np.float64)
        self._G = np.empty((cap_pts, fd, d), dtype=np.float64) if has_g else None
        self._sizes = np.empty((cap_clusters,), dtype=np.int64)
        self._depth = np.empty((cap_clusters,), dtype=np.int64)
        self._tree = (
            {name: np.empty((cap_clusters,), dtype=dt) for name, dt in TREE_FIELDS.items()}
            if track_trees
            else None
        )
        self.n_pts = 0
        self.n_clusters = 0

    def append(
        self, X: Array, F: Array, G: Array | None, sizes: Array, depth: Array | None = None,
        **tree: Array,
    ) -> None:
        """
        Append clusters given in contiguous order (device or host arrays). ``G`` is
        stored only if the builder was created with ``has_g``; ``depth`` defaults to 0.
        ``tree`` must be exactly the ``TREE_FIELDS`` if the builder tracks trees,
        and empty otherwise.
        """
        if self._G is not None and G is None:
            raise ValueError("This builder carries gradients, but G was not given.")
        expected = set(TREE_FIELDS) if self._tree is not None else set()
        if set(tree) != expected:
            raise ValueError(
                f"This builder expects tree fields {sorted(expected)}; got {sorted(tree)}."
            )
        X_host, F_host = _to_host(X), _to_host(F)
        G_host = _to_host(G) if G is not None and self._G is not None else None
        sizes_host = _to_host(sizes)
        if int(sizes_host.sum()) != X_host.shape[0] or len(F_host) != X_host.shape[0]:
            raise ValueError(
                f"Cluster sizes add up to {int(sizes_host.sum())}, but X has "
                f"{X_host.shape[0]} rows and F has {len(F_host)}."
            )

        pts_end = self.n_pts + int(X_host.shape[0])
        clusters_end = self.n_clusters + int(sizes_host.shape[0])
        # Rows written into reserved capacity become resident too, so check every
        # append (plus the copy if the buffers must grow), not only growth.
        grow = pts_end > self._X.shape[0]
        new_rows = pts_end if grow else pts_end - self.n_pts
        check_host_memory(new_rows * self._row_bytes(), f"The next level's queue ({pts_end:,} points)")
        self._reserve(pts_end, clusters_end)

        self._X[self.n_pts : pts_end] = X_host
        self._F[self.n_pts : pts_end] = F_host
        if G_host is not None:
            self._G[self.n_pts : pts_end] = G_host
        self._sizes[self.n_clusters : clusters_end] = sizes_host
        self._depth[self.n_clusters : clusters_end] = 0 if depth is None else _to_host(depth)
        for name, value in tree.items():
            self._tree[name][self.n_clusters : clusters_end] = _to_host(value)
        self.n_pts, self.n_clusters = pts_end, clusters_end

    def build(self) -> ClusterBatch:
        """Return the collected clusters, dropping any empty ones."""
        if self.n_clusters == 0:
            return ClusterBatch.empty(self.d, self.fd)
        X = self._X[: self.n_pts]
        F = self._F[: self.n_pts]
        G = self._G[: self.n_pts] if self._G is not None else None
        sizes = self._sizes[: self.n_clusters]
        tree = (
            {name: buf[: self.n_clusters] for name, buf in self._tree.items()}
            if self._tree is not None
            else {}
        )
        batch = ClusterBatch(X, F, G, sizes, self._depth[: self.n_clusters], **tree)
        nonempty = sizes > 0
        return batch if np.all(nonempty) else batch.select(nonempty)

    def _row_bytes(self) -> int:
        return 8 * (self.d + self.fd + (self.fd * self.d if self._G is not None else 0))

    def _reserve(self, n_pts: int, n_clusters: int) -> None:
        """Grow the buffers (at least doubling) to hold ``n_pts`` / ``n_clusters``."""
        if n_pts > self._X.shape[0]:
            cap = max(n_pts, 2 * self._X.shape[0])
            self._X = _grown(self._X, cap, self.n_pts)
            self._F = _grown(self._F, cap, self.n_pts)
            if self._G is not None:
                self._G = _grown(self._G, cap, self.n_pts)
        if n_clusters > self._sizes.shape[0]:
            cap = max(n_clusters, 2 * self._sizes.shape[0])
            self._sizes = _grown(self._sizes, cap, self.n_clusters)
            self._depth = _grown(self._depth, cap, self.n_clusters)
            if self._tree is not None:
                self._tree = {
                    name: _grown(buf, cap, self.n_clusters) for name, buf in self._tree.items()
                }


def _grown(buf: Array, capacity: int, n_used: int) -> Array:
    """A copy of ``buf`` with room for ``capacity`` rows, on the same backend (host or device); the first ``n_used`` are kept."""
    empty = np.empty if isinstance(buf, np.ndarray) else xp.empty
    new = empty((capacity, *buf.shape[1:]), dtype=buf.dtype)
    new[:n_used] = buf[:n_used]
    return new


@dataclass(frozen=True)
class ParamSpec:
    """Shape (excluding the leading row axis) and dtype of one stored column."""

    shape: tuple[int, ...]
    dtype: Any = np.float64


class ColumnBuffer:
    """
    Growable columns with one row per record (e.g. per accepted cluster), each
    column one contiguous array: in host memory, or on the device with
    ``on_device`` (e.g. the split trees, which the GPU reads).

    ``append`` must receive exactly the declared columns, each with the declared
    trailing shape, so a column can never be left unwritten or silently broadcast.
    """

    def __init__(self, columns: dict[str, ParamSpec], capacity: int, on_device: bool = False):
        self.columns = dict(columns)
        cap = max(1, int(capacity))
        empty = xp.empty if on_device else np.empty
        self._to_storage = xp.asarray if on_device else _to_host
        self._data = {
            name: empty((cap, *spec.shape), dtype=spec.dtype)
            for name, spec in self.columns.items()
        }
        self.n_rows = 0

    def append(self, rows: dict[str, Array]) -> None:
        """Append rows given as {column: array with a leading row axis} (device or host)."""
        if rows.keys() != self.columns.keys():
            missing = sorted(self.columns.keys() - rows.keys())
            extra = sorted(rows.keys() - self.columns.keys())
            raise KeyError(f"Column mismatch: missing {missing}, unexpected {extra}")
        host = {name: self._to_storage(value) for name, value in rows.items()}
        n_new = {int(v.shape[0]) for v in host.values()}
        if len(n_new) != 1:
            raise ValueError(f"Columns have different numbers of rows: {n_new}")
        for name, value in host.items():
            if tuple(value.shape[1:]) != self.columns[name].shape:
                raise ValueError(
                    f"Column {name!r} has rows of shape {tuple(value.shape[1:])}; "
                    f"declared {self.columns[name].shape}."
                )
        end = self.n_rows + n_new.pop()

        cap = next(iter(self._data.values())).shape[0]
        if end > cap:
            new_cap = max(end, cap * 2)
            self._data = {
                name: _grown(buf, new_cap, self.n_rows)
                for name, buf in self._data.items()
            }
        for name, value in host.items():
            self._data[name][self.n_rows : end] = value
        self.n_rows = end

    def result(self) -> dict[str, Array]:
        """The filled rows of every column (views, on the buffer's backend)."""
        return {name: buf[: self.n_rows] for name, buf in self._data.items()}
