"""
Host-side containers for queues of clusters.

``ClusterBatch`` is a rank's queue of clusters stored contiguously in host memory;
it is what the refinement loop processes, what work stealing sends between ranks,
and what each level hands to the next. ``ClusterQueueBuilder`` accumulates the
children produced during one level into the next level's ``ClusterBatch``.
``ColumnBuffer`` accumulates one row of named columns per accepted cluster.

Queues can outgrow host RAM: every rejected cluster is augmented to at least
``2 * min_pts_per_cluster`` points, so a deep level in high dimension holds tens of
millions of points. Queue buffers therefore check the available host memory before
they grow and raise ``MemoryError`` with advice, rather than letting the operating
system kill the job.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import psutil

from src import Array, _to_host


@dataclass(frozen=True)
class QueueColumn:
    """
    One array carried by every queued cluster.

    ``scope="point"`` columns have one row per point, ``"cluster"`` columns one row per
    cluster. ``trailing(d, fd, h)`` is the shape after the leading axis, where ``h`` is the
    number of constraint rows a region-mode cluster carries. The position of a column in
    ``QUEUE_COLUMNS`` is its MPI tag offset, so columns are only ever appended.
    """

    name: str
    scope: Literal["point", "cluster"]
    dtype: Any
    trailing: Callable[[int, int, int], tuple[int, ...]]


# Every column a queue can carry, in wire order (see src.engine.work_stealing.send_clusters).
# X, F, sizes and depth are always present; G only when gradients are carried per point;
# root, path_code and constraints only in region mode (src.region), where constraints are
# the signed ancestor half-spaces ``a^T u <= b`` stored as rows ``[a | b]`` in the
# normalized coordinates of the domain, zero-padded (``0 <= 0``) to a common ``h``.
QUEUE_COLUMNS: tuple[QueueColumn, ...] = (
    QueueColumn("X", "point", np.float64, lambda d, fd, h: (d,)),
    QueueColumn("F", "point", np.float64, lambda d, fd, h: (fd,)),
    QueueColumn("G", "point", np.float64, lambda d, fd, h: (fd, d)),
    QueueColumn("sizes", "cluster", np.int64, lambda d, fd, h: ()),
    QueueColumn("depth", "cluster", np.int64, lambda d, fd, h: ()),
    QueueColumn("root", "cluster", np.int64, lambda d, fd, h: ()),
    QueueColumn("path_code", "cluster", np.int64, lambda d, fd, h: ()),
    QueueColumn("constraints", "cluster", np.float64, lambda d, fd, h: (h, d + 1)),
)
REGION_COLUMNS = ("root", "path_code", "constraints")


@dataclass
class ClusterBatch:
    """
    Clusters stored contiguously in host memory.

    Cluster ``k`` owns rows ``starts[k] : starts[k] + sizes[k]`` of every point column
    (``X``, ``F`` and, if present, ``G``). ``depth[k]`` is the cluster's depth in its tree
    (roots are 0); it travels with the cluster when work is stolen, so the depth limit
    does not depend on which rank processes it. Region mode adds the node identity
    ``(root, path_code)`` and the ancestor ``constraints``, which also travel, so a stolen
    cluster needs nothing from its victim's history.
    """

    X: np.ndarray  # (n_pts, d)
    F: np.ndarray  # (n_pts, fd)
    G: np.ndarray | None  # (n_pts, fd, d), or None when gradients are not carried
    sizes: np.ndarray  # (n_clusters,) int64
    depth: np.ndarray | None = None  # (n_clusters,) int64; None means all roots
    root: np.ndarray | None = None  # (n_clusters,) int64, region mode
    path_code: np.ndarray | None = None  # (n_clusters,) int64, region mode (root = 1)
    constraints: np.ndarray | None = None  # (n_clusters, h, d + 1), region mode
    starts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.starts = np.r_[0, np.cumsum(self.sizes)[:-1]]
        if self.depth is None:
            self.depth = np.zeros(len(self.sizes), dtype=np.int64)

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

    @property
    def is_region(self) -> bool:
        return self.root is not None

    def columns(self) -> dict[str, np.ndarray]:
        """The present columns, in ``QUEUE_COLUMNS`` order."""
        return {c.name: getattr(self, c.name) for c in QUEUE_COLUMNS if getattr(self, c.name) is not None}

    def _take(self, rows, clusters) -> "ClusterBatch":
        """A batch of ``rows`` of every point column and ``clusters`` of every cluster column."""
        scope = {c.name: c.scope for c in QUEUE_COLUMNS}
        taken = {name: a[rows] if scope[name] == "point" else a[clusters] for name, a in self.columns().items()}
        return ClusterBatch(**{"G": None, **taken})

    def split(self, cluster_idx: int) -> tuple["ClusterBatch", "ClusterBatch"]:
        """Return (clusters before ``cluster_idx``, clusters from ``cluster_idx`` on)."""
        pt = int(self.starts[cluster_idx])
        return (
            self._take(slice(None, pt), slice(None, cluster_idx)),
            self._take(slice(pt, None), slice(cluster_idx, None)),
        )

    def select(self, keep: np.ndarray) -> "ClusterBatch":
        """The clusters where ``keep`` (n_clusters,) is True, in order."""
        return self._take(np.repeat(keep, self.sizes), keep)


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
    ``region_rows`` (region mode) adds the node identity and ``region_rows`` constraint
    rows per cluster; shorter constraint blocks are zero-padded, which adds ``0 <= 0``.
    """

    def __init__(
        self, d: int, fd: int, has_g: bool, capacity_pts: int, capacity_clusters: int,
        *, region_rows: int | None = None,
    ):  # fmt: skip
        self.d, self.fd = d, fd
        self.region_rows = region_rows
        h = region_rows or 0
        present = {"X", "F", "sizes", "depth"} | ({"G"} if has_g else set())
        if region_rows is not None:
            present |= set(REGION_COLUMNS)
        self._columns = [c for c in QUEUE_COLUMNS if c.name in present]
        caps = {"point": max(1, capacity_pts), "cluster": max(1, capacity_clusters)}
        self._buf = {
            c.name: np.empty((caps[c.scope], *c.trailing(d, fd, h)), dtype=c.dtype) for c in self._columns
        }
        self.n_pts = 0
        self.n_clusters = 0

    def append(
        self, X: Array, F: Array, G: Array | None, sizes: Array, depth: Array | None = None,
        *, root: Array | None = None, path_code: Array | None = None, constraints: Array | None = None,
    ) -> None:  # fmt: skip
        """
        Append clusters given in contiguous order (device or host arrays). ``G`` is
        stored only if the builder was created with ``has_g``; ``depth`` defaults to 0.
        A region-mode builder requires ``root``, ``path_code`` and ``constraints``.
        """
        if "G" in self._buf and G is None:
            raise ValueError("This builder carries gradients, but G was not given.")
        region = {"root": root, "path_code": path_code, "constraints": constraints}
        if self.region_rows is not None and any(v is None for v in region.values()):
            raise ValueError("A region-mode builder needs root, path_code and constraints.")
        sizes_host = _to_host(sizes)
        values = {"X": _to_host(X), "F": _to_host(F), "sizes": sizes_host}
        if "G" in self._buf:
            values["G"] = _to_host(G)
        if self.region_rows is not None:
            values.update({k: _to_host(v) for k, v in region.items()})
            c = values["constraints"]
            if c.shape[1] > self.region_rows:
                raise ValueError(
                    f"Clusters carry {c.shape[1]} constraint rows; this builder holds {self.region_rows}."
                )
        n_new = int(values["X"].shape[0])
        if int(sizes_host.sum()) != n_new or len(values["F"]) != n_new:
            raise ValueError(
                f"Cluster sizes add up to {int(sizes_host.sum())}, but X has "
                f"{n_new} rows and F has {len(values['F'])}."
            )

        pts_end = self.n_pts + n_new
        clusters_end = self.n_clusters + int(sizes_host.shape[0])
        # Rows written into reserved capacity become resident too, so check every
        # append (plus the copy if the buffers must grow), not only growth.
        grow = pts_end > self._buf["X"].shape[0]
        new_rows = pts_end if grow else pts_end - self.n_pts
        check_host_memory(new_rows * self._row_bytes(), f"The next level's queue ({pts_end:,} points)")
        self._reserve(pts_end, clusters_end)

        bounds = {"point": (self.n_pts, pts_end), "cluster": (self.n_clusters, clusters_end)}
        for c in self._columns:
            lo, hi = bounds[c.scope]
            if c.name == "depth":
                self._buf["depth"][lo:hi] = 0 if depth is None else _to_host(depth)
            elif c.name == "constraints":
                block = values["constraints"]
                self._buf["constraints"][lo:hi, : block.shape[1]] = block
                self._buf["constraints"][lo:hi, block.shape[1] :] = 0.0
            else:
                self._buf[c.name][lo:hi] = values[c.name]
        self.n_pts, self.n_clusters = pts_end, clusters_end

    def build(self) -> ClusterBatch:
        """Return the collected clusters, dropping any empty ones."""
        if self.n_clusters == 0:
            return ClusterBatch.empty(self.d, self.fd)
        ends = {"point": self.n_pts, "cluster": self.n_clusters}
        batch = ClusterBatch(**{"G": None, **{c.name: self._buf[c.name][: ends[c.scope]] for c in self._columns}})
        nonempty = batch.sizes > 0
        return batch if np.all(nonempty) else batch.select(nonempty)

    def _row_bytes(self) -> int:
        return sum(
            int(np.prod(c.trailing(self.d, self.fd, self.region_rows or 0))) * np.dtype(c.dtype).itemsize
            for c in self._columns
            if c.scope == "point"
        )

    def _reserve(self, n_pts: int, n_clusters: int) -> None:
        """Grow the buffers (at least doubling) to hold ``n_pts`` / ``n_clusters``."""
        need = {"point": (n_pts, self.n_pts), "cluster": (n_clusters, self.n_clusters)}
        for scope, (n, used) in need.items():
            cols = [c.name for c in self._columns if c.scope == scope]
            cap = self._buf[cols[0]].shape[0]
            if n > cap:
                new_cap = max(n, 2 * cap)
                for name in cols:
                    self._buf[name] = _grown(self._buf[name], new_cap, used)


def _grown(buf: np.ndarray, capacity: int, n_used: int) -> np.ndarray:
    """A copy of ``buf`` with room for ``capacity`` rows; the first ``n_used`` are kept."""
    new = np.empty((capacity, *buf.shape[1:]), dtype=buf.dtype)
    new[:n_used] = buf[:n_used]
    return new


@dataclass(frozen=True)
class ParamSpec:
    """Shape (excluding the leading row axis) and dtype of one stored column."""

    shape: tuple[int, ...]
    dtype: Any = np.float64


class ColumnBuffer:
    """
    Growable host-side columns with one row per accepted cluster.

    ``append`` must receive exactly the declared columns, each with the declared
    trailing shape, so a column can never be left unwritten or silently broadcast.
    """

    def __init__(self, columns: dict[str, ParamSpec], capacity: int):
        self.columns = dict(columns)
        cap = max(1, int(capacity))
        self._data = {
            name: np.empty((cap, *spec.shape), dtype=spec.dtype)
            for name, spec in self.columns.items()
        }
        self.n_rows = 0

    def append(self, rows: dict[str, Array]) -> None:
        """Append rows given as {column: array with a leading row axis} (device or host)."""
        if rows.keys() != self.columns.keys():
            missing = sorted(self.columns.keys() - rows.keys())
            extra = sorted(rows.keys() - self.columns.keys())
            raise KeyError(f"Column mismatch: missing {missing}, unexpected {extra}")
        host = {name: _to_host(value) for name, value in rows.items()}
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

    def result(self) -> dict[str, np.ndarray]:
        """The filled rows of every column."""
        return {name: buf[: self.n_rows] for name, buf in self._data.items()}
