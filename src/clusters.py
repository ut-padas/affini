"""
Host-side containers for queues of clusters.

``ClusterBatch`` is a rank's queue of clusters stored contiguously in host memory;
it is what the refinement loop processes, what work stealing sends between ranks,
and what each level hands to the next. ``ClusterQueueBuilder`` accumulates the
children produced during one level into the next level's ``ClusterBatch``.
``ColumnBuffer`` accumulates one row of named columns per accepted cluster.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src import Array, _to_host


@dataclass
class ClusterBatch:
    """
    Clusters stored contiguously in host memory.

    Cluster ``k`` owns rows ``starts[k] : starts[k] + sizes[k]`` of ``X``, ``F``
    and (if present) ``G``.
    """

    X: np.ndarray  # (n_pts, d)
    F: np.ndarray  # (n_pts, fd)
    G: np.ndarray | None  # (n_pts, fd, d), or None when gradients are not carried
    sizes: np.ndarray  # (n_clusters,) int64
    starts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.starts = np.r_[0, np.cumsum(self.sizes)[:-1]]

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
        return (
            ClusterBatch(self.X[:pt], self.F[:pt], G_head, self.sizes[:cluster_idx]),
            ClusterBatch(self.X[pt:], self.F[pt:], G_tail, self.sizes[cluster_idx:]),
        )


class ClusterQueueBuilder:
    """
    Growable host-side buffers that collect clusters (one level's children) and
    turn them into a ``ClusterBatch``.

    The number of points only becomes known batch by batch, so the buffers are
    written in place and grown by doubling, instead of keeping a list of arrays
    and concatenating at the end.
    """

    def __init__(
        self, d: int, fd: int, has_g: bool, capacity_pts: int, capacity_clusters: int
    ):
        self.d, self.fd = d, fd
        cap_pts, cap_clusters = max(1, capacity_pts), max(1, capacity_clusters)
        self._X = np.empty((cap_pts, d), dtype=np.float64)
        self._F = np.empty((cap_pts, fd), dtype=np.float64)
        self._G = np.empty((cap_pts, fd, d), dtype=np.float64) if has_g else None
        self._sizes = np.empty((cap_clusters,), dtype=np.int64)
        self.n_pts = 0
        self.n_clusters = 0

    def append(self, X: Array, F: Array, G: Array | None, sizes: Array) -> None:
        """
        Append clusters given in contiguous order (device or host arrays). ``G`` is
        stored only if the builder was created with ``has_g``.
        """
        X_host, F_host = _to_host(X), _to_host(F)
        G_host = _to_host(G) if G is not None and self._G is not None else None
        sizes_host = _to_host(sizes)

        pts_end = self.n_pts + int(X_host.shape[0])
        clusters_end = self.n_clusters + int(sizes_host.shape[0])
        self._reserve(pts_end, clusters_end)

        self._X[self.n_pts : pts_end] = X_host
        self._F[self.n_pts : pts_end] = F_host
        if G_host is not None:
            self._G[self.n_pts : pts_end] = G_host
        self._sizes[self.n_clusters : clusters_end] = sizes_host
        self.n_pts, self.n_clusters = pts_end, clusters_end

    def build(self) -> ClusterBatch:
        """Return the collected clusters, dropping any empty ones."""
        if self.n_clusters == 0:
            return ClusterBatch.empty(self.d, self.fd)
        X = self._X[: self.n_pts]
        F = self._F[: self.n_pts]
        G = self._G[: self.n_pts] if self._G is not None else None
        sizes = self._sizes[: self.n_clusters]

        nonempty = sizes > 0
        if not np.all(nonempty):
            keep = np.repeat(nonempty, sizes)
            X, F, sizes = X[keep], F[keep], sizes[nonempty]
            if G is not None:
                G = G[keep]
        return ClusterBatch(X, F, G, sizes)

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

    ``append`` must receive exactly the declared columns, so a column can never be
    allocated but left unwritten.
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
