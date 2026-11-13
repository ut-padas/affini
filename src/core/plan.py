"""
The contract between the refinement loop (``src.engine.refinement``) and an algorithm plan.

The loop owns everything both algorithms share: queues, batching by device memory, host to
device transfers, work stealing, level statistics and profiling. A plan owns what differs:
its output columns, how much device memory a queued cluster takes, where a batch may end, and
what happens to one batch (``process_batch``). Two plans implement it: ``AfinniPlan``
(``src.engine.afinni_plan``) and ``RegionPlan`` (``src.region.plan``).
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src import Array
from src.core.clusters import ClusterBatch, ClusterQueueBuilder, ColumnBuffer, ParamSpec
from src.core.evaluation import Evaluator


@dataclass
class DeviceBatch:
    """
    One batch of whole queued clusters, moved to the device by the loop.

    ``take()`` hands the point arrays over, so the plan holds the only references and can
    release them early (otherwise they pin the batch through augmentation and splitting,
    the batch's peak memory). ``queue[cut]`` are the batch's clusters in the queue as it
    was when the batch was cut: work stealing inside ``progress()`` gives away only later
    clusters, so a plan reads per-cluster columns (depth, root, constraints) before it
    first calls ``progress()``.
    """

    X: Array | None
    F: Array | None
    G: Array | None
    sizes: Array  # (K,) points per cluster, on the device
    starts: Array  # (K,) first row of each cluster in X, on the device
    queue: ClusterBatch
    cut: slice
    level: int  # this rank's processing round
    index: int  # batch number within the level
    n_batches: int  # batches planned for the level (for logs; stealing can change it)
    progress: Callable[[], None]  # answers steal requests; never gives this batch away

    def take(self) -> tuple[Array, Array, Array | None]:
        """The points, values and gradients; the batch drops its references."""
        arrays = self.X, self.F, self.G
        self.X = self.F = self.G = None
        return arrays


Timer = Callable[[str], AbstractContextManager[Any]]


class Plan(Protocol):
    """
    What the refinement loop asks of an algorithm plan. ``tests/unit/test_plans.py`` checks
    that both plans provide every member.
    """

    label: str  # a short name for profiles, plots and the monitor
    steal_settings: dict[str, Any]  # keyword arguments for make_work_stealer
    batch_floor: int  # fewest points a batch may be capped to when device memory is short
    keeps_fallback_leaves: bool  # clusters stopped above tolerance still become (fallback) leaves

    def leaf_columns(self, model) -> dict[str, ParamSpec]:
        """Columns of one accepted leaf."""

    def split_columns(self) -> dict[str, ParamSpec] | None:
        """Columns of one split record, or None if the plan keeps no tree."""

    def work_sizes(self, sizes: np.ndarray) -> np.ndarray:
        """Device points each queued cluster will occupy (the unit of batching and stealing)."""

    def bytes_per_point(self) -> int:
        """Peak device bytes per point of ``work_sizes``, for the batch-size cap."""

    def constraint_rows(self, queue: ClusterBatch) -> int | None:
        """Constraint rows the children of ``queue`` carry, or None if clusters carry none."""

    def batch_end(self, cumsum: np.ndarray, start: int, index: int, batch_size: int) -> int:
        """One past the last cluster of the batch starting at cluster ``start`` (> start)."""

    def process_batch(self, batch: DeviceBatch, *, components, evaluator: Evaluator, accepted: ColumnBuffer,
                      splits: ColumnBuffer | None, next_level: ClusterQueueBuilder, timer: Timer) -> dict[str, int]:  # fmt: skip
        """
        Fit, accept and split one batch: leaves go to ``accepted``, split records to
        ``splits``, children to ``next_level``. Returns the counts ``accepted``,
        ``rejected``, ``generated``, ``fallback_clusters``, ``fallback_pts`` and ``n_pts``.
        """

    def profile(self) -> dict[str, Any]:
        """Entries this plan adds to the rank's ``profiling_stats`` at the end of refinement."""
