"""One rank's refinement loop: batches of clusters fitted, accepted or split level by level, with work stealing between ranks."""

import itertools
import logging
import time
from dataclasses import dataclass

import numpy as np

from src import IS_CUPY_BACKEND, xp
from src.afinni import AFINNIPlan
from src.config import TrainingConfig
from src.core.clusters import ClusterBatch, ClusterQueueBuilder, ColumnBuffer
from src.core.evaluation import Evaluator
from src.core.plan import DeviceBatch, Plan
from src.engine.work_stealing import make_work_stealer
from src.infra.backend import MPI  # real mpi4py.MPI, or a single-rank stand-in
from src.infra.logging_utils import log_mem_stats
from src.infra.profiling import Profiler

logger = logging.getLogger(__name__)


@dataclass
class _Refinement:
    """What every level of one rank's refinement shares: settings, plan, outputs, stealer and running totals."""

    config: TrainingConfig
    plan: Plan
    components: object
    evaluator: Evaluator
    accepted: ColumnBuffer  # the leaves (S in the companion), one row each
    splits: ColumnBuffer | None
    stealer: object
    profiler: Profiler
    rank: int
    mempool: object
    generated: int = 0
    unresolved_clusters: int = 0  # AFINNI: never accepted; ART: kept as fallback leaves
    unresolved_pts: int = 0

    def load_of(self, queue: ClusterBatch) -> int:
        """A queue's work in the plan's unit (device points), for load reports and stealing."""
        return int(self.plan.work_sizes(queue.sizes).sum())


def refine_rank(
    config: TrainingConfig,
    evaluator: Evaluator,
    clusters: ClusterBatch,
    comm: MPI.Intracomm,
    profiler: Profiler,
    plan: Plan | None = None,
):
    """
    One rank's breadth-first refinement: level by level, the queued clusters (host memory)
    go to the device in batches until none is left. Returns the finalized output buffers
    plus counts.

    This is the loop shared by both algorithm plans (``src.core.plan``): it owns
    batching, device transfers, queues, work stealing and statistics, and each batch
    goes to ``plan.process_batch``. Without ``plan`` it runs ``AFINNIPlan``. A plan's
    split records are returned in the buffers under ``"split:<column>"`` keys.

    Algorithm settings (tolerances, limits, seed, work stealing, ...) are read
    from ``config``; ``config.max_clusters_est`` is the total across all ranks.
    Function evaluations go through ``evaluator``; the returned evaluation count
    covers only the evaluations made here. Load balancing between ranks is
    handled by the work stealer from ``src.engine.work_stealing``. Timings, level
    statistics and events go to ``profiler`` (its ``stats`` dict).
    """
    rank, size = comm.Get_rank(), comm.Get_size()
    d, fd = clusters.d, clusters.fd
    plan = plan if plan is not None else AFINNIPlan(config, d, fd, rank)
    mempool = None
    if IS_CUPY_BACKEND:
        import cupy as cp

        mempool = cp.get_default_memory_pool()

    components = config.components()
    capacity = max(1, config.max_clusters_est // max(1, size))
    split_columns = plan.split_columns()
    profiler.stats["plan"] = plan.label
    profiler.event("run_start", algorithm=config.algorithm, plan=plan.label)
    run = _Refinement(
        config, plan, components, evaluator,
        accepted=ColumnBuffer(plan.leaf_columns(components.local_model), capacity=capacity),
        splits=ColumnBuffer(split_columns, capacity=capacity) if split_columns is not None else None,
        stealer=make_work_stealer(config.use_work_stealing, comm, d, fd, profiler, **plan.steal_settings),
        profiler=profiler, rank=rank, mempool=mempool,
    )  # fmt: skip
    evals_at_start = evaluator.n_evals
    level = 0  # this rank's processing rounds; tree depth is tracked per cluster
    queue = clusters
    run.stealer.exchange_initial_loads(run.load_of(queue))

    while True:
        run.stealer.poll_state()
        run.stealer.set_load(run.load_of(queue) if queue.n_clusters > 0 else 0)
        # Clusters deeper than max_levels were already set aside, so an empty queue is the
        # only reason to stop or steal.
        if queue.n_clusters == 0:
            retired, stolen = run.stealer.idle_step(False)
            if retired:
                break
            if stolen is not None:
                queue = stolen
            continue
        queue = _refine_level(run, queue, level)
        level += 1

    profiler.level = profiler.batch = None
    total_f_evals = evaluator.n_evals - evals_at_start
    # Clusters still queued here were never accepted: refinement stopped at
    # max_levels before they met the tolerance.
    run.unresolved_clusters += queue.n_clusters
    run.unresolved_pts += queue.n_pts
    profiler.stats["num_unresolved_clusters"] = run.unresolved_clusters
    profiler.stats["num_unresolved_pts"] = run.unresolved_pts
    if run.unresolved_clusters > 0 and plan.keeps_fallback_leaves:
        logger.warning(
            f"R{rank:02}: {run.unresolved_clusters:,} ART nodes ({run.unresolved_pts:,} points) ended "
            f"above tolerance (max_levels={config.max_levels} or no valid split); they keep their "
            f"fitted model as a leaf with terminal_reason != 'accepted'."
        )
    elif run.unresolved_clusters > 0:
        logger.warning(
            f"R{rank:02}: {run.unresolved_clusters:,} clusters ({run.unresolved_pts:,} points) "
            f"reached max_levels={config.max_levels} unresolved; no accepted leaf holds "
            f"their points."
        )
    profiler.stats.update(plan.profile())
    buffers = run.accepted.result()
    if run.splits is not None:
        buffers.update({f"split:{k}": v for k, v in run.splits.result().items()})
    return buffers, run.accepted.n_rows, total_f_evals, run.generated, level


def _refine_level(run: _Refinement, queue: ClusterBatch, level: int) -> ClusterBatch:
    """
    Process every queued cluster in batches and return the next level's queue: the
    children of this level's rejected clusters, minus those deeper than ``max_levels``,
    after answering steal requests at the level boundary.
    """
    config, plan, profiler, rank = run.config, run.plan, run.profiler, run.rank
    level_stats = profiler.stats["level_stats"]
    num_clusters = queue.n_clusters
    t_level = time.perf_counter()
    profiler.level, profiler.batch = int(level), None
    if config.log_mem_stats:
        log_mem_stats(f"R{rank:02} Start: L{level:02}")
    level_stats.append({
        "rank": rank, "level": int(level), "clusters_start": int(num_clusters),
        "size_min": int(float(np.min(queue.sizes))), "size_mean": float(np.mean(queue.sizes)),
        "size_median": int(float(np.median(queue.sizes))), "size_max": int(float(np.max(queue.sizes))),
        "t_start": profiler.now(),
    })  # fmt: skip

    level_work = plan.work_sizes(queue.sizes)
    level_total_pts = int(np.sum(level_work))
    cumsum = np.cumsum(level_work)
    largest = int(np.max(level_work)) if len(level_work) else 0
    if largest > config.batch_size:
        logger.warning(
            f"R{rank:02} L{level:02}: a cluster of {largest:,} points exceeds batch_size="
            f"{config.batch_size:,}; its batch may exceed the memory budget"
        )

    # Children of this level's rejected clusters, collected batch by batch.
    next_level = ClusterQueueBuilder(
        queue.d, queue.fd, has_g=config.requires_grad_everywhere,
        # Children rarely exceed twice the parents' points (augmentation targets
        # 2 * count); untouched capacity is only address space.
        capacity_pts=2 * level_total_pts, capacity_clusters=2 * num_clusters,
        constraint_rows=plan.constraint_rows(queue),
    )  # fmt: skip

    num_batches = level_total_pts // config.batch_size + 1
    if profiler.events is not None:
        # Batches end at cluster boundaries, so count the distinct batch ends
        # (an upper bound if work is stolen mid-level).
        ends = np.searchsorted(cumsum, config.batch_size * np.arange(1, num_batches + 1))
        planned = len(np.unique(np.minimum(ends + 1, num_clusters)))
        profiler.event("level_start", n_clusters=int(num_clusters), n_pts=int(queue.n_pts), n_batches=int(planned))
    if config.log_mem_stats:
        log_mem_stats(f"R{rank:02} Start: L{level:02} Batching")

    # The plan chooses where each batch ends; the loop runs until every queued cluster was
    # processed, so oversized clusters are never dropped.
    start, good, fallback, level_generated = 0, 0, 0, 0
    for b in itertools.count():
        if start >= queue.n_clusters:
            break
        logger.info(f"R{rank:02} L{level:02}: Processing Batch {b + 1}/{max(num_batches, b + 1)}")
        profiler.batch = b
        # Mid-level work stealing: give away unprocessed clusters.
        queue = run.stealer.serve_requests(queue, start, level, batch=b)
        run.stealer.poll_state()
        cumsum = cumsum[: queue.n_clusters]
        if start >= queue.n_clusters:
            break
        end = plan.batch_end(cumsum, start, b, config.batch_size)

        def progress(batch_end: int = end, lvl: int = level, bt: int = b) -> None:
            """Answer steal requests from inside long split work (the soft split search); never gives this batch away."""
            nonlocal queue
            queue = run.stealer.serve_requests(queue, batch_end, lvl, batch=bt)
            run.stealer.poll_state()

        stats = _run_batch(run, queue, slice(start, end), next_level, level, b, num_batches, progress)
        good += stats["accepted"]
        fallback += stats["fallback_clusters"]
        level_generated += stats["generated"]
        run.unresolved_clusters += stats["fallback_clusters"]
        run.unresolved_pts += stats["fallback_pts"]
        start = end
    run.generated += level_generated

    with profiler.phase("time_transition"):
        next_queue = next_level.build()
        # Children deeper than max_levels are never refined: they stay unresolved.
        too_deep = next_queue.depth > config.max_levels
        if too_deep.any():
            run.unresolved_clusters += int(too_deep.sum())
            run.unresolved_pts += int(next_queue.sizes[too_deep].sum())
            next_queue = next_queue.select(~too_deep)
        if next_queue.n_clusters == 0:
            logger.info(f"R{rank:02} L{level:02}: No bad clusters to refine, terminating.")
        # Level-boundary work stealing.
        run.stealer.publish_load(run.load_of(next_queue))
        queue = run.stealer.serve_requests(next_queue, 0, level)

    level_stats[-1].update({
        "clusters_level_end": int(len(queue.sizes)), "good_clusters": int(good),
        "fallback_clusters": int(fallback), "num_pts_next_level": int(queue.X.shape[0]),
        "generated_pts": int(level_generated),
    })  # fmt: skip
    if config.log_mem_stats:
        log_mem_stats(f"R{rank:02} End: L{level:02} Transition")
    # No gc.collect() here: the loop creates no reference cycles holding arrays,
    # so refcounting frees them; a full collection cost ~50 ms per level.
    with profiler.phase("time_gc"):
        del next_level, next_queue

    logger.info(
        f"R{rank:02} L{level:02}: Clusters (start): {num_clusters:,}, Clusters (end): {len(queue.sizes):,}, "
        f"Good Clusters: {good:,}, Num Pts (next level): {queue.X.shape[0]:,}"
    )
    if len(queue.sizes) > 0:
        logger.info(
            f"R{rank:02} L{level:02}: Cluster Sizes - min: {int(queue.sizes.min())}, "
            f"mean: {queue.sizes.mean():.1f}, median: {int(np.median(queue.sizes))}, "
            f"max: {int(queue.sizes.max())}"
        )
    level_runtime = time.perf_counter() - t_level
    logger.info(f"Runtime - R{rank:02} L{level:02}: {level_runtime:.3f}s")
    level_stats[-1].update({"runtime_sec": level_runtime, "t_end": profiler.now()})
    profiler.event("level_end", **level_stats[-1])
    return queue


def _run_batch(run: _Refinement, queue: ClusterBatch, cut: slice, next_level: ClusterQueueBuilder, level: int,
               index: int, n_batches: int, progress) -> dict[str, int]:  # fmt: skip
    """Move ``queue[cut]`` to the device, hand it to the plan, and report the batch."""
    with run.profiler.phase("time_slicing"):
        sizes_cpu = queue.sizes[cut]
        starts_cpu = queue.starts[cut]
        lo, hi = int(starts_cpu[0]), int(starts_cpu[-1] + sizes_cpu[-1])
        batch = DeviceBatch(
            X=xp.asarray(queue.X[lo:hi]), F=xp.asarray(queue.F[lo:hi]),
            G=xp.asarray(queue.G[lo:hi]) if queue.G is not None else None,
            sizes=xp.asarray(sizes_cpu), starts=xp.asarray(starts_cpu - lo),
            queue=queue, cut=cut, level=level, index=index, n_batches=n_batches, progress=progress,
        )  # fmt: skip
    # The plan times its own stages under the shared phase keys.
    stats = run.plan.process_batch(
        batch, components=run.components, evaluator=run.evaluator, accepted=run.accepted, splits=run.splits,
        next_level=next_level, timer=run.profiler.phase,
    )  # fmt: skip
    run.profiler.event("batch", batch=index, accepted=stats["accepted"], rejected=stats["rejected"], n_pts=stats["n_pts"])
    del batch
    if run.mempool is not None:
        run.mempool.free_all_blocks()
    return stats
