"""
The adaptive clustering engine: initial clustering on rank 0 (``initial_clustering``,
also callable on its own), scatter to MPI ranks,
the batched fit / accept / augment / split refinement loop on each rank (with work
stealing between ranks), and the gather of accepted leaves on rank 0.

Scientific choices live in the components of ``ClusteringConfig`` (``src.config``);
this module owns batching, device transfers, buffers, MPI and profiling, so a new
component never has to change it.
"""

import dataclasses
import itertools
import logging
import os
import time
from typing import Any, Optional

import numpy as np

from src import (
    IS_CUPY_BACKEND,
    Array,
    _to_host,
    xp,
)
from src.adaptive_clustering_utils import (
    combine_generated_data,
    save_generated_data,
    uniform_grid_cluster_labels,
)
from src.backend import (  # MPI: real mpi4py.MPI, or a single-rank stand-in
    MPI,
)
from src.clusters import ClusterBatch, ClusterQueueBuilder, ColumnBuffer
from src.config import ClusteringConfig
from src.evaluation import Evaluator
from src.local_models import ClusterFitData, deploy_leaves, output_columns
from src.logging_utils import log_mem_stats
from src.numerics import convert_to_2d_array
from src.profiling import ORACLE_TIME_KEY, Profiler, new_profiling_stats
from src.region import TERMINAL_REASONS, Domain, RegionPlan, Roots, RoutingTable
from src.sampling import augment_clusters, augmentation_targets
from src.splitting import SplitData
from src.utils import repeats
from src.work_stealing import (
    make_work_stealer,
    recv_clusters,
    send_clusters,
)

# Initial scatter of clusters from rank 0 (payload tags SCATTER_TAG + 1..4); kept
# well away from the work-stealing tags.
SCATTER_TAG = 1000
LOCAL_RESULT_TAG = 2000
# One scope (list of profilers) per running run_collective operation, innermost last, so
# a nested operation (e.g. a predict() inside an oracle) only cleans up its own. A stack
# per process: concurrent operations from several threads are not supported.
_PROFILER_SCOPES: list[list] = []  # each rank's merged leaves and profile, sent to rank 0

logger = logging.getLogger(__name__)
TINY_EPS = 1e-12

_CHECK_CHUNK = 1 << 22  # rows per gather in the split check (bounds its temporary memory)


def _check_split(splitter, order, child_sizes, counts, labels) -> None:
    """
    A split must permute every row exactly once, keep each row inside its parent
    cluster (``labels`` is sorted, so ``labels[order]`` must equal it), and cut each
    cluster into two children whose sizes add up to the parent's.
    """
    n_rows = int(labels.shape[0])
    if order.shape != (n_rows,) or child_sizes.shape != (2 * len(counts),):
        raise ValueError(
            f"Split strategy {splitter.name!r} returned order {tuple(order.shape)} and "
            f"child sizes {tuple(child_sizes.shape)}; expected ({n_rows},) and "
            f"({2 * len(counts)},)."
        )
    if not bool(
        xp.all(child_sizes >= 0) & xp.all(child_sizes[0::2] + child_sizes[1::2] == counts)
    ):
        raise ValueError(
            f"Split strategy {splitter.name!r} returned child sizes that are negative "
            f"or do not add up to their cluster's size."
        )
    # ``order`` has n_rows entries, so it is a permutation iff it marks every row;
    # a boolean mark array (1 byte per row) avoids bincount's large temporaries.
    marked = xp.zeros(n_rows, dtype=xp.bool_)
    marked[order] = True
    same_cluster = all(
        bool(xp.all(labels[order[s : s + _CHECK_CHUNK]] == labels[s : s + _CHECK_CHUNK]))
        for s in range(0, n_rows, _CHECK_CHUNK)
    )
    if not (bool(xp.all(marked)) and same_cluster):
        raise ValueError(
            f"Split strategy {splitter.name!r} returned an order that drops, duplicates "
            f"or moves rows between clusters."
        )


def plan_label(config: ClusteringConfig) -> str:
    """A short name for the algorithm plan of ``config``, for profiles, plots and the monitor."""
    if config.algorithm == "region":
        return f"AFINNI region mode (ART; {config.region.split_search} splits, {config.region.roots} roots)"
    return "AFINNI"


def _threads_for_rank0(node_ranks: int) -> int:
    """Threads for work done by rank 0 alone: its own share (``OMP_NUM_THREADS``, default 1 under MPI) times the ranks on its node."""
    env = os.environ.get("OMP_NUM_THREADS", "")
    per_rank = int(env) if env.isdigit() else (1 if node_ranks > 1 else os.cpu_count() or 1)
    return max(1, per_rank * node_ranks)


def initial_clustering(X: Array, config: ClusteringConfig, *, threads: int | None = None) -> np.ndarray:
    """
    The (N,) int64 host labels of the initial clustering that training with
    ``config`` starts from (``initial_clustering_method``, ``num_initial_clusters``,
    ``seed``); deterministic for a given seed and backend. ``threads`` (CPU k-means
    only) defaults to ``OMP_NUM_THREADS``, else every core.
    """
    method = config.initial_clustering_method
    if method == "kmeans":
        # Custom k-means (src/kmeans.py). The CPU path keeps the previous two Lloyd
        # iterations; training passes the cores of rank 0's node, whose other ranks wait.
        from src.kmeans import kmeans_labels

        if IS_CUPY_BACKEND:
            labels = kmeans_labels(xp.asarray(X, dtype=xp.float64), config.num_initial_clusters, config.seed)
        else:
            labels = kmeans_labels(
                np.asarray(_to_host(X), dtype=np.float64), config.num_initial_clusters, config.seed,
                max_iter=2, threads=_threads_for_rank0(1) if threads is None else threads,
            )  # fmt: skip
    elif method == "knndbscan":
        from knndbscan import knn_dbscan

        labels = knn_dbscan(
            xp.asarray(X, dtype=xp.float64), eps=0.5, min_samples=config.min_pts_per_cluster,
            device="gpu" if IS_CUPY_BACKEND else "cpu",
        )  # fmt: skip
    elif method == "uniform_grid":
        labels, _ = uniform_grid_cluster_labels(_to_host(X), config.num_initial_clusters)
    else:
        raise ValueError(f"Invalid initial clustering method: {method!r}")
    return np.asarray(_to_host(labels)).astype(np.int64)


def _run_local_refinement(
    config: ClusteringConfig,
    evaluator: Evaluator,
    clusters: ClusterBatch,
    comm: MPI.Intracomm,
    profiler: Profiler,
    plan: Optional["RegionPlan"] = None,
):
    """
    Core local BFS refinement loop for GPU adaptive clustering, run independently
    on each MPI rank. Takes this rank's pre-sorted clusters (host memory) and
    returns finalized output buffers plus counts.

    This is the executor shared by both algorithm plans: it owns batching, device
    transfers, queues, work stealing and statistics. Each batch is processed by the
    AFINNI plan (the inline body below, in the original order and arithmetic, so its
    results are bit-identical) or, with ``plan``, by
    ``RegionPlan.process_batch``, whose split records are returned in the buffers under
    ``"node:<column>"`` keys.

    Algorithm settings (tolerances, limits, seed, work stealing, ...) are read
    from ``config``; ``config.max_clusters_est`` is the total across all ranks.
    Function evaluations go through ``evaluator``; the returned evaluation count
    covers only the evaluations made here. Load balancing between ranks is
    handled by the work stealer from ``src.work_stealing``. Timings, level
    statistics and events go to ``profiler`` (its ``stats`` dict).
    """
    rank, size = comm.Get_rank(), comm.Get_size()
    profiling_stats = profiler.stats
    timer = profiler.phase
    d, fd = clusters.d, clusters.fd
    queue = clusters

    mempool = None
    if IS_CUPY_BACKEND:
        import cupy as cp

        mempool = cp.get_default_memory_pool()

    # The components (local model, split, sampler, acceptance, trust region), and
    # host-side output columns for the accepted clusters (the shared columns plus
    # the model's parameters).
    components = config.components()
    model = components.local_model
    splitter = components.split_strategy
    acceptance = components.acceptance
    trust_region = components.trust_region
    capacity = max(1, config.max_clusters_est // max(1, size))
    accepted = ColumnBuffer(
        output_columns(model, d, fd) if plan is None else plan.leaf_columns(model), capacity=capacity
    )
    splits = ColumnBuffer(plan.split_columns(), capacity=capacity) if plan is not None else None
    if plan is None:
        def load_of(q: ClusterBatch) -> int:
            return q.n_pts

        def work_sizes(sizes: np.ndarray) -> np.ndarray:
            return sizes
    else:  # a region node's device footprint is its topped-up size, not its queued size
        work_sizes = plan.work_sizes

        def load_of(q: ClusterBatch) -> int:
            return int(work_sizes(q.sizes).sum())
    num_generated_pts = 0
    evals_at_start = evaluator.n_evals
    level = 0  # this rank's processing rounds; tree depth is tracked per cluster
    unresolved_clusters = unresolved_pts = 0

    sampler = components.sampler
    profiling_stats["plan"] = plan_label(config)
    profiler.event("run_start", algorithm=config.algorithm, plan=profiling_stats["plan"])
    stealer = make_work_stealer(
        config.use_work_stealing, comm, d, fd, profiler, **(plan.steal_settings if plan is not None else {})
    )
    stealer.exchange_initial_loads(load_of(queue))

    while True:
        num_clusters = queue.n_clusters
        stealer.poll_state()
        stealer.set_load(load_of(queue) if num_clusters > 0 else 0)

        # --- Phase 3: Idle / Stealing Phase ---
        # (clusters deeper than max_levels were already set aside, so an empty
        # queue is the only reason to stop or steal)
        if num_clusters == 0:
            retired, stolen = stealer.idle_step(False)
            if retired:
                break
            if stolen is not None:
                queue = stolen
            # Loop back around to check termination and steal opportunities again.
            continue

        # --- Phase 1: The Compute Phase (Mid-Level) ---
        _t_level = time.perf_counter()
        profiler.level, profiler.batch = int(level), None
        if config.log_mem_stats:
            log_mem_stats(f"R{rank:02} Start: L{level:02}")

        profiling_stats["level_stats"].append(
            {
                "rank": rank,
                "level": int(level),
                "clusters_start": int(num_clusters),
                "size_min": int(float(np.min(queue.sizes))),
                "size_mean": float(np.mean(queue.sizes)),
                "size_median": int(float(np.median(queue.sizes))),
                "size_max": int(float(np.max(queue.sizes))),
                "t_start": profiler.now(),
            }
        )
        level_generated = 0

        level_work = work_sizes(queue.sizes)
        level_total_pts = int(np.sum(level_work))
        aug_cumsum_cpu = np.cumsum(level_work)
        largest = int(np.max(level_work)) if len(level_work) else 0
        if largest > config.batch_size:
            logger.warning(
                f"R{rank:02} L{level:02}: a cluster of {largest:,} points exceeds batch_size="
                f"{config.batch_size:,}; it is processed alone and may exceed the memory budget"
            )

        # Children of this level's rejected clusters, collected batch by batch.
        next_level = ClusterQueueBuilder(
            d,
            fd,
            has_g=config.requires_grad_everywhere,
            # Children rarely exceed twice the parents' points (augmentation targets
            # 2 * count); untouched capacity is only address space.
            capacity_pts=2 * level_total_pts,
            capacity_clusters=2 * num_clusters,
            # Children carry one more constraint row than the deepest parent.
            region_rows=None if plan is None else int(queue.depth.max()) + 1,
        )

        num_batches = level_total_pts // config.batch_size + 1
        start_cluster_idx = 0
        K_good = 0
        K_fallback = 0  # region mode: nodes that kept a fallback leaf this level
        if profiler.events is not None:
            # Batches end at cluster boundaries, so count the distinct batch ends
            # (an upper bound if work is stolen mid-level).
            ends = np.searchsorted(aug_cumsum_cpu, config.batch_size * np.arange(1, num_batches + 1))
            planned = len(np.unique(np.minimum(ends + 1, num_clusters)))
            profiler.event(
                "level_start", n_clusters=int(num_clusters), n_pts=int(queue.n_pts),
                n_batches=int(planned),
            )  # fmt: skip

        if config.log_mem_stats:
            log_mem_stats(f"R{rank:02} Start: L{level:02} Batching")

        # Batch ends follow the grid (b + 1) * batch_size of cumulative points. A cluster
        # larger than a whole batch is processed on its own and the grid re-anchored after
        # it (batch_offset); the loop runs until every queued cluster was processed, so
        # oversized clusters are never dropped (they once silently were).
        batch_offset = 0
        for b in itertools.count():
            if start_cluster_idx >= queue.n_clusters:
                break
            logger.info(
                f"R{rank:02} L{level:02}: Processing Batch {b + 1}/{max(num_batches, b + 1)}"
            )
            profiler.batch = b

            # --- Mid-Level Work Stealing: give away unprocessed clusters ---
            queue = stealer.serve_requests(queue, start_cluster_idx, level, batch=b)
            stealer.poll_state()
            num_clusters = queue.n_clusters
            aug_cumsum_cpu = aug_cumsum_cpu[:num_clusters]

            target_aug_count = (b + 1) * config.batch_size + batch_offset
            end_cluster_idx = int(np.searchsorted(aug_cumsum_cpu, target_aug_count))
            end_cluster_idx = min(end_cluster_idx + 1, num_clusters)
            if start_cluster_idx >= num_clusters:
                break
            if end_cluster_idx <= start_cluster_idx:
                batch_offset = int(aug_cumsum_cpu[start_cluster_idx]) - (b + 1) * config.batch_size
                end_cluster_idx = start_cluster_idx + 1

            with timer("time_slicing"):
                sizes_cpu = queue.sizes[start_cluster_idx:end_cluster_idx]
                depth_cpu = queue.depth[start_cluster_idx:end_cluster_idx]
                starts_local_cpu = queue.starts[start_cluster_idx:end_cluster_idx]
                slice_start = int(starts_local_cpu[0])
                slice_end = int(starts_local_cpu[-1] + sizes_cpu[-1])

                # --- MOVE ONLY THIS BATCH TO THE GPU ---
                pts = xp.asarray(queue.X[slice_start:slice_end])
                f_vals = xp.asarray(queue.F[slice_start:slice_end])
                g_vals = (
                    xp.asarray(queue.G[slice_start:slice_end])
                    if queue.G is not None
                    else None
                )

                # Move structural arrays to GPU for the math logic
                sizes = xp.asarray(sizes_cpu)
                starts_local_shifted = xp.asarray(starts_local_cpu - slice_start)

            if plan is not None:
                cut = slice(start_cluster_idx, end_cluster_idx)
                def progress(batch_end: int = end_cluster_idx, lvl: int = level, bt: int = b) -> None:
                    """Answer steal requests from inside long split work (B4); never gives this batch away."""
                    nonlocal queue
                    queue = stealer.serve_requests(queue, batch_end, lvl, batch=bt)
                    stealer.poll_state()

                # The plan times its own stages under the shared phase keys.
                batch_stats = plan.process_batch(
                    X=pts, F=f_vals, G=g_vals, sizes=sizes, depth=xp.asarray(depth_cpu),
                    root=xp.asarray(queue.root[cut]), path_code=xp.asarray(queue.path_code[cut]),
                    constraints=xp.asarray(queue.constraints[cut]), components=components,
                    evaluator=evaluator, accepted=accepted, splits=splits, next_level=next_level,
                    progress=progress, timer=timer,
                )  # fmt: skip
                K_good += batch_stats["accepted"]
                num_generated_pts += batch_stats["generated"]
                level_generated += batch_stats["generated"]
                unresolved_clusters += batch_stats["fallback_clusters"]
                K_fallback += batch_stats["fallback_clusters"]
                unresolved_pts += batch_stats["fallback_pts"]
                profiler.event(
                    "batch", batch=b, accepted=batch_stats["accepted"], rejected=batch_stats["rejected"],
                    n_pts=batch_stats["n_pts"],
                )  # fmt: skip
                start_cluster_idx = end_cluster_idx
                del pts, f_vals, g_vals
                if mempool is not None:
                    mempool.free_all_blocks()
                continue

            aug_sizes = sizes
            K_batch = len(sizes)

            X_batch = pts
            F_batch = f_vals
            G_batch = g_vals

            aug_starts = starts_local_shifted
            aug_ids = repeats(xp.arange(K_batch), sizes)

            with timer("time_error_calc"):
                mus = xp.add.reduceat(X_batch, aug_starts) / aug_sizes[:, None]

                f_mus, g_mus = evaluator(
                    mus, need_grad=config.requires_grad_centroids
                )

                # Fit the local model on every cluster, evaluate it at the
                # cluster's points, and accept the clusters within tolerance.
                fit_data = ClusterFitData(
                    X=X_batch,
                    F=F_batch,
                    G=G_batch,
                    labels=aug_ids,
                    starts=aug_starts,
                    sizes=aug_sizes,
                    centroids=mus,
                    f_c=f_mus,
                    g_c=g_mus,
                )
                surrogate_params = model.fit(fit_data)
                F_hat = model.evaluate(
                    {**surrogate_params, "centroids": mus, "f": f_mus},
                    X_batch,
                    aug_ids,
                )
                is_good, E_abs, E_rel = acceptance(F_batch, F_hat, aug_starts, aug_sizes)
                del F_hat, fit_data

            if config.log_mem_stats:
                log_mem_stats(
                    f" R{rank:02} L{level:02} B{b:02}/{num_batches:02}: After Error Metrics"
                )

            # 5. Handle Accepted ("Good") Clusters
            with timer("time_filter"):
                good_idx_local = xp.nonzero(is_good)[0]
                K_good_batch = int(good_idx_local.shape[0])
                K_good += K_good_batch

            if K_good_batch > 0:
                with timer("time_good_Q"):
                    leaves = deploy_leaves(
                        model, trust_region, acceptance,
                        ClusterFitData(
                            X=X_batch, F=F_batch, G=G_batch, labels=aug_ids, starts=aug_starts,
                            sizes=aug_sizes, centroids=mus, f_c=f_mus, g_c=g_mus,
                        ),
                        good_idx_local, surrogate_params, E_abs, E_rel,
                    )  # fmt: skip
                    if int(leaves.failed.shape[0]) > 0:
                        # Clusters the finalized model fails are split instead.
                        is_good[leaves.failed] = False
                        K_good -= int(leaves.failed.shape[0])
                        K_good_batch -= int(leaves.failed.shape[0])

                with timer("time_good_d2h"):
                    accepted.append(leaves.rows)
                    del leaves

            # 6. Handle Rejected ("Bad") Clusters -> Split
            with timer("time_filter"):
                bad_idx_local = xp.nonzero(~is_good)[0]
                n_bad = int(bad_idx_local.shape[0])
                n_batch_pts = int(X_batch.shape[0])

            if n_bad > 0:
                with timer("time_augment"):
                    bad_mask = (~is_good)[aug_ids]
                    X_bad = X_batch[bad_mask]
                    F_bad = F_batch[bad_mask]
                    G_bad = G_batch[bad_mask] if G_batch is not None else None
                    # The whole batch is not needed past this point: release it before
                    # augmentation and splitting, which hold the batch's peak memory.
                    X_batch = F_batch = G_batch = bad_mask = None
                    bad_counts = aug_sizes[bad_idx_local]

                    # --- Augment small rejected clusters before splitting ---
                    target_bad_counts = augmentation_targets(
                        bad_counts, config.min_pts_per_cluster
                    )

                    total_target_pts = int(xp.sum(target_bad_counts))
                    total_bad_input_pts = int(xp.sum(bad_counts))
                    planned_synth_pts = total_target_pts - total_bad_input_pts
                    bad_count_min = int(xp.min(bad_counts))
                    bad_count_mean = float(xp.mean(bad_counts))
                    bad_count_max = int(xp.max(bad_counts))
                    expansion = total_target_pts / max(total_bad_input_pts, 1)

                    logger.info(
                        f"R{rank:02} L{level:02} B{b + 1}: rejected-cluster augmentation plan | "
                        f"bad_clusters={n_bad:,} "
                        f"input_pts={total_bad_input_pts:,} "
                        f"bad_count[min/mean/max]={bad_count_min:,}/{bad_count_mean:,.1f}/{bad_count_max:,} "
                        f"target_pts={total_target_pts:,} "
                        f"synthetic_pts={planned_synth_pts:,} "
                        f"expansion={expansion:.2f}x"
                    )

                    rng = xp.random.default_rng(config.seed + level + b + rank * 1000)
                    aug = augment_clusters(
                        X_bad,
                        F_bad,
                        G_bad,
                        bad_counts,
                        target_bad_counts,
                        sampler,
                        evaluator,
                        rng,
                        centroids=mus[bad_idx_local],
                    )
                    X_bad_aug, F_bad_aug, G_bad_aug = aug.X, aug.F, aug.G
                    aug_dense_labels_bad = aug.labels
                    if aug.X_new is not None:
                        num_generated_pts += int(aug.X_new.shape[0])
                        level_generated += int(aug.X_new.shape[0])
                        if config.save_all_gen_training_data:
                            save_generated_data(
                                aug.X_new,
                                config.generated_data_name(d),
                            )
                    # The augmented arrays hold copies of the rejected rows.
                    del aug, X_bad, F_bad, G_bad
                    if mempool is not None:
                        # Hand cached blocks back before the split, the batch's largest
                        # allocation burst (a few cudaFree calls per batch).
                        mempool.free_all_blocks()

                with timer("time_bad_split"):
                    # Cut each rejected cluster in two along its split direction:
                    # its points are sorted by their projection onto that direction,
                    # and child_sizes gives the size of each half.
                    sort_order, child_sizes = splitter.split(
                        SplitData(
                            X=X_bad_aug,
                            F=F_bad_aug,
                            G=G_bad_aug,
                            labels=aug_dense_labels_bad,
                            counts=target_bad_counts,
                            centroids=mus[bad_idx_local],
                            f_c=f_mus[bad_idx_local],
                            g_c=g_mus[bad_idx_local] if g_mus is not None else None,
                            hessian_fn=evaluator.hessian_fn,
                        )
                    )
                    _check_split(
                        splitter, sort_order, child_sizes, target_bad_counts,
                        aug_dense_labels_bad,
                    )  # fmt: skip

                with timer("time_bad_d2h"):
                    # Children: each cluster cut along its split direction.
                    next_level.append(
                        X_bad_aug[sort_order],
                        F_bad_aug[sort_order],
                        G_bad_aug[sort_order]
                        if config.requires_grad_everywhere and G_bad_aug is not None
                        else None,
                        child_sizes,
                        depth=np.repeat(depth_cpu[_to_host(bad_idx_local)] + 1, 2),
                    )

                with timer("time_gc"):
                    del (
                        X_bad_aug,
                        F_bad_aug,
                        G_bad_aug,
                        sort_order,
                    )
                    if mempool is not None:
                        mempool.free_all_blocks()

            profiler.event(
                "batch", batch=b, accepted=K_good_batch, rejected=n_bad,
                n_pts=n_batch_pts,
            )  # fmt: skip
            start_cluster_idx = end_cluster_idx
            with timer("time_gc"):
                del X_batch, F_batch, G_batch
                if mempool is not None:
                    mempool.free_all_blocks()

        # --- Phase 2: The Level Boundary (Transition & Steal Response) ---
        with timer("time_transition"):
            next_queue = next_level.build()
            # Children deeper than max_levels are never refined: they stay unresolved.
            too_deep = next_queue.depth > config.max_levels
            if too_deep.any():
                unresolved_clusters += int(too_deep.sum())
                unresolved_pts += int(next_queue.sizes[too_deep].sum())
                next_queue = next_queue.select(~too_deep)
            if next_queue.n_clusters == 0:
                logger.info(
                    f"R{rank:02} L{level:02}: No bad clusters to refine, terminating."
                )

            # --- Level-Boundary Work Stealing ---
            stealer.publish_load(load_of(next_queue))
            queue = stealer.serve_requests(next_queue, 0, level)

        if profiling_stats["level_stats"]:
            profiling_stats["level_stats"][-1].update(
                {
                    "clusters_level_end": int(len(queue.sizes)),
                    "good_clusters": int(K_good),
                    "fallback_clusters": int(K_fallback),
                    "num_pts_next_level": int(queue.X.shape[0]),
                    "generated_pts": int(level_generated),
                }
            )

        if config.log_mem_stats:
            log_mem_stats(f"R{rank:02} End: L{level:02} Transition")

        # No gc.collect() here: the loop creates no reference cycles holding arrays,
        # so refcounting frees them; a full collection cost ~50 ms per level.
        with timer("time_gc"):
            del (
                next_level,
                next_queue,
            )

        logger.info(
            f"R{rank:02} L{level:02}: Clusters (start): {num_clusters:,}, "
            f"Clusters (end): {len(queue.sizes):,}, "
            f"Good Clusters: {K_good:,}, "
            f"Num Pts (next level): {queue.X.shape[0]:,}"
        )
        if len(queue.sizes) > 0:
            logger.info(
                f"R{rank:02} L{level:02}: Cluster Sizes - min: {int(queue.sizes.min())}, "
                f"mean: {queue.sizes.mean():.1f}, median: {int(np.median(queue.sizes))}, "
                f"max: {int(queue.sizes.max())}"
            )
        level_runtime = time.perf_counter() - _t_level
        logger.info(f"Runtime - R{rank:02} L{level:02}: {level_runtime:.3f}s")

        if profiling_stats["level_stats"]:
            profiling_stats["level_stats"][-1].update(
                {"runtime_sec": level_runtime, "t_end": profiler.now()}
            )
            profiler.event("level_end", **profiling_stats["level_stats"][-1])
        level += 1

    profiler.level = profiler.batch = None
    total_f_evals = evaluator.n_evals - evals_at_start
    # Clusters still queued here were never accepted: refinement stopped at
    # max_levels before they met the tolerance.
    unresolved_clusters += queue.n_clusters
    unresolved_pts += queue.n_pts
    profiling_stats["num_unresolved_clusters"] = unresolved_clusters
    profiling_stats["num_unresolved_pts"] = unresolved_pts
    if unresolved_clusters > 0 and plan is not None:
        logger.warning(
            f"R{rank:02}: {unresolved_clusters:,} region nodes ({unresolved_pts:,} points) ended "
            f"above tolerance (max_levels={config.max_levels} or no valid split); they keep their "
            f"fitted model as a leaf with terminal_reason != 'accepted'."
        )
    elif unresolved_clusters > 0:
        logger.warning(
            f"R{rank:02}: {unresolved_clusters:,} clusters ({unresolved_pts:,} points) "
            f"reached max_levels={config.max_levels} unresolved; their region has no "
            f"accepted leaves."
        )
    if plan is not None:
        profiling_stats["region"] = dict(plan.counters)
    buffers = accepted.result()
    if splits is not None:
        buffers.update({f"node:{k}": v for k, v in splits.result().items()})
    return buffers, accepted.n_rows, total_f_evals, num_generated_pts, level


# ---------------------------------------------------------------------------
# Public entry point (MPI-aware)
# ---------------------------------------------------------------------------
def adaptive_clustering(
    X: Array,
    config: ClusteringConfig,
    evaluator: Evaluator,
    *,
    Y: Optional[Array] = None,
    initial_labels: Optional[Array] = None,
) -> dict[str, Any] | None:
    """
    Train collectively on every MPI rank (see ``_adaptive_clustering``).

    Failure protocol: in a single process an exception propagates unchanged. Under MPI
    an exception on any rank, in any phase (initial evaluation and clustering, scatter,
    refinement, collection), may leave the other ranks blocked in a collective or a
    receive, so the failing rank logs its diagnostic (rank and traceback) and calls
    ``MPI_Abort``, which ends every rank of the job with a non-zero status.
    """
    return run_collective(
        lambda: _adaptive_clustering(X, config, evaluator, Y=Y, initial_labels=initial_labels), "training"
    )


def run_collective(operation, what: str):
    """
    Run a collective ``operation()`` with the MPI failure policy; the single owner of
    fatal-failure handling for the public collective entry points (``adaptive_clustering``,
    ``AFINNISurrogate.train`` and ``predict``), so wrappers are never nested.

    In a single process the original exception propagates unchanged. Under MPI an
    exception on any rank may leave peers blocked in a collective or receive, so no
    further collective is attempted: the rank logs a best-effort diagnostic (rank,
    phase, traceback) and calls ``MPI_Abort`` in ``finally``, so a logging error cannot
    skip it. Only ``Exception`` is handled; ``KeyboardInterrupt``/``SystemExit`` are left
    to the MPI launcher. A logging handler that blocks forever is out of scope.
    """
    comm = MPI.COMM_WORLD
    scope: list = []
    _PROFILER_SCOPES.append(scope)
    try:
        return operation()  # on success the operation closed its profilers itself
    except Exception:
        # Release the profiler streams this operation opened (not a parent's); a cleanup
        # error must neither replace the original exception nor prevent the abort below.
        for profiler in scope:
            try:
                profiler.close()
            except Exception:
                pass
        if comm.Get_size() > 1:
            try:
                logger.critical(f"R{comm.Get_rank():02}: {what} failed; aborting all ranks.", exc_info=True)
            finally:
                comm.Abort(1)
        raise
    finally:
        # Equal empty lists still belong to different calls; unwind by stack position.
        _PROFILER_SCOPES.pop()


def _adaptive_clustering(
    X: Array,
    config: ClusteringConfig,
    evaluator: Evaluator,
    *,
    Y: Optional[Array] = None,
    initial_labels: Optional[Array] = None,
) -> dict[str, Any] | None:
    """
    Train collectively on every MPI rank; return the merged leaves on rank 0, None elsewhere.

    Only rank 0's ``X``, ``Y`` (a precomputed f(X)) and ``initial_labels`` are read;
    the initial points count as evaluations even when ``Y`` is given.
    ``initial_labels`` (one non-negative integer per row of ``X``, e.g. from
    ``initial_clustering``) replaces the initial clustering, which is then recorded
    as ``profiling_stats["initial_labels_given"]``; they are checked before ``f`` is
    evaluated. ``config`` is copied, so runtime
    changes (the GPU batch-size cap) never reach the
    caller; ``evaluator.n_evals`` keeps counting across calls. The result holds the
    shared columns, the local model's columns, the convergence status and
    ``profiling_stats``.
    """
    config = dataclasses.replace(config)  # local copy; the caller's config is untouched
    region_mode = config.algorithm == "region"
    if region_mode and initial_labels is not None:
        raise ValueError("initial_labels do not apply to algorithm='region': its roots come from region.roots.")
    if not config.requires_no_grads and not evaluator.has_grad:
        raise ValueError(
            f"Selected mode {config.mode.value} requires gradient information. Please provide g_fn or oracle_fn."
        )
    if config.requires_hessian_fn and evaluator.hessian_fn is None:
        raise ValueError(
            f"Selected mode {config.mode.value} requires an exact Hessian. Please provide hessian_fn."
        )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Ranks sharing rank 0's node (collective, so every rank calls it).
    node_ranks = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_size() if size > 1 else 1

    _start_time = time.perf_counter()

    # A start time shared by all ranks, so event times are comparable across ranks.
    t0 = comm.bcast(time.time() if rank == 0 else None, root=0)
    profiling_stats = new_profiling_stats()
    profiler = Profiler(
        profiling_stats,
        rank=rank,
        timing=config.profile_timing,
        events=config.profile_events,
        stream_dir=config.profile_dir,
        t0=t0,
    )
    if _PROFILER_SCOPES:  # closed by the enclosing run_collective if the operation fails
        _PROFILER_SCOPES[-1].append(profiler)
    oracle_seconds_at_start = evaluator.seconds
    timer = profiler.phase

    d_bcast, fd_bcast = 0, 0

    if rank == 0:
        if config.save_all_gen_training_data:
            _, DIM = X.shape
            save_generated_data(
                X, config.generated_data_name(DIM)
            )

        with timer("time_init"):
            X0_gpu = xp.asarray(X, dtype=xp.float64)
            N0, d_bcast = X0_gpu.shape
            profiling_stats["N0"], profiling_stats["total_f_evals"] = N0, N0
            if initial_labels is not None:
                initial_labels = np.asarray(_to_host(initial_labels))
                valid = initial_labels.shape == (N0,) and np.issubdtype(initial_labels.dtype, np.integer)
                if not valid or (N0 and initial_labels.min() < 0):
                    raise ValueError(
                        f"initial_labels must be {N0} non-negative integers, one per training point; "
                        f"got shape {initial_labels.shape} and dtype {initial_labels.dtype}."
                    )
            profiling_stats["initial_labels_given"] = initial_labels is not None

            # Per-point gradients are only carried when the mode uses them at
            # every point; centroid gradients are evaluated during refinement.
            # The N0 initial points always count as evaluations, even with Y.
            need_point_grads = config.requires_grad_everywhere
            if Y is not None:
                F0_gpu = convert_to_2d_array(xp.asarray(Y, dtype=xp.float64))
                G0_gpu = (
                    evaluator(X0_gpu, need_grad=True)[1] if need_point_grads else None
                )
            else:
                F0_gpu, G0_gpu = evaluator(X0_gpu, need_grad=need_point_grads)
            fd_bcast = F0_gpu.shape[1]

            if region_mode:
                domain = Domain.resolve(config.domain, _to_host(X0_gpu))
                labels0 = None
            else:
                labels0 = (
                    initial_clustering(X0_gpu, config, threads=_threads_for_rank0(node_ranks))
                    if initial_labels is None
                    else initial_labels.astype(np.int64)
                )

            if not region_mode:
                sort_idx = np.argsort(labels0)
                labels0_sorted = labels0[sort_idx]
                X0_sorted = _to_host(X0_gpu)[sort_idx]
                F0_sorted = _to_host(F0_gpu)[sort_idx]
                G0_sorted = _to_host(G0_gpu)[sort_idx] if G0_gpu is not None else None

                uniq_labels, all_counts = np.unique(labels0_sorted, return_counts=True)
                counts_chunks = np.array_split(all_counts, size)
        if not region_mode:
            logger.info(
                f"Initial clustering done. Number of clusters: {uniq_labels.shape[0]}."
            )

    plan = None
    if region_mode:
        # Every rank builds the same plan; only the domain (from rank 0's points for
        # "data_bbox") has to be broadcast.
        bounds = comm.bcast((domain.lower, domain.upper) if rank == 0 else None, root=0)
        d_r = comm.bcast(d_bcast if rank == 0 else None, root=0)
        fd_r = comm.bcast(fd_bcast if rank == 0 else None, root=0)
        domain = Domain(*bounds)
        centres = None
        if config.region.roots == "voronoi":
            # The initial clustering's centres (means of its clusters), in normalized
            # coordinates; roots are then assigned by exact nearest centre, not by its labels.
            if rank == 0:
                labels_v = initial_clustering(X0_gpu, config, threads=_threads_for_rank0(node_ranks))
                U0 = _to_host(domain.normalize(X0_gpu))
                inverse = np.unique(labels_v, return_inverse=True)[1]
                centres = np.zeros((int(inverse.max()) + 1, U0.shape[1]))
                np.add.at(centres, inverse, U0)
                centres /= np.bincount(inverse)[:, None]
            centres = comm.bcast(centres, root=0)
        roots = Roots.build(config.region.roots, d_r, config.num_initial_clusters, centres)
        plan = RegionPlan(config, d_r, fd_r, domain, roots)
        if rank == 0:
            all_roots = plan.initial_queue(_to_host(X0_gpu), _to_host(F0_gpu), _to_host(G0_gpu) if G0_gpu is not None else None)
            chunks = np.array_split(np.arange(all_roots.n_clusters), size)
            logger.info(f"Region roots: {roots.n_roots} ({config.region.roots}), domain {domain.lower} .. {domain.upper}.")

    with timer("time_scatter"):
        d_bcast, fd_bcast = comm.bcast((d_bcast, fd_bcast), root=0)
        # Every rank checks, so a tolerance of the wrong length fails before any rank refines.
        config.components().acceptance.check_output_dim(fd_bcast)

        # Rank 0 splits the sorted clusters into one contiguous chunk per rank.
        if rank == 0 and region_mode:
            for r in range(size):
                keep = np.zeros(all_roots.n_clusters, dtype=bool)
                keep[chunks[r]] = True
                chunk = all_roots.select(keep)
                if r == 0:
                    local_clusters = chunk
                else:
                    send_clusters(comm, chunk, dest=r, tag=SCATTER_TAG)
        elif rank == 0:
            current_idx = 0
            for r in range(size):
                num_pts_in_rank = int(np.sum(counts_chunks[r]))
                end_idx = current_idx + num_pts_in_rank
                chunk = ClusterBatch(
                    X0_sorted[current_idx:end_idx],
                    F0_sorted[current_idx:end_idx],
                    G0_sorted[current_idx:end_idx] if G0_sorted is not None else None,
                    counts_chunks[r].astype(np.int64),
                )
                if r == 0:
                    local_clusters = chunk
                else:
                    logger.info(
                        f"R00: Sending initial data chunk to R{r:02} | "
                        f"pts={chunk.n_pts:,} clusters={chunk.n_clusters:,}"
                    )
                    send_clusters(comm, chunk, dest=r, tag=SCATTER_TAG)
                current_idx = end_idx
        else:
            logger.info(f"R{rank:02}: Waiting to receive initial data chunk from R00.")
            local_clusters = recv_clusters(comm, 0, d_bcast, fd_bcast, tag=SCATTER_TAG)

    # Hardware-aware sizing is more appropriate now that the split projection no
    # longer materializes an (N, d, d) tensor. Keep the estimate conservative,
    # though: a rejected batch can still be augmented to roughly 2x its original
    # size and carries several live per-point work arrays during splitting.
    if IS_CUPY_BACKEND:
        import cupy as cp

        # The batch cap below depends on free VRAM, and a different cap changes batch
        # boundaries (and so the random streams): release cached pool blocks first so
        # the reading does not depend on what the pool happens to hold.
        cp.get_default_memory_pool().free_all_blocks()
        free_vram_bytes, total_vram_bytes = cp.cuda.Device().mem_info
        usable_vram_bytes = int(free_vram_bytes * 0.65)
    else:
        free_vram_bytes = total_vram_bytes = 32 * 1024**3
        usable_vram_bytes = free_vram_bytes

    # Approximate live storage during AFFINE_COV_HESS rejected-cluster handling:
    # - base batch X/F/labels
    # - augmented X/F/labels, dx, residual work arrays, split scores/sort buffers
    # - the current tangent residual path still expands A_tangent[label] to
    #   (N, fd, d), so account for that until it is fused as well.
    base_batch_bytes_per_point = (d_bcast + fd_bcast + 1) * 8
    if plan is not None:  # the region plan's own footprint, with the same admission rule
        base_batch_bytes_per_point = plan.bytes_per_point()
    split_augmented_bytes_per_point = (
        3 * d_bcast  # X_aug, dx, selected/source point work
        + fd_bcast * d_bcast
        + 3 * fd_bcast
        + 8  # labels, masks, scalar work arrays, sort/scores
    ) * 8
    augmentation_factor = 2
    estimated_peak_bytes_per_point = (
        base_batch_bytes_per_point
        + augmentation_factor * split_augmented_bytes_per_point
    ) if plan is None else base_batch_bytes_per_point

    hardware_batch_cap = usable_vram_bytes // max(1, estimated_peak_bytes_per_point)
    # AFINNI mode keeps its floor of 50,000 points, so its batches stay bit-identical.
    batch_floor = 50_000 if plan is None else 1_000
    if hardware_batch_cap < batch_floor:
        logger.warning(
            f"Estimated device memory fits only {hardware_batch_cap:,} points per batch "
            f"({estimated_peak_bytes_per_point:,.0f} bytes/point); using the floor of "
            f"{batch_floor:,}, which may run out of memory"
        )
    dynamic_batch_cap = int(np.clip(hardware_batch_cap, batch_floor, 20_000_000))
    if config.batch_size > dynamic_batch_cap:
        logger.info(
            "Hardware VRAM scaling: reducing batch_size from %s to %s for d=%s, "
            "fd=%s using %.2f GB free / %.2f GB total VRAM and an estimated "
            "%.0f bytes/point peak footprint.",
            f"{config.batch_size:,}",
            f"{dynamic_batch_cap:,}",
            d_bcast,
            fd_bcast,
            free_vram_bytes / 1e9,
            total_vram_bytes / 1e9,
            float(estimated_peak_bytes_per_point),
        )
        config.batch_size = dynamic_batch_cap
    # What this run actually used; the caller's (requested) config is unchanged.
    profiling_stats["effective_config"] = {
        "batch_size": int(config.batch_size), "max_levels": int(config.max_levels),
    }  # fmt: skip

    buffers, out_count, local_f_evals, local_gen_pts, last_level = _run_local_refinement(
        config=config,
        evaluator=evaluator,
        clusters=local_clusters,
        comm=comm,
        profiler=profiler,
        plan=plan,
    )
    # Region mode leaves no points unresolved: nodes stopped early keep a fallback leaf,
    # reported separately so the monitor does not show them as queued work.
    unresolved = 0 if plan is not None else int(profiling_stats["num_unresolved_pts"])
    profiler.event(
        "done", leaves=int(out_count), generated=int(local_gen_pts), unresolved_pts=unresolved,
        fallback_pts=int(profiling_stats["num_unresolved_pts"]) if plan is not None else 0,
    )  # fmt: skip

    local_result_host = dict(buffers)
    profiling_stats["num_levels"] = last_level
    profiling_stats["num_generated_pts"] = local_gen_pts
    profiling_stats["total_f_evals"] += local_f_evals
    profiling_stats["num_final_clusters"] = out_count
    profiling_stats["time_total"] = time.perf_counter() - _start_time
    profiling_stats[ORACLE_TIME_KEY] = evaluator.seconds - oracle_seconds_at_start

    active_time = sum(
        profiling_stats.get(k, 0.0)
        for k in ["time_state_update", "time_mpi_send", "time_mpi_recv"]
    )
    profiling_stats["time_idle"] = max(
        0.0, profiling_stats.get("time_idle", 0.0) - active_time
    )
    local_result_host["_profiling_stats"] = profiling_stats

    with timer("time_gather"):
        if rank != 0:
            logger.info(
                f"R{rank:03}: Sending local results ({out_count:,} clusters) to rank 0."
            )
            comm.send(local_result_host, dest=0, tag=LOCAL_RESULT_TAG)
        else:
            logger.info("R00: Merging results from all ranks.")
            all_results = [local_result_host]
            for r in range(1, size):
                all_results.append(comm.recv(source=r, tag=LOCAL_RESULT_TAG))
            logger.info(f"R00: Received local results from {size:,} ranks.")

    if rank == 0:
        all_profiling_stats = [r.pop("_profiling_stats") for r in all_results]
        merged = {
            k: xp.asarray(np.concatenate([r[k] for r in all_results], axis=0))
            for k in all_results[0]
        }

        consolidated_profiling = {
            "time_total": time.perf_counter() - _start_time,
            "t0": t0,
            "N0": all_profiling_stats[0].get("N0", 0),
            "initial_labels_given": all_profiling_stats[0].get("initial_labels_given", False),
            "num_levels": max(ps["num_levels"] for ps in all_profiling_stats),
            "num_final_clusters": int(merged["centroids"].shape[0]),
            "effective_config": all_profiling_stats[0].get("effective_config", {}),
            "level_stats": sorted(
                [entry for ps in all_profiling_stats for entry in ps["level_stats"]],
                key=lambda e: (e["level"], e["rank"]),
            ),
            "per_rank": all_profiling_stats,
        }

        for k in [
            k
            for k in all_profiling_stats[0]
            if k.startswith("time_") and k != "time_total"
        ]:
            consolidated_profiling[k] = sum(
                ps.get(k, 0.0) for ps in all_profiling_stats
            )
        for k in [
            "num_generated_pts",
            "total_f_evals",
            "num_unresolved_clusters",
            "num_unresolved_pts",
        ]:
            consolidated_profiling[k] = sum(ps.get(k, 0) for ps in all_profiling_stats)
        consolidated_profiling["converged"] = (
            consolidated_profiling["num_unresolved_clusters"] == 0
        )
        consolidated_profiling["plan"] = all_profiling_stats[0].get("plan")
        if plan is not None:
            nodes = {k[len("node:"):]: _to_host(merged.pop(k)) for k in list(merged) if k.startswith("node:")}
            routing = RoutingTable.build(
                plan.domain, plan.roots, merged["root"], merged["path_code"],
                nodes["root"], nodes["path_code"], nodes["w"], nodes["z"],
            )  # fmt: skip
            merged["routing"] = routing.arrays()
            reasons = np.bincount(_to_host(merged["terminal_reason"]).astype(np.int64), minlength=len(TERMINAL_REASONS))
            consolidated_profiling["terminal_reasons"] = dict(zip(TERMINAL_REASONS, map(int, reasons), strict=True))
            consolidated_profiling["num_split_nodes"] = int(nodes["root"].shape[0])
            consolidated_profiling["region"] = {
                k: sum(ps.get("region", {}).get(k, 0) for ps in all_profiling_stats) for k in plan.counters
            }

        merged.update(
            {
                "num_levels": consolidated_profiling["num_levels"],
                "num_generated_pts": consolidated_profiling["num_generated_pts"],
                "total_f_evals": consolidated_profiling["total_f_evals"],
                "converged": consolidated_profiling["converged"],
                "num_unresolved_clusters": consolidated_profiling["num_unresolved_clusters"],
                "num_unresolved_pts": consolidated_profiling["num_unresolved_pts"],
                "profiling_stats": consolidated_profiling,
            }
        )

        # Combine all the files if we are saving
        if config.save_all_gen_training_data:
            combine_generated_data(
                config.generated_data_name(d_bcast)
            )

        profiler.close()
        return merged
    profiler.close()
    return None
