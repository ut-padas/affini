"""One rank's refinement loop: batches of clusters fitted, accepted or split level by level, with work stealing between ranks."""

import itertools
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src import (
    IS_CUPY_BACKEND,
    Array,
    _to_host,
    xp,
)
from src.components.local_models import ClusterFitData, deploy_leaves, output_columns
from src.components.sampling import augment_clusters, augmentation_targets
from src.components.splitting import SplitData
from src.config import ClusteringConfig
from src.engine.clusters import ClusterBatch, ClusterQueueBuilder, ColumnBuffer
from src.engine.evaluation import Evaluator
from src.engine.io import save_generated_data
from src.engine.work_stealing import (
    make_work_stealer,
)
from src.infra.backend import (  # MPI: real mpi4py.MPI, or a single-rank stand-in
    MPI,
)
from src.infra.logging_utils import log_mem_stats
from src.infra.profiling import Profiler
from src.numerics import _reduce_sum_1d, repeats
from src.region import RegionPlan

logger = logging.getLogger(__name__)

_CHECK_CHUNK = 1 << 22  # rows per gather in the split check (bounds its temporary memory)


def plan_label(config: ClusteringConfig) -> str:
    """A short name for the algorithm plan of ``config``, for profiles, plots and the monitor."""
    if config.algorithm == "region":
        return f"AFINNI region mode (ART; {config.region.split_search} splits, {config.region.roots} roots)"
    return "AFINNI"


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



@dataclass
class _BatchArrays:
    """
    One batch's points, values and gradients on the device. ``_afinni_batch`` takes them over
    and drops each as soon as it is done with it, so the caller must not keep references (they
    would hold the batch through augmentation and splitting, the batch's peak memory).
    """

    X: Array | None
    F: Array | None
    G: Array | None


def _afinni_batch(arrays: _BatchArrays, sizes: Array, starts: Array, depth_cpu: np.ndarray, *, config: ClusteringConfig,
                  components, evaluator: Evaluator, accepted: ColumnBuffer, next_level: ClusterQueueBuilder, timer,
                  profiler: Profiler, mempool, rank: int, level: int, b: int, num_batches: int, d: int) -> tuple[int, int]:  # fmt: skip
    """
    One AFINNI-mode batch: fit the local model on every cluster, append the accepted ones (with
    their trust regions) to ``accepted``, and augment and split the rejected ones into
    ``next_level``. Returns (accepted clusters, generated points).
    """
    model, splitter, acceptance = components.local_model, components.split_strategy, components.acceptance
    trust_region, sampler = components.trust_region, components.sampler
    generated = 0
    aug_sizes = sizes
    K_batch = len(sizes)

    X_batch, F_batch, G_batch = arrays.X, arrays.F, arrays.G
    arrays.X = arrays.F = arrays.G = None  # this function now holds the only references

    aug_starts = starts
    aug_ids = repeats(xp.arange(K_batch), sizes)

    with timer("time_error_calc"):
        mus = _reduce_sum_1d(X_batch, aug_starts) / aug_sizes[:, None]

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
                generated += int(aug.X_new.shape[0])
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
    with timer("time_gc"):
        del X_batch, F_batch, G_batch
        if mempool is not None:
            mempool.free_all_blocks()
    return K_good_batch, generated


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
    handled by the work stealer from ``src.engine.work_stealing``. Timings, level
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

    # The components (the AFINNI batch step takes the rest from ``components``), and
    # host-side output columns for the accepted clusters (the shared columns plus
    # the model's parameters).
    components = config.components()
    model = components.local_model
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
                f"{config.batch_size:,}; its batch may exceed the memory budget"
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

        # AFINNI batches end at the first cluster crossing the grid (b + 1) * batch_size of
        # cumulative points, so a batch can exceed batch_size by its last cluster; a cluster
        # that would start a batch beyond the grid is processed alone and the grid
        # re-anchored after it (batch_offset). Region batches hold whole nodes up to
        # batch_size (see below). The loop runs until every queued cluster was processed,
        # so oversized clusters are never dropped (they once silently were).
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

            if start_cluster_idx >= num_clusters:
                break
            if plan is not None:
                # Region mode: the longest run of whole nodes that fits batch_size, or one
                # oversized node alone. Its trees do not depend on batch boundaries.
                before = int(aug_cumsum_cpu[start_cluster_idx - 1]) if start_cluster_idx else 0
                end_cluster_idx = int(np.searchsorted(aug_cumsum_cpu, before + config.batch_size, side="right"))
                end_cluster_idx = min(max(end_cluster_idx, start_cluster_idx + 1), num_clusters)
            else:
                # AFINNI mode keeps its grid, so its batches (and seeds) stay bit-identical:
                # the cluster crossing a grid line joins that batch.
                target_aug_count = (b + 1) * config.batch_size + batch_offset
                end_cluster_idx = int(np.searchsorted(aug_cumsum_cpu, target_aug_count))
                end_cluster_idx = min(end_cluster_idx + 1, num_clusters)
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

            arrays = _BatchArrays(pts, f_vals, g_vals)
            del pts, f_vals, g_vals
            k_good_batch, generated = _afinni_batch(
                arrays, sizes, starts_local_shifted, depth_cpu, config=config, components=components,
                evaluator=evaluator, accepted=accepted, next_level=next_level, timer=timer, profiler=profiler,
                mempool=mempool, rank=rank, level=level, b=b, num_batches=num_batches, d=d,
            )  # fmt: skip
            K_good += k_good_batch
            num_generated_pts += generated
            level_generated += generated
            start_cluster_idx = end_cluster_idx

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
