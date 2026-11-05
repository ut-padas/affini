import dataclasses
import gc
import logging
import os
import time
from typing import Any, Optional

import numpy as np

from src import (
    HAS_CUML,
    IS_CUPY_BACKEND,
    Array,
    _get_mpi_rank,
    _get_mpi_size,
    _to_host,
    xp,
)
from src.adaptive_clustering_utils import (
    ClusteringConfig,
    CudaTimer,
    Evaluator,
    SurrogateMode,
    combine_generated_data,
    compute_accepted_stats,
    convert_to_2d_array,
    evaluate_cluster_errors,
    save_generated_data,
    uniform_grid_cluster_labels,
)
from src.backend import MPI  # real mpi4py.MPI, or a single-rank stand-in without MPI
from src.clusters import ClusterBatch, ClusterQueueBuilder, ColumnBuffer
from src.local_models import make_local_model, output_columns
from src.logging_utils import log_mem_stats
from src.sampling import augment_clusters, augmentation_targets, make_sampler
from src.splitting import SplitData, split_strategy_for_mode
from src.utils import repeats
from src.work_stealing import (
    make_work_stealer,
    recv_clusters,
    send_clusters,
)

INITIAL_CLUSTERING_FILE = "initial_clustering_labels.npy"
# Initial scatter of clusters from rank 0 (payload tags SCATTER_TAG + 1..4); kept
# well away from the work-stealing tags.
SCATTER_TAG = 1000

logger = logging.getLogger(__name__)
TINY_EPS = 1e-12

def _run_local_refinement(
    config: ClusteringConfig,
    evaluator: Evaluator,
    clusters: ClusterBatch,
    F_max_global: Array,
    comm: MPI.Intracomm,
    profiling_stats: dict,
):
    """
    Core local BFS refinement loop for GPU adaptive clustering, run independently
    on each MPI rank. Takes this rank's pre-sorted clusters (host memory) and
    returns finalized output buffers plus counts.

    Algorithm settings (tolerances, limits, seed, work stealing, ...) are read
    from ``config``; ``config.max_clusters_est`` is the total across all ranks.
    Function evaluations go through ``evaluator``; the returned evaluation count
    covers only the evaluations made here. Load balancing between ranks is
    handled by the work stealer from ``src.work_stealing``.
    """
    rank, size = comm.Get_rank(), comm.Get_size()
    d, fd = clusters.d, clusters.fd
    queue = clusters

    mempool = None
    if IS_CUPY_BACKEND:
        import cupy as cp

        mempool = cp.get_default_memory_pool()

    # The function fitted on each cluster, and host-side output columns for the
    # accepted clusters (the shared columns plus the model's parameters).
    model = make_local_model(config)
    splitter = split_strategy_for_mode(config.mode, config.min_pts_per_cluster)
    if model.needs_centroid_grad and not config.requires_grad_centroids:
        raise ValueError(f"{model.name} needs gradients at the centroids")
    accepted = ColumnBuffer(
        output_columns(model, d, fd),
        capacity=max(1, config.max_clusters_est // max(1, size)),
    )
    num_generated_pts = 0
    evals_at_start = evaluator.n_evals
    level = 0

    sampler = make_sampler(config.sampling_method)
    stealer = make_work_stealer(
        config.use_work_stealing, comm, d, fd, profiling_stats, config.profile_timing
    )
    stealer.exchange_initial_loads(queue.n_pts)

    while True:
        num_clusters = queue.n_clusters
        stealer.poll_state()
        stealer.set_load(queue.n_pts if num_clusters > 0 else 0)

        # --- Phase 3: Idle / Stealing Phase ---
        if num_clusters == 0 or level > config.max_levels:
            retired, stolen = stealer.idle_step(level > config.max_levels)
            if retired:
                break
            if stolen is not None:
                queue = stolen
            # Loop back around to check termination and steal opportunities again.
            continue

        # --- Phase 1: The Compute Phase (Mid-Level) ---
        _t_level = time.perf_counter()
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
            }
        )

        level_total_pts = int(np.sum(queue.sizes))
        aug_cumsum_cpu = np.cumsum(queue.sizes)

        # Children of this level's rejected clusters, collected batch by batch.
        next_level = ClusterQueueBuilder(
            d,
            fd,
            has_g=config.requires_grad_everywhere,
            capacity_pts=level_total_pts,
            capacity_clusters=2 * num_clusters,
        )

        num_batches = level_total_pts // config.batch_size + 1
        start_cluster_idx = 0
        K_good = 0

        if config.log_mem_stats:
            log_mem_stats(f"R{rank:02} Start: L{level:02} Batching")

        for b in range(num_batches):
            logger.info(
                f"R{rank:02} L{level:02}: Processing Batch {b + 1}/{num_batches}"
            )

            # --- Mid-Level Work Stealing: give away unprocessed clusters ---
            queue = stealer.serve_requests(queue, start_cluster_idx, level, batch=b)
            stealer.poll_state()
            num_clusters = queue.n_clusters
            aug_cumsum_cpu = aug_cumsum_cpu[:num_clusters]

            target_aug_count = (b + 1) * config.batch_size
            end_cluster_idx = int(np.searchsorted(aug_cumsum_cpu, target_aug_count))
            end_cluster_idx = min(end_cluster_idx + 1, num_clusters)
            if start_cluster_idx >= end_cluster_idx:
                break

            with CudaTimer("time_slicing", profiling_stats, config.profile_timing):
                sizes_cpu = queue.sizes[start_cluster_idx:end_cluster_idx]
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

            aug_sizes = sizes
            K_batch = len(sizes)

            X_batch = pts
            F_batch = f_vals
            G_batch = g_vals

            aug_starts = starts_local_shifted
            aug_ids = repeats(xp.arange(K_batch), sizes)

            with CudaTimer("time_error_calc", profiling_stats, config.profile_timing):
                mus = xp.add.reduceat(X_batch, aug_starts) / aug_sizes[:, None]

                f_mus, g_mus = evaluator(
                    mus, need_grad=config.requires_grad_centroids
                )

                # Fit the local model on every cluster, evaluate it at the
                # cluster's points, and accept the clusters within tolerance.
                is_good, E_cluster, surrogate_params = evaluate_cluster_errors(
                    config,
                    model,
                    X_batch,
                    F_batch,
                    mus,
                    f_mus,
                    g_mus,
                    aug_starts,
                    aug_sizes,
                    aug_ids,
                    F_max_global,
                )

            if config.log_mem_stats:
                log_mem_stats(
                    f" R{rank:02} L{level:02} B{b:02}/{num_batches:02}: After Error Metrics"
                )

            # 5. Handle Accepted ("Good") Clusters
            with CudaTimer("time_filter", profiling_stats, config.profile_timing):
                good_idx_local = xp.nonzero(is_good)[0]
                K_good_batch = int(good_idx_local.shape[0])
                K_good += K_good_batch

            if K_good_batch > 0:
                with CudaTimer("time_good_Q", profiling_stats, config.profile_timing):
                    good_mask = is_good[aug_ids]
                    X_good = X_batch[good_mask]
                    G_good = (
                        G_batch[good_mask]
                        if (queue.G is not None and G_batch is not None)
                        else None
                    )
                    good_counts = aug_sizes[good_idx_local]
                    dense_labels_good = repeats(xp.arange(K_good_batch), good_counts)

                    # Shape matrix Q of each accepted cluster (used by the
                    # trust-region test at inference), and the second-order
                    # Hessian H when it is stored.
                    Qs, Hs = compute_accepted_stats(
                        config,
                        X_good,
                        G_good,
                        mus[good_idx_local],
                        g_mus[good_idx_local] if g_mus is not None else None,
                        dense_labels_good,
                        good_counts,
                    )

                with CudaTimer("time_good_d2h", profiling_stats, config.profile_timing):
                    rows = {
                        "centroids": mus[good_idx_local],
                        "f": f_mus[good_idx_local],
                        "Q": Qs,
                        "E_cluster": E_cluster[good_idx_local],
                        "counts": good_counts.astype(xp.int64),
                        **{k: v[good_idx_local] for k, v in surrogate_params.items()},
                    }
                    if Hs is not None:
                        rows["H"] = Hs  # second-order Taylor term (accepted clusters only)
                    accepted.append(rows)

            # 6. Handle Rejected ("Bad") Clusters -> Split
            with CudaTimer("time_filter", profiling_stats, config.profile_timing):
                bad_idx_local = xp.nonzero(~is_good)[0]
                n_bad = int(bad_idx_local.shape[0])

            if n_bad > 0:
                with CudaTimer("time_bad_split", profiling_stats, config.profile_timing):
                    bad_mask = (~is_good)[aug_ids]
                    X_bad = X_batch[bad_mask]
                    F_bad = F_batch[bad_mask]
                    G_bad = (
                        G_batch[bad_mask]
                        if (queue.G is not None and G_batch is not None)
                        else None
                    )
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
                    )
                    X_bad_aug, F_bad_aug, G_bad_aug = aug.X, aug.F, aug.G
                    aug_dense_labels_bad = aug.labels
                    if aug.X_new is not None:
                        num_generated_pts += int(aug.X_new.shape[0])
                        if config.save_all_gen_training_data:
                            save_generated_data(
                                aug.X_new,
                                f"train_data_atol_{config.atol}_rtol_{config.rtol}_d{d}",
                            )
                    del aug

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

                with CudaTimer("time_bad_d2h", profiling_stats, config.profile_timing):
                    # Children: each cluster cut along its split direction.
                    next_level.append(
                        X_bad_aug[sort_order],
                        F_bad_aug[sort_order],
                        G_bad_aug[sort_order]
                        if config.requires_grad_everywhere and G_bad_aug is not None
                        else None,
                        child_sizes,
                    )

                with CudaTimer("time_gc", profiling_stats, config.profile_timing):
                    del (
                        X_bad_aug,
                        F_bad_aug,
                        G_bad_aug,
                        sort_order,
                    )
                    if mempool is not None:
                        mempool.free_all_blocks()

            start_cluster_idx = end_cluster_idx
            with CudaTimer("time_gc", profiling_stats, config.profile_timing):
                del X_batch, F_batch, G_batch
                if mempool is not None:
                    mempool.free_all_blocks()

        # --- Phase 2: The Level Boundary (Transition & Steal Response) ---
        with CudaTimer("time_transition", profiling_stats, config.profile_timing):
            next_queue = next_level.build()
            if next_queue.n_clusters == 0:
                logger.info(
                    f"R{rank:02} L{level:02}: No bad clusters to refine, terminating."
                )

            # --- Level-Boundary Work Stealing ---
            stealer.publish_load(next_queue.n_pts)
            queue = stealer.serve_requests(next_queue, 0, level)

        if profiling_stats["level_stats"]:
            profiling_stats["level_stats"][-1].update(
                {
                    "clusters_level_end": int(len(queue.sizes)),
                    "good_clusters": int(K_good),
                    "num_pts_next_level": int(queue.X.shape[0]),
                }
            )

        if config.log_mem_stats:
            log_mem_stats(f"R{rank:02} End: L{level:02} Transition")

        with CudaTimer("time_gc", profiling_stats, config.profile_timing):
            del (
                next_level,
                next_queue,
            )
            gc.collect()

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
            profiling_stats["level_stats"][-1].update({"runtime_sec": level_runtime})
        level += 1

    total_f_evals = evaluator.n_evals - evals_at_start
    return accepted.result(), accepted.n_rows, total_f_evals, num_generated_pts, level


# ---------------------------------------------------------------------------
# Public entry point (MPI-aware)
# ---------------------------------------------------------------------------
def adaptive_clustering(
    X: Array,
    config: ClusteringConfig,
    evaluator: Evaluator,
    *,
    Y: Optional[Array] = None,
) -> dict[str, Any] | None:
    """
    Adaptive Clustering for Vector-Valued Functions (GPU Accelerated).

    Args:
        X: (N, d) training points. Only rank 0's X is used; other ranks may pass an
            empty array.
        config: algorithm settings. It is not modified; runtime adjustments to
            max_levels and batch_size are made on a copy.
        evaluator: evaluates f (and its gradient / Hessian where the mode needs
            them). Its n_evals counter keeps increasing across calls.
        Y: optional precomputed f(X), shape (N, fd) or (N,). The N initial points
            count as evaluations either way.

    Returns:
        On rank 0, the merged result dict (centroids, f, Q, E_cluster, counts,
        and optional g, H, A_fit/b_fit, plus profiling_stats); None on
        other ranks.

    config.save_initial_clustering saves the initial cluster labels to
    INITIAL_CLUSTERING_FILE. config.use_existing_initial_clustering skips the initial
    clustering and loads the labels from that file if it exists and has one label per
    row of X; otherwise clustering is done as normal.
    """
    config = dataclasses.replace(config)  # local copy; the caller's config is untouched
    if not config.requires_no_grads and not evaluator.has_grad:
        raise ValueError(
            f"Selected mode {config.mode.value} requires gradient information. Please provide g_fn or oracle_fn."
        )
    if (
        split_strategy_for_mode(config.mode, config.min_pts_per_cluster).needs_hessian_fn
        and evaluator.hessian_fn is None
    ):
        raise ValueError(
            f"Selected mode {config.mode.value} requires an exact Hessian. Please provide hessian_fn."
        )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # With one rank there are no stealing opportunities, so respect the
    # caller's refinement limit even when use_work_stealing defaults to True.
    if config.use_work_stealing and size > 1 and config.max_levels <= 100:
        _safe_max_levels = 1000
        logger.warning(
            f"Work stealing enabled but max_levels is only {config.max_levels}. Overriding max_levels to {_safe_max_levels} to ensure sufficient stealing opportunities across levels."
        )
        config.max_levels = _safe_max_levels

    _start_time = time.perf_counter()

    profiling_stats = {
        "N0": 0,
        "num_levels": 0,
        "num_generated_pts": 0,
        "total_f_evals": 0,
        "num_final_clusters": 0,
        "time_total": 0.0,
        "time_init": 0.0,
        "time_scatter": 0.0,
        "time_state_update": 0.0,
        "time_idle": 0.0,
        "time_find_req_synth": 0.0,
        "time_mpi_send": 0.0,
        "time_mpi_recv": 0.0,
        "time_slicing": 0.0,
        "time_synth_gen": 0.0,
        "time_error_calc": 0.0,
        "time_filter": 0.0,
        "time_good_Q": 0.0,
        "time_good_d2h": 0.0,
        "time_bad_split": 0.0,
        "time_bad_d2h": 0.0,
        "time_transition": 0.0,
        "time_gc": 0.0,
        "time_gather": 0.0,
        "level_stats": [],
    }

    F_max_global_host = None
    d_bcast, fd_bcast = 0, 0

    if rank == 0:
        if config.save_all_gen_training_data:
            _, DIM = X.shape
            save_generated_data(
                X, f"train_data_atol_{config.atol}_rtol_{config.rtol}_d{DIM}"
            )

        with CudaTimer("time_init", profiling_stats, config.profile_timing):
            X0_gpu = xp.asarray(X, dtype=xp.float64)
            N0, d_bcast = X0_gpu.shape
            profiling_stats["N0"], profiling_stats["total_f_evals"] = N0, N0

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

            F_max_global_host = _to_host(xp.max(xp.abs(F0_gpu), axis=0))

            labels0 = None
            if config.use_existing_initial_clustering and os.path.exists(INITIAL_CLUSTERING_FILE):
                saved_labels = np.load(INITIAL_CLUSTERING_FILE)
                if saved_labels.ndim == 1 and saved_labels.shape[0] == N0:
                    labels0 = saved_labels.astype(np.int64)
                    logger.info(f"Loaded initial clustering from {INITIAL_CLUSTERING_FILE}.")
                else:
                    logger.warning(
                        f"{INITIAL_CLUSTERING_FILE} has shape {saved_labels.shape}, expected ({N0},). Clustering as normal."
                    )

            if labels0 is None:
                if config.initial_clustering_method == "kmeans":
                    if HAS_CUML:
                        from cuml.cluster import KMeans as cuKMeans

                        km = cuKMeans(n_clusters=config.num_initial_clusters, random_state=config.seed)
                        labels0 = km.fit_predict(X0_gpu)
                    else:
                        from sklearn.cluster import KMeans

                        km = KMeans(
                            n_clusters=config.num_initial_clusters, random_state=config.seed, max_iter=2
                        )
                        labels0 = km.fit_predict(_to_host(X0_gpu))
                    labels0 = _to_host(labels0).astype(np.int64)

                elif config.initial_clustering_method == "knndbscan":
                    from knndbscan import knn_dbscan

                    labels0 = _to_host(
                        knn_dbscan(
                            X0_gpu,
                            eps=0.5,
                            min_samples=config.min_pts_per_cluster,
                            device="gpu" if IS_CUPY_BACKEND else "cpu",
                        )
                    ).astype(np.int64)

                elif config.initial_clustering_method == "uniform_grid":
                    labels0, _ = uniform_grid_cluster_labels(
                        _to_host(X0_gpu), config.num_initial_clusters
                    )
                    labels0 = labels0.astype(np.int64)

                else:
                    raise ValueError(f"Invalid: {config.initial_clustering_method}")

                if config.save_initial_clustering:
                    np.save(INITIAL_CLUSTERING_FILE, labels0)
                    logger.info(f"Saved initial clustering to {INITIAL_CLUSTERING_FILE}.")

            sort_idx = np.argsort(labels0)
            labels0_sorted = labels0[sort_idx]
            X0_sorted = _to_host(X0_gpu)[sort_idx]
            F0_sorted = _to_host(F0_gpu)[sort_idx]
            G0_sorted = _to_host(G0_gpu)[sort_idx] if G0_gpu is not None else None

            uniq_labels, all_counts = np.unique(labels0_sorted, return_counts=True)
            counts_chunks = np.array_split(all_counts, size)
        logger.info(
            f"Initial clustering done. Number of clusters: {uniq_labels.shape[0]}."
        )

    with CudaTimer("time_scatter", profiling_stats, config.profile_timing):
        F_max_global_host = comm.bcast(F_max_global_host, root=0)
        d_bcast, fd_bcast = comm.bcast((d_bcast, fd_bcast), root=0)

        # Rank 0 splits the sorted clusters into one contiguous chunk per rank.
        if rank == 0:
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

    # KEEP DATA ON THE CPU (Host RAM)
    F_max_global = xp.asarray(F_max_global_host, dtype=xp.float64)

    # Hardware-aware sizing is more appropriate now that the split projection no
    # longer materializes an (N, d, d) tensor. Keep the estimate conservative,
    # though: a rejected batch can still be augmented to roughly 2x its original
    # size and carries several live per-point work arrays during splitting.
    if IS_CUPY_BACKEND:
        import cupy as cp

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
    )

    hardware_batch_cap = usable_vram_bytes // max(1, estimated_peak_bytes_per_point)
    dynamic_batch_cap = int(np.clip(hardware_batch_cap, 50_000, 20_000_000))
    if config.batch_size > dynamic_batch_cap:
        logger.info(
            "Hardware VRAM scaling: reducing BATCH_SIZE from %s to %s for d=%s, "
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

    buffers, out_count, local_f_evals, local_gen_pts, last_level = (
        _run_local_refinement(
            config=config,
            evaluator=evaluator,
            clusters=local_clusters,
            F_max_global=F_max_global,
            comm=comm,
            profiling_stats=profiling_stats,
        )
    )

    local_result_host = dict(buffers)
    profiling_stats["num_levels"] = last_level
    profiling_stats["num_generated_pts"] = local_gen_pts
    profiling_stats["total_f_evals"] += local_f_evals
    profiling_stats["num_final_clusters"] = out_count
    profiling_stats["time_total"] = time.perf_counter() - _start_time

    active_time = sum(
        profiling_stats.get(k, 0.0)
        for k in ["time_state_update", "time_mpi_send", "time_mpi_recv"]
    )
    profiling_stats["time_idle"] = max(
        0.0, profiling_stats.get("time_idle", 0.0) - active_time
    )
    local_result_host["_profiling_stats"] = profiling_stats

    with CudaTimer("time_gather", profiling_stats, config.profile_timing):
        LOCAL_RESULT_TAG = 2000
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
            "N0": all_profiling_stats[0].get("N0", 0),
            "num_levels": max(ps["num_levels"] for ps in all_profiling_stats),
            "num_final_clusters": int(merged["centroids"].shape[0]),
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
        for k in ["num_generated_pts", "total_f_evals"]:
            consolidated_profiling[k] = sum(ps.get(k, 0) for ps in all_profiling_stats)

        merged.update(
            {
                "num_levels": consolidated_profiling["num_levels"],
                "num_generated_pts": consolidated_profiling["num_generated_pts"],
                "total_f_evals": consolidated_profiling["total_f_evals"],
                "profiling_stats": consolidated_profiling,
            }
        )

        # Combine all the files if we are saving
        if config.save_all_gen_training_data:
            combine_generated_data(
                f"train_data_atol_{config.atol}_rtol_{config.rtol}_d{d_bcast}"
            )

        return merged
    return None


if __name__ == "__main__":
    import json
    from functools import partial

    from src import RESULTS_DIR
    from src.heuristic_parameters import (
        get_MIN_PTS_PER_CLUSTER,
        get_N_TRAIN,
        get_NUM_INITIAL_CLUSTERS,
    )
    from src.logging_utils import setup_logging
    from src.plotter_utils import (
        plot_adaptive_clustering_results,
        plot_mpi_level_heatmap,
        plot_mpi_rank_time_breakdown,
    )
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging(console_level=logging.DEBUG)
    _rank = _get_mpi_rank()
    _size = _get_mpi_size()

    # Create partial Rosenbrock functions (all ranks need these for oracle calls)
    A_ROSENBROCK = 100.0
    rosenbrock = partial(rosenbrock, a=A_ROSENBROCK)
    rosenbrock_grad = partial(rosenbrock_grad, a=A_ROSENBROCK)

    DIM = 2
    ATOL = 0.0001
    RTOL = 0.01
    PROB_LIMS = (-1.0, 3)
    N_TRAIN = get_N_TRAIN(DIM)
    NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(DIM)
    MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(DIM)
    USE_WORK_STEALING = True
    ERROR_METRIC = "L2"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    ERROR_EVAL_MODE = SurrogateMode.AFFINE_COV_HESS

    OUTPUT_DIR = RESULTS_DIR / "adaptive_clustering_experimental"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_fpath = (
        OUTPUT_DIR
        / f"mpi_sz{_size}_rosen{A_ROSENBROCK}_ntrain{N_TRAIN}_dim{DIM}_atol{ATOL}_rtol{RTOL}_err{ERROR_EVAL_MODE.value}_metric{ERROR_METRIC}_init{INITIAL_CLUSTERING_METHOD}_nclust{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}.png"
    )

    if _rank == 0:
        xp.random.seed(0)
        X = xp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(N_TRAIN, DIM))
    else:
        X = xp.empty((0, DIM), dtype=xp.float64)

    result = adaptive_clustering(
        X,
        ClusteringConfig(
            mode=ERROR_EVAL_MODE,
            atol=ATOL,
            rtol=RTOL,
            error_metric=ERROR_METRIC,
            initial_clustering_method=INITIAL_CLUSTERING_METHOD,
            num_initial_clusters=NUM_INITIAL_CLUSTERS,
            min_pts_per_cluster=MIN_PTS_PER_CLUSTER,
            log_mem_stats=True,
            profile_timing=True,
            use_work_stealing=USE_WORK_STEALING,
            max_levels=200,
        ),
        Evaluator(f_fn=rosenbrock, g_fn=rosenbrock_grad),
    )

    if _rank != 0:
        MPI.Finalize()
        raise SystemExit(0)

    assert result, "Rank 0 should have the result"
    n_centroids = result["centroids"].shape[0]
    print(f"Num clusters found: {n_centroids:,}")
    print(f"Time taken: {result['profiling_stats']['time_total']:.3f}s")

    # Store json of all profiling stats
    json_fpath = img_fpath.with_suffix(".json")
    json_data = {
        "IS_CUPY_BACKEND": IS_CUPY_BACKEND,
        "mpi_size": _size,
        "rosenbrock_a": A_ROSENBROCK,
        "dim": DIM,
        "atol": ATOL,
        "rtol": RTOL,
        "n_train": N_TRAIN,
        "num_clusters": n_centroids,
        "train_time_sec": result["profiling_stats"]["time_total"],
        "init_time_sec": result["profiling_stats"]["time_init"],
        "gather_time_sec": result["profiling_stats"]["time_gather"],
        "num_levels": result["num_levels"],
        "num_generated_pts": result["num_generated_pts"],
        "total_f_evals": result["total_f_evals"],
        "error_metric": ERROR_METRIC,
        "error_eval_mode": ERROR_EVAL_MODE.value,
        "initial_clustering_method": INITIAL_CLUSTERING_METHOD,
        "num_initial_clusters": NUM_INITIAL_CLUSTERS,
        "min_pts_per_cluster": MIN_PTS_PER_CLUSTER,
    }
    with open(json_fpath, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Saved profiling stats to {json_fpath}")

    if DIM == 2 and n_centroids <= 10_000:
        title = f"Rosenbrock (a={A_ROSENBROCK}) (Atol={ATOL}, Rtol={RTOL}, ErrMode={ERROR_EVAL_MODE.value}, Metric={ERROR_METRIC}, N_init={NUM_INITIAL_CLUSTERS}, MinPts={MIN_PTS_PER_CLUSTER}, Steal={USE_WORK_STEALING}) (N0={N_TRAIN:,}, N_clusters={n_centroids:,})"
        plot_adaptive_clustering_results(
            centroids=_to_host(result["centroids"]),
            Q=_to_host(result["Q"]),
            domain=PROB_LIMS,
            f_fn=rosenbrock,
            img_fpath=img_fpath,
            title=title,
        )

    # Save MPI rank time breakdown plot
    time_breakdown_path = img_fpath.with_name(
        img_fpath.stem + "_mpi_time_breakdown.png"
    )
    plot_mpi_rank_time_breakdown(result["profiling_stats"], time_breakdown_path)

    # MPI × Level workload heatmap (good_clusters is the primary workload signal)
    for _heatmap_col in (
        "good_clusters",
        "clusters_start",
        "num_pts_next_level",
        "runtime_sec",
    ):
        _hm_path = img_fpath.with_name(
            img_fpath.stem + f"_mpi_heatmap_{_heatmap_col}.png"
        )
        title = f"MPI x Level Heatmap ({_heatmap_col}) (Train Time={result['profiling_stats']['time_total']:.1f}s) (N_CLUSTERS={n_centroids:,})"
        plot_mpi_level_heatmap(result, _hm_path, values_col=_heatmap_col, title=title)

    # Save npz file of results
    npz_fpath = img_fpath.with_suffix(".npz")
    result = {k: _to_host(v) for k, v in result.items()}
    xp.savez_compressed(npz_fpath, **result)
    print(f"Saved results to {npz_fpath}")
