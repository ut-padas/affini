import gc
import logging
import time
from typing import Any, Callable, Literal, Optional

import numpy as np
from mpi4py import MPI

from src import IS_CUPY_BACKEND, Array, _get_mpi_rank, _get_mpi_size, _to_host, xp
from src.adaptive_clustering_utils import (
    ClusteringConfig,
    CudaTimer,
    SurrogateMode,
    adjust_buffer_capacity_host,
    allocate_output_buffers_host,
    compute_accepted_stats,
    convert_to_2d_array,
    convert_to_3d_array,
    evaluate_cluster_errors,
    split_rejected_clusters,
    uniform_grid_cluster_labels,
    save_generated_data,
    combine_generated_data,
)
from src.logging_utils import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)
TINY_EPS = 1e-12

# --- Custom MPI Tags for Decentralized Work-Stealing ---
TAG_STATE_BROADCAST = 101
TAG_STEAL_REQ = 102
TAG_STEAL_GRANT = 103
TAG_STEAL_DENY = 104
TAG_TERM = 105

# --- Work-Stealing Tuning Parameters ---
MIN_LOAD_THRESHOLD = 1_000_000  # Min points to be eligible for stealing
STEAL_SIZE = 500_000  # Target point count to steal

SLEEP_INTERVAL = 0.05  # Seconds to sleep when idle or after failed steal


def _run_local_refinement(
    config: ClusteringConfig,
    CLUSTER_PTS: Array,
    CLUSTER_F: Array,
    CLUSTER_G: Optional[Array],
    CLUSTER_SIZES: Array,
    CLUSTER_STARTS: Array,
    F_max_global: Array,
    d: int,
    fd: int,
    f_fn: Optional[Callable],
    g_fn: Optional[Callable],
    oracle_fn: Optional[Callable],
    TOL: float,
    MIN_PTS_PER_CLUSTER: int,
    MAX_CLUSTERS_EST: int,
    SEED: int,
    Q_SHRINK: float,
    MAX_LEVELS: int,
    BATCH_SIZE: int,
    USE_WORK_STEALING: bool,
    LOG_MEM_STATS: bool,
    PROFILE_TIMING: bool,
    SAVE_ALL_GEN_CLUSTER_DATA: bool,
    rank: int,
    size: int,
    profiling_stats: dict,
    comm: MPI.Intracomm,
):
    """
    Core local BFS refinement loop for GPU adaptive clustering, run independently
    on each MPI rank. Accepts pre-sorted GPU data for the rank's partition and
    returns finalized output buffers plus counts.
    """
    mempool = None
    if IS_CUPY_BACKEND:
        import cupy as cp

        mempool = cp.get_default_memory_pool()

    # Allocate Output Buffers on CPU (Host) to save VRAM
    buffers, out_cap = allocate_output_buffers_host(
        MAX_CLUSTERS_EST,
        d,
        fd,
        store_hess=config.store_hess,
        store_affine_fit=config.store_affine_fits,
        no_gradient=config.requires_no_grads,
    )
    out_count = 0
    num_generated_pts = 0
    total_f_evals = 0
    level = 0

    is_terminated = False
    status = MPI.Status()
    local_load = int(CLUSTER_PTS.shape[0])

    if USE_WORK_STEALING:
        # Architecture State for Decentralized Work-Stealing
        global_loads = np.zeros(size, dtype=np.int64)
        global_loads[rank] = local_load

        # Pre-allocate 1-element arrays for C-level MPI state updates
        state_send_buf = np.empty(1, dtype=np.int64)
        state_recv_buf = np.empty(1, dtype=np.int64)

        def receive_and_update_state():
            """
            Helper function (async gossip receiver): Check for and process any incoming load updates from other ranks.
            """
            while comm.Iprobe(
                source=MPI.ANY_SOURCE, tag=TAG_STATE_BROADCAST, status=status
            ):
                src = status.Get_source()
                comm.Recv(state_recv_buf, source=src, tag=TAG_STATE_BROADCAST)
                global_loads[src] = state_recv_buf[0]

        def broadcast_state():
            """
            Helper function: Broadcast our current load to all other ranks.
            """
            state_send_buf[0] = local_load
            for r in range(size):
                if r != rank:
                    # 8 bytes is well within the MPI eager limit; this returns instantly.
                    comm.Send(state_send_buf, dest=r, tag=TAG_STATE_BROADCAST)

        # Broadcast initial state
        broadcast_state()
    else:

        def receive_and_update_state():
            pass

        def broadcast_state():
            pass

    # --- Decentralized Work-Stealing Main Loop ---
    while not is_terminated:
        num_clusters = len(CLUSTER_SIZES)
        local_load = int(CLUSTER_PTS.shape[0] if num_clusters > 0 else 0)

        if USE_WORK_STEALING:
            with CudaTimer("time_state_update", profiling_stats, PROFILE_TIMING):
                receive_and_update_state()
            global_loads[rank] = local_load

        # --- Phase 3: Idle / Stealing Phase ---
        if num_clusters == 0 or level > MAX_LEVELS:
            if not USE_WORK_STEALING:
                logger.info(
                    f"R{rank:02}: No clusters to process or exceeded max levels. Terminating."
                )
                is_terminated = True
                break

            with CudaTimer("time_idle", profiling_stats, PROFILE_TIMING):
                # A rank that hit MAX_LEVELS may still hold data but will never process
                # it. Ensure we advertise 0 so the termination condition works correctly.
                if local_load != 0:
                    local_load = 0
                    global_loads[rank] = 0
                    broadcast_state()
                if np.sum(global_loads) == 0:
                    # Heuristic termination: everyone claims to be empty.
                    # Drain any pending steal requests (sent before we went idle) so
                    # the requesting ranks are not stuck in their spin-wait forever.
                    while comm.Iprobe(
                        source=MPI.ANY_SOURCE, tag=TAG_STEAL_REQ, status=status
                    ):
                        req_src = status.Get_source()
                        comm.recv(source=req_src, tag=TAG_STEAL_REQ)
                        comm.send("TERMINATING", dest=req_src, tag=TAG_STEAL_DENY)
                    logger.info(f"R{rank:02}: Global loads sum to 0. Terminating.")
                    is_terminated = True
                    break

                # Get the rank with the highest load as our steal target
                target_rank = int(np.argmax(global_loads))

                # Gatekeeper Check
                if global_loads[target_rank] >= MIN_LOAD_THRESHOLD:
                    logger.debug(
                        f"R{rank:02}: Idle. Attempting steal from R{target_rank:02} "
                        f"(Load: {global_loads[target_rank]:,}) [my load: {local_load:,}]"
                    )
                    comm.send(rank, dest=target_rank, tag=TAG_STEAL_REQ)
                    _t_steal_wait = time.perf_counter()

                    waiting = True
                    while waiting:
                        with CudaTimer(
                            "time_state_update", profiling_stats, PROFILE_TIMING
                        ):
                            receive_and_update_state()

                        # Escape if the target has gone idle/terminated (load -> 0)
                        if global_loads[target_rank] == 0:
                            logger.debug(
                                f"R{rank:02}: Steal target R{target_rank:02} load dropped to 0 "
                                f"while waiting. Aborting steal."
                            )
                            if comm.Iprobe(source=target_rank, tag=TAG_STEAL_DENY):
                                comm.recv(source=target_rank, tag=TAG_STEAL_DENY)
                            elif comm.Iprobe(source=target_rank, tag=TAG_STEAL_GRANT):
                                meta = comm.recv(
                                    source=target_rank, tag=TAG_STEAL_GRANT
                                )
                                _X = np.empty((meta["n_pts"], d), dtype=np.float64)
                                _F = np.empty((meta["n_pts"], fd), dtype=np.float64)
                                comm.Recv(
                                    _X, source=target_rank, tag=TAG_STEAL_GRANT + 1
                                )
                                comm.Recv(
                                    _F, source=target_rank, tag=TAG_STEAL_GRANT + 2
                                )
                                if meta["has_g"]:
                                    _G = np.empty(
                                        (meta["n_pts"], fd, d), dtype=np.float64
                                    )
                                    comm.Recv(
                                        _G, source=target_rank, tag=TAG_STEAL_GRANT + 3
                                    )
                                _s = np.empty(meta["n_clusters"], dtype=np.int64)
                                comm.Recv(
                                    _s, source=target_rank, tag=TAG_STEAL_GRANT + 4
                                )
                            waiting = False
                            continue

                        # Check for steal response
                        if comm.Iprobe(source=target_rank, tag=TAG_STEAL_GRANT):
                            _t_recv = time.perf_counter()

                            meta = comm.recv(source=target_rank, tag=TAG_STEAL_GRANT)

                            X_host = np.empty((meta["n_pts"], d), dtype=np.float64)
                            F_host = np.empty((meta["n_pts"], fd), dtype=np.float64)
                            G_host = (
                                np.empty((meta["n_pts"], fd, d), dtype=np.float64)
                                if meta["has_g"]
                                else None
                            )
                            sizes_host = np.empty(meta["n_clusters"], dtype=np.int64)

                            comm.Recv(
                                X_host, source=target_rank, tag=TAG_STEAL_GRANT + 1
                            )
                            comm.Recv(
                                F_host, source=target_rank, tag=TAG_STEAL_GRANT + 2
                            )
                            if meta["has_g"]:
                                comm.Recv(
                                    G_host, source=target_rank, tag=TAG_STEAL_GRANT + 3  # type: ignore
                                )
                            comm.Recv(
                                sizes_host, source=target_rank, tag=TAG_STEAL_GRANT + 4
                            )

                            # Keep stolen data on the CPU
                            CLUSTER_PTS = X_host
                            CLUSTER_F = F_host
                            CLUSTER_G = G_host
                            CLUSTER_SIZES = sizes_host
                            CLUSTER_STARTS = np.r_[0, np.cumsum(CLUSTER_SIZES)[:-1]]

                            local_load = int(CLUSTER_PTS.shape[0])
                            global_loads[rank] = local_load
                            broadcast_state()

                            _t_wait = time.perf_counter() - _t_steal_wait
                            _t_transfer = time.perf_counter() - _t_recv
                            logger.info(
                                f"R{rank:02}: Steal OK from R{target_rank:02} | "
                                f"pts={local_load:,} clusters={len(CLUSTER_SIZES):,} | "
                                f"waited={_t_wait:.3f}s transfer={_t_transfer:.3f}s"
                            )
                            waiting = False

                            if PROFILE_TIMING:
                                profiling_stats["time_mpi_recv"] += _t_transfer

                        elif comm.Iprobe(source=target_rank, tag=TAG_STEAL_DENY):
                            deny_reason = comm.recv(
                                source=target_rank, tag=TAG_STEAL_DENY
                            )
                            _t_wait = time.perf_counter() - _t_steal_wait
                            logger.debug(
                                f"R{rank:02}: Steal denied by R{target_rank:02} "
                                f"(reason={deny_reason}, waited {_t_wait:.3f}s). Sleeping."
                            )
                            time.sleep(SLEEP_INTERVAL)
                            waiting = False
                else:
                    time.sleep(SLEEP_INTERVAL)

            # Loop back around to check termination condition and steal opportunities again.
            continue

        # --- Phase 1: The Compute Phase (Mid-Level) ---
        _t_level = time.perf_counter()
        if LOG_MEM_STATS:
            log_mem_stats(f"R{rank:02} Start: L{level:02}")

        profiling_stats["level_stats"].append(
            {
                "rank": rank,
                "level": int(level),
                "clusters_start": int(num_clusters),
                "size_min": int(float(np.min(CLUSTER_SIZES))),
                "size_mean": float(np.mean(CLUSTER_SIZES)),
                "size_median": int(float(np.median(CLUSTER_SIZES))),
                "size_max": int(float(np.max(CLUSTER_SIZES))),
            }
        )

        level_total_pts = int(np.sum(CLUSTER_SIZES))
        aug_sizes_cpu = CLUSTER_SIZES
        aug_cumsum_cpu = np.cumsum(aug_sizes_cpu)

        (
            next_level_X_list,
            next_level_F_list,
            next_level_G_list,
            next_level_sizes_list,
        ) = ([], [], [], [])

        num_batches = level_total_pts // BATCH_SIZE + 1
        start_cluster_idx = 0
        K_good = 0

        if LOG_MEM_STATS:
            log_mem_stats(f"R{rank:02} Start: L{level:02} Batching")

        for b in range(num_batches):
            logger.info(f"R{rank:02} L{level:02}: Processing Batch {b+1}/{num_batches}")

            if USE_WORK_STEALING:
                # --- Mid-Level Work Stealing ---
                while comm.Iprobe(
                    source=MPI.ANY_SOURCE, tag=TAG_STEAL_REQ, status=status
                ):
                    req_src = status.Get_source()
                    comm.recv(source=req_src, tag=TAG_STEAL_REQ)

                    unprocessed_clusters = num_clusters - start_cluster_idx

                    if unprocessed_clusters <= 1:
                        comm.send(
                            "MID_LEVEL_NO_CLUSTERS", dest=req_src, tag=TAG_STEAL_DENY
                        )
                        continue

                    unprocessed_sizes = CLUSTER_SIZES[start_cluster_idx:]
                    unprocessed_pts = int(np.sum(unprocessed_sizes))

                    if unprocessed_pts > MIN_LOAD_THRESHOLD:
                        _t_send = time.perf_counter()

                        rev_sizes_cpu = unprocessed_sizes[::-1]
                        rev_cumsum = np.cumsum(rev_sizes_cpu)
                        split_rev_idx = int(np.searchsorted(rev_cumsum, STEAL_SIZE))

                        give_clusters = split_rev_idx + 1
                        give_clusters = min(give_clusters, unprocessed_clusters - 1)

                        split_idx = num_clusters - give_clusters
                        pt_split = int(CLUSTER_STARTS[split_idx])

                        # Arrays are already on CPU, no need for _to_host
                        X_send = CLUSTER_PTS[pt_split:]
                        F_send = CLUSTER_F[pt_split:]
                        G_send = CLUSTER_G[pt_split:] if CLUSTER_G is not None else None
                        sizes_send = CLUSTER_SIZES[split_idx:]

                        meta = {
                            "n_pts": int(X_send.shape[0]),
                            "n_clusters": int(sizes_send.shape[0]),
                            "has_g": G_send is not None,
                        }
                        comm.send(meta, dest=req_src, tag=TAG_STEAL_GRANT)

                        comm.Send(
                            np.ascontiguousarray(X_send),
                            dest=req_src,
                            tag=TAG_STEAL_GRANT + 1,
                        )
                        comm.Send(
                            np.ascontiguousarray(F_send),
                            dest=req_src,
                            tag=TAG_STEAL_GRANT + 2,
                        )
                        if meta["has_g"]:
                            comm.Send(
                                np.ascontiguousarray(G_send),
                                dest=req_src,
                                tag=TAG_STEAL_GRANT + 3,
                            )
                        comm.Send(
                            np.ascontiguousarray(sizes_send),
                            dest=req_src,
                            tag=TAG_STEAL_GRANT + 4,
                        )

                        CLUSTER_PTS = CLUSTER_PTS[:pt_split]
                        CLUSTER_F = CLUSTER_F[:pt_split]
                        if CLUSTER_G is not None:
                            CLUSTER_G = CLUSTER_G[:pt_split]
                        CLUSTER_SIZES = CLUSTER_SIZES[:split_idx]
                        CLUSTER_STARTS = CLUSTER_STARTS[:split_idx]

                        aug_sizes_cpu = aug_sizes_cpu[:split_idx]
                        aug_cumsum_cpu = aug_cumsum_cpu[:split_idx]

                        num_clusters = split_idx
                        local_load = int(CLUSTER_PTS.shape[0])
                        global_loads[rank] = local_load
                        broadcast_state()

                        _t_send_elapsed = time.perf_counter() - _t_send
                        logger.info(
                            f"R{rank:02} L{level:02} B{b+1}: Granted mid-level steal to R{req_src:02} | "
                            f"gave {X_send.shape[0]:,} pts "
                            f"{sizes_send.shape[0]:,} clusters | "
                            f"send={_t_send_elapsed:.3f}s | "
                            f"remaining pts={local_load:,} clusters={num_clusters:,}"
                        )
                        if PROFILE_TIMING:
                            profiling_stats["time_mpi_send"] += _t_send_elapsed
                    else:
                        deny_reason = f"MID_LEVEL_LOAD ({unprocessed_pts:,} <= threshold {MIN_LOAD_THRESHOLD:,})"
                        comm.send(deny_reason, dest=req_src, tag=TAG_STEAL_DENY)

                with CudaTimer("time_state_update", profiling_stats, PROFILE_TIMING):
                    receive_and_update_state()

            target_aug_count = (b + 1) * BATCH_SIZE
            end_cluster_idx = int(np.searchsorted(aug_cumsum_cpu, target_aug_count))
            end_cluster_idx = min(end_cluster_idx + 1, num_clusters)
            if start_cluster_idx >= end_cluster_idx:
                break

            with CudaTimer("time_slicing", profiling_stats, PROFILE_TIMING):
                sizes_cpu = CLUSTER_SIZES[start_cluster_idx:end_cluster_idx]
                starts_local_cpu = CLUSTER_STARTS[start_cluster_idx:end_cluster_idx]
                slice_start = int(starts_local_cpu[0])
                slice_end = int(starts_local_cpu[-1] + sizes_cpu[-1])

                # --- MOVE ONLY THIS BATCH TO THE GPU ---
                pts = xp.asarray(CLUSTER_PTS[slice_start:slice_end])
                f_vals = xp.asarray(CLUSTER_F[slice_start:slice_end])
                g_vals = (
                    xp.asarray(CLUSTER_G[slice_start:slice_end])
                    if CLUSTER_G is not None
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

            with CudaTimer("time_error_calc", profiling_stats, PROFILE_TIMING):
                mus = xp.add.reduceat(X_batch, aug_starts) / aug_sizes[:, None]

                if oracle_fn is not None:
                    F_mus_val, G_mus_val = oracle_fn(mus)
                    f_mus = convert_to_2d_array(F_mus_val)
                    if config.requires_grad_centroids:
                        g_mus = convert_to_3d_array(G_mus_val)
                    else:
                        g_mus = None
                        del G_mus_val

                else:
                    assert f_fn
                    f_mus = convert_to_2d_array(f_fn(mus))
                    g_mus = (
                        convert_to_3d_array(g_fn(mus))
                        if config.requires_grad_centroids and g_fn
                        else None
                    )

                total_f_evals += mus.shape[0]

                # ==========================================
                # MATH PIPELINE STAGE 1
                # ==========================================
                is_good, E_cluster, surrogate_params = evaluate_cluster_errors(
                    config,
                    X_batch,
                    F_batch,
                    mus,
                    f_mus,
                    g_mus,
                    aug_starts,
                    aug_sizes,
                    aug_ids,
                    F_max_global,
                    TOL,
                )

            if LOG_MEM_STATS:
                log_mem_stats(
                    f" R{rank:02} L{level:02} B{b:02}/{num_batches:02}: After Error Metrics"
                )

            # 5. Handle Accepted ("Good") Clusters
            with CudaTimer("time_filter", profiling_stats, PROFILE_TIMING):
                good_idx_local = xp.nonzero(is_good)[0]
                K_good_batch = int(good_idx_local.shape[0])
                K_good += K_good_batch

            if K_good_batch > 0:
                with CudaTimer("time_good_Q", profiling_stats, PROFILE_TIMING):
                    good_mask = is_good[aug_ids]
                    X_good = X_batch[good_mask]
                    G_good = (
                        G_batch[good_mask]
                        if (CLUSTER_G is not None and G_batch is not None)
                        else None
                    )
                    good_counts = aug_sizes[good_idx_local]
                    dense_labels_good = repeats(xp.arange(K_good_batch), good_counts)

                    # ==========================================
                    # MATH PIPELINE STAGE 2
                    # ==========================================
                    Qs, Hs, radii, L_obs = compute_accepted_stats(
                        config,
                        X_good,
                        G_good,
                        mus[good_idx_local],
                        g_mus[good_idx_local] if g_mus is not None else None,
                        dense_labels_good,
                        good_counts,
                        Q_SHRINK,
                    )

                with CudaTimer("time_good_d2h", profiling_stats, PROFILE_TIMING):
                    end_out = out_count + K_good_batch
                    buffers, out_cap = adjust_buffer_capacity_host(
                        buffers,
                        out_count,
                        out_cap,
                        end_out,
                        config.store_hess,
                        config.store_affine_fits,
                        not (
                            config.requires_grad_everywhere
                            or config.requires_grad_centroids
                        ),
                    )

                    buffers["centroids"][out_count:end_out] = _to_host(
                        mus[good_idx_local]
                    )
                    buffers["f"][out_count:end_out] = _to_host(f_mus[good_idx_local])
                    buffers["Q"][out_count:end_out] = _to_host(Qs)
                    buffers["E_cluster"][out_count:end_out] = _to_host(
                        E_cluster[good_idx_local]
                    )
                    buffers["radii"][out_count:end_out] = _to_host(radii)
                    buffers["counts"][out_count:end_out] = _to_host(
                        good_counts.astype(xp.int64)
                    )

                    if config.requires_grad_centroids and g_mus is not None:
                        buffers["g"][out_count:end_out] = _to_host(
                            g_mus[good_idx_local]
                        )
                    if L_obs is not None:
                        buffers["L_obs"][out_count:end_out] = _to_host(L_obs)
                    if Hs is not None:
                        buffers["H"][out_count:end_out] = _to_host(Hs)
                    if "A" in surrogate_params and "b" in surrogate_params:
                        buffers["A_fit"][out_count:end_out] = _to_host(
                            surrogate_params["A"][good_idx_local]
                        )
                        buffers["b_fit"][out_count:end_out] = _to_host(
                            surrogate_params["b"][good_idx_local]
                        )
                    elif "H" in surrogate_params and "w" in surrogate_params and "b" in surrogate_params:
                        buffers["H_fit"][out_count:end_out] = _to_host(
                            surrogate_params["H"][good_idx_local]
                        )
                        buffers["w_fit"][out_count:end_out] = _to_host(
                            surrogate_params["w"][good_idx_local]
                        )
                        buffers["b_fit"][out_count:end_out] = _to_host(
                            surrogate_params["b"][good_idx_local]
                        )

                    out_count = end_out

            # 6. Handle Rejected ("Bad") Clusters -> Split
            with CudaTimer("time_filter", profiling_stats, PROFILE_TIMING):
                bad_idx_local = xp.nonzero(~is_good)[0]
                n_bad = int(bad_idx_local.shape[0])

            if n_bad > 0:
                with CudaTimer("time_bad_split", profiling_stats, PROFILE_TIMING):
                    bad_mask = (~is_good)[aug_ids]
                    X_bad = X_batch[bad_mask]
                    F_bad = F_batch[bad_mask]
                    G_bad = (
                        G_batch[bad_mask]
                        if (CLUSTER_G is not None and G_batch is not None)
                        else None
                    )
                    bad_counts = aug_sizes[bad_idx_local]

                    # --- Dynamically augment bad clusters before splitting ---
                    target_bad_counts = xp.where(
                        bad_counts < 2 * MIN_PTS_PER_CLUSTER,
                        xp.maximum(2 * bad_counts, 2 * MIN_PTS_PER_CLUSTER),
                        bad_counts,
                    )

                    total_target_pts = int(xp.sum(target_bad_counts))

                    X_bad_aug = xp.empty((total_target_pts, d), dtype=xp.float64)
                    F_bad_aug = xp.empty((total_target_pts, fd), dtype=xp.float64)
                    G_bad_aug = (
                        xp.empty((total_target_pts, fd, d), dtype=xp.float64)
                        if G_bad is not None
                        else None
                    )

                    target_starts = xp.r_[0, xp.cumsum(target_bad_counts)[:-1]]
                    aug_dense_labels_bad = repeats(xp.arange(n_bad), target_bad_counts)

                    # Map indices to interleave original and synthetic points correctly
                    pt_idxs_in_cluster = (
                        xp.arange(total_target_pts, dtype=xp.int32)
                        - target_starts[aug_dense_labels_bad]
                    )
                    is_orig = pt_idxs_in_cluster < bad_counts[aug_dense_labels_bad]

                    # Copy original points over
                    X_bad_aug[is_orig] = X_bad
                    F_bad_aug[is_orig] = F_bad
                    if G_bad is not None and G_bad_aug is not None:
                        G_bad_aug[is_orig] = G_bad

                    is_synth = ~is_orig
                    num_synth = int(xp.sum(is_synth))

                    if num_synth > 0:
                        num_generated_pts += num_synth
                        synth_c_ids = aug_dense_labels_bad[is_synth]
                        rng = xp.random.default_rng(SEED + level + b + rank * 1000)

                        bad_starts = xp.r_[0, xp.cumsum(bad_counts)[:-1]]
                        c_starts_synth = bad_starts[synth_c_ids]
                        c_counts_synth = bad_counts[synth_c_ids]

                        rand1 = (rng.random(num_synth) * c_counts_synth).astype(
                            xp.int64
                        )
                        rand2 = (rng.random(num_synth) * (c_counts_synth - 1)).astype(
                            xp.int64
                        ) + 1

                        idx1 = c_starts_synth + rand1
                        idx2 = c_starts_synth + (rand1 + rand2) % c_counts_synth

                        p1 = X_bad[idx1]
                        p2 = X_bad[idx2]
                        alpha = rng.random((num_synth, 1)).astype(xp.float64)
                        X_synth_vals = p1 * alpha + p2 * (1.0 - alpha)

                        if SAVE_ALL_GEN_CLUSTER_DATA is True:
                            save_generated_data(X_synth_vals, f'train_data_tol_{TOL}_d{d}')

                        X_bad_aug[is_synth] = X_synth_vals

                        # Evaluate F and G for synthetic points
                        if oracle_fn is not None:
                            if G_bad is None:
                                assert f_fn
                                F_syn = f_fn(X_synth_vals)
                                F_bad_aug[is_synth] = convert_to_2d_array(F_syn)
                            else:
                                assert G_bad_aug is not None
                                F_syn, G_syn = oracle_fn(X_synth_vals)
                                F_bad_aug[is_synth] = convert_to_2d_array(F_syn)
                                G_bad_aug[is_synth] = convert_to_3d_array(G_syn)
                        else:
                            assert f_fn
                            F_bad_aug[is_synth] = convert_to_2d_array(
                                f_fn(X_synth_vals)
                            )
                            if g_fn and G_bad_aug is not None:
                                G_bad_aug[is_synth] = convert_to_3d_array(
                                    g_fn(X_synth_vals)
                                )

                        total_f_evals += num_synth

                    # ==========================================
                    # MATH PIPELINE STAGE 3
                    # ==========================================
                    bad_surrogate_params = {
                        k: v[bad_idx_local] for k, v in surrogate_params.items()
                    }
                    sort_order = split_rejected_clusters(
                        config,
                        X_bad_aug,
                        F_bad_aug,
                        G_bad_aug,
                        mus[bad_idx_local],
                        f_mus[bad_idx_local],
                        aug_dense_labels_bad,
                        target_bad_counts,
                        MIN_PTS_PER_CLUSTER,
                        bad_surrogate_params,
                    )

                with CudaTimer("time_bad_d2h", profiling_stats, PROFILE_TIMING):
                    # Append the sorted augmented arrays
                    next_level_X_list.append(_to_host(X_bad_aug[sort_order]))
                    next_level_F_list.append(_to_host(F_bad_aug[sort_order]))
                    if config.requires_grad_everywhere and G_bad_aug is not None:
                        next_level_G_list.append(_to_host(G_bad_aug[sort_order]))

                    # Split the new target counts: TODO: understand
                    half = target_bad_counts // 2
                    other = target_bad_counts - half
                    new_sz = xp.zeros(2 * n_bad, dtype=xp.int64)
                    new_sz[0::2] = half
                    new_sz[1::2] = other
                    next_level_sizes_list.append(_to_host(new_sz))

            start_cluster_idx = end_cluster_idx
            with CudaTimer("time_gc", profiling_stats, PROFILE_TIMING):
                del X_batch, F_batch, G_batch
                if mempool is not None:
                    mempool.free_all_blocks()

        # --- Phase 2: The Level Boundary (Transition & Steal Response) ---
        with CudaTimer("time_transition", profiling_stats, PROFILE_TIMING):
            if len(next_level_sizes_list) > 0:
                next_X = np.concatenate(next_level_X_list)
                next_F = np.concatenate(next_level_F_list)
                next_G = (
                    np.concatenate(next_level_G_list)
                    if config.requires_grad_everywhere
                    else None
                )
                next_sizes = np.concatenate(next_level_sizes_list)

                valid = next_sizes > 0
                if not np.all(valid):
                    keep = np.repeat(valid, next_sizes)
                    next_X, next_F, next_sizes = (
                        next_X[keep],
                        next_F[keep],
                        next_sizes[valid],
                    )
                    if next_G is not None:
                        next_G = next_G[keep]
            else:
                logger.info(
                    f"R{rank:02} L{level:02}: No bad clusters to refine, terminating."
                )

                # Prep empty arrays so we gracefully loop back up to Phase 3
                next_X = np.array([])
                next_F = np.array([])
                next_G = None
                next_sizes = np.array([])

            total_next_pts = np.sum(next_sizes) if len(next_sizes) > 0 else 0
            local_load = int(total_next_pts)

            if USE_WORK_STEALING:
                global_loads[rank] = local_load
                broadcast_state()

                while comm.Iprobe(
                    source=MPI.ANY_SOURCE, tag=TAG_STEAL_REQ, status=status
                ):
                    req_src = status.Get_source()
                    comm.recv(source=req_src, tag=TAG_STEAL_REQ)

                    if total_next_pts > MIN_LOAD_THRESHOLD and len(next_sizes) > 1:
                        _t_send = time.perf_counter()

                        rev_cumsum = np.cumsum(next_sizes[::-1])
                        split_rev_idx = np.searchsorted(rev_cumsum, STEAL_SIZE)
                        split_idx = len(next_sizes) - 1 - split_rev_idx
                        split_idx = max(1, split_idx)

                        pt_split = int(np.sum(next_sizes[:split_idx]))

                        X_send = next_X[pt_split:]
                        F_send = next_F[pt_split:]
                        G_send = next_G[pt_split:] if next_G is not None else None
                        sizes_send = next_sizes[split_idx:]

                        meta = {
                            "n_pts": int(X_send.shape[0]),
                            "n_clusters": int(sizes_send.shape[0]),
                            "has_g": G_send is not None,
                        }
                        comm.send(meta, dest=req_src, tag=TAG_STEAL_GRANT)

                        comm.Send(
                            np.ascontiguousarray(X_send),
                            dest=req_src,
                            tag=TAG_STEAL_GRANT + 1,
                        )
                        comm.Send(
                            np.ascontiguousarray(F_send),
                            dest=req_src,
                            tag=TAG_STEAL_GRANT + 2,
                        )
                        if meta["has_g"]:
                            comm.Send(
                                np.ascontiguousarray(G_send),
                                dest=req_src,
                                tag=TAG_STEAL_GRANT + 3,
                            )
                        comm.Send(
                            np.ascontiguousarray(sizes_send),
                            dest=req_src,
                            tag=TAG_STEAL_GRANT + 4,
                        )

                        next_X = next_X[:pt_split]
                        next_F = next_F[:pt_split]
                        if next_G is not None:
                            next_G = next_G[:pt_split]
                        next_sizes = next_sizes[:split_idx]
                        total_next_pts = np.sum(next_sizes)

                        local_load = int(total_next_pts)
                        global_loads[rank] = local_load
                        broadcast_state()

                        _t_send_elapsed = time.perf_counter() - _t_send
                        logger.info(
                            f"R{rank:02} L{level:02}: Granted steal to R{req_src:02} | "
                            f"gave {X_send.shape[0]:,} pts "
                            f"{sizes_send.shape[0]:,} clusters | "
                            f"send={_t_send_elapsed:.3f}s | "
                            f"remaining pts={total_next_pts:,} clusters={len(next_sizes):,}"
                        )

                        if PROFILE_TIMING:
                            profiling_stats["time_mpi_send"] += _t_send_elapsed
                    else:
                        if total_next_pts <= MIN_LOAD_THRESHOLD:
                            deny_reason = f"LOAD ({total_next_pts:,} <= threshold {MIN_LOAD_THRESHOLD:,})"
                        else:
                            deny_reason = f"STEAL_SIZE (only {len(next_sizes)} cluster(s), need > 1)"
                        comm.send(deny_reason, dest=req_src, tag=TAG_STEAL_DENY)
                        logger.debug(
                            f"R{rank:02} L{level:02}: Denied steal to R{req_src:02}: {deny_reason}"
                        )

            # Remove xp.asarray - keep the next level queued up in CPU RAM
            if len(next_sizes) > 0:
                CLUSTER_PTS = next_X
                CLUSTER_F = next_F
                CLUSTER_G = next_G
                CLUSTER_SIZES = next_sizes
                CLUSTER_STARTS = np.r_[0, np.cumsum(CLUSTER_SIZES)[:-1]]
            else:
                CLUSTER_PTS = np.array([])
                CLUSTER_SIZES = np.array([])

        if profiling_stats["level_stats"]:
            profiling_stats["level_stats"][-1].update(
                {
                    "clusters_level_end": int(len(CLUSTER_SIZES)),
                    "good_clusters": int(K_good),
                    "num_pts_next_level": int(CLUSTER_PTS.shape[0]),
                }
            )

        if LOG_MEM_STATS:
            log_mem_stats(f"R{rank:02} End: L{level:02} Transition")

        with CudaTimer("time_gc", profiling_stats, PROFILE_TIMING):
            del (
                next_level_X_list,
                next_level_F_list,
                next_level_G_list,
                next_level_sizes_list,
                next_X,
                next_F,
                next_G,
                next_sizes,
            )
            gc.collect()

        logger.info(
            f"R{rank:02} L{level:02}: Clusters (start): {num_clusters:,}, "
            f"Clusters (end): {len(CLUSTER_SIZES):,}, "
            f"Good Clusters: {K_good:,}, "
            f"Num Pts (next level): {CLUSTER_PTS.shape[0]:,}"
        )
        if len(CLUSTER_SIZES) > 0:
            logger.info(
                f"R{rank:02} L{level:02}: Cluster Sizes - min: {int(CLUSTER_SIZES.min())}, "
                f"mean: {CLUSTER_SIZES.mean():.1f}, median: {int(np.median(CLUSTER_SIZES))}, "
                f"max: {int(CLUSTER_SIZES.max())}"
            )
        level_runtime = time.perf_counter() - _t_level
        logger.info(f"Runtime - R{rank:02} L{level:02}: {level_runtime:.3f}s")

        if profiling_stats["level_stats"]:
            profiling_stats["level_stats"][-1].update({"runtime_sec": level_runtime})
        level += 1

    return buffers, out_count, total_f_evals, num_generated_pts, level


# ---------------------------------------------------------------------------
# Public entry point (MPI-aware)
# ---------------------------------------------------------------------------
def adaptive_clustering(
    X: Array,
    *,
    f_fn: Optional[Callable[[Array], Array]] = None,
    g_fn: Optional[Callable[[Array], Array]] = None,
    oracle_fn: Optional[Callable[[Array], tuple[Array, Array]]] = None,
    TOL: float,
    ERROR_EVAL_MODE: SurrogateMode,
    ERROR_METRIC: Literal["L2", "Linf-global", "Linf-local"] = "Linf-global",
    INITIAL_CLUSTERING_METHOD: Literal[
        "kmeans", "knndbscan", "uniform_grid"
    ] = "kmeans",
    NUM_INITIAL_CLUSTERS: int = 100,
    MIN_PTS_PER_CLUSTER: int = 100,
    MAX_CLUSTERS_EST: int = 100_000,
    SEED: int = 0,
    Q_SHRINK: float = 0.05,
    MAX_LEVELS: int = 50,
    BATCH_SIZE: int = 10_000_000,
    USE_WORK_STEALING: bool = True,
    STORE_HESSIAN_PER_CLUSTER: bool = False,
    STORE_AFFINE_FIT_PER_CLUSTER: bool = False,
    LOG_MEM_STATS: bool = False,
    PROFILE_TIMING: bool = False,
    SAVE_ALL_GEN_TRAINING_DATA: bool = True,
) -> dict[str, Any] | None:
    """
    Adaptive Clustering for Vector-Valued Functions (GPU Accelerated).
    """
    if ERROR_EVAL_MODE in (
        SurrogateMode.AFFINE_COV_HESS,
        SurrogateMode.AFFINE_COV_NO_HESS,
    ):
        if not STORE_AFFINE_FIT_PER_CLUSTER:
            STORE_AFFINE_FIT_PER_CLUSTER = True
            logger.warning(
                f"Selected mode {ERROR_EVAL_MODE.value} requires storing affine fit parameters. Overriding STORE_AFFINE_FIT_PER_CLUSTER to True."
            )

    if USE_WORK_STEALING:
        if MAX_LEVELS <= 100:
            _safe_MAX_LEVELS = 1000
            logger.warning(
                f"Work stealing enabled but MAX_LEVELS is only {MAX_LEVELS}. Overriding MAX_LEVELS to {_safe_MAX_LEVELS} to ensure sufficient stealing opportunities across levels."
            )
            MAX_LEVELS = _safe_MAX_LEVELS

    config = ClusteringConfig(
        mode=ERROR_EVAL_MODE,
        error_metric=ERROR_METRIC,
        store_hess=STORE_HESSIAN_PER_CLUSTER,
        store_affine_fits=STORE_AFFINE_FIT_PER_CLUSTER,
    )
    NO_GRADS = config.requires_no_grads
    if NO_GRADS is False and (g_fn is None and oracle_fn is None):
        raise ValueError(
            f"Selected mode {ERROR_EVAL_MODE.value} requires gradient information. Please provide g_fn or oracle_fn."
        )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
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

        if SAVE_ALL_GEN_TRAINING_DATA is True:
            _, DIM = X.shape
            save_generated_data(X, f'train_data_tol_{TOL}_d{DIM}')

        with CudaTimer("time_init", profiling_stats, PROFILE_TIMING):
            X0_gpu = xp.asarray(X, dtype=xp.float64)
            N0, d_bcast = X0_gpu.shape
            profiling_stats["N0"], profiling_stats["total_f_evals"] = N0, N0

            if oracle_fn:
                F_out, G_out = oracle_fn(X0_gpu)
                F0_gpu = convert_to_2d_array(F_out)
                if NO_GRADS:
                    del G_out
                    G0_gpu = None
                else:
                    G0_gpu = convert_to_3d_array(G_out)
            else:
                assert f_fn
                F0_gpu = convert_to_2d_array(f_fn(X0_gpu))
                G0_gpu = (
                    convert_to_3d_array(g_fn(X0_gpu)) if not NO_GRADS and g_fn else None
                )
            fd_bcast = F0_gpu.shape[1]

            F_max_global_host = _to_host(xp.max(xp.abs(F0_gpu), axis=0))

            if INITIAL_CLUSTERING_METHOD == "kmeans":
                if IS_CUPY_BACKEND:
                    from cuml.cluster import KMeans as cuKMeans

                    km = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
                    labels0 = km.fit_predict(X0_gpu)
                else:
                    from sklearn.cluster import KMeans

                    km = KMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
                    labels0 = km.fit_predict(_to_host(X0_gpu))
                labels0 = _to_host(labels0).astype(np.int64)

            elif INITIAL_CLUSTERING_METHOD == "knndbscan":
                from knndbscan import knn_dbscan

                labels0 = _to_host(
                    knn_dbscan(
                        X0_gpu,
                        eps=0.5,
                        min_samples=MIN_PTS_PER_CLUSTER,
                        device="gpu" if IS_CUPY_BACKEND else "cpu",
                    )
                ).astype(np.int64)

            elif INITIAL_CLUSTERING_METHOD == "uniform_grid":
                labels0, _ = uniform_grid_cluster_labels(
                    _to_host(X0_gpu), NUM_INITIAL_CLUSTERS
                )
                labels0 = labels0.astype(np.int64)

            else:
                raise ValueError(f"Invalid: {INITIAL_CLUSTERING_METHOD}")

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

    with CudaTimer("time_scatter", profiling_stats, PROFILE_TIMING):
        F_max_global_host = comm.bcast(F_max_global_host, root=0)
        d_bcast, fd_bcast = comm.bcast((d_bcast, fd_bcast), root=0)

        META_TAG = 999
        X_TAG = 1000
        F_TAG = 1001
        G_TAG = 1002
        SIZES_TAG = 1003

        if rank == 0:
            current_idx = 0
            for r in range(size):
                num_pts_in_rank = int(np.sum(counts_chunks[r]))
                end_idx = current_idx + num_pts_in_rank
                chunk_X = X0_sorted[current_idx:end_idx]
                chunk_F = F0_sorted[current_idx:end_idx]
                chunk_G = (
                    G0_sorted[current_idx:end_idx] if G0_sorted is not None else None
                )
                chunk_sizes = counts_chunks[r].astype(np.int64)

                if r == 0:
                    local_dict = {
                        "X": chunk_X,
                        "F": chunk_F,
                        "G": chunk_G,
                        "sizes": chunk_sizes,
                    }
                else:
                    logger.info(
                        f"R00: Sending initial data chunk to R{r:02} | "
                        f"pts={chunk_X.shape[0]:,} clusters={chunk_sizes.shape[0]:,}"
                    )
                    meta_arr = np.array(
                        [
                            chunk_X.shape[0],
                            chunk_sizes.shape[0],
                            1 if chunk_G is not None else 0,
                        ],
                        dtype=np.int64,
                    )
                    comm.Send(meta_arr, dest=r, tag=META_TAG)
                    comm.Send(np.ascontiguousarray(chunk_X), dest=r, tag=X_TAG)
                    comm.Send(np.ascontiguousarray(chunk_F), dest=r, tag=F_TAG)
                    if chunk_G is not None:
                        comm.Send(np.ascontiguousarray(chunk_G), dest=r, tag=G_TAG)
                    comm.Send(np.ascontiguousarray(chunk_sizes), dest=r, tag=SIZES_TAG)
                current_idx = end_idx
        else:
            logger.info(f"R{rank:02}: Waiting to receive initial data chunk from R00.")
            meta_arr = np.empty(3, dtype=np.int64)
            comm.Recv(meta_arr, source=0, tag=META_TAG)

            n_pts, n_clusters, has_g = meta_arr[0], meta_arr[1], meta_arr[2] == 1
            chunk_X = np.empty((n_pts, d_bcast), dtype=np.float64)
            chunk_F = np.empty((n_pts, fd_bcast), dtype=np.float64)
            chunk_G = (
                np.empty((n_pts, fd_bcast, d_bcast), dtype=np.float64)
                if has_g
                else None
            )
            chunk_sizes = np.empty(n_clusters, dtype=np.int64)

            comm.Recv(chunk_X, source=0, tag=X_TAG)
            comm.Recv(chunk_F, source=0, tag=F_TAG)
            if has_g and chunk_G is not None:
                comm.Recv(chunk_G, source=0, tag=G_TAG)
            comm.Recv(chunk_sizes, source=0, tag=SIZES_TAG)
            local_dict = {
                "X": chunk_X,
                "F": chunk_F,
                "G": chunk_G,
                "sizes": chunk_sizes,
            }

    # KEEP DATA ON THE CPU (Host RAM)
    local_X = local_dict["X"]
    local_F = local_dict["F"]
    local_G = local_dict["G"]
    local_sizes = local_dict["sizes"]
    local_starts = np.r_[0, np.cumsum(local_sizes)[:-1]]
    F_max_global = xp.asarray(F_max_global_host, dtype=xp.float64)

    buffers, out_count, local_f_evals, local_gen_pts, last_level = (
        _run_local_refinement(
            config=config,
            CLUSTER_PTS=local_X,
            CLUSTER_F=local_F,
            CLUSTER_G=local_G,
            CLUSTER_SIZES=local_sizes,
            CLUSTER_STARTS=local_starts,
            F_max_global=F_max_global,
            d=d_bcast,
            fd=fd_bcast,
            f_fn=f_fn,
            g_fn=g_fn,
            oracle_fn=oracle_fn,
            TOL=TOL,
            MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
            MAX_CLUSTERS_EST=max(1, MAX_CLUSTERS_EST // max(1, size)),
            SEED=SEED,
            Q_SHRINK=Q_SHRINK,
            MAX_LEVELS=MAX_LEVELS,
            BATCH_SIZE=BATCH_SIZE,
            USE_WORK_STEALING=USE_WORK_STEALING,
            LOG_MEM_STATS=LOG_MEM_STATS,
            PROFILE_TIMING=PROFILE_TIMING,
            SAVE_ALL_GEN_CLUSTER_DATA=SAVE_ALL_GEN_TRAINING_DATA,
            rank=rank,
            size=size,
            profiling_stats=profiling_stats,
            comm=comm,
        )
    )

    local_result_host = {k: buffers[k][:out_count] for k in buffers}
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

    with CudaTimer("time_gather", profiling_stats, PROFILE_TIMING):
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
        if SAVE_ALL_GEN_TRAINING_DATA is True:
            combine_generated_data(f'train_data_tol_{TOL}_d{d_bcast}')

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
    TOL = 0.02
    PROB_LIMS = (-1.0, 3)
    N_TRAIN = get_N_TRAIN(DIM)
    NUM_INITIAL_CLUSTERS = get_NUM_INITIAL_CLUSTERS(DIM)
    MIN_PTS_PER_CLUSTER = get_MIN_PTS_PER_CLUSTER(DIM)
    USE_WORK_STEALING = True
    ERROR_METRIC = "L2"
    INITIAL_CLUSTERING_METHOD = "kmeans"
    ERROR_EVAL_MODE = SurrogateMode.AFFINE_COV_HESS

    OUTPUT_DIR = RESULTS_DIR / "adaptive_clustering_experimental_v2"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_fpath = (
        OUTPUT_DIR
        / f"mpi_sz{_size}_rosen{A_ROSENBROCK}_ntrain{N_TRAIN}_dim{DIM}_tol{TOL}_err{ERROR_EVAL_MODE.value}_metric{ERROR_METRIC}_init{INITIAL_CLUSTERING_METHOD}_nclust{NUM_INITIAL_CLUSTERS}_minpts{MIN_PTS_PER_CLUSTER}_steal{USE_WORK_STEALING}.png"
    )

    if _rank == 0:
        xp.random.seed(0)
        X = xp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(N_TRAIN, DIM))
    else:
        X = xp.empty((0, DIM), dtype=xp.float64)

    result = adaptive_clustering(
        X=X,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        TOL=TOL,
        ERROR_EVAL_MODE=ERROR_EVAL_MODE,
        ERROR_METRIC=ERROR_METRIC,
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        LOG_MEM_STATS=True,
        PROFILE_TIMING=True,
        USE_WORK_STEALING=USE_WORK_STEALING,
        MAX_LEVELS=200,
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
        "tol": TOL,
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
        title = f"Rosenbrock (a={A_ROSENBROCK}) (Tol={TOL}, ErrMode={ERROR_EVAL_MODE.value}, Metric={ERROR_METRIC}, N_init={NUM_INITIAL_CLUSTERS}, MinPts={MIN_PTS_PER_CLUSTER}, Steal={USE_WORK_STEALING}) (N0={N_TRAIN:,}, N_clusters={n_centroids:,})"
        plot_adaptive_clustering_results(
            centroids=_to_host(result["centroids"]),
            Q=_to_host(result["Q"]),
            radii=_to_host(result["radii"]),
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
