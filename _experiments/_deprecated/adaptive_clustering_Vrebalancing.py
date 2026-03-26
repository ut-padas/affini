import gc
import logging
import time
from typing import Any, Callable, Literal, Optional

import numpy as np
from mpi4py import MPI
from sklearn.cluster import KMeans

from src import IS_CUPY_BACKEND, Array, _get_mpi_rank, _get_mpi_size, _to_host, xp
from src.adaptive_clustering_utils import (
    CudaTimer,
    adjust_buffer_capacity_host,
    allocate_output_buffers_host,
    convert_to_2d_array,
    convert_to_3d_array,
    get_n_smallest_per_block,
    run_fused_metrics,
    run_fused_q_dist,
    run_fused_score,
    run_fused_stats,
    run_fused_taylor_comp_abs,
    run_fused_taylor_metric,
    scatter_maximum_1d,
)
from src.logging_utils import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)
TINY_EPS = 1e-12


def _rebalance_workload(comm, next_X, next_F, next_G, next_sizes, d, fd, BATCH_SIZE):
    """
    Synchronous load balancing with threshold checks.
    """
    mpi_size = comm.Get_size()
    rank = comm.Get_rank()

    # 1. CHEAP MPI CALL: Gather workload stats
    local_pts = len(next_X)
    all_pts = comm.allgather(local_pts)

    total_pts = sum(all_pts)
    avg_pts = total_pts / mpi_size
    max_pts = max(all_pts)

    # 2. THRESHOLD GATES
    if total_pts == 0:
        return next_X, next_F, next_G, next_sizes

    # The "Long Tail" Bypass
    if total_pts < BATCH_SIZE:
        return next_X, next_F, next_G, next_sizes

    # The Imbalance Ratio Check (Only rebalance if max is 1.5x the average)
    IMBALANCE_TOLERANCE = 1.5
    if max_pts <= IMBALANCE_TOLERANCE * avg_pts:
        return next_X, next_F, next_G, next_sizes

    # --- EXPENSIVE MPI CALLS BEGIN HERE ---
    if rank == 0:
        logger.info(
            f"R00: Severe imbalance detected (Max: {max_pts:,}, Avg: {avg_pts:,.1f}). Global rebalancing..."
        )

    # Gather all sizes to deterministically partition
    all_sizes = comm.allgather(next_sizes)
    global_sizes = (
        np.concatenate(all_sizes)
        if any(len(s) > 0 for s in all_sizes)
        else np.array([], dtype=np.int64)
    )

    # Assign clusters to ranks to evenly distribute points
    target_pts = total_pts / mpi_size
    assignments = np.zeros(len(global_sizes), dtype=int)
    current_pts, p_idx = 0, 0

    for i, sz in enumerate(global_sizes):
        assignments[i] = p_idx
        current_pts += sz
        if current_pts >= target_pts and p_idx < mpi_size - 1:
            p_idx += 1
            current_pts = 0

    # Map my local clusters to their new destination ranks
    global_offsets = np.r_[0, np.cumsum([len(s) for s in all_sizes])[:-1]]
    local_start_idx = global_offsets[rank]
    my_destinations = assignments[local_start_idx : local_start_idx + len(next_sizes)]

    # Pack data
    send_data = {i: {"X": [], "F": [], "G": [], "sizes": []} for i in range(mpi_size)}
    local_c_starts = np.r_[0, np.cumsum(next_sizes)[:-1]] if len(next_sizes) > 0 else []

    for local_c_idx, dest_rank in enumerate(my_destinations):
        sz = next_sizes[local_c_idx]
        start = local_c_starts[local_c_idx]
        end = start + sz
        send_data[dest_rank]["sizes"].append(sz)
        if sz > 0:
            send_data[dest_rank]["X"].append(next_X[start:end])
            send_data[dest_rank]["F"].append(next_F[start:end])
            send_data[dest_rank]["G"].append(next_G[start:end])

    # Format for sending
    for i in range(mpi_size):
        send_data[i]["sizes"] = np.array(send_data[i]["sizes"], dtype=np.int64)  # type: ignore
        send_data[i]["X"] = (  # type: ignore
            np.concatenate(send_data[i]["X"], axis=0)
            if send_data[i]["X"]
            else np.empty((0, d), dtype=np.float64)
        )
        send_data[i]["F"] = (  # type: ignore
            np.concatenate(send_data[i]["F"], axis=0)
            if send_data[i]["F"]
            else np.empty((0, fd), dtype=np.float64)
        )
        send_data[i]["G"] = (  # type: ignore
            np.concatenate(send_data[i]["G"], axis=0)
            if send_data[i]["G"]
            else np.empty((0, fd, d), dtype=np.float64)
        )

    # Exchange data
    send_list = [send_data[i] for i in range(mpi_size)]
    recv_list = comm.alltoall(send_list)

    # Reassemble received data
    new_X = (
        np.concatenate([d["X"] for d in recv_list if len(d["X"]) > 0], axis=0)
        if any(len(d["X"]) > 0 for d in recv_list)
        else np.empty((0, d), dtype=np.float64)
    )
    new_F = (
        np.concatenate([d["F"] for d in recv_list if len(d["F"]) > 0], axis=0)
        if any(len(d["F"]) > 0 for d in recv_list)
        else np.empty((0, fd), dtype=np.float64)
    )
    new_G = (
        np.concatenate([d["G"] for d in recv_list if len(d["G"]) > 0], axis=0)
        if any(len(d["G"]) > 0 for d in recv_list)
        else np.empty((0, fd, d), dtype=np.float64)
    )
    new_sizes = (
        np.concatenate([d["sizes"] for d in recv_list if len(d["sizes"]) > 0])
        if any(len(d["sizes"]) > 0 for d in recv_list)
        else np.empty(0, dtype=np.int64)
    )

    return new_X, new_F, new_G, new_sizes


# ---------------------------------------------------------------------------
# Local BFS refinement loop (shared by each MPI rank)
# ---------------------------------------------------------------------------
def _run_local_refinement(
    comm,
    # Initial sorted GPU arrays for this rank's partition
    CLUSTER_PTS,  # (N_local, d)
    CLUSTER_F,  # (N_local, fd)
    CLUSTER_G,  # (N_local, fd, d)
    CLUSTER_SIZES,  # (K_local,)
    CLUSTER_STARTS,  # (K_local,)
    F_max_global,  # (fd,)  frozen global scale broadcast from rank 0
    # Dimensions
    d: int,
    fd: int,
    # Oracles
    f_fn,
    g_fn,
    oracle_fn,
    # Config
    TOL: float,
    ERROR_METRIC: str,
    MIN_PTS_PER_CLUSTER: int,
    MAX_CLUSTERS_EST: int,
    SEED: int,
    Q_SHRINK: float,
    MAX_LEVELS: int,
    BATCH_SIZE: int,
    CLUSTERING_MATRIX: str,
    IGNORE_GRADIENT: bool,
    STORE_HESSIAN_SQRT_PER_CLUSTER: bool,
    STORE_TREE_DATA: bool,
    LOG_MEM_STATS: bool,
    PROFILE_TIMING: bool,
    rank: int,
    profiling_stats: dict,
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

    num_generated_pts = 0
    total_f_evals = 0
    tree_data: dict = {}

    # Allocate Output Buffers on CPU (Host) to save VRAM
    buffers, out_cap = allocate_output_buffers_host(
        MAX_CLUSTERS_EST, d, fd, STORE_HESSIAN_SQRT_PER_CLUSTER
    )
    out_count = 0

    # --- BFS Loop ---
    level = 0
    K_good = 0
    while level <= MAX_LEVELS:
        _t_level = time.perf_counter()
        if LOG_MEM_STATS:
            log_mem_stats(f"R{rank:02} Start: L{level:02}")

        num_clusters = len(CLUSTER_SIZES)

        # --- NEW GLOBAL EXIT CHECK ---
        # 1 if this rank has work, 0 if it is idle
        local_has_work = 1 if num_clusters > 0 else 0
        global_has_work = comm.allreduce(local_has_work, op=MPI.MAX)

        if global_has_work == 0:
            logger.info(
                f"R{rank:02} L{level:02}: Global workload empty, terminating BFS."
            )
            break
        # -----------------------------

        # Collect Level Stats (Guarded against empty arrays)
        if num_clusters > 0:
            s_min = int(float(xp.min(CLUSTER_SIZES)))
            s_mean = float(xp.mean(CLUSTER_SIZES))
            s_med = int(float(xp.median(CLUSTER_SIZES)))
            s_max = int(float(xp.max(CLUSTER_SIZES)))
        else:
            s_min, s_mean, s_med, s_max = 0, 0.0, 0, 0

        profiling_stats["level_stats"].append(
            {
                "rank": rank,
                "level": int(level),
                "clusters_start": int(num_clusters),
                "size_min": s_min,
                "size_mean": s_mean,
                "size_median": s_med,
                "size_max": s_max,
            }
        )

        # Calculate augmentation needs
        with CudaTimer("time_find_req_synth", profiling_stats, PROFILE_TIMING):
            level_augmented_sizes = xp.maximum(CLUSTER_SIZES, MIN_PTS_PER_CLUSTER)
            level_total_aug_pts = int(xp.sum(level_augmented_sizes))

        aug_sizes_cpu = _to_host(level_augmented_sizes)
        aug_cumsum_cpu = np.cumsum(aug_sizes_cpu)

        # Host Staging Buffers for the NEXT level
        next_level_X_list = []
        next_level_F_list = []
        next_level_G_list = []
        next_level_sizes_list = []

        # --- Batch Processing Loop ---
        num_batches = level_total_aug_pts // BATCH_SIZE + 1
        start_cluster_idx = 0

        if LOG_MEM_STATS:
            log_mem_stats(f"R{rank:02} Start: L{level:02} Batching")

        K_good = 0
        for b in range(num_batches):
            logger.info(f"R{rank:02} L{level:02}: Processing Batch {b+1}/{num_batches}")

            target_aug_count = (b + 1) * BATCH_SIZE
            end_cluster_idx = int(np.searchsorted(aug_cumsum_cpu, target_aug_count))
            end_cluster_idx = min(end_cluster_idx + 1, num_clusters)
            if start_cluster_idx >= end_cluster_idx:
                break

            # 1. Zero-Copy Slicing
            with CudaTimer("time_slicing", profiling_stats, PROFILE_TIMING):
                sizes = CLUSTER_SIZES[start_cluster_idx:end_cluster_idx]
                starts_local = CLUSTER_STARTS[start_cluster_idx:end_cluster_idx]
                slice_start = int(starts_local[0])
                slice_end = int(starts_local[-1] + sizes[-1])

                pts = CLUSTER_PTS[slice_start:slice_end]
                f_vals = CLUSTER_F[slice_start:slice_end]
                g_vals = CLUSTER_G[slice_start:slice_end]
                starts_local_shifted = starts_local - slice_start

            # 2. Augmentation
            with CudaTimer("time_synth_gen", profiling_stats, PROFILE_TIMING):
                aug_sizes = level_augmented_sizes[start_cluster_idx:end_cluster_idx]
                aug_total = int(xp.sum(aug_sizes))
                K_batch = len(aug_sizes)

                X_batch = xp.empty((aug_total, d), dtype=xp.float64)
                F_batch = xp.empty((aug_total, fd), dtype=xp.float64)
                G_batch = xp.empty((aug_total, fd, d), dtype=xp.float64)

                aug_starts = xp.r_[0, xp.cumsum(aug_sizes)[:-1]]
                aug_ids = repeats(xp.arange(K_batch), aug_sizes)

                pt_idxs_in_cluster = (
                    xp.arange(aug_total, dtype=xp.int32) - aug_starts[aug_ids]
                )
                is_real = pt_idxs_in_cluster < sizes[aug_ids]
                X_batch[is_real] = pts
                F_batch[is_real] = f_vals
                G_batch[is_real] = g_vals

                is_synth = ~is_real
                num_synth = int(xp.sum(is_synth))
                num_generated_pts += num_synth
                if num_synth > 0:
                    if LOG_MEM_STATS:
                        log_mem_stats(
                            f" R{rank:02} L{level:02} B{b:02}/{num_batches:02}: Start Augmentation"
                        )
                    synth_c_ids = aug_ids[is_synth]
                    rng = xp.random.default_rng(SEED + level + b + rank * 1000)

                    c_starts = starts_local_shifted[synth_c_ids]
                    c_counts = sizes[synth_c_ids]
                    rand1 = (rng.random(num_synth) * c_counts).astype(xp.int64)
                    rand2 = (rng.random(num_synth) * (c_counts - 1)).astype(
                        xp.int64
                    ) + 1
                    idx1 = c_starts + rand1
                    idx2 = c_starts + (rand1 + rand2) % c_counts

                    p1 = pts[idx1]
                    p2 = pts[idx2]
                    alpha = rng.random((num_synth, 1)).astype(xp.float64)
                    X_synth_vals = p1 * alpha + p2 * (1.0 - alpha)
                    X_batch[is_synth] = X_synth_vals

                    if oracle_fn is not None:
                        F_syn, G_syn = oracle_fn(X_synth_vals)
                        F_batch[is_synth] = convert_to_2d_array(F_syn)
                        if IGNORE_GRADIENT:
                            G_batch[is_synth] = 0.0
                        else:
                            G_batch[is_synth] = convert_to_3d_array(G_syn)
                    else:
                        assert f_fn is not None
                        F_batch[is_synth] = convert_to_2d_array(f_fn(X_synth_vals))
                        if IGNORE_GRADIENT:
                            G_batch[is_synth] = 0.0
                        else:
                            assert g_fn is not None
                            G_batch[is_synth] = convert_to_3d_array(g_fn(X_synth_vals))
                    total_f_evals += X_synth_vals.shape[0]
                    del (
                        c_starts,
                        c_counts,
                        rand1,
                        rand2,
                        idx1,
                        idx2,
                        p1,
                        p2,
                        alpha,
                        X_synth_vals,
                    )

            if LOG_MEM_STATS:
                log_mem_stats(
                    f" R{rank:02} L{level:02} B{b:02}/{num_batches:02}: After Augmentation"
                )

            # 3. Compute Centroids & Reference Gradients
            with CudaTimer("time_error_calc", profiling_stats, PROFILE_TIMING):
                mus = (
                    xp.add.reduceat(X_batch, aug_starts) / aug_sizes[:, None]
                )  # (K_batch, d)

                if oracle_fn is not None:
                    F_mus_val, G_mus_val = oracle_fn(mus)
                    f_mus = convert_to_2d_array(F_mus_val)
                    if IGNORE_GRADIENT:
                        g_mus = xp.zeros((K_batch, fd, d), dtype=xp.float64)
                    else:
                        g_mus = convert_to_3d_array(G_mus_val)
                else:
                    assert f_fn is not None
                    f_mus = convert_to_2d_array(f_fn(mus))
                    if IGNORE_GRADIENT:
                        g_mus = xp.zeros((K_batch, fd, d), dtype=xp.float64)
                    else:
                        assert g_fn is not None
                        g_mus = convert_to_3d_array(g_fn(mus))

                total_f_evals += mus.shape[0]

                # 4. Error Metrics & Acceptance Logic
                if ERROR_METRIC == "Linf-local":
                    E_cluster_rel_max = xp.zeros((K_batch,), dtype=xp.float64)
                    E_cluster_abs_max = xp.zeros((K_batch,), dtype=xp.float64)

                    for j in range(fd):
                        F_max_j = xp.zeros((K_batch,), dtype=xp.float64)
                        scatter_maximum_1d(F_max_j, aug_ids, xp.abs(F_batch[:, j]))

                        pointwise_tay_err_j = run_fused_taylor_comp_abs(
                            X_batch, mus, aug_ids, F_batch, f_mus, g_mus, j
                        )
                        E_abs_j = xp.zeros((K_batch,), dtype=xp.float64)
                        scatter_maximum_1d(E_abs_j, aug_ids, pointwise_tay_err_j)
                        scale = xp.maximum(F_max_j, TINY_EPS)
                        E_rel_j = E_abs_j / scale
                        E_cluster_rel_max = xp.maximum(E_cluster_rel_max, E_rel_j)
                        E_cluster_abs_max = xp.maximum(E_cluster_abs_max, E_abs_j)

                    is_good = (E_cluster_rel_max < TOL) | (E_cluster_abs_max < TOL)
                    E_cluster = E_cluster_rel_max

                else:
                    if ERROR_METRIC == "L2":
                        scales_ker = xp.ones((fd,), dtype=xp.float64)
                        metric_mode = 0
                    else:
                        scales_ker = F_max_global
                        metric_mode = 1

                    err_pts = run_fused_taylor_metric(
                        X_batch,
                        mus,
                        aug_ids,
                        F_batch,
                        f_mus,
                        g_mus,
                        scales_ker,
                        metric_mode,
                    )

                    if ERROR_METRIC == "L2":
                        sum_sq = xp.add.reduceat(err_pts, aug_starts)
                        E_cluster = xp.sqrt(sum_sq)
                    else:
                        E_cluster = xp.zeros((K_batch,), dtype=xp.float64)
                        scatter_maximum_1d(E_cluster, aug_ids, err_pts)

                    is_good = E_cluster < TOL

            if LOG_MEM_STATS:
                log_mem_stats(
                    f" R{rank:02} L{level:02} B{b:02}/{num_batches:02}: After Error Metrics"
                )

            # 5. Handle Accepted ("Good") Clusters
            with CudaTimer("time_filter", profiling_stats, PROFILE_TIMING):
                good_idx_local = xp.nonzero(is_good)[0]
                K_good = int(good_idx_local.shape[0])

            if K_good > 0:
                with CudaTimer("time_good_Q", profiling_stats, PROFILE_TIMING):
                    good_mask = is_good[aug_ids]
                    X_good = X_batch[good_mask]
                    G_good = G_batch[good_mask]
                    good_counts = aug_sizes[good_idx_local]
                    dense_labels_good = repeats(xp.arange(K_good), good_counts)

                    radii_pt, ratio_pt = run_fused_metrics(
                        X_good,
                        G_good,
                        mus[good_idx_local],
                        g_mus[good_idx_local],
                        dense_labels_good,
                    )
                    radii = xp.zeros((K_good,), dtype=radii_pt.dtype)
                    L_obs = xp.zeros((K_good,), dtype=ratio_pt.dtype)
                    scatter_maximum_1d(radii, dense_labels_good, radii_pt)
                    scatter_maximum_1d(L_obs, dense_labels_good, ratio_pt)

                    Cov, _ = run_fused_stats(
                        X_good,
                        mus[good_idx_local],
                        dense_labels_good,
                        K_good,
                        d,
                        fd,
                        xp.zeros(K_good, dtype=xp.int32),
                        calc_E=False,
                    )
                    Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

                    if Q_SHRINK > 0.0:
                        tr = xp.trace(Cov, axis1=1, axis2=2)
                        scale_tr = tr / max(d, 1)
                        I = xp.eye(d, dtype=Cov.dtype)[None, :, :]
                        Cov = (1.0 - Q_SHRINK) * Cov + Q_SHRINK * (
                            scale_tr[:, None, None] * I
                        )
                        Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

                    Lambda, U = xp.linalg.eigh(Cov)
                    Lambda_inv = 1.0 / xp.maximum(Lambda, TINY_EPS)

                    dist_sq = run_fused_q_dist(
                        X_good, mus[good_idx_local], dense_labels_good, U, Lambda_inv
                    )
                    max_dist = xp.zeros((K_good,), dtype=dist_sq.dtype)
                    scatter_maximum_1d(max_dist, dense_labels_good, dist_sq)
                    scale_fac = 1.0 / xp.maximum(max_dist, TINY_EPS)

                    Qs = (
                        U * (scale_fac[:, None] * Lambda_inv)[:, None, :]
                    ) @ U.transpose(0, 2, 1)

                    H_sqrt = None
                    if STORE_HESSIAN_SQRT_PER_CLUSTER:
                        g_mus_exp = g_mus[good_idx_local][dense_labels_good]
                        dx = X_good - mus[good_idx_local][dense_labels_good]
                        lin = xp.einsum("nmd,nd->nm", g_mus_exp, dx)
                        diff_vals = F_batch[good_mask] - (
                            f_mus[good_idx_local][dense_labels_good] + lin
                        )

                        good_starts = xp.r_[0, xp.cumsum(good_counts)[:-1]]
                        sq_err = xp.add.reduceat(diff_vals**2, good_starts)
                        sq_val = xp.add.reduceat(F_batch[good_mask] ** 2, good_starts)
                        scores = xp.sqrt(sq_err) / (xp.sqrt(sq_val) + TINY_EPS)
                        active_comps_good = xp.argmax(scores, axis=1).astype(xp.int32)

                        C_h, E_h = run_fused_stats(
                            X_good,
                            mus[good_idx_local],
                            dense_labels_good,
                            K_good,
                            d,
                            fd,
                            active_comps_good,
                            G=G_good,
                            g_mus=g_mus[good_idx_local],
                            calc_E=True,
                        )
                        C_h = 0.5 * (C_h + C_h.transpose(0, 2, 1))
                        S_h = E_h + E_h.transpose(0, 2, 1)

                        dvals, U_h = xp.linalg.eigh(C_h)
                        dvals = xp.maximum(dvals, TINY_EPS)
                        Gtilde = U_h.transpose(0, 2, 1) @ S_h @ U_h
                        F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])
                        _, V_h = xp.linalg.eigh(F_mat)

                        Z = U_h @ V_h
                        M = (Z * xp.sqrt(xp.abs(_))[:, None, :]) @ Z.transpose(0, 2, 1)
                        H_sqrt = 0.5 * (M + M.transpose(0, 2, 1))

                with CudaTimer("time_good_d2h", profiling_stats, PROFILE_TIMING):
                    end_out = out_count + K_good
                    buffers, out_cap = adjust_buffer_capacity_host(
                        buffers,
                        out_count,
                        out_cap,
                        end_out,
                        STORE_HESSIAN_SQRT_PER_CLUSTER,
                    )
                    buffers["centroids"][out_count:end_out] = _to_host(
                        mus[good_idx_local]
                    )
                    buffers["f"][out_count:end_out] = _to_host(f_mus[good_idx_local])
                    buffers["g"][out_count:end_out] = _to_host(g_mus[good_idx_local])
                    buffers["Q"][out_count:end_out] = _to_host(Qs)
                    if H_sqrt is not None:
                        buffers["H_sqrt"][out_count:end_out] = _to_host(H_sqrt)
                    buffers["E_cluster"][out_count:end_out] = _to_host(
                        E_cluster[good_idx_local]
                    )
                    buffers["radii"][out_count:end_out] = _to_host(radii)
                    buffers["L_obs"][out_count:end_out] = _to_host(L_obs)
                    buffers["counts"][out_count:end_out] = _to_host(
                        good_counts.astype(xp.int64)
                    )
                    out_count = end_out

                if STORE_TREE_DATA:
                    _src = out_count - K_good
                    _td_entry = {
                        "centroids": buffers["centroids"][_src:out_count].copy(),
                        "f": buffers["f"][_src:out_count].copy(),
                        "g": buffers["g"][_src:out_count].copy(),
                        "radii": buffers["radii"][_src:out_count].copy(),
                        "L_obs": buffers["L_obs"][_src:out_count].copy(),
                        "E_cluster": buffers["E_cluster"][_src:out_count].copy(),
                        "counts": buffers["counts"][_src:out_count].copy(),
                        "Q": buffers["Q"][_src:out_count].copy(),
                        "points_in_clusters": [
                            _to_host(X_good[dense_labels_good == i])
                            for i in range(K_good)
                        ],
                        "mpi_rank": np.full((K_good,), rank, dtype=np.int64),
                    }
                    if level not in tree_data:
                        tree_data[level] = _td_entry
                    else:
                        existing = tree_data[level]
                        for dk in (
                            "centroids",
                            "f",
                            "g",
                            "radii",
                            "L_obs",
                            "E_cluster",
                            "counts",
                            "Q",
                            "mpi_rank",
                        ):
                            if dk in existing and dk in _td_entry:
                                existing[dk] = np.concatenate(
                                    [existing[dk], _td_entry[dk]], axis=0
                                )
                        existing.setdefault("points_in_clusters", []).extend(
                            _td_entry.get("points_in_clusters", [])
                        )

            if LOG_MEM_STATS:
                log_mem_stats(
                    f" R{rank:02} L{level:02} B{b:02}/{num_batches:02}: After Good Clusters"
                )

            # 6. Handle Rejected ("Bad") Clusters -> Split
            with CudaTimer("time_filter", profiling_stats, PROFILE_TIMING):
                bad_idx_local = xp.nonzero(~is_good)[0]
                n_bad = int(bad_idx_local.shape[0])

            if n_bad > 0:
                with CudaTimer("time_bad_split", profiling_stats, PROFILE_TIMING):
                    bad_mask = (~is_good)[aug_ids]
                    X_bad = X_batch[bad_mask]
                    F_bad = F_batch[bad_mask]
                    G_bad = G_batch[bad_mask]
                    bad_counts = aug_sizes[bad_idx_local]
                    dense_labels_bad = repeats(xp.arange(n_bad), bad_counts)

                    if IGNORE_GRADIENT or CLUSTERING_MATRIX == "None":
                        # Fallback: PCA Split (Split along variance of X)
                        M_split = xp.eye(d, dtype=xp.float64)[None, :, :].repeat(
                            n_bad, axis=0
                        )
                        C, _ = run_fused_stats(
                            X_bad,
                            mus[bad_idx_local],
                            dense_labels_bad,
                            n_bad,
                            d,
                            fd,
                            xp.zeros(n_bad, dtype=xp.int32),
                            calc_E=False,
                        )
                        C = 0.5 * (C + C.transpose(0, 2, 1))
                        CovZ = C

                    elif CLUSTERING_MATRIX == "Hessian_LSQ":
                        # Hessian Split: Split along eigenvector of Hessian estimate
                        # 1. Determine "Active Component" (Output dim with highest error)
                        g_mus_exp_bad = g_mus[bad_idx_local][dense_labels_bad]
                        dx_bad = X_bad - mus[bad_idx_local][dense_labels_bad]
                        lin_bad = xp.einsum("nmd,nd->nm", g_mus_exp_bad, dx_bad)
                        diff_bad = F_bad - (
                            f_mus[bad_idx_local][dense_labels_bad] + lin_bad
                        )

                        bad_starts = xp.r_[0, xp.cumsum(bad_counts)[:-1]]
                        sq_err = xp.add.reduceat(diff_bad**2, bad_starts)
                        sq_val = xp.add.reduceat(F_bad**2, bad_starts)
                        scores = xp.sqrt(sq_err) / (xp.sqrt(sq_val) + TINY_EPS)
                        active_comps = xp.argmax(scores, axis=1).astype(xp.int32)

                        # 2. Accumulate Stats and Solve for Hessian
                        C, E = run_fused_stats(
                            X_bad,
                            mus[bad_idx_local],
                            dense_labels_bad,
                            n_bad,
                            d,
                            fd,
                            active_comps,
                            G=G_bad,
                            g_mus=g_mus[bad_idx_local],
                            calc_E=True,
                        )
                        C = 0.5 * (C + C.transpose(0, 2, 1))
                        S = E + E.transpose(0, 2, 1)

                        dvals, U = xp.linalg.eigh(C)
                        dvals = xp.maximum(dvals, TINY_EPS)
                        Gtilde = U.transpose(0, 2, 1) @ S @ U
                        F_mat = Gtilde / (dvals[:, :, None] + dvals[:, None, :])
                        _, V = xp.linalg.eigh(F_mat)

                        # M = |H|^1/2
                        Z = U @ V
                        M_split = (Z * xp.sqrt(xp.abs(_))[:, None, :]) @ Z.transpose(
                            0, 2, 1
                        )
                        M_split = 0.5 * (M_split + M_split.transpose(0, 2, 1))

                        # Covariance in transformed space Z = M * dx
                        CovZ = M_split @ C @ M_split.transpose(0, 2, 1)

                    elif CLUSTERING_MATRIX == "Cov_Lvl_curves":
                        # Approximate Hessian using inverse covariance formulation.

                        # 1. Compute the linear surrogate errors
                        lin_bad = xp.einsum(
                            "nmd,nd->nm",
                            g_mus[bad_idx_local][dense_labels_bad],
                            X_bad - mus[bad_idx_local][dense_labels_bad],
                        )
                        diff_bad = F_bad - (
                            f_mus[bad_idx_local][dense_labels_bad] + lin_bad
                        )
                        E_bad = xp.linalg.norm(diff_bad, axis=1)

                        # 2. Sort and pick out the smallest n
                        n_est_cov = (
                            MIN_PTS_PER_CLUSTER // 3
                        )  # this is empirical for now
                        cov_est_mask = get_n_smallest_per_block(
                            E_bad, dense_labels_bad, n_est_cov
                        )

                        # 3. Use these points to compute the sample covariances for the Hessian
                        H_inv, _ = run_fused_stats(
                            X_bad[cov_est_mask],
                            mus[bad_idx_local],
                            dense_labels_bad[cov_est_mask],
                            n_bad,
                            d,
                            fd,
                            xp.zeros(n_bad, dtype=xp.int32),
                            calc_E=False,
                        )
                        H_inv = 0.5 * (H_inv + H_inv.transpose(0, 2, 1))

                        # 3. Compute square root inverse of H_inv to get the scaling matrix for the split
                        dvals, U = xp.linalg.eigh(H_inv)
                        dvals_sqrt_inv = 1.0 / xp.sqrt(
                            xp.maximum(xp.abs(dvals), TINY_EPS)
                        )
                        M_split = (U * dvals_sqrt_inv[:, None]) @ U.transpose(0, 2, 1)
                        M_split = 0.5 * (M_split + M_split.transpose(0, 2, 1))

                        # 4. Full covariance
                        C, _ = run_fused_stats(
                            X_bad,
                            mus[bad_idx_local],
                            dense_labels_bad,
                            n_bad,
                            d,
                            fd,
                            xp.zeros(n_bad, dtype=xp.int32),
                            calc_E=False,
                        )
                        C = 0.5 * (C + C.transpose(0, 2, 1))

                        # 5. Covariance in transformed space
                        CovZ = M_split @ C @ M_split.transpose(0, 2, 1)

                    # Get Split Direction (Dominant Eigenvector)
                    _, V_z = xp.linalg.eigh(CovZ)
                    v = V_z[:, :, -1]

                    # Project Points & Sort
                    scores_proj = run_fused_score(
                        X_bad, mus[bad_idx_local], dense_labels_bad, M_split, v
                    )
                    sort_order = xp.lexsort(xp.stack([scores_proj, dense_labels_bad]))

                with CudaTimer("time_bad_d2h", profiling_stats, PROFILE_TIMING):
                    # Collect Sorted Data for Next Level
                    next_level_X_list.append(_to_host(X_bad[sort_order]))
                    next_level_F_list.append(_to_host(F_bad[sort_order]))
                    next_level_G_list.append(_to_host(G_bad[sort_order]))

                    # Calculate Child Sizes (Binary Split)
                    half = bad_counts // 2
                    other = bad_counts - half
                    new_sz = xp.zeros(2 * n_bad, dtype=xp.int64)  # (2 * n_bad,)
                    new_sz[0::2] = half
                    new_sz[1::2] = other
                    next_level_sizes_list.append(_to_host(new_sz))

                with CudaTimer("time_gc", profiling_stats, PROFILE_TIMING):
                    del (
                        M_split,
                        C,
                        scores_proj,
                        sort_order,
                        bad_mask,
                        X_bad,
                        F_bad,
                        G_bad,
                        dense_labels_bad,
                        bad_counts,
                        half,
                        other,
                    )

            if LOG_MEM_STATS:
                log_mem_stats(
                    f" R{rank:02} L{level:02} B{b:02}/{num_batches:02}: After Bad Clusters"
                )

            start_cluster_idx = end_cluster_idx
            del X_batch, F_batch, G_batch
            if mempool is not None:
                mempool.free_all_blocks()

        # --- Level Transition ---
        # The break decision must be global: _rebalance_workload uses MPI collectives
        # (allgather / alltoall), so all ranks must agree to exit together or all
        # must continue.  A rank-local break would deadlock any rank that is still
        # inside a collective.
        global_bad = comm.allreduce(len(next_level_X_list), op=MPI.SUM)
        if global_bad == 0:
            logger.info(
                f"R{rank:02} L{level:02}: All clusters accepted globally — terminating refinement."
            )
            break

        with CudaTimer("time_transition", profiling_stats, PROFILE_TIMING):
            # Ranks that accepted all their clusters locally contribute empty arrays so
            # they can still participate in the collective rebalance call below.
            if next_level_X_list:
                next_X = np.concatenate(next_level_X_list)
                next_F = np.concatenate(next_level_F_list)
                next_G = np.concatenate(next_level_G_list)
                next_sizes = np.concatenate(next_level_sizes_list)
            else:
                next_X = np.empty((0, d), dtype=np.float64)
                next_F = np.empty((0, fd), dtype=np.float64)
                next_G = np.empty((0, fd, d), dtype=np.float64)
                next_sizes = np.empty(0, dtype=np.int64)

            # Rebalance workload across ranks for the next level based on the new cluster
            # sizes
            next_X, next_F, next_G, next_sizes = _rebalance_workload(
                comm, next_X, next_F, next_G, next_sizes, d, fd, BATCH_SIZE
            )

            valid = next_sizes > 0
            if not np.all(valid):
                keep = np.repeat(valid, next_sizes)
                CLUSTER_PTS = xp.asarray(next_X[keep])
                CLUSTER_F = xp.asarray(next_F[keep])
                CLUSTER_G = xp.asarray(next_G[keep])
                CLUSTER_SIZES = xp.asarray(next_sizes[valid])
            else:
                CLUSTER_PTS = xp.asarray(next_X)
                CLUSTER_F = xp.asarray(next_F)
                CLUSTER_G = xp.asarray(next_G)
                CLUSTER_SIZES = xp.asarray(next_sizes)

            CLUSTER_STARTS = xp.r_[0, xp.cumsum(CLUSTER_SIZES)[:-1]]

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
                f"mean: {CLUSTER_SIZES.mean():.1f}, median: {int(xp.median(CLUSTER_SIZES))}, "
                f"max: {int(CLUSTER_SIZES.max())}"
            )
        level_runtime = time.perf_counter() - _t_level
        logger.info(f"Runtime - R{rank:02} L{level:02}: {level_runtime:.3f}s")
        if profiling_stats["level_stats"]:
            profiling_stats["level_stats"][-1].update({"runtime_sec": level_runtime})

        level += 1

    return buffers, out_count, total_f_evals, num_generated_pts, tree_data, level


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
    ERROR_METRIC: Literal["L2", "Linf-global", "Linf-local"] = "Linf-global",
    INITIAL_CLUSTERING_METHOD: Literal["kmeans", "knndbscan"] = "kmeans",
    CLUSTERING_MATRIX: Literal["Hessian_LSQ", "Cov_Lvl_curves", "None"] = "Hessian_LSQ",
    NUM_INITIAL_CLUSTERS: int = 100,
    MIN_PTS_PER_CLUSTER: int = 100,
    MAX_CLUSTERS_EST: int = 100_000,
    SEED: int = 0,
    Q_SHRINK: float = 0.05,
    MAX_LEVELS: int = 50,
    BATCH_SIZE: int = 10_000_000,
    IGNORE_GRADIENT: bool = False,
    STORE_HESSIAN_SQRT_PER_CLUSTER: bool = False,
    STORE_TREE_DATA: bool = False,
    LOG_MEM_STATS: bool = False,
    PROFILE_TIMING: bool = False,
) -> dict[str, Any] | None:
    """
    Adaptive Clustering for Vector-Valued Functions (GPU Accelerated).

    Constructs a piecewise-local linear surrogate model for a function F: R^d \to R^fd.
    The algorithm recursively splits clusters where the local first-order Taylor approximation fails to meet the specified `TOL`.

    Architecture (Sort-Streaming):
    To handle datasets larger than GPU memory, this implementation uses a "Sort-Streaming" approach:
    1.  **Physical Sort:** Points are sorted by cluster label in VRAM, allowing zero-copy slicing.
    2.  **Streaming:** The current level is processed in batches on the GPU.
    3.  **Offloading:** Accepted ("Good") clusters are immediately moved to CPU host memory.
    4.  **Refinement:** Rejected ("Bad") clusters are split and streamed to a CPU staging buffer
        for the next level.

    **Error Metrics:**
    Let E(x) = F(x) - (F(μ_k) + ∇F(μ_k)(x - μ_k)).
    -   L2: RMSE-style check. √(∑_{x ∈ C_k} ‖E(x)‖_F^2) < TOL.
        (Strict total energy check).
    -   Linf-global: Max relative error normalized by global range.
        max_{x ∈ C_k} max_j |E_j(x)| / max_y |F_j(y)| < TOL.
    -   Linf-local: Max relative error normalized by local cluster range.
        max_{x ∈ C_k} max_j |E_j(x)| / max_{y ∈ C_k} |F_j(y)| < TOL.
        Includes hybrid acceptance (Absolute < TOL) to handle zero-crossings.

    Args:
        X: Input training points. Shape (N, d).
        f_fn: Callable returning function values. Output shape (N, fd).
        g_fn: Callable returning gradients/Jacobians. Output shape (N, fd, d).
        TOL: Error tolerance for cluster acceptance.
        ERROR_METRIC: Metric for acceptance ('L2', 'Linf-global', 'Linf-local').
        INITIAL_CLUSTERING_METHOD: Method for initial clustering
            ('kmeans', 'knndbscan').
        CLUSTERING_MATRIX: Matrix for spectral clustering of "Bad" clusters
            ('Hessian_LSQ', 'Cov_Lvl_curves', 'None').
        NUM_INITIAL_CLUSTERS: Number of seeds for initial K-Means.
        MIN_PTS_PER_CLUSTER: Minimum points per cluster. If a cluster has fewer,
            synthetic points are generated via convex combinations to stabilize geometry.
        MAX_CLUSTERS_EST: Initial capacity for CPU output buffers (auto-expands).
        SEED: Random seed for initialization and synthetic generation.
        Q_SHRINK: Shrinkage factor a for covariance: Σ ← (1-a)Σ + a (Tr(Σ)/d)I.
        MAX_LEVELS: Maximum recursion depth for splitting.
        BATCH_SIZE: Maximum number of points processed in a single GPU kernel launch.
        IGNORE_GRADIENT: If True, assumes ∇F = 0 (Voronoi/Piecewise-Constant
            approximation).
        STORE_HESSIAN_SQRT_PER_CLUSTER: If True, computes and stores |H|^{1/2} for every
            accepted cluster (useful for anisotropic trust regions downstream).
        LOG_MEM_STATS: If True, prints VRAM usage at key steps.
        PROFILE_TIMING: If True, tracks detailed execution times for performance
            profiling.

    Returns:
        A dictionary containing the surrogate model parameters:
        -   `centroids`: (K, d) Cluster centers.
        -   `f`: (K, fd) Function values at centroids.
        -   `g`: (K, fd, d) Gradients/Jacobians at centroids.
        -   `Q`: (K, d, d) Trust region shape matrices (inverse covariance).
        -   `radii`: (K,) Trust region radii.
        -   `L_obs`: (K,) Observed Lipschitz constants.
        -   `E_cluster`: (K,) Max observed error per cluster.
        -   `counts`: (K,) Number of points support per cluster.
        -   `H_sqrt`: (K, d, d) (Optional) Symmetric Hessian square roots.
        -   `tree_data`: (dict) (Optional) Per-level cluster data keyed by level index, for visualization/debugging.
        -   `num_levels`: (int) Total levels refined.
        -   `num_generated_pts`: (int) Number of generated points.
        -  `total_f_evals`: (int) Total function evaluations performed.
        -  `profiling_stats`: (dict) Detailed timing and statistics if PROFILE_TIMING is True.
    """

    # Validate function arguments
    if oracle_fn is not None:
        if f_fn is not None or g_fn is not None:
            raise ValueError(
                "Cannot specify both oracle_fn and f_fn/g_fn. Use either oracle_fn OR both f_fn and g_fn."
            )
    else:
        if f_fn is None or g_fn is None:
            raise ValueError("Must specify either oracle_fn OR both f_fn and g_fn.")

    if STORE_TREE_DATA:
        logger.warning(
            "STORE_TREE_DATA=True: This will store per-cluster point data for all finalized clusters, which can consume a lot of memory, and slow down the algorithm. Use for debugging/visualization purposes, but be cautious with large datasets or many levels of refinement."
        )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    _start_time = time.perf_counter()

    # Per-rank profiling stats (merged by rank 0 at the end)
    profiling_stats = {
        "N0": 0,
        "num_levels": 0,
        "num_generated_pts": 0,
        "total_f_evals": 0,
        "num_final_clusters": 0,
        "time_total": 0.0,
        "time_init": 0.0,
        "time_slicing": 0.0,
        "time_find_req_synth": 0.0,
        "time_synth_gen": 0.0,
        "time_error_calc": 0.0,
        "time_filter": 0.0,
        "time_good_Q": 0.0,
        "time_good_d2h": 0.0,
        "time_bad_split": 0.0,
        "time_bad_d2h": 0.0,
        "time_transition": 0.0,
        "time_gc": 0.0,
        "level_stats": [],
    }

    # ------------------------------------------------------------------
    # 1. Rank 0: Evaluate, Cluster, Partition, Prepare scatter payloads
    # ------------------------------------------------------------------
    F_max_global_host: np.ndarray | None = None
    scatter_data: list | None = None
    d_bcast: int = 0
    fd_bcast: int = 0

    if rank == 0:
        logger.info(f"MPI Size: {size}")

        with CudaTimer("time_init", profiling_stats, PROFILE_TIMING):
            X0_gpu = xp.asarray(X, dtype=xp.float64)
            N0, d_bcast = X0_gpu.shape
            profiling_stats["N0"] = N0

            if LOG_MEM_STATS:
                log_mem_stats("R00: Start Adaptive Clustering")

            _t0 = time.perf_counter()
            if oracle_fn is not None:
                F_out, G_out = oracle_fn(X0_gpu)
                F0_gpu = convert_to_2d_array(F_out)
                fd_bcast = F0_gpu.shape[1]
                if IGNORE_GRADIENT:
                    G0_gpu = xp.zeros((N0, fd_bcast, d_bcast), dtype=xp.float64)
                else:
                    G0_gpu = convert_to_3d_array(G_out)
            else:
                assert f_fn is not None
                F0_gpu = convert_to_2d_array(f_fn(X0_gpu))
                fd_bcast = F0_gpu.shape[1]
                if IGNORE_GRADIENT:
                    G0_gpu = xp.zeros((N0, fd_bcast, d_bcast), dtype=xp.float64)
                else:
                    assert g_fn is not None
                    G0_gpu = convert_to_3d_array(g_fn(X0_gpu))

            logger.info(f"R00: N0={N0:,}, d={d_bcast}, fd={fd_bcast}")
            logger.info(f"Runtime - R00: Initial f/g: {time.perf_counter()-_t0:.3f}s")
            if LOG_MEM_STATS:
                log_mem_stats("R00: After Initial F/G Evaluation")

            # Compute global scale for Linf-global metric
            F_max_global_host = _to_host(xp.max(xp.abs(F0_gpu), axis=0))

            # Initial Clustering
            _t1 = time.perf_counter()
            if INITIAL_CLUSTERING_METHOD == "kmeans":
                if IS_CUPY_BACKEND:
                    from cuml.cluster import KMeans as cuKMeans

                    km = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
                    labels0 = km.fit_predict(X0_gpu)
                else:
                    km = KMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
                    labels0 = km.fit_predict(_to_host(X0_gpu))
                labels0 = (
                    labels0.astype(np.int64)
                    if isinstance(labels0, np.ndarray)
                    else _to_host(labels0).astype(np.int64)
                )
                logger.info(
                    f"Runtime - R00: Initial KMeans {time.perf_counter()-_t1:.3f}s"
                )

            elif INITIAL_CLUSTERING_METHOD == "knndbscan":
                from knndbscan import knn_dbscan

                _device = "gpu" if IS_CUPY_BACKEND else "cpu"
                labels0 = _to_host(
                    knn_dbscan(
                        X0_gpu,
                        eps=0.5,
                        min_samples=MIN_PTS_PER_CLUSTER,
                        device=_device,
                        n_jobs=-1,
                    )
                ).astype(np.int64)
                num_initial_clusters = int(np.unique(labels0).shape[0])
                logger.info(
                    f"R00: Initial kNN-DBSCAN: {num_initial_clusters:,} clusters"
                )
                logger.info(
                    f"Runtime - R00: Initial kNN-DBSCAN {time.perf_counter()-_t1:.3f}s"
                )
            else:
                raise ValueError(
                    f"Unsupported INITIAL_CLUSTERING_METHOD: {INITIAL_CLUSTERING_METHOD}"
                )

            if LOG_MEM_STATS:
                log_mem_stats("R00: After Initial Clustering")

            # Move data to host for safe MPI scattering
            X0_host = _to_host(X0_gpu)
            F0_host = _to_host(F0_gpu)
            G0_host = _to_host(G0_gpu)
            del X0_gpu, F0_gpu, G0_gpu

            # Partition by label chunks so full clusters stay together
            unique_labels = np.unique(labels0)

            if unique_labels.shape[0] < size:
                raise ValueError(
                    f"Number of unique labels ({unique_labels.shape[0]}) is less than MPI "
                    f"size ({size}). Increase NUM_INITIAL_CLUSTERS or reduce MPI size."
                )

            label_chunks = np.array_split(unique_labels, size)
            scatter_data = []
            for chunk in label_chunks:
                mask = np.isin(labels0, chunk)
                sub_labels = labels0[mask]
                sort_idx = np.argsort(sub_labels, kind="stable")
                sub_labels_sorted = sub_labels[sort_idx]
                _, sub_counts = np.unique(sub_labels_sorted, return_counts=True)
                scatter_data.append(
                    {
                        "X": X0_host[mask][sort_idx],
                        "F": F0_host[mask][sort_idx],
                        "G": G0_host[mask][sort_idx],
                        "sizes": sub_counts.astype(np.int64),
                    }
                )
    else:
        scatter_data = None

    # ------------------------------------------------------------------
    # 2. Broadcast F_max_global and dimensions; scatter data
    # ------------------------------------------------------------------
    F_max_global_host = comm.bcast(F_max_global_host, root=0)
    d_bcast, fd_bcast = comm.bcast((d_bcast, fd_bcast), root=0)

    local_dict = comm.scatter(scatter_data, root=0)

    local_X = xp.asarray(local_dict["X"], dtype=xp.float64)
    local_F = xp.asarray(local_dict["F"], dtype=xp.float64)
    local_G = xp.asarray(local_dict["G"], dtype=xp.float64)
    local_sizes = xp.asarray(local_dict["sizes"], dtype=xp.int64)
    local_starts = xp.r_[0, xp.cumsum(local_sizes)[:-1]]

    d = d_bcast
    fd = fd_bcast
    F_max_global = xp.asarray(F_max_global_host, dtype=xp.float64)

    logger.info(
        f"R{rank:03}: Received {int(local_X.shape[0]):,} pts "
        f"({int(local_sizes.shape[0]):,} clusters). Starting BFS Refinement."
    )

    # ------------------------------------------------------------------
    # 3. Local BFS refinement
    # ------------------------------------------------------------------
    buffers, out_count, local_f_evals, local_gen_pts, local_tree_data, last_level = (
        _run_local_refinement(
            comm=comm,
            CLUSTER_PTS=local_X,
            CLUSTER_F=local_F,
            CLUSTER_G=local_G,
            CLUSTER_SIZES=local_sizes,
            CLUSTER_STARTS=local_starts,
            F_max_global=F_max_global,
            d=d,
            fd=fd,
            f_fn=f_fn,
            g_fn=g_fn,
            oracle_fn=oracle_fn,
            TOL=TOL,
            ERROR_METRIC=ERROR_METRIC,
            MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
            MAX_CLUSTERS_EST=max(1, MAX_CLUSTERS_EST // max(1, size)),
            SEED=SEED,
            Q_SHRINK=Q_SHRINK,
            MAX_LEVELS=MAX_LEVELS,
            BATCH_SIZE=BATCH_SIZE,
            CLUSTERING_MATRIX=CLUSTERING_MATRIX,
            IGNORE_GRADIENT=IGNORE_GRADIENT,
            STORE_HESSIAN_SQRT_PER_CLUSTER=STORE_HESSIAN_SQRT_PER_CLUSTER,
            STORE_TREE_DATA=STORE_TREE_DATA,
            LOG_MEM_STATS=LOG_MEM_STATS,
            PROFILE_TIMING=PROFILE_TIMING,
            rank=rank,
            profiling_stats=profiling_stats,
        )
    )

    logger.info(
        f"R{rank:03}: Local BFS done. Found {out_count:,} clusters. "
        f"Local f-evals={local_f_evals:,}. Time={time.perf_counter()-_start_time:.2f}s"
    )

    # ------------------------------------------------------------------
    # 4. Pack local results to host dict
    # ------------------------------------------------------------------
    local_result_host = {k: buffers[k][:out_count] for k in buffers}
    if STORE_TREE_DATA:
        local_result_host["tree_data"] = local_tree_data

    # Stamp per-rank scalars into profiling_stats before the gather
    profiling_stats["num_levels"] = last_level
    profiling_stats["num_generated_pts"] = local_gen_pts
    profiling_stats["total_f_evals"] = local_f_evals
    profiling_stats["num_final_clusters"] = out_count
    profiling_stats["time_total"] = time.perf_counter() - _start_time
    # Piggyback profiling stats on the existing gather (no extra comms round)
    local_result_host["_profiling_stats"] = profiling_stats

    # ------------------------------------------------------------------
    # 5. Gather & merge on rank 0
    # ------------------------------------------------------------------
    all_results = comm.gather(local_result_host, root=0)

    if rank == 0:
        logger.info("R00: Merging results from all ranks.")

        # Extract per-rank profiling stats piggybacked on the gather payload
        all_profiling_stats: list[dict] = [r.pop("_profiling_stats") for r in all_results]  # type: ignore

        array_keys = [k for k in all_results[0] if k not in ("tree_data",)]  # type: ignore

        merged: dict[str, Any] = {}
        for k in array_keys:
            merged[k] = np.concatenate([r[k] for r in all_results], axis=0)  # type: ignore
            if IS_CUPY_BACKEND:
                merged[k] = xp.asarray(merged[k])

        if STORE_TREE_DATA:
            merged["tree_data"] = {}
            for r in all_results:  # type: ignore
                for lvl, data in r.get("tree_data", {}).items():
                    if lvl not in merged["tree_data"]:
                        merged["tree_data"][lvl] = {
                            k: v.copy() if isinstance(v, np.ndarray) else v
                            for k, v in data.items()
                        }
                    else:
                        existing = merged["tree_data"][lvl]
                        for dk in (
                            "centroids",
                            "f",
                            "g",
                            "radii",
                            "L_obs",
                            "E_cluster",
                            "counts",
                            "Q",
                            "mpi_rank",
                        ):
                            if dk in existing and dk in data:
                                existing[dk] = np.concatenate(
                                    [existing[dk], data[dk]], axis=0
                                )
                        existing.setdefault("points_in_clusters", []).extend(
                            data.get("points_in_clusters", [])
                        )

        total_clusters = int(merged["centroids"].shape[0])
        _total_time = time.perf_counter() - _start_time

        # ------------------------------------------------------------------
        # Build profiling_stats: consolidated (reduced across ranks) + per_rank
        # ------------------------------------------------------------------
        _timer_keys = [
            k
            for k in all_profiling_stats[0]
            if k.startswith("time_") and k != "time_total"
        ]
        _sum_keys = ["num_generated_pts", "total_f_evals"]

        consolidated_profiling: dict[str, Any] = {
            # Wall-clock total is the slowest rank (critical path)
            "time_total": _total_time,
            # N0 is only populated by rank 0
            "N0": all_profiling_stats[0].get("N0", 0),
            # Deepest level reached across all ranks
            "num_levels": max(ps["num_levels"] for ps in all_profiling_stats),
            # All final clusters already merged above
            "num_final_clusters": total_clusters,
            # Flat list of every per-rank level entry, sorted by (level, rank)
            "level_stats": sorted(
                [entry for ps in all_profiling_stats for entry in ps["level_stats"]],
                key=lambda e: (e["level"], e["rank"]),
            ),
        }
        # Summed timers (cumulative compute across all ranks)
        for k in _timer_keys:
            consolidated_profiling[k] = sum(
                ps.get(k, 0.0) for ps in all_profiling_stats
            )
        # Summed scalar counts
        for k in _sum_keys:
            consolidated_profiling[k] = sum(ps.get(k, 0) for ps in all_profiling_stats)
        # Attach the per-rank breakdown
        consolidated_profiling["per_rank"] = all_profiling_stats

        merged["num_levels"] = consolidated_profiling["num_levels"]
        merged["num_generated_pts"] = consolidated_profiling["num_generated_pts"]
        merged["total_f_evals"] = consolidated_profiling["total_f_evals"]
        merged["profiling_stats"] = consolidated_profiling

        logger.info(
            f"R00: final clusters={total_clusters:,}, " f"total time={_total_time:.2f}s"
        )
        return merged
    else:
        return None


if __name__ == "__main__":
    import argparse
    import os
    from functools import partial

    from src import FIGURES_DIR
    from src.logging_utils import setup_logging
    from src.plotter_utils import (
        plot_adaptive_clustering_results,
        plot_mpi_level_heatmap,
    )
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging(console_level=logging.DEBUG)

    _rank = _get_mpi_rank()
    _size = _get_mpi_size()

    AC_PLOTS_DIR = FIGURES_DIR / "adaptive_clustering_rosenbrock_rebalancing"

    # --------------------------------------------------------------------------
    # Argument parsing & setup
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Adaptive GPU Clustering Example with Rosenbrock Function"
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Dimensionality of the problem"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Tolerance for training and evaluation",
    )
    parser.add_argument(
        "--n-train", type=float, default=100_000, help="Number of training samples"
    )
    parser.add_argument(
        "--n-init-clusters",
        type=int,
        default=50,
        help="Number of initial clusters for adaptive clustering",
    )
    parser.add_argument(
        "--n-min-per-cluster",
        type=int,
        default=500,
        help="Minimum points per cluster for adaptive clustering",
    )
    parser.add_argument(
        "--error-metric",
        type=str,
        default="Linf-local",
        choices=["L2", "Linf-global", "Linf-local"],
        help="Error metric for adaptive clustering",
    )
    parser.add_argument(
        "--initial-clustering-method",
        type=str,
        default="kmeans",
        choices=["kmeans", "knndbscan"],
        help="Method for initial clustering",
    )
    parser.add_argument(
        "--rosenbrock-a",
        type=float,
        default=100.0,
        help="Rosenbrock function parameter 'a'",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="",
        help="Subdirectory for output files",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-execution even if output file exists",
    )

    args = parser.parse_args()
    DIM = args.dim
    TOL = args.tol
    N_TRAIN = int(args.n_train)
    NUM_INITIAL_CLUSTERS = args.n_init_clusters
    MIN_PTS_PER_CLUSTER = args.n_min_per_cluster
    ERROR_METRIC = args.error_metric
    INITIAL_CLUSTERING_METHOD = args.initial_clustering_method
    A_ROSENBROCK = args.rosenbrock_a
    OUTPUT_SUBDIR = args.output_subdir
    FORCE_REEXEC = args.force

    # Check if output already exists
    AC_PLOTS_DIR = AC_PLOTS_DIR / OUTPUT_SUBDIR if OUTPUT_SUBDIR else AC_PLOTS_DIR
    AC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    img_fpath = (
        AC_PLOTS_DIR
        / f"mpi_sz{_size}_ac_rosen{A_ROSENBROCK}_tol{TOL}_error_metric_{ERROR_METRIC}_init{NUM_INITIAL_CLUSTERS}_min{MIN_PTS_PER_CLUSTER}_ntrain{N_TRAIN}_dim{DIM}_{INITIAL_CLUSTERING_METHOD}.png"
    )
    _skip = DIM == 2 and img_fpath.exists() and not FORCE_REEXEC
    if _skip:
        print(f"Plot file {img_fpath} already exists. Skipping execution.")

    # Create partial Rosenbrock functions (all ranks need these for oracle calls)
    rosenbrock = partial(rosenbrock, a=A_ROSENBROCK)
    rosenbrock_grad = partial(rosenbrock_grad, a=A_ROSENBROCK)

    # --------------------------------------------------------------------------
    # Generate data on rank 0 only; other ranks pass a dummy empty array.
    # adaptive_clustering() scatters work internally so only rank 0 needs X.
    # --------------------------------------------------------------------------
    if _rank == 0:
        xp.random.seed(0)
        print(f"Generating {N_TRAIN:,} points (d={DIM})...")
        PROB_LIMS = (-1.0, 3)
        X = xp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(N_TRAIN, DIM))
        print("Starting Adaptive Clustering...")
    else:
        PROB_LIMS = (-1.0, 3)
        X = xp.empty((0, DIM), dtype=xp.float64)

    STORE_TREE_DATA = False
    results = adaptive_clustering(
        X,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        TOL=TOL,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS * DIM,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        ERROR_METRIC=ERROR_METRIC,
        INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
        STORE_TREE_DATA=STORE_TREE_DATA,
        LOG_MEM_STATS=False,
        PROFILE_TIMING=True,
    )

    # Only rank 0 gets results; other MPI workers exit cleanly here.
    if _rank != 0:
        MPI.Finalize()
        raise SystemExit(0)

    assert results is not None, "Rank 0 should have results after adaptive_clustering"

    print("-" * 40)
    print(f"Final Cluster Count: {results['centroids'].shape[0]:,}")

    E_cluster_vals = results["E_cluster"]
    counts = results["counts"]
    print(
        "E_cluster stats: min/mean/max =",
        xp.round(E_cluster_vals.min(), 6),
        xp.round(E_cluster_vals.mean(), 6),
        xp.round(E_cluster_vals.max(), 6),
    )
    print(
        "Counts stats: min/mean/max =",
        int(counts.min()),
        f"{counts.mean():.1f}",
        int(counts.max()),
    )

    if STORE_TREE_DATA:
        tree_data = results["tree_data"]
        print("\nTree Data Summary:")
        for lvl, data in tree_data.items():
            print(f"Level {lvl:02}: Clusters: {data['centroids'].shape[0]:,}")

    n_centroids = results["centroids"].shape[0]
    if DIM == 2 and n_centroids <= 10_000:
        title = f"Rosenbrock (a={A_ROSENBROCK}) (TOL={TOL}, ERROR_METRIC={ERROR_METRIC}), N_CLUSTERS={n_centroids:,}"
        plot_adaptive_clustering_results(
            results,
            domain=PROB_LIMS,
            f_fn=rosenbrock,
            img_fpath=img_fpath,
            title=title,
        )
    else:
        print("Skipping plot (d != 2 or too many clusters).")

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
        title = f"MPI x Level Heatmap ({_heatmap_col}) (Train Time={results['profiling_stats']['time_total']:.1f}s) (N_CLUSTERS={n_centroids:,})"
        plot_mpi_level_heatmap(results, _hm_path, values_col=_heatmap_col, title=title)

    # Save npz file of results
    npz_fpath = img_fpath.with_suffix(".npz")
    results = {k: _to_host(v) for k, v in results.items()}
    xp.savez_compressed(npz_fpath, **results)

    # Finalize MPI cleanly, then use os._exit to bypass Python atexit/GC
    # shutdown which triggers a UCX/knndbscan libucp destructor segfault.
    MPI.Finalize()
    os._exit(0)
