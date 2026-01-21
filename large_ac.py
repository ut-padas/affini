import gc
import logging
import time
from typing import Any, Callable, Literal

import numpy as np
from cuml.cluster import KMeans as cuKMeans
from cupyx import scatter_max as cupy_scatter_max
from sklearn.cluster import KMeans

from src import IS_CUPY_BACKEND, Array, xp
from src.logging_config import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)

TINY_EPS = 1e-12


def cleanup_memory():
    """Forces garbage collection and GPU memory release."""
    gc.collect()
    if IS_CUPY_BACKEND:
        xp.get_default_memory_pool().free_all_blocks()


def scatter_maximum_1d(out: Array, idx: Array, vals: Array) -> None:
    if IS_CUPY_BACKEND:
        cupy_scatter_max(out, idx, vals)
    else:
        xp.maximum.at(out, idx, vals)


def flatten_function_array(F: Array) -> Array:
    F = xp.asarray(F)
    if getattr(F, "ndim", 1) == 2 and F.shape[1] == 1:
        return F[:, 0]
    return F.reshape((-1,))


def flatten_gradient_array(G: Array) -> Array:
    G = xp.asarray(G)
    if getattr(G, "ndim", 1) == 3 and G.shape[1] == 1:
        return G[:, 0, :]
    if getattr(G, "ndim", 1) == 2:
        return G
    raise ValueError(f"Unexpected gradient shape {G.shape}; expected (N,d) or (N,1,d).")


def compute_label_segments(sorted_labels: Array) -> tuple[Array, Array, Array]:
    uniq, starts, counts = xp.unique(
        sorted_labels, return_index=True, return_counts=True
    )
    return uniq, starts, counts


def fill_synthetic_points_chunked(
    X_out: Array,
    labels_out: Array,
    X_pool: Array,
    idx_sorted: Array,
    starts: Array,
    counts: Array,
    uniq_labels: Array,
    synth_needs: Array,
    SEED: int,
    level: int,
    CHUNK_SIZE: int = 1_000_000,
) -> None:
    total_synth = int(X_out.shape[0])
    rng = xp.random.default_rng(SEED + level)

    k_synth_all = repeats(xp.arange(int(synth_needs.shape[0])), synth_needs)
    labels_out[:] = uniq_labels[k_synth_all]

    for chunk_start in range(0, total_synth, CHUNK_SIZE):
        chunk_end = min(total_synth, chunk_start + CHUNK_SIZE)
        curr_size = chunk_end - chunk_start

        k_chunk = k_synth_all[chunk_start:chunk_end]
        m = counts[k_chunk]
        s0 = starts[k_chunk]

        rand_i = rng.random(curr_size)
        i = (xp.floor(rand_i * m)).astype(xp.int64)

        rand_j = rng.random(curr_size)
        j = (xp.floor(rand_j * (m - 1))).astype(xp.int64)
        j = j + (j >= i)

        if isinstance(idx_sorted, slice):
            global_i = (idx_sorted.start or 0) + s0 + i
            global_j = (idx_sorted.start or 0) + s0 + j
        else:
            global_i = idx_sorted[s0 + i]
            global_j = idx_sorted[s0 + j]

        a = rng.random((curr_size, 1)).astype(X_pool.dtype)
        p1 = X_pool[global_i]
        p2 = X_pool[global_j]
        X_out[chunk_start:chunk_end] = a * p1 + (1.0 - a) * p2


def calculate_Q(
    DeltaX: Array,
    starts: Array,
    counts: Array,
    K: int,
    *,
    Q_EPS_AXIS: float = 1e-8,
    Q_SHRINK: float = 0.0,
) -> tuple[Array, Array]:
    N, d = DeltaX.shape
    dtype = DeltaX.dtype

    # Accumulate Covariance
    Cov = xp.empty((K, d, d), dtype=dtype)
    for i in range(d):
        dxi = DeltaX[:, i]
        for j in range(i, d):
            # sum(x_i * x_j) for each cluster
            val = xp.add.reduceat(dxi * DeltaX[:, j], starts)
            Cov[:, i, j] = val
            if i != j:
                Cov[:, j, i] = val
    Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

    # Covariance Shrinkage
    if Q_SHRINK > 0.0:
        d_int = int(Cov.shape[1])
        tr = xp.trace(Cov, axis1=1, axis2=2)
        scale_tr = tr / max(d_int, 1)
        I = xp.eye(d_int, dtype=Cov.dtype)[None, :, :]
        Cov = (1.0 - Q_SHRINK) * Cov + Q_SHRINK * (scale_tr[:, None, None] * I)
        Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

    # Eigendecomposition
    Lambda, U = xp.linalg.eigh(Cov)

    # Invert eigenvalues to get the shape matrix diagonal
    # Since Cov is shrunk/PSD, Lambda > 0 usually, but we clamp for safety
    Lambda = xp.maximum(Lambda, Q_EPS_AXIS)
    Lambda_inv = 1.0 / Lambda  # (K, d)

    # Calculate Projections
    # Computes p = U^T x without creating (N, d, d)
    map_n_to_k = repeats(xp.arange(K), counts)
    proj = xp.empty((N, d), dtype=dtype)

    for j in range(d):
        accum = xp.zeros(N, dtype=dtype)
        for m in range(d):
            # Dot product of point x with j-th eigenvector u_j
            # U[k, m, j] is the m-th component of the j-th eigenvector for cluster k
            u_elem = U[map_n_to_k, m, j]
            accum += u_elem * DeltaX[:, m]
        proj[:, j] = accum

    # Calculate Scaled Covariance Size
    # dist^2 = sum(proj_j^2 / lambda_j)
    L_inv_points = Lambda_inv[map_n_to_k]  # (N, d)
    dist_sq = xp.sum((proj * proj) * L_inv_points, axis=1)  # (N,)

    # Find max distance per cluster
    max_dist_sq = xp.zeros((K,), dtype=proj.dtype)
    scatter_maximum_1d(max_dist_sq, map_n_to_k, dist_sq)

    # Prevent divide-by-zero for single-point clusters
    max_dist_sq = xp.maximum(max_dist_sq, Q_EPS_AXIS)

    # Scale factor alpha = 1 / max_dist
    scale = 1.0 / max_dist_sq  # (K,)

    # Construct Q
    # Q = scale * (U Lambda_inv U^T)
    # effectively new eigenvalues are scale / Lambda
    final_D = scale[:, None] * Lambda_inv  # (K, d)

    # Reconstruct Q matrix
    Q = (U * final_D[:, None, :]) @ U.transpose(0, 2, 1)

    return 0.5 * (Q + Q.transpose(0, 2, 1)), Cov


def estimate_hessian_sqrt_symmetric(
    DeltaX: Array,
    DeltaG: Array,
    starts: Array,
    *,
    H_RIDGE: float = 1e-18,
) -> Array:
    N, d = DeltaX.shape
    K = starts.shape[0]

    C = xp.empty((K, d, d), dtype=DeltaX.dtype)
    E = xp.empty((K, d, d), dtype=DeltaX.dtype)

    for i in range(d):
        dxi = DeltaX[:, i]
        for j in range(i, d):
            val = xp.add.reduceat(dxi * DeltaX[:, j], starts)
            C[:, i, j] = val
            if i != j:
                C[:, j, i] = val

    for i in range(d):
        dxi = DeltaX[:, i]
        for j in range(d):
            E[:, i, j] = xp.add.reduceat(dxi * DeltaG[:, j], starts)

    C = 0.5 * (C + C.transpose(0, 2, 1))
    S = E + E.transpose(0, 2, 1)

    dvals, U = xp.linalg.eigh(C)
    dvals = xp.maximum(dvals, H_RIDGE)

    Gtilde = U.transpose(0, 2, 1) @ S @ U
    denom = dvals[:, :, None] + dvals[:, None, :]
    F = Gtilde / denom

    h_eigvals, V = xp.linalg.eigh(F)
    Z = U @ V
    w_abs_sqrt = xp.sqrt(xp.abs(h_eigvals))

    M = (Z * w_abs_sqrt[:, None, :]) @ Z.transpose(0, 2, 1)
    return 0.5 * (M + M.transpose(0, 2, 1))


def get_dominant_eigenvector(Cov: Array) -> Array:
    w, V = xp.linalg.eigh(Cov)
    return V[:, :, -1]


def compute_median_split_decision(
    scores: Array,
    cluster_id: Array,
    starts: Array,
    counts: Array,
) -> Array:
    N = int(scores.shape[0])
    if N == 0:
        return scores >= 0

    idx_by_score = xp.argsort(scores)
    cid_by_score = cluster_id[idx_by_score]
    idx_group = xp.argsort(cid_by_score)
    order = idx_by_score[idx_group]
    scores_sorted = scores[order]

    i0 = starts + (counts - 1) // 2
    i1 = starts + counts // 2
    median = 0.5 * (scores_sorted[i0] + scores_sorted[i1])
    return scores >= median[cluster_id]


def allocate_output_buffers(
    MAX_CLUSTERS_EST: int, d: int, dtype: Any
) -> tuple[dict, int]:
    cap = max(1, int(MAX_CLUSTERS_EST))
    buffers = {
        "centroids": np.empty((cap, d), dtype=dtype),
        "f": np.empty((cap,), dtype=dtype),
        "g": np.empty((cap, d), dtype=dtype),
        "Q": np.empty((cap, d, d), dtype=dtype),
        "E_rel": np.empty((cap,), dtype=dtype),
        "counts": np.empty((cap,), dtype=np.int64),
    }
    return buffers, cap


def adjust_buffer_capacity(
    buffers: dict, current_count: int, current_cap: int, required_total: int
) -> tuple[dict, int]:
    if required_total <= current_cap:
        return buffers, current_cap
    d = int(buffers["centroids"].shape[1])
    new_cap = max(required_total, max(1, current_cap) * 2)
    dtype = buffers["centroids"].dtype
    logger.warning(
        f"Resizing output buffers (CPU): {current_cap:,} -> {new_cap:,} clusters"
    )
    new_buffers = {
        "centroids": np.empty((new_cap, d), dtype=dtype),
        "f": np.empty((new_cap,), dtype=dtype),
        "g": np.empty((new_cap, d), dtype=dtype),
        "Q": np.empty((new_cap, d, d), dtype=dtype),
        "E_rel": np.empty((new_cap,), dtype=dtype),
        "counts": np.empty((new_cap,), dtype=np.int64),
    }
    if current_count > 0:
        for k in new_buffers:
            new_buffers[k][:current_count] = buffers[k][:current_count]
    del buffers
    return new_buffers, int(new_cap)


def adaptive_clustering(
    X: Array,
    *,
    f_fn: Callable[[Array], Array],
    g_fn: Callable[[Array], Array],
    TOL: float,
    ERROR_SCALING_METHOD: Literal["global", "local"] = "global",
    NUM_INITIAL_CLUSTERS: int = 100,
    MIN_PTS_PER_CLUSTER: int = 100,
    MAX_CLUSTERS_EST: int = 100_000,
    SEED: int = 0,
    SYNTH_BUDGET_FACTOR: float = 2.0,
    Q_EPS_AXIS: float = 1e-8,
    Q_SHRINK: float = 0.05,
    H_RIDGE: float = 1e-18,
    MAX_LEVELS: int = 50,
    LOG_MEM_STATS: bool = False,
) -> dict[str, Array]:
    _start_time = time.perf_counter()
    X0 = xp.asarray(X)
    N0, d = X0.shape
    capacity = int(N0 * SYNTH_BUDGET_FACTOR)

    X_pool = xp.empty((capacity, d), dtype=X0.dtype)
    X_pool[:N0] = X0
    n_total = int(N0)

    _t0 = time.perf_counter()
    F_pool = xp.empty((capacity,), dtype=X0.dtype)
    G_pool = xp.empty((capacity, d), dtype=X0.dtype)
    F_pool[:N0] = flatten_function_array(f_fn(X_pool[:N0]))
    G_pool[:N0] = flatten_gradient_array(g_fn(X_pool[:N0]))
    logger.info(f"Runtime - Initial f/g eval: {time.perf_counter() - _t0:.3f}s")

    F_inf = xp.max(xp.abs(F_pool[:N0]))
    synth_added_prev_level = False

    _t1 = time.perf_counter()
    if IS_CUPY_BACKEND:
        km = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
        labels0 = km.fit_predict(X_pool[:N0])
    else:
        km = KMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
        labels0 = km.fit_predict(xp.asarray(X_pool[:N0]))
    logger.info(f"Runtime - Initial KMeans: {time.perf_counter() - _t1:.3f}s")

    labels_pool = xp.empty((capacity,), dtype=xp.int64)
    labels_pool[:N0] = labels0.astype(xp.int64)
    active = xp.zeros((capacity,), dtype=bool)
    active[:N0] = True

    out_dtype = np.float32 if X0.dtype == xp.float32 else np.float64
    buffers, out_cap = allocate_output_buffers(MAX_CLUSTERS_EST, d, out_dtype)
    out_count = 0

    del X, X0, labels0
    cleanup_memory()

    level = 0
    while level <= MAX_LEVELS:
        t_level = time.perf_counter()
        if LOG_MEM_STATS:
            log_mem_stats(f"Start: L{level:02}")

        if synth_added_prev_level:
            F_inf = xp.max(xp.abs(F_pool[:n_total]))

        if F_inf < TINY_EPS:
            logger.warning(
                f"Level {level:03}: F_inf = {F_inf} is very small "
                f"TOL-based decisions may be unreliable."
            )

        idx_active = xp.nonzero(active[:n_total])[0]
        n_active = int(idx_active.shape[0])
        if n_active == 0:
            logger.info(f"Level {level:03}: no active points remain, terminating.")
            break

        synth_added_this_level = False

        # Sort active points by label
        label_active = labels_pool[idx_active]
        perm = xp.argsort(label_active)
        idx_sorted = idx_active[perm]
        label_sorted = label_active[perm]

        del label_active, perm
        cleanup_memory()

        uniq_labels, starts, counts = compute_label_segments(label_sorted)
        K_batch = int(uniq_labels.shape[0])

        # ---- Generate Synthetic Points ----
        need = xp.maximum(0, MIN_PTS_PER_CLUSTER - counts)
        total_synth = int(xp.sum(need))

        if total_synth > 0:
            M = total_synth
            physical_end = n_total + M
            should_compact = (physical_end > capacity) or (
                n_active < 0.6 * n_total and n_total > 1e6
            )

            if should_compact:
                n_compacted = n_active
                req_cap = n_compacted + M
                new_capacity = (
                    int(max(capacity * 1.5, req_cap + N0 * 0.5))
                    if req_cap > capacity
                    else capacity
                )

                logger.info(
                    f"Evacuating VRAM to System RAM. Resizing: {capacity:.1e} -> {new_capacity:.1e}"
                )

                def to_host(arr_slice):
                    return (
                        arr_slice.get()
                        if hasattr(arr_slice, "get")
                        else np.array(arr_slice)
                    )

                host_X = to_host(X_pool[idx_sorted])
                host_F = to_host(F_pool[idx_sorted])
                host_G = to_host(G_pool[idx_sorted])
                host_lbl = to_host(label_sorted)

                # WIPE GPU
                del X_pool, F_pool, G_pool, labels_pool, active
                cleanup_memory()

                # REALLOCATE
                X_pool = xp.empty((new_capacity, d), dtype=out_dtype)
                F_pool = xp.empty((new_capacity,), dtype=out_dtype)
                G_pool = xp.empty((new_capacity, d), dtype=out_dtype)
                labels_pool = xp.empty((new_capacity,), dtype=xp.int64)
                active = xp.zeros((new_capacity,), dtype=bool)

                # RESTORE
                if hasattr(X_pool, "set"):
                    X_pool[:n_compacted].set(host_X)
                    F_pool[:n_compacted].set(host_F)
                    G_pool[:n_compacted].set(host_G)
                    labels_pool[:n_compacted].set(host_lbl)
                else:
                    X_pool[:n_compacted] = xp.asarray(host_X)
                    F_pool[:n_compacted] = xp.asarray(host_F)
                    G_pool[:n_compacted] = xp.asarray(host_G)
                    labels_pool[:n_compacted] = xp.asarray(host_lbl)

                del host_X, host_F, host_G, host_lbl
                cleanup_memory()

                capacity = new_capacity
                n_total = n_compacted
                active[:n_total] = True
                idx_sorted = slice(0, n_active)
                n_plus_M = n_total + M

            else:
                n_plus_M = n_total + M

            fill_synthetic_points_chunked(
                X_out=X_pool[n_total:n_plus_M],
                labels_out=labels_pool[n_total:n_plus_M],
                X_pool=X_pool,
                idx_sorted=idx_sorted,
                starts=starts,
                counts=counts,
                uniq_labels=uniq_labels,
                synth_needs=need,
                SEED=SEED,
                level=level,
            )
            active[n_total:n_plus_M] = True

            # Eval synthetic
            CHUNK_SIZE = 1_000_000
            for chunk_start in range(0, M, CHUNK_SIZE):
                chunk_end = min(M, chunk_start + CHUNK_SIZE)
                p_start = n_total + chunk_start
                p_end = n_total + chunk_end
                x_batch = X_pool[p_start:p_end]
                F_pool[p_start:p_end] = flatten_function_array(f_fn(x_batch))
                G_pool[p_start:p_end] = flatten_gradient_array(g_fn(x_batch))

            n_total += M
            synth_added_this_level = True

            # Rebuild segment info
            idx_active = xp.nonzero(active[:n_total])[0]
            label_active = labels_pool[idx_active]
            perm = xp.argsort(label_active)
            idx_sorted = idx_active[perm]
            label_sorted = label_active[perm]
            del label_active, perm
            uniq_labels, starts, counts = compute_label_segments(label_sorted)
            K_batch = int(uniq_labels.shape[0])

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Padding")

        # ---- Error Estimation (CHUNKED) ----
        mus = xp.zeros((K_batch, d), dtype=X_pool.dtype)
        ERR_CHUNK = 5_000_000
        N_curr = n_active if isinstance(idx_sorted, slice) else idx_sorted.shape[0]

        def get_idx_chunk(s, e):
            if isinstance(idx_sorted, slice):
                return slice((idx_sorted.start or 0) + s, (idx_sorted.start or 0) + e)
            else:
                return idx_sorted[s:e]

        # 1. Compute Centroids
        for s in range(0, N_curr, ERR_CHUNK):
            e = min(N_curr, s + ERR_CHUNK)
            indices = get_idx_chunk(s, e)
            X_batch = X_pool[indices]
            labels_batch = label_sorted[s:e]

            batch_uniq, batch_starts = xp.unique(labels_batch, return_index=True)
            cluster_idx = xp.searchsorted(uniq_labels, batch_uniq)
            batch_sums = xp.add.reduceat(X_batch, batch_starts)

            if IS_CUPY_BACKEND:
                import cupyx

                cupyx.scatter_add(mus, cluster_idx, batch_sums)
            else:
                xp.add.at(mus, cluster_idx, batch_sums)

            del X_batch, labels_batch, batch_sums

        mus /= counts[:, None]

        # 2. Compute Max Error
        E_abs = xp.zeros((K_batch,), dtype=X_pool.dtype)
        F_max_cluster = xp.zeros((K_batch,), dtype=X_pool.dtype)

        f_mus_all = flatten_function_array(f_fn(mus))
        g_mus_all = flatten_gradient_array(g_fn(mus))

        for s in range(0, N_curr, ERR_CHUNK):
            e = min(N_curr, s + ERR_CHUNK)
            indices = get_idx_chunk(s, e)

            X_batch = X_pool[indices]
            F_batch = F_pool[indices]

            labels_batch = label_sorted[s:e]
            cluster_idx = xp.searchsorted(uniq_labels, labels_batch)

            dx_batch = X_batch - mus[cluster_idx]
            f_lin = f_mus_all[cluster_idx] + xp.sum(
                dx_batch * g_mus_all[cluster_idx], axis=1
            )
            err_batch = xp.abs(F_batch - f_lin)

            scatter_maximum_1d(E_abs, cluster_idx, err_batch)
            if ERROR_SCALING_METHOD == "local":
                scatter_maximum_1d(F_max_cluster, cluster_idx, xp.abs(F_batch))

            del X_batch, F_batch, dx_batch, f_lin, err_batch, labels_batch

        cleanup_memory()

        if ERROR_SCALING_METHOD == "global":
            E_rel = E_abs / F_inf
            is_good = E_rel < TOL
        else:
            E_rel = E_abs / xp.maximum(F_max_cluster, TINY_EPS)
            is_good = (E_rel < TOL) | (E_abs < TOL)

        # ---- Finalize Good Clusters ----
        good_mask = is_good
        good_idx = xp.nonzero(good_mask)[0]
        n_good = int(good_idx.shape[0])

        if n_good > 0:
            point_is_good = repeats(good_mask, counts)
            if isinstance(idx_sorted, slice):
                local_good_indices = xp.nonzero(point_is_good)[0]
                idx_good = (idx_sorted.start or 0) + local_good_indices
            else:
                idx_good = idx_sorted[point_is_good]

            X_good = X_pool[idx_good]
            good_counts = counts[good_idx]
            good_starts = xp.r_[0, xp.cumsum(good_counts)[:-1]]

            map_good_n_to_k = repeats(xp.arange(n_good), good_counts)
            dx_good = X_good - mus[good_idx][map_good_n_to_k]
            del X_good

            Qs, _ = calculate_Q(
                dx_good,
                good_starts,
                good_counts,
                n_good,
                Q_EPS_AXIS=Q_EPS_AXIS,
                Q_SHRINK=Q_SHRINK,
            )
            del dx_good
            cleanup_memory()

            end_out = out_count + n_good
            buffers, out_cap = adjust_buffer_capacity(
                buffers, out_count, out_cap, end_out
            )

            def to_cpu(arr):
                return arr.get() if hasattr(arr, "get") else arr

            buffers["centroids"][out_count:end_out] = to_cpu(mus[good_idx])
            buffers["f"][out_count:end_out] = to_cpu(f_mus_all[good_idx])
            buffers["g"][out_count:end_out] = to_cpu(g_mus_all[good_idx])
            buffers["Q"][out_count:end_out] = to_cpu(Qs)
            buffers["E_rel"][out_count:end_out] = to_cpu(E_rel[good_idx])
            buffers["counts"][out_count:end_out] = to_cpu(
                counts[good_idx].astype(xp.int64)
            )
            out_count = end_out

            active[idx_good] = False
            del idx_good, Qs
            cleanup_memory()

        bad_mask = ~good_mask
        bad_idx = xp.nonzero(bad_mask)[0]
        K_bad = int(bad_idx.shape[0])

        if K_bad == 0:
            logger.info(f"Level {level:03}: no bad clusters remain, terminating.")
            break

        # ---- Split Bad Clusters ----
        point_is_bad = repeats(bad_mask, counts)
        if isinstance(idx_sorted, slice):
            local_bad_indices = xp.nonzero(point_is_bad)[0]
            idx_bad = (idx_sorted.start or 0) + local_bad_indices
        else:
            idx_bad = idx_sorted[point_is_bad]

        dx_bad = (
            X_pool[idx_bad] - mus[bad_idx][repeats(xp.arange(K_bad), counts[bad_idx])]
        )
        G_bad = G_pool[idx_bad]
        gmu_bad = g_mus_all[bad_idx][repeats(xp.arange(K_bad), counts[bad_idx])]
        dG_bad = G_bad - gmu_bad
        del G_bad

        bad_counts = counts[bad_idx]
        bad_starts = xp.r_[0, xp.cumsum(bad_counts)[:-1]]

        M_hess = estimate_hessian_sqrt_symmetric(
            dx_bad, dG_bad, bad_starts, H_RIDGE=H_RIDGE
        )
        del dG_bad

        map_bad_n_to_k = repeats(xp.arange(K_bad), bad_counts)
        N_bad = dx_bad.shape[0]
        Z = xp.empty((N_bad, d), dtype=dx_bad.dtype)
        for i in range(d):
            accum = xp.zeros(N_bad, dtype=dx_bad.dtype)
            for j in range(d):
                m_elem = M_hess[map_bad_n_to_k, i, j]
                accum += m_elem * dx_bad[:, j]
            Z[:, i] = accum
        del dx_bad, M_hess
        cleanup_memory()

        CovZ = xp.empty((K_bad, d, d), dtype=Z.dtype)
        for i in range(d):
            zi = Z[:, i]
            for j in range(i, d):
                val = xp.add.reduceat(zi * Z[:, j], bad_starts)
                CovZ[:, i, j] = val
                if i != j:
                    CovZ[:, j, i] = val
        CovZ = 0.5 * (CovZ + CovZ.transpose(0, 2, 1))
        v = get_dominant_eigenvector(CovZ)
        v_exp = v[map_bad_n_to_k]
        scores = xp.sum(Z * v_exp, axis=1)
        del Z, v, v_exp, CovZ

        split_right = compute_median_split_decision(
            scores, map_bad_n_to_k, bad_starts, bad_counts
        )
        del scores

        old_label_values = uniq_labels[bad_idx]
        old_per_point = old_label_values[map_bad_n_to_k]
        new_labels = old_per_point * 2 + split_right.astype(xp.int64)
        labels_pool[idx_bad] = new_labels

        del idx_bad, new_labels, split_right, old_label_values, old_per_point
        del point_is_bad, bad_mask, bad_idx, good_mask, good_idx
        del mus, f_mus_all, g_mus_all, E_abs, E_rel, F_max_cluster

        cleanup_memory()

        synth_added_prev_level = synth_added_this_level
        logger.info(
            f"Level {level:03}: Clusters={K_batch:,}, active={n_active:,}, "
            f"pool_used={n_total:.1e}/{capacity:.1e} (Active: {n_active/n_total:.0%})"
        )
        logger.info(f"Runtime - Level {level:03}: {time.perf_counter()-t_level:.3f}s")
        level += 1

    if level > MAX_LEVELS:
        logger.warning(f"Reached MAX_LEVELS={MAX_LEVELS}, terminating.")

    final_count = int(out_count)
    logger.info(f"Adaptive clustering complete: {final_count:,} final clusters.")
    logger.info(f"Runtime - Total: {time.perf_counter() -_start_time:.3f}s")

    out_dict = {k: buffers[k][:final_count] for k in buffers}
    out_dict["num_levels"] = level
    return out_dict


if __name__ == "__main__":
    import argparse
    from functools import partial

    from src import FIGURES_DIR
    from src.logging_config import setup_logging
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging(console_level=logging.DEBUG)

    AC_PLOTS_DIR = FIGURES_DIR / "adaptive_clustering_rosenbrock"

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Adaptive GPU Clustering Example with Rosenbrock Function"
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Dimensionality of the problem"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-2, help="Tolerance for training and evaluation"
    )
    parser.add_argument(
        "--n-train", type=float, default=100_000, help="Number of training samples"
    )
    parser.add_argument(
        "--n-init-clusters",
        type=int,
        default=20,
        help="Number of initial clusters for adaptive clustering",
    )
    parser.add_argument(
        "--n-min-per-cluster",
        type=int,
        default=500,
        help="Minimum points per cluster for adaptive clustering",
    )
    parser.add_argument(
        "--error-scaling-method",
        type=str,
        default="global",
        choices=["local", "global"],
        help="Type of error scaling for relative error computation",
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
    ERROR_SCALING_METHOD = args.error_scaling_method
    A_ROSENBROCK = args.rosenbrock_a
    OUTPUT_SUBDIR = args.output_subdir
    FORCE_REEXEC = args.force

    # Check if file already exists
    AC_PLOTS_DIR = AC_PLOTS_DIR / OUTPUT_SUBDIR if OUTPUT_SUBDIR else AC_PLOTS_DIR
    AC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    img_fpath = (
        AC_PLOTS_DIR
        / f"adaptive_clustering_rosenbrock_{A_ROSENBROCK}_tol{TOL}_init{NUM_INITIAL_CLUSTERS}_min{MIN_PTS_PER_CLUSTER}_{ERROR_SCALING_METHOD}.png"
    )
    if DIM == 2 and img_fpath.exists() and not FORCE_REEXEC:
        print(f"Plot file {img_fpath} already exists. Skipping execution.")
        exit(0)

    # Creat partial Rosenbrock functions
    rosenbrock = partial(rosenbrock, a=A_ROSENBROCK)
    rosenbrock_grad = partial(rosenbrock_grad, a=A_ROSENBROCK)

    xp.random.seed(0)
    print(f"Generating {N_TRAIN:,} points (d={DIM})...")
    PROB_LIMS = (-1.0, 3)
    X = xp.random.uniform(PROB_LIMS[0], PROB_LIMS[1], size=(N_TRAIN, DIM))

    print("Starting Adaptive Clustering...")
    results = adaptive_clustering(
        X,
        f_fn=rosenbrock,
        g_fn=rosenbrock_grad,
        TOL=TOL,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
        SYNTH_BUDGET_FACTOR=100,
        ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
    )

    print("-" * 40)
    print(f"Final Cluster Count: {results['centroids'].shape[0]:,}")

    E_rel_vals = results["E_rel"]
    counts = results["counts"]
    print(
        "E_rel stats: min/mean/max =",
        xp.round(E_rel_vals.min(), 6),
        xp.round(E_rel_vals.mean(), 6),
        xp.round(E_rel_vals.max(), 6),
    )
    print(
        "Counts stats: min/mean/max =",
        int(counts.min()),
        f"{counts.mean():.1f}",
        int(counts.max()),
    )

    if DIM == 2 and len(results["centroids"]) <= 10_000:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Ellipse

        if IS_CUPY_BACKEND:
            results = {k: xp.asnumpy(v) for k, v in results.items()}

        # 1 plot 2 subplots (1X2)
        # Subplot 1: scatter of final cluster centroids on contourf
        # Subplot 2: scatter of final cluster centroids with trust ellipses on contourf

        x = np.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        y = np.linspace(PROB_LIMS[0], PROB_LIMS[1], 400)
        X_grid, Y_grid = np.meshgrid(x, y)
        Z_grid = rosenbrock(np.vstack([X_grid.ravel(), Y_grid.ravel()]).T).reshape(
            X_grid.shape
        )

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        cs = axs[0].contourf(
            X_grid,
            Y_grid,
            Z_grid,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(cs, ax=axs[0])
        axs[0].set_title("Rosenbrock Function Contour")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].set_aspect("equal")

        # Subplot 2: with ellipses
        cs2 = axs[1].contourf(
            X_grid,
            Y_grid,
            Z_grid,
            levels=50,
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(cs2, ax=axs[1])
        axs[1].set_title("Adaptive Clustering with Trust Ellipses")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].set_aspect("equal")
        axs[1].scatter(
            results["centroids"][:, 0],
            results["centroids"][:, 1],
            c="red",
            s=10,
            label="Centroids",
        )

        # Draw ellipses
        level = 1.0
        for i in range(results["centroids"].shape[0]):
            mu = results["centroids"][i]
            Q = results["Q"][i]

            # Eigen-decomposition of Q
            w, V = np.linalg.eigh(Q)
            w = np.maximum(w, 1e-12)

            order = np.argsort(w)[::-1]
            w = w[order]
            V = V[:, order]

            a = np.sqrt(level / w[0])
            b = np.sqrt(level / w[1])

            angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))

            ellipse = Ellipse(
                xy=(mu[0], mu[1]),
                width=2 * a,
                height=2 * b,
                angle=angle,
                edgecolor="red",
                facecolor="none",
                lw=1,
                alpha=0.5,
            )
            axs[1].add_patch(ellipse)
        axs[1].legend()
        plt.tight_layout()
        fig.suptitle(
            f"Adaptive Clustering: Rosenbrock (a={A_ROSENBROCK}) (TOL={TOL} ({ERROR_SCALING_METHOD}), N_INIT={NUM_INITIAL_CLUSTERS}, N_MIN={MIN_PTS_PER_CLUSTER}), Final Clusters: {results['centroids'].shape[0]:,}",
            fontsize=16,
        )

        plt.subplots_adjust(top=0.88)
        plt.savefig(img_fpath, dpi=300)
        print(f"Saved figure to {img_fpath}")
    else:
        print("Skipping plot (d != 2 or too many clusters).")
