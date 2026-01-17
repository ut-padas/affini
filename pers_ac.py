import logging
import time
from typing import Callable, Dict, Optional, Tuple

from cuml.cluster import KMeans as cuKMeans
from cupyx import scatter_max as cupy_scatter_max
from sklearn.cluster import KMeans

from src import IS_CUPY_BACKEND, Array, xp
from src.logging_config import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)


def _scatter_max_1d(out: Array, idx: Array, vals: Array) -> None:
    """out[idx[i]] = max(out[idx[i]], vals[i])"""
    if IS_CUPY_BACKEND:
        cupy_scatter_max(out, idx, vals)
    else:
        xp.maximum.at(out, idx, vals)


def _ensure_2d_f(F: Array) -> Array:
    """Return f(x) as (N,)"""
    F = xp.asarray(F)
    if getattr(F, "ndim", 1) == 2 and F.shape[1] == 1:
        return F[:, 0]
    return F.reshape((-1,))


def _ensure_2d_g(G: Array) -> Array:
    """Return g(x) as (N,d)"""
    G = xp.asarray(G)
    if getattr(G, "ndim", 1) == 3 and G.shape[1] == 1:
        return G[:, 0, :]
    if getattr(G, "ndim", 1) == 2:
        return G
    raise ValueError(f"Unexpected gradient shape {G.shape}; expected (N,d) or (N,1,d).")


def _segment_starts_counts(sorted_labels: Array) -> Tuple[Array, Array, Array]:
    """uniq_labels, starts, counts for a label-sorted array."""
    uniq, starts, counts = xp.unique(sorted_labels, return_index=True, return_counts=True)  # type: ignore
    return uniq, starts, counts


def generate_persistent_convex_samples_from_sorted_active(
    X_sorted: Array,
    starts: Array,
    counts: Array,
    uniq_labels: Array,
    MIN_PTS_PER_CLUSTER: int,
    *,
    SEED: int,
    level: int,
) -> Tuple[Optional[Array], Optional[Array], bool]:
    """
    Vectorized convex padding for a label-sorted active array.

    Returns:
        X_synth: (M,d) or None
        labels_synth: (M,) or None   (global label values, not local indices)
        added: bool
    """
    # If any cluster has <3 points, convex-combo padding is ill-defined (degenerate).
    if xp.any(counts < 3):
        bad = uniq_labels[xp.nonzero(counts < 3)[0][:10]]
        raise RuntimeError(
            f"Level {level}: cluster(s) with <3 points detected. "
            f"Example labels: {bad.tolist()} (showing up to 10)."
        )

    need = xp.maximum(0, MIN_PTS_PER_CLUSTER - counts)  # (K,)
    total_synth = int(xp.sum(need))
    if total_synth == 0:
        return None, None, False

    if IS_CUPY_BACKEND:
        rs = xp.random.RandomState(SEED + level)
        rand = rs.random
    else:
        rng = xp.random.default_rng(SEED + level)
        rand = rng.random

    # Map each synth sample to a cluster k in [0..K-1]
    k_synth = repeats(xp.arange(int(need.shape[0])), need)  # (M,)
    labels_synth = uniq_labels[k_synth]  # (M,)

    # For each synth sample, choose two DISTINCT points inside its cluster segment.
    m = counts[k_synth]  # (M,)
    s0 = starts[k_synth]  # (M,)

    # i ~ Uniform{0,...,m-1}
    i = (xp.floor(rand(total_synth) * m)).astype(xp.int64)
    # j ~ Uniform{0,...,m-2}, then shift to skip i
    j = (xp.floor(rand(total_synth) * (m - 1))).astype(xp.int64)
    j = j + (j >= i)

    idx_i = s0 + i
    idx_j = s0 + j

    a = rand((total_synth, 1)).astype(X_sorted.dtype)  # (M,1)
    X_synth = a * X_sorted[idx_i] + (1.0 - a) * X_sorted[idx_j]  # (M,d)
    return X_synth, labels_synth.astype(xp.int64), True


# -----------------------------------------------------------------------------
# Batched linear algebra building blocks (with shrinkage)
# -----------------------------------------------------------------------------
def batched_calculate_Q(
    DeltaX: Array,
    starts: Array,
    counts: Array,
    K: int,
    *,
    Q_EPS_AXIS: float = 1e-8,
    Q_SHRINK: float = 0.0,
) -> Array:
    """
    PCA-aligned trust ellipsoid Q_k using max-abs projections, with covariance shrinkage.
    """
    # Cov_k = sum dx dx^T
    XX_all = DeltaX[:, :, None] * DeltaX[:, None, :]  # (N,d,d)
    Cov = xp.add.reduceat(XX_all, starts)  # (K,d,d)
    Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

    if Q_SHRINK > 0.0:
        # Sigma = (1-a)Cov + a*(tr(Cov)/d)*I, per cluster
        d = int(Cov.shape[1])
        tr = xp.trace(Cov, axis1=1, axis2=2)  # (K,)
        scale = tr / max(d, 1)
        I = xp.eye(d, dtype=Cov.dtype)[None, :, :]  # (1,d,d)
        Cov = (1.0 - Q_SHRINK) * Cov + Q_SHRINK * (scale[:, None, None] * I)
        Cov = 0.5 * (Cov + Cov.transpose(0, 2, 1))

    # PCA directions U (cols) from eigh
    _, U = xp.linalg.eigh(Cov)

    # projections: proj_i = U_k^T dx_i
    map_n_to_k = repeats(xp.arange(K), counts)  # (N,)
    U_exp = U[map_n_to_k]  # (N,d,d)
    proj = (U_exp.transpose(0, 2, 1) @ DeltaX[:, :, None])[:, :, 0]  # (N,d)
    proj_abs = xp.abs(proj)

    # d_axes[k,j] = max_i |proj[i,j]| within cluster k
    d_axes = xp.zeros((K, proj_abs.shape[1]), dtype=proj_abs.dtype)
    _scatter_max_1d(d_axes, map_n_to_k, proj_abs)

    d_axes = xp.maximum(d_axes, Q_EPS_AXIS)
    Dinv2 = 1.0 / (d_axes * d_axes)  # (K,d)

    # Q = U diag(Dinv2) U^T
    Q = (U * Dinv2[:, None, :]) @ U.transpose(0, 2, 1)
    return 0.5 * (Q + Q.transpose(0, 2, 1))


def batched_estimate_hessian_symmetric(
    DeltaX: Array,
    DeltaG: Array,
    starts: Array,
    *,
    H_RIDGE: float = 1e-18,
) -> Array:
    """
    Symmetric Hessian fit via Sylvester solve in eigen-basis:
      C = sum dx dx^T, E = sum dx dg^T, S = E + E^T
      Solve H C + C H = S.
    """
    XX_all = DeltaX[:, :, None] * DeltaX[:, None, :]  # (N,d,d)
    XG_all = DeltaX[:, :, None] * DeltaG[:, None, :]  # (N,d,d)

    C = xp.add.reduceat(XX_all, starts)
    E = xp.add.reduceat(XG_all, starts)
    C = 0.5 * (C + C.transpose(0, 2, 1))
    S = E + E.transpose(0, 2, 1)

    dvals, U = xp.linalg.eigh(C)
    dvals = xp.maximum(dvals, H_RIDGE)

    Gtilde = U.transpose(0, 2, 1) @ S @ U
    denom = dvals[:, :, None] + dvals[:, None, :]
    F = Gtilde / denom
    H = U @ F @ U.transpose(0, 2, 1)
    return 0.5 * (H + H.transpose(0, 2, 1))


def batched_sqrt_abs_symmetric(H: Array) -> Array:
    """M_k = sqrt(|H_k|) for symmetric H_k."""
    w, U = xp.linalg.eigh(0.5 * (H + H.transpose(0, 2, 1)))
    w = xp.sqrt(xp.abs(w))
    M = (U * w[:, None, :]) @ U.transpose(0, 2, 1)
    return 0.5 * (M + M.transpose(0, 2, 1))


def batched_top_eigvec(Cov: Array) -> Array:
    """Dominant eigenvector (K,d) of symmetric Cov (K,d,d)."""
    w, V = xp.linalg.eigh(0.5 * (Cov + Cov.transpose(0, 2, 1)))
    # w asc => last is largest
    return V[:, :, -1]


def batched_median_split_decision(
    scores: Array,
    cluster_id: Array,
    starts: Array,
    counts: Array,
) -> Array:
    """
    Global radix-like sort to get per-cluster medians; returns decision (scores >= median_k).
    """
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
    median = 0.5 * (scores_sorted[i0] + scores_sorted[i1])  # (K,)

    return scores >= median[cluster_id]


# -----------------------------------------------------------------------------
# Output buffer management
# -----------------------------------------------------------------------------
def allocate_output_buffers(
    MAX_CLUSTERS_EST: int, d: int, dtype
) -> Tuple[Dict[str, Array], int]:
    """Pre-allocate output buffers."""
    cap = max(1, int(MAX_CLUSTERS_EST))
    buffers = {
        "centroids": xp.empty((cap, d), dtype=dtype),
        "f": xp.empty((cap,), dtype=dtype),
        "g": xp.empty((cap, d), dtype=dtype),
        "Q": xp.empty((cap, d, d), dtype=dtype),
        "E_rel": xp.empty((cap,), dtype=dtype),
        "counts": xp.empty((cap,), dtype=xp.int64),
    }
    return buffers, cap


def adjust_buffer_capacity(
    buffers: Dict[str, Array],
    current_count: int,
    current_cap: int,
    required_total: int,
) -> Tuple[Dict[str, Array], int]:
    """Grow output buffers if needed."""
    if required_total <= current_cap:
        return buffers, current_cap

    d = int(buffers["centroids"].shape[1])
    new_cap = max(required_total, max(1, current_cap) * 2)
    dtype = buffers["centroids"].dtype

    logger.warning(f"Resizing output buffers: {current_cap} -> {new_cap} clusters")

    new_buffers = {
        "centroids": xp.empty((new_cap, d), dtype=dtype),
        "f": xp.empty((new_cap,), dtype=dtype),
        "g": xp.empty((new_cap, d), dtype=dtype),
        "Q": xp.empty((new_cap, d, d), dtype=dtype),
        "E_rel": xp.empty((new_cap,), dtype=dtype),
        "counts": xp.empty((new_cap,), dtype=xp.int64),
    }
    if current_count > 0:
        for k in new_buffers:
            new_buffers[k][:current_count] = buffers[k][:current_count]
    return new_buffers, int(new_cap)


# -----------------------------------------------------------------------------
# Main algorithm: Persistent pool padding + shrinkage + RTOL-only accept
# -----------------------------------------------------------------------------
def adaptive_clustering(
    X: Array,
    *,
    f_fn: Callable[[Array], Array],
    g_fn: Callable[[Array], Array],
    RTOL: float,
    NUM_INITIAL_CLUSTERS: int = 100,
    MIN_PTS_PER_CLUSTER: int = 100,
    MAX_CLUSTERS_EST: int = 100_000,
    SEED: int = 0,
    LOG_MEM_STATS: bool = False,
    # Pool sizing
    SYNTH_BUDGET: Optional[int] = None,
    SYNTH_BUDGET_FACTOR: float = 1.0,
    # Geometry stabilizers
    Q_EPS_AXIS: float = 1e-8,
    Q_SHRINK: float = 0.05,
    COVZ_SHRINK: float = 0.05,
    H_RIDGE: float = 1e-18,
    MAX_LEVELS: int = 10_000,
) -> Dict[str, Array]:
    """
    BFS adaptive clustering with *persistent* convex padding via a preallocated pool.

    Key properties:
      - Synthetic points become indistinguishable from real points (persist across levels).
      - Pool never resizes (GPU-friendly); we just advance a write pointer.
      - RTOL-only accept criterion: accept cluster k if E_rel,k < RTOL.
      - Q and split-direction PCA use covariance shrinkage to prevent PSD issues.
    """
    X0 = xp.asarray(X)
    N0, d = X0.shape

    if SYNTH_BUDGET is None:
        # Rule of thumb: allow up to factor*N synth points (can be tuned).
        SYNTH_BUDGET = int(max(0, SYNTH_BUDGET_FACTOR * N0))

    capacity = int(N0 + SYNTH_BUDGET)
    if capacity < N0:
        raise ValueError("Pool capacity underflow (check SYNTH_BUDGET).")

    if LOG_MEM_STATS:
        log_mem_stats("Start: Adaptive Clustering (pool alloc)")

    # ---- Pool allocation (fixed capacity) ----
    X_pool = xp.empty((capacity, d), dtype=X0.dtype)
    X_pool[:N0] = X0
    n_total = int(N0)

    # Precompute f/g for initial points
    _t0 = time.perf_counter()
    F_pool = xp.empty((capacity,), dtype=X0.dtype)
    G_pool = xp.empty((capacity, d), dtype=X0.dtype)

    F_pool[:N0] = _ensure_2d_f(f_fn(X_pool[:N0]))
    G_pool[:N0] = _ensure_2d_g(g_fn(X_pool[:N0]))
    logger.info(f"Initial f/g on {N0:,} pts: {time.perf_counter()-_t0:.3f}s")

    # Initial global scale (updated only at *next* level after synth insertion)
    tiny = 1e-30
    F_inf = float(xp.max(xp.abs(F_pool[:N0]))) + tiny
    synth_added_prev_level = False

    # Initial clustering on initial (real) set only
    _t1 = time.perf_counter()
    if IS_CUPY_BACKEND:
        km = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
        labels0 = km.fit_predict(X_pool[:N0])
    else:
        km = KMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED, n_init="auto")
        labels0 = km.fit_predict(xp.asarray(X_pool[:N0]))
    labels_pool = xp.empty((capacity,), dtype=xp.int64)
    labels_pool[:N0] = labels0.astype(xp.int64)
    logger.info(f"Initial KMeans: {time.perf_counter()-_t1:.3f}s")

    # Active mask (all points in pool that are still being processed)
    active = xp.zeros((capacity,), dtype=bool)
    active[:N0] = True

    # Outputs
    buffers, out_cap = allocate_output_buffers(MAX_CLUSTERS_EST, d, X0.dtype)
    out_count = 0

    level = 0
    while level <= MAX_LEVELS:
        if LOG_MEM_STATS:
            log_mem_stats(f"Start: L{level:02}")

        # Deferred F_inf update: only at level boundary, only if synth was added last level
        if synth_added_prev_level:
            F_inf = float(xp.max(xp.abs(F_pool[:n_total]))) + tiny

        # Active indices within [0..n_total)
        idx_active = xp.nonzero(active[:n_total])[0]
        n_active = int(idx_active.shape[0])
        if n_active == 0:
            break

        t_level = time.perf_counter()
        synth_added_this_level = False

        # Sort active points by label (segmented layout)
        lab_active = labels_pool[idx_active]
        perm = xp.argsort(lab_active)  # stable not guaranteed, but OK
        idx_sorted = idx_active[perm]
        lab_sorted = lab_active[perm]
        uniq_labels, starts, counts = _segment_starts_counts(lab_sorted)
        K_batch = int(uniq_labels.shape[0])

        # ---------------------------------------------------------------------
        # Persistent padding step: write synth points into pool, mark active
        # ---------------------------------------------------------------------
        X_sorted = X_pool[idx_sorted]  # (n_active, d), label-sorted
        X_synth, labels_synth, added = (
            generate_persistent_convex_samples_from_sorted_active(
                X_sorted,
                starts,
                counts,
                uniq_labels,
                MIN_PTS_PER_CLUSTER,
                SEED=SEED,
                level=level,
            )
        )
        if added:
            M = int(X_synth.shape[0])  # type: ignore
            if n_total + M > capacity:
                raise MemoryError(
                    f"Pool capacity exceeded: need {n_total+M} > {capacity}. "
                    f"Increase SYNTH_BUDGET / SYNTH_BUDGET_FACTOR."
                )

            # Write into pool
            X_pool[n_total : n_total + M] = X_synth
            labels_pool[n_total : n_total + M] = labels_synth
            active[n_total : n_total + M] = True

            # Evaluate f/g for synth points and cache
            F_pool[n_total : n_total + M] = _ensure_2d_f(f_fn(X_synth))
            G_pool[n_total : n_total + M] = _ensure_2d_g(g_fn(X_synth))

            n_total += M
            synth_added_this_level = True

            # Rebuild active segmentation INCLUDING newly-added points (same level)
            idx_active = xp.nonzero(active[:n_total])[0]
            lab_active = labels_pool[idx_active]
            perm = xp.argsort(lab_active)
            idx_sorted = idx_active[perm]
            lab_sorted = lab_active[perm]
            uniq_labels, starts, counts = _segment_starts_counts(lab_sorted)
            K_batch = int(uniq_labels.shape[0])

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Padding")

        # ---------------------------------------------------------------------
        # Per-cluster centroids, Taylor error, accept/reject (RTOL-only)
        # ---------------------------------------------------------------------
        X_sorted = X_pool[idx_sorted]
        F_sorted = F_pool[idx_sorted]
        G_sorted = G_pool[idx_sorted]

        # Centroids mu_k
        sums = xp.add.reduceat(X_sorted, starts)  # (K,d)
        mus = sums / counts[:, None]  # (K,d)

        f_mus = _ensure_2d_f(f_fn(mus))  # (K,)
        g_mus = _ensure_2d_g(g_fn(mus))  # (K,d)

        # Map each point to local cluster index
        map_n_to_k = repeats(xp.arange(K_batch), counts)  # (n_active,)
        dx = X_sorted - mus[map_n_to_k]  # (n_active,d)
        f_lin = f_mus[map_n_to_k] + xp.sum(dx * g_mus[map_n_to_k], axis=1)
        err_point = xp.abs(F_sorted - f_lin)

        E_abs = xp.zeros((K_batch,), dtype=err_point.dtype)
        _scatter_max_1d(E_abs, map_n_to_k, err_point)
        E_rel = E_abs / F_inf
        is_good = E_rel < RTOL

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Error")

        # ---------------------------------------------------------------------
        # Finalize good clusters: compute Q, write outputs, deactivate their points
        # ---------------------------------------------------------------------
        good_idx = xp.nonzero(is_good)[0]
        if int(good_idx.shape[0]) > 0:
            # Build a packed DeltaX for only-good points (contiguous)
            point_good_mask = is_good[map_n_to_k]
            dx_good = dx[point_good_mask]

            good_counts = counts[good_idx]
            good_starts = xp.r_[0, xp.cumsum(good_counts)[:-1]]

            Qs = batched_calculate_Q(
                dx_good,
                good_starts,
                good_counts,
                int(good_idx.shape[0]),
                Q_EPS_AXIS=Q_EPS_AXIS,
                Q_SHRINK=Q_SHRINK,
            )

            n_good = int(good_idx.shape[0])
            end_out = out_count + n_good
            buffers, out_cap = adjust_buffer_capacity(
                buffers, out_count, out_cap, end_out
            )

            buffers["centroids"][out_count:end_out] = mus[good_idx]
            buffers["f"][out_count:end_out] = f_mus[good_idx]
            buffers["g"][out_count:end_out] = g_mus[good_idx]
            buffers["Q"][out_count:end_out] = Qs
            buffers["E_rel"][out_count:end_out] = E_rel[good_idx]
            buffers["counts"][out_count:end_out] = counts[good_idx].astype(xp.int64)
            out_count = end_out

            # Deactivate all points (real+synth) in good clusters
            global_ids_good = idx_sorted[point_good_mask]
            active[global_ids_good] = False

        # If no bad clusters remain, done
        bad_idx = xp.nonzero(~is_good)[0]
        if int(bad_idx.shape[0]) == 0:
            synth_added_prev_level = synth_added_this_level
            break

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Finalize good")

        # ---------------------------------------------------------------------
        # Split bad clusters (Hessian-informed), update labels in-place for ALL points
        # ---------------------------------------------------------------------
        point_bad_mask = (~is_good)[map_n_to_k]
        idx_bad = idx_sorted[point_bad_mask]  # global indices in pool
        dx_bad = dx[point_bad_mask]
        G_bad = G_sorted[point_bad_mask]
        gmu_bad = g_mus[map_n_to_k][point_bad_mask]
        dG_bad = G_bad - gmu_bad

        bad_counts = counts[bad_idx]
        bad_starts = xp.r_[0, xp.cumsum(bad_counts)[:-1]]
        K_bad = int(bad_idx.shape[0])

        # Hessian fit + |H|^{1/2}
        H = batched_estimate_hessian_symmetric(
            dx_bad, dG_bad, bad_starts, H_RIDGE=H_RIDGE
        )
        M = batched_sqrt_abs_symmetric(H)

        # z = M_k dx
        map_bad_n_to_k = repeats(xp.arange(K_bad), bad_counts)
        M_exp = M[map_bad_n_to_k]
        Z = (M_exp @ dx_bad[:, :, None])[:, :, 0]

        # CovZ with shrinkage
        ZZ_all = Z[:, :, None] * Z[:, None, :]
        CovZ = xp.add.reduceat(ZZ_all, bad_starts)
        CovZ = 0.5 * (CovZ + CovZ.transpose(0, 2, 1))
        if COVZ_SHRINK > 0.0:
            tr = xp.trace(CovZ, axis1=1, axis2=2)
            scale = tr / max(d, 1)
            I = xp.eye(d, dtype=CovZ.dtype)[None, :, :]
            CovZ = (1.0 - COVZ_SHRINK) * CovZ + COVZ_SHRINK * (scale[:, None, None] * I)
            CovZ = 0.5 * (CovZ + CovZ.transpose(0, 2, 1))

        v = batched_top_eigvec(CovZ)
        v_exp = v[map_bad_n_to_k]
        scores = xp.sum(Z * v_exp, axis=1)

        split_right = batched_median_split_decision(
            scores, map_bad_n_to_k, bad_starts, bad_counts
        )

        # Update labels for ALL points in bad subset (including synth)
        old_label_values = uniq_labels[bad_idx]  # (K_bad,)
        old_per_point = old_label_values[map_bad_n_to_k]  # (N_bad,)

        # left: old*2, right: old*2+1
        new_labels = old_per_point * 2 + split_right.astype(xp.int64)
        labels_pool[idx_bad] = new_labels

        # End-of-level bookkeeping
        synth_added_prev_level = synth_added_this_level

        logger.info(
            f"Level {level:03}: active={n_active:,}, clusters={K_batch:,}, "
            f"good={int(good_idx.shape[0]):,}, bad={int(bad_idx.shape[0]):,}, "
            f"pool_used={n_total:,}/{capacity:,}, time={time.perf_counter()-t_level:.3f}s"
        )

        level += 1

    # Truncate outputs
    final_count = int(out_count)
    logger.info(f"Adaptive clustering complete: {final_count:,} final clusters.")
    return {k: buffers[k][:final_count] for k in buffers}


if __name__ == "__main__":
    import argparse
    from functools import partial

    from src import FIGURES_DIR
    from src.logging_config import setup_logging
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    setup_logging(console_level=logging.DEBUG)

    AC_PLOTS_DIR = FIGURES_DIR / "adaptive_clustering_rosenbrock"
    AC_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
        "--n-train", type=int, default=100_000, help="Number of training samples"
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
        "--rosenbrock-a",
        type=float,
        default=100.0,
        help="Rosenbrock function parameter 'a'",
    )

    args = parser.parse_args()
    DIM = args.dim
    TOL = args.tol
    N_TRAIN = args.n_train
    NUM_INITIAL_CLUSTERS = args.n_init_clusters
    MIN_PTS_PER_CLUSTER = args.n_min_per_cluster
    A_ROSENBROCK = args.rosenbrock_a

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
        RTOL=TOL,
        NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
        MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
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

    if DIM == 2 and len(results["centroids"]) <= 3000:
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
            f"Adaptive Clustering: Rosenbrock (a={A_ROSENBROCK}) (TOL={TOL}, N_INIT={NUM_INITIAL_CLUSTERS}, N_MIN={MIN_PTS_PER_CLUSTER}), Final Clusters: {results['centroids'].shape[0]:,}",
            fontsize=16,
        )

        plt.subplots_adjust(top=0.88)
        fname = (
            AC_PLOTS_DIR
            / f"adaptive_clustering_rosenbrock_{A_ROSENBROCK}_tol{TOL}_init{NUM_INITIAL_CLUSTERS}_min{MIN_PTS_PER_CLUSTER}_persistent.png"
        )
        plt.savefig(fname, dpi=300)
        print(f"Saved figure to {fname}")
    else:
        print("Skipping plot (d != 2 or too many clusters).")
