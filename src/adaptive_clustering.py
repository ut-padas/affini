import logging
import time
from typing import Any, Callable, Literal, Optional

from cuml.cluster import KMeans as cuKMeans
from cupyx import scatter_max as cupy_scatter_max
from sklearn.cluster import KMeans

from src import IS_CUPY_BACKEND, Array, xp
from src.logging_config import log_mem_stats
from src.utils import repeats

logger = logging.getLogger(__name__)

TINY_EPS = 1e-12


def scatter_maximum_1d(out: Array, idx: Array, vals: Array) -> None:
    """
    1D segmented max scatter: ``out[idx[i]] = max(out[idx[i]], vals[i])``.
    Uses `cupyx.scatter_max` on GPU and `numpy.maximum.at` semantics on CPU.
    Eg:
        out = [0, 0, 0]
        idx = [0, 1, 0, 2, 1]
        vals = [1, 2, 3, 4, 0]
        scatter_maximum_1d(out, idx, vals)
        # out is now [3, 2, 4]

    Args:
        out: (K,) output array updated in-place.
        idx: (N,) integer indices in [0, K).
        vals: (N,) values to reduce via max per index.
    """
    if IS_CUPY_BACKEND:
        cupy_scatter_max(out, idx, vals)
    else:
        xp.maximum.at(out, idx, vals)


def flatten_function_array(F: Array) -> Array:
    """
    Normalize function values to a flat (N,) array.
    The user-provided `f_fn` may return either (N,), (N,1), or something reshapeable
    to (N,).

    Args:
        F: Array-like of function values.

    Returns:
        F_flat: (N,) array.
    """
    F = xp.asarray(F)
    if getattr(F, "ndim", 1) == 2 and F.shape[1] == 1:
        return F[:, 0]
    return F.reshape((-1,))


def flatten_gradient_array(G: Array) -> Array:
    """
    Normalize gradients to a dense (N, d) array.
    Accepts either:
      - (N, d) gradients, or
      - (N, 1, d) gradients (common when batching Jacobians).

    Args:
        G: Array-like gradients.

    Returns:
        G2: (N, d) gradient array.
    """
    G = xp.asarray(G)
    if getattr(G, "ndim", 1) == 3 and G.shape[1] == 1:
        return G[:, 0, :]
    if getattr(G, "ndim", 1) == 2:
        return G
    raise ValueError(f"Unexpected gradient shape {G.shape}; expected (N,d) or (N,1,d).")


def compute_label_segments(sorted_labels: Array) -> tuple[Array, Array, Array]:
    """Compute segmentation metadata for a label-sorted 1D label array.

    Args:
        sorted_labels: (N,) labels sorted in nondecreasing order.

    Returns:
        uniq: (K,) unique label values (ascending).
        starts: (K,) start indices of each label-run in `sorted_labels`.
        counts: (K,) run lengths.
    """
    uniq, starts, counts = xp.unique(
        sorted_labels, return_index=True, return_counts=True
    )
    return uniq, starts, counts


def generate_sythetic_points(
    X_sorted: Array,
    starts: Array,
    counts: Array,
    uniq_labels: Array,
    MIN_PTS_PER_CLUSTER: int,
    *,
    SEED: int,
    level: int,
) -> tuple[Optional[Array], Optional[Array]]:
    """
    Batch generate synthetic point generation via convex combinations within clusters.

    For each cluster whose active count is below `MIN_PTS_PER_CLUSTER`, generate the
    required number of synthetic points via convex combinations of *two distinct*
    points from the same cluster:

        x_synth = a * x_i + (1 - a) * x_j,    a ~ Uniform(0, 1)

    This stabilizes covariance / Hessian-estimation steps in later geometry, while the
    pool design makes synthetic points persist across levels.

    Args:
        X_sorted: (N_active, d) active points sorted by label.
        starts: (K,) cluster segment starts within `X_sorted`.
        counts: (K,) active counts per cluster.
        uniq_labels: (K,) label values for each segment.
        MIN_PTS_PER_CLUSTER: minimum desired points per cluster.
        SEED: base RNG seed.
        level: current refinement level; used to derive deterministic RNG stream.

    Returns:
        X_synth: (M, d) synthetic points, or None if no padding needed.
        labels_synth: (M,) *global* label values associated to each synthetic point,
            or None if no padding needed.
    """
    # If any cluster has <3 points, convex-combo padding is ill-defined (degenerate).
    if xp.any(counts < 3):
        bad = uniq_labels[xp.nonzero(counts < 3)[0][:10]]
        raise RuntimeError(
            f"Level {level}: cluster(s) with <3 points detected. "
            f"Example labels: {bad.tolist()} (showing up to 10)."
        )

    # How many synthetic points we need per cluster.
    need = xp.maximum(0, MIN_PTS_PER_CLUSTER - counts)  # (K,)
    total_synth = int(xp.sum(need))
    if total_synth == 0:
        return None, None

    rng = xp.random.default_rng(SEED + level)
    rand = rng.random

    # Map each synthetic sample to a local segment id k in [0..K-1].
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
    return X_synth, labels_synth.astype(xp.int64)


def calculate_Q(
    DeltaX: Array,
    starts: Array,
    counts: Array,
    K: int,
    *,
    Q_EPS_AXIS: float = 1e-8,
    Q_SHRINK: float = 0.0,
) -> tuple[Array, Array]:
    """
    PCA-aligned trust ellipsoid $Q_k$ using max-abs projections, with covariance shrinkage.

    This constructs, for each cluster k, a PSD matrix Q_k defining an ellipsoid:
        { y : y^T Q_k y <= 1 }
    aligned with PCA directions of the cluster, with axis lengths chosen by maximum
    absolute projection of centered points onto those axes.

    Args:
        DeltaX: (N, d) centered points for a *packed* batch (cluster segments contiguous).
        starts: (K,) segment starts within `DeltaX`.
        counts: (K,) points per segment.
        K: number of segments/clusters.
        Q_EPS_AXIS: lower bound on per-axis extent (avoids divide-by-zero).
        Q_SHRINK: covariance shrinkage factor in [0,1]. When >0:
            Cov <- (1-a) Cov + a * (tr(Cov)/d) I.

    Returns:
        Q: (K, d, d) PSD ellipsoid matrices.
    """
    # Cov_k = sum_i (dx_i dx_i^T) for points belonging to cluster k.
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

    # PCA directions U (columns).
    _, U = xp.linalg.eigh(Cov)

    # Projection scores onto PCA axes: proj_i = U_k^T dx_i.
    map_n_to_k = repeats(xp.arange(K), counts)  # (N,)
    U_exp = U[map_n_to_k]  # (N,d,d)
    proj = (U_exp.transpose(0, 2, 1) @ DeltaX[:, :, None])[:, :, 0]  # (N,d)
    proj_abs = xp.abs(proj)

    # Axis extents: d_axes[k,j] = max_i |proj[i,j]| within cluster k.
    d_axes = xp.zeros((K, proj_abs.shape[1]), dtype=proj_abs.dtype)
    scatter_maximum_1d(d_axes, map_n_to_k, proj_abs)

    d_axes = xp.maximum(d_axes, Q_EPS_AXIS)
    Dinv2 = 1.0 / (d_axes * d_axes)  # (K,d)

    # Q = U diag(Dinv2) U^T
    Q = (U * Dinv2[:, None, :]) @ U.transpose(0, 2, 1)
    return 0.5 * (Q + Q.transpose(0, 2, 1)), Cov


def estimate_hessian_sqrt_symmetric(
    DeltaX: Array,
    DeltaG: Array,
    starts: Array,
    *,
    H_RIDGE: float = 1e-18,
) -> Array:
    """
    Symmetric Hessian estimation via least-squares fit (Sylvester solve in eigenbasis),
    followed by |H|^{1/2} computation via eigendecomposition.
    Solve: H C + C H = S,  where
        C = sum_i (Δx_i Δx_i^T)
        S = sum_i (Δx_i Δg_i^T + Δg_i Δx_i^T)

    Args:
        DeltaX: (N, d) centered points for a packed batch.
        DeltaG: (N, d) centered gradients, i.e. (g(x_i - g(mu_k)) per point.
        starts: (K,) segment starts.
        H_RIDGE: floor for eigenvalues of C (ridge for numerical stability).

    Returns:
        M: (K, d, d) symmetric matrices with M_k @ M_k = |H_k|.
    """
    XX_all = DeltaX[:, :, None] * DeltaX[:, None, :]  # (N,d,d)
    XG_all = DeltaX[:, :, None] * DeltaG[:, None, :]  # (N,d,d)

    C = xp.add.reduceat(XX_all, starts)  # (K,d,d)
    E = xp.add.reduceat(XG_all, starts)  # (K,d,d)

    # Symmetrize C and S
    C = 0.5 * (C + C.transpose(0, 2, 1))
    S = E + E.transpose(0, 2, 1)

    # 1. Eigendecompose C to find the basis U
    dvals, U = xp.linalg.eigh(C)
    dvals = xp.maximum(dvals, H_RIDGE)  # (K,d)

    # 2. Solve Sylvester Equation in U-basis to get F
    # Gtilde = U.T @ S @ U
    Gtilde = U.transpose(0, 2, 1) @ S @ U  # (K,d,d)
    denom = dvals[:, :, None] + dvals[:, None, :]  # (K,d,d)
    F = Gtilde / denom  # (K,d,d)

    # 3. Diagonalize F directly (F is H in the rotated basis)
    # F is symmetric, so we use eigh
    h_eigvals, V = xp.linalg.eigh(F)

    # 4. Recover eigenvectors of H: Z = U @ V
    # Because H = U @ F @ U.T and F = V @ diag(w) @ V.T
    # H = (U @ V) @ diag(w) @ (U @ V).T = Z @ diag(w) @ Z.T
    Z = U @ V  # (K,d,d)

    # 5. Construct M using sqrt(|w|) and Z
    w_abs_sqrt = xp.sqrt(xp.abs(h_eigvals))  # (K,d)

    # M = Z @ diag(w') @ Z.T
    M = (Z * w_abs_sqrt[:, None, :]) @ Z.transpose(0, 2, 1)

    return 0.5 * (M + M.transpose(0, 2, 1))


def get_dominant_eigenvector(Cov: Array) -> Array:
    """Dominant eigenvector of symmetric covariance matrices.

    Args:
        Cov: (K, d, d) symmetric matrices.

    Returns:
        v: (K, d) top eigenvector per matrix.
    """
    # w asc => last is largest
    w, V = xp.linalg.eigh(Cov)
    return V[:, :, -1]


def compute_median_split_decision(
    scores: Array,
    cluster_id: Array,
    starts: Array,
    counts: Array,
) -> Array:
    """
    Compute per-cluster median split decisions for a packed (clustered) score vector.

    Given per-point scores `scores` and a per-point cluster id `cluster_id` (local ids
    like 0..K-1), return a boolean decision:
        decision[i] = scores[i] >= median(scores within that cluster).

    Implementation note:
        This uses a two-stage argsort (by score, then by cluster id) to obtain scores
        sorted *within clusters* without explicit Python loops.
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


def allocate_output_buffers(
    MAX_CLUSTERS_EST: int, d: int, dtype: Any
) -> tuple[dict[str, Array], int]:
    """Allocate output buffers for finalized clusters.

    Args:
        MAX_CLUSTERS_EST: initial capacity estimate for finalized clusters.
        d: dimensionality of the input space.
        dtype: floating dtype for outputs.

    Returns:
        buffers: dict of arrays sized to `cap`.
        cap: allocated capacity (number of clusters).

    Notes:
        This matches the contract of `adaptive_clustering` return value.
    """
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
    buffers: dict[str, Array],
    current_count: int,
    current_cap: int,
    required_total: int,
) -> tuple[dict[str, Array], int]:
    """Grow output buffers if the finalized cluster count exceeds current capacity.

    Args:
        buffers: current output buffers.
        current_count: number of clusters currently written.
        current_cap: current capacity.
        required_total: required capacity after the next write.

    Returns:
        new_buffers: potentially reallocated buffers.
        new_cap: updated capacity.
    """
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

    # Delete old buffers to free memory.
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
    LOG_MEM_STATS: bool = False,
    # Pool sizing
    SYNTH_BUDGET: Optional[int] = None,
    SYNTH_BUDGET_FACTOR: float = 1.0,
    # Geometry stabilizers
    Q_EPS_AXIS: float = 1e-8,
    Q_SHRINK: float = 0.05,
    COVZ_SHRINK: float = 0.05,
    H_RIDGE: float = 1e-18,
    MAX_LEVELS: int = 50,
) -> dict[str, Array]:
    """
    BFS adaptive clustering with *persistent* convex padding via a preallocated pool.
    This routine constructs a piecewise-local surrogate of a scalar function f(x) over
    a point cloud X by forming clusters whose members are well-approximated by a
    first-order Taylor model about the cluster centroid μ_k:

            f(x) ≈ f(μ_k) + (x - μ_k)^T g(μ_k),

    where g(·) denotes the gradient (∇f). A cluster is accepted using TOL:

            E_abs,k = max_{x in cluster k} |f(x) - f_lin(x; μ_k)|
            E_rel,k = E_abs,k / max_{x in active pool} |f(x)|
            accept if E_rel,k < TOL

    Clusters that fail are split using a second-order-informed direction derived from
    a symmetric Hessian estimate H_k fitted from (Δx, Δg) within the cluster.
    The split direction is computed in transformed coordinates z = |H_k|^{1/2} Δx,
    followed by choosing the top PCA direction of Cov(z). Points are partitioned by a
    within-cluster median split along that direction.

    Key properties (pool design):
        - Synthetic points persist across levels and become indistinguishable from real
            points in later geometry calculations.
        - The pool never resizes (to keep it efficient on GPUs), we only advance a write
        pointer.
        - Function values and gradients are cached for every pool entry.

    Args:
        X: (N0, d) initial real point cloud.
        f_fn: callable mapping (N, d) -> (N,) or (N,1) values.
        g_fn: callable mapping (N, d) -> (N, d) or (N,1,d) gradients.
        TOL: tolerance for accept criterion.
        ERROR_SCALING_METHOD: "global" uses max |f| over all points for normalization when
            computing E_rel; "local" uses max |f| within each cluster.
        NUM_INITIAL_CLUSTERS: initial KMeans cluster count.
        MIN_PTS_PER_CLUSTER: minimum points desired per active cluster (padding target).
        MAX_CLUSTERS_EST: initial allocation for output buffers (can grow).
        SEED: RNG seed for KMeans and padding.
        LOG_MEM_STATS: whether to log memory stats at key points.
        SYNTH_BUDGET: maximum number of synthetic points allowed in the pool.
        SYNTH_BUDGET_FACTOR: if SYNTH_BUDGET is None, sets SYNTH_BUDGET ≈ factor*N0.
        Q_EPS_AXIS: minimum PCA axis extent when building Q (avoids singular Q).
        Q_SHRINK: covariance shrinkage for Q computation.
        COVZ_SHRINK: covariance shrinkage in Z-space for split direction.
        H_RIDGE: eigenvalue floor for Hessian fit stability.
        MAX_LEVELS: maximum refinement iterations.

    Returns:
        outputs: dict with arrays truncated to number of accepted clusters:
            - centroids: (K_final, d)
            - f: (K_final,)
            - g: (K_final, d)
            - Q: (K_final, d, d)
            - E_rel: (K_final,)
            - counts: (K_final,) total (real+synth) points in each accepted cluster
    """
    _start_time = time.perf_counter()
    X0 = xp.asarray(X)  # (N0, d)
    N0, d = X0.shape

    if SYNTH_BUDGET is None:
        SYNTH_BUDGET = int(max(0, SYNTH_BUDGET_FACTOR * N0))

    # Pool capacity
    capacity = int(N0 + SYNTH_BUDGET)
    if LOG_MEM_STATS:
        log_mem_stats("Start: Adaptive Clustering (pool alloc)")

    # ---- Pool allocation (fixed capacity) ----
    # X_pool: (capacity, d) stores all points (real then synthetic as appended).
    X_pool = xp.empty((capacity, d), dtype=X0.dtype)
    X_pool[:N0] = X0
    n_total = int(N0)

    # Precompute and cache f/g for the initial real points.
    # F_pool: (capacity,) scalar values; G_pool: (capacity, d) gradients ∇f.
    _t0 = time.perf_counter()
    F_pool = xp.empty((capacity,), dtype=X0.dtype)
    G_pool = xp.empty((capacity, d), dtype=X0.dtype)

    F_pool[:N0] = flatten_function_array(f_fn(X_pool[:N0]))
    G_pool[:N0] = flatten_gradient_array(g_fn(X_pool[:N0]))
    logger.info(f"Runtime - Initial f/g eval: {time.perf_counter()-_t0:.3f}s")

    # Global scaling for relative error. Updated only at next level boundary after
    # synthetic insertion to avoid oscillating threshold mid-iteration.
    F_inf = xp.max(xp.abs(F_pool[:N0]))
    synth_added_prev_level = False

    # Initial clustering on initial (real) set only.
    _t1 = time.perf_counter()
    if IS_CUPY_BACKEND:
        km = cuKMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
        labels0 = km.fit_predict(X_pool[:N0])
    else:
        km = KMeans(n_clusters=NUM_INITIAL_CLUSTERS, random_state=SEED)
        labels0 = km.fit_predict(xp.asarray(X_pool[:N0]))

    labels_pool = xp.empty((capacity,), dtype=xp.int64)
    labels_pool[:N0] = labels0.astype(xp.int64)
    logger.info(f"Runtime - Initial KMeans: {time.perf_counter()-_t1:.3f}s")

    # Active mask (all pool points currently being processed). `active[i] == True`
    # means pool index i participates in the current refinement frontier.
    active = xp.zeros((capacity,), dtype=bool)
    active[:N0] = True

    # Outputs
    buffers, out_cap = allocate_output_buffers(MAX_CLUSTERS_EST, d, X0.dtype)
    out_count = 0

    level = 0
    while level <= MAX_LEVELS:
        t_level = time.perf_counter()
        if LOG_MEM_STATS:
            log_mem_stats(f"Start: L{level:02}")

        # Deferred F_inf update: level boundary only, and only if padding happened
        # in the previous level.
        if synth_added_prev_level:
            F_inf = xp.max(xp.abs(F_pool[:n_total]))

        if F_inf < TINY_EPS:
            logger.warning(
                f"Level {level:03}: F_inf = {F_inf} is very small "
                f"TOL-based decisions may be unreliable."
            )

        # Active indices within the currently-used pool prefix [0..n_total).
        idx_active = xp.nonzero(active[:n_total])[0]
        n_active = int(idx_active.shape[0])
        if n_active == 0:
            logger.info(f"Level {level:03}: no active points remain, terminating.")
            break

        synth_added_this_level = False

        # Sort active points by label to create contiguous segments per cluster.
        # label_active: (n_active,) global label values.
        label_active = labels_pool[idx_active]
        perm = xp.argsort(label_active)
        idx_sorted = idx_active[perm]
        label_sorted = label_active[perm]
        uniq_labels, starts, counts = compute_label_segments(label_sorted)
        K_batch = int(uniq_labels.shape[0])

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Setup")

        # ---------------------------------------------------------------------
        # Generate per-cluster convex-combination points and append them to the pool.
        # ---------------------------------------------------------------------
        X_sorted = X_pool[idx_sorted]  # (n_active, d), label-sorted
        X_synth, labels_synth = generate_sythetic_points(
            X_sorted,
            starts,
            counts,
            uniq_labels,
            MIN_PTS_PER_CLUSTER,
            SEED=SEED,
            level=level,
        )
        if X_synth is not None:
            M = int(X_synth.shape[0])  # type: ignore
            n_plus_M = n_total + M
            if n_plus_M > capacity:
                raise MemoryError(
                    f"Pool capacity exceeded: need {n_plus_M} > {capacity}. "
                    f"Increase SYNTH_BUDGET / SYNTH_BUDGET_FACTOR."
                )

            # Write new synthetic points into the pool tail and mark them active.
            X_pool[n_total:n_plus_M] = X_synth
            labels_pool[n_total:n_plus_M] = labels_synth
            active[n_total:n_plus_M] = True

            # Evaluate f/g for synthetic points and cache.
            # (These are real gradients of f at synthetic locations.)
            F_pool[n_total:n_plus_M] = flatten_function_array(f_fn(X_synth))
            G_pool[n_total:n_plus_M] = flatten_gradient_array(g_fn(X_synth))

            n_total += M
            synth_added_this_level = True

            # Rebuild active segmentation INCLUDING newly-added points (same level)
            # so that error evaluation and potential split decisions see the padded
            # geometry immediately.
            idx_active = xp.nonzero(active[:n_total])[0]
            label_active = labels_pool[idx_active]
            perm = xp.argsort(label_active)
            idx_sorted = idx_active[perm]
            label_sorted = label_active[perm]
            uniq_labels, starts, counts = compute_label_segments(label_sorted)
            K_batch = int(uniq_labels.shape[0])

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Padding")

        # ---------------------------------------------------------------------
        # Per-cluster centroids, first-order Taylor prediction, and max error.
        # Accept/reject uses TOL-only.
        # ---------------------------------------------------------------------
        X_sorted = X_pool[idx_sorted]
        F_sorted = F_pool[idx_sorted]
        G_sorted = G_pool[idx_sorted]

        # Centroids μ_k for each label segment.
        sums = xp.add.reduceat(X_sorted, starts)  # (K,d)
        mus = sums / counts[:, None]  # (K,d)

        f_mus = flatten_function_array(f_fn(mus))  # (K,)
        g_mus = flatten_gradient_array(g_fn(mus))  # (K,d)

        # Map each point to its local segment id k in [0..K_batch).
        map_n_to_k = repeats(xp.arange(K_batch), counts)  # (n_active,)
        # Centered coordinates Δx.
        dx = X_sorted - mus[map_n_to_k]  # (n_active,d)
        # First-order Taylor model around μ_k.
        f_lin = f_mus[map_n_to_k] + xp.sum(dx * g_mus[map_n_to_k], axis=1)
        err_point = xp.abs(F_sorted - f_lin)

        # Cluster-wise max error E_abs,k = max_i err_point within each segment.
        E_abs = xp.zeros((K_batch,), dtype=err_point.dtype)
        scatter_maximum_1d(E_abs, map_n_to_k, err_point)

        if ERROR_SCALING_METHOD == "global":
            E_rel = E_abs / F_inf
            is_good = E_rel < TOL
        else:
            # Cluster-wise max abs value F_max,k = max_i |f(x_i)| within each segment.
            F_max_cluster = xp.zeros((K_batch,), dtype=F_sorted.dtype)
            scatter_maximum_1d(F_max_cluster, map_n_to_k, xp.abs(F_sorted))

            # Relative error using local normalization.
            E_rel = E_abs / xp.maximum(F_max_cluster, TINY_EPS)
            is_good = (E_rel < TOL) | (E_abs < TOL)

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Error Computation")

        # ---------------------------------------------------------------------
        # Finalize good clusters:
        #  - compute Q from centered points in accepted clusters,
        #  - append results to output buffers,
        #  - deactivate their pool points so they are no longer processed.
        # ---------------------------------------------------------------------
        good_idx = xp.nonzero(is_good)[0]
        if int(good_idx.shape[0]) > 0:
            # Build a packed DeltaX for only-good points (contiguous).
            point_good_mask = is_good[map_n_to_k]
            dx_good = dx[point_good_mask]

            good_counts = counts[good_idx]
            good_starts = xp.r_[0, xp.cumsum(good_counts)[:-1]]

            Qs, _ = calculate_Q(
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

            # Deactivate all points (real + synthetic) in accepted clusters.
            global_ids_good = idx_sorted[point_good_mask]
            active[global_ids_good] = False

        # If no bad clusters remain, done
        bad_idx = xp.nonzero(~is_good)[0]
        if int(bad_idx.shape[0]) == 0:
            synth_added_prev_level = synth_added_this_level
            logger.info(f"Level {level:03}: no bad clusters remain, terminating.")
            break

        if LOG_MEM_STATS:
            log_mem_stats(f"After: L{level:02} Finalize good clusters")

        # ---------------------------------------------------------------------
        # Split bad clusters (Hessian-informed) and update labels in-place for ALL
        # pool points in those clusters (real + synthetic).
        # ---------------------------------------------------------------------
        point_bad_mask = (~is_good)[map_n_to_k]
        idx_bad = idx_sorted[point_bad_mask]  # global indices in pool
        # Extract the packed bad subset.
        dx_bad = dx[point_bad_mask]
        G_bad = G_sorted[point_bad_mask]
        gmu_bad = g_mus[map_n_to_k][point_bad_mask]
        # Center gradients per cluster: Δg = g(x) - g(μ_k).
        dG_bad = G_bad - gmu_bad

        bad_counts = counts[bad_idx]
        bad_starts = xp.r_[0, xp.cumsum(bad_counts)[:-1]]
        K_bad = int(bad_idx.shape[0])

        # Hessian fit (symmetric) + |H|^{1/2} transform.
        M = estimate_hessian_sqrt_symmetric(dx_bad, dG_bad, bad_starts, H_RIDGE=H_RIDGE)

        # Transform coordinates: z = M_k Δx (cluster-wise).
        map_bad_n_to_k = repeats(xp.arange(K_bad), bad_counts)
        M_exp = M[map_bad_n_to_k]
        Z = (M_exp @ dx_bad[:, :, None])[:, :, 0]

        # Cov(Z) with (optional) shrinkage to stabilize eigenvectors.
        ZZ_all = Z[:, :, None] * Z[:, None, :]
        CovZ = xp.add.reduceat(ZZ_all, bad_starts)
        CovZ = 0.5 * (CovZ + CovZ.transpose(0, 2, 1))
        if COVZ_SHRINK > 0.0:
            # Sigma = (1-a)Cov + a*(tr(Cov)/d)*I, per cluster
            tr = xp.trace(CovZ, axis1=1, axis2=2)
            scale = tr / max(d, 1)
            I = xp.eye(d, dtype=CovZ.dtype)[None, :, :]
            CovZ = (1.0 - COVZ_SHRINK) * CovZ + COVZ_SHRINK * (scale[:, None, None] * I)
            CovZ = 0.5 * (CovZ + CovZ.transpose(0, 2, 1))

        # Split direction per cluster = top eigenvector of CovZ.
        v = get_dominant_eigenvector(CovZ)
        v_exp = v[map_bad_n_to_k]
        # Per-point scalar projection scores in Z-space.
        scores = xp.sum(Z * v_exp, axis=1)

        split_right = compute_median_split_decision(
            scores, map_bad_n_to_k, bad_starts, bad_counts
        )

        # Update labels for ALL points in bad subset (including synthetic).
        # Tree-like label encoding: left child = old*2, right child = old*2+1.
        old_label_values = uniq_labels[bad_idx]  # (K_bad,)
        old_per_point = old_label_values[map_bad_n_to_k]  # (N_bad,)

        # left: old*2, right: old*2+1
        new_labels = old_per_point * 2 + split_right.astype(xp.int64)
        labels_pool[idx_bad] = new_labels

        # End-of-level bookkeeping
        synth_added_prev_level = synth_added_this_level
        logger.info(
            f"Level {level:03}: Clusters={K_batch:,}, active={n_active:,}, "
            f"good={int(good_idx.shape[0]):,}, bad={int(bad_idx.shape[0]):,}, "
            f"synth={n_total - N0:,}, "
            f"pool_used={n_total:.1e}/{capacity:.1e} ({n_total/capacity:.0%}), "
            f"F_inf={F_inf:.1e}"
        )
        logger.info(f"Runtime - Level {level:03}: {time.perf_counter()-t_level:.3f}s")
        level += 1

    if level > MAX_LEVELS:
        logger.warning(f"Reached MAX_LEVELS={MAX_LEVELS}, terminating.")

    # Truncate outputs
    final_count = int(out_count)
    logger.info(f"Adaptive clustering complete: {final_count:,} final clusters.")
    logger.info(f"Runtime - Total: {time.perf_counter() -_start_time:.3f}s")
    return {k: buffers[k][:final_count] for k in buffers}


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
