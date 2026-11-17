"""
Exact trust-region containment: every leaf whose trust ellipsoid ``(x - c_i)^T T_i (x - c_i) <= 1``
contains a query, for ``AdaptiveSurrogate.predict(exact_outside=True)``.

``predict`` looks for a query's leaf among its ``kappa`` Euclidean-nearest centres. A long
ellipsoid whose centre lies farther away can contain the query and still be missed, so the
outside flag, and ``p_o``, depend on ``kappa``. This search makes the flag exact for the queries
that ``predict`` flags. A pair is decided by three tests, each a necessary condition for the next
and computed in float64 from one batched ``eigh`` per model:

1. The ellipsoid's axis-aligned bounding box, half-width ``sqrt((T^-1)_kk)`` on axis k.
2. ``a |dx|^2 + g (u . dx)^2 <= 1``, a lower bound on ``dx^T T dx`` (``a <= lambda_min``,
   ``u`` the eigenvector of ``lambda_max``, ``g = lambda_max - lambda_min``).
3. The d^2 quadratic form itself.

The eigenvalues are lowered by their rounding error before the boxes and the bound are formed, so
neither can reject a containing leaf.

Trained leaves are needles: one long axis and thin elsewhere. At d = 10 on a 181,543-leaf model
the median ellipsoid's ball radius is about half the domain, so balls overlap almost every query,
but a needle's bounding box is about 1/sqrt(d) of its length on each axis. On the CPU the leaves
therefore sit in a tree of median splits with buckets of ``CPU_BUCKET`` leaves, each node holding
the union of its leaves' boxes, and a query only visits the subtrees whose box contains it: about
7% of the leaves there, against 66% for the same tree with balls. The GPU tests every pair in one
fused kernel (``contained_mask``), whose throughput makes the scan cheap.

Cost: one tree walk per query on the CPU, about 2 d operations per pair on the GPU, plus d^2 for
the few pairs that pass the first two tests. Cheap for the queries ``predict`` flags near the
training data; it grows towards a search of every leaf when most queries lie far from it.
"""

import numpy as np

from src import IS_CUPY_BACKEND, Array, kernels, xp

# Bytes of trust matrices per batched eigh call (its workspace is a few copies).
EIG_BLOCK_BYTES = 256 << 20
# Bytes of the GPU's uint8 (queries x leaves) mask per launch.
MASK_BLOCK_BYTES = 256 << 20
# Leaves per bucket of the CPU's box tree; buckets of 16 visited 7% of the leaves at d = 10, 64
# visited 27%, as larger buckets' union boxes widen.
CPU_BUCKET = 16
# Axes (the widest by centre extent) each tree node tries for its split.
SPLIT_CANDIDATES = 4
# Containing leaves a query may report on the CPU before its row is retried with more room.
CPU_HITS_PER_QUERY = 64
_U64 = 2.0**-53


def build_box_tree(C: np.ndarray, lo: np.ndarray, hi: np.ndarray, bucket: int) -> dict[str, np.ndarray]:
    """
    A tree of median splits over the leaf centres ``C``, down to ``bucket`` leaves, with every
    node's box the union of its leaves' boxes ``[lo, hi]``. Each node is split at the median of
    the axis, among its ``SPLIT_CANDIDATES`` widest, whose two children have the least total box
    volume relative to the node's (the chance that a query uniform in the node's box enters a
    child): overlapping children make a query visit both. Returns ``perm`` (the leaves in tree
    order) and per node ``node_lo``, ``node_hi``, ``start``, ``end`` (its leaves are
    ``perm[start:end]``), ``left`` (-1 for a bucket; the right child is ``left + 1``) and
    ``axes``, the axes from the narrowest box side to the widest, the order in which a query is
    most likely to fall outside early. Children are numbered after their parent.
    """
    n, d = C.shape
    perm = np.arange(n)
    left, start, end = [-1], [0], [n]
    stack = [0]
    while stack:
        nd = stack.pop()
        s, e = start[nd], end[nd]
        if e - s <= bucket:
            continue
        ids = perm[s:e]
        mid = s + (e - s) // 2
        width = hi[ids].max(axis=0) - lo[ids].min(axis=0)
        best, best_cost = None, np.inf
        for axis in np.argsort(C[ids].min(axis=0) - C[ids].max(axis=0))[:SPLIT_CANDIDATES]:
            order = ids[np.argsort(C[ids, axis], kind="stable")]
            cost = 0.0
            for part in (order[: mid - s], order[mid - s :]):
                ratio = (hi[part].max(axis=0) - lo[part].min(axis=0)) / width
                cost += np.exp(np.sum(np.log(np.clip(np.where(np.isfinite(ratio), ratio, 1.0), 1e-300, 1.0))))
            if cost < best_cost:
                best, best_cost = order, cost
        perm[s:e] = best
        left[nd] = len(left)
        for a, b in ((s, mid), (mid, e)):
            left.append(-1)
            start.append(a)
            end.append(b)
        stack += [left[nd] + 1, left[nd]]
    left_a, start_a, end_a = (np.asarray(v, dtype=np.int64) for v in (left, start, end))
    node_lo, node_hi = np.empty((len(left), d)), np.empty((len(left), d))
    lo_p, hi_p = lo[perm], hi[perm]
    for nd in range(len(left) - 1, -1, -1):  # children before parents
        if left_a[nd] < 0:
            node_lo[nd] = lo_p[start_a[nd] : end_a[nd]].min(axis=0)
            node_hi[nd] = hi_p[start_a[nd] : end_a[nd]].max(axis=0)
        else:
            c = left_a[nd]
            node_lo[nd] = np.minimum(node_lo[c], node_lo[c + 1])
            node_hi[nd] = np.maximum(node_hi[c], node_hi[c + 1])
    axes = np.argsort(node_hi - node_lo, axis=1, kind="stable")  # inf widths last
    return {"perm": perm, "node_lo": node_lo, "node_hi": node_hi, "left": left_a, "start": start_a, "end": end_a,
            "axes": np.ascontiguousarray(axes)}  # fmt: skip


class TrustRegionContainment:
    """
    Containment index over a model's leaves, built once (the bounds, and on the CPU the box tree)
    and reused by every query batch. Holds references to the caller's centroids and trust
    matrices, which must not change while it is in use.
    """

    def __init__(self, centroids: Array, trust_matrices: Array):
        self.centroids = xp.ascontiguousarray(xp.asarray(centroids, dtype=xp.float64))
        self.trust_matrices = xp.ascontiguousarray(xp.asarray(trust_matrices, dtype=xp.float64))
        n, d = self.centroids.shape
        lam_min, lam_max, u = xp.empty(n), xp.empty(n), xp.empty((n, d))
        half = xp.empty((n, d))
        rows = max(1, EIG_BLOCK_BYTES // (8 * d * d))
        for s in range(0, n, rows):
            w, V = xp.linalg.eigh(self.trust_matrices[s : s + rows])
            # eigh is backward stable: its eigenpairs are exact for a matrix within about
            # d u64 |T| of T. Lowering every eigenvalue by 32 d u64 lambda_max gives a matrix
            # below T, whose ellipsoid holds T's: its boxes and its bound can only accept more.
            err = 32.0 * d * _U64 * xp.abs(w[:, -1:])
            safe = w - err
            ok = xp.all(safe > 0, axis=1)
            h2 = xp.einsum("nkj,nj->nk", V * V, 1.0 / xp.where(safe > 0, safe, 1.0))
            half[s : s + rows] = xp.where(ok[:, None], xp.sqrt(h2) * (1.0 + 1e-12), xp.inf)
            lam_min[s : s + rows], lam_max[s : s + rows], u[s : s + rows] = safe[:, 0], w[:, -1], V[:, :, -1]
        self.a = lam_min
        self.g = xp.maximum(lam_max - (lam_min + 2 * 32.0 * d * _U64 * xp.abs(lam_max)), 0.0)
        self.u = xp.ascontiguousarray(u)
        # The bound's own d-term sums round by about d u64 of their value, near 1 where it matters.
        self.lim = 1.0 + 4.0 * d * _U64
        if IS_CUPY_BACKEND:
            self.centroids_t = xp.ascontiguousarray(self.centroids.T)  # (d, n): coalesced reads
            self.u_t = xp.ascontiguousarray(self.u.T)
        else:
            C = self.centroids
            tree = build_box_tree(C, C - half, C + half, CPU_BUCKET)
            perm = tree["perm"]
            # The leaves' own arrays in tree order, so a bucket reads contiguous rows; the trust
            # matrices stay in place and are read through perm by the few pairs that need them.
            self.tree = {**tree, "lo": np.ascontiguousarray((C - half)[perm]),
                         "hi": np.ascontiguousarray((C + half)[perm]), "C": np.ascontiguousarray(C[perm]),
                         "U": np.ascontiguousarray(self.u[perm]), "a": self.a[perm], "g": self.g[perm]}  # fmt: skip

    def containing(self, queries: Array) -> tuple[Array, Array, Array]:
        """
        ``(rows, leaves, dist_sq)`` of every (query, leaf) pair with ``dist_sq = dx^T T dx <= 1``,
        sorted by row, then by leaf.
        """
        Q = xp.ascontiguousarray(xp.asarray(queries, dtype=xp.float64))
        m = Q.shape[0]
        n = int(self.centroids.shape[0])
        if m == 0 or n == 0:
            empty = xp.empty(0, dtype=xp.int64)
            return empty, empty, xp.empty(0)
        rows, leaves = self._pairs_gpu(Q) if IS_CUPY_BACKEND else self._pairs_cpu(Q)
        # The same expression as predict's candidate distances, on the few pairs found.
        dx = Q[rows] - self.centroids[leaves]
        dist = xp.einsum("ni,nij,nj->n", dx, self.trust_matrices[leaves], dx)
        keep = dist <= 1.0  # predict's test, in case its rounding differs from the search's
        return rows[keep], leaves[keep], dist[keep]

    def _pairs_cpu(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from src.neighbors.containment_cpu import collect

        t = self.tree
        args = (t["node_lo"], t["node_hi"], t["axes"], t["left"], t["start"], t["end"], t["lo"], t["hi"], t["C"], t["U"],
                t["a"], t["g"], self.trust_matrices, t["perm"], self.lim)  # fmt: skip
        m = Q.shape[0]
        leaves = np.empty((m, CPU_HITS_PER_QUERY), np.int64)
        nhits = np.empty(m, np.int64)
        collect(Q, *args, leaves, nhits)
        rows = np.repeat(np.arange(m, dtype=np.int64), nhits)
        if np.all(nhits <= CPU_HITS_PER_QUERY):
            found = leaves[np.arange(CPU_HITS_PER_QUERY)[None, :] < nhits[:, None]]
        else:  # rows that found more leaves than they hold: those again, with room for all
            over = np.flatnonzero(nhits > CPU_HITS_PER_QUERY)
            wide = np.empty((over.size, int(nhits[over].max())), np.int64)
            collect(Q[over], *args, wide, np.empty(over.size, np.int64))
            parts = [leaves[i, : nhits[i]] for i in range(m)]
            for k, i in enumerate(over):
                parts[i] = wide[k, : nhits[i]]
            found = np.concatenate(parts)
        order = np.lexsort((found, rows))
        return rows[order], found[order]

    def _pairs_gpu(self, Q: Array) -> tuple[Array, Array]:
        kernel = kernels.module("containment").get_function("contained_mask")
        m, d = Q.shape
        n = int(self.centroids.shape[0])
        block = int(max(1, min(65535, MASK_BLOCK_BYTES // n)))
        mask = xp.empty((min(block, m), n), dtype=xp.uint8)
        found_rows, found_leaves = [], []
        for s in range(0, m, block):
            b = min(block, m - s)
            kernel(((n + 255) // 256, b), (256,),
                   (Q[s : s + b], self.centroids_t, self.u_t, self.a, self.g, self.trust_matrices,
                    np.int64(n), np.int32(d), np.float64(self.lim), mask),
                   shared_mem=8 * d)  # fmt: skip
            r, j = xp.nonzero(mask[:b])
            found_rows.append(r.astype(xp.int64) + s)
            found_leaves.append(j.astype(xp.int64))
        return xp.concatenate(found_rows), xp.concatenate(found_leaves)


def best_containing(rows: Array, leaves: Array, dist_sq: Array, n_rows: int, k: int) -> tuple[Array, Array]:
    """
    ``(leaves, dist_sq)``, each ``(n_rows, k)``: every row's ``k`` containing leaves of smallest
    distance (ties: smaller leaf), padded with leaf 0 at distance ``inf``.
    """
    out_i = xp.zeros((n_rows, k), dtype=xp.int64)
    out_d = xp.full((n_rows, k), xp.inf)
    if rows.shape[0] == 0:
        return out_i, out_d
    order = xp.lexsort(xp.stack([leaves.astype(xp.float64), dist_sq, rows.astype(xp.float64)]))
    rows, leaves, dist_sq = rows[order], leaves[order], dist_sq[order]
    rank = xp.arange(rows.shape[0]) - xp.searchsorted(rows, rows, side="left")
    keep = rank < k
    out_i[rows[keep], rank[keep]] = leaves[keep]
    out_d[rows[keep], rank[keep]] = dist_sq[keep]
    return out_i, out_d
