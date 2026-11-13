"""Per-node split search arithmetic: the batched rows, hard SSE and the B4 soft-oblique objective and step."""
from dataclasses import dataclass, field

import numpy as np

from src import IS_CUPY_BACKEND, Array, _to_host, kernels, xp
from src.numerics import (
    _reduce_sum_1d,
    count_groups,
    det_einsum,
    grouped_cross,
    segment_cross,
    segment_layout,
    spd_factor,
    spd_solve,
    spd_solve_factored,
)
from src.region.geometry import plane_scores


@dataclass
class _SplitRows:
    """
    The rows of the nodes being split, with their affine split features (training views
    only). No per-point outer product is ever formed, which would cost ``N p^2`` memory
    traffic per Gram and dominate at high dimension, and a node's arithmetic never depends
    on the other nodes of its batch. NumPy: batched matrix products over the nodes grouped
    by point count (no padding). CuPy: ``segment_cross`` over each node's contiguous rows,
    reproducible between runs, where cuBLAS-batched products pick algorithms (and so
    rounding) by batch count.
    """

    U: Array  # (N, d) normalized points
    Y: Array  # (N, fd) outputs scaled by each node's tolerance scale
    Phi: Array  # (N, p) the leaf model's split features ([u, 1] for affine leaves)
    labels: Array  # (N,)
    counts: Array  # (k,)
    starts: Array = field(init=False)
    lam: Array = field(init=False)  # (k,) fixed ridge per node
    groups: list = field(init=False)  # NumPy: (node ids (g,), row ids (g, c)) per distinct count c

    def __post_init__(self) -> None:
        self.starts = xp.r_[0, xp.cumsum(self.counts)[:-1]]
        tr = segment_sum(xp.sum(self.Phi**2, axis=1), self.starts)
        self.lam = 1e-8 * tr / self.Phi.shape[1] + 1e-300
        self.groups = [] if IS_CUPY_BACKEND else count_groups(self.counts, self.starts)
        self._blocks = {id(a): [a[rows] for _, rows in self.groups] for a in (self.Phi, self.Y)}
        if IS_CUPY_BACKEND:  # the kernels index with int64 labels; the chunk layout is shared by every sum
            self.labels = self.labels.astype(xp.int64, copy=False)
            self._layout = segment_layout(self.starts, self.counts)

    def reduce(self, a: Array) -> Array:
        """Per-node sums of the rows of ``a`` (N, ...)."""
        if IS_CUPY_BACKEND:
            flat = a.reshape(a.shape[0], -1).astype(xp.float64, copy=False)
            out = segment_cross(flat, None, self.starts, self.counts, layout=self._layout)[:, :, 0]
            return out.reshape(-1, *a.shape[1:])
        return segment_sum(a, self.starts)

    def cross(self, A: Array, B: Array, w: Array | None = None) -> Array:
        """(k, a, b) per-node products ``A_k^T diag(w_k) B_k`` of row arrays (N, a), (N, b)."""
        if IS_CUPY_BACKEND:
            return segment_cross(A, B, self.starts, self.counts, w, symmetric=A is B, layout=self._layout)
        Ab = self._blocks.get(id(A)) or [A[rows] for _, rows in self.groups]
        # The same block objects for A is B: NumPy then takes its symmetric A^T A path, whose
        # rounding the CPU trees were built with.
        Bb = Ab if B is A else self._blocks.get(id(B)) or [B[rows] for _, rows in self.groups]
        return grouped_cross(self.groups, Ab, Bb, w)

    def gram(self, w: Array | None = None) -> tuple[Array, Array]:
        """The weighted Gram ``Phi^T W Phi`` (k, p, p) and right-hand side ``Phi^T W Y`` (k, p, fd)."""
        return self.cross(self.Phi, self.Phi, w), self.cross(self.Phi, self.Y, w)

    def cut_sums(self, order: Array, pos: Array) -> tuple[Array, ...]:
        """
        Gram, right-hand side and ``sum |y|^2`` of both sides of every cut: ``(GL, BL, CL,
        GR, BR, CR)``, each (k, n_cuts, ...), where cut ``i`` of node ``k`` puts the
        ``pos[k, i]`` lowest rows of ``order`` (rows sorted by score within each node) on
        the left. Cut positions depend only on a node's count, so a count group shares its
        bin edges: each bin's sums are one batched product over its contiguous sorted rows,
        and the sides are prefix and suffix sums over bins. That is one pass over the data
        instead of one per cut, and the right side is summed directly rather than taken
        from the node total by subtraction (which cancels near a good fit).
        """
        k, m = int(self.counts.shape[0]), int(pos.shape[1])
        p, fd = self.Phi.shape[1], self.Y.shape[1]
        if IS_CUPY_BACKEND:
            # Every (node, bin) is a contiguous run of the score-sorted rows: one call each.
            edges = xp.concatenate([xp.zeros((k, 1), dtype=xp.int64), pos.astype(xp.int64),
                                    self.counts.astype(xp.int64)[:, None]], axis=1)  # fmt: skip
            b_starts = (self.starts[:, None] + edges[:, :-1]).reshape(-1)
            b_counts = xp.diff(edges, axis=1).reshape(-1)
            P, Ys = self.Phi[order], self.Y[order]
            G = segment_cross(P, P, b_starts, b_counts, symmetric=True).reshape(k, m + 1, p, p)
            B = segment_cross(P, Ys, b_starts, b_counts).reshape(k, m + 1, p, fd)
            C = xp.trace(segment_cross(Ys, Ys, b_starts, b_counts, symmetric=True), axis1=1, axis2=2)
            sides = []
            for S in (G, B, C.reshape(k, m + 1)):
                L, R = xp.empty((k, m) + S.shape[2:]), xp.empty((k, m) + S.shape[2:])
                X = int(np.prod(S.shape[2:], dtype=np.int64))
                _soft_kernel("bin_sides", k * X, S, L, R, np.int64(k), np.int32(m), np.int64(X))
                sides.append((L, R))
            return tuple(s[0] for s in sides) + tuple(s[1] for s in sides)
        out = [xp.empty(shape) for shape in ((k, m, p, p), (k, m, p, fd), (k, m)) * 2]
        pos_host = np.asarray(_to_host(pos))
        y2 = xp.sum(self.Y**2, axis=1)
        for nodes, rows in self.groups:
            c = int(rows.shape[1])
            edges = np.r_[0, pos_host[int(_to_host(nodes[0]))], c]
            sorted_rows = order[rows]  # (g, c): each node's rows by increasing score
            P, Yg, y2g = self.Phi[sorted_rows], self.Y[sorted_rows], y2[sorted_rows]
            bins = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
            G = xp.stack([xp.swapaxes(P[:, a:b], 1, 2) @ P[:, a:b] for a, b in bins], axis=1)
            B = xp.stack([xp.swapaxes(P[:, a:b], 1, 2) @ Yg[:, a:b] for a, b in bins], axis=1)
            C = xp.stack([xp.sum(y2g[:, a:b], axis=1) for a, b in bins], axis=1)
            for slot, S in zip((0, 1, 2), (G, B, C), strict=True):
                out[slot][nodes] = xp.cumsum(S, axis=1)[:, :-1]
                out[slot + 3][nodes] = xp.flip(xp.cumsum(xp.flip(S, axis=1), axis=1), axis=1)[:, 1:]
        return tuple(out)

    def solve(self, G: Array, B: Array) -> Array:
        """Ridge solutions ``(G + lam I)^-1 B`` for (k, ..., p, p) Grams (lam fixed per node)."""
        p = G.shape[-1]
        lam = self.lam.reshape((-1,) + (1,) * (G.ndim - 1))
        return spd_solve(G + lam * xp.eye(p), B)  # SPD: lam > 0 and the weights are floored


def _hard_sse(rows: _SplitRows, w: Array, z: Array, min_side: int) -> tuple[Array, Array]:
    """
    Direct residual sum of squares of the two hard children ``w^T u < z`` / ``>= z``,
    each refit on its own rows (both Grams formed directly, no complementary
    subtraction), and whether both sides keep ``min_side`` points. Used to rank final
    candidates: the prefix-statistic SSE is only a screen (it cancels badly near a good fit).
    """
    side = plane_scores(rows.U, w[rows.labels]) >= z[rows.labels]
    right = side.astype(xp.float64)
    n_right = rows.reduce(right)
    ok = (n_right >= min_side) & (rows.counts - n_right >= min_side) & xp.isfinite(z)
    sse = xp.zeros(int(rows.counts.shape[0]))
    for mask in (1.0 - right, right):
        G, B = rows.gram(mask)
        beta = rows.solve(G, B)  # (k, p, fd)
        res = rows.Y - det_einsum("np,npf->nf", rows.Phi, beta[rows.labels])
        sse = sse + rows.reduce(mask * xp.sum(res**2, axis=1))
    return xp.where(ok, sse, xp.inf), ok


def segment_sum(a: Array, starts: Array) -> Array:
    """
    Sums of the row blocks ``a[starts[i]:starts[i + 1]]`` (the last runs to the end), for any
    number of trailing axes: ``_reduce_sum_1d`` (exact per block on CuPy too), and on NumPy
    2-D ``add.reduceat`` for multi-axis input, the form the CPU trees were built with.
    """
    if a.ndim == 1:
        return _reduce_sum_1d(a.astype(xp.float64), starts)
    if IS_CUPY_BACKEND:
        return _reduce_sum_1d(a, starts)
    return xp.add.reduceat(a.reshape(a.shape[0], -1), starts, axis=0).reshape(-1, *a.shape[1:])


def _soft_gates(rows: _SplitRows, w: Array, z: Array, T: Array, delta: float) -> tuple[Array, Array, Array]:
    """Floored soft memberships ``(W_L, W_R)`` and ``sigma(s) sigma(-s)``, with ``s = (w^T u - z) / T``."""
    if IS_CUPY_BACKEND:  # one fused pass instead of about a dozen elementwise kernels
        WL, WR, sp_sm = (xp.empty(int(rows.U.shape[0])) for _ in range(3))
        _soft_kernel("soft_gates", int(rows.U.shape[0]),
                     rows.U, rows.labels, w, z, T, np.float64(delta), WL, WR, sp_sm, np.int64(rows.U.shape[0]),
                     np.int32(rows.U.shape[1]))  # fmt: skip
        return WL, WR, sp_sm
    out = tuple(np.empty(int(rows.U.shape[0])) for _ in range(3))
    _cpu_soft_gates()(np.ascontiguousarray(rows.U), np.ascontiguousarray(rows.labels), np.ascontiguousarray(w),
                      np.ascontiguousarray(z), np.ascontiguousarray(T), float(delta), *out)  # fmt: skip
    return out


_CPU_SOFT_GATES = None


def _cpu_soft_gates():
    """
    ``_soft_gates`` on the CPU as one serial Numba pass over the rows (compiled on first use): the
    score, the stable logistic of ``_sigmoid`` and the floors without about ten temporaries.
    Serial on purpose: B4 makes thousands of calls, many on small batches, and a parallel region's
    start-up made the whole training 13% slower, while the serial loop made it 19% faster.
    """
    global _CPU_SOFT_GATES
    if _CPU_SOFT_GATES is None:
        import numba as nb

        @nb.njit(cache=True, error_model="numpy")  # IEEE inf/NaN on division by zero, as NumPy and CUDA
        def gates(U, lab, w, z, T, delta, WL, WR, sp_sm):  # pragma: no cover - compiled
            for i in range(U.shape[0]):
                k = lab[i]
                s = 0.0
                for j in range(U.shape[1]):
                    s += U[i, j] * w[k, j]
                s = (s - z[k]) / T[k]
                e = np.exp(-abs(s))
                big, small = 1.0 / (1.0 + e), e / (1.0 + e)
                sp = big if s >= 0 else small
                sm = big if -s >= 0 else small
                WL[i] = delta + (1 - 2 * delta) * sm
                WR[i] = delta + (1 - 2 * delta) * sp
                sp_sm[i] = sp * sm

        _CPU_SOFT_GATES = gates
    return _CPU_SOFT_GATES


_SOFT_MODULE = None


def _soft_kernel(name: str, n_threads: int, *args) -> None:
    """Launch one of the B4 kernels over ``n_threads`` threads (the module is compiled on first use)."""
    global _SOFT_MODULE
    if _SOFT_MODULE is None:
        _SOFT_MODULE = kernels.module("region_b4")
    if n_threads:
        args = tuple(xp.ascontiguousarray(a) if isinstance(a, xp.ndarray) else a for a in args)
        _SOFT_MODULE.get_function(name)(((n_threads + 255) // 256,), (256,), args)


def _soft_step(rows: _SplitRows, w, z, g_w, g_z, J, gap, pen, rL, rR, sp_sum, T, n, delta, active, stops, trials: int):
    """
    One B4 descent step on the GPU in three node-level kernels and one segmented sum: project
    the gradient and set ``alpha = T / |g|``; evaluate the frozen loss of every halving
    ``alpha 2^-m`` (m < ``trials``) at once; let each node take its first trial that passes
    the Armijo and margin tests, which is the step the sequential halving loop on the CPU
    takes, and count why the others stopped. Updates ``w``, ``z``, ``active`` and ``stops``
    in place with no host synchronization.
    """
    k, d = w.shape
    N = int(rows.U.shape[0])
    g_w = xp.ascontiguousarray(g_w)  # projected in place, then read by the selection
    gn2, alpha, margin = xp.empty(k), xp.empty(k), xp.empty(k)
    finite = xp.empty(k, dtype=xp.bool_)
    inv_t, z_t = xp.empty((k, trials)), xp.empty((k, trials))
    eps = float(np.finfo(np.float64).eps)
    _soft_kernel("soft_node_prep", k, w, g_w, z, g_z, J, gap, T, active, gn2, alpha, margin, finite, inv_t, z_t,
                 np.int32(k), np.int32(d), np.int32(trials), np.float64(eps))  # fmt: skip
    # Trials in two groups, the second only for nodes still waiting (most step by trial 3).
    stepped = xp.zeros(k, dtype=xp.bool_)
    groups = [(0, min(4, trials))] + ([(4, trials - 4)] if trials > 4 else [])
    for gi, (m0, mt) in enumerate(groups):
        terms = xp.empty((N, mt))
        _soft_kernel("soft_trial_terms", N, rows.U, rows.labels, w, g_w, inv_t, z_t, alpha, T, np.float64(delta),
                     rL, rR, active, finite, stepped, terms, np.int64(N), np.int32(d), np.int32(trials),
                     np.int32(m0), np.int32(mt))  # fmt: skip
        S = rows.reduce(terms)  # (k, mt)
        _soft_kernel("soft_node_select", k, w, z, g_w, inv_t, z_t, S, pen, n, J, gn2, alpha, margin, finite,
                     sp_sum, T, active, stepped, stops, np.int32(k), np.int32(d), np.int32(trials), np.int32(m0),
                     np.int32(mt), np.int32(int(gi == len(groups) - 1)))  # fmt: skip


def _soft_objective(rows: _SplitRows, w: Array, z: Array, T: Array, delta: float):
    """
    The consistent soft objective ``J = min_B F`` per node and its gradient in ``(w, z)``
    (not yet projected to ``|w| = 1``). With ``B`` at its minimizer, the envelope theorem
    leaves only the gate derivative: ``dJ/ds_i = (r_R,i - r_L,i) (1 - 2 delta) sigma sigma' / n``.
    Also returns the frozen residuals and penalty (for line-search trials), the inner
    solves' suboptimality estimate ``sum_s tr(R^T H^-1 R) / n`` and each node's sum of
    ``sigma(s) sigma(-s)``. ``R = Phi^T W (Y - Phi B) - lam B`` is formed from the data
    residual, not as ``Phi^T W Y - (H + lam I) B`` from the sums, which cancels to rounding
    noise near a fit.
    """
    if IS_CUPY_BACKEND:
        return _soft_objective_fused(rows, w, z, T, delta)
    n = rows.counts.astype(xp.float64)
    lab = rows.labels
    WL, WR, sp_sm = _soft_gates(rows, w, z, T, delta)
    res, pen, gap = [], 0.0, 0.0
    for Wt in (WL, WR):
        H, Bs = rows.gram(Wt)
        beta = rows.solve(H, Bs)
        E = rows.Y - det_einsum("np,npf->nf", rows.Phi, beta[lab])
        res.append(xp.sum(E**2, axis=1))
        pen = pen + rows.lam * xp.sum(beta**2, axis=(1, 2))
        R = rows.cross(rows.Phi, E, Wt) - rows.lam[:, None, None] * beta
        gap = gap + xp.sum(R * rows.solve(H, R), axis=(1, 2)) / n
    rL, rR = res
    J = (rows.reduce(WL * rL + WR * rR) + pen) / n
    a = (rR - rL) * (1 - 2 * delta) * sp_sm / T[lab] / n[lab]
    return J, rows.reduce(a[:, None] * rows.U), -rows.reduce(a), rL, rR, pen, gap, rows.reduce(sp_sm)


def _soft_objective_fused(rows: _SplitRows, w: Array, z: Array, T: Array, delta: float):
    """
    ``_soft_objective`` on the GPU in about 20 launches instead of about 100: both children
    share each segmented product, solve and residual pass, and every per-node sum of the
    objective and gradient comes from one reduction of ``soft_row_terms``.
    """
    n = rows.counts.astype(xp.float64)
    N, (k, d), fd = int(rows.U.shape[0]), w.shape, rows.Y.shape[1]
    p = rows.Phi.shape[1]
    WL, WR, sp_sm = _soft_gates(rows, w, z, T, delta)
    W = xp.stack([WL, WR])  # (2, N)
    H = segment_cross(rows.Phi, rows.Phi, rows.starts, rows.counts, W, symmetric=True, layout=rows._layout)
    Bs = segment_cross(rows.Phi, rows.Y, rows.starts, rows.counts, W, layout=rows._layout)  # (2, k, p, fd)
    H = H + rows.lam[None, :, None, None] * xp.eye(p)
    L = spd_factor(H)  # each child's system is factored once and used for both solves
    beta = spd_solve_factored(L, Bs)
    E, r = xp.empty((2, N, fd)), xp.empty((2, N))
    _soft_kernel("soft_residuals", 2 * N, rows.Phi, rows.Y, rows.labels, beta, E, r, np.int64(N), np.int32(p),
                 np.int32(fd), np.int32(k))  # fmt: skip
    pen = rows.lam * xp.sum(beta**2, axis=(0, 2, 3))
    R = segment_cross(rows.Phi, E, rows.starts, rows.counts, W, layout=rows._layout) - rows.lam[None, :, None, None] * beta
    gap = xp.sum(R * spd_solve_factored(L, R), axis=(0, 2, 3)) / n
    terms = xp.empty((N, d + 3))
    _soft_kernel("soft_row_terms", N, rows.U, rows.labels, WL, WR, sp_sm, r, T, n, np.float64(delta), terms,
                 np.int64(N), np.int32(d))  # fmt: skip
    S = rows.reduce(terms)  # (k, d + 3)
    J = (S[:, 0] + pen) / n
    return J, S[:, 1 : d + 1], -S[:, d + 1], r[0], r[1], pen, gap, S[:, d + 2]


def _unit(a: Array) -> Array:
    """Rows of ``a`` scaled to unit length; zero or non-finite rows become NaN (an invalid split)."""
    norm = xp.linalg.norm(a, axis=-1, keepdims=True)
    ok = xp.isfinite(norm) & (norm > 0)
    return xp.where(ok, a / xp.where(ok, norm, 1.0), xp.nan)


def _sigmoid(s: Array) -> Array:
    """The logistic function without overflow for large ``|s|`` and without clipping."""
    e = xp.exp(-xp.abs(s))
    return xp.where(s >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _interleave(a: Array, b: Array, first: Array) -> Array:
    """Rows of ``a`` where ``first`` is True and rows of ``b`` elsewhere, in order."""
    out = xp.empty((int(first.shape[0]), *a.shape[1:]), dtype=a.dtype)
    out[first] = a
    out[~first] = b
    return out
