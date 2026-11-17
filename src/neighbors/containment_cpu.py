"""
The CPU search of ``src.neighbors.containment`` (Numba): a walk down a bounding-box tree over the
leaves, in its own module so that it is compiled once and cached on disk (closures are not
cacheable), and imported only when a CPU containment search first runs.
"""

import numba as nb
import numpy as np

# Deeper than any tree built over up to 2^60 leaves (two entries per level).
_STACK = 128


@nb.njit(inline="always", cache=True)
def walk(q, node_lo, node_hi, axes, left, start, end, lo, hi, C, U, a, g, T, perm, lim, out):
    """
    Writes the leaves (original indices) containing query q to out, as many as fit, and returns
    how many there are. A subtree is skipped when q lies outside its box on some axis; in a
    bucket, each leaf's own box, then the bound a |dx|^2 + g (u . dx)^2 <= lim, then the d^2
    form decide.
    """
    d = q.shape[0]
    stack = np.empty(_STACK, np.int64)
    stack[0] = 0
    sp = 1
    n_hit = 0
    while sp > 0:
        sp -= 1
        nd = stack[sp]
        inside = True
        for kk in range(d):  # narrowest side first: most likely to reject early
            k = axes[nd, kk]
            if q[k] < node_lo[nd, k] or q[k] > node_hi[nd, k]:
                inside = False
                break
        if not inside:
            continue
        if left[nd] >= 0:
            stack[sp] = left[nd] + 1  # the right child follows the left in the node arrays
            stack[sp + 1] = left[nd]
            sp += 2
            continue
        for j in range(start[nd], end[nd]):
            box = True
            for k in range(d):
                if q[k] < lo[j, k] or q[k] > hi[j, k]:
                    box = False
                    break
            if not box:
                continue
            d2 = 0.0
            p = 0.0
            for k in range(d):
                t = q[k] - C[j, k]
                d2 += t * t
                p += U[j, k] * t
            if a[j] * d2 + g[j] * p * p > lim:
                continue
            Tj = T[perm[j]]
            s = 0.0
            for k in range(d):
                r = 0.0
                for col in range(d):
                    r += Tj[k, col] * (q[col] - C[j, col])
                s += (q[k] - C[j, k]) * r
            if s <= 1.0:
                if n_hit < out.shape[0]:
                    out[n_hit] = perm[j]
                n_hit += 1
    return n_hit


@nb.njit(parallel=True, cache=True)
def collect(Q, node_lo, node_hi, axes, left, start, end, lo, hi, C, U, a, g, T, perm, lim, leaves, nhits):
    """
    Every query's containing leaves in its row of leaves (as many as the row holds, in tree
    order); nhits[i] counts them all, so the caller sees an overflow and retries with longer rows.
    """
    for i in nb.prange(Q.shape[0]):
        nhits[i] = walk(Q[i], node_lo, node_hi, axes, left, start, end, lo, hi, C, U, a, g, T, perm, lim, leaves[i])
