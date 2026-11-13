"""Deterministic segmented sums and products over clusters stored contiguously, and per-block ordering helpers."""

import numpy as np

from src import IS_CUPY_BACKEND, Array, _to_host, kernels, xp

if IS_CUPY_BACKEND:  # sources in src/kernels/*.cu
    import cupy as cp

    _seg = kernels.module("segment_cross")
    _stats = kernels.module("cluster_stats")
    _kern_seg_max = _stats.get_function("segmented_max_float64")
    _kern_seg_cross_partial = _seg.get_function("segment_cross_partial")
    _kern_seg_cross_reduce = _seg.get_function("segment_cross_reduce")
    _kern_seg_cross_tiled = _seg.get_function("segment_cross_tiled")


SEGMENT_CHUNK = 256

# From this many output entries per segment, segment_cross loads each chunk into shared
# memory once (same bits as the untiled kernel; fewer FP64 operations and global reads).
SEGMENT_TILED_MIN_E = 32


def repeats(values: Array, repeats: Array) -> Array:
    """
    Repeat elements of an array.

    Args:
        values (Array): The values to repeat (N,)
        repeats (Array): How many times to repeat each value (N,)

    Returns:
        cp.ndarray: The expanded array

    Example::
        values = cp.array([10, 20, 30])
        repeats = cp.array([3, 0, 2])
        result = repeats(values, repeats)
        # result is cp.array([10, 10, 10, 30, 30])
    """
    # 1. Calculate the output positions (offsets)
    # If repeats is [3, 0, 2], offsets is [0, 3, 3, 5]
    offsets = xp.zeros(len(repeats) + 1, dtype=xp.int32)
    offsets[1:] = xp.cumsum(repeats)

    total_size = int(offsets[-1])

    # 2. Create the output array
    # We use -1 as a placeholder to identify start positions
    indices = xp.zeros(total_size, dtype=xp.int32)

    # 3. Mark the start of each segment
    # We use boolean masking to handle cases where repeats[i] == 0
    diff_mask = repeats > 0
    indices[offsets[:-1][diff_mask]] = 1

    # 4. Use cumsum to propagate the segment IDs
    # [1, 0, 0, 1, 0] -> [1, 1, 1, 2, 2]
    # Then subtract 1 to get back to 0-based indexing
    segment_ids = xp.cumsum(indices) - 1

    # 5. Map these indices back to the original values
    # Filter 'values' to only include those actually being repeated
    # to maintain alignment with the cumulative segment IDs
    valid_values = values[diff_mask]
    return valid_values[segment_ids]


def _reduce_max_1d(vals: Array, starts: Array) -> Array:
    """Segmented maximum reduction for contiguous blocks."""
    if IS_CUPY_BACKEND:
        K = starts.shape[0]
        N = vals.shape[0]

        # Ensure proper types for the C-kernel (double, long long) and memory layout
        vals_c = cp.ascontiguousarray(vals, dtype=cp.float64)
        starts_c = cp.ascontiguousarray(starts, dtype=cp.int64)
        out = cp.empty(K, dtype=cp.float64)

        # Standard CUDA block/grid sizing
        threads_per_block = 256
        blocks_per_grid = (K + threads_per_block - 1) // threads_per_block
        _kern_seg_max(
            (blocks_per_grid,), (threads_per_block,), (vals_c, starts_c, out, K, N)
        )

        # Recast if the original array wasn't float64 (e.g., float32)
        if vals.dtype != cp.float64:
            out = out.astype(vals.dtype)
        return out
    else:
        # Standard NumPy fallback
        return xp.maximum.reduceat(vals, starts)


def _reduce_sum_1d(vals: Array, starts: Array) -> Array:
    """
    Sums of the contiguous row blocks starting at ``starts`` (any trailing shape; blocks
    must be non-empty). NumPy: ``add.reduceat``. CuPy, floating point: ``segment_cross``'s
    sequential chunked sums, because CuPy's ``add.reduceat`` subtracts prefix sums, so a
    block's sum depends on, and loses precision to, every row before it in the batch
    (and the float prefix scan varies between runs). Integer sums stay on ``add.reduceat``,
    which is exact for them.
    """
    if not IS_CUPY_BACKEND or not xp.issubdtype(vals.dtype, xp.floating):
        return xp.add.reduceat(vals, starts)
    n = vals.shape[0]
    counts = xp.diff(xp.asarray(starts, dtype=xp.int64), append=n)
    flat = vals.reshape(n, -1)
    return segment_cross(flat, None, starts, counts)[:, :, 0].reshape(-1, *vals.shape[1:]).astype(vals.dtype, copy=False)


def segment_sum(values: Array, labels: Array, K: int) -> Array:
    """
    ``out[k] = sum of values[labels == k]`` (zeros for empty k). On NumPy bit-identical to
    ``xp.add.at`` but several times faster: a CSR 0/1 indicator times the rows adds each
    cluster's rows sequentially in input order, starting from zero. On CuPy the rows are
    grouped by a stable sort and summed with ``add.reduceat`` (a segmented reduction with
    a fixed order), not with ``add.at``, whose atomic order changes the bits between runs.
    """
    if IS_CUPY_BACKEND:  # any label order: a stable sort makes the blocks contiguous
        labels = xp.asarray(labels).astype(xp.int64, copy=False)
        order = xp.argsort(labels)  # CuPy's argsort is stable
        starts, counts = _segments_of(labels[order], K)
        flat = values[order].reshape(values.shape[0], -1).astype(xp.float64, copy=False)
        out = segment_cross(flat, None, starts, counts)[:, :, 0]
        return out.reshape((K, *values.shape[1:])).astype(values.dtype, copy=False)
    import scipy.sparse as sp

    n = len(labels)
    # CSR rows must list columns in input order; labels are usually already sorted.
    if n < 2 or bool(xp.all(labels[1:] >= labels[:-1])):
        order = xp.arange(n)
    else:
        order = xp.argsort(labels, kind="stable")
    counts = xp.bincount(labels, minlength=K)
    indicator = sp.csr_matrix(
        (xp.ones(n), order, xp.r_[0, xp.cumsum(counts)]), shape=(K, n)
    )
    flat = values.reshape(n, -1)
    return xp.asarray(indicator @ flat).reshape((K, *values.shape[1:]))


def _segments_of(labels: Array, K: int) -> tuple[Array, Array]:
    """
    ``(starts, counts)`` of the K label blocks of *sorted* labels, the batch layout
    (``repeat(arange(K), sizes)`` or an order-preserving subset of it) that every training
    caller passes. Found by binary search, with no host synchronization: this runs a few
    times per batch, and a sortedness check or ``bincount`` (which reads ``max`` on the
    host) would each cost a round trip. Unsorted labels give wrong segments.
    """
    labels = xp.asarray(labels).astype(xp.int64, copy=False)
    edges = xp.searchsorted(labels, xp.arange(K + 1, dtype=xp.int64))
    return edges[:-1], xp.diff(edges)


def segment_layout(starts: Array, counts: Array) -> tuple[Array, Array, Array]:
    """``(starts, counts, cumulative chunk counts)`` as ``segment_cross`` needs them; build once per batch and pass as ``layout``."""
    counts = xp.asarray(counts).astype(xp.int64, copy=False)
    return xp.asarray(starts).astype(xp.int64, copy=False), counts, xp.cumsum((counts + SEGMENT_CHUNK - 1) // SEGMENT_CHUNK)


def segment_cross(
    A: Array, B: Array, starts: Array, counts: Array, w: Array | None = None,
    a_off: Array | None = None, b_off: Array | None = None, symmetric: bool = False,
    layout: tuple[Array, Array, Array] | None = None,
) -> Array:  # fmt: skip
    """
    ``out[k] = sum_i (A_i - a_off[k]) w_i (B_i - b_off[k])^T`` over the contiguous rows
    ``starts[k] : starts[k] + counts[k]``, shape (K, P, Q); zero for empty segments.
    ``symmetric`` (``A is B`` with equal offsets) computes the upper triangle once;
    ``B=None`` means a column of ones (plain segmented sums, (K, P, 1)). A 2-D ``w`` (S, N)
    gives S weighted products in one launch, (S, K, P, Q), with ``B`` shared (N, Q) or one
    per weight vector (S, N, Q).

    CuPy: one thread per (256-row chunk, output entry) sums its chunk sequentially, then
    one thread per (segment, entry) sums the chunks in order. No atomics and no per-row
    outer products: the bits are the same between runs and do not depend on the other
    segments of the batch, and it is faster than an atomicAdd kernel
    (2-30x at 1e6-1e7 rows) and than cuBLAS-batched grouped products for p <= 21.
    NumPy (reference and tests only; NumPy callers use their own exact paths): a
    per-row outer product reduced with ``add.reduceat``.
    """
    K, P, Q = int(counts.shape[0]), A.shape[1], 1 if B is None else B.shape[-1]
    if w is not None and w.ndim == 2 and not IS_CUPY_BACKEND:
        per_b = B is not None and B.ndim == 3
        return xp.stack([segment_cross(A, B[i] if per_b else B, starts, counts, w[i], a_off, b_off, symmetric)
                         for i in range(w.shape[0])])  # fmt: skip
    if not IS_CUPY_BACKEND:
        B = xp.ones((A.shape[0], 1)) if B is None else B
        Ar = A - (0.0 if a_off is None else xp.repeat(a_off, counts, axis=0))
        Br = B - (0.0 if b_off is None else xp.repeat(b_off, counts, axis=0))
        prod = (Ar if w is None else Ar * w[:, None])[:, :, None] * Br[:, None, :]
        out = xp.zeros((K, P, Q))
        nz = xp.nonzero(counts)[0]
        if nz.size:
            out[nz] = xp.add.reduceat(prod, starts[nz], axis=0)
        return out
    er, ec = _segment_cross_entries(P, Q, symmetric)
    E, CH = int(er.shape[0]), SEGMENT_CHUNK
    S = 1 if w is None or w.ndim == 1 else int(w.shape[0])
    multi = w is not None and w.ndim == 2
    out = xp.empty((S, K, P, Q) if multi else (K, P, Q))
    if K == 0 or E == 0:
        return out
    starts, counts, cum = layout if layout is not None else segment_layout(starts, counts)
    n_chunks = A.shape[0] // CH + K  # an upper bound, so the launch needs no host sync
    partial = xp.empty(S * n_chunks * E)
    A = xp.ascontiguousarray(A, dtype=xp.float64)
    flags = int(w is not None) | int(a_off is not None) << 1 | int(b_off is not None) << 2 | int(B is None) << 3
    B = A if B is A or B is None else xp.ascontiguousarray(B, dtype=xp.float64)
    opt = [A if a is None else xp.ascontiguousarray(a, dtype=xp.float64) for a in (w, a_off, b_off)]
    N = A.shape[0]
    b_stride = N * Q if multi and B is not A and B.ndim == 3 else 0
    T = 256
    # One shared tile when B's adjusted rows equal A's: same array, same offsets, no weights.
    same = B is A and a_off is b_off and w is None
    width = P if same or flags & 8 else P + Q
    TR = min(32, (40 * 1024) // (8 * max(1, width)))
    if SEGMENT_TILED_MIN_E <= E and TR >= 4:  # entry groups of 8 x threads lift any cap on E
        threads = min(T, ((E + 31) // 32) * 32)
        _kern_seg_cross_tiled(
            (S * n_chunks, -(-E // (8 * threads))), (threads,),
            (A, B, *opt, cum, starts, counts, er, ec, partial, np.int64(n_chunks), np.int32(K),
             np.int32(E), np.int32(P), np.int32(Q), np.int32(CH), np.int32(flags), np.int32(S),
             np.int64(N if multi else 0), np.int64(b_stride), np.int32(TR), np.int32(int(same))),
            shared_mem=8 * TR * width,
        )  # fmt: skip
    else:
        _kern_seg_cross_partial(
            ((S * n_chunks * E + T - 1) // T,), (T,),
            (A, B, *opt, cum, starts, counts, er, ec, partial, np.int64(n_chunks), np.int32(K),
             np.int32(E), np.int32(P), np.int32(Q), np.int32(CH), np.int32(flags), np.int32(S),
             np.int64(N if multi else 0), np.int64(b_stride)),
        )  # fmt: skip
    _kern_seg_cross_reduce(
        ((S * K * E + T - 1) // T,), (T,),
        (partial, cum, counts, er, ec, out, np.int32(K), np.int32(E), np.int32(P), np.int32(Q),
         np.int32(CH), np.int32(int(symmetric)), np.int32(S), np.int64(n_chunks)),
    )  # fmt: skip
    return out

_SEGMENT_CROSS_ENTRIES: dict = {}


def _segment_cross_entries(P: int, Q: int, symmetric: bool) -> tuple[Array, Array]:
    """Row and column index of every computed output entry (the upper triangle if symmetric), cached per shape."""
    key = (P, Q, symmetric)
    if key not in _SEGMENT_CROSS_ENTRIES:
        rc = np.array([(r, c) for r in range(P) for c in range(Q) if not symmetric or c >= r], dtype=np.int32)
        _SEGMENT_CROSS_ENTRIES[key] = (xp.asarray(rc[:, 0].copy()), xp.asarray(rc[:, 1].copy()))
    return _SEGMENT_CROSS_ENTRIES[key]


def count_groups(counts: Array, starts: Array) -> list[tuple[Array, Array]]:
    """
    Blocks grouped by size: one ``(block ids (g,), row ids (g, c))`` per distinct size
    ``c``, for blocks stored contiguously from ``starts``. Gathering a group gives a dense
    (g, c, ...) array, so per-block products are batched matrix products with no padding
    and no per-row outer products, and a block's arithmetic does not depend on the
    other blocks in its batch.
    """
    host = np.asarray(_to_host(counts))
    groups = []
    for c in np.unique(host):
        ids = xp.asarray(np.nonzero(host == c)[0])
        groups.append((ids, starts[ids][:, None] + xp.arange(int(c))[None, :]))
    return groups


def grouped_cross(groups: list, A: list[Array], B: list[Array], w: Array | None = None) -> Array:
    """(k, a, b) per-block products ``A_k^T diag(w_k) B_k`` of ``count_groups``-gathered blocks."""
    k = sum(int(ids.shape[0]) for ids, _ in groups)
    out = xp.empty((k, A[0].shape[-1], B[0].shape[-1]))
    for (ids, rows), Ag, Bg in zip(groups, A, B, strict=True):
        Aw = Ag if w is None else Ag * w[rows][..., None]
        out[ids] = xp.swapaxes(Aw, 1, 2) @ Bg
    return out


def sort_by_block_then_value(values: Array, blocks: Array) -> Array:
    """
    The permutation of ``xp.lexsort((values, blocks))``: a stable sort by value
    followed by a stable sort by block gives the same order, about 3x faster on both
    backends (CuPy's argsort is always stable; its lexsort is a slow merge sort).
    """
    kind = {} if IS_CUPY_BACKEND else {"kind": "stable"}
    by_value = xp.argsort(values, **kind)
    return by_value[xp.argsort(blocks[by_value], **kind)]


def get_n_smallest_per_block(values: Array, blocks: Array, n: int) -> Array:
    """Returns a boolean mask for the n smallest values in each block."""
    sort_idx = sort_by_block_then_value(values, blocks)
    block_sizes = xp.bincount(blocks)
    block_starts = xp.concatenate((xp.array([0]), xp.cumsum(block_sizes[:-1])))
    if IS_CUPY_BACKEND:
        # Rank within the block = sorted position - block start; marking rows in one
        # pass avoids the six N-length index arrays of the general form below.
        mask = xp.zeros(len(blocks), dtype=xp.bool_)
        _mark_n_smallest()(sort_idx, xp.ascontiguousarray(blocks, dtype=xp.int64),
                           block_starts.astype(xp.int64), xp.int64(n), mask)
        return mask

    offsets = xp.zeros(len(blocks), dtype=xp.int64)
    diffs = xp.diff(block_starts, prepend=0)
    offsets[block_starts] = diffs
    local_indices = xp.cumsum(offsets)

    inv_sort_idx = xp.empty_like(sort_idx)
    inv_sort_idx[sort_idx] = xp.arange(len(sort_idx), dtype=sort_idx.dtype)
    sorted_local_indices = inv_sort_idx - local_indices
    return sorted_local_indices < n

_MARK_KERNEL = None


def _mark_n_smallest():
    global _MARK_KERNEL
    if _MARK_KERNEL is None:
        _MARK_KERNEL = cp.ElementwiseKernel(
            "int64 order, raw int64 blocks, raw int64 starts, int64 n",
            "raw bool mask",
            "if (i - starts[blocks[order]] < n) mask[order] = true;",
            "afinni_mark_n_smallest",
        )
    return _MARK_KERNEL
