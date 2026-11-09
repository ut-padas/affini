"""
k-means for the initial clustering (``initial_clustering_method="kmeans"``) on both
backends, replacing cuML's ``KMeans`` (GPU) and scikit-learn's ``KMeans`` (CPU). Runs
once per training, before refinement, so its memory never overlaps the refinement peak.
``kmeans_labels`` dispatches on the active backend; both paths share the algorithm.

Scheme (as in cuVS / cuML defaults: k-means|| seeding, Lloyd, ``max_iter=300``,
``tol=1e-4``):

- Seeding (k-means||): start from one random point, then a few rounds that each keep
  every point independently with probability ``min(1, l * d^2 / phi)`` (``l`` =
  oversampling x k), so the candidates come from a few passes over the data instead of
  k sequential ones; the candidates are weighted by how many points they are nearest
  to, and greedy k-means++ (best of ``2 + ln k`` D^2-weighted draws per centre, as in
  scikit-learn) picks the k centres among them.
- Assignment: the fused distance + nearest-neighbour kernel in ``src.knn`` with k=1
  (float32 distances, float64 data read in place), so the (points x centres) distance
  matrix is never stored. Accuracy policy: assignments are ``kneighbors_unverified``,
  i.e. exact up to float32 resolution of the centred coordinates; a point almost
  equidistant from two centres may go to either, which can change the Lloyd trajectory
  and so the tree; validated by objective checks against float64 inertia on translated
  data, not as a general bound. The exact float64 verification used for prediction is
  not paid here.
- Update: centroid sums in float64 by a deterministic kernel (one block per cluster,
  rows added in a fixed order), so repeated runs give identical clusters; atomics would
  not. Empty clusters keep their previous centre.
- Convergence: total squared centre shift <= ``tol`` x mean feature variance, checked
  every ``CHECK_EVERY`` iterations (each check waits for the device).

Measured on an RTX 2000 Ada laptop GPU against cuML 26.06 at cuML's iteration counts:
4.5-7.2x faster for 200k-2M points, d = 5-100, k = 92-500, with inertia within 0.1% in
four of five cases and 0.7% higher in one. Results differ from cuML's (different
seeding), so trees built from them differ but are statistically equivalent.

CPU path (Numba + BLAS; the pipeline calls it with ``max_iter=2`` as before):

- Seeding: the same k-means||, with each round lowering every point's distance to its
  nearest candidate in place on the float64 data (a compiled kernel) and recording
  that candidate, so the candidate weights need no extra pass over the data.
- Lloyd: one fused pass per iteration over fixed-size chunks (scikit-learn's chunked
  EM step): nearest centre (direct distances for d <= ``CPU_DIRECT_MAX_D``, otherwise
  a BLAS block of ``x.c`` and the ``|c|^2 - 2 x.c`` epilogue, as the argmin needs no
  ``|x|^2``), label, and chunk-local centre sums. Chunk sums are added in chunk order,
  so results do not depend on the thread count. Stops when no label changes or the
  centre shift is below ``tol``; a final assignment makes labels match the centres.
- Threads: ``threads`` if given (rank 0 borrows its node's idle MPI ranks' cores),
  else ``OMP_NUM_THREADS``, else all cores.
- Measured against scikit-learn 1.7 at 2-3 iterations: 1.4-2.3x faster on one thread
  and 2.7-8.6x on eight, for 200k-2M points, d = 5-100, k = 92-500, with inertia within
  -0.95% to +0.14% of scikit-learn's.
- Caveat: run to full convergence it can need more iterations than scikit-learn (300
  vs 221 in one 5-D test, from the different seeding), although each iteration is as
  fast or faster; the pipeline only runs a few iterations.

Independent reimplementation of ideas from (no code copied; cuVS is Apache-2.0,
scikit-learn is BSD-3-Clause):

- cuVS k-means fit loop, k-means|| seeding and device-side convergence check:
  https://github.com/NVIDIA/cuvs/blob/920fc837e370b76345ab91dcbd699da40525a51b/cpp/src/cluster/detail/kmeans.cuh
- cuVS assignment and centroid helpers (min cluster and distance, batching):
  https://github.com/NVIDIA/cuvs/blob/920fc837e370b76345ab91dcbd699da40525a51b/cpp/src/cluster/detail/kmeans_common.cuh
- cuVS fused distance + argmin (the idea behind using the fused kNN kernel with k=1):
  https://github.com/NVIDIA/cuvs/blob/920fc837e370b76345ab91dcbd699da40525a51b/cpp/src/distance/detail/fused_distance_nn/fused_l2_nn.cuh
- scikit-learn chunked Lloyd EM step (fused assignment + chunk-local sums, argmin
  without |x|^2):
  https://github.com/scikit-learn/scikit-learn/blob/6e37063f7ab7ef7a7e257b9d44bc3ec6ee1b1bcd/sklearn/cluster/_k_means_lloyd.pyx
- scikit-learn greedy k-means++ (best of 2 + ln k candidates), used on the candidates:
  https://github.com/scikit-learn/scikit-learn/blob/6e37063f7ab7ef7a7e257b9d44bc3ec6ee1b1bcd/sklearn/cluster/_kmeans.py
"""

import math
import os

import numpy as np

from src import Array

CHECK_EVERY = 8  # Lloyd iterations between convergence checks
SEED_ROUNDS = 5  # k-means|| sampling rounds
OVERSAMPLING = 2.0  # expected candidates per round, in multiples of k (cuML default)

# One block per cluster; threads stride over dimensions and add the cluster's rows
# (``order[starts[c]:starts[c+1]]``) in a fixed order, so sums are reproducible.
_CENTROID_SUM_SRC = r"""
extern "C" __global__ void centroid_sums(const double* __restrict__ X,
                                         const long long* __restrict__ order,
                                         const long long* __restrict__ starts,
                                         int d, double* __restrict__ sums) {
    const int c = blockIdx.x;
    const long long b = starts[c], e = starts[c + 1];
    for (int t = threadIdx.x; t < d; t += blockDim.x) {
        double s = 0.0;
        for (long long r = b; r < e; ++r) s += X[order[r] * d + t];
        sums[(long long)c * d + t] = s;
    }
}
"""
_SUM_KERNEL = None


def _nearest(X: Array, C: Array) -> tuple[Array, Array]:
    """``(labels, squared distances)`` of every row of ``X`` to its nearest row of ``C``."""
    import cupy as cp

    from src.knn import GpuExactKNN

    # k-means tolerates ties within float32 resolution, so the float64 verification of
    # the exact search is skipped; GpuExactKNN centres on C and subtracts in float64.
    dist, idx = GpuExactKNN(C).kneighbors_unverified(X, 1)
    return idx[:, 0], dist[:, 0].astype(cp.float64) ** 2


def _centroids(X: Array, labels: Array, previous: Array) -> Array:
    """New centres (float64, deterministic); empty clusters keep their previous centre."""
    import cupy as cp

    global _SUM_KERNEL
    if _SUM_KERNEL is None:
        _SUM_KERNEL = cp.RawKernel(_CENTROID_SUM_SRC, "centroid_sums")
    k, d = previous.shape
    order = cp.argsort(labels)  # stable (CuPy), so each cluster's rows keep input order
    counts = cp.bincount(labels, minlength=k)
    starts = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(counts)])
    sums = cp.empty((k, d), dtype=cp.float64)
    _SUM_KERNEL((k,), (min(128, max(32, d)),), (X, order, starts, cp.int32(d), sums))
    centres = previous.copy()
    nonempty = counts > 0
    centres[nonempty] = sums[nonempty] / counts[nonempty, None]
    return centres


def _greedy_kmeans_pp(P: Array, w: Array, k: int, rng: np.random.Generator) -> Array:
    """Weighted greedy k-means++ over a small candidate set ``P`` with weights ``w``."""
    import cupy as cp

    n = int(P.shape[0])
    trials = 2 + int(math.log(k))
    C = cp.empty((k, P.shape[1]))
    C[0] = P[int(rng.choice(n, p=cp.asnumpy(w / w.sum())))]
    d2 = ((P - C[0]) ** 2).sum(axis=1)
    draws = cp.asarray(rng.random((k, trials)))  # drawn up front: no host sync per centre
    for c in range(1, k):
        cum = cp.cumsum(w * d2)
        picks = cp.minimum(cp.searchsorted(cum, draws[c] * cum[-1]), n - 1)
        cand = P[picks]  # (trials, d)
        cand_d2 = cp.minimum(d2[:, None], ((P[:, None, :] - cand[None]) ** 2).sum(axis=-1))
        best = cp.argmin((w[:, None] * cand_d2).sum(axis=0))
        C[c] = cand[best]
        d2 = cp.take(cand_d2, best, axis=1)
    return C


def _kmeans_parallel_seed(X: Array, k: int, rng: np.random.Generator) -> Array:
    """k-means|| seeding: oversampled candidates from a few passes, reduced to k centres."""
    import cupy as cp

    n = int(X.shape[0])
    C = X[int(rng.integers(n))][None, :]
    _, d2 = _nearest(X, C)
    for _ in range(SEED_ROUNDS):
        phi = float(d2.sum())
        if phi == 0.0:
            break
        keep = cp.asarray(rng.random(n)) < cp.minimum(1.0, OVERSAMPLING * k * d2 / phi)
        n_new = int(keep.sum())
        if n_new == 0:
            continue
        new = X[keep]
        C = cp.concatenate([C, new])
        d2 = cp.minimum(d2, _nearest(X, new)[1])
    if C.shape[0] <= k:
        # Too few distinct candidates (tiny or degenerate data): pad with random rows.
        extra = X[cp.asarray(rng.choice(n, size=k - C.shape[0], replace=n < k))]
        return cp.concatenate([C, extra])
    labels, _ = _nearest(X, C)
    weights = cp.bincount(labels, minlength=C.shape[0]).astype(cp.float64) + 1e-12
    return _greedy_kmeans_pp(C, weights, k, rng)


def kmeans_labels(
    X: Array, k: int, seed: int, max_iter: int = 300, tol: float = 1e-4, threads: int | None = None
) -> Array:
    """Cluster labels (int64, on the active backend) of the rows of ``X`` into ``k`` clusters."""
    from src import IS_CUPY_BACKEND

    if not IS_CUPY_BACKEND:
        return _kmeans_labels_cpu(np.asarray(X), k, seed, max_iter, tol, threads)
    import cupy as cp

    X = cp.ascontiguousarray(X, dtype=cp.float64)
    rng = np.random.default_rng(seed)
    C = _kmeans_parallel_seed(X, k, rng)
    threshold = tol * float(cp.mean(cp.var(X, axis=0)))
    labels, _ = _nearest(X, C)
    for it in range(1, max_iter + 1):
        new = _centroids(X, labels, C)
        shift = cp.sum((new - C) ** 2)
        C = new
        labels, _ = _nearest(X, C)
        if (it % CHECK_EVERY == 0 or it == max_iter) and float(shift) <= threshold:
            break
    return labels


# --------------------------------------------------------------------------- CPU
CPU_DIRECT_MAX_D = 16  # above this, distances come from BLAS blocks
_CPU_BLOCK_BYTES = 64 * 2**20  # bound on BLAS blocks and chunk-local partial sums
_CPU_KERNELS = None


def _cpu_kernels():
    """Compile the Numba kernels on first use (cached on disk afterwards)."""
    global _CPU_KERNELS
    if _CPU_KERNELS is not None:
        return _CPU_KERNELS
    import numba as nb

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def nearest_update_direct(X, Ct, base, d2, lab):
        # Low d: distances to all m candidates, built dimension by dimension over the
        # transposed (d, m) candidates (the inner loop over candidates vectorizes).
        n, d = X.shape
        m = Ct.shape[1]
        CH = 4096
        for c in nb.prange((n + CH - 1) // CH):
            dist = np.empty(m)
            for i in range(c * CH, min(n, (c + 1) * CH)):
                for j in range(m):
                    dist[j] = 0.0
                for t in range(d):
                    xt = X[i, t]
                    for j in range(m):
                        diff = xt - Ct[t, j]
                        dist[j] += diff * diff
                best = d2[i]
                bj = -1
                for j in range(m):
                    if dist[j] < best:
                        best = dist[j]
                        bj = j
                if bj >= 0:
                    d2[i] = best
                    lab[i] = base + bj

    @nb.njit(parallel=True, cache=True)
    def nearest_update_blas(ip, x_sq, c_sq, row0, base, d2, lab):
        # High d: the same update from a BLAS block ip = X[row0:row0+m] @ new.T.
        m, k = ip.shape
        for r in nb.prange(m):
            i = row0 + r
            best = d2[i]
            bj = -1
            for j in range(k):
                v = max(x_sq[i] + c_sq[j] - 2.0 * ip[r, j], 0.0)
                if v < best:
                    best = v
                    bj = j
            if bj >= 0:
                d2[i] = best
                lab[i] = base + bj

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def lloyd_direct(X, Ct, chunk, labels, sums, counts, changed):
        # Low d, one fused pass per chunk: distances to all centres, argmin, label,
        # chunk-local sums; changed[c] counts label changes (strict convergence).
        n, d = X.shape
        k = Ct.shape[1]
        for c in nb.prange(sums.shape[0]):
            dist = np.empty(k)
            for i in range(c * chunk, min(n, (c + 1) * chunk)):
                for j in range(k):
                    dist[j] = 0.0
                for t in range(d):
                    xt = X[i, t]
                    for j in range(k):
                        diff = xt - Ct[t, j]
                        dist[j] += diff * diff
                bj = 0
                best = dist[0]
                for j in range(1, k):
                    if dist[j] < best:
                        best = dist[j]
                        bj = j
                if labels[i] != bj:
                    changed[c] += 1
                labels[i] = bj
                counts[c, bj] += 1.0
                for t in range(d):
                    sums[c, bj, t] += X[i, t]

    @nb.njit(parallel=True, cache=True)
    def lloyd_blas(X, ip, c_sq, row0, chunk, labels, sums, counts, changed):
        # High d, after ip = X[row0:row0+m] @ C.T: per chunk, argmin of |c|^2 - 2 x.c,
        # label and chunk-local sums in one pass (blocks are aligned to chunks).
        m, k = ip.shape
        d = X.shape[1]
        first = row0 // chunk
        for cc in nb.prange((row0 + m - 1) // chunk - first + 1):
            c = first + cc
            for i in range(max(row0, c * chunk), min(row0 + m, (c + 1) * chunk)):
                r = i - row0
                bj = 0
                best = c_sq[0] - 2.0 * ip[r, 0]
                for j in range(1, k):
                    v = c_sq[j] - 2.0 * ip[r, j]
                    if v < best:
                        best = v
                        bj = j
                if labels[i] != bj:
                    changed[c] += 1
                labels[i] = bj
                counts[c, bj] += 1.0
                for t in range(d):
                    sums[c, bj, t] += X[i, t]

    _CPU_KERNELS = (nearest_update_direct, nearest_update_blas, lloyd_direct, lloyd_blas)
    return _CPU_KERNELS


def _cpu_nearest_update(X, x_sq, new, base, d2, lab):
    """Lower ``d2`` / ``lab`` where one of the ``new`` candidates (indices ``base``...) is nearer."""
    direct, blas, _, _ = _cpu_kernels()
    if X.shape[1] <= CPU_DIRECT_MAX_D:
        direct(X, np.ascontiguousarray(new.T), base, d2, lab)
        return
    c_sq = np.einsum("ij,ij->i", new, new)
    rows = max(1, _CPU_BLOCK_BYTES // (8 * new.shape[0]))
    for r0 in range(0, X.shape[0], rows):
        blas(X[r0 : r0 + rows] @ new.T, x_sq, c_sq, r0, base, d2, lab)


def _cpu_lloyd_step(X, C, labels):
    """One fused Lloyd iteration: labels for ``C`` (in place) and the new centres; also the number of label changes."""
    _, _, direct, blas = _cpu_kernels()
    n, d = X.shape
    k = C.shape[0]
    chunk = max(4096, -(-n * k * d * 8 // _CPU_BLOCK_BYTES))  # bounds the partial-sum memory
    nc = -(-n // chunk)
    sums = np.zeros((nc, k, d))
    counts = np.zeros((nc, k))
    changed = np.zeros(nc, dtype=np.int64)
    if d <= CPU_DIRECT_MAX_D:
        direct(X, np.ascontiguousarray(C.T), chunk, labels, sums, counts, changed)
    else:
        c_sq = np.einsum("ij,ij->i", C, C)
        rows = max(chunk, _CPU_BLOCK_BYTES // (8 * k))
        rows -= rows % chunk  # blocks aligned to chunks
        for r0 in range(0, n, rows):
            blas(X, X[r0 : r0 + rows] @ C.T, c_sq, r0, chunk, labels, sums, counts, changed)
    total_s, total_c = sums[0].copy(), counts[0].copy()
    for i in range(1, nc):  # chunk order: identical for any thread count
        total_s += sums[i]
        total_c += counts[i]
    new = C.copy()
    nonempty = total_c > 0
    new[nonempty] = total_s[nonempty] / total_c[nonempty, None]
    return new, int(changed.sum())


def _cpu_greedy_kmeans_pp(P, w, k, rng):
    """Weighted greedy k-means++ over the candidates; distances from one BLAS product per centre."""
    n = P.shape[0]
    trials = 2 + int(math.log(k))
    C = np.empty((k, P.shape[1]))
    C[0] = P[int(rng.choice(n, p=w / w.sum()))]
    d2 = ((P - C[0]) ** 2).sum(axis=1)
    p_sq = np.einsum("ij,ij->i", P, P)
    for c in range(1, k):
        cum = np.cumsum(w * d2)
        picks = np.minimum(np.searchsorted(cum, rng.random(trials) * cum[-1]), n - 1)
        dist = np.maximum(p_sq[:, None] - 2.0 * (P @ P[picks].T) + p_sq[picks][None, :], 0.0)
        cand_d2 = np.minimum(d2[:, None], dist)
        best = int(np.argmin((w[:, None] * cand_d2).sum(axis=0)))
        C[c] = P[picks[best]]
        d2 = cand_d2[:, best]
    return C


def _kmeans_labels_cpu(X, k, seed, max_iter, tol, threads):
    import numba as nb

    # Centre in float64 first: k-means is translation invariant, and the BLAS path's
    # expanded distances (|x|^2 + |c|^2 - 2 x.c) cancel catastrophically for data far
    # from the origin (inertia was 935x too large with a 1e8 offset at d=30).
    X = np.asarray(X, dtype=np.float64)
    X = np.ascontiguousarray(X - X.mean(axis=0))
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if threads is None:
        env = os.environ.get("OMP_NUM_THREADS", "")
        threads = int(env) if env.isdigit() else nb.config.NUMBA_NUM_THREADS
    previous = nb.get_num_threads()
    nb.set_num_threads(max(1, min(int(threads), nb.config.NUMBA_NUM_THREADS)))
    try:
        # k-means|| seeding, tracking each point's nearest candidate for the weights.
        x_sq = np.einsum("ij,ij->i", X, X)
        C = X[int(rng.integers(n))][None, :]
        d2 = np.full(n, np.inf)
        lab = np.zeros(n, dtype=np.int64)
        _cpu_nearest_update(X, x_sq, C, 0, d2, lab)
        for _ in range(SEED_ROUNDS):
            phi = float(d2.sum())
            if phi == 0.0:
                break
            keep = rng.random(n) < np.minimum(1.0, OVERSAMPLING * k * d2 / phi)
            if keep.any():
                new = X[keep]
                _cpu_nearest_update(X, x_sq, new, C.shape[0], d2, lab)
                C = np.concatenate([C, new])
        if C.shape[0] <= k:
            # Too few distinct candidates (tiny or degenerate data): pad with random rows.
            C = np.concatenate([C, X[rng.choice(n, size=k - C.shape[0], replace=n < k)]])
        else:
            weights = np.bincount(lab, minlength=C.shape[0]).astype(np.float64) + 1e-12
            C = _cpu_greedy_kmeans_pp(C, weights, k, rng)

        threshold = tol * float(np.mean(np.var(X, axis=0)))
        labels = np.full(n, -1, dtype=np.int64)
        for _ in range(max_iter):
            new, n_changed = _cpu_lloyd_step(X, C, labels)
            if n_changed == 0:  # strict convergence: C is a fixed point
                break
            shift = float(((new - C) ** 2).sum())
            C = new
            if shift <= threshold:
                break
        _cpu_lloyd_step(X, C, labels)  # final assignment: labels match the returned centres
        return labels
    finally:
        nb.set_num_threads(previous)
