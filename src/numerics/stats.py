"""Per-cluster statistics and projections (covariances, cross-covariances, distances, residual norms) and array shapes."""


from src import IS_CUPY_BACKEND, Array, kernels, xp
from src.numerics.segments import _segments_of, segment_cross, segment_sum

if IS_CUPY_BACKEND:
    pass

if IS_CUPY_BACKEND:  # sources in src/kernels/*.cu
    _stats = kernels.module("cluster_stats")
    _kern_centroid_distance = _stats.get_function("centroid_distance")
    _kern_taylor_residual = _stats.get_function("taylor_residual_norm")
    _kern_fused_split_proj = _stats.get_function("fused_split_projection")
    _kern_fused_mah_proj = _stats.get_function("fused_mahalanobis_projection")


def run_fused_stats(
    X, mus, labels, K, D, FD, active_comps, G=None, g_mus=None, calc_E=False
):
    """
    Accumulates Covariance (C) and Gradient Interaction (E) matrices. On CuPy through
    ``segment_cross``, so the sums are reproducible and independent of the batch; there
    the labels must be sorted (``_segments_of``), as every batch layout is.
    """
    if IS_CUPY_BACKEND:  # labels sorted, as every batch stores its clusters contiguously
        starts, counts = _segments_of(labels, K)
        C_out = segment_cross(X, X, starts, counts, a_off=mus, b_off=mus, symmetric=True)
        if not calc_E:
            return C_out, xp.array([], dtype=xp.float64)
        ac = xp.asarray(active_comps).astype(xp.int64, copy=False)
        G_sel = G[xp.arange(X.shape[0]), ac[xp.asarray(labels, dtype=xp.int64)], :]
        E_out = segment_cross(X, G_sel, starts, counts, a_off=mus, b_off=g_mus[xp.arange(K), ac, :])
        return C_out, E_out
    else:
        dx = X - mus[labels]
        C_out = segment_sum(dx[:, :, None] * dx[:, None, :], labels, K)

        if calc_E:
            assert G is not None and g_mus is not None
            active_j = active_comps[labels]
            n_idx = xp.arange(X.shape[0])
            g_x = G[n_idx, active_j, :]
            g_mu = g_mus[labels, active_j, :]
            dg = g_x - g_mu
            E_out = segment_sum(dx[:, :, None] * dg[:, None, :], labels, K)
        else:
            E_out = xp.array([], dtype=xp.float64)
        return C_out, E_out


def run_fused_cross_cov(X, F, mus, f_mus, labels, K, D, FD):
    """Accumulates Cross-Covariance matrix (dx * df^T); on CuPy through ``segment_cross`` (reproducible; sorted labels)."""
    if IS_CUPY_BACKEND:  # labels sorted, as every batch stores its clusters contiguously
        starts, counts = _segments_of(labels, K)
        return segment_cross(X, F, starts, counts, a_off=mus, b_off=f_mus)

    dx = X - mus[labels]
    df = F - f_mus[labels]
    return segment_sum(dx[:, :, None] * df[:, None, :], labels, K)


def run_fused_split_projection(X, mus, labels, M_split, v):
    """
    Computes ``dot(M_split[label] @ (x - mu[label]), v[label])`` per point
    without materializing an ``(N, d, d)`` indexed matrix tensor on the GPU.
    """
    if IS_CUPY_BACKEND:
        N, D = X.shape
        scores_out = xp.empty((N,), dtype=xp.float64)
        labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)
        block = 256
        grid = (N + block - 1) // block
        _kern_fused_split_proj(
            (grid,),
            (block,),
            (
                xp.ascontiguousarray(X),
                xp.ascontiguousarray(mus),
                labels_64,
                xp.ascontiguousarray(M_split),
                xp.ascontiguousarray(v),
                scores_out,
                N,
                D,
            ),
        )
        return scores_out

    dx = X - mus[labels]
    z = xp.einsum("nij,nj->ni", M_split[labels], dx)
    return xp.sum(z * v[labels], axis=1)


def centroid_distances(X, mus, labels):
    """``|x - mu[label]|`` per point; on the GPU without the (N, d) offset array."""
    if IS_CUPY_BACKEND:
        N, D = X.shape
        out = xp.empty((N,), dtype=xp.float64)
        _kern_centroid_distance(
            ((N + 255) // 256,), (256,),
            (xp.ascontiguousarray(X), xp.ascontiguousarray(mus),
             xp.ascontiguousarray(labels, dtype=xp.int64), out, xp.int64(N), xp.int32(D)),
        )  # fmt: skip
        return out
    return xp.linalg.norm(X - mus[labels], axis=1)


def taylor_residual_norms(X, F, mus, f_c, labels, A):
    """
    ``|F - f_c[l] - A[l] (x - mu[l])|`` per point for ``A`` (K, fd, d). On the GPU
    this avoids the (N, d), (N, fd, d) and (N, fd) temporaries of the NumPy form,
    which at d=100, fd=10 is about 8 KB per point.
    """
    if IS_CUPY_BACKEND:
        N, D = X.shape
        FD = F.shape[1]
        out = xp.empty((N,), dtype=xp.float64)
        _kern_taylor_residual(
            ((N + 255) // 256,), (256,),
            (xp.ascontiguousarray(X), xp.ascontiguousarray(F), xp.ascontiguousarray(mus),
             xp.ascontiguousarray(f_c), xp.ascontiguousarray(labels, dtype=xp.int64),
             xp.ascontiguousarray(A), out, xp.int64(N), xp.int32(D), xp.int32(FD)),
        )  # fmt: skip
        return out
    dx = X - mus[labels]
    lin = xp.einsum("nmd,nd->nm", A[labels], dx)
    return xp.linalg.norm(F - (f_c[labels] + lin), axis=1)


def run_fused_mahalanobis_projection(dx, labels, U):
    """
    Computes ``dx @ U[label]`` per point without materializing
    ``U[labels]`` as an ``(N, d, d)`` tensor.

    This is used in accepted-cluster trust ellipsoid construction, where
    materializing the indexed eigenvector tensor can be O(N*d^2) memory.
    """
    if IS_CUPY_BACKEND:
        N, D = dx.shape
        proj_out = xp.empty((N, D), dtype=xp.float64)
        labels_64 = xp.ascontiguousarray(labels, dtype=xp.int64)
        block = 256
        grid = (N + block - 1) // block
        _kern_fused_mah_proj(
            (grid,),
            (block,),
            (
                xp.ascontiguousarray(dx),
                labels_64,
                xp.ascontiguousarray(U),
                proj_out,
                N,
                D,
            ),
        )
        return proj_out

    return xp.einsum("ni,nij->nj", dx, U[labels])


def _symmetric_point_covariance(
    X: Array, centroids: Array, labels: Array, K: int
) -> Array:
    """(K, d, d) covariance of each cluster's points about its centroid."""
    Cov, _ = run_fused_stats(
        X, centroids, labels, K, X.shape[1], 0, None, calc_E=False
    )
    return 0.5 * (Cov + Cov.transpose(0, 2, 1))


def convert_to_2d_array(F: Array) -> Array:
    """
    Ensures that F is a 2D array of shape (N, fd). If F is 1D, it converts it to (N, 1).
    """
    F = xp.asarray(F)
    if F.ndim == 1:
        return F[:, None]  # Convert (N,) to (N, 1)
    return F


def convert_to_3d_array(G: Array) -> Array:
    """
    Ensures that G is a 3D array of shape (N, 1, d). If G is 2D, it converts it to (N, 1, d).
    """
    G = xp.asarray(G)
    if G.ndim == 2:
        return G[:, None, :]  # Convert (N, d) to (N, 1, d)
    return G
