"""
Randomized-QR based fast 2-means using optimal 1D projections or median split.

This implements the math sketched in docs image (randomized range finder +
subspace iterations with unpivoted QR), then performs a 1D partition on the
resulting principal direction. API mirrors two_means.two_means_pca_threshold.

References:
- Halko, Martinsson, Tropp. Finding structure with randomness: Probabilistic
  algorithms for constructing approximate matrix decompositions. SIAM Review, 2011.
- Project notes: randomized QR sketch (image) with batched QR steps.
"""

import numpy as np

from src.two_means import find_optimal_split_1d


def _ensure_zero_mean_columns(Z: np.ndarray) -> np.ndarray:
    """Ensure each feature/column of Z has zero mean; return a copy if needed."""
    col_means = Z.mean(axis=0)
    if np.allclose(col_means, 0.0):
        return Z
    return Z - col_means


def _median_threshold(labels_proj: np.ndarray) -> tuple[np.ndarray, float]:
    """Return labels from a median threshold on a 1D array.

    Returns labels in {0,1} and the threshold used.
    """
    tau = float(np.median(labels_proj))
    labels = (labels_proj > tau).astype(np.int64)
    # Put points on the threshold into the left cluster to avoid empty right
    on_tau = np.isclose(labels_proj, tau)
    labels[on_tau] = 0
    return labels, tau


def _labels_sse_split(s: np.ndarray) -> np.ndarray:
    """Labels via optimal SSE split of 1D projections s.

    Returns labels in the original order.
    """
    n = s.shape[0]
    order = np.argsort(s)
    s_sorted = s[order]
    best_k = int(find_optimal_split_1d(s_sorted))
    best_k = min(max(best_k, 1), n - 1)

    labels_sorted = np.zeros(n, dtype=np.int64)
    labels_sorted[best_k:] = 1
    labels = np.empty(n, dtype=np.int64)
    labels[order] = labels_sorted
    return labels


def _safe_centroids(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute 2 centroids in Z-space given labels in {0,1}, avoiding empty sets."""
    n = Z.shape[0]
    k0 = int(np.sum(labels == 0))
    k1 = n - k0
    if k0 == 0 or k1 == 0:
        # Fallback to balanced split by order if degenerate
        s = np.arange(n)
        order = np.argsort(s)
        labels = np.empty(n, dtype=np.int64)
        labels[order[: n // 2]] = 0
        labels[order[n // 2 :]] = 1
        k0 = n // 2
        k1 = n - k0
    c0 = Z[labels == 0].mean(axis=0)
    c1 = Z[labels == 1].mean(axis=0)
    return np.vstack([c0, c1])


def two_means_random_qr_threshold(
    X_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    oversampling: int = 7,
    power_iter: int = 1,
    random_state: int | None = None,
    threshold: str = "median",  # "median" (per sketch) or "sse" (optimal 1D)
    use_small_svd: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomized-QR 2-means using a fast principal direction estimation.

    Args:
        X_norm: Normalized data of shape (n, d), ideally zero-mean per feature.
        mean: Mean used for normalization, shape (d,)
        std: Stddev used for normalization, shape (d,)
        oversampling: Oversampling parameter p; effective subspace size ℓ = 1 + p.
        power_iter: Number of subspace power iterations (0, 1, or 2 are typical).
        random_state: Seed for reproducibility.
        threshold: "median" to follow sketch; "sse" to use optimal 1D SSE scan.
        use_small_svd: Use SVD on small matrix B = Q^T Z to get direction.

    Returns:
        labels: (n,) int64 in {0,1}
        centroids: (2, d) in original, unnormalized space
    """
    Z = np.asarray(X_norm)
    n, d = Z.shape

    if n < 2:
        labels = np.zeros(n, dtype=np.int64)
        centroids = np.array([Z.mean(axis=0), Z.mean(axis=0)]) * std + mean
        return labels, centroids

    # Ensure zero-mean per feature if not already
    Z = _ensure_zero_mean_columns(Z)

    rng = np.random.default_rng(random_state)
    ell = 1 + max(int(oversampling), 0)
    ell = min(ell, d)  # cannot exceed feature dimension

    # Random test matrix (feature-space since Z is n x d)
    G = rng.standard_normal(size=(d, ell))

    # Range finder with subspace iterations; follow the batched QR sketch.
    Y = Z @ G  # (n, ell)
    for _ in range(max(power_iter, 0)):
        Q1, _ = np.linalg.qr(Y, mode="reduced")  # (n, ell)
        Y = Z.T @ Q1  # (d, ell)
        Q2, _ = np.linalg.qr(Y, mode="reduced")  # (d, ell)
        Y = Z @ Q2  # (n, ell)

    Q, _ = np.linalg.qr(Y, mode="reduced")  # (n, ell)

    # Small matrix whose right singular vectors approximate those of Z
    # (standard in randomized SVD)
    B = Q.T @ Z  # (ell, d)

    try:
        if use_small_svd:
            # Compute top right singular vector of B
            # B = U_hat S V^T  => principal direction u = first row of V^T (transposed)
            _, _, Vt = np.linalg.svd(B, full_matrices=False)
            u = Vt[0].astype(Z.dtype)
        else:
            # Alternative without SVD(B): one more projection to feature space and QR
            Yf = Z.T @ Q  # (d, ell)
            Qf, _ = np.linalg.qr(Yf, mode="reduced")
            u = Qf[:, 0]
    except np.linalg.LinAlgError:
        # Fallback to a coordinate with max variance
        var = Z.var(axis=0)
        u = np.zeros(d, dtype=Z.dtype)
        u[int(np.argmax(var))] = 1.0

    # 1D projection
    s = Z @ u  # (n,)

    if threshold == "sse":
        labels = _labels_sse_split(s)
    else:  # median threshold (default, per the sketch)
        labels, _ = _median_threshold(s)
        # Avoid degenerate empty cluster; if so, fall back to SSE split
        if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
            labels = _labels_sse_split(s)

    # Centroids in normalized space then unnormalize
    centroids_norm = _safe_centroids(Z, labels)
    centroids = centroids_norm * std + mean

    return labels, centroids


def numba_warmup():
    # Warm up by exercising the code path once with tiny arrays
    rng = np.random.default_rng(0)
    dummy_data = rng.standard_normal((10, 5))
    dummy_mean = np.zeros(5)
    dummy_std = np.ones(5)
    two_means_random_qr_threshold(
        dummy_data, dummy_mean, dummy_std, oversampling=3, power_iter=1
    )


numba_warmup()
