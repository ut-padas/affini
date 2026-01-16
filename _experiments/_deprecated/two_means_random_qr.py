"""
Randomized-QR based fast 2-means using optimal 1D projections.

References:
- Halko, Martinsson, Tropp. Finding structure with randomness: Probabilistic
  algorithms for constructing approximate matrix decompositions. SIAM Review, 2011.
- Project notes: randomized QR sketch (image) with batched QR steps.
"""

import numpy as np

from src.two_means import find_optimal_split_1d


def two_means_random_qr_threshold(
    X_norm: np.ndarray,
    mean: np.ndarray,
    *,
    ell: int = 8,
    power_iter: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomized-QR 2-means using a fast principal direction estimation.

    Args:
        X_norm: Normalized data of shape (n, d), ideally zero-mean per feature.
        mean: Mean used for normalization, shape (d,)
        ell: Subspace size.
        power_iter: Number of subspace power iterations (0, 1, or 2 are typical).
        random_state: Seed for reproducibility.
        threshold: "median" to follow sketch; "sse" to use optimal 1D SSE scan.
        use_small_svd: Use SVD on small matrix B = Q^T Z to get direction.

    Returns:
        labels: (n,) int64 in {0,1}
        centroids: (2, d) in original, unnormalized space
    """
    Z = X_norm
    X0 = X_norm.T
    d, n = X0.shape
    if n < 2:
        raise ValueError("Need at least 2 data points for 2-means clustering.")

    rng = np.random.default_rng(seed)
    ell = min(ell, d)  # cannot exceed feature dimension

    # Random Gaussian test matrix
    Omega = rng.standard_normal(size=(n, ell))

    # Range finder with subspace iterations; follow the batched QR sketch.
    Y = X0 @ Omega
    Q, _ = np.linalg.qr(Y)
    for _ in range(max(power_iter, 0)):
        Y = X0 @ (X0.T @ Q)
        Q, _ = np.linalg.qr(Y)

    u = Q[:, 0]  # First basis vector approximates top principal direction

    # Project and sort
    s = Z @ u  # shape (n,)
    order = np.argsort(s)
    s_sorted = s[order]  # sorted 1D projections
    Z_sorted = Z[order]  # sorted original data (for centroid calculation later)

    # Find optimal split point in 1D projected space
    best_k = find_optimal_split_1d(s_sorted)

    # Assign labels according to best threshold (first best_k sorted points -> cluster 0)
    labels_sorted = np.zeros(n, dtype=np.int64)
    labels_sorted[best_k:] = 1
    # Unsort to original order
    labels = np.empty(n, dtype=np.int64)
    labels[order] = labels_sorted

    # Compute centroids efficiently using prefix sums in multidimensional space
    total_Z_sum = np.sum(Z, axis=0)  # Sum across all points (d,)
    c0_sum = np.sum(Z_sorted[:best_k], axis=0)  # Sum of first best_k points
    c1_sum = total_Z_sum - c0_sum

    c0_norm = c0_sum / best_k
    c1_norm = c1_sum / (n - best_k)
    centroids_norm = np.vstack([c0_norm, c1_norm])
    centroids = centroids_norm + mean

    return labels, centroids
