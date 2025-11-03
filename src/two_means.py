"""
Fast PCA-based 2-means clustering using optimal 1D projections.
References:
- ./docs/Two-Means PCA.pdf
"""

import numpy as np
from numba import njit


@njit
def find_optimal_split_1d(s_sorted: np.ndarray) -> int:
    """
    Find optimal split point in sorted 1D projections to minimize within-cluster SSE.

    Args:
        s_sorted: Sorted 1D projections of shape (n,)

    Returns:
        best_k: Index to split at (first best_k points in cluster 0, rest in cluster 1)
    """
    # Prefix sums for exact SSE scan in 1D projected space
    # SSE(S) = sum(x^2) - sum(x)^2 / |S| for 1D projections
    pref_sum = np.cumsum(s_sorted)  # (n,) - sum of 1D projections
    pref_sq = np.cumsum(s_sorted**2)  # (n,) - sum of squared 1D projections
    total_sum = pref_sum[-1]
    total_sq = pref_sq[-1]

    # Vectorized computation of all SSE values
    n = len(s_sorted)
    k_vals = np.arange(1, n)

    left_sum = pref_sum[:-1]
    right_sum = total_sum - left_sum
    left_sq = pref_sq[:-1]
    right_sq = total_sq - left_sq

    sse_left = left_sq - (left_sum * left_sum) / k_vals
    rn = n - k_vals
    sse_right = right_sq - (right_sum * right_sum) / rn
    sse = sse_left + sse_right

    best_k = int(np.argmin(sse) + 1)  # +1 because k_vals starts at 1

    return best_k


def two_means_pca_threshold(
    X_norm: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    A fast, PCA-based k=2 clustering method that finds an optimal split using 1D projections.

    Args:
        X_norm: Normalized data of shape (n, d), zero-mean per dimension
        mean: Mean used for normalization, shape (d,)
        std: Stddev used for normalization, shape (d,)

    Returns:
        labels: Cluster labels of shape (n,), values in {0, 1}
        centroids: Centroids of shape (2, d) in original (unnormalized) space.

    References:
    - ./docs/Two-Means PCA.pdf
    """
    Z = X_norm  # already zero-mean per dimension
    n, d = Z.shape
    if n < 2:
        # Degenerate; fallback to trivial assignment
        labels = np.zeros(n, dtype=np.int64)
        centroids = np.array([Z.mean(axis=0), Z.mean(axis=0)]) * std + mean
        return labels, centroids

    # Top principal direction via covariance matrix (faster for small d)
    # For small d (typical in this project), computing Z^T @ Z is faster than full SVD
    # u shape (d,)
    try:
        # Compute covariance matrix: Z^T @ Z / (n-1), but we only need eigenvector
        cov = Z.T @ Z  # (d, d) - much smaller than (n, n) for n >> d
        _, eigvecs = np.linalg.eigh(cov)  # eigh is faster for symmetric matrices
        u = eigvecs[:, -1]  # Last column = eigenvector with largest eigenvalue
    except np.linalg.LinAlgError:
        # Fallback: use the direction of max variance coordinate
        var = Z.var(axis=0)
        e = np.zeros(d)
        e[int(np.argmax(var))] = 1.0
        u = e

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
    # Already have sorted data: Z_sorted
    # For cluster 0 (first best_k points): sum and divide by best_k
    # For cluster 1 (remaining points): use (total_sum - c0_sum) / (n - best_k)
    total_Z_sum = np.sum(Z, axis=0)  # Sum across all points (d,)
    c0_sum = np.sum(Z_sorted[:best_k], axis=0)  # Sum of first best_k points
    c1_sum = total_Z_sum - c0_sum

    c0_norm = c0_sum / best_k
    c1_norm = c1_sum / (n - best_k)
    centroids_norm = np.vstack([c0_norm, c1_norm])
    centroids = centroids_norm * std + mean

    return labels, centroids


def numba_warmup():
    # Warm up Numba compilation
    dummy_data = np.random.randn(10, 5)
    dummy_mean = np.zeros(5)
    dummy_std = np.ones(5)
    two_means_pca_threshold(dummy_data, dummy_mean, dummy_std)


numba_warmup()
