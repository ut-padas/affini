"""
Fast PCA-based 2-means clustering using optimal 1D projections.
References:
- ./docs/Two-Means PCA.pdf
"""

import numpy as np


def find_optimal_split_1d(s_sorted: np.ndarray) -> int:
    """
    Find optimal split point in sorted 1D projections to minimize within-cluster SSE.

    Args:
        s_sorted: Sorted 1D projections of shape (n,)

    Returns:
        best_k: Index to split at (first best_k points in cluster 0, rest in cluster 1)
    """
    # Prefix sums for exact SSE scan in 1D projected space
    pref_sum = np.cumsum(s_sorted)  # (n,) - sum of 1D projections
    pref_sq = np.cumsum(s_sorted**2)  # (n,) - sum of squared 1D projections
    total_sum = pref_sum[-1]
    total_sq = pref_sq[-1]

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


def estimate_hessian_abs_sqrt(X: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Estimates the absolute value square root of the Hessian |H|^{1/2}
    for a single cluster of centered points and gradients.

    Args:
        X: (N, d) centered points (x_i - mu)
        G: (N, d) centered gradients (g_i - g_mu)

    Returns:
        M: (d, d) symmetric matrix where M @ M = |H|
    """
    d = X.shape[1]
    TINY_EPS = 1e-12

    # 1. Compute C (Covariance-like) and S (Symmetrized Gradient Outer Product)
    # Equivalent to C = X.T @ X and S = X.T @ G + G.T @ X
    C = X.T @ X
    S = X.T @ G + G.T @ X

    # 2. Eigendecompose C to enter the coordinate system where it is diagonal
    # C = U @ diag(lambdas) @ U.T
    lambdas, U = np.linalg.eigh(C)

    # Clip eigenvalues to avoid division by zero in the Sylvester solve
    lambdas = np.maximum(lambdas, TINY_EPS)

    # 3. Rotate S into the U-basis: F_tilde = U.T @ S @ U
    S_tilde = U.T @ S @ U

    # 4. Solve the Sylvester equation in the diagonal basis
    # Entry-wise: f_ij = s_tilde_ij / (lambda_i + lambda_j)
    # This represents the Hessian H in the U-basis.
    denom = lambdas[:, None] + lambdas[None, :]
    F = S_tilde / denom

    # 5. Eigendecompose F to handle the absolute value and square root
    # F = V @ diag(w) @ V.T
    w, V = np.linalg.eigh(F)

    # 6. Recover the eigenvectors of the original H: Z = U @ V
    Z = U @ V

    # 7. Construct M = Z @ diag(sqrt(|w|)) @ Z.T
    # This gives us the matrix square root of the absolute Hessian.
    w_abs_sqrt = np.sqrt(np.abs(w))

    # Efficiently compute Z @ diag(w_abs_sqrt) @ Z.T
    M = (Z * w_abs_sqrt) @ Z.T

    # Ensure final symmetry against floating point drift
    return 0.5 * (M + M.T)


def two_means_pca_threshold(
    X_norm: np.ndarray,
    mean: np.ndarray,
    split_method: str = "optimal",
) -> tuple[np.ndarray, np.ndarray]:
    """
    A fast, PCA-based k=2 clustering method that finds a split using 1D projections.

    Args:
        X_norm: Normalized data of shape (n, d), zero-mean per dimension
        mean: Mean used for normalization, shape (d,)
        split_method: Strategy for splitting the projected data.
                      'optimal' - Minimizes SSE (original behavior).
                      'median'  - Splits at the exact middle index (forces balanced tree).

    Returns:
        labels: Cluster labels of shape (n,), values in {0, 1}
        centroids: Centroids of shape (2, d) in original (unnormalized) space.
    """
    Z = X_norm  # already zero-mean per dimension
    n, d = Z.shape
    if n < 2:
        # Degenerate; fallback to trivial assignment
        labels = np.zeros(n, dtype=np.int64)
        centroids = np.array([Z.mean(axis=0), Z.mean(axis=0)]) + mean
        return labels, centroids

    # Top principal direction via covariance matrix
    try:
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

    # --- Determine Split Point ---
    if split_method == "optimal":
        best_k = find_optimal_split_1d(s_sorted)
    elif split_method == "median":
        best_k = n // 2
        # Edge case: if n is odd, n//2 is fine. If n=2, index 1.
        # Ensure we don't split at 0 or n (though n < 2 handled above)
        best_k = max(1, min(n - 1, best_k))
    else:
        raise ValueError(
            f"Unknown split_method: {split_method}. Use 'optimal' or 'median'."
        )

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
