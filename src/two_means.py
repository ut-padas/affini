import numpy as np


def two_means_pca_threshold(
    X_norm: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast, rigorous k=2 split:
    1) Compute top principal direction u of X_norm (SVD on centered data; X_norm already z-scored per-dim).
    2) Project onto u, sort, and scan all thresholds exactly using prefix sums to minimize within-cluster SSE.

    Returns labels and centroids in original (unnormalized) space.
    Complexity: O(n d + n log n), excellent for n up to a few thousand.
    """
    Z = X_norm  # already zero-mean per dimension
    n, d = Z.shape
    if n < 2:
        # Degenerate; fallback to trivial assignment
        labels = np.zeros(n, dtype=int)
        centroids = np.array([Z.mean(axis=0), Z.mean(axis=0)]) * std + mean
        return labels, centroids

    # Top principal direction via SVD on Z (n x d), d is small in this project
    # u shape (d,)
    try:
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        u = Vt[0]
    except np.linalg.LinAlgError:
        # Fallback: use the direction of max variance coordinate
        var = Z.var(axis=0)
        e = np.zeros(d)
        e[int(np.argmax(var))] = 1.0
        u = e

    # Project and sort
    s = Z @ u  # shape (n,)
    order = np.argsort(s)
    Z_sorted = Z[order]

    # Prefix sums for exact SSE scan in normalized space
    # SSE(S) = sum(||x||^2) - ||sum(x)||^2 / |S|
    pref_sum = np.cumsum(Z_sorted, axis=0)  # (n, d)
    pref_sq = np.cumsum(np.einsum("ij,ij->i", Z_sorted, Z_sorted))  # (n,)
    total_sum = pref_sum[-1]
    total_sq = pref_sq[-1]

    best_k = 1
    best_sse = np.inf
    # Scan all split points 1..n-1
    for k in range(1, n):
        left_sum = pref_sum[k - 1]
        right_sum = total_sum - left_sum
        left_sq = pref_sq[k - 1]
        right_sq = total_sq - left_sq

        sse_left = left_sq - (left_sum @ left_sum) / k
        rn = n - k
        sse_right = right_sq - (right_sum @ right_sum) / rn
        sse = sse_left + sse_right
        if sse < best_sse:
            best_sse = sse
            best_k = k

    # Assign labels according to best threshold (first best_k sorted points -> cluster 0)
    labels_sorted = np.zeros(n, dtype=int)
    labels_sorted[best_k:] = 1
    # Unsort to original order
    labels = np.empty(n, dtype=int)
    labels[order] = labels_sorted

    # Compute centroids in normalized space and map back
    sum_left = pref_sum[best_k - 1]
    k = best_k
    rn = n - k
    c0_norm = sum_left / k
    c1_norm = (total_sum - sum_left) / rn
    centroids_norm = np.vstack([c0_norm, c1_norm])
    centroids = centroids_norm * std + mean

    return labels, centroids
