import time

import cupy as cp
from cuml.cluster import KMeans as cuKMeans

_HAS_GPU = cp.is_available()
if not _HAS_GPU:
    raise ImportError("GPU not available.")

CP_DTYPE = cp.float64
SEED = 0
N_SAMPLES = 100_000
N_INIT_CLUSTERS = 100
ABS_TOL = 1e-5
REL_TOL = 1e-3


def sample_function(X: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    X: (n_samples, 3)
    f(x) = [x1^2 + x2^2, x1 - x2 + x3] (n_samples, 2)
    Jacobian/Gradient: (n_samples, 2, 3)
    [[2*x1, 2*x2, 0],
        [1,    -1,  1]]
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    f1 = x1**2 + x2**2
    f2 = x1 - x2 + x3
    F = cp.stack([f1, f2], axis=1)

    g1 = cp.stack([2 * x1, 2 * x2, cp.zeros_like(x1)], axis=1)
    g2 = cp.stack([cp.ones_like(x1), -cp.ones_like(x1), cp.ones_like(x1)], axis=1)
    J = cp.stack([g1, g2], axis=1)
    return F, J


rng = cp.random.default_rng(SEED)
X = rng.random((N_SAMPLES, 3), dtype=CP_DTYPE)  # type: ignore

print(f"Running on GPU. N_SAMPLES={N_SAMPLES:2e}")
tic = time.perf_counter()
F, J = sample_function(X)
toc = time.perf_counter()
print(f"Function and Jacobian computed in {toc - tic:0.4f} seconds.")


print(f"Performing KMeans clustering with {N_INIT_CLUSTERS} clusters.")
tic = time.perf_counter()
km = cuKMeans(n_clusters=N_INIT_CLUSTERS, random_state=SEED)
labels = km.fit_predict(X)
centroids = km.cluster_centers_
toc = time.perf_counter()
print(f"KMeans clustering completed in {toc - tic:0.4f} seconds.")

print("Calculating function and Jacobian at centroids.")
tic = time.perf_counter()
F_centroids, J_centroids = sample_function(centroids)
toc = time.perf_counter()
print(f"Function and Jacobian at centroids computed in {toc - tic:0.4f} seconds.")

print(
    "Calculating first-order Taylor approximation of f at all points using corresponding centroids."
)
tic = time.perf_counter()
centroids_expanded = centroids[labels]
dx = X - centroids_expanded
f_taylor = F_centroids[labels] + cp.einsum("ijk,ik->ij", J_centroids[labels], dx)
toc = time.perf_counter()
print(f"First-order Taylor approximation computed in {toc - tic:0.4f} seconds.")

print("Evaluating approximation errors.")
tic = time.perf_counter()
errors_in_f = cp.linalg.norm(F - f_taylor, axis=1)
max_error = cp.max(errors_in_f)
mean_error = cp.mean(errors_in_f)
std_error = cp.std(errors_in_f)
print(
    f"Taylor approximation errors -- max: {max_error:0.4e}, mean: {mean_error:0.4e}, std: {std_error:0.4e}."
)

# Flag each point where the error exceeds tolerances
tolerance_violations = cp.where(
    (errors_in_f > ABS_TOL) | (errors_in_f > REL_TOL * cp.linalg.norm(F, axis=1))
)[0]
num_violations = tolerance_violations.size
print(f"Number of points exceeding tolerances: {num_violations} out of {N_SAMPLES}.")
toc = time.perf_counter()
print(f"Error evaluation completed in {toc - tic:0.4f} seconds.")

# Get cluster IDs of the violations
clusters_with_violations = cp.unique(labels[tolerance_violations])
clusters_with_no_violations = cp.setdiff1d(
    cp.arange(N_INIT_CLUSTERS), clusters_with_violations
)
print(f"Clusters with no violations: {clusters_with_no_violations}")

# For those with violations, resplit into 2 clusters
if clusters_with_violations.size > 0:
    print("Resplitting clusters with violations into 2 sub-clusters each.")
    tic = time.perf_counter()
    for cluster_id in clusters_with_violations:
        cluster_points = X[labels == cluster_id]
        if cluster_points.shape[0] < 2:
            raise ValueError("Not enough points to resplit.")
        sub_km = cuKMeans(n_clusters=2, random_state=SEED)
        sub_labels = sub_km.fit_predict(cluster_points)
        # Here you would typically store or process the new sub-clusters
    toc = time.perf_counter()
    print(f"Resplitting completed in {toc - tic:0.4f} seconds.")
