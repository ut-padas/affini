import time

import numpy as np
from sklearn.cluster import KMeans

NP_DTYPE = np.float64
SEED = 0
N_SAMPLES = 10_000
N_INIT_CLUSTERS = 100
ABS_TOL = 1e-5
REL_TOL = 1e-3


def sample_function(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    F = np.stack([f1, f2], axis=1)

    g1 = np.stack([2 * x1, 2 * x2, np.zeros_like(x1)], axis=1)
    g2 = np.stack([np.ones_like(x1), -np.ones_like(x1), np.ones_like(x1)], axis=1)
    J = np.stack([g1, g2], axis=1)
    return F, J


rng = np.random.default_rng(SEED)
X = rng.random((N_SAMPLES, 3), dtype=NP_DTYPE)

print(f"Running on CPU. N_SAMPLES={N_SAMPLES:2e}")
tic = time.perf_counter()
F, J = sample_function(X)
toc = time.perf_counter()
print(f"Function and Jacobian computed in {toc - tic:0.4f} seconds.")

print(f"Performing KMeans clustering with {N_INIT_CLUSTERS} clusters.")
tic = time.perf_counter()
km = KMeans(n_clusters=N_INIT_CLUSTERS, random_state=SEED)
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
f_taylor = F_centroids[labels] + np.einsum("ijk,ik->ij", J_centroids[labels], dx)
toc = time.perf_counter()
print(f"First-order Taylor approximation computed in {toc - tic:0.4f} seconds.")

print("Evaluating approximation errors.")
tic = time.perf_counter()
errors_in_f = np.linalg.norm(F - f_taylor, axis=1)
max_error = np.max(errors_in_f)
mean_error = np.mean(errors_in_f)
std_error = np.std(errors_in_f)
print(
    f"Taylor approximation errors -- max: {max_error:0.4e}, mean: {mean_error:0.4e}, std: {std_error:0.4e}."
)

# Flag each point where the error exceeds tolerances
tolerance_violations = np.where(
    (errors_in_f > ABS_TOL) | (errors_in_f > REL_TOL * np.linalg.norm(F, axis=1))
)[0]
num_violations = tolerance_violations.size
print(f"Number of points exceeding tolerances: {num_violations} out of {N_SAMPLES}.")
toc = time.perf_counter()
print(f"Error evaluation completed in {toc - tic:0.4f} seconds.")

# Get cluster IDs of the violations
clusters_with_violations = np.unique(labels[tolerance_violations])
clusters_with_no_violations = np.setdiff1d(
    np.arange(N_INIT_CLUSTERS), clusters_with_violations
)
print(f"Clusters with no violations: {clusters_with_no_violations}")
