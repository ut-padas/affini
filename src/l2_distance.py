import faiss
from cuml.neighbors import NearestNeighbors as cuMLNearestNeighbors

from src import IS_CUPY_BACKEND, Array, xp


def _knn_l2_distance_cpu(
    query_points: Array, centroids: Array, trust_radii: Array, k_nn: int
) -> tuple[Array, Array, Array]:
    """
    Find K nearest neighbors using FAISS with trust radius checking.

    This is the main function you should use for K nearest neighbor search.

    Arg:
        query_points : xp.ndarray, shape (M, D)
            Query points to find neighbors for
        centroids : xp.ndarray, shape (K, D)
            Centroid points to search from
        trust_radii : xp.ndarray, shape (K,)
            Trust radius for each centroid
        k_nn : int
            Number of nearest neighbors to find

    Returns:
    --------
    indices : xp.ndarray, shape (M, K_NN)
        Indices of K nearest neighbors for each query point
    distances : xp.ndarray, shape (M, K_NN)
        L2 distances to K nearest neighbors for each query point
    unguaranteed_mask : xp.ndarray, shape (M,)
        Boolean mask indicating if nearest neighbor distance > trust radius
    """
    M, D = query_points.shape
    K, _ = centroids.shape

    # Ensure k_nn doesn't exceed number of centroids
    k_nn = min(k_nn, K)

    # Ensure data types are correct for FAISS (float32)
    centroids_f32 = centroids.astype(xp.float32)
    query_points_f32 = query_points.astype(xp.float32)

    # Build FAISS index
    index = faiss.IndexFlatL2(D)  # L2 distance index
    index.add(centroids_f32)  # type: ignore

    # Search for K nearest neighbors
    distances_sq, indices = index.search(query_points_f32, k_nn)  # type: ignore

    # Convert squared distances to L2 distances
    distances = xp.sqrt(distances_sq)

    # Check if nearest neighbor distance exceeds trust radius
    nearest_neighbor_indices = indices[:, 0]  # (M,)
    nearest_neighbor_distances = distances[:, 0]  # (M,)
    nearest_neighbor_trust_radii = trust_radii[nearest_neighbor_indices]  # (M,)

    # Create unguaranteed mask
    unguaranteed_mask = nearest_neighbor_distances > nearest_neighbor_trust_radii

    return indices, distances, unguaranteed_mask


def _knn_l2_distance_gpu(
    query_points: Array, centroids: Array, trust_radii: Array, k_nn: int
) -> tuple[Array, Array, Array]:
    """
    Find K nearest neighbors using cuML with trust radius checking.
    """
    index = cuMLNearestNeighbors(metric="euclidean", n_neighbors="brute")
    index.fit(centroids)
    distances, indices = index.kneighbors(query_points, n_neighbors=k_nn)

    # Check if nearest neighbor distance exceeds trust radius
    nearest_neighbor_indices = indices[:, 0]  # (M,)
    nearest_neighbor_distances = distances[:, 0]  # (M,)
    nearest_neighbor_trust_radii = trust_radii[nearest_neighbor_indices]  # (M,)

    # Create unguaranteed mask
    unguaranteed_mask = nearest_neighbor_distances > nearest_neighbor_trust_radii
    return indices, distances, unguaranteed_mask


def knn_l2_distance(
    query_points: Array, centroids: Array, trust_radii: Array, k_nn: int
) -> tuple[Array, Array, Array]:
    """
    Find K nearest neighbors using either FAISS (CPU) or cuML (GPU) with trust radius checking.

    This is the main function you should use for K nearest neighbor search.

    Arg:
        query_points: (M, D) Query points.
        centroids: (K, D) Centroid centroids.
        trust_radii: (K,) Trust radius for each centroid.
        k_nn: Number of nearest neighbors to find.

    Returns:
        nn_indices: (M, K_NN) Indices of K nearest neighbors for each query point.
        nn_distances: (M, K_NN) L2 distances to K nearest neighbors for each query point.
        unguaranteed_mask: (M,) Boolean mask indicating if nearest neighbor distance > trust radius.
    """
    if IS_CUPY_BACKEND:
        return _knn_l2_distance_gpu(query_points, centroids, trust_radii, k_nn)
    else:
        return _knn_l2_distance_cpu(query_points, centroids, trust_radii, k_nn)
