import faiss
import numpy as np


def knn_l2_distance(
    query_points: np.ndarray, centroids: np.ndarray, trust_radii: np.ndarray, k_nn: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find K nearest neighbors using FAISS with trust radius checking.

    This is the main function you should use for K nearest neighbor search.

    Parameters:
    -----------
    query_points : np.ndarray, shape (M, D)
        Query points to find neighbors for
    centroids : np.ndarray, shape (K, D)
        Centroid points to search from
    trust_radii : np.ndarray, shape (K,)
        Trust radius for each centroid
    k_nn : int
        Number of nearest neighbors to find

    Returns:
    --------
    indices : np.ndarray, shape (M, K_NN)
        Indices of K nearest neighbors for each query point
    distances : np.ndarray, shape (M, K_NN)
        L2 distances to K nearest neighbors for each query point
    unguaranteed_mask : np.ndarray, shape (M,)
        Boolean mask indicating if nearest neighbor distance > trust radius
    """

    M, D = query_points.shape
    K, _ = centroids.shape

    # Ensure k_nn doesn't exceed number of centroids
    k_nn = min(k_nn, K)

    # Ensure data types are correct for FAISS (float32)
    centroids_f32 = centroids.astype(np.float32)
    query_points_f32 = query_points.astype(np.float32)

    # Build FAISS index
    index = faiss.IndexFlatL2(D)  # L2 distance index
    index.add(centroids_f32)  # type: ignore

    # Search for K nearest neighbors
    distances_sq, indices = index.search(query_points_f32, k_nn)  # type: ignore

    # Convert squared distances to L2 distances
    distances = np.sqrt(distances_sq)

    # Check if nearest neighbor distance exceeds trust radius
    nearest_neighbor_indices = indices[:, 0]  # (M,)
    nearest_neighbor_distances = distances[:, 0]  # (M,)
    nearest_neighbor_trust_radii = trust_radii[nearest_neighbor_indices]  # (M,)

    # Create unguaranteed mask
    unguaranteed_mask = nearest_neighbor_distances > nearest_neighbor_trust_radii

    return indices, distances, unguaranteed_mask
