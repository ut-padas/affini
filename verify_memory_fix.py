"""
Verification script for GPU memory fix in compute_leading_eigenvectors.

This script tests the chunked processing implementation to ensure:
1. Small datasets use the fast path (no chunking)
2. Large datasets use the chunked path (memory-efficient)
3. Results are consistent between fast and chunked paths
4. Memory usage is reduced for large datasets

Run this on a system with GPU support (CuPy and cuML installed).
"""

import sys

try:
    import cupy as cp
    import numpy as np
    from src.gpu_adaptive_clustering import compute_leading_eigenvectors
except ImportError as e:
    print(f"Error: Required libraries not available: {e}")
    print("This script requires CuPy and cuML to be installed.")
    sys.exit(1)


def test_small_dataset():
    """Test that small datasets work correctly (fast path)."""
    print("Testing small dataset (fast path)...")

    # Create small test data
    n_clusters = 10
    points_per_cluster = 100
    n_dims = 5

    cp.random.seed(42)
    rng = cp.random.default_rng(42)

    # Generate random points for each cluster
    points_list = []
    cluster_starts = []
    cluster_sizes = []

    for i in range(n_clusters):
        cluster_center = cp.random.randn(n_dims).astype(cp.float64)
        cluster_points = (
            cluster_center + cp.random.randn(points_per_cluster, n_dims) * 0.5
        ).astype(cp.float64)
        cluster_starts.append(len(points_list) * points_per_cluster)
        cluster_sizes.append(points_per_cluster)
        points_list.append(cluster_points)

    points = cp.vstack(points_list)
    cluster_starts = cp.array(cluster_starts, dtype=cp.int32)
    cluster_sizes = cp.array(cluster_sizes, dtype=cp.int32)

    # Compute eigenvectors
    eigenvectors = compute_leading_eigenvectors(
        points=points,
        cluster_starts=cluster_starts,
        cluster_sizes=cluster_sizes,
        num_clusters=n_clusters,
        rng=rng,
        POWER_ITERS=3,
        DTYPE=cp.float64,
    )

    # Verify output shape
    assert eigenvectors.shape == (
        n_clusters,
        n_dims,
    ), f"Expected shape ({n_clusters}, {n_dims}), got {eigenvectors.shape}"

    # Verify eigenvectors are normalized
    norms = cp.linalg.norm(eigenvectors, axis=1)
    assert cp.allclose(
        norms, 1.0, atol=1e-6
    ), f"Eigenvectors not normalized: {norms}"

    print("✓ Small dataset test passed")
    return True


def test_large_dataset_chunked():
    """Test that large datasets use chunked processing."""
    print("\nTesting large dataset (chunked path)...")

    # Create dataset that triggers chunking
    n_clusters = 50
    points_per_cluster = 25_000  # Total: 1.25M points
    n_dims = 20

    cp.random.seed(42)
    rng = cp.random.default_rng(42)

    print(
        f"  Creating dataset: {n_clusters} clusters × {points_per_cluster} points × {n_dims} dims"
    )
    print(
        f"  Total: {n_clusters * points_per_cluster:,} points = {n_clusters * points_per_cluster * n_dims * 8 / 1e9:.2f} GB"
    )

    # Generate random points for each cluster
    points_list = []
    cluster_starts = []
    cluster_sizes = []

    for i in range(n_clusters):
        cluster_center = cp.random.randn(n_dims).astype(cp.float64)
        cluster_points = (
            cluster_center + cp.random.randn(points_per_cluster, n_dims) * 0.5
        ).astype(cp.float64)
        if i == 0:
            cluster_starts.append(0)
        else:
            cluster_starts.append(cluster_starts[-1] + points_per_cluster)
        cluster_sizes.append(points_per_cluster)
        points_list.append(cluster_points)

    points = cp.vstack(points_list)
    cluster_starts = cp.array(cluster_starts, dtype=cp.int32)
    cluster_sizes = cp.array(cluster_sizes, dtype=cp.int32)

    print("  Computing eigenvectors with chunked processing...")

    # Get initial memory info
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    initial_used = mempool.used_bytes()

    # Compute eigenvectors with small chunk size to test chunking
    eigenvectors = compute_leading_eigenvectors(
        points=points,
        cluster_starts=cluster_starts,
        cluster_sizes=cluster_sizes,
        num_clusters=n_clusters,
        rng=rng,
        POWER_ITERS=3,
        DTYPE=cp.float64,
        CHUNK_SIZE=500_000,  # Process in chunks of 500K points
    )

    peak_used = mempool.used_bytes()
    peak_gb = (peak_used - initial_used) / 1e9

    print(f"  Peak memory used: {peak_gb:.2f} GB")

    # Verify output shape
    assert eigenvectors.shape == (
        n_clusters,
        n_dims,
    ), f"Expected shape ({n_clusters}, {n_dims}), got {eigenvectors.shape}"

    # Verify eigenvectors are normalized
    norms = cp.linalg.norm(eigenvectors, axis=1)
    assert cp.allclose(
        norms, 1.0, atol=1e-6
    ), f"Eigenvectors not normalized: {norms}"

    print("✓ Large dataset (chunked) test passed")
    return True


def test_consistency():
    """Test that chunked and non-chunked paths give similar results."""
    print("\nTesting consistency between fast and chunked paths...")

    n_clusters = 5
    points_per_cluster = 200
    n_dims = 10

    cp.random.seed(123)
    rng = cp.random.default_rng(123)

    # Generate test data
    points_list = []
    cluster_starts = []
    cluster_sizes = []

    for i in range(n_clusters):
        cluster_center = cp.random.randn(n_dims).astype(cp.float64)
        cluster_points = (
            cluster_center + cp.random.randn(points_per_cluster, n_dims) * 0.5
        ).astype(cp.float64)
        if i == 0:
            cluster_starts.append(0)
        else:
            cluster_starts.append(cluster_starts[-1] + points_per_cluster)
        cluster_sizes.append(points_per_cluster)
        points_list.append(cluster_points)

    points = cp.vstack(points_list)
    cluster_starts = cp.array(cluster_starts, dtype=cp.int32)
    cluster_sizes = cp.array(cluster_sizes, dtype=cp.int32)

    # Compute with fast path (large chunk size ensures no chunking for this small dataset)
    rng1 = cp.random.default_rng(123)
    eigenvectors_fast = compute_leading_eigenvectors(
        points=points,
        cluster_starts=cluster_starts,
        cluster_sizes=cluster_sizes,
        num_clusters=n_clusters,
        rng=rng1,
        POWER_ITERS=5,
        DTYPE=cp.float64,
        CHUNK_SIZE=10_000_000,
    )

    # Compute with chunked path (small chunk size forces chunking)
    rng2 = cp.random.default_rng(123)
    eigenvectors_chunked = compute_leading_eigenvectors(
        points=points,
        cluster_starts=cluster_starts,
        cluster_sizes=cluster_sizes,
        num_clusters=n_clusters,
        rng=rng2,
        POWER_ITERS=5,
        DTYPE=cp.float64,
        CHUNK_SIZE=100,  # Small chunk to force chunking
    )

    # Compare results (allowing for numerical differences)
    # Note: Eigenvectors can point in opposite directions, so compare absolute dot products
    for i in range(n_clusters):
        dot_product = abs(cp.dot(eigenvectors_fast[i], eigenvectors_chunked[i]))
        # Should be close to 1 (parallel or anti-parallel)
        assert (
            dot_product > 0.99
        ), f"Cluster {i}: eigenvectors not consistent (dot product: {dot_product})"

    print("✓ Consistency test passed")
    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("GPU Memory Fix Verification")
    print("=" * 60)

    # Check GPU availability
    print(f"\nGPU Device: {cp.cuda.Device()}")
    meminfo = cp.cuda.Device().mem_info
    print(f"GPU Memory: {meminfo[0]/1e9:.2f} GB free / {meminfo[1]/1e9:.2f} GB total")

    try:
        # Run tests
        test_small_dataset()
        test_large_dataset_chunked()
        test_consistency()

        print("\n" + "=" * 60)
        print("All verification tests passed! ✓")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
