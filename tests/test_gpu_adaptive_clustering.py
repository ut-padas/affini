import cupy as cp
import pytest

from src.gpu_adaptive_clustering import (
    CP_DTYPE,
    SEED,
    generate_extra_points,
    get_centroid,
    median_cut_projection,
    randomized_qr_gpu,
    taylor_errors,
)


class TestRandomizedQR:
    """Test cases for randomized_qr_gpu function."""

    def test_randomized_qr_vs_svd_high_power_iters(self):
        """
        Test that randomized QR with high power iterations matches SVD leading vector.
        """
        cp.random.seed(SEED)
        rng = cp.random.default_rng(SEED)

        # Create a test matrix with known structure
        n, d = 1000, 50
        X = cp.random.standard_normal((n, d), dtype=CP_DTYPE)

        # Get leading left singular vector using randomized QR with high power iterations
        q_rqr = randomized_qr_gpu(X, rng, ell=min(20, d), power_iterations=10)

        # Get leading left singular vector using SVD
        U, S, Vt = cp.linalg.svd(X, full_matrices=False)
        q_svd = Vt[0, :]  # Leading right singular vector (corresponds to left in X^T)

        # The vectors should be aligned (dot product close to ±1)
        dot_product = abs(cp.dot(q_rqr, q_svd))
        assert cp.allclose(
            dot_product, 1.0, atol=1e-5
        ), f"Dot product {dot_product} should be close to 1"

        # Both should be unit vectors
        assert abs(cp.linalg.norm(q_rqr) - 1.0) < 1e-10, "q_rqr should be a unit vector"
        assert abs(cp.linalg.norm(q_svd) - 1.0) < 1e-10, "q_svd should be a unit vector"


class TestMedianCutProjection:
    """Test cases for median_cut_projection function."""

    def test_median_cut_splits_evenly(self):
        """Test that median cut splits points roughly evenly."""
        cp.random.seed(SEED)

        # Create points along a line
        n = 1000
        X_1d = cp.linspace(-5.0, 5.0, n)
        X = X_1d.reshape(-1, 1)  # (n, 1) # type: ignore
        q = cp.array([1.0], dtype=CP_DTYPE)  # Project along x-axis

        left, right = median_cut_projection(X, q)

        # Should split roughly in half
        n_left = int(left.sum())
        n_right = int(right.sum())

        assert n_left + n_right == n
        assert abs(n_left - n_right) <= 1  # At most 1 difference due to odd n

        # Left points should be <= median, right points > median
        projections = X @ q
        median_val = cp.median(projections)

        assert cp.all(projections[left] <= median_val)
        assert cp.all(projections[right] > median_val)

    def test_median_cut_2d_diagonal(self):
        """Test median cut on 2D points with diagonal projection."""
        cp.random.seed(SEED)

        # Create 2D grid points
        x = cp.linspace(-2, 2, 20)
        y = cp.linspace(-2, 2, 20)
        X1, X2 = cp.meshgrid(x, y)
        X = cp.stack([X1.flatten(), X2.flatten()], axis=1)  # (400, 2)

        # Project along diagonal
        q = cp.array([1.0, 1.0], dtype=CP_DTYPE)
        q /= cp.linalg.norm(q)

        left, right = median_cut_projection(X, q)

        # Verify split properties
        assert left.sum() + right.sum() == X.shape[0]
        assert left.sum() > 0 and right.sum() > 0

        # Check that the split follows the median
        projections = X @ q
        median_val = cp.median(projections)

        assert cp.all(projections[left] <= median_val)
        assert cp.all(projections[right] > median_val)


class TestGetCentroid:
    """Test cases for get_centroid function."""

    def test_centroid_single_point(self):
        """Test centroid of single point."""
        X = cp.array([[1.0, 2.0, 3.0]], dtype=CP_DTYPE)
        centroid = get_centroid(X)

        cp.testing.assert_array_equal(centroid, X[0])

    def test_centroid_symmetric_points(self):
        """Test centroid of symmetric points."""
        # Points symmetric around origin
        X = cp.array(
            [
                [+1.0, +2.0],
                [-1.0, -2.0],
                [+2.0, -1.0],
                [-2.0, +1.0],
            ],
            dtype=CP_DTYPE,
        )

        centroid = get_centroid(X)
        expected = cp.array([0.0, 0.0], dtype=CP_DTYPE)

        cp.testing.assert_allclose(centroid, expected)

    def test_centroid_known_points(self):
        """Test centroid of known points."""
        X = cp.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            dtype=CP_DTYPE,
        )

        centroid = get_centroid(X)
        expected = cp.array([0.75, 0.75, 0.75], dtype=CP_DTYPE)

        cp.testing.assert_allclose(centroid, expected)


class TestTaylorErrors:
    """Test cases for taylor_errors function."""

    def test_taylor_errors_vs_loop_implementation(self):
        """Test that einsum implementation matches explicit loop implementation."""
        cp.random.seed(SEED)

        # Set up test data
        n, d, fd = 50, 4, 3
        rng = cp.random.default_rng(SEED)
        assert rng is not None, "RNG initialization failed."

        X = rng.standard_normal((n, d), dtype=CP_DTYPE)
        F = rng.standard_normal((n, fd), dtype=CP_DTYPE)
        centroid = rng.standard_normal((1, d), dtype=CP_DTYPE)
        F_c = rng.standard_normal((1, fd), dtype=CP_DTYPE)
        J_c = rng.standard_normal((1, fd, d), dtype=CP_DTYPE)

        # Einsum implementation (original)
        err_einsum = taylor_errors(X, F, centroid, F_c, J_c)

        # Loop implementation for comparison
        dx = X - centroid  # (n, d)
        f_taylor_loop = cp.zeros((n, fd), dtype=CP_DTYPE)

        for i in range(n):
            # For each point i, compute F_c + J_c @ dx[i]
            f_taylor_loop[i] = F_c[0] + J_c[0] @ dx[i]

        err_loop = cp.linalg.norm(F - f_taylor_loop, axis=1)

        # Compare results
        cp.testing.assert_allclose(err_einsum, err_loop)


class TestGenerateExtraPoints:
    """Test cases for generate_extra_points function."""

    def test_convex_combination_property(self):
        """Test that generated points are convex combinations of original points."""
        cp.random.seed(SEED)
        rng = cp.random.default_rng(SEED)

        # Original points
        X = cp.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [1.0, 2.0],
                [0.0, 1.0],
            ],
            dtype=CP_DTYPE,
        )

        min_points = 10
        new_X = generate_extra_points(X, min_points, rng)

        # Check that we got the expected number of points
        expected_extra = max(1, min_points - X.shape[0] + 1)
        assert (
            new_X.shape[0] == expected_extra
        ), f"Expected {expected_extra} points, got {new_X.shape[0]}"
        assert (
            new_X.shape[1] == X.shape[1]
        ), f"Expected dimension {X.shape[1]}, got {new_X.shape[1]}"

        # Each generated point should be inside the convex hull of original points
        # For convex combination λ*p1 + (1-λ)*p2, point should be on line segment
        Q = X.shape[0]
        for i in range(new_X.shape[0]):
            point = new_X[i]

            # Find the closest pair of original points
            min_dist = float("inf")

            for j in range(Q):
                for k in range(Q):
                    if j != k:
                        # Check if point lies on line segment between X[j] and X[k]
                        p1, p2 = X[j], X[k]

                        # Project point onto line segment
                        v = p2 - p1
                        if cp.linalg.norm(v) > 1e-10:
                            t = cp.dot(point - p1, v) / cp.dot(v, v)
                            t = cp.clip(t, 0, 1)
                            proj = p1 + t * v
                            dist = cp.linalg.norm(point - proj)

                            if dist < min_dist:
                                min_dist = dist

            # The point should be very close to some line segment
            assert (
                min_dist < 1e-8
            ), f"Point {point} not on any line segment, min_dist={min_dist}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
