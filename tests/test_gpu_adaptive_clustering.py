import cupy as cp
import pytest

from src.gpu_adaptive_clustering import (
    CP_DTYPE,
    SEED,
    generate_extra_points,
    median_cut_projection,
    randomized_qr_gpu,
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
        n, d = 1000, 20
        X = cp.random.standard_normal((n, d), dtype=CP_DTYPE)

        # Get leading left singular vector using randomized QR with high power iterations
        q_rqr = randomized_qr_gpu(X, rng, ell=d, power_iterations=50)

        # Get leading left singular vector using SVD
        U, S, Vt = cp.linalg.svd(X, full_matrices=False)
        q_svd = Vt[0, :]  # Leading right singular vector (corresponds to left in X^T)

        # The vectors should be aligned (dot product close to ±1)
        dot_product = float(abs(cp.dot(q_rqr, q_svd)))
        assert (
            dot_product > 0.99
        ), "Randomized QR vector does not match SVD leading vector"

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
