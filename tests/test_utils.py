import unittest

import numpy as np

from src import xp
from src.utils import repeats


class TestRepeats(unittest.TestCase):
    def test_basic_functionality(self):
        """Test basic repeat functionality with simple cases."""
        values = xp.array([1, 2, 3])
        repeat_counts = xp.array([2, 3, 1])
        expected = xp.array([1, 1, 2, 2, 2, 3])
        result = repeats(values, repeat_counts)

        if hasattr(result, "get"):  # CuPy array
            result = result.get()
        if hasattr(expected, "get"):  # CuPy array
            expected = expected.get()

        np.testing.assert_array_equal(result, expected)

    def test_zero_repeats(self):
        """Test handling of zero repeats and mixed patterns."""
        values = xp.array([10, 20, 30, 40])
        repeat_counts = xp.array([1, 0, 3, 2])
        expected = xp.array([10, 30, 30, 30, 40, 40])
        result = repeats(values, repeat_counts)

        if hasattr(result, "get"):
            result = result.get()
        if hasattr(expected, "get"):
            expected = expected.get()

        np.testing.assert_array_equal(result, expected)

    def test_empty_arrays(self):
        """Test edge cases with empty inputs and all-zero repeats."""
        # Empty arrays
        values = xp.array([])
        repeat_counts = xp.array([], dtype=xp.int32)
        result = repeats(values, repeat_counts)
        self.assertEqual(len(result), 0)

        # All zero repeats
        values = xp.array([1, 2, 3])
        repeat_counts = xp.array([0, 0, 0])
        result = repeats(values, repeat_counts)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
