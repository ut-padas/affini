"""Fast NumPy replacements in src/numerics.py must reproduce the operations they replace exactly."""

import numpy as np
import pytest

from src import IS_CUPY_BACKEND, xp
from tests._helpers import host


@pytest.mark.skipif(IS_CUPY_BACKEND, reason="CuPy keeps its atomic add.at, whose order varies per run")
def test_segment_sum_is_bit_identical_to_add_at():
    from src.numerics import segment_sum

    rng = np.random.default_rng(0)
    for trial in range(50):
        K, n, d = int(rng.integers(1, 30)), int(rng.integers(0, 2000)), int(rng.integers(1, 5))
        labels = rng.integers(0, K, n)
        if trial % 2:
            labels = np.sort(labels)
        values = rng.normal(size=(n, d, d)) * 10 ** rng.uniform(-3, 3, (n, 1, 1))
        ref = np.zeros((K, d, d))
        np.add.at(ref, labels, values)
        np.testing.assert_array_equal(host(segment_sum(xp.asarray(values), xp.asarray(labels), K)), ref)


def test_sort_by_block_then_value_matches_lexsort_with_ties():
    from src.numerics import sort_by_block_then_value

    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(1, 2000))
        blocks, values = np.sort(rng.integers(0, 30, n)), np.round(rng.normal(size=n), 1)
        np.testing.assert_array_equal(
            host(sort_by_block_then_value(xp.asarray(values), xp.asarray(blocks))),
            np.lexsort(np.stack([values, blocks])),
        )
