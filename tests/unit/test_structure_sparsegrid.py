"""SparseCheb (src/structure/sparsegrid.py): stopping with a one-step look-ahead."""

import numpy as np

from src.structure import sparsegrid
from src.structure.problems import PROBLEMS


def test_the_look_ahead_finds_what_the_surpluses_stop_before():
    p = PROBLEMS["genz_corner_peak"]
    lo, hi = p.domain(2)
    X = p.test_points(2)
    fX = p.values(X)
    atol = 1e-3 * np.max(np.abs(fX))
    model = sparsegrid.build(p.values, lo, hi, atol, seed=2)
    assert model.look_aheads >= 1
    assert np.max(np.abs(model.predict(X) - fX)) <= atol
