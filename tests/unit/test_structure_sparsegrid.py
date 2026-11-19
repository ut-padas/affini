"""SparseCheb (src/structure/schemes/sparsegrid.py): the check set and the look-ahead it triggers."""

import numpy as np

from src.problems import FUNCTIONS
from src.structure.schemes import sparsegrid


def test_a_failed_check_opens_what_the_surpluses_stopped_before():
    p = FUNCTIONS["genz_corner_peak"]
    lo, hi = p.domain(2)
    X = p.test_points(2)
    fX = p.values(X)
    atol = 1e-3 * np.max(np.abs(fX))
    model = sparsegrid.build(p.values, lo, hi, atol, seed=2)
    assert model.checks >= 1
    assert np.max(np.abs(model.predict(X) - fX)) <= atol


def test_the_check_set_is_sobol_points_plus_the_corners():
    lo, hi = np.array([-1.0, 0.0, 2.0]), np.array([1.0, 1.0, 3.0])
    X = sparsegrid.check_points(lo, hi, 16, seed=0)
    assert X.shape == (16 + 8, 3)
    assert np.all((X >= lo) & (X <= hi))
    corners = {tuple(c) for c in X[16:]}
    assert (-1.0, 0.0, 2.0) in corners and (1.0, 1.0, 3.0) in corners
