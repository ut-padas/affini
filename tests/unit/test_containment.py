"""Tests for src/neighbors/containment.py against a brute-force search over every (query, leaf) pair."""

import numpy as np
import pytest

from src import xp
from src.neighbors import TrustRegionContainment, best_containing
from tests._helpers import host


def _elongated_leaves(rng, n, d, offset=0.0):
    """Centres and trust matrices of ellipsoids with axes from 1e-3 to 3 (aspect up to 3,000)."""
    C = offset + rng.uniform(-1, 1, (n, d))
    Q = np.linalg.qr(rng.standard_normal((n, d, d)))[0]
    axes = np.exp(rng.uniform(np.log(1e-3), np.log(3.0), (n, d)))
    T = Q @ ((1.0 / axes**2)[:, :, None] * np.swapaxes(Q, 1, 2))
    return C, T, Q * axes[:, None, :]


def _brute(C, T, X):
    dx = X[:, None, :] - C[None, :, :]
    return np.einsum("qni,nij,qnj->qn", dx, T, dx)


@pytest.mark.parametrize("d, offset", [(2, 0.0), (7, 0.0), (30, 0.0), (7, 1e6)])
def test_containing_finds_exactly_the_pairs_brute_force_finds(d, offset):
    rng = np.random.default_rng(d)
    C, T, axes = _elongated_leaves(rng, 300, d, offset)
    # Half the queries in a leaf's own axes, about half of them inside it; half anywhere.
    i = rng.integers(0, 300, 200)
    z = rng.standard_normal((200, d)) / np.sqrt(d)
    X = np.concatenate([C[i] + np.einsum("qij,qj->qi", axes[i], z), offset + rng.uniform(-1.5, 1.5, (200, d))])
    rows, leaves, dist = (host(a) for a in TrustRegionContainment(xp.asarray(C), xp.asarray(T)).containing(xp.asarray(X)))
    ref = _brute(C, T, X)
    clear = np.abs(ref - 1.0) > 1e-9  # pairs on the boundary may go either way
    found = np.zeros_like(ref, dtype=bool)
    found[rows, leaves] = True
    np.testing.assert_array_equal(found[clear], (ref <= 1.0)[clear])
    assert found.sum() > 50  # the test exercises real containment
    np.testing.assert_allclose(dist, ref[rows, leaves], rtol=1e-10)
    assert np.all(np.diff(rows) >= 0)


def test_a_singular_trust_matrix_has_an_infinite_ball_and_is_still_exact():
    C = np.zeros((2, 3))
    T = np.stack([np.diag([1.0, 1.0, 0.0]), np.eye(3)])  # leaf 0: an infinite cylinder along x3
    X = np.array([[0.5, 0.0, 1e6], [2.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
    rows, leaves, _ = (host(a) for a in TrustRegionContainment(xp.asarray(C), xp.asarray(T)).containing(xp.asarray(X)))
    assert sorted(zip(rows.tolist(), leaves.tolist(), strict=True)) == [(0, 0), (2, 0), (2, 1)]


def test_best_containing_keeps_the_k_nearest_per_row_and_pads():
    rows = xp.asarray([0, 0, 0, 2])
    leaves = xp.asarray([5, 3, 9, 4])
    dist = xp.asarray([0.5, 0.2, 0.5, 0.9])
    idx, d = (host(a) for a in best_containing(rows, leaves, dist, 3, 2))
    np.testing.assert_array_equal(idx, [[3, 5], [0, 0], [4, 0]])  # equal distances: smaller leaf first
    np.testing.assert_array_equal(d, [[0.2, 0.5], [np.inf, np.inf], [0.9, np.inf]])


def test_queries_with_more_containing_leaves_than_a_row_holds_get_all_of_them(monkeypatch):
    import src.neighbors.containment as containment

    rng = np.random.default_rng(11)
    C, T, _ = _elongated_leaves(rng, 200, 3)
    X = C[rng.integers(0, 200, 300)]  # at the centres: every query is in its own leaf, often in more
    index = TrustRegionContainment(xp.asarray(C), xp.asarray(T))
    whole = [host(a) for a in index.containing(xp.asarray(X))]
    monkeypatch.setattr(containment, "CPU_HITS_PER_QUERY", 1)
    small = [host(a) for a in index.containing(xp.asarray(X))]
    for a, b in zip(whole, small, strict=True):
        np.testing.assert_array_equal(a, b)
    assert np.bincount(whole[0]).max() > 1  # some query has more hits than one slot


def test_the_box_tree_holds_every_leaf_once_and_nests_its_boxes():
    from src.neighbors.containment import build_box_tree

    rng = np.random.default_rng(12)
    C = rng.uniform(-1, 1, (1000, 4))
    h = rng.uniform(0.01, 0.2, (1000, 4))
    t = build_box_tree(C, C - h, C + h, 16)
    np.testing.assert_array_equal(np.sort(t["perm"]), np.arange(1000))
    for nd in range(len(t["left"])):
        ids = t["perm"][t["start"][nd] : t["end"][nd]]
        assert np.all(t["node_lo"][nd] <= (C - h)[ids].min(axis=0)) and np.all(t["node_hi"][nd] >= (C + h)[ids].max(axis=0))
        if t["left"][nd] < 0:
            assert t["end"][nd] - t["start"][nd] <= 16
