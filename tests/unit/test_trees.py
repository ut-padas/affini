"""Tests for src/trees.py: Morton IDs and the split-hyperplane forest."""

import numpy as np
import pytest

from src import xp
from src.config import ClusteringConfig
from src.trees import DEPTH_MASK, MAX_DEPTH, BinaryTreeForest, child_morton
from tests._helpers import host


def _child(m: int, side: int) -> int:
    return int(child_morton(np.array([m], dtype=np.int64), np.array([side]))[0])


def test_child_morton_records_path_and_depth():
    left, right = _child(0, 0), _child(0, 1)
    assert (left & DEPTH_MASK, right & DEPTH_MASK) == (1, 1)
    assert right >> 62 == 1 and left >> 62 == 0
    assert _child(right, 1) >> 61 == 0b11 and _child(right, 1) & DEPTH_MASK == 2


def test_child_morton_reaches_max_depth_without_overflow():
    m = 0
    for _ in range(MAX_DEPTH):
        m = _child(m, 1)
    assert m > 0 and m & DEPTH_MASK == MAX_DEPTH
    assert (m >> 6) == (1 << MAX_DEPTH) - 1  # every path bit set, none in the depth bits


def _grow_forest(rng, n_roots=4, levels=6, d=3):
    """
    Split clusters level by level, as the refinement loop does, recording each
    split in a forest (tiny capacity, so it must grow). Returns the forest and the
    expected per-root preorder of node ids (stored in W[:, 0]).
    """
    forest = BinaryTreeForest(d, capacity=1)
    children = {}  # node id -> [side-0 child id, side-1 child id]
    first_split = {}  # root -> node id of its first split
    # Queue entries: (root, parent row, side, morton, parent node id).
    queue = [(r, -1, 0, 0, None) for r in range(n_roots) if r != 2]  # root 2 is never split
    next_id = 0
    for _ in range(levels):
        split = [c for c in queue if rng.random() < 0.7]
        if not split:
            break
        k = len(split)
        ids = np.arange(next_id, next_id + k)
        next_id += k
        W = np.zeros((k, d))
        W[:, 0] = ids
        root, parent, side, morton, parent_id = (np.array(col) for col in zip(*split))
        rows = host(forest.append(
            W, rng.standard_normal(k), parent.astype(np.int64), side.astype(np.int8),
            root.astype(np.int64), morton.astype(np.int64),
        ))  # fmt: skip
        np.testing.assert_array_equal(rows, ids)  # arrival order: rows are the node ids
        for i, nid in enumerate(ids):
            if parent_id[i] is None:
                first_split[int(root[i])] = int(nid)
            else:
                children.setdefault(int(parent_id[i]), [None, None])[int(side[i])] = int(nid)
        queue = [
            (int(root[i]), int(rows[i]), s, _child(int(morton[i]), s), int(ids[i]))
            for i in range(k)
            for s in (0, 1)
        ]

    def preorder(nid):
        yield nid
        for c in children.get(nid, [None, None]):
            if c is not None:
                yield from preorder(c)

    expected = {r: list(preorder(first_split[r])) for r in sorted(first_split)}
    return forest, expected


def test_sort_gives_contiguous_trees_in_preorder():
    forest, expected = _grow_forest(np.random.default_rng(0))
    parent_before, ids_before = host(forest.parent).copy(), host(forest.W[:, 0]).copy()
    new_row = host(forest.sort())

    np.testing.assert_array_equal(host(forest.W[:, 0]), np.concatenate(list(expected.values())))
    np.testing.assert_array_equal(forest.roots, list(expected))
    np.testing.assert_array_equal(np.diff(forest.tree_starts), [len(v) for v in expected.values()])
    # new_row maps old rows to new ones, and parents were remapped with it.
    np.testing.assert_array_equal(host(forest.W[:, 0])[new_row], ids_before)
    np.testing.assert_array_equal(
        host(forest.parent)[new_row], np.where(parent_before >= 0, new_row[parent_before], -1)
    )


def test_single_trees_have_local_parents_consistent_with_morton_ids():
    forest, expected = _grow_forest(np.random.default_rng(1))
    forest.sort()
    for r, ids in expected.items():
        tree = forest.tree(r)
        assert tree.n_nodes == len(ids)
        np.testing.assert_array_equal(host(tree.W[:, 0]), ids)
        parent, side, morton = host(tree.parent), host(tree.side), host(tree.morton)
        assert parent[0] == -1 and morton[0] == 0 and np.all(parent[1:] >= 0)
        assert np.all(parent[1:] < np.arange(1, len(ids)))  # preorder: parents come first
        np.testing.assert_array_equal(
            child_morton(morton[parent[1:]], side[1:]), morton[1:]
        )
    assert forest.tree(2).n_nodes == 0  # a root that was accepted without splitting


def test_frozen_and_unsorted_states_are_enforced():
    forest = BinaryTreeForest(2)
    with pytest.raises(RuntimeError, match="sort"):
        forest.tree(0)
    with pytest.raises(ValueError, match="Expected W"):
        forest.append(np.zeros((2, 3)), np.zeros(2), *(np.zeros(2, dtype=np.int64),) * 4)
    forest.sort()  # an empty forest sorts to no trees
    assert forest.tree(0).n_nodes == 0
    with pytest.raises(RuntimeError, match="frozen"):
        forest.append(np.zeros((1, 2)), np.zeros(1), *(np.zeros(1, dtype=np.int64),) * 4)


def test_forest_lives_on_the_active_backend():
    forest = BinaryTreeForest(2, capacity=1)
    forest.append(np.ones((3, 2)), np.zeros(3), *(np.full(3, -1, dtype=np.int64),) * 2,
                  np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64))  # fmt: skip
    assert isinstance(forest.W, type(xp.empty(0)))


def test_tree_tracking_limits_depth_to_what_morton_ids_hold():
    with pytest.raises(ValueError, match="max_levels"):
        ClusteringConfig(atol=0.1, track_trees=True, use_work_stealing=False, max_levels=MAX_DEPTH)
    ClusteringConfig(atol=0.1, track_trees=True, use_work_stealing=False, max_levels=MAX_DEPTH - 1)
