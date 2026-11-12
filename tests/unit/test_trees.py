"""Tests for src/trees.py: Morton IDs and the binary tree forest."""

import numpy as np
import pytest

from src import xp
from src.config import ClusteringConfig
from src.trees import DEPTH_MASK, MAX_DEPTH, BinaryTreeForest, child_morton, side_of
from tests._helpers import host


def _child(m: int, side: int) -> int:
    return int(child_morton(np.array([m], dtype=np.int64), np.array([side]))[0])


def test_child_morton_records_path_and_depth():
    left, right = _child(0, 0), _child(0, 1)
    assert (left & DEPTH_MASK, right & DEPTH_MASK) == (1, 1)
    assert right >> 62 == 1 and left >> 62 == 0
    assert _child(right, 1) >> 61 == 0b11 and _child(right, 1) & DEPTH_MASK == 2


def test_side_is_the_last_path_bit():
    m = np.array([0, _child(0, 0), _child(0, 1), _child(_child(0, 1), 0), _child(_child(0, 0), 1)])
    np.testing.assert_array_equal(side_of(m), [0, 0, 1, 0, 1])


def test_child_morton_reaches_max_depth_without_overflow():
    m = 0
    for _ in range(MAX_DEPTH):
        m = _child(m, 1)
    assert m > 0 and m & DEPTH_MASK == MAX_DEPTH and int(side_of(np.array([m]))[0]) == 1
    assert (m >> 6) == (1 << MAX_DEPTH) - 1  # every path bit set, none in the depth bits


def _grow_forest(rng, n_roots=4, levels=6, d=3):
    """
    Split clusters level by level, as the refinement loop does: each cluster is
    split (an internal node), accepted or, at the last level, left unresolved (a
    leaf node). Tiny capacity, so the tables must grow. Returns the forest and the
    expected per-root preorder of node ids, stored in ``W[:, 0]`` and ``leaf``.
    """
    forest = BinaryTreeForest(d, capacity=1)
    children = {}  # node id -> [side-0 child id, side-1 child id]
    first = {}  # root -> id of its top node
    queue = [(r, -1, 0, None) for r in range(n_roots)]  # (root, parent row, morton, parent id)
    next_id = 0
    for level in range(levels + 1):
        if not queue:
            break
        split = rng.random(len(queue)) < (0.7 if level < levels else 0.0)
        ids = np.arange(next_id, next_id + len(queue))
        next_id += len(queue)
        root, parent, morton = (np.array([c[i] for c in queue], dtype=np.int64) for i in range(3))
        for i, nid in enumerate(ids):
            pid = queue[i][3]
            if pid is None:
                first[int(root[i])] = int(nid)
            else:
                children.setdefault(pid, [None, None])[int(side_of(morton[i : i + 1])[0])] = int(nid)
        if (~split).any():
            forest.add_leaves(parent[~split], root[~split], morton[~split], ids[~split])
        k = int(split.sum())
        if not k:
            break
        W = np.zeros((k, d))
        W[:, 0] = ids[split]
        rows = host(forest.add_splits(
            W, rng.standard_normal(k), rng.standard_normal((k, d)), parent[split], root[split], morton[split]
        ))  # fmt: skip
        queue = [
            (int(root[split][i]), int(rows[i]), _child(int(morton[split][i]), s), int(ids[split][i]))
            for i in range(k)
            for s in (0, 1)
        ]

    def preorder(nid):
        yield nid
        for c in children.get(nid, [None, None]):
            if c is not None:
                yield from preorder(c)

    return forest, {r: list(preorder(first[r])) for r in sorted(first)}


def _ids(table):
    return host(table["W"][:, 0]).astype(int) if "W" in table else host(table["leaf"])


def test_sort_gives_contiguous_trees_in_preorder():
    forest, expected = _grow_forest(np.random.default_rng(0))
    forest.sort()
    for r, order in expected.items():
        tree = forest.tree(r)
        internal, leaves = set(_ids(tree.internal_nodes)), set(_ids(tree.leaf_nodes))
        assert tree.n_leaves == tree.n_internal + 1 and internal | leaves == set(order)
        # Each table lists its nodes in the tree's preorder.
        np.testing.assert_array_equal(_ids(tree.internal_nodes), [i for i in order if i in internal])
        np.testing.assert_array_equal(_ids(tree.leaf_nodes), [i for i in order if i in leaves])
    assert list(forest.roots) == list(expected)


def test_parents_are_remapped_to_sorted_rows():
    forest, _ = _grow_forest(np.random.default_rng(1))
    before = {k: host(v).copy() for k, v in forest.internal_nodes.items()}
    leaves_before = {k: host(v).copy() for k, v in forest.leaf_nodes.items()}
    forest.sort()
    after, leaves_after = forest.internal_nodes, forest.leaf_nodes
    # Compare by node id: each node's parent must still be the same node.
    pid = lambda W, p: np.where(p >= 0, W[np.maximum(p, 0), 0], -1)  # noqa: E731
    by_id = lambda ids, vals: dict(zip(ids.tolist(), vals.tolist()))  # noqa: E731
    assert by_id(before["W"][:, 0], pid(before["W"], before["parent"])) == by_id(
        host(after["W"][:, 0]), pid(host(after["W"]), host(after["parent"]))
    )
    assert by_id(leaves_before["leaf"], pid(before["W"], leaves_before["parent"])) == by_id(
        host(leaves_after["leaf"]), pid(host(after["W"]), host(leaves_after["parent"]))
    )


def test_single_trees_have_local_parents_consistent_with_morton_ids():
    forest, expected = _grow_forest(np.random.default_rng(2))
    forest.sort()
    for r in expected:
        tree = forest.tree(r)
        internal = {k: host(v) for k, v in tree.internal_nodes.items()}
        for table in (internal, {k: host(v) for k, v in tree.leaf_nodes.items()}):
            has = table["parent"] >= 0
            np.testing.assert_array_equal(
                child_morton(internal["morton"][table["parent"][has]], side_of(table["morton"][has])),
                table["morton"][has],
            )
        assert np.all(internal["parent"][1:] < np.arange(1, tree.n_internal))  # parents come first
    assert forest.tree(10**6).n_leaves == 0  # a root this forest does not have


def test_frozen_and_unsorted_states_are_enforced():
    forest = BinaryTreeForest(2)
    with pytest.raises(RuntimeError, match="sort"):
        forest.tree(0)
    with pytest.raises(ValueError, match="shape"):
        forest.add_splits(np.zeros((2, 3)), np.zeros(2), np.zeros((2, 2)), *(np.zeros(2, dtype=np.int64),) * 3)
    forest.add_leaves(*(np.zeros(1, dtype=np.int64),) * 4)  # a root accepted without splitting
    forest.sort()
    assert forest.tree(0).n_leaves == 1 and forest.tree(0).n_internal == 0
    with pytest.raises(RuntimeError, match="frozen"):
        forest.add_leaves(*(np.zeros(1, dtype=np.int64),) * 4)


def test_host_round_trip_keeps_the_sorted_forest():
    forest, _ = _grow_forest(np.random.default_rng(3), d=2)
    forest.root_centroids = xp.asarray(np.arange(8.0).reshape(4, 2))
    forest.sort()
    again = BinaryTreeForest.sorted_from(forest.to_host(), d=2)
    for kind in ("internal_nodes", "leaf_nodes"):
        for k, v in getattr(forest, kind).items():
            np.testing.assert_array_equal(host(getattr(again, kind)[k]), host(v))
    np.testing.assert_array_equal(host(again.root_centroids), host(forest.root_centroids))
    assert isinstance(again.internal_nodes["W"], type(xp.empty(0)))  # on the active backend


def test_tree_tracking_limits_depth_to_what_morton_ids_hold():
    with pytest.raises(ValueError, match="max_levels"):
        ClusteringConfig(atol=0.1, track_trees=True, use_work_stealing=False, max_levels=MAX_DEPTH)
    ClusteringConfig(atol=0.1, track_trees=True, use_work_stealing=False, max_levels=MAX_DEPTH - 1)


def test_find_rows_locates_every_node_and_only_those():
    forest, expected = _grow_forest(np.random.default_rng(4))
    forest.sort()
    for kind in ("internal_nodes", "leaf_nodes"):
        table = getattr(forest, kind)
        n = len(host(table["root"]))
        np.testing.assert_array_equal(host(forest.find_rows(kind, table["root"], table["morton"])), np.arange(n))
    leaves = forest.leaf_nodes
    # A leaf's Morton ID is not an internal node, and roots this forest lacks find nothing.
    assert np.all(host(forest.find_rows("internal_nodes", leaves["root"], leaves["morton"])) == -1)
    missing = xp.asarray(np.array([10**6, -1], dtype=np.int64))
    assert np.all(host(forest.find_rows("leaf_nodes", missing, xp.zeros(2, dtype=xp.int64))) == -1)
