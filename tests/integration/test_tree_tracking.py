"""Training with ``track_trees``: the recorded split forest and the leaves' place in it."""

import numpy as np
import pytest

from src import xp
from src.modes import SurrogateMode
from src.surrogate import AFINNISurrogate
from tests._helpers import assert_backend_equal, host, train_surrogate, tree_problems

rng = np.random.default_rng(3)
X_TRAIN = rng.uniform(-1, 1, (3_000, 2))
N_ROOTS = 6


def f(x):
    return (xp.sin(3 * x[:, 0]) * xp.cos(2 * x[:, 1]))[:, None]


def train(**kw):
    opts = dict(
        f_fn=f, mode=SurrogateMode.AFFINE_COV_HESS, atol=0.02, rtol=0.02,
        num_initial_clusters=N_ROOTS, min_pts_per_cluster=40, use_work_stealing=False,
    )  # fmt: skip
    return train_surrogate(X_TRAIN, **{**opts, **kw})


@pytest.fixture(scope="module")
def tracked():
    return train(track_trees=True)


def test_tracking_does_not_change_the_leaves(tracked):
    plain = train()
    assert tracked.N_src == plain.N_src
    assert tracked.profiling_stats["total_f_evals"] == plain.profiling_stats["total_f_evals"]
    for name in ("X_src", "f_src", "Qs_src"):
        assert_backend_equal(getattr(tracked, name), getattr(plain, name), separate_runs=True)
    assert plain.tree_forest is None


@pytest.mark.parametrize("cut", ["median", "sign"])
@pytest.mark.parametrize("mode", [SurrogateMode.AFFINE_COV_HESS, SurrogateMode.NEAREST_NEIGHBOR])
def test_forest_and_leaves_are_consistent(mode, cut):
    split = {"split_strategy": ("covariance", {"cut": cut})} if mode == SurrogateMode.NEAREST_NEIGHBOR else {}
    s = train(mode=mode, track_trees=True, **split)
    assert s.converged and tree_problems(s) == []
    forest = s.tree_forest
    # A converged binary tree has one more leaf than splits, and every leaf is accepted.
    assert forest.n_leaves == s.N_src == forest.n_internal + N_ROOTS
    assert host(forest.root_centroids).shape == (N_ROOTS, 2)


def test_leaf_roots_match_their_initial_clusters(tracked):
    """Each leaf's centroid is nearer its own root's centroid than most others (k-means cells)."""
    leaves = tracked.tree_forest.leaf_nodes
    roots = host(leaves["root"])
    assert set(roots) == set(range(N_ROOTS))
    c, rc = host(tracked.X_src)[host(leaves["leaf"])], host(tracked.tree_forest.root_centroids)
    nearest = np.argmin(((c[:, None, :] - rc[None]) ** 2).sum(-1), axis=1)
    assert np.mean(nearest == roots) > 0.9


def test_unresolved_clusters_become_leaf_nodes_without_leaf_data():
    with pytest.warns(RuntimeWarning, match="max_levels"):
        s = train(track_trees=True, max_levels=4)
    assert not s.converged and s.N_src > 0 and tree_problems(s) == []
    leaves = s.tree_forest.leaf_nodes
    assert int((host(leaves["leaf"]) < 0).sum()) == s.num_unresolved_clusters
    assert int(host(s.tree_forest.internal_nodes["morton"] & 63).max()) <= 4


def test_sorted_trees_are_preorder_blocks(tracked):
    forest = tracked.tree_forest
    assert forest.is_sorted and list(forest.roots) == list(range(N_ROOTS))
    n_internal = n_leaves = 0
    for r in forest.roots:
        tree = forest.tree(int(r))
        for table in (tree.internal_nodes, tree.leaf_nodes):
            assert np.all(host(table["root"]) == r)
            assert np.all(np.diff(host(table["morton"])) > 0)  # unique and ascending: preorder
        assert tree.n_leaves == tree.n_internal + 1
        n_internal, n_leaves = n_internal + tree.n_internal, n_leaves + tree.n_leaves
    assert (n_internal, n_leaves) == (forest.n_internal, forest.n_leaves)


def test_retraining_without_tracking_clears_the_trees(tracked):
    s = AFINNISurrogate()
    train(surrogate=s, track_trees=True)
    assert s.tree_forest is not None
    train(surrogate=s)
    assert s.tree_forest is None


def test_forest_plot_renders_every_node_edge_and_trust_region(tracked, tmp_path):
    from matplotlib.collections import EllipseCollection, LineCollection, PathCollection

    from src.plotting import plot_binary_tree_forest

    path = tmp_path / "trees.png"
    fig = plot_binary_tree_forest(tracked, (-1.0, 1.0), f, path=path)
    assert path.exists()
    ax_tree, ax_trust = fig.axes[:2]
    internal_parent = host(tracked.tree_forest.internal_nodes["parent"])
    leaf_parent = host(tracked.tree_forest.leaf_nodes["parent"])
    n_mid = int((internal_parent >= 0).sum())  # internal nodes below the roots
    (edges,) = [c for c in ax_tree.collections if isinstance(c, LineCollection)]
    # One edge from every node below a root to its parent (converged: every leaf accepted).
    assert len(edges.get_segments()) == n_mid + int((leaf_parent >= 0).sum())
    points = [len(c.get_offsets()) for c in ax_tree.collections if isinstance(c, PathCollection)]
    assert points == [n_mid, tracked.N_src, N_ROOTS]  # internal nodes, leaves, roots
    (ellipses,) = [c for c in ax_trust.collections if isinstance(c, EllipseCollection)]
    assert len(ellipses.get_offsets()) == tracked.N_src
    assert [len(c.get_offsets()) for c in ax_trust.collections if isinstance(c, PathCollection)] == [tracked.N_src]


def test_forest_plot_needs_tracked_trees():
    from src.plotting import plot_binary_tree_forest

    with pytest.raises(ValueError, match="track_trees=True"):
        plot_binary_tree_forest(train(), (-1.0, 1.0), f)


def test_forest_plot_refuses_multi_output_functions():
    from src.plotting import plot_binary_tree_forest

    def two_outputs(x):
        return xp.concatenate([f(x), 2 * f(x)], axis=1)

    s = train(f_fn=two_outputs, track_trees=True)
    with pytest.raises(ValueError, match="single-output function only; this surrogate has 2 outputs"):
        plot_binary_tree_forest(s, (-1.0, 1.0), two_outputs)
