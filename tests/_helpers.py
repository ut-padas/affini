"""Helpers shared by tests: backend-aware comparisons (NumPy or CuPy) and a training shorthand."""

import numpy as np

from src import IS_CUPY_BACKEND, _to_host, xp
from src.config import ClusteringConfig
from src.evaluation import Evaluator
from src.surrogate import AFINNISurrogate

# CuPy reductions/einsum may order floating-point operations differently from
# NumPy (and from one code path to another), so results agree to rounding only.
GPU_RTOL = 1e-12
# Two separate GPU trainings: atomicAdd order differs per run and the differences
# pass through the least-squares solves, reaching a few 1e-12 on small values.
GPU_RUNS_RTOL = 1e-10


def host(a):
    """``a`` as a NumPy array, whatever the backend."""
    return np.asarray(_to_host(a))


def assert_backend_equal(
    actual, desired, rtol: float = GPU_RTOL, atol: float = 0.0, separate_runs: bool = False
):
    """
    Exact equality on the NumPy backend; equality to ``rtol`` (relative) and
    ``atol`` on CuPy, where operation order can differ. ``separate_runs`` compares
    results of two trainings, with ``GPU_RUNS_RTOL`` and an absolute floor at 1e-12
    of the largest value (for entries near zero).
    """
    actual, desired = host(actual), host(desired)
    if IS_CUPY_BACKEND and separate_runs:
        scale = float(np.max(np.abs(desired))) if desired.size else 0.0
        rtol, atol = max(rtol, GPU_RUNS_RTOL), max(atol, 1e-12 * scale)
    if IS_CUPY_BACKEND:
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
    else:
        np.testing.assert_array_equal(actual, desired)


EVALUATOR_FIELDS = ("f_fn", "g_fn", "oracle_fn", "hessian_fn")


def train_surrogate(X, surrogate=None, *, Y_train=None, initial_labels=None, **options):
    """
    Train ``surrogate`` (a new ``AFINNISurrogate`` by default) and return it. The
    ``Evaluator`` fields among ``options`` build the evaluator and the rest the
    ``ClusteringConfig``, so tests can keep one dict of options per case.
    """
    evaluator = Evaluator(**{k: options.pop(k) for k in EVALUATOR_FIELDS if k in options})
    surrogate = AFINNISurrogate() if surrogate is None else surrogate
    surrogate.train(xp.asarray(X), ClusteringConfig(**options), evaluator, Y_train=Y_train, initial_labels=initial_labels)
    return surrogate


def tree_problems(surrogate) -> list[str]:
    """
    Inconsistencies in a tree-tracking surrogate's forest (empty if none). Every
    node must agree with its parent's root and Morton ID and come after its parent
    (preorder), every child slot (parent, side) must hold exactly one node, the
    accepted leaf nodes must index each trained leaf once, and every node's centroid
    must lie on its side of each ancestor's hyperplane.
    """
    from src.trees import child_morton, side_of

    forest = surrogate.tree_forest
    if forest is None or not forest.is_sorted:
        return ["no sorted forest"]
    internal = {k: host(v) for k, v in forest.internal_nodes.items()}
    leaves = {k: host(v) for k, v in forest.leaf_nodes.items()}
    n = len(internal["b"])
    problems = []
    accepted = leaves["leaf"][leaves["leaf"] >= 0]
    if sorted(accepted.tolist()) != list(range(surrogate.N_src)):
        problems.append("accepted leaf nodes do not index every trained leaf exactly once")
    if host(forest.root_centroids).shape[0] <= max(internal["root"].max(initial=-1), leaves["root"].max(initial=-1)):
        problems.append("a root has no centroid")
    for what, table, rows in (("internal", internal, np.arange(n)), ("leaf", leaves, None)):
        p = table["parent"]
        has = p >= 0
        if np.any(p >= n) or (rows is not None and np.any(p[has] >= rows[has])):
            problems.append(f"{what} parents out of range or after their children")
            continue
        if np.any(table["morton"][~has] != 0):
            problems.append(f"{what} nodes without a parent have a non-root Morton ID")
        if np.any(table["root"][has] != internal["root"][p[has]]):
            problems.append(f"{what} nodes and their parents disagree on the root")
        m = table["morton"][has]
        if np.any(m != child_morton(internal["morton"][p[has]], side_of(m))):
            problems.append(f"{what} Morton IDs do not extend their parents'")
    slots = np.concatenate([
        2 * t["parent"][t["parent"] >= 0] + side_of(t["morton"][t["parent"] >= 0]) for t in (internal, leaves)
    ])  # fmt: skip
    if len(np.unique(slots)) != len(slots) or len(slots) != 2 * n:
        problems.append(f"{2 * n} child slots but {len(slots)} children ({len(np.unique(slots))} distinct)")

    # Walk every accepted leaf and every internal node up to its root, checking that
    # its centroid lies on its side of each ancestor hyperplane.
    keep = leaves["leaf"] >= 0
    c = np.concatenate([host(surrogate.X_src)[leaves["leaf"][keep]], internal["centroid"]])
    p = np.concatenate([leaves["parent"][keep], internal["parent"]])
    m = np.concatenate([leaves["morton"][keep], internal["morton"]])
    while np.any(p >= 0):
        on = p >= 0
        dx = c[on] - internal["centroid"][p[on]]
        value = np.einsum("kd,kd->k", internal["W"][p[on]], dx) + internal["b"][p[on]]
        if np.any((value > 0) != (side_of(m[on]) == 1)):
            problems.append("a centroid lies on the wrong side of an ancestor hyperplane")
            break
        m[on], p[on] = internal["morton"][p[on]], internal["parent"][p[on]]
    return problems
