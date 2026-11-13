"""Both algorithm plans satisfy the contract the refinement loop relies on (src/core/plan.py)."""

import inspect

import numpy as np
import pytest

from src.afinni import AFINNIPlan
from src.art import ARTPlan, Domain, Roots
from src.config import ARTSettings, TrainingConfig
from src.core.clusters import ClusterBatch
from src.core.plan import Plan


def _plans():
    afinni = AFINNIPlan(TrainingConfig(atol=0.1), 2, 1, rank=0)
    cfg = TrainingConfig(atol=0.1, algorithm="art", domain=([0.0, 0.0], [1.0, 1.0]), art=ARTSettings(sample_multiplier=10))
    art = ARTPlan(cfg, 2, 1, Domain(np.zeros(2), np.ones(2)), Roots.build("box", 2, 1))
    return {"afinni": afinni, "art": art}


@pytest.mark.parametrize("name", ["afinni", "art"])
def test_plan_provides_every_member_of_the_contract(name):
    plan = _plans()[name]
    for member in Plan.__annotations__:
        assert hasattr(plan, member), member
    for member, spec in vars(Plan).items():
        if callable(spec) and not member.startswith("_"):
            assert callable(getattr(plan, member, None)), member
            expected = list(inspect.signature(spec).parameters)[1:]
            assert list(inspect.signature(getattr(plan, member)).parameters)[: len(expected)] == expected, member


@pytest.mark.parametrize("name", ["afinni", "art"])
def test_batch_ends_cover_the_queue_once_in_order_and_are_never_empty(name):
    plan = _plans()[name]
    sizes = np.array([5, 40, 3, 3, 120, 7, 7, 7, 1])
    cumsum = np.cumsum(plan.work_sizes(sizes))
    start, index, ends = 0, 0, []
    while start < len(sizes):
        end = plan.batch_end(cumsum, start, index, 50)
        assert start < end <= len(sizes)
        ends.append(end)
        start, index = end, index + 1
    assert ends[-1] == len(sizes)


def test_afinni_batch_grid_admits_the_crossing_cluster_and_reanchors_behind_the_grid():
    """AFINNI's rule: a batch ends after the cluster that crosses (index + 1) * batch_size."""
    plan = AFINNIPlan(TrainingConfig(atol=0.1), 2, 1, rank=0)
    cumsum = np.cumsum([30, 30, 200, 10, 10])  # 30, 60, 260, 270, 280
    assert plan.batch_end(cumsum, 0, 0, 50) == 2  # 60 crosses 50: the crossing cluster joins
    assert plan.batch_end(cumsum, 2, 1, 50) == 3  # the oversized cluster crosses 100 alone
    # Grid line 150 is already behind cluster 3: it goes alone and the grid moves to 270 - 150.
    assert plan.batch_end(cumsum, 3, 2, 50) == 4
    assert plan.batch_end(cumsum, 4, 3, 50) == 5
    assert plan.batch_end(cumsum, 0, 0, 50) == 2  # a new level's first batch resets the anchor


def test_art_batches_hold_whole_nodes_within_the_batch_size():
    plan = _plans()["art"]
    queue = ClusterBatch(np.zeros((0, 2)), np.zeros((0, 1)), None, np.zeros(4, dtype=np.int64))
    work = plan.work_sizes(queue.sizes)
    cumsum = np.cumsum(work)
    end = plan.batch_end(cumsum, 0, 0, int(2 * work[0]))
    assert end == 2 and cumsum[end - 1] <= 2 * work[0]


def test_afinni_batch_releases_its_points_before_splitting():
    """Augmentation and splitting hold a batch's peak memory, so the batch's own points must be gone by then."""
    import contextlib
    import dataclasses
    import weakref

    from src import xp
    from src.core.clusters import ClusterQueueBuilder, ColumnBuffer
    from src.core.evaluation import Evaluator
    from src.core.plan import DeviceBatch

    config = TrainingConfig(atol=1e-4, rtol=1e-4, min_pts_per_cluster=8)
    plan = AFINNIPlan(config, 2, 1, rank=0)
    sizes = np.full(4, 30, dtype=np.int64)
    X = xp.asarray(np.random.default_rng(0).uniform(-1, 1, (int(sizes.sum()), 2)))
    alive = weakref.ref(X)
    real = config.components().split_strategy
    seen = []

    class Spy:
        name = real.name

        def split(self, data):
            seen.append(alive() is None)
            return real.split(data)

    components = dataclasses.replace(config.components(), split_strategy=Spy())
    evaluator = Evaluator(f_fn=lambda x: xp.sin(4 * x[:, :1]) * xp.cos(3 * x[:, 1:]))
    batch = DeviceBatch(
        X=X, F=evaluator(X)[0], G=None, sizes=xp.asarray(sizes), starts=xp.asarray(np.r_[0, np.cumsum(sizes)[:-1]]),
        queue=ClusterBatch(np.zeros((0, 2)), np.zeros((0, 1)), None, sizes), cut=slice(0, 4), level=0, index=0,
        n_batches=1, progress=lambda: None,
    )  # fmt: skip
    del X
    stats = plan.process_batch(
        batch, components=components, evaluator=evaluator,
        accepted=ColumnBuffer(plan.leaf_columns(components.local_model), capacity=8), splits=None,
        next_level=ClusterQueueBuilder(2, 1, has_g=False, capacity_pts=1024, capacity_clusters=16),
        timer=lambda key: contextlib.nullcontext(),
    )  # fmt: skip
    assert stats["rejected"] == 4 and seen == [True]
