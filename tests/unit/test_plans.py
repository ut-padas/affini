"""Both algorithm plans satisfy the contract the refinement loop relies on (src/core/plan.py)."""

import inspect

import numpy as np
import pytest

from src.config import ClusteringConfig, RegionSettings
from src.core.clusters import ClusterBatch
from src.core.plan import Plan
from src.engine.afinni_plan import AfinniPlan
from src.region import Domain, RegionPlan, Roots


def _plans():
    afinni = AfinniPlan(ClusteringConfig(atol=0.1), 2, 1, rank=0)
    cfg = ClusteringConfig(atol=0.1, algorithm="region", domain=([0.0, 0.0], [1.0, 1.0]), region=RegionSettings(sample_multiplier=10))
    region = RegionPlan(cfg, 2, 1, Domain(np.zeros(2), np.ones(2)), Roots.build("box", 2, 1))
    return {"afinni": afinni, "region": region}


@pytest.mark.parametrize("name", ["afinni", "region"])
def test_plan_provides_every_member_of_the_contract(name):
    plan = _plans()[name]
    for member in Plan.__annotations__:
        assert hasattr(plan, member), member
    for member, spec in vars(Plan).items():
        if callable(spec) and not member.startswith("_"):
            assert callable(getattr(plan, member, None)), member
            expected = list(inspect.signature(spec).parameters)[1:]
            assert list(inspect.signature(getattr(plan, member)).parameters)[: len(expected)] == expected, member


@pytest.mark.parametrize("name", ["afinni", "region"])
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
    """AFINNI's original rule: a batch ends after the cluster that crosses (index + 1) * batch_size."""
    plan = AfinniPlan(ClusteringConfig(atol=0.1), 2, 1, rank=0)
    cumsum = np.cumsum([30, 30, 200, 10, 10])  # 30, 60, 260, 270, 280
    assert plan.batch_end(cumsum, 0, 0, 50) == 2  # 60 crosses 50: the crossing cluster joins
    assert plan.batch_end(cumsum, 2, 1, 50) == 3  # the oversized cluster crosses 100 alone
    # Grid line 150 is already behind cluster 3: it goes alone and the grid moves to 270 - 150.
    assert plan.batch_end(cumsum, 3, 2, 50) == 4
    assert plan.batch_end(cumsum, 4, 3, 50) == 5
    assert plan.batch_end(cumsum, 0, 0, 50) == 2  # a new level's first batch resets the anchor


def test_region_batches_hold_whole_nodes_within_the_batch_size():
    plan = _plans()["region"]
    queue = ClusterBatch(np.zeros((0, 2)), np.zeros((0, 1)), None, np.zeros(4, dtype=np.int64))
    work = plan.work_sizes(queue.sizes)
    cumsum = np.cumsum(work)
    end = plan.batch_end(cumsum, 0, 0, int(2 * work[0]))
    assert end == 2 and cumsum[end - 1] <= 2 * work[0]
