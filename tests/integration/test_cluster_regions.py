"""Training with a sampler that needs regions: the domain box cut by each cluster's ancestor hyperplanes."""

import numpy as np
import pytest

from src import xp
from src.config import ClusteringConfig
from src.modes import SurrogateMode
from src.sampling import ConvexCombinationSampler
from src.utils import repeats
from tests._helpers import assert_backend_equal, host, train_surrogate, tree_problems

rng = np.random.default_rng(3)
X_TRAIN = rng.uniform(-1, 1, (3_000, 2))


def f(x):
    return (xp.sin(3 * x[:, 0]) * xp.cos(2 * x[:, 1]))[:, None]


class RegionCheckingSampler(ConvexCombinationSampler):
    """
    Convex combinations, which never leave a convex region, that record whether every
    existing and new point lies in its cluster's region. Not registered: tests pass
    an instance.
    """

    name = "region_checking"
    needs_regions = True

    def __init__(self):
        self.calls = []  # (planes per cluster, depths, every point inside, domain)

    def sample(self, clusters, n_new, rng):
        new = super().sample(clusters, n_new, rng)
        regions = clusters.regions
        old_inside = regions.contains(clusters.X, repeats(xp.arange(clusters.n_clusters), clusters.counts))
        new_inside = regions.contains(new, repeats(xp.arange(clusters.n_clusters), n_new))
        self.calls.append((
            int(regions.planes.shape[1]), host(regions.depth),
            bool(xp.all(old_inside)) and bool(xp.all(new_inside)), host(regions.domain),
        ))  # fmt: skip
        return new


def train(**kw):
    opts = {
        "f_fn": f, "mode": SurrogateMode.AFFINE_COV_HESS, "atol": 0.02, "rtol": 0.02, "num_initial_clusters": 1,
        "min_pts_per_cluster": 40, "use_work_stealing": False, "track_trees": True,
    }  # fmt: skip
    return train_surrogate(X_TRAIN, **{**opts, **kw})


def test_every_point_lies_in_its_regions_and_results_are_unchanged():
    sampler = RegionCheckingSampler()
    s = train(sampler=sampler)
    assert s.converged and tree_problems(s) == []
    assert sampler.calls and all(inside for _, _, inside, _ in sampler.calls)
    # Without work stealing a batch holds one depth, so no planes are padding.
    assert all(np.all(depths == D) for D, depths, _, _ in sampler.calls)
    assert max(D for D, _, _, _ in sampler.calls) >= 4
    bbox = np.stack([X_TRAIN.min(axis=0), X_TRAIN.max(axis=0)])
    assert all(np.array_equal(domain, bbox) for *_, domain in sampler.calls)

    # Handing the sampler regions changes nothing else.
    plain = train(track_trees=False)
    assert s.N_src == plain.N_src
    assert s.profiling_stats["total_f_evals"] == plain.profiling_stats["total_f_evals"]
    for name in ("X_src", "f_src", "Qs_src"):
        assert_backend_equal(getattr(s, name), getattr(plain, name), separate_runs=True)


def test_a_given_domain_reaches_the_sampler():
    sampler = RegionCheckingSampler()
    train(sampler=sampler, domain=([-2, -1], [1, 3]))
    np.testing.assert_array_equal(sampler.calls[0][3], [[-2.0, -1.0], [1.0, 3.0]])


@pytest.mark.parametrize(
    ("options", "labels", "match"),
    [
        ({"track_trees": False}, None, "track_trees=True"),
        ({"num_initial_clusters": 2}, None, "num_initial_clusters=1"),
        ({"domain": ([-0.5, -1], [1, 1])}, None, "outside the domain"),
        ({"domain": ([-1, -1, -1], [1, 1, 1])}, None, "dimension 3"),
        ({}, np.arange(len(X_TRAIN)) % 2, "single label"),
    ],
)
def test_regions_need_trees_one_root_and_a_domain_around_the_points(options, labels, match):
    with pytest.raises(ValueError, match=match):
        train_surrogate(
            X_TRAIN, initial_labels=labels,
            **{"f_fn": f, "atol": 0.02, "num_initial_clusters": 1, "use_work_stealing": False,
               "track_trees": True, "sampler": RegionCheckingSampler(), **options},
        )  # fmt: skip


def test_the_data_domain_must_not_be_flat():
    X_flat = np.c_[X_TRAIN[:, 0], np.zeros(len(X_TRAIN))]
    with pytest.raises(ValueError, match="flat"):
        train_surrogate(
            X_flat, f_fn=f, atol=0.02, num_initial_clusters=1, use_work_stealing=False,
            track_trees=True, sampler=RegionCheckingSampler(),
        )  # fmt: skip


def test_domain_is_normalized_and_validated():
    config = ClusteringConfig(atol=0.1, domain=(np.array([0, 0]), [1, 2.5]))
    assert config.domain == ((0.0, 0.0), (1.0, 2.5))
    for bad in (([0, 0], [1]), ([0, 1], [1, 1]), ([], []), (1, 2), ([0], [1], [2])):
        with pytest.raises(ValueError, match="domain"):
            ClusteringConfig(atol=0.1, domain=bad)
