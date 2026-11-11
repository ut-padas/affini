"""Tests for src/region.py: random streams, roots, polytope sampling, cuts and routing."""

from collections import defaultdict

import numpy as np
import pytest

import src.work_stealing as ws
from src import _to_host, xp
from src.clusters import ClusterBatch, ClusterQueueBuilder
from src.region import (
    PURPOSE_TRAIN,
    PURPOSE_VALIDATION,
    Domain,
    HitAndRunSampler,
    RegionPoints,
    Roots,
    RoutingTable,
    counter_uniform,
    gap_cuts,
    node_keys,
    plane_scores,
)
from src.sampling import ClusterPoints
from tests._helpers import host


def test_counter_uniform_is_deterministic_open_and_stream_separated():
    keys = node_keys(3, xp.asarray([0, 0, 1]), xp.asarray([1, 2, 1]))
    assert len(set(_to_host(keys).tolist())) == 3
    counter = xp.arange(100_000)
    u = counter_uniform(keys[0], PURPOSE_TRAIN, counter)
    np.testing.assert_array_equal(_to_host(u), _to_host(counter_uniform(keys[0], PURPOSE_TRAIN, counter)))
    assert float(u.min()) > 0 and float(u.max()) < 1
    assert abs(float(u.mean()) - 0.5) < 5e-3 and abs(float(u.var()) - 1 / 12) < 2e-3
    other = counter_uniform(keys[0], PURPOSE_VALIDATION, counter)
    assert abs(float(xp.corrcoef(u, other)[0, 1])) < 0.02


def test_grid_roots_cover_the_unit_cube_and_assign_faces_upward():
    roots = Roots.build("grid", 2, 6)
    assert roots.n_roots == 6 and int(np.prod(roots.bins)) == 6
    lo, hi = roots.bounds()
    U = xp.asarray(np.random.default_rng(0).random((1000, 2)))
    r = roots.root_of(U)
    assert bool(xp.all((lo[r] <= U) & (hi[r] >= U)))
    # A point on an interior face belongs to the upper cell; the domain's upper face to the last.
    on_face, above = xp.asarray([[0.5, 0.1], [1.0, 1.0]]), xp.asarray([[0.5 + 1e-9, 0.1], [1.0 - 1e-9, 1.0 - 1e-9]])
    np.testing.assert_array_equal(_to_host(roots.root_of(on_face)), _to_host(roots.root_of(above)))


def test_hit_and_run_stays_in_the_polytope_and_is_near_uniform():
    """Lower triangle u0 + u1 <= 1 of the unit square: mean (1/3, 1/3)."""
    domain, roots = Domain(np.zeros(2), np.ones(2)), Roots.build("box", 2, 1)
    w = np.array([1.0, 1.0]) / np.sqrt(2)
    constraints = xp.asarray(np.r_[w, 1 / np.sqrt(2)][None, None, :])
    seeds = xp.asarray([[0.2, 0.2], [0.1, 0.5]])
    geo = RegionPoints(xp.asarray([0]), xp.asarray([1]), node_keys(0, xp.asarray([0]), xp.asarray([2])), constraints, PURPOSE_TRAIN)
    X = HitAndRunSampler(domain, roots, walk_steps=30).sample(ClusterPoints(seeds, xp.asarray([2]), region=geo), xp.asarray([20_000]))
    assert bool(xp.all(X >= 0)) and bool(xp.all(X <= 1))
    assert float(xp.max(plane_scores(X, xp.asarray(w)[None, :]))) <= 1 / np.sqrt(2) + 1e-12
    np.testing.assert_allclose(_to_host(X.mean(axis=0)), [1 / 3, 1 / 3], atol=0.01)


def test_depth_zero_sampling_is_exactly_uniform_in_the_root_cell():
    domain, roots = Domain(np.array([-2.0, 0.0]), np.array([2.0, 1.0])), Roots.build("grid", 2, 2)
    lo, hi = roots.bounds()
    geo = RegionPoints(xp.asarray([1]), xp.asarray([0]), node_keys(0, xp.asarray([1]), xp.asarray([1])), xp.zeros((1, 0, 3)), PURPOSE_TRAIN)
    X = HitAndRunSampler(domain, roots, 5).sample(ClusterPoints(xp.zeros((0, 2)), xp.asarray([0]), region=geo), xp.asarray([5000]))
    U = domain.normalize(X)
    assert bool(xp.all((lo[1] <= U) & (hi[1] >= U)))


def test_gap_cuts_never_split_tied_scores():
    scores = xp.asarray([0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 5.0, 6.0])
    labels = xp.asarray([0] * 6 + [1] * 2)
    valid, pos, z = gap_cuts(scores, labels, xp.asarray([6, 2]), min_side=1)
    # The median position 3 sits inside the tie at 1.0; the nearest gaps are 1 and 4.
    assert bool(valid[0]) and int(pos[0]) in (1, 4) and float(z[0]) == float(scores[int(pos[0])])
    assert bool(valid[1]) and int(pos[1]) == 1
    valid, _, _ = gap_cuts(xp.asarray([1.0, 1.0, 1.0]), xp.asarray([0, 0, 0]), xp.asarray([3]), min_side=1)
    assert not bool(valid[0])


def _tree():
    """Root (box) split at u0 = 0.5; its right child split at u1 = 0.25."""
    domain, roots = Domain(np.zeros(2), np.ones(2)), Roots.build("box", 2, 1)
    return domain, roots, {
        "leaf_root": [0, 0, 0], "leaf_path": [2, 6, 7], "split_root": [0, 0], "split_path": [1, 3],
        "w": np.array([[1.0, 0.0], [0.0, 1.0]]), "z": np.array([0.5, 0.25]),
    }


def test_routing_follows_the_stored_predicates():
    domain, roots, tree = _tree()
    table = RoutingTable.build(domain, roots, **tree)
    leaf, outside = table.route(xp.asarray([[0.1, 0.9], [0.7, 0.1], [0.7, 0.9], [0.5, 0.25], [1.5, 0.9]]))
    np.testing.assert_array_equal(_to_host(leaf), [0, 1, 2, 2, 2])  # boundary goes right
    np.testing.assert_array_equal(_to_host(outside), [False, False, False, False, True])
    again = RoutingTable.from_arrays(table.arrays())
    np.testing.assert_array_equal(_to_host(again.route(xp.asarray([[0.7, 0.1]]))[0]), [1])


@pytest.mark.parametrize("broken", ["missing_child", "duplicate", "orphan"])
def test_routing_table_rejects_inconsistent_trees(broken):
    domain, roots, tree = _tree()
    if broken == "missing_child":
        tree["leaf_root"], tree["leaf_path"] = [0, 0], [2, 6]
    elif broken == "duplicate":
        tree["leaf_root"], tree["leaf_path"] = [0, 0, 0, 0], [2, 6, 7, 7]
    else:
        tree["leaf_root"], tree["leaf_path"] = [0, 0, 0, 0], [2, 6, 7, 9]
    with pytest.raises(ValueError):
        RoutingTable.build(domain, roots, **tree)


def test_builder_pads_constraints_and_split_select_keep_region_columns():
    builder = ClusterQueueBuilder(2, 1, False, capacity_pts=1, capacity_clusters=1, region_rows=2)
    builder.append(
        np.ones((3, 2)), np.ones((3, 1)), None, np.array([1, 2]), depth=np.array([1, 1]),
        root=np.array([0, 0]), path_code=np.array([2, 3]), constraints=np.full((2, 1, 3), 7.0),
    )  # fmt: skip
    batch = builder.build()
    assert batch.constraints.shape == (2, 2, 3) and np.all(batch.constraints[:, 1] == 0)
    head, tail = batch.split(1)
    np.testing.assert_array_equal(tail.path_code, [3])
    np.testing.assert_array_equal(batch.select(np.array([False, True])).root, [0])
    with pytest.raises(ValueError, match="region-mode builder"):
        builder.append(np.ones((1, 2)), np.ones((1, 1)), None, np.array([1]))


class _Wire:
    """Records point-to-point traffic between two in-process ends."""

    def __init__(self):
        self.objects, self.buffers = {}, defaultdict(list)

    def send(self, obj, dest, tag):
        self.objects[tag] = obj

    def Send(self, buf, dest, tag):  # noqa: N802
        self.buffers[tag].append(np.array(buf))

    def recv(self, source, tag):
        return self.objects[tag]

    def Recv(self, buf, source, tag):  # noqa: N802
        buf[...] = self.buffers[tag].pop(0)


def test_wire_format_round_trips_region_columns_on_table_offsets():
    rng = np.random.default_rng(0)
    batch = ClusterBatch(
        rng.random((5, 2)), rng.random((5, 1)), None, np.array([2, 3]), np.array([3, 3]),
        root=np.array([1, 4]), path_code=np.array([9, 12]), constraints=rng.random((2, 3, 3)),
    )  # fmt: skip
    wire = _Wire()
    ws.send_clusters(wire, batch, dest=1, tag=500)
    assert sorted(wire.buffers) == [501, 502, 504, 505, 506, 507, 508]  # no G: 503 unused
    got = ws.recv_clusters(wire, 0, 2, 1, tag=500)
    for name, value in batch.columns().items():
        np.testing.assert_array_equal(got.columns()[name], value)
    wire.objects[500]["version"] = 1
    with pytest.raises(RuntimeError, match="format 1"):
        ws.recv_clusters(wire, 0, 2, 1, tag=500)


def test_batched_grams_match_direct_sums_and_ignore_other_nodes():
    """Grams are per-node batched products, equal to the direct sums and bitwise batch-independent."""
    from src.region import _SplitRows

    rng = np.random.default_rng(0)
    counts = [7, 5, 7, 9]  # two nodes share a count group, two are alone in theirs
    U = xp.asarray(rng.random((sum(counts), 3)))
    Y = xp.asarray(rng.standard_normal((sum(counts), 2)))
    Phi = xp.concatenate([U, xp.ones((sum(counts), 1))], axis=1)
    labels = xp.asarray(np.repeat(np.arange(4), counts))
    rows = _SplitRows(U, Y, Phi, labels, xp.asarray(counts))
    w = xp.asarray(rng.random(sum(counts)))
    G, B = rows.gram(w)
    P, Yh, Wh, lab = host(Phi), host(Y), host(w), host(labels)
    for k in range(4):
        m = lab == k
        np.testing.assert_allclose(host(G[k]), (P[m] * Wh[m, None]).T @ P[m], rtol=1e-13)
        np.testing.assert_allclose(host(B[k]), (P[m] * Wh[m, None]).T @ Yh[m], rtol=1e-13)
    # The last node alone gives bitwise the same Gram as inside the batch.
    m = lab == 3
    alone = _SplitRows(U[m], Y[m], Phi[m], xp.zeros(9, dtype=xp.int64), xp.asarray([9]))
    np.testing.assert_array_equal(host(alone.gram(w[m])[0][0]), host(G[3]))


def _rows(rng, n=60, d=2):
    from src.region import _SplitRows

    U = xp.asarray(rng.random((n, d)))
    Y = xp.asarray(np.where(host(U)[:, :1] > 0.4, 1.0, -1.0) + 0.1 * rng.standard_normal((n, 1)))
    Phi = xp.concatenate([U, xp.ones((n, 1))], axis=1)
    return _SplitRows(U, Y, Phi, xp.zeros(n, dtype=xp.int64), xp.asarray([n]))


def test_soft_objective_gradient_matches_finite_differences():
    """The envelope-theorem gradient of J = min_B F, checked against central differences."""
    from src.region import _soft_objective

    rows = _rows(np.random.default_rng(1))
    w, z, T = xp.asarray([[0.8, 0.6]]), xp.asarray([0.45]), xp.asarray([0.08])
    _, g_w, g_z, *_ = _soft_objective(rows, w, z, T, 1e-3)
    h = 1e-6
    fd = []
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        e = xp.asarray(e)
        Jp = _soft_objective(rows, w + e[None, :2], z + e[2], T, 1e-3)[0]
        Jm = _soft_objective(rows, w - e[None, :2], z - e[2], T, 1e-3)[0]
        fd.append(float((Jp - Jm)[0]) / (2 * h))
    np.testing.assert_allclose([*host(g_w)[0], float(host(g_z)[0])], fd, rtol=1e-5, atol=1e-9)


def test_sigmoid_is_stable_at_extreme_arguments():
    from src.region import _sigmoid

    s = xp.asarray([-1e4, -40.0, 0.0, 40.0, 1e4])
    out = host(_sigmoid(s))
    assert np.all(np.isfinite(out)) and out[2] == 0.5 and out[0] == 0.0 and out[-1] == 1.0
    np.testing.assert_allclose(out + host(_sigmoid(-s)), 1.0)


def test_soft_refinement_never_returns_a_worse_hard_split_than_its_incumbent():
    from src.config import ClusteringConfig
    from src.region import RegionPlan, _hard_sse

    cfg = ClusteringConfig(atol=1e-3, algorithm="region", domain=([0, 0], [1, 1]),
                           region={"sample_multiplier": 10, "split_search": "soft"})  # fmt: skip
    plan = RegionPlan(cfg, 2, 1, Domain(np.zeros(2), np.ones(2)), Roots.build("box", 2, 1))
    rows = _rows(np.random.default_rng(2))
    w0, z0 = xp.asarray([[0.0, 1.0]]), xp.asarray([0.5])  # a poor incumbent: the wrong axis
    sse0, _ = _hard_sse(rows, w0, z0, plan.min_side)
    w, z = plan._soft_refine(rows, w0, z0, sse0, None)
    sse, ok = _hard_sse(rows, w, z, plan.min_side)
    assert bool(ok[0]) and float(sse[0]) <= float(sse0[0])
    assert plan.counters["soft_improved"] == 1  # the step in x0 = 0.4 is found
    reasons = ("soft_gradient_converged", "soft_line_search_stalled", "soft_gate_saturated",
               "soft_nonfinite", "soft_iteration_limit")  # fmt: skip
    assert sum(plan.counters[r] for r in reasons) == 1  # one termination reason per refined node


def test_drift_check_flags_short_chains_from_packed_seeds_and_extension_clears_it():
    """Seeds packed in one corner of a node: 2 steps cannot spread them; doubling chains do."""
    domain, roots = Domain(np.zeros(2), np.ones(2)), Roots.build("box", 2, 1)
    constraints = xp.asarray(np.r_[1.0, 0.0, 1.0][None, None, :])  # u0 <= 1: the whole square
    seeds = xp.asarray(np.full((5, 2), 0.01) + 0.001 * np.arange(5)[:, None])
    keys = node_keys(0, xp.asarray([0]), xp.asarray([2]))

    def run(max_steps):
        geo = RegionPoints(xp.asarray([0]), xp.asarray([1]), keys, constraints, PURPOSE_TRAIN)
        X = HitAndRunSampler(domain, roots, walk_steps=2, max_walk_steps=max_steps).sample(
            ClusterPoints(seeds, xp.asarray([5]), region=geo), xp.asarray([400])
        )
        return X, bool(geo.insufficient[0])

    _, flagged = run(2)
    assert flagged
    X, flagged = run(64)
    assert not flagged
    np.testing.assert_allclose(host(X.mean(axis=0)), [0.5, 0.5], atol=0.05)


def test_paired_uniforms_are_open_uniform_and_uncorrelated():
    """One hash gives two uniforms (high and low 32 bits): both uniform, independent of each other."""
    from src.region import counter_uniform_pairs

    keys = node_keys(1, xp.asarray([0]), xp.asarray([5]))
    u = host(counter_uniform_pairs(keys[0], PURPOSE_TRAIN, xp.arange(100_000)))
    assert u.shape == (100_000, 2) and u.min() > 0 and u.max() < 1
    np.testing.assert_allclose(u.mean(axis=0), 0.5, atol=5e-3)
    np.testing.assert_allclose(u.var(axis=0), 1 / 12, atol=2e-3)
    assert abs(np.corrcoef(u[:, 0], u[:, 1])[0, 1]) < 0.02
    np.testing.assert_array_equal(u, host(counter_uniform_pairs(keys[0], PURPOSE_TRAIN, xp.arange(100_000))))


def _walk(roots, seeds, n_new, d, max_steps=64, walk_steps=8):
    """Hit-and-run in the whole unit cube (one redundant cut u0 <= 1), seeded at ``seeds``."""
    domain = Domain(np.zeros(d), np.ones(d))
    constraints = xp.asarray(np.r_[1.0, np.zeros(d - 1), 1.0][None, None, :])
    geo = RegionPoints(xp.asarray([0]), xp.asarray([2]), node_keys(0, xp.asarray([0]), xp.asarray([2])), constraints, PURPOSE_TRAIN)
    X = HitAndRunSampler(domain, roots, walk_steps=walk_steps, max_walk_steps=max_steps).sample(
        ClusterPoints(xp.asarray(seeds), xp.asarray([len(seeds)]), region=geo), xp.asarray([n_new])
    )
    return host(X), bool(geo.insufficient[0])


def test_a_one_centre_voronoi_root_walks_exactly_like_the_box():
    """The cell's own centre gives no bisector, so the walk sees only the box and the cut."""
    seeds = np.random.default_rng(0).random((4, 2)) * 0.5
    box, flag_box = _walk(Roots.build("box", 2, 1), seeds, 300, 2)
    vor, flag_vor = _walk(Roots.build("voronoi", 2, 1, centres=np.array([[0.3, 0.6]])), seeds, 300, 2)
    np.testing.assert_array_equal(vor, box)
    assert flag_vor == flag_box


def test_duplicate_voronoi_centres_are_dropped_keeping_the_first():
    roots = Roots.build("voronoi", 2, 3, centres=np.array([[0.2, 0.2], [0.8, 0.8], [0.2, 0.2]]))
    np.testing.assert_array_equal(roots.centres, [[0.2, 0.2], [0.8, 0.8]])


def test_chains_stuck_in_a_20d_corner_are_recovered_or_flagged():
    """From the corner nearly every random chord has zero length; coordinate moves get out."""
    d, corner = 20, np.zeros((5, 20))
    _, flagged = _walk(Roots.build("box", d, 1), corner, 400, d, max_steps=64)
    assert flagged  # moved but not mixed within the budget: never reported as fine
    X, flagged = _walk(Roots.build("box", d, 1), corner, 400, d, max_steps=4096)
    assert not flagged and np.all(np.any(X != 0.0, axis=1)) and np.all((X >= 0) & (X <= 1))
    np.testing.assert_allclose(X.mean(axis=0), 0.5, atol=0.06)


@pytest.mark.parametrize("seed", range(8))
def test_no_endpoint_is_left_on_a_corner_without_a_flag(seed):
    """A minority of unmoved chains (34% expected at d = 4, 8 steps) is recovered too, or the node flagged."""
    d = 4
    domain = Domain(np.zeros(d), np.ones(d))
    constraints = xp.asarray(np.r_[1.0, np.zeros(d - 1), 1.0][None, None, :])
    geo = RegionPoints(xp.asarray([0]), xp.asarray([2]), node_keys(seed, xp.asarray([0]), xp.asarray([2])), constraints, PURPOSE_TRAIN)
    X = host(HitAndRunSampler(domain, Roots.build("box", d, 1), walk_steps=8, max_walk_steps=8).sample(
        ClusterPoints(xp.zeros((3, d)), xp.asarray([3]), region=geo), xp.asarray([20])
    ))  # fmt: skip
    assert bool(geo.insufficient[0]) or not np.any(np.all(X == 0.0, axis=1))


@pytest.mark.parametrize("separation", [1e-6, 1e-8, 1e-10, 1e-12])
def test_close_voronoi_centres_keep_every_sample_in_its_own_thin_cell(separation):
    """Bisectors from centre differences: expanding them cancelled and put every sample in a neighbour's cell."""
    base = 0.695
    roots = Roots.build("voronoi", 1, 3, np.array([[base - separation], [base], [base + separation]]))
    geo = RegionPoints(xp.asarray([1]), xp.asarray([0]), node_keys(2, xp.asarray([1]), xp.asarray([1])), xp.zeros((1, 0, 2)), PURPOSE_TRAIN)
    sampler = HitAndRunSampler(Domain(np.zeros(1), np.ones(1)), roots, walk_steps=8, max_walk_steps=32)
    P = sampler.sample(ClusterPoints(xp.asarray([[base]]), xp.asarray([1]), region=geo), xp.asarray([200]))
    assert np.all(host(roots.root_of(P)) == 1) and not bool(geo.insufficient[0])
    np.testing.assert_allclose([host(P).min() - base, host(P).max() - base], [-separation / 2, separation / 2], rtol=0.05)


def test_narrow_score_clouds_far_from_the_origin_keep_a_resolved_temperature():
    """Translating a node far from the origin must not collapse B4's gate temperature (raw moments cancelled)."""
    from src.config import ClusteringConfig
    from src.region import RegionPlan, _SplitRows

    cfg = ClusteringConfig(atol=1e-3, algorithm="region", domain=([0, 0], [1, 1]),
                           region={"split_search": "soft", "soft_iters": 1})  # fmt: skip
    plan = RegionPlan(cfg, 2, 1, Domain(np.zeros(2), np.ones(2)), Roots.build("box", 2, 1))
    n = 200
    U = np.c_[0.9 + 1e-8 * np.linspace(-1, 1, n), np.random.default_rng(0).random(n)]
    Y = np.where(U[:, :1] > 0.9 + 2e-9, 1.0, -1.0)
    rows = _SplitRows(xp.asarray(U), xp.asarray(Y), xp.asarray(np.c_[U, np.ones(n)]), xp.zeros(n, dtype=xp.int64), xp.asarray([n]))
    plan._soft_refine(rows, xp.asarray([[1.0, 0.0]]), xp.asarray([0.9]), xp.asarray([np.inf]), None)
    assert plan.counters["soft_gate_saturated"] == 0


def test_drift_ignores_rounding_sized_shifts_of_a_narrow_node():
    domain, roots = Domain(np.zeros(2), np.ones(2)), Roots.build("box", 2, 1)
    sampler = HitAndRunSampler(domain, roots, walk_steps=2)
    U = xp.asarray(0.9 + 1e-8 * np.random.default_rng(0).random((20, 2)))
    owner, walk = xp.zeros(20, dtype=xp.int64), xp.ones(20, dtype=bool)
    assert not bool(sampler._drifting(U, U + 1e-12, owner, walk, xp.asarray([20]))[0])
    assert bool(sampler._drifting(U, U + 1e-8, owner, walk, xp.asarray([20]))[0])


def test_threshold_search_skips_nodes_too_small_to_split_beside_valid_ones():
    """A sparse node (fewer than 2 * min_side points, as B1 can produce) gets no candidate and no cross-node read."""
    from src.config import ClusteringConfig
    from src.region import RegionPlan, _SplitRows

    cfg = ClusteringConfig(atol=1e-3, algorithm="region", domain=([0, 0], [1, 1]), region={"split_search": "bins"})
    plan = RegionPlan(cfg, 2, 1, Domain(np.zeros(2), np.ones(2)), Roots.build("box", 2, 1))
    big = _rows(np.random.default_rng(3))
    n_small = 2 * plan.min_side - 1
    U = xp.concatenate([big.U, xp.asarray(np.random.default_rng(4).random((n_small, 2)))])
    Y = xp.concatenate([big.Y, xp.ones((n_small, 1))])
    Phi = xp.concatenate([U, xp.ones((U.shape[0], 1))], axis=1)
    labels = xp.asarray(np.r_[np.zeros(60), np.ones(n_small)].astype(np.int64))
    rows = _SplitRows(U, Y, Phi, labels, xp.asarray([60, n_small]))
    bank = xp.asarray(np.broadcast_to(np.eye(2), (2, 2, 2)).copy())
    _, z, sse = plan._threshold_search(rows, bank)
    assert np.isfinite(host(z)[0]) and np.isfinite(host(sse)[0])
    assert np.isnan(host(z)[1]) and np.isinf(host(sse)[1])
    alone = _SplitRows(big.U, big.Y, big.Phi, big.labels, big.counts)
    _, z_alone, _ = plan._threshold_search(alone, bank[:1])
    assert float(host(z_alone)[0]) == float(host(z)[0])


def test_a_62_level_skinny_tree_with_many_roots_routes_to_its_deepest_leaf():
    """Child indices, not packed keys: full path-code depth even with many roots."""
    depth = 62
    domain, roots = Domain(np.zeros(2), np.ones(2)), Roots.build("grid", 2, 1024)
    # Root 1023 halves u0 repeatedly towards 1: path codes 1, 3, 7, ... (always the right child).
    split_path = [(1 << (k + 1)) - 1 for k in range(depth)]
    leaf_path = [p << 1 for p in split_path] + [(split_path[-1] << 1) | 1]
    lo, hi = (host(b)[1023] for b in roots.bounds())
    z = [lo[0] + (hi[0] - lo[0]) * (1 - 0.5 ** (k + 1)) for k in range(depth)]
    others = [r for r in range(1024) if r != 1023]
    table = RoutingTable.build(
        domain, roots, leaf_root=others + [1023] * (depth + 1), leaf_path=[1] * 1023 + leaf_path,
        split_root=[1023] * depth, split_path=split_path, w=np.tile([1.0, 0.0], (depth, 1)), z=np.array(z),
    )  # fmt: skip
    q = xp.asarray([[float(hi[0]), float((lo[1] + hi[1]) / 2)]])
    leaf, outside = table.route(q)
    assert int(host(leaf)[0]) == table.n_leaves - 1 and not bool(host(outside)[0])
    again = RoutingTable.from_arrays(table.arrays())
    assert int(host(again.route(q)[0])[0]) == table.n_leaves - 1


def test_routing_table_loads_the_packed_key_format():
    domain, roots, tree = _tree()
    shift = 8
    sk = (np.array(tree["split_root"]) << shift) | np.array(tree["split_path"])
    lk = (np.array(tree["leaf_root"]) << shift) | np.array(tree["leaf_path"])
    order = np.argsort(lk)
    old = {
        "domain_lower": np.zeros(2), "domain_upper": np.ones(2), "root_bins": roots.bins, "shift": shift,
        "split_keys": sk, "leaf_keys": lk[order], "leaf_index": order, "w": tree["w"], "z": tree["z"],
    }  # fmt: skip
    new = RoutingTable.build(domain, roots, **tree)
    q = xp.asarray(np.random.default_rng(1).random((200, 2)))
    np.testing.assert_array_equal(host(RoutingTable.from_arrays(old).route(q)[0]), host(new.route(q)[0]))


@pytest.mark.parametrize("broken", ["nan_plane", "bad_shape", "unknown_root", "zero_width", "bad_bins"])
def test_loaded_routing_tables_are_validated(broken):
    domain, roots, tree = _tree()
    a = RoutingTable.build(domain, roots, **tree).arrays()
    if broken == "nan_plane":
        a["w"] = a["w"].copy()
        a["w"][0, 0] = np.nan
    elif broken == "bad_shape":
        a["z"] = a["z"][:1]
    elif broken == "unknown_root":
        a["leaf_root"] = np.asarray(a["leaf_root"]).copy()
        a["leaf_root"][0] = 5
    elif broken == "zero_width":
        a["domain_upper"] = np.array([1.0, 0.0])
    else:
        a["root_bins"] = np.array([1, 0])
    with pytest.raises(ValueError):
        RoutingTable.from_arrays(a)


@pytest.mark.parametrize("query", [[[np.nan, 0.5]], [[0.5, np.inf]], [[0.5, 0.5, 0.5]]])
def test_routing_rejects_non_finite_or_misshaped_queries(query):
    domain, roots, tree = _tree()
    with pytest.raises(ValueError):
        RoutingTable.build(domain, roots, **tree).route(xp.asarray(query))
