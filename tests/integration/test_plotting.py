"""Plots render from real and synthetic profiles, and their tables hold the right numbers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import plotting as P  # noqa: E402
from src.infra.profiling import new_profiling_stats  # noqa: E402
from src.surrogate import AFINNISurrogate  # noqa: E402
from tests._helpers import train_surrogate  # noqa: E402


def synthetic_profile(n_ranks=3):
    """A 3-rank profile with phases, loads, levels and two steals from rank 0."""
    per_rank = []
    for r in range(n_ranks):
        ps = new_profiling_stats()
        ps.update(time_total=10.0, time_error_calc=2.0 + r, time_bad_split=1.0, time_idle=3.0 - r, time_state_update=9.0)
        ps["level_stats"] = [
            {"rank": r, "level": lvl, "clusters_start": 4 * 2**lvl, "good_clusters": 2**lvl,
             "num_pts_next_level": 1000 // (lvl + 1), "generated_pts": 10 * lvl, "runtime_sec": 0.1}
            for lvl in range(4)
        ]  # fmt: skip
        ps["events"] = [
            {"kind": "phase", "phase": "time_error_calc", "t": 0.0, "dur": 2.0, "rank": r, "level": 0, "batch": 0},
            {"kind": "phase", "phase": "time_idle", "t": 2.0, "dur": 3.0, "rank": r, "level": None, "batch": None},
            {"kind": "level_start", "t": 0.0, "rank": r, "level": 0, "n_pts": 1000, "n_clusters": 4},
            {"kind": "load", "t": 1.0, "rank": r, "level": 0, "n_pts": 500},
        ]
        per_rank.append(ps)
    per_rank[0]["events"] += [
        {"kind": "steal_grant", "t": 1.5, "rank": 0, "level": 1, "thief": 1, "n_pts": 300, "n_clusters": 3, "mid_level": True},
        {"kind": "steal_grant", "t": 1.7, "rank": 0, "level": 1, "thief": 2, "n_pts": 200, "n_clusters": 2, "mid_level": False},
    ]
    return {
        "per_rank": per_rank, "time_total": 10.0, "num_final_clusters": 42,
        "level_stats": [e for ps in per_rank for e in ps["level_stats"]],
    }  # fmt: skip


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_tables_hold_the_profile_numbers():
    prof = synthetic_profile()
    bd = P.phase_breakdown_frame(prof)
    assert bd.loc[1, "Fit + error"] == 3.0 and bd.loc[2, "Idle"] == 1.0
    assert "Work stealing" in bd and bd["Work stealing"].sum() == 0.0  # nested phases excluded
    assert np.allclose(bd.drop(columns=["total"]).sum(axis=1), 10.0)  # untimed fills the gap
    steals = P.steals_frame(prof)
    assert steals[["victim", "thief", "n_pts"]].values.tolist() == [[0, 1, 300], [0, 2, 200]]
    assert len(P.level_stats_frame(prof)) == 12
    assert P.events_frame(prof)["t"].is_monotonic_increasing


@pytest.mark.parametrize(
    "plot",
    [
        P.plot_rank_time_breakdown,
        P.plot_level_heatmap,
        P.plot_level_progress,
        P.plot_timeline,
        P.plot_load_over_time,
        P.plot_steal_matrix,
        P.plot_profile_dashboard,
    ],
)
def test_every_profile_plot_renders_and_saves(plot, tmp_path):
    fig = plot(synthetic_profile(), tmp_path / f"{plot.__name__}.png")
    assert isinstance(fig, plt.Figure) and (tmp_path / f"{plot.__name__}.png").stat().st_size > 0


def test_steal_matrix_includes_ranks_without_steals():
    fig = P.plot_steal_matrix(synthetic_profile(n_ranks=5))
    assert len(fig.axes[0].get_xticks()) == 5


def test_plots_from_a_real_training_run(tmp_path):
    s = AFINNISurrogate()
    train_surrogate(
        np.random.default_rng(0).uniform(-1, 1, (2_000, 2)), s,
        f_fn=lambda x: (1 + x[:, 0] ** 2 + 4 * x[:, 1] ** 2)[:, None], atol=0.02, rtol=0.02,
        num_initial_clusters=4, min_pts_per_cluster=40, use_work_stealing=False,
        profile_timing=True, profile_events=True, profile_dir=tmp_path / "stream",
    )  # fmt: skip
    for plot in (P.plot_rank_time_breakdown, P.plot_level_heatmap, P.plot_level_progress,
                 P.plot_timeline, P.plot_load_over_time, P.plot_profile_dashboard):  # fmt: skip
        assert isinstance(plot(s), plt.Figure)
    from src.infra.profiling import read_event_streams

    assert isinstance(P.plot_timeline(read_event_streams(tmp_path / "stream")), plt.Figure)
    with pytest.raises(ValueError, match="No steals"):
        P.plot_steal_matrix(s)  # single rank: nothing to show, with a clear message


def test_missing_data_gives_actionable_errors():
    s = synthetic_profile()
    for ps in s["per_rank"]:
        ps.pop("events")
        for k in list(ps):
            if k.startswith("time_") and k != "time_total":
                ps[k] = 0.0
    with pytest.raises(ValueError, match="profile_timing"):
        P.plot_rank_time_breakdown(s)
    with pytest.raises(ValueError, match="profile_events"):
        P.plot_timeline(s)


def test_region_and_comparison_plots_render(tmp_path):
    """Region partition and tree statistics, error maps, error CDF and metric bars, from a real model."""
    from src import xp
    from src.problems.spherical_piecewise import spherical_piecewise

    X = np.random.default_rng(0).uniform(-3, 3, (150, 2))
    region = train_surrogate(
        X, f_fn=spherical_piecewise, rtol=1e-2, error_metric="pointwise", algorithm="region",
        domain=([-3, -3], [3, 3]), max_levels=5, use_work_stealing=False, region={"sample_multiplier": 10},
    )  # fmt: skip
    afinni = train_surrogate(
        X, f_fn=spherical_piecewise, rtol=0.05, num_initial_clusters=4, min_pts_per_cluster=20, max_levels=4,
        use_work_stealing=False,
    )  # fmt: skip
    Xq = xp.asarray(np.random.default_rng(1).uniform(-3, 3, (500, 2)))
    truth = spherical_piecewise(Xq)
    err = P.pointwise_relative_error(truth, region._predict_route_local(Xq)[0])
    assert err.shape == (500,) and np.all(err >= 0)
    paths = {name: tmp_path / f"{name}.png" for name in ("partition", "stats", "maps", "cdf", "bars")}
    P.plot_region_partition(region, paths["partition"], f_fn=spherical_piecewise, tol=1e-2, resolution=60)
    P.plot_region_tree_stats(region, paths["stats"])
    P.plot_error_maps({"afinni": lambda q: afinni._predict_local(q, 1, 10, False)[0],
                       "region": lambda q: region._predict_route_local(q)[0]},
                      spherical_piecewise, (-3, -3), (3, 3), paths["maps"], tol=1e-2, resolution=40)  # fmt: skip
    P.plot_error_cdf({"region": err}, paths["cdf"], tol=1e-2)
    P.plot_method_comparison({"a": {"leaves": 10, "relative_l2": 0.1}, "b": {"leaves": 20, "relative_l2": 0.05}},
                             paths["bars"])  # fmt: skip
    assert all(p.stat().st_size > 0 for p in paths.values())
    plt.close("all")
    with pytest.raises(ValueError, match="region-mode model"):
        P.plot_region_tree_stats(afinni)


def test_spherical_piecewise_matches_art_preset_values():
    from src.problems.spherical_piecewise import OFFSET, RADII, spherical_piecewise

    X = np.array([[0.0, 0.0], [RADII[2], 0.0], [3.0, 3.0]])
    np.testing.assert_allclose(spherical_piecewise(X)[:, 0], [OFFSET + 5.0, OFFSET + 5.0 - 0.25 * RADII[2], OFFSET + 0.3])
