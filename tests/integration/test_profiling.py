"""Profiler behaviour (timings, events, coalescing, streams) and what training records."""

import json
import time

import numpy as np
import pytest

from src import xp
from src.modes import SurrogateMode
from src.profiling import (
    ORACLE_TIME_KEY,
    PHASE_BY_KEY,
    PHASES,
    Profiler,
    new_profiling_stats,
    read_event_stream,
    read_event_streams,
)
from src.surrogate import AFINNISurrogate

X = np.random.default_rng(4).uniform(-1, 1, (3_000, 2))


def f(x):
    return (1 + x[:, 0] ** 2 + 4 * x[:, 1] ** 2)[:, None]


def train(**kw):
    s = AFINNISurrogate()
    s.train(
        xp.asarray(X), f_fn=f, ATOL=0.02, RTOL=0.02, ERROR_METRIC="Linf",
        NUM_INITIAL_CLUSTERS=8, MIN_PTS_PER_CLUSTER=40, USE_WORK_STEALING=False, **kw,
    )  # fmt: skip
    return s


# --- Profiler unit behaviour ---------------------------------------------------------


def test_disabled_profiler_records_nothing():
    stats = new_profiling_stats()
    p = Profiler(stats)
    with p.phase("time_bad_split"):
        pass
    p.event("steal_grant", thief=1)
    assert not p.enabled and "events" not in stats and stats["time_bad_split"] == 0.0


def test_timing_accumulates_totals_without_events():
    stats = new_profiling_stats()
    p = Profiler(stats, timing=True)
    for _ in range(3):
        with p.phase("time_bad_split"):
            time.sleep(0.002)
    p.add_time("time_mpi_send", 0.5)
    assert stats["time_bad_split"] >= 0.006 and stats["time_mpi_send"] == 0.5
    assert "events" not in stats


def test_events_carry_level_batch_rank_and_shared_epoch():
    stats = new_profiling_stats()
    p = Profiler(stats, rank=3, events=True, t0=time.time() - 10.0)
    p.level, p.batch = 2, 1
    with p.phase("time_error_calc"):
        pass
    p.event("batch", accepted=4)
    phase, batch = stats["events"]
    assert phase["kind"] == "phase" and phase["phase"] == "time_error_calc"
    assert (phase["level"], phase["batch"], phase["rank"]) == (2, 1, 3)
    assert phase["t"] >= 10.0 and phase["dur"] >= 0.0
    assert batch == {"kind": "batch", "t": batch["t"], "level": 2, "accepted": 4, "rank": 3}
    assert stats["time_error_calc"] == 0.0  # totals need timing=True


def test_repeated_phases_and_denials_are_coalesced_but_nested_phases_skipped():
    stats = new_profiling_stats()
    p = Profiler(stats, events=True)
    for _ in range(50):
        with p.phase("time_idle"):
            with p.phase("time_state_update"):  # nested: totals only
                pass
        p.event("steal_denied", victim=0, reason="IDLE")
    kinds = [(e["kind"], e.get("phase")) for e in stats["events"]]
    # Alternating idle/denial events cannot merge with each other...
    assert ("phase", "time_state_update") not in kinds
    # ...but back-to-back repeats do.
    stats2 = new_profiling_stats()
    p2 = Profiler(stats2, events=True)
    for _ in range(50):
        with p2.phase("time_idle"):
            pass
    for _ in range(20):
        p2.event("steal_denied", victim=0, reason="IDLE")
    p2.event("steal_denied", victim=1, reason="IDLE")
    idle, denied0, denied1 = stats2["events"]
    assert idle["count"] == 50 and idle["dur"] >= 0
    assert denied0["count"] == 20 and denied0["t_last"] >= denied0["t"]
    assert denied1["victim"] == 1 and "count" not in denied1


def test_phases_far_apart_are_not_merged():
    stats = new_profiling_stats()
    p = Profiler(stats, events=True)
    with p.phase("time_idle"):
        pass
    time.sleep(0.3)
    with p.phase("time_idle"):
        pass
    assert len(stats["events"]) == 2


def test_event_stream_is_json_lines_and_tolerates_a_partial_line(tmp_path):
    stats = new_profiling_stats()
    p = Profiler(stats, rank=2, stream_dir=tmp_path)
    with p.phase("time_init"):
        pass
    p.event("steal_grant", thief=1, n_pts=10, n_clusters=1)
    p.event("load", n_pts=5)
    p.close()
    p.close()  # idempotent
    path = tmp_path / "rank_002.jsonl"
    events = read_event_stream(path)
    assert [e["kind"] for e in events] == ["phase", "steal_grant", "load"]
    assert events == json.loads(json.dumps(stats["events"]))
    with open(path, "a") as fh:
        fh.write('{"kind": "pha')  # a writer interrupted mid-line
    assert len(read_event_stream(path)) == 3
    assert list(read_event_streams(tmp_path)) == [2]


def test_phase_table_is_consistent():
    keys = [p.key for p in PHASES]
    assert len(keys) == len(set(keys)) and all(k.startswith("time_") for k in keys)
    assert set(keys) <= set(new_profiling_stats())
    assert PHASE_BY_KEY["time_state_update"].nested and not PHASE_BY_KEY["time_idle"].nested


# --- What training records ------------------------------------------------------------


def test_profiling_off_by_default_records_levels_only():
    s = train()
    ps = s.profiling_stats
    assert all("events" not in r for r in ps["per_rank"])
    assert ps["level_stats"] and all("t_start" in e and "t_end" in e for e in ps["level_stats"])
    assert sum(e["good_clusters"] for e in ps["level_stats"]) == s.N_src
    assert all(ps[p.key] == 0.0 for p in PHASES)


def test_timing_totals_fit_inside_the_total_time():
    ps = train(PROFILE_TIMING=True).profiling_stats["per_rank"][0]
    timed = sum(ps[p.key] for p in PHASES if not p.nested)
    assert 0 < timed <= ps["time_total"] * 1.01
    for key in ("time_init", "time_error_calc", "time_good_Q", "time_augment", "time_bad_split"):
        assert ps[key] > 0, key
    assert 0 < ps[ORACLE_TIME_KEY] < ps["time_total"]


def test_events_describe_every_level_and_batch(tmp_path):
    s = train(PROFILE_EVENTS=True, PROFILE_DIR=tmp_path)
    ps = s.profiling_stats
    events = ps["per_rank"][0]["events"]
    kinds = [e["kind"] for e in events]
    assert kinds.count("level_start") == kinds.count("level_end") == len(ps["level_stats"])
    batches = [e for e in events if e["kind"] == "batch"]
    assert sum(e["accepted"] for e in batches) == s.N_src
    for start in (e for e in events if e["kind"] == "level_start"):  # planned = actual batches (one rank)
        assert start["n_batches"] == sum(b["level"] == start["level"] for b in batches)
    phases = [e for e in events if e["kind"] == "phase"]
    assert {e["phase"] for e in phases} >= {"time_init", "time_error_calc", "time_bad_split"}
    assert all(e["t"] >= 0 and e["dur"] >= 0 for e in phases)
    assert max(e["t"] + e["dur"] for e in phases) <= ps["time_total"] + 1.0
    # The live stream holds the same events.
    assert read_event_streams(tmp_path)[0] == json.loads(json.dumps(events))


def test_profiles_survive_save_and_load(tmp_path):
    s = train(PROFILE_EVENTS=True)
    loaded = AFINNISurrogate.load_model(s.save(tmp_path / "m.npz"))
    assert loaded.profiling_stats["per_rank"][0]["events"] == json.loads(
        json.dumps(s.profiling_stats["per_rank"][0]["events"])
    )


@pytest.mark.parametrize("mode", [SurrogateMode.AFFINE_COV_HESS])
def test_profiling_does_not_change_results(mode):
    a, b = train(), train(PROFILE_TIMING=True, PROFILE_EVENTS=True)
    assert a.N_src == b.N_src
    np.testing.assert_array_equal(np.asarray(a.counts_src), np.asarray(b.counts_src))


def test_streamed_idle_time_reaches_the_file_within_about_a_second(tmp_path):
    """A rank idling for a long time must not look frozen to a live monitor."""
    p = Profiler(new_profiling_stats(), stream_dir=tmp_path)
    t_end = time.time() + 1.3
    while time.time() < t_end:
        with p.phase("time_idle"):
            time.sleep(0.05)
    # Still idling (not closed): the first second is already in the file.
    streamed = read_event_stream(tmp_path / "rank_000.jsonl")
    assert streamed and streamed[0]["phase"] == "time_idle" and streamed[0]["dur"] >= 0.9
    p.close()
    total = sum(e["dur"] for e in read_event_stream(tmp_path / "rank_000.jsonl"))
    assert 1.2 <= total <= 1.5
