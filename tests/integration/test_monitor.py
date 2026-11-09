"""The terminal monitor folds live event streams into per-rank status, including partial writes."""

import json

import numpy as np
from rich.console import Console

from src.monitor import RankStatus, StreamMonitor, main
from src.surrogate import AFINNISurrogate
from tests._helpers import train_surrogate


def _write(path, events, tail=""):
    with open(path, "a") as fh:
        fh.writelines(json.dumps(e) + "\n" for e in events)
        fh.write(tail)


def _text(monitor) -> str:
    console = Console(record=True, width=160)
    console.print(monitor.render())
    return console.export_text()


def test_monitor_follows_streams_as_they_grow(tmp_path):
    r0, r1 = tmp_path / "rank_000.jsonl", tmp_path / "rank_001.jsonl"
    _write(r0, [
        {"kind": "level_start", "t": 0.0, "rank": 0, "level": 0, "n_pts": 1000, "n_clusters": 4},
        {"kind": "phase", "phase": "time_error_calc", "t": 0.1, "dur": 0.2, "rank": 0, "level": 0, "batch": 0},
        {"kind": "batch", "t": 0.3, "rank": 0, "level": 0, "batch": 0, "accepted": 3, "rejected": 1},
    ], tail='{"kind": "steal_gr')  # rank 0 is mid-write
    _write(r1, [{"kind": "phase", "phase": "time_idle", "t": 0.0, "dur": 0.5, "rank": 1, "level": None, "batch": None}])
    mon = StreamMonitor(tmp_path)
    assert mon.poll() == 4
    s0, s1 = mon.ranks[0], mon.ranks[1]
    assert (s0.state, s0.level, s0.batch, s0.leaves, s0.queue_pts) == ("Fit + error", 0, 0, 3, 1000)
    assert s1.state == "idle" and s1.idle_share == 1.0
    assert not mon.finished

    # The writer finishes the line and continues.
    _write(r0, [], tail='ant", "t": 0.4, "rank": 0, "level": 0, "thief": 1, "n_pts": 400, "n_clusters": 2}\n')
    _write(r1, [
        {"kind": "steal_received", "t": 0.4, "rank": 1, "level": None, "victim": 0, "n_pts": 400, "n_clusters": 2},
        {"kind": "done", "t": 0.9, "rank": 1, "level": None, "leaves": 2, "unresolved_pts": 0},
        {"kind": "phase", "phase": "time_gather", "t": 0.9, "dur": 0.1, "rank": 1, "level": None, "batch": None},
    ])
    assert mon.poll() == 4
    assert (s0.steals_out, s0.pts_out, s1.steals_in, s1.pts_in) == (1, 400, 1, 400)
    assert s1.done and s1.state == "done" and s1.leaves == 2  # the gather after "done" keeps the state
    text = _text(mon)
    assert "R0 gave 400 pts (2 clusters) to R1" in text and "ranks done 1/2" in text
    _write(r0, [{"kind": "done", "t": 1.0, "rank": 0, "level": None, "leaves": 3, "unresolved_pts": 0}])
    mon.poll()
    assert mon.finished and mon.poll() == 0


def test_rank_status_reports_unresolved_points_when_done():
    s = RankStatus(0)
    s.apply({"kind": "level_end", "t": 1.0, "level": 3, "generated_pts": 50, "num_pts_next_level": 700, "clusters_level_end": 6})
    assert (s.generated, s.queue_pts, s.queue_clusters) == (50, 700, 6)
    s.apply({"kind": "done", "t": 2.0, "leaves": 9, "unresolved_pts": 700})
    assert s.done and s.queue_pts == 700 and s.leaves == 9


def test_monitor_cli_snapshot_after_a_real_run(tmp_path, capsys):
    s = AFINNISurrogate()
    train_surrogate(
        np.random.default_rng(0).uniform(-1, 1, (3_000, 2)), s,
        f_fn=lambda x: (1 + x[:, 0] ** 2 + 4 * x[:, 1] ** 2)[:, None], atol=0.02, rtol=0.02,
        num_initial_clusters=4, min_pts_per_cluster=40, use_work_stealing=False, profile_dir=tmp_path,
    )  # fmt: skip
    mon = StreamMonitor(tmp_path)
    mon.poll()
    assert mon.finished and mon.ranks[0].leaves == s.N_src
    main([str(tmp_path), "--once"])
    assert f"leaves {s.N_src:,}" in capsys.readouterr().out


def test_monitor_restarts_a_rank_when_its_stream_is_replaced(tmp_path):
    path = tmp_path / "rank_000.jsonl"
    _write(path, [{"kind": "batch", "t": 0.1, "rank": 0, "level": 0, "batch": 0, "accepted": 5, "rejected": 0}] * 3)
    mon = StreamMonitor(tmp_path)
    mon.poll()
    assert mon.ranks[0].leaves == 15
    # A new run deletes the directory and writes a shorter file in its place.
    path.unlink()
    _write(path, [{"kind": "batch", "t": 0.1, "rank": 0, "level": 0, "batch": 0, "accepted": 2, "rejected": 0}])
    mon.poll()
    assert mon.ranks[0].leaves == 2


def test_monitor_skips_corrupt_lines(tmp_path):
    _write(tmp_path / "rank_000.jsonl", [], tail='{"kind": "load"}{"broken": 1}\nnot json\n')
    _write(tmp_path / "rank_000.jsonl", [{"kind": "load", "t": 1.0, "rank": 0, "level": None, "n_pts": 7}])
    mon = StreamMonitor(tmp_path)
    assert mon.poll() == 1 and mon.ranks[0].queue_pts == 7


def test_batch_progress_and_seen_labels():
    from src.monitor import _batch_label, _seen_label

    s = RankStatus(0)
    s.apply({"kind": "level_start", "t": 0.0, "level": 2, "n_pts": 10, "n_clusters": 2, "n_batches": 5})
    assert _batch_label(s) == "0/5"
    s.apply({"kind": "batch", "t": 0.1, "level": 2, "batch": 1, "accepted": 0, "rejected": 2})
    assert _batch_label(s) == "2/5"
    s.apply({"kind": "phase", "phase": "time_idle", "t": 0.2, "dur": 0.1, "level": None, "batch": None})
    assert s.level == 2  # idle time does not erase the last level
    assert _seen_label(3.25, False) == "3.2s ago" and _seen_label(1.0, True) == "-"
