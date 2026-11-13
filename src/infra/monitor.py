"""
Live terminal monitor for a training run started with ``profile_dir``: per-rank state
(phase, level, batch), queued work, accepted leaves, idle share and work stealing,
read from the ranks' JSON-lines event streams (see ``src.infra.profiling``). The header names
the algorithm plan (AFINNI, or ART with its split search and roots); for
ART it also counts the points in fallback leaves (nodes stopped at the depth
limit or without a valid split), which are finished work, not queued work.

    python -m src.infra.monitor <profile_dir> [--interval 0.5] [--once]

It only reads files, so it runs from another shell (or node with a shared file
system) while the job runs, or afterwards to replay the final state, and adds no
MPI traffic. Event coalescing in the profiler means the newest idle stretch of a
rank appears once that rank records its next event.
"""

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.infra.profiling import PHASE_BY_KEY, fmt_count

_LOGGED_KINDS = ("steal_grant", "level_end", "retired", "done")
# Wide enough for every phase group and "asking R<rank> for work".
_STATE_WIDTH = max(24, *(len(p.group) for p in PHASE_BY_KEY.values()))
_BAR_WIDTH = 12


@dataclass
class RankStatus:
    """What one rank is doing, folded from its events in order."""

    rank: int
    state: str = "starting"
    level: int | None = None
    batch: int | None = None
    n_batches: int | None = None  # planned batches in the current level
    queue_pts: int = 0
    queue_clusters: int = 0
    leaves: int = 0
    generated: int = 0
    idle_s: float = 0.0
    busy_s: float = 0.0
    t_last: float = 0.0
    steals_out: int = 0
    steals_in: int = 0
    pts_out: int = 0
    pts_in: int = 0
    done: bool = False
    plan: str | None = None  # from the "run_start" event
    art: bool = False  # the run is ART (algorithm="art"): its stopped nodes are fallback leaves
    fallback_pts: int = 0  # ART: points of nodes that kept a fallback leaf

    @property
    def idle_share(self) -> float:
        total = self.idle_s + self.busy_s
        return self.idle_s / total if total else 0.0

    def apply(self, e: dict[str, Any]) -> None:
        """Update from one event (see ``src.infra.profiling`` for the event kinds)."""
        kind = e["kind"]
        self.t_last = max(self.t_last, e.get("t", 0.0) + e.get("dur", 0.0))
        if kind == "run_start":
            self.plan, self.art = e.get("plan"), e.get("algorithm") == "art"
        elif kind == "phase":
            phase = PHASE_BY_KEY.get(e["phase"])
            if e["phase"] == "time_idle":
                self.idle_s += e["dur"]
                self.state = "idle"
            else:
                self.busy_s += e["dur"]
                if not self.done:  # the final gather comes after "done"
                    self.state = phase.group if phase else e["phase"]
            if e.get("level") is not None:
                self.level, self.batch = e["level"], e.get("batch", self.batch)
        elif kind == "level_start":
            self.level, self.batch = e["level"], None
            self.queue_pts, self.queue_clusters = e["n_pts"], e["n_clusters"]
            self.n_batches = e.get("n_batches")
        elif kind == "batch":
            self.leaves += e["accepted"]
            self.batch = e["batch"]
        elif kind == "level_end":
            self.generated += e.get("generated_pts", 0)
            self.queue_pts = e.get("num_pts_next_level", self.queue_pts)
            self.queue_clusters = e.get("clusters_level_end", self.queue_clusters)
        elif kind == "load":
            self.queue_pts = e["n_pts"]
        elif kind == "steal_request":
            self.state = f"asking R{e['victim']} for work"
        elif kind == "steal_grant":
            self.steals_out += 1
            self.pts_out += e["n_pts"]
        elif kind == "steal_received":
            self.steals_in += 1
            self.pts_in += e["n_pts"]
        elif kind == "retire_start":
            self.state = "retiring"
        elif kind == "retired":
            self.state = "retired"
        elif kind == "done":
            self.done, self.state = True, "done"
            self.level = self.batch = self.n_batches = None
            self.leaves, self.queue_pts = e["leaves"], e.get("unresolved_pts", 0)
            self.fallback_pts = e.get("fallback_pts", 0)


@dataclass
class StreamMonitor:
    """Incrementally read every ``rank_<r>.jsonl`` in ``stream_dir`` and fold the events."""

    stream_dir: Path
    ranks: dict[int, RankStatus] = field(default_factory=dict)
    recent: deque = field(default_factory=lambda: deque(maxlen=8))
    n_events: int = 0
    _offsets: dict[Path, int] = field(default_factory=dict)
    _inodes: dict[Path, int] = field(default_factory=dict)

    def poll(self) -> int:
        """Read whatever was appended since the last poll; returns the number of new events."""
        new = 0
        for path in sorted(Path(self.stream_dir).glob("rank_*.jsonl")):
            rank = int(path.stem.split("_")[1])
            try:
                stat = path.stat()
                with open(path, "rb") as fh:
                    # A new run replaces or truncates the file: start this rank afresh.
                    if self._inodes.get(path) != stat.st_ino or stat.st_size < self._offsets.get(path, 0):
                        self._inodes[path], self._offsets[path] = stat.st_ino, 0
                        self.ranks[rank] = RankStatus(rank)
                    fh.seek(self._offsets[path])
                    data = fh.read()
            except FileNotFoundError:  # removed between listing and opening
                continue
            status = self.ranks.setdefault(rank, RankStatus(rank))
            # Only complete lines: a writer may be in the middle of the last one.
            end = data.rfind(b"\n") + 1
            self._offsets[path] += end
            for line in data[:end].splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                status.apply(event)
                if event["kind"] in _LOGGED_KINDS:
                    self.recent.append(event)
                new += 1
        self.n_events += new
        return new

    @property
    def finished(self) -> bool:
        return bool(self.ranks) and all(s.done for s in self.ranks.values())

    def render(self) -> Group:
        """The current view: a summary line, one row per rank, and recent milestones."""
        ranks = [self.ranks[r] for r in sorted(self.ranks)]
        elapsed = max((s.t_last for s in ranks), default=0.0)
        n_done = sum(s.done for s in ranks)
        plan = next((s.plan for s in ranks if s.plan), "AFINNI")
        art = any(s.art for s in ranks)
        summary = Text.assemble(
            (f"Training: {plan}  ", "bold"),
            f"{elapsed:7.1f}s   ranks done {n_done}/{len(ranks)}   ",
            f"leaves {sum(s.leaves for s in ranks):,}   ",
            f"queued {fmt_count(sum(s.queue_pts for s in ranks))} pts   ",
            f"in fallback leaves {fmt_count(sum(s.fallback_pts for s in ranks))} pts   " if art else "",
            f"steals {sum(s.steals_out for s in ranks)}   events {self.n_events:,}",
        )
        # Fixed widths (+2 for cell padding): content-sized columns would resize
        # whenever a value changes.
        table = Table(header_style="bold")
        for col, justify, width in (
            ("Rank", "right", 4), ("State", "left", _STATE_WIDTH), ("Level", "right", 5),
            ("Batch", "right", 9), ("Queued pts", "right", 10), ("Load", "left", _BAR_WIDTH),
            ("Clusters", "right", 9), ("Leaves", "right", 9), ("Generated", "right", 9),
            ("Idle", "right", 4), ("Gave work", "right", 13), ("Got work", "right", 13),
            ("Last updated", "right", 12),
        ):  # fmt: skip
            table.add_column(
                col, justify=justify, width=max(width, len(col)) + 2, no_wrap=True, overflow="ellipsis"
            )
        max_queue = max((s.queue_pts for s in ranks), default=0) or 1
        for s in ranks:
            bar = "█" * round(_BAR_WIDTH * s.queue_pts / max_queue)
            table.add_row(
                str(s.rank),
                Text(s.state, style="green" if s.done else ("yellow" if s.state == "idle" else "")),
                "-" if s.level is None else str(s.level),
                _batch_label(s),
                fmt_count(s.queue_pts), Text(bar, style="cyan"), f"{s.queue_clusters:,}",
                f"{s.leaves:,}", fmt_count(s.generated), f"{s.idle_share:.0%}",
                _steal_label(s.steals_out, s.pts_out), _steal_label(s.steals_in, s.pts_in),
                _seen_label(elapsed - s.t_last, s.done),
            )  # fmt: skip
        log = Table.grid(padding=(0, 2))
        for e in self.recent:
            if e["kind"] == "steal_grant":
                msg = f"R{e['rank']} gave {fmt_count(e['n_pts'])} pts ({e['n_clusters']} clusters) to R{e['thief']}"
            elif e["kind"] == "level_end":
                msg = f"R{e['rank']} finished level {e['level']}: {e.get('good_clusters', 0):,} accepted, {fmt_count(e.get('num_pts_next_level', 0))} pts queued"
            else:
                msg = f"R{e['rank']} {e['kind']}"
            log.add_row(f"{e['t']:7.2f}s", msg)
        return Group(summary, table, Panel(log, title="recent", title_align="left"))


def _batch_label(s: RankStatus) -> str:
    """``current/planned`` batches of the rank's level ("-" between levels)."""
    if s.batch is None:
        return "-" if s.n_batches is None else f"0/{s.n_batches}"
    return f"{s.batch + 1}/{s.n_batches}" if s.n_batches else str(s.batch + 1)


def _steal_label(n: int, pts: int) -> str:
    """``"<grants> × <points>"``, e.g. ``3× 20.3k pts``; "-" when there were none."""
    return f"{n}× {fmt_count(pts)} pts" if n else "-"


def _seen_label(ago: float, done: bool) -> str:
    """How long ago, relative to the newest event of any rank, this rank last reported."""
    return "-" if done else f"{max(ago, 0.0):.1f}s ago"


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("stream_dir", type=Path, help="the profile_dir given to training")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between refreshes")
    p.add_argument("--once", action="store_true", help="print the current state once and exit")
    args = p.parse_args(argv)
    monitor = StreamMonitor(args.stream_dir)
    console = Console()
    if args.once:
        monitor.poll()
        console.print(monitor.render())
        return
    with Live(monitor.render(), console=console, refresh_per_second=4, screen=False) as live:
        try:
            while True:
                monitor.poll()
                live.update(monitor.render())
                if monitor.finished:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
