"""
Per-rank profiling of adaptive clustering: phase timings, a timeline of events, and
live per-rank event streams for monitoring.

Three levels of detail, all off by default except the always-cheap level statistics:

- ``timing`` (``profile_timing``): accumulate wall time per phase into the existing
  ``profiling_stats["time_*"]`` totals; on CuPy the device is synchronized around
  each phase so kernel time is attributed correctly (this slows training).
- ``events`` (``profile_events``): also record a timeline: one interval per phase
  (with its level and batch), steal requests/grants/denials, load samples and level
  summaries. Times are seconds since a start time shared by all ranks. To keep the
  timeline small, repeats are coalesced: back-to-back intervals of the same phase
  (idle polling, for instance) become one interval, and repeated identical steal
  requests/denials become one event with a ``count``; nested phases are not recorded.
- ``stream_dir`` (``profile_dir``): additionally append every event as one JSON line to
  ``<stream_dir>/rank_<r>.jsonl`` as it happens, so a monitor can follow a running
  (MPI) job by tailing files, with no extra MPI traffic.

``PHASES`` is the single list of phase keys, their display labels and colours, used by
the plots in ``src.plotting`` and by any monitor, so a new phase only needs one entry.
Phases marked ``nested`` run inside another phase (e.g. state polling inside idle
time); they are excluded from stacked totals to avoid double counting.
"""

import json
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import IS_CUPY_BACKEND


@dataclass(frozen=True)
class Phase:
    """A timed phase: its ``profiling_stats`` key, display group and colour."""

    key: str
    group: str
    color: str
    nested: bool = False


# Order matters: it is the stacking order of breakdown plots (bottom to top).
PHASES: tuple[Phase, ...] = (
    Phase("time_init", "Preprocessing", "#4e79a7"),
    Phase("time_scatter", "MPI scatter", "#b07aa1"),
    Phase("time_slicing", "H2D: batch transfer", "#a0cbe8"),
    Phase("time_error_calc", "Fit + error", "#f28e2b"),
    Phase("time_filter", "Fit + error", "#f28e2b"),
    Phase("time_good_Q", "Trust regions + finalize", "#59a14f"),
    Phase("time_good_d2h", "D2H: accepted leaves", "#8cd17d"),
    Phase("time_augment", "Augment points", "#ffbe7d"),
    Phase("time_bad_split", "Split rejected", "#e15759"),
    Phase("time_bad_d2h", "D2H: children", "#ff9d9a"),
    Phase("time_transition", "Level transition", "#bab0ac"),
    Phase("time_gc", "Level transition", "#bab0ac"),
    Phase("time_idle", "Idle", "#9c755f"),
    Phase("time_state_update", "Work stealing", "#d4a6c8", nested=True),
    Phase("time_mpi_send", "Work stealing", "#d4a6c8"),
    Phase("time_mpi_recv", "Work stealing", "#d4a6c8", nested=True),
    Phase("time_gather", "MPI gather", "#76b7b2"),
)
PHASE_BY_KEY = {p.key: p for p in PHASES}

# Events of these kinds that repeat back to back with the same fields are merged.
_COALESCED_KINDS = frozenset({"steal_request", "steal_deny", "steal_denied"})
_COALESCE_GAP = 0.25  # seconds between same-phase intervals that are still merged
# A merged event is finalized (and streamed) once it spans this long, so a live
# monitor sees a long idle stretch within about a second instead of when it ends.
_STREAM_FLUSH_S = 1.0

# Not a phase of the loop: wall time spent inside the user's function (and gradient),
# which overlaps the phases that call it (fit/error, augmentation).
ORACLE_TIME_KEY = "time_oracle"


def phase_groups() -> dict[str, list[str]]:
    """Display group -> phase keys, in stacking order."""
    groups: dict[str, list[str]] = {}
    for p in PHASES:
        groups.setdefault(p.group, []).append(p.key)
    return groups


def group_colors() -> dict[str, str]:
    """Display group -> colour (the colour of its first phase)."""
    colors: dict[str, str] = {}
    for p in PHASES:
        colors.setdefault(p.group, p.color)
    return colors


def new_profiling_stats() -> dict[str, Any]:
    """The per-rank ``profiling_stats`` dict with every counter and phase total at zero."""
    return {
        "N0": 0,
        "num_levels": 0,
        "num_generated_pts": 0,
        "total_f_evals": 0,
        "num_final_clusters": 0,
        "num_unresolved_clusters": 0,
        "num_unresolved_pts": 0,
        "time_total": 0.0,
        ORACLE_TIME_KEY: 0.0,
        **{p.key: 0.0 for p in PHASES},
        "level_stats": [],
    }


_NVTX = IS_CUPY_BACKEND and os.environ.get("AFINNI_NVTX") == "1"


def _nvtx_range(key: str):
    from cupyx.profiler import time_range

    phase = PHASE_BY_KEY.get(key)
    return time_range(phase.group if phase else key)


class Profiler:
    """
    One rank's profiler. Owns that rank's ``stats`` dict (phase totals and
    ``level_stats``) and its event timeline (``stats["events"]`` when enabled).

    Not thread-safe: the refinement loop is single-threaded per rank.
    """

    def __init__(
        self,
        stats: dict[str, Any],
        rank: int = 0,
        timing: bool = False,
        events: bool = False,
        stream_dir: str | Path | None = None,
        t0: float | None = None,
    ):
        self.stats = stats
        self.rank = rank
        self.timing = bool(timing)
        self.events: list[dict[str, Any]] | None = (
            [] if (events or stream_dir is not None) else None
        )
        if self.events is not None:
            stats["events"] = self.events
        self.t0 = time.time() if t0 is None else t0  # shared epoch: see set_epoch
        self.level: int | None = None
        self.batch: int | None = None
        self._stream = None
        self._last: dict[str, Any] | None = None  # last event; may still be merged into
        if stream_dir is not None:
            path = Path(stream_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._stream = open(path / f"rank_{rank:03d}.jsonl", "w", buffering=1)

    @property
    def enabled(self) -> bool:
        return self.timing or self.events is not None

    def set_epoch(self, t0: float) -> None:
        """Use a start time shared by all ranks (``time.time()`` broadcast from rank 0)."""
        self.t0 = t0

    def now(self) -> float:
        return time.time() - self.t0

    def phase(self, key: str):
        """
        Context manager timing phase ``key`` (a no-op when profiling is off). With
        ``AFINNI_NVTX=1`` on the GPU it also opens an NVTX range named after the
        phase, so Nsight Systems timelines are labelled; this adds no synchronization.
        """
        if not self.enabled:
            return _nvtx_range(key) if _NVTX else nullcontext()
        return self._phase(key)

    @contextmanager
    def _phase(self, key: str):
        with _nvtx_range(key) if _NVTX else nullcontext():
            yield from self._timed_phase(key)

    def _timed_phase(self, key: str):
        if IS_CUPY_BACKEND and self.timing:
            import cupy as cp

            cp.cuda.Device().synchronize()
        start = time.perf_counter()
        t_start = self.now() if self.events is not None else 0.0
        try:
            yield
        finally:
            if IS_CUPY_BACKEND and self.timing:
                import cupy as cp

                cp.cuda.Device().synchronize()
            elapsed = time.perf_counter() - start
            if self.timing:
                self.stats[key] = self.stats.get(key, 0.0) + elapsed
            phase = PHASE_BY_KEY.get(key)
            if self.events is not None and not (phase and phase.nested):
                self._emit(
                    {"kind": "phase", "phase": key, "t": t_start, "dur": elapsed,
                     "level": self.level, "batch": self.batch}
                )  # fmt: skip

    def add_time(self, key: str, seconds: float) -> None:
        """Add a duration measured elsewhere (e.g. an MPI transfer) to ``key``'s total."""
        if self.timing:
            self.stats[key] = self.stats.get(key, 0.0) + seconds

    def event(self, kind: str, **fields: Any) -> None:
        """Record an instantaneous event (ignored unless events are enabled)."""
        if self.events is not None:
            self._emit({"kind": kind, "t": self.now(), "level": self.level, **fields})

    def load(self, n_pts: int, n_clusters: int) -> None:
        """Sample this rank's queued work (points and clusters)."""
        self.event("load", n_pts=int(n_pts), n_clusters=int(n_clusters))

    def _emit(self, record: dict[str, Any]) -> None:
        record["rank"] = self.rank
        last = self._last
        if last is not None and self._merge(last, record):
            if self._stream is not None and record["t"] - last["t"] >= _STREAM_FLUSH_S:
                self._write(last)
                self._last = None
            return
        self._write(last)
        self.events.append(record)
        self._last = record
        if record["kind"] in ("level_end", "steal_grant", "steal_received", "retired", "done"):
            self._write(record)  # milestones reach the stream immediately
            self._last = None

    @staticmethod
    def _merge(last: dict[str, Any], record: dict[str, Any]) -> bool:
        """Fold ``record`` into ``last`` if it only repeats it (see the module docstring)."""
        if record["kind"] != last["kind"]:
            return False
        if record["kind"] == "phase":
            same = all(last[k] == record[k] for k in ("phase", "level", "batch"))
            gap = record["t"] - (last["t"] + last["dur"])
            if same and gap <= _COALESCE_GAP:
                last["dur"] = record["t"] + record["dur"] - last["t"]
                last["count"] = last.get("count", 1) + 1
                return True
            return False
        if record["kind"] in _COALESCED_KINDS:
            keys = set(record) - {"t", "waited", "victim_load"}
            if all(last.get(k) == record[k] for k in keys):
                last["count"] = last.get("count", 1) + 1
                last["t_last"] = record["t"]
                return True
        return False

    def _write(self, record: dict[str, Any] | None) -> None:
        if record is not None and self._stream is not None:
            self._stream.write(json.dumps(record) + "\n")

    def close(self) -> None:
        """Flush and close the event stream (idempotent)."""
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                self._stream = stream  # _write needs it
                self._write(self._last)
            finally:
                # A failing final write must still release the file.
                self._stream, self._last = None, None
                stream.close()


def read_event_stream(path: str | Path) -> list[dict[str, Any]]:
    """Events from one ``rank_<r>.jsonl`` stream, skipping a partially written last line."""
    events = []
    with open(path) as fh:
        for line in fh:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return events


def read_event_streams(stream_dir: str | Path) -> dict[int, list[dict[str, Any]]]:
    """Every rank's events from a ``profile_dir``, keyed by rank."""
    out = {}
    for path in sorted(Path(stream_dir).glob("rank_*.jsonl")):
        out[int(path.stem.split("_")[1])] = read_event_stream(path)
    return out


def fmt_count(value: float) -> str:
    """Compact count label (950, 12.3k, 4.1M), shared by the live monitor and the profiling and ART plots."""
    for div, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= div:
            return f"{value / div:.3g}{suffix}"
    return f"{value:.3g}" if value != int(value) else f"{int(value)}"
