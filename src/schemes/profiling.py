"""
Profiling and logging of a scheme's training: named phases on ``infra.profiling.Profiler``, one record per
round, and the ``TrainingReport`` a trained model carries.

Every scheme times the same phase groups (``PHASES``), so schemes compare stage by stage: ``explore``
(directions, edges, pilot gradients), ``sample`` (choosing points), ``fit``, ``check`` (judging the model),
``adapt`` (deciding what to refine) and ``compress``. The oracle's time is the ``Oracle``'s own count and
overlaps the phases that call it. Phases are coarse (one per stage or round, never per point or fibre), so
timing is always on. With ``stream_dir`` the profiler also streams every phase and round as one JSON line
(``<stream_dir>/rank_000.jsonl``), for a monitor to follow a run; see TODO.md for the monitor.

A round is a scheme's unit of refinement (a sweep, a budget doubling, a batch of checks); ``round_end``
logs one line at INFO and records the round's fields in the report.
"""

import logging
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.infra.profiling import Profiler

PHASES = ("explore", "sample", "fit", "check", "adapt", "compress")
RESERVED = ("kind", "t", "level", "rank", "scheme")  # fields every streamed event carries itself


@dataclass
class TrainingReport:
    """What a training run cost and did: oracle use, wall time per phase, and one record per round."""

    scheme: str
    settings: dict = field(default_factory=dict)
    evals: int = 0  # points the oracle evaluated
    jacobians: int = 0  # points at which the oracle's Jacobian was evaluated
    seconds: float = 0.0  # wall time of the whole training
    oracle_seconds: float = 0.0  # of which inside the oracle
    phases: dict = field(default_factory=dict)  # phase -> seconds (the oracle's time included)
    rounds: list = field(default_factory=list)  # one dict per round, in order

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """One line: evaluations, time and its split over the phases and the oracle."""
        parts = ", ".join(f"{k} {v:.2f} s" for k, v in self.phases.items() if v > 0)
        return (f"{self.scheme}: {self.evals:,} evaluations, {self.jacobians:,} Jacobians, {self.seconds:.2f} s "
                f"(oracle {self.oracle_seconds:.2f} s; {parts}), {len(self.rounds)} rounds")  # fmt: skip


class Training:
    """
    One scheme's training run: times its phases, logs and records its rounds, and on ``finish`` attaches
    the ``TrainingReport`` to the model. Single-threaded, like the scheme it serves.
    """

    def __init__(self, scheme: str, settings, oracle, stream_dir: str | Path | None = None):
        self.scheme, self.oracle = scheme, oracle
        self.logger = logging.getLogger(f"src.schemes.{scheme}")
        self.stats: dict[str, Any] = dict.fromkeys(PHASES, 0.0)
        self.profiler = Profiler(self.stats, rank=0, timing=True, events=stream_dir is not None, stream_dir=stream_dir)
        self.report = TrainingReport(scheme, settings.to_dict() if hasattr(settings, "to_dict") else dict(settings or {}))
        self.start = time.perf_counter()
        self.profiler.event("start", scheme=scheme)

    def phase(self, key: str) -> AbstractContextManager:
        """Time one stage under the phase group ``key`` (one of ``PHASES``)."""
        if key not in PHASES:
            raise ValueError(f"unknown phase {key!r}; expected one of {PHASES}")
        return self.profiler.phase(key)

    def round_end(self, **fields: Any) -> None:
        """Close a round: its fields (plain numbers and strings) go to the report, the log and the event stream."""
        if clash := sorted(set(fields) & set(RESERVED)):
            raise ValueError(f"round fields {clash} are reserved for the event stream")
        record = {"round": len(self.report.rounds), "evals": self.oracle.evals, **fields}
        self.report.rounds.append(record)
        self.profiler.level = record["round"]
        self.profiler.event("round_end", scheme=self.scheme, **record)
        self.logger.info("%s round %d: %s", self.scheme, record["round"],
                         ", ".join(f"{k} {_fmt(v)}" for k, v in record.items() if k != "round"))  # fmt: skip

    def finish(self, model):
        """Fill the report from the oracle and the phase totals, attach it as ``model.report`` and return the model."""
        r = self.report
        r.evals, r.jacobians, r.oracle_seconds = self.oracle.evals, self.oracle.jacobians, self.oracle.seconds
        r.seconds = time.perf_counter() - self.start
        r.phases = {key: float(self.stats[key]) for key in PHASES}
        self.profiler.event("done", scheme=self.scheme, evals=r.evals, seconds=r.seconds)
        self.profiler.close()
        model.report = r
        self.logger.info("%s", r.summary())
        return model


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)
