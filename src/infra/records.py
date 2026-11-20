"""
Records of long runs: one JSON object per finished case, written atomically the moment the case ends, so a
run that stops resumes at its first unfinished case. The benchmark's store, both companions' convergence
studies and the manuscript study keep their cases through this module.

Every record carries an envelope beside its own fields:

- ``schema``, the record format (``SCHEMA``); ``upgrade`` brings an older record to it when it is read.
- ``status``: ``done``, ``failed`` (with ``failed``, the reason, and ``failure``: type, message, traceback)
  or ``skipped``.
- ``recorded`` (local time, ISO) and ``provenance``: the commit, whether tracked files differed from it, a
  hash of the library's sources, the machine (CPU, cores, memory, GPU; never the host name), the Python and
  library versions, and the environment settings that change a run.

Whether a record still stands is decided by its method alone (``stale``): it is reused unless the method's
results changed after it was written, as the caller's table of deliberate changes declares
(``src.benchmark.table.RESULTS_CHANGED``). An unrelated edit never forces a rerun, so a change to a method's
results must be entered there; ``provenance.code`` still tells which sources wrote a record.

The probes measure what a case costs: ``Probe`` its wall time and peak memory, ``timed_oracle`` the time
spent inside the function being approximated (to tell it from the method's own work), ``per_query_us`` the
inference time per query, ``saved_bytes`` the stored model's size and ``error_stats`` the error
distribution. A record should keep every one of them, so that a later question about a run is answered from
its record rather than by running it again.
"""

import contextlib
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from functools import cache
from importlib import metadata
from pathlib import Path

import numpy as np

from src import PROJECT_DIR

SCHEMA = 2
ENVELOPE = ("key", "schema", "status", "recorded", "provenance", "commit", "migrated")
# The settings that change a run, by name: a wildcard would also catch variables holding paths.
ENVIRONMENT = ("SURROGATE_DEVICE", "SURROGATE_MPI", "SURROGATE_MPI_NP", "SURROGATE_ODE_WORKERS", "OMP_NUM_THREADS",
               "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")  # fmt: skip
LIBRARIES = ("numpy", "scipy", "numba", "cupy-cuda12x", "cupy", "mpi4py", "psutil", "matplotlib")
PROVENANCE = ("commit", "describe", "tracked_changes", "code", "python", "libraries", "system", "machine", "cpu",
              "cores", "ram_gib", "gpu", "environment")  # fmt: skip


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def atomic_write(path: Path, write) -> None:
    """``write(tmp_path)`` then rename into ``path``: concurrent readers and writers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}-{os.getpid()}.tmp{path.suffix}")
    write(tmp)
    tmp.replace(path)


def plain(o):
    """A value JSON cannot hold as itself: a NumPy scalar or a device scalar as a number, anything else as its text."""
    if isinstance(o, np.generic):
        return o.item()
    return float(o) if hasattr(o, "__float__") else str(o)


def write_json(path: Path, obj: dict, indent: int | None = None) -> None:
    """``obj`` as JSON (``plain`` for what JSON cannot hold), atomically."""
    text = json.dumps(obj, indent=indent, default=plain)
    atomic_write(path, lambda tmp: tmp.write_text(text))


def read_json(path: Path) -> dict | None:
    """The object in ``path``, or None when the file is missing or not valid JSON (written by something other than ``write_json``)."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def git(*args: str) -> str | None:
    """``git args`` in the repository, its output; None outside a checkout or when git fails."""
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@cache
def code_hash() -> str:
    """A hash of the library's Python and CUDA sources (16 hex digits): the code a record was written by."""
    h = hashlib.sha256()
    src = PROJECT_DIR / "src"
    for path in [*sorted(src.rglob("*.py")), *sorted(src.rglob("*.cu"))]:
        h.update(str(path.relative_to(PROJECT_DIR)).encode() + b"\0" + path.read_bytes())
    return h.hexdigest()[:16]


@cache
def provenance() -> dict:
    """The commit, the code, the machine and the library versions this process runs with (no host name); computed once."""
    libs = {}
    for name in LIBRARIES:
        with contextlib.suppress(metadata.PackageNotFoundError):
            libs[name] = metadata.version(name)
    cpu = platform.processor() or None
    with contextlib.suppress(OSError, StopIteration):
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.lower().startswith(("model name", "cpu model")))  # fmt: skip
    ram = None
    with contextlib.suppress(ImportError):
        import psutil

        ram = round(psutil.virtual_memory().total / 2**30, 1)
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "describe": git("describe", "--always", "--dirty"),
            "tracked_changes": None if status is None else bool(status), "code": code_hash(),
            "python": platform.python_version(), "libraries": libs, "system": platform.system(), "machine": platform.machine(),
            "cpu": cpu, "cores": os.cpu_count(), "ram_gib": ram, "gpu": gpu(),
            "environment": {k: os.environ[k] for k in ENVIRONMENT if k in os.environ}}  # fmt: skip


def gpu() -> dict | None:
    """The CUDA device in use (name, memory, runtime), or None on the NumPy backend."""
    from src import IS_CUPY_BACKEND

    if not IS_CUPY_BACKEND:
        return None
    import cupy

    props = cupy.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    return {"name": name, "memory_gib": round(props["totalGlobalMem"] / 2**30, 1), "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion()}


def portable(text: str) -> str:
    """``text`` without this machine's paths: the repository, the Python installs, the home folder and the user name."""
    prefixes = {str(PROJECT_DIR): "<repo>", sys.prefix: "<python>", sys.base_prefix: "<python>", str(Path.home()): "~"}
    for prefix in sorted(prefixes, key=len, reverse=True):
        text = text.replace(prefix, prefixes[prefix])
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    return re.sub(rf"/{re.escape(user)}(?=/|\b)", "/<user>", text) if user else text


def failure(err: BaseException) -> dict:
    """What a record keeps of an exception: its type, message and the end of its traceback, without this machine's paths."""
    return {"type": type(err).__name__, "message": portable(str(err))[:2000], "traceback": portable(traceback.format_exc())[-4000:]}


def stale(record: dict, changed: dict, method: str | None = None) -> bool:
    """
    Whether ``record`` predates its method's last change to its results: ``changed`` maps a method to
    (ISO time, what changed), and ``method`` (default: the record's own) names the entry that applies.
    """
    since = changed.get(method or record.get("method"), ("", ""))[0]
    return record.get("recorded", "") < since


def upgrade(record: dict) -> dict:
    """
    ``record`` in the current schema. A schema-1 record (the benchmark's first format) gets its status from
    ``failed``, and a provenance holding only what it recorded, its ``commit`` (``git describe``), every other
    field None: the code and machine that wrote it were not recorded.
    """
    if record.get("schema", 1) >= SCHEMA:
        return record
    known = dict.fromkeys(PROVENANCE) | {"describe": record.get("commit")}
    return {**record, "schema": SCHEMA, "status": record.get("status") or ("failed" if "failed" in record else "done"),
            "provenance": record.get("provenance") or known}  # fmt: skip


def envelope(record: dict) -> dict:
    """``record`` with the current envelope: schema, status, time and this process's provenance."""
    status = record.get("status") or ("failed" if "failed" in record else "done")
    return {**record, "schema": SCHEMA, "status": status, "recorded": now(), "provenance": provenance()}


class RecordStore:
    """
    Records keyed by a dict of identifying fields, one file each under ``root``, at ``path_of(key)`` or, by
    default, at a hash of the key. A record stores its key under ``key``; ``get`` returns a case's own fields.
    """

    def __init__(self, root: Path, path_of=None):
        self.root = Path(root)
        self.path_of = path_of

    def path(self, key: dict) -> Path:
        if self.path_of is not None:
            return self.path_of(key)
        return self.root / f"{hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]}.json"

    def read(self, key: dict) -> dict | None:
        record = read_json(self.path(key))
        return None if record is None else upgrade(record)

    def write(self, key: dict, record: dict, arrays: dict | None = None) -> dict:
        """Record one case; its arrays (``.npz`` beside it) first, so a record on disk always has its arrays. Returns it as written."""
        path = self.path(key)
        if arrays:
            atomic_write(path.with_suffix(".npz"), lambda tmp: np.savez_compressed(tmp, **arrays))
        out = envelope({"key": key, **record})
        write_json(path, out)
        return json.loads(json.dumps(out, default=plain))

    def records(self) -> list[dict]:
        """Every record under ``root``, upgraded, in path order."""
        found = (read_json(p) for p in sorted(self.root.rglob("*.json")))
        return [upgrade(r) for r in found if r is not None]

    def get(self, key: dict, compute, *args, changed: dict | None = None, method: str | None = None) -> dict:
        """
        The case's fields: from its record when there is one that is not ``stale`` (against ``changed``, for
        ``method``, by default the key's), otherwise from ``compute(*args)``, recorded before they are returned.
        Both come back as JSON reads them, so a resumed run produces the same output as an uninterrupted one.
        """
        record = self.read(key)
        if record is None or (changed is not None and stale(record, changed, method or key.get("method"))):
            record = self.write(key, compute(*args))
        return {k: v for k, v in record.items() if k not in ENVELOPE}


class Probe:
    """
    A block's wall time (``seconds``) and peak memory above its start: host (``peak_mb``, the process's resident
    set sampled every 20 ms by a background thread, so a peak shorter than that can be missed) and, on CuPy,
    the device memory pool's size at the end (``device_mb``; the pool keeps what it allocated, so this is its
    peak unless the code freed the pool).
    """

    def __enter__(self) -> "Probe":
        self.seconds = self.peak_mb = self.device_mb = None
        self._start = self._peak = self._rss()
        self._stop = threading.Event()
        self._thread = None
        if self._start is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._peak = max(self._peak, self._rss() or 0)
            self.peak_mb = round((self._peak - self._start) / 2**20, 1)
        from src import IS_CUPY_BACKEND

        if IS_CUPY_BACKEND:
            from src import xp

            self.device_mb = round(xp.get_default_memory_pool().total_bytes() / 2**20, 1)

    @staticmethod
    def _rss() -> int | None:
        try:
            import psutil

            return psutil.Process().memory_info().rss
        except (ImportError, OSError):
            return None

    def _sample(self) -> None:
        while not self._stop.wait(0.02):
            self._peak = max(self._peak, self._rss() or 0)

    def fields(self) -> dict:
        """The record's fields: ``wall_s``, ``peak_host_mb`` and ``peak_device_mb``."""
        return {"wall_s": self.seconds, "peak_host_mb": self.peak_mb, "peak_device_mb": self.device_mb}


def timed_oracle(f):
    """``f`` with a running total of the wall time spent inside it (``total["s"]``) and of the calls (``total["calls"]``)."""
    total = {"s": 0.0, "calls": 0}

    def g(X):
        t0 = time.perf_counter()
        try:
            return f(X)
        finally:
            total["s"] += time.perf_counter() - t0
            total["calls"] += 1

    return g, total


def timed(fn, *args):
    """``fn(*args)`` and its wall time in seconds."""
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def per_query_us(predict, X: np.ndarray):
    """
    ``predict(X)`` and its wall time per query in microseconds, after a warm-up call on a few points, so
    one-time costs (compilation, a first index build) are not charged to the queries.
    """
    predict(X[: min(len(X), 64)])
    out, seconds = timed(predict, X)
    return out, 1e6 * seconds / len(X)


def saved_bytes(save, model) -> int | None:
    """The size of the file ``save(model, path)`` writes: the model's storage. None when it cannot be saved."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = save(model, Path(tmp) / "model.npz")
        except Exception:  # noqa: BLE001  (the size is optional: a case must not fail over it)
            return None
        path = Path(path) if path is not None else Path(tmp) / "model.npz"
        return path.stat().st_size if path.exists() else None


def error_stats(e: np.ndarray, digits: int = 3) -> dict:
    """Per-point errors in units of the tolerance: the worst, 99th and 99.9th percentiles, mean, and share within (%)."""
    e = np.asarray(e, dtype=np.float64)
    if e.size == 0:
        return {"worst": None, "p99": None, "p999": None, "mean": None, "within": None}

    def r(v):
        return float(f"{v:.{digits}g}")

    return {"worst": r(e.max()), "p99": r(np.quantile(e, 0.99)), "p999": r(np.quantile(e, 0.999)), "mean": r(e.mean()),
            "within": round(100 * float(np.mean(e <= 1)), 2)}  # fmt: skip


def data_hash(*arrays) -> str:
    """A hash of float64 arrays (16 hex digits): a case's problem by its box, test points and values, in its key."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]
