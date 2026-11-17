"""
One run of the high-dimensional study in its own process: train one variant on one problem at
one dimension and tolerance, score it on the fixed test set, and write one JSON record.

Run it through ``sweep.py``, which gives every run a fresh process, a time limit and a result
file. Directly (from the repository root)::

    SURROGATE_DEVICE=gpu SURROGATE_MPI=off .venv/bin/python -m studies.high_dim.run_one \
        --variant art_affine --problem rosenbrock --d 40 --tol 0.1 --out /tmp/run.json

The record is written before training starts (``status: "running"``) and replaced at the
end, so a process killed by the kernel or the time limit still leaves its settings behind;
the sweep then fills in the failure. Memory is measured three ways on the GPU: the peak of
CuPy's default pool (an allocator wrapper, exact for every CuPy allocation, including the
cuSOLVER workspaces CuPy allocates), the device's used memory from NVML sampled every 20 ms
(includes the CUDA context and other processes; ``nvml_baseline_bytes`` is subtracted in
``gpu_peak_bytes``), and the process's own entry in NVML where the driver reports it (not
under WSL). Host memory: peak RSS (``getrusage``) and the sampled RSS.
"""

import argparse
import json
import logging
import os
import re
import resource
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

SAMPLE_INTERVAL = 0.02  # seconds between NVML / RSS samples


class TimeLimitError(Exception):
    """Raised in the main thread by SIGALRM when the run exceeds its time limit."""


class ProgressLog(logging.Handler):
    """Keep the deepest refinement level and the last message logged, so a failed run still reports how far it got."""

    LEVEL = re.compile(r"R\d+ L(\d+)")

    def __init__(self):
        super().__init__(logging.INFO)
        self.level_reached, self.last = -1, ""

    def emit(self, record):
        msg = record.getMessage()
        m = self.LEVEL.search(msg)
        if m:
            self.level_reached = max(self.level_reached, int(m.group(1)))
        self.last = msg[:300]


class MemoryMonitor:
    """Peak device and host memory of this process while it runs (see the module docstring)."""

    def __init__(self, gpu: bool):
        import psutil

        self.gpu, self.proc, self.pid = gpu, psutil.Process(), os.getpid()
        self.pool_peak = 0
        self.nvml_peak = self.nvml_baseline = self.nvml_total = self.proc_peak = None
        self.rss_peak = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._handle = None
        if gpu:
            import cupy as cp

            pool = cp.get_default_memory_pool()

            def malloc(size):
                mem = pool.malloc(size)
                total = pool.total_bytes()
                if total > self.pool_peak:
                    self.pool_peak = total
                return mem

            cp.cuda.set_allocator(malloc)
            try:
                import pynvml

                pynvml.nvmlInit()
                index = int(cp.cuda.Device().id)
                self._nvml = pynvml
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.nvml_baseline, self.nvml_total, self.nvml_peak = int(info.used), int(info.total), int(info.used)
            except Exception as e:  # NVML is a diagnostic: a missing library must not fail the run
                logging.getLogger(__name__).warning(f"NVML unavailable ({e}); GPU peak from the CuPy pool only.")
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while not self._stop.is_set():
            try:
                self.rss_peak = max(self.rss_peak, self.proc.memory_info().rss)
                if self._handle is not None:
                    self.nvml_peak = max(self.nvml_peak, int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used))
                    try:
                        for p in self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                            if p.pid == self.pid and p.usedGpuMemory is not None:
                                self.proc_peak = max(self.proc_peak or 0, int(p.usedGpuMemory))
                    except Exception:
                        pass
            except Exception:
                pass
            self._stop.wait(SAMPLE_INTERVAL)

    def report(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=1.0)
        out = {
            "host_rss_peak_bytes": max(self.rss_peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "gpu_pool_peak_bytes": self.pool_peak if self.gpu else None,
            "nvml_peak_bytes": self.nvml_peak, "nvml_baseline_bytes": self.nvml_baseline,
            "nvml_total_bytes": self.nvml_total, "nvml_process_peak_bytes": self.proc_peak,
        }  # fmt: skip
        out["gpu_peak_bytes"] = (
            self.nvml_peak - self.nvml_baseline if self.nvml_peak is not None else (self.pool_peak if self.gpu else None)
        )
        return out


def classify(exc: BaseException) -> str:
    """The failure mode of an exception: gpu_oom, host_oom, time_limit, no_leaf or error."""
    name = type(exc).__name__
    text = str(exc)
    if isinstance(exc, TimeLimitError):
        return "time_limit"
    if name == "OutOfMemoryError" or "CUDA_ERROR_OUT_OF_MEMORY" in text or "cudaErrorMemoryAllocation" in text:
        return "gpu_oom"
    if "CUSOLVER_STATUS_ALLOC_FAILED" in text or "CUBLAS_STATUS_ALLOC_FAILED" in text:
        return "gpu_oom"
    if isinstance(exc, MemoryError):
        return "host_oom"
    if isinstance(exc, RuntimeError) and "no cluster" in text.lower():
        return "no_leaf"
    return "error"


def where(exc: BaseException) -> str:
    """The innermost frame inside the repository's ``src/`` (file:line function), else the innermost frame."""
    frames = traceback.extract_tb(exc.__traceback__)
    inside = [f for f in frames if f"{os.sep}src{os.sep}" in f.filename]
    f = (inside or frames or [None])[-1]
    return "" if f is None else f"{Path(f.filename).name}:{f.lineno} {f.name}"


def tree_depth(model) -> int | None:
    """Deepest leaf: ART from the locational codes (one bit per cut behind a leading 1)."""
    codes = model.art_leaves.get("locational_code") if model.art_leaves else None
    if codes is None or len(codes) == 0:
        return None
    return int(np.max(np.floor(np.log2(np.asarray(codes, dtype=np.float64)))))


def score(model, problem, d: int, n_test: int, atol: float, predict_batch: int, coarse_factor: int) -> dict:
    """Test errors on the fixed test set, relative to the problem's scale (std of f); ``coarse_factor`` applies to AFINNI's kNN prediction only."""
    from src import to_host
    from studies.high_dim.problems import scoring_set

    Xq, Fq, scale = scoring_set(problem, d, n_test)
    t = time.perf_counter()
    pred, outside = model.predict(Xq, coarse_factor=coarse_factor, batch_size=predict_batch)
    t_predict = time.perf_counter() - t
    pred, outside = np.asarray(to_host(pred)), np.asarray(to_host(outside)).astype(bool)
    err = np.max(np.abs(pred - Fq), axis=1)
    out = {
        "scale": scale, "time_predict": t_predict, "n_test": int(n_test),
        "err_max_rel": float(np.max(err) / scale), "err_rms_rel": float(np.sqrt(np.mean(err**2)) / scale),
        "err_p99_rel": float(np.quantile(err, 0.99) / scale), "frac_within_tol": float(np.mean(err < atol)),
        "frac_outside": float(np.mean(outside)),
    }  # fmt: skip
    if outside.any() and not outside.all():
        inside = ~outside
        out["err_max_rel_inside"] = float(np.max(err[inside]) / scale)
        out["frac_within_tol_inside"] = float(np.mean(err[inside] < atol))
    return out


def summarize(model, train_seconds: float) -> dict:
    """What the study records from a trained model's ``profiling_stats``."""
    ps = model.profiling_stats
    phases = {k: float(v) for k, v in ps.items() if k.startswith("time_") and isinstance(v, int | float)}
    levels = [
        {k: e.get(k) for k in ("level", "clusters_start", "good_clusters", "fallback_clusters", "num_pts_next_level",
                               "generated_pts", "runtime_sec")}  # fmt: skip
        for e in ps.get("level_stats", [])
    ]
    out = {
        "time_train": train_seconds, "phases": phases, "leaves": int(model.N_src),
        "f_evals": int(ps.get("total_f_evals", 0)), "levels": int(ps.get("num_levels", 0)),
        "max_depth": tree_depth(model), "converged": bool(model.converged),
        "unresolved_clusters": int(model.num_unresolved_clusters), "unresolved_pts": int(model.num_unresolved_pts),
        "effective_batch_size": ps.get("effective_config", {}).get("batch_size"), "level_stats": levels,
    }  # fmt: skip
    if model.art_leaves:
        out["leaf_status"] = model.leaf_status_counts()
        out["art_counters"] = ps.get("art", {})
    return out


def main(argv=None) -> int:
    from studies.high_dim.problems import PROBLEMS
    from studies.high_dim.settings import VARIANTS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--problem", required=True, choices=sorted(PROBLEMS))
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--tol", type=float, required=True, help="tolerance relative to the problem's scale")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-test", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=0.0, help="seconds; 0 = none (the sweep also enforces it)")
    parser.add_argument("--library-min-pts", action="store_true", help="keep min_pts_per_cluster = 100")
    parser.add_argument("--max-levels", type=int, default=None)
    parser.add_argument("--predict-bytes", type=float, default=5e8,
                        help="bound on the kNN gather per prediction chunk (k_coarse d^2 doubles per query)")  # fmt: skip
    parser.add_argument("--coarse-factor", type=int, default=50,
                        help="AFINNI's kNN candidates per query (kappa); fewer miss elongated trust regions")  # fmt: skip
    args = parser.parse_args(argv)

    for var in ("SURROGATE_DEVICE", "SURROGATE_MPI"):
        if var not in os.environ:
            print(f"Set {var} explicitly (AGENTS.md): a missing GPU or MPI must not pass silently.", file=sys.stderr)
            return 2

    progress = ProgressLog()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger().addHandler(progress)

    from src import IS_CUPY_BACKEND, xp
    from src.core.evaluation import Evaluator
    from src.surrogate import AdaptiveSurrogate
    from studies.high_dim.problems import scoring_set, uniform_points
    from studies.high_dim.settings import (
        MAX_LEVELS,
        build_config,
        n_coefficients,
        n_initial_points,
    )

    variant, problem, d = VARIANTS[args.variant], PROBLEMS[args.problem], args.d
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, _, scale = scoring_set(problem, d, args.n_test)
    atol = args.tol * scale
    record = {
        "variant": variant.name, "algorithm": variant.algorithm, "leaf": variant.leaf, "problem": problem.name,
        "exact": variant.leaf in problem.exact_for, "d": d, "tol": args.tol, "atol": atol, "coarse_factor": args.coarse_factor, "scale": scale,
        "device": "gpu" if IS_CUPY_BACKEND else "cpu", "seed": args.seed, "library_min_pts": args.library_min_pts,
        "n_coefficients": n_coefficients(variant.leaf, d), "status": "running", "started": time.time(),
    }  # fmt: skip
    if IS_CUPY_BACKEND:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        record["gpu_name"] = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    out_path.write_text(json.dumps(record, indent=1))

    monitor = MemoryMonitor(IS_CUPY_BACKEND)
    if args.time_limit > 0:
        def on_alarm(signum, frame):
            raise TimeLimitError(f"time limit of {args.time_limit:.0f} s")

        signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, args.time_limit)
    stage = "setup"
    try:
        config = build_config(variant, d, atol, seed=args.seed, library_min_pts=args.library_min_pts,
                              max_levels=args.max_levels or MAX_LEVELS)  # fmt: skip
        record["config"] = json.loads(json.dumps(config.to_dict(), default=str))
        X0 = uniform_points(n_initial_points(variant, d, args.library_min_pts), d, seed=1000 + args.seed)
        record["n0"] = int(X0.shape[0])
        model = AdaptiveSurrogate()
        stage = "train"
        t = time.perf_counter()
        model.train(xp.asarray(X0), config, Evaluator(f_fn=problem.f))
        record.update(summarize(model, time.perf_counter() - t))
        stage = "predict"
        k_coarse = args.coarse_factor  # k_nn * coarse_factor with predict's k_nn = 1
        predict_batch = int(max(16, min(5000, args.predict_bytes // (8 * k_coarse * d * d))))
        record.update(score(model, problem, d, args.n_test, atol, predict_batch, args.coarse_factor))
        signal.setitimer(signal.ITIMER_REAL, 0)
        record["status"] = "ok"
        record["reached"] = bool(record["converged"])
        if not record["converged"]:
            record["failure"] = "not_converged"
    except BaseException as exc:  # every failure is recorded, including the time limit
        signal.setitimer(signal.ITIMER_REAL, 0)
        record.update(status="failed", reached=False, failure=classify(exc), failure_stage=stage,
                      failure_where=where(exc), failure_message=f"{type(exc).__name__}: {str(exc)[:500]}")  # fmt: skip
        traceback.print_exc()
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        record.update(monitor.report())
        record["level_reached_log"] = progress.level_reached
        record["last_log"] = progress.last
        record["finished"] = time.time()
        out_path.write_text(json.dumps(record, indent=1, default=str))
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
