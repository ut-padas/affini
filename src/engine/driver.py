"""
The refinement engine's driver: ``refine`` trains either algorithm plan (``AFINNIPlan`` or
``ARTPlan``, chosen by ``config.algorithm``) collectively on every MPI rank. Rank 0 evaluates
the training points and builds the initial queue (AFINNI's initial clustering, or ART's
roots), the queue is scattered to the ranks, each rank runs the batched level-by-level loop
(``src.engine.refinement.refine_rank``, with work stealing between ranks), and rank 0 gathers
and merges the leaves.

Scientific choices live in the plans and the components of ``ClusteringConfig``
(``src.config``); the engine owns batching, device transfers, buffers, MPI and profiling, so a
new component never has to change it.
"""

import dataclasses
import logging
import time
from typing import Any, Optional

import numpy as np

from src import IS_CUPY_BACKEND, Array, _to_host, xp
from src.afinni import AFINNIPlan
from src.art import ARTPlan, Domain, Roots
from src.config import ClusteringConfig
from src.core.clusters import ClusterBatch
from src.core.evaluation import Evaluator
from src.engine.initial import _threads_for_rank0, initial_clustering
from src.engine.refinement import refine_rank
from src.engine.work_stealing import recv_clusters, send_clusters
from src.infra.backend import MPI  # real mpi4py.MPI, or a single-rank stand-in
from src.infra.io import combine_generated_data, save_generated_data
from src.infra.profiling import ORACLE_TIME_KEY, Profiler, new_profiling_stats
from src.numerics import convert_to_2d_array

# Initial scatter of clusters from rank 0 (payload tags SCATTER_TAG + 1..4); kept
# well away from the work-stealing tags.
SCATTER_TAG = 1000
LOCAL_RESULT_TAG = 2000  # each rank's merged leaves and profile, sent to rank 0
# One scope (list of profilers) per running run_collective operation, innermost last, so
# a nested operation (e.g. a predict() inside an oracle) only cleans up its own. A stack
# per process: concurrent operations from several threads are not supported.
_PROFILER_SCOPES: list[list] = []

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point (MPI-aware)
# ---------------------------------------------------------------------------
def refine(
    X: Array,
    config: ClusteringConfig,
    evaluator: Evaluator,
    *,
    Y: Optional[Array] = None,
    initial_labels: Optional[Array] = None,
) -> dict[str, Any] | None:
    """
    Train ``config.algorithm`` collectively on every MPI rank (see ``_refine``).

    Failure protocol: in a single process an exception propagates unchanged. Under MPI
    an exception on any rank, in any phase (initial evaluation and clustering, scatter,
    refinement, collection), may leave the other ranks blocked in a collective or a
    receive, so the failing rank logs its diagnostic (rank and traceback) and calls
    ``MPI_Abort``, which ends every rank of the job with a non-zero status.
    """
    return run_collective(
        lambda: _refine(X, config, evaluator, Y=Y, initial_labels=initial_labels), "training"
    )


def run_collective(operation, what: str):
    """
    Run a collective ``operation()`` with the MPI failure policy; the single owner of
    fatal-failure handling for the public collective entry points (``refine``,
    ``AFINNISurrogate.train`` and ``predict``), so wrappers are never nested.

    In a single process the original exception propagates unchanged. Under MPI an
    exception on any rank may leave peers blocked in a collective or receive, so no
    further collective is attempted: the rank logs a best-effort diagnostic (rank,
    phase, traceback) and calls ``MPI_Abort`` in ``finally``, so a logging error cannot
    skip it. Only ``Exception`` is handled; ``KeyboardInterrupt``/``SystemExit`` are left
    to the MPI launcher. A logging handler that blocks forever is out of scope.
    """
    comm = MPI.COMM_WORLD
    scope: list = []
    _PROFILER_SCOPES.append(scope)
    try:
        return operation()  # on success the operation closed its profilers itself
    except Exception:
        # Release the profiler streams this operation opened (not a parent's); a cleanup
        # error must neither replace the original exception nor prevent the abort below.
        for profiler in scope:
            try:
                profiler.close()
            except Exception:
                pass
        if comm.Get_size() > 1:
            try:
                logger.critical(f"R{comm.Get_rank():02}: {what} failed; aborting all ranks.", exc_info=True)
            finally:
                comm.Abort(1)
        raise
    finally:
        # Equal empty lists still belong to different calls; unwind by stack position.
        _PROFILER_SCOPES.pop()


def _refine(
    X: Array,
    config: ClusteringConfig,
    evaluator: Evaluator,
    *,
    Y: Optional[Array] = None,
    initial_labels: Optional[Array] = None,
) -> dict[str, Any] | None:
    """
    Train collectively on every MPI rank; return the merged leaves on rank 0, None elsewhere.

    Only rank 0's ``X``, ``Y`` (a precomputed f(X)) and ``initial_labels`` are read;
    the initial points count as evaluations even when ``Y`` is given.
    ``initial_labels`` (one non-negative integer per row of ``X``, e.g. from
    ``initial_clustering``) replaces the initial clustering, which is then recorded
    as ``profiling_stats["initial_labels_given"]``; they are checked before ``f`` is
    evaluated. ``config`` is copied, so runtime changes (the GPU batch-size cap) never
    reach the caller; ``evaluator.n_evals`` keeps counting across calls. The result holds
    the shared columns, the local model's columns, the convergence status and
    ``profiling_stats``.
    """
    config = dataclasses.replace(config)  # local copy; the caller's config is untouched
    _check_inputs(config, evaluator, initial_labels)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    # Ranks sharing rank 0's node (collective, so every rank calls it).
    node_ranks = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_size() if size > 1 else 1
    start_time = time.perf_counter()
    # A start time shared by all ranks, so event times are comparable across ranks.
    t0 = comm.bcast(time.time() if rank == 0 else None, root=0)
    profiling_stats = new_profiling_stats()
    profiler = Profiler(
        profiling_stats, rank=rank, timing=config.profile_timing, events=config.profile_events,
        stream_dir=config.profile_dir, t0=t0,
    )  # fmt: skip
    if _PROFILER_SCOPES:  # closed by the enclosing run_collective if the operation fails
        _PROFILER_SCOPES[-1].append(profiler)
    oracle_seconds_at_start = evaluator.seconds

    data = None
    if rank == 0:
        if config.save_all_gen_training_data:
            save_generated_data(X, config.generated_data_name(X.shape[1]))
        with profiler.phase("time_init"):
            data = _initial_data(X, Y, initial_labels, config, evaluator, profiling_stats, node_ranks)
    d, fd = comm.bcast((data[0].shape[1], data[1].shape[1]) if rank == 0 else None, root=0)
    # Every rank checks, so a tolerance of the wrong length fails before any rank refines.
    config.components().acceptance.check_output_dim(fd)
    plan = _make_plan(config, comm, data[0] if rank == 0 else None, d, fd, node_ranks)

    with profiler.phase("time_scatter"):
        queue = plan.initial_queue(*(_to_host(a) if a is not None else None for a in data[:3]), data[3]) if rank == 0 else None
        local_clusters = _scatter(comm, queue, d, fd)
        del queue
    # Rank 0's device copies of the training data stay alive through the free-memory reading,
    # so the cap (and with it the batch boundaries) is the same as when they were kept for the
    # whole run; afterwards they only cost memory.
    _cap_batch_size(config, plan, d, fd)
    del data
    # What this run actually used; the caller's (requested) config is unchanged.
    profiling_stats["effective_config"] = {
        "batch_size": int(config.batch_size), "max_levels": int(config.max_levels),
    }  # fmt: skip

    buffers, out_count, local_f_evals, local_gen_pts, last_level = refine_rank(
        config=config, evaluator=evaluator, clusters=local_clusters, comm=comm, profiler=profiler, plan=plan,
    )  # fmt: skip
    # ART leaves no points unresolved: nodes stopped early keep a fallback leaf,
    # reported separately so the monitor does not show them as queued work.
    left = int(profiling_stats["num_unresolved_pts"])
    profiler.event(
        "done", leaves=int(out_count), generated=int(local_gen_pts),
        unresolved_pts=0 if plan.keeps_fallback_leaves else left, fallback_pts=left if plan.keeps_fallback_leaves else 0,
    )  # fmt: skip
    profiling_stats["num_levels"] = last_level
    profiling_stats["num_generated_pts"] = local_gen_pts
    profiling_stats["total_f_evals"] += local_f_evals
    profiling_stats["num_final_clusters"] = out_count
    profiling_stats["time_total"] = time.perf_counter() - start_time
    profiling_stats[ORACLE_TIME_KEY] = evaluator.seconds - oracle_seconds_at_start
    active_time = sum(profiling_stats.get(k, 0.0) for k in ["time_state_update", "time_mpi_send", "time_mpi_recv"])
    profiling_stats["time_idle"] = max(0.0, profiling_stats.get("time_idle", 0.0) - active_time)
    local_result = {**buffers, "_profiling_stats": profiling_stats}

    with profiler.phase("time_gather"):
        if rank != 0:
            logger.info(f"R{rank:03}: Sending local results ({out_count:,} clusters) to rank 0.")
            comm.send(local_result, dest=0, tag=LOCAL_RESULT_TAG)
        else:
            logger.info("R00: Merging results from all ranks.")
            all_results = [local_result] + [comm.recv(source=r, tag=LOCAL_RESULT_TAG) for r in range(1, size)]
            logger.info(f"R00: Received local results from {size:,} ranks.")
    merged = None
    if rank == 0:
        merged = _merge_results(all_results, plan, start_time, t0)
        if config.save_all_gen_training_data:
            combine_generated_data(config.generated_data_name(d))
    profiler.close()
    return merged


def _check_inputs(config: ClusteringConfig, evaluator: Evaluator, initial_labels) -> None:
    """Reject settings the evaluator cannot serve before any rank evaluates or communicates."""
    if config.algorithm == "art" and initial_labels is not None:
        raise ValueError("initial_labels do not apply to algorithm='art': its roots come from art.roots.")
    if not config.requires_no_grads and not evaluator.has_grad:
        raise ValueError(
            f"Selected mode {config.mode.value} requires gradient information. Please provide g_fn or oracle_fn."
        )
    if config.requires_hessian_fn and evaluator.hessian_fn is None:
        raise ValueError(
            f"Selected mode {config.mode.value} requires an exact Hessian. Please provide hessian_fn."
        )


def _initial_data(X, Y, initial_labels, config: ClusteringConfig, evaluator: Evaluator, profiling_stats: dict,
                  node_ranks: int) -> tuple[Array, Array, Array | None, np.ndarray | None]:  # fmt: skip
    """
    Rank 0: the training points with their values (and per-point gradients when the mode
    carries them) on the device, and AFINNI's initial cluster labels (None in ART,
    whose roots come from the plan). The labels are checked before ``f`` is evaluated.
    """
    X0 = xp.asarray(X, dtype=xp.float64)
    N0 = X0.shape[0]
    profiling_stats["N0"], profiling_stats["total_f_evals"] = N0, N0
    if initial_labels is not None:
        initial_labels = np.asarray(_to_host(initial_labels))
        valid = initial_labels.shape == (N0,) and np.issubdtype(initial_labels.dtype, np.integer)
        if not valid or (N0 and initial_labels.min() < 0):
            raise ValueError(
                f"initial_labels must be {N0} non-negative integers, one per training point; "
                f"got shape {initial_labels.shape} and dtype {initial_labels.dtype}."
            )
    profiling_stats["initial_labels_given"] = initial_labels is not None

    # Per-point gradients are only carried when the mode uses them at every point;
    # centroid gradients are evaluated during refinement. The N0 initial points always
    # count as evaluations, even with Y.
    need_point_grads = config.requires_grad_everywhere
    if Y is not None:
        F0 = convert_to_2d_array(xp.asarray(Y, dtype=xp.float64))
        G0 = evaluator(X0, need_grad=True)[1] if need_point_grads else None
    else:
        F0, G0 = evaluator(X0, need_grad=need_point_grads)
    labels = None
    if config.algorithm != "art":
        labels = (
            initial_clustering(X0, config, threads=_threads_for_rank0(node_ranks))
            if initial_labels is None
            else initial_labels.astype(np.int64)
        )
    return X0, F0, G0, labels


def _make_plan(config: ClusteringConfig, comm, X0: Array | None, d: int, fd: int, node_ranks: int):
    """
    The same plan on every rank (collective). ART broadcasts the domain (from rank
    0's points for ``domain="data_bbox"``) and, for Voronoi roots, the initial clustering's
    centres in normalized coordinates; roots are then assigned by exact nearest centre, not
    by the clustering's labels.
    """
    rank = comm.Get_rank()
    if config.algorithm != "art":
        return AFINNIPlan(config, d, fd, rank)
    bounds = None
    if rank == 0:
        domain = Domain.resolve(config.domain, _to_host(X0))
        bounds = (domain.lower, domain.upper)
    domain = Domain(*comm.bcast(bounds, root=0))
    centres = None
    if config.art.roots == "voronoi":
        if rank == 0:
            labels_v = initial_clustering(X0, config, threads=_threads_for_rank0(node_ranks))
            U0 = _to_host(domain.normalize(X0))
            inverse = np.unique(labels_v, return_inverse=True)[1]
            centres = np.zeros((int(inverse.max()) + 1, U0.shape[1]))
            np.add.at(centres, inverse, U0)
            centres /= np.bincount(inverse)[:, None]
        centres = comm.bcast(centres, root=0)
    roots = Roots.build(config.art.roots, d, config.num_initial_clusters, centres)
    if rank == 0:
        logger.info(f"ART roots: {roots.n_roots} ({config.art.roots}), domain {domain.lower} .. {domain.upper}.")
    return ARTPlan(config, d, fd, domain, roots)


def _scatter(comm, queue: ClusterBatch | None, d: int, fd: int) -> ClusterBatch:
    """Rank 0 cuts its whole initial ``queue`` into one contiguous run of clusters per rank (views, not copies)."""
    rank, size = comm.Get_rank(), comm.Get_size()
    if rank != 0:
        logger.info(f"R{rank:02}: Waiting to receive initial data chunk from R00.")
        return recv_clusters(comm, 0, d, fd, tag=SCATTER_TAG)
    counts = [len(c) for c in np.array_split(np.arange(queue.n_clusters), size)]
    rest, local = queue, None
    for r, count in enumerate(counts):
        if count < rest.n_clusters:
            chunk, rest = rest.split(count)
        else:  # the last chunk, then empty ones (with the same columns) for ranks beyond the clusters
            chunk, rest = rest, rest.select(np.zeros(rest.n_clusters, dtype=bool))
        if r == 0:
            local = chunk
        else:
            logger.info(f"R00: Sending initial data chunk to R{r:02} | pts={chunk.n_pts:,} clusters={chunk.n_clusters:,}")
            send_clusters(comm, chunk, dest=r, tag=SCATTER_TAG)
    return local


def _cap_batch_size(config: ClusteringConfig, plan, d: int, fd: int) -> None:
    """
    Lower ``config.batch_size`` to what 65% of the free device memory holds at the plan's
    ``bytes_per_point``, but not below ``plan.batch_floor``. The cap sets batch boundaries,
    and with them the random streams, so it must be computed the same way every run.
    """
    if IS_CUPY_BACKEND:
        import cupy as cp

        # Release cached pool blocks first, so the reading does not depend on what the
        # pool happens to hold.
        cp.get_default_memory_pool().free_all_blocks()
        free_vram_bytes, total_vram_bytes = cp.cuda.Device().mem_info
        usable_vram_bytes = int(free_vram_bytes * 0.65)
    else:
        free_vram_bytes = total_vram_bytes = 32 * 1024**3
        usable_vram_bytes = free_vram_bytes
    bytes_per_point = plan.bytes_per_point()
    hardware_batch_cap = usable_vram_bytes // max(1, bytes_per_point)
    if hardware_batch_cap < plan.batch_floor:
        logger.warning(
            f"Estimated device memory fits only {hardware_batch_cap:,} points per batch "
            f"({bytes_per_point:,.0f} bytes/point); using the floor of "
            f"{plan.batch_floor:,}, which may run out of memory"
        )
    dynamic_batch_cap = int(np.clip(hardware_batch_cap, plan.batch_floor, 20_000_000))
    if config.batch_size > dynamic_batch_cap:
        logger.info(
            f"Hardware VRAM scaling: reducing batch_size from {config.batch_size:,} to {dynamic_batch_cap:,} "
            f"for d={d}, fd={fd} using {free_vram_bytes / 1e9:.2f} GB free / {total_vram_bytes / 1e9:.2f} GB "
            f"total VRAM and an estimated {float(bytes_per_point):.0f} bytes/point peak footprint."
        )
        config.batch_size = dynamic_batch_cap


def _merge_results(all_results: list[dict], plan, start_time: float, t0: float) -> dict[str, Any]:
    """Rank 0: concatenate every rank's leaves (rank order) and consolidate their profiles."""
    per_rank = [r.pop("_profiling_stats") for r in all_results]
    merged = {k: xp.asarray(np.concatenate([r[k] for r in all_results], axis=0)) for k in all_results[0]}
    profile = {
        "time_total": time.perf_counter() - start_time,
        "t0": t0,
        "N0": per_rank[0].get("N0", 0),
        "initial_labels_given": per_rank[0].get("initial_labels_given", False),
        "num_levels": max(ps["num_levels"] for ps in per_rank),
        "num_final_clusters": int(merged["centroids"].shape[0]),
        "effective_config": per_rank[0].get("effective_config", {}),
        "level_stats": sorted([e for ps in per_rank for e in ps["level_stats"]], key=lambda e: (e["level"], e["rank"])),
        "per_rank": per_rank,
    }
    for k in [k for k in per_rank[0] if k.startswith("time_") and k != "time_total"]:
        profile[k] = sum(ps.get(k, 0.0) for ps in per_rank)
    for k in ["num_generated_pts", "total_f_evals", "num_unresolved_clusters", "num_unresolved_pts"]:
        profile[k] = sum(ps.get(k, 0) for ps in per_rank)
    profile["converged"] = profile["num_unresolved_clusters"] == 0
    profile["plan"] = per_rank[0].get("plan")
    plan.merge_results(merged, profile, per_rank)
    merged.update({
        k: profile[k] for k in (
            "num_levels", "num_generated_pts", "total_f_evals", "converged", "num_unresolved_clusters", "num_unresolved_pts",
        )
    })  # fmt: skip
    merged["profiling_stats"] = profile
    return merged
