"""
The adaptive clustering engine: initial clustering on rank 0 (``initial_clustering``,
also callable on its own), scatter to MPI ranks,
the batched fit / accept / augment / split refinement loop on each rank (with work
stealing between ranks), and the gather of accepted leaves on rank 0.

Scientific choices live in the components of ``ClusteringConfig`` (``src.config``);
this module owns batching, device transfers, buffers, MPI and profiling, so a new
component never has to change it.
"""

import dataclasses
import logging
import time
from typing import Any, Optional

import numpy as np

from src import (
    IS_CUPY_BACKEND,
    Array,
    _to_host,
    xp,
)
from src.config import ClusteringConfig
from src.engine.clusters import ClusterBatch
from src.engine.evaluation import Evaluator
from src.engine.initial import _threads_for_rank0, initial_clustering
from src.engine.io import combine_generated_data, save_generated_data
from src.engine.refinement import _run_local_refinement
from src.engine.work_stealing import (
    recv_clusters,
    send_clusters,
)
from src.infra.backend import (  # MPI: real mpi4py.MPI, or a single-rank stand-in
    MPI,
)
from src.infra.profiling import ORACLE_TIME_KEY, Profiler, new_profiling_stats
from src.numerics import convert_to_2d_array
from src.region import TERMINAL_REASONS, Domain, RegionPlan, Roots, RoutingTable

# Initial scatter of clusters from rank 0 (payload tags SCATTER_TAG + 1..4); kept
# well away from the work-stealing tags.
SCATTER_TAG = 1000
LOCAL_RESULT_TAG = 2000
# One scope (list of profilers) per running run_collective operation, innermost last, so
# a nested operation (e.g. a predict() inside an oracle) only cleans up its own. A stack
# per process: concurrent operations from several threads are not supported.
_PROFILER_SCOPES: list[list] = []  # each rank's merged leaves and profile, sent to rank 0

logger = logging.getLogger(__name__)
TINY_EPS = 1e-12













# ---------------------------------------------------------------------------
# Public entry point (MPI-aware)
# ---------------------------------------------------------------------------
def adaptive_clustering(
    X: Array,
    config: ClusteringConfig,
    evaluator: Evaluator,
    *,
    Y: Optional[Array] = None,
    initial_labels: Optional[Array] = None,
) -> dict[str, Any] | None:
    """
    Train collectively on every MPI rank (see ``_adaptive_clustering``).

    Failure protocol: in a single process an exception propagates unchanged. Under MPI
    an exception on any rank, in any phase (initial evaluation and clustering, scatter,
    refinement, collection), may leave the other ranks blocked in a collective or a
    receive, so the failing rank logs its diagnostic (rank and traceback) and calls
    ``MPI_Abort``, which ends every rank of the job with a non-zero status.
    """
    return run_collective(
        lambda: _adaptive_clustering(X, config, evaluator, Y=Y, initial_labels=initial_labels), "training"
    )


def run_collective(operation, what: str):
    """
    Run a collective ``operation()`` with the MPI failure policy; the single owner of
    fatal-failure handling for the public collective entry points (``adaptive_clustering``,
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


def _adaptive_clustering(
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
    evaluated. ``config`` is copied, so runtime
    changes (the GPU batch-size cap) never reach the
    caller; ``evaluator.n_evals`` keeps counting across calls. The result holds the
    shared columns, the local model's columns, the convergence status and
    ``profiling_stats``.
    """
    config = dataclasses.replace(config)  # local copy; the caller's config is untouched
    region_mode = config.algorithm == "region"
    if region_mode and initial_labels is not None:
        raise ValueError("initial_labels do not apply to algorithm='region': its roots come from region.roots.")
    if not config.requires_no_grads and not evaluator.has_grad:
        raise ValueError(
            f"Selected mode {config.mode.value} requires gradient information. Please provide g_fn or oracle_fn."
        )
    if config.requires_hessian_fn and evaluator.hessian_fn is None:
        raise ValueError(
            f"Selected mode {config.mode.value} requires an exact Hessian. Please provide hessian_fn."
        )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Ranks sharing rank 0's node (collective, so every rank calls it).
    node_ranks = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_size() if size > 1 else 1

    _start_time = time.perf_counter()

    # A start time shared by all ranks, so event times are comparable across ranks.
    t0 = comm.bcast(time.time() if rank == 0 else None, root=0)
    profiling_stats = new_profiling_stats()
    profiler = Profiler(
        profiling_stats,
        rank=rank,
        timing=config.profile_timing,
        events=config.profile_events,
        stream_dir=config.profile_dir,
        t0=t0,
    )
    if _PROFILER_SCOPES:  # closed by the enclosing run_collective if the operation fails
        _PROFILER_SCOPES[-1].append(profiler)
    oracle_seconds_at_start = evaluator.seconds
    timer = profiler.phase

    d_bcast, fd_bcast = 0, 0

    if rank == 0:
        if config.save_all_gen_training_data:
            _, DIM = X.shape
            save_generated_data(
                X, config.generated_data_name(DIM)
            )

        with timer("time_init"):
            X0_gpu = xp.asarray(X, dtype=xp.float64)
            N0, d_bcast = X0_gpu.shape
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

            # Per-point gradients are only carried when the mode uses them at
            # every point; centroid gradients are evaluated during refinement.
            # The N0 initial points always count as evaluations, even with Y.
            need_point_grads = config.requires_grad_everywhere
            if Y is not None:
                F0_gpu = convert_to_2d_array(xp.asarray(Y, dtype=xp.float64))
                G0_gpu = (
                    evaluator(X0_gpu, need_grad=True)[1] if need_point_grads else None
                )
            else:
                F0_gpu, G0_gpu = evaluator(X0_gpu, need_grad=need_point_grads)
            fd_bcast = F0_gpu.shape[1]

            if region_mode:
                domain = Domain.resolve(config.domain, _to_host(X0_gpu))
                labels0 = None
            else:
                labels0 = (
                    initial_clustering(X0_gpu, config, threads=_threads_for_rank0(node_ranks))
                    if initial_labels is None
                    else initial_labels.astype(np.int64)
                )

            if not region_mode:
                sort_idx = np.argsort(labels0)
                labels0_sorted = labels0[sort_idx]
                X0_sorted = _to_host(X0_gpu)[sort_idx]
                F0_sorted = _to_host(F0_gpu)[sort_idx]
                G0_sorted = _to_host(G0_gpu)[sort_idx] if G0_gpu is not None else None

                uniq_labels, all_counts = np.unique(labels0_sorted, return_counts=True)
                counts_chunks = np.array_split(all_counts, size)
        if not region_mode:
            logger.info(
                f"Initial clustering done. Number of clusters: {uniq_labels.shape[0]}."
            )

    plan = None
    if region_mode:
        # Every rank builds the same plan; only the domain (from rank 0's points for
        # "data_bbox") has to be broadcast.
        bounds = comm.bcast((domain.lower, domain.upper) if rank == 0 else None, root=0)
        d_r = comm.bcast(d_bcast if rank == 0 else None, root=0)
        fd_r = comm.bcast(fd_bcast if rank == 0 else None, root=0)
        domain = Domain(*bounds)
        centres = None
        if config.region.roots == "voronoi":
            # The initial clustering's centres (means of its clusters), in normalized
            # coordinates; roots are then assigned by exact nearest centre, not by its labels.
            if rank == 0:
                labels_v = initial_clustering(X0_gpu, config, threads=_threads_for_rank0(node_ranks))
                U0 = _to_host(domain.normalize(X0_gpu))
                inverse = np.unique(labels_v, return_inverse=True)[1]
                centres = np.zeros((int(inverse.max()) + 1, U0.shape[1]))
                np.add.at(centres, inverse, U0)
                centres /= np.bincount(inverse)[:, None]
            centres = comm.bcast(centres, root=0)
        roots = Roots.build(config.region.roots, d_r, config.num_initial_clusters, centres)
        plan = RegionPlan(config, d_r, fd_r, domain, roots)
        if rank == 0:
            all_roots = plan.initial_queue(_to_host(X0_gpu), _to_host(F0_gpu), _to_host(G0_gpu) if G0_gpu is not None else None)
            chunks = np.array_split(np.arange(all_roots.n_clusters), size)
            logger.info(f"Region roots: {roots.n_roots} ({config.region.roots}), domain {domain.lower} .. {domain.upper}.")

    with timer("time_scatter"):
        d_bcast, fd_bcast = comm.bcast((d_bcast, fd_bcast), root=0)
        # Every rank checks, so a tolerance of the wrong length fails before any rank refines.
        config.components().acceptance.check_output_dim(fd_bcast)

        # Rank 0 splits the sorted clusters into one contiguous chunk per rank.
        if rank == 0 and region_mode:
            for r in range(size):
                keep = np.zeros(all_roots.n_clusters, dtype=bool)
                keep[chunks[r]] = True
                chunk = all_roots.select(keep)
                if r == 0:
                    local_clusters = chunk
                else:
                    send_clusters(comm, chunk, dest=r, tag=SCATTER_TAG)
        elif rank == 0:
            current_idx = 0
            for r in range(size):
                num_pts_in_rank = int(np.sum(counts_chunks[r]))
                end_idx = current_idx + num_pts_in_rank
                chunk = ClusterBatch(
                    X0_sorted[current_idx:end_idx],
                    F0_sorted[current_idx:end_idx],
                    G0_sorted[current_idx:end_idx] if G0_sorted is not None else None,
                    counts_chunks[r].astype(np.int64),
                )
                if r == 0:
                    local_clusters = chunk
                else:
                    logger.info(
                        f"R00: Sending initial data chunk to R{r:02} | "
                        f"pts={chunk.n_pts:,} clusters={chunk.n_clusters:,}"
                    )
                    send_clusters(comm, chunk, dest=r, tag=SCATTER_TAG)
                current_idx = end_idx
        else:
            logger.info(f"R{rank:02}: Waiting to receive initial data chunk from R00.")
            local_clusters = recv_clusters(comm, 0, d_bcast, fd_bcast, tag=SCATTER_TAG)

    # Hardware-aware sizing is more appropriate now that the split projection no
    # longer materializes an (N, d, d) tensor. Keep the estimate conservative,
    # though: a rejected batch can still be augmented to roughly 2x its original
    # size and carries several live per-point work arrays during splitting.
    if IS_CUPY_BACKEND:
        import cupy as cp

        # The batch cap below depends on free VRAM, and a different cap changes batch
        # boundaries (and so the random streams): release cached pool blocks first so
        # the reading does not depend on what the pool happens to hold.
        cp.get_default_memory_pool().free_all_blocks()
        free_vram_bytes, total_vram_bytes = cp.cuda.Device().mem_info
        usable_vram_bytes = int(free_vram_bytes * 0.65)
    else:
        free_vram_bytes = total_vram_bytes = 32 * 1024**3
        usable_vram_bytes = free_vram_bytes

    # Approximate live storage during AFFINE_COV_HESS rejected-cluster handling:
    # - base batch X/F/labels
    # - augmented X/F/labels, dx, residual work arrays, split scores/sort buffers
    # - the current tangent residual path still expands A_tangent[label] to
    #   (N, fd, d), so account for that until it is fused as well.
    base_batch_bytes_per_point = (d_bcast + fd_bcast + 1) * 8
    if plan is not None:  # the region plan's own footprint, with the same admission rule
        base_batch_bytes_per_point = plan.bytes_per_point()
    split_augmented_bytes_per_point = (
        3 * d_bcast  # X_aug, dx, selected/source point work
        + fd_bcast * d_bcast
        + 3 * fd_bcast
        + 8  # labels, masks, scalar work arrays, sort/scores
    ) * 8
    augmentation_factor = 2
    estimated_peak_bytes_per_point = (
        base_batch_bytes_per_point
        + augmentation_factor * split_augmented_bytes_per_point
    ) if plan is None else base_batch_bytes_per_point

    hardware_batch_cap = usable_vram_bytes // max(1, estimated_peak_bytes_per_point)
    # AFINNI mode keeps its floor of 50,000 points, so its batches stay bit-identical.
    batch_floor = 50_000 if plan is None else 1_000
    if hardware_batch_cap < batch_floor:
        logger.warning(
            f"Estimated device memory fits only {hardware_batch_cap:,} points per batch "
            f"({estimated_peak_bytes_per_point:,.0f} bytes/point); using the floor of "
            f"{batch_floor:,}, which may run out of memory"
        )
    dynamic_batch_cap = int(np.clip(hardware_batch_cap, batch_floor, 20_000_000))
    if config.batch_size > dynamic_batch_cap:
        logger.info(
            "Hardware VRAM scaling: reducing batch_size from %s to %s for d=%s, "
            "fd=%s using %.2f GB free / %.2f GB total VRAM and an estimated "
            "%.0f bytes/point peak footprint.",
            f"{config.batch_size:,}",
            f"{dynamic_batch_cap:,}",
            d_bcast,
            fd_bcast,
            free_vram_bytes / 1e9,
            total_vram_bytes / 1e9,
            float(estimated_peak_bytes_per_point),
        )
        config.batch_size = dynamic_batch_cap
    # What this run actually used; the caller's (requested) config is unchanged.
    profiling_stats["effective_config"] = {
        "batch_size": int(config.batch_size), "max_levels": int(config.max_levels),
    }  # fmt: skip

    buffers, out_count, local_f_evals, local_gen_pts, last_level = _run_local_refinement(
        config=config,
        evaluator=evaluator,
        clusters=local_clusters,
        comm=comm,
        profiler=profiler,
        plan=plan,
    )
    # Region mode leaves no points unresolved: nodes stopped early keep a fallback leaf,
    # reported separately so the monitor does not show them as queued work.
    unresolved = 0 if plan is not None else int(profiling_stats["num_unresolved_pts"])
    profiler.event(
        "done", leaves=int(out_count), generated=int(local_gen_pts), unresolved_pts=unresolved,
        fallback_pts=int(profiling_stats["num_unresolved_pts"]) if plan is not None else 0,
    )  # fmt: skip

    local_result_host = dict(buffers)
    profiling_stats["num_levels"] = last_level
    profiling_stats["num_generated_pts"] = local_gen_pts
    profiling_stats["total_f_evals"] += local_f_evals
    profiling_stats["num_final_clusters"] = out_count
    profiling_stats["time_total"] = time.perf_counter() - _start_time
    profiling_stats[ORACLE_TIME_KEY] = evaluator.seconds - oracle_seconds_at_start

    active_time = sum(
        profiling_stats.get(k, 0.0)
        for k in ["time_state_update", "time_mpi_send", "time_mpi_recv"]
    )
    profiling_stats["time_idle"] = max(
        0.0, profiling_stats.get("time_idle", 0.0) - active_time
    )
    local_result_host["_profiling_stats"] = profiling_stats

    with timer("time_gather"):
        if rank != 0:
            logger.info(
                f"R{rank:03}: Sending local results ({out_count:,} clusters) to rank 0."
            )
            comm.send(local_result_host, dest=0, tag=LOCAL_RESULT_TAG)
        else:
            logger.info("R00: Merging results from all ranks.")
            all_results = [local_result_host]
            for r in range(1, size):
                all_results.append(comm.recv(source=r, tag=LOCAL_RESULT_TAG))
            logger.info(f"R00: Received local results from {size:,} ranks.")

    if rank == 0:
        all_profiling_stats = [r.pop("_profiling_stats") for r in all_results]
        merged = {
            k: xp.asarray(np.concatenate([r[k] for r in all_results], axis=0))
            for k in all_results[0]
        }

        consolidated_profiling = {
            "time_total": time.perf_counter() - _start_time,
            "t0": t0,
            "N0": all_profiling_stats[0].get("N0", 0),
            "initial_labels_given": all_profiling_stats[0].get("initial_labels_given", False),
            "num_levels": max(ps["num_levels"] for ps in all_profiling_stats),
            "num_final_clusters": int(merged["centroids"].shape[0]),
            "effective_config": all_profiling_stats[0].get("effective_config", {}),
            "level_stats": sorted(
                [entry for ps in all_profiling_stats for entry in ps["level_stats"]],
                key=lambda e: (e["level"], e["rank"]),
            ),
            "per_rank": all_profiling_stats,
        }

        for k in [
            k
            for k in all_profiling_stats[0]
            if k.startswith("time_") and k != "time_total"
        ]:
            consolidated_profiling[k] = sum(
                ps.get(k, 0.0) for ps in all_profiling_stats
            )
        for k in [
            "num_generated_pts",
            "total_f_evals",
            "num_unresolved_clusters",
            "num_unresolved_pts",
        ]:
            consolidated_profiling[k] = sum(ps.get(k, 0) for ps in all_profiling_stats)
        consolidated_profiling["converged"] = (
            consolidated_profiling["num_unresolved_clusters"] == 0
        )
        consolidated_profiling["plan"] = all_profiling_stats[0].get("plan")
        if plan is not None:
            nodes = {k[len("node:"):]: _to_host(merged.pop(k)) for k in list(merged) if k.startswith("node:")}
            routing = RoutingTable.build(
                plan.domain, plan.roots, merged["root"], merged["path_code"],
                nodes["root"], nodes["path_code"], nodes["w"], nodes["z"],
            )  # fmt: skip
            merged["routing"] = routing.arrays()
            reasons = np.bincount(_to_host(merged["terminal_reason"]).astype(np.int64), minlength=len(TERMINAL_REASONS))
            consolidated_profiling["terminal_reasons"] = dict(zip(TERMINAL_REASONS, map(int, reasons), strict=True))
            consolidated_profiling["num_split_nodes"] = int(nodes["root"].shape[0])
            consolidated_profiling["region"] = {
                k: sum(ps.get("region", {}).get(k, 0) for ps in all_profiling_stats) for k in plan.counters
            }

        merged.update(
            {
                "num_levels": consolidated_profiling["num_levels"],
                "num_generated_pts": consolidated_profiling["num_generated_pts"],
                "total_f_evals": consolidated_profiling["total_f_evals"],
                "converged": consolidated_profiling["converged"],
                "num_unresolved_clusters": consolidated_profiling["num_unresolved_clusters"],
                "num_unresolved_pts": consolidated_profiling["num_unresolved_pts"],
                "profiling_stats": consolidated_profiling,
            }
        )

        # Combine all the files if we are saving
        if config.save_all_gen_training_data:
            combine_generated_data(
                config.generated_data_name(d_bcast)
            )

        profiler.close()
        return merged
    profiler.close()
    return None
