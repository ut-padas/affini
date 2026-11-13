"""
Watch a multi-rank training run live: 8 MPI ranks on the CPU train a 5-D Rosenbrock
surrogate (2M points, tol 0.007: about 85 s and ~60 work steals on a 22-core laptop)
while streaming profiling events, which ``src.infra.monitor`` displays from another terminal.

    # terminal 1: train (add --stealing off to compare without work stealing)
    uv run mpiexec -n 8 python -m examples.example03_monitoring_afinni
    # terminal 2: watch (exits when every rank is done; add --once for a snapshot)
    uv run python -m src.infra.monitor scratch/example03_monitoring_afinni/stealing/stream

Afterwards, rank 0 saves the profile and every profile plot (dashboard, time
breakdown, level heatmaps, forest growth, timeline, load, steals) next to the stream,
under ``scratch/example03_monitoring_afinni/stealing`` or ``.../no_stealing``, so runs with
and without work stealing can be compared side by side.
Steal thresholds are lowered so work stealing is visible at this size; production
runs keep the defaults.
"""

import os

# Before importing src: run on the CPU, one thread per rank.
os.environ.setdefault("AFINNI_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import src.engine.work_stealing as ws  # noqa: E402
from src import PROJECT_DIR, get_mpi_rank, get_mpi_size, xp  # noqa: E402
from src.components.modes import SurrogateMode  # noqa: E402
from src.config import (  # noqa: E402
    TrainingConfig,  # noqa: E402
    get_MIN_PTS_PER_CLUSTER,
    get_NUM_INITIAL_CLUSTERS,
)
from src.core.evaluation import Evaluator  # noqa: E402
from src.infra.backend import MPI  # noqa: E402
from src.problems.rosenbrock import rosenbrock  # noqa: E402
from src.surrogate import AFINNISurrogate  # noqa: E402

BASE_DIR = PROJECT_DIR / "scratch" / "example03_monitoring_afinni"
N_TEST, SEED = 50_000, 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--dim", type=int, default=5)
    p.add_argument("--n-train", type=int, default=2_000_000)
    p.add_argument("--tol", type=float, default=0.007)
    p.add_argument("--init", default="kmeans", choices=["kmeans", "uniform_grid"])
    # A relative test refines without end where f is near 0 (Rosenbrock's minimum), so
    # the demo uses only an absolute one, ATOL = tol * max|f| over the training data.
    p.add_argument("--metric", default="Linf", choices=["Linf", "L2"])
    p.add_argument("--stealing", default="on", choices=["on", "off"], help="MPI work stealing")
    args = p.parse_args()
    stealing = args.stealing == "on"
    OUT_DIR = BASE_DIR / ("stealing" if stealing else "no_stealing")
    STREAM_DIR = OUT_DIR / "stream"
    DIM, N_TRAIN, TOL = args.dim, args.n_train, args.tol
    rank, size = get_mpi_rank(), get_mpi_size()
    ws.MIN_LOAD_THRESHOLD, ws.STEAL_SIZE = 50_000, 25_000
    f = partial(rosenbrock, a=100.0)

    X_train = Y_train = Q = None
    if rank == 0:
        # Other ranks only create their stream files once training starts, after
        # rank 0 joins it, so clearing an old run here cannot race with them.
        shutil.rmtree(STREAM_DIR, ignore_errors=True)
        rng = np.random.default_rng(SEED)
        X_train = xp.asarray(rng.uniform(-2.0, 3.0, (N_TRAIN, DIM)))
        Y_train = f(X_train)  # passed to train(), so evaluated only once
        Q = xp.asarray(rng.uniform(-2.0, 3.0, (N_TEST, DIM)))
        print(f"Training on {size} rank(s), work stealing {args.stealing}. Watch with:\n"
              f"  uv run python -m src.infra.monitor {Path(STREAM_DIR).relative_to(PROJECT_DIR)}", flush=True)  # fmt: skip

    # Only rank 0 holds the data, but every rank must train with the same tolerance.
    ATOL = MPI.COMM_WORLD.bcast(TOL * float(xp.max(xp.abs(Y_train))) if rank == 0 else None, root=0)

    surrogate = AFINNISurrogate()
    t0 = time.perf_counter()
    config = TrainingConfig(
        mode=SurrogateMode.AFFINE_COV_HESS,
        atol=ATOL,
        error_metric=args.metric,
        initial_clustering_method=args.init,
        num_initial_clusters=get_NUM_INITIAL_CLUSTERS(DIM),
        min_pts_per_cluster=get_MIN_PTS_PER_CLUSTER(DIM),
        batch_size=50_000,  # several batches per level, so mid-level steals can happen
        seed=SEED,
        use_work_stealing=stealing,
        max_levels=1000,  # the same depth limit with and without stealing
        profile_timing=True,
        profile_dir=STREAM_DIR,
    )
    surrogate.train(X_train if rank == 0 else xp.empty((0, DIM)), config, Evaluator(f_fn=f), Y_train=Y_train)
    train_time = time.perf_counter() - t0
    pred, outside = surrogate.predict(Q)  # collective; results on rank 0

    if rank == 0:
        from src import plotting as P

        truth = f(Q).reshape(pred.shape)
        err = float(xp.linalg.norm(pred - truth) / xp.linalg.norm(truth))
        ps = surrogate.profiling_stats
        steals = len(P.steals_frame(ps))
        print(
            f"Trained in {train_time:.1f}s: {surrogate.N_src:,} leaves, {ps['num_levels']} levels, "
            f"{ps['total_f_evals']:,} evaluations, {steals} steals; held-out relative L2 error {err:.2e} "
            f"({float(xp.mean(outside)):.1%} of queries outside every trust region)."
        )
        (OUT_DIR / "profile.json").write_text(json.dumps(ps, default=float))
        # Every profile plot (see docs/USAGE.md#plots); the steal plot needs steals.
        plots = OUT_DIR / "plots"
        P.plot_profile_dashboard(ps, plots / "dashboard.png")
        P.plot_rank_time_breakdown(ps, plots / "time_breakdown.png")
        P.plot_level_heatmap(ps, plots / "level_heatmap_clusters.png", value="clusters_start")
        P.plot_level_heatmap(ps, plots / "level_heatmap_runtime.png", value="runtime_sec", cmap="Blues")
        P.plot_level_progress(ps, plots / "forest_growth.png")
        P.plot_timeline(ps, plots / "timeline.png")
        P.plot_load_over_time(ps, plots / "load_over_time.png")
        if steals:
            P.plot_steal_matrix(ps, plots / "steal_matrix.png")
        print(f"Saved {OUT_DIR / 'profile.json'} and the plots in {plots}")
        print(f"SUMMARY ranks={size} dim={DIM} n={N_TRAIN} tol={TOL} init={args.init} metric={args.metric} stealing={args.stealing} "
              f"train_s={train_time:.1f} steals={steals} leaves={surrogate.N_src}", flush=True)
