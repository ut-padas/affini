"""
End-to-end work-stealing check on a real training run. Launched under an MPI
launcher by tests/mpi/test_mpi.py.

At the default MIN_LOAD_THRESHOLD (1,000,000 points) no steal ever happens at
test size, so the thresholds are lowered here. For each of a few seeds this
trains the Rosenbrock smoke surrogate with work stealing on and checks that:

- every steal request received exactly one reply, and every grant and denial was
  consumed (an unconsumed message kills ranks at MPI_Finalize);
- nothing is still in flight once all ranks have finished;
- accepted cluster totals are consistent and centroids are unique.

Point conservation is checked separately by _work_stealing_worker.py.

Unlike tests/mpi/_work_stealing_worker.py (which accepts every cluster at level 0
and so only reaches the mid-level grant path), this test starts with a single
cluster on rank 0. Its first refinement evaluation waits for a steal request,
which stays pending until the level boundary. This forces a boundary grant
without relying on relative rank speeds.
"""

import json
import logging

import numpy as np

from tests.mpi._mpi_worker import RESULT_PREFIX

SEEDS = (0, 1, 2)
N_TRAIN, TOL = 40_000, 0.05


def main() -> None:
    import src
    import src.engine.driver as driver
    import src.engine.work_stealing as ws
    from src.problems.rosenbrock import rosenbrock
    from src.surrogate import AFINNISurrogate
    from tests._helpers import train_surrogate

    real_mpi = driver.MPI
    world = real_mpi.COMM_WORLD
    rank = world.Get_rank()
    req, grant, deny = ws.TAG_STEAL_REQ, ws.TAG_STEAL_GRANT, ws.TAG_STEAL_DENY
    sent_name = {req: "req_sent", grant: "grant_sent", deny: "deny_sent"}
    recv_name = {req: "req_recv", grant: "grant_recv", deny: "deny_recv"}
    counts = dict.fromkeys(
        [*sent_name.values(), *recv_name.values(), "boundary_grants"], 0
    )

    class GrantLog(logging.Handler):
        def emit(self, record):
            if "Granted steal to" in record.getMessage():
                counts["boundary_grants"] += 1

    logger = logging.getLogger("src.engine.work_stealing")
    logger.setLevel(logging.INFO)
    logger.addHandler(GrantLog())
    logger.propagate = False

    class CountingComm:
        """Delegates to the real communicator; counts steal-protocol messages."""

        def __getattr__(self, name):
            return getattr(world, name)

        def send(self, obj, dest, tag=0):
            if tag in sent_name:
                counts[sent_name[tag]] += 1
            return world.send(obj, dest=dest, tag=tag)

        def recv(self, *args, **kwargs):
            obj = world.recv(*args, **kwargs)
            if kwargs.get("tag") in recv_name:
                counts[recv_name[kwargs["tag"]]] += 1
            return obj

    class MPIProxy:
        COMM_WORLD = CountingComm()

        def __getattr__(self, name):
            return getattr(real_mpi, name)

    driver.MPI = MPIProxy()
    ws.MIN_LOAD_THRESHOLD = 1_000  # default 1,000,000 never triggers at test size
    ws.STEAL_SIZE = 2_000  # partial steals leave work on the victim

    runs, status = [], real_mpi.Status()
    for seed in SEEDS:
        counts.update(dict.fromkeys(counts, 0))
        rng = np.random.default_rng(seed)
        X = (
            src.xp.asarray(rng.uniform(-2.0, 3.0, (N_TRAIN, 2)))
            if rank == 0
            else src.xp.empty((0, 2))
        )
        boundary_request_queued = False

        def evaluate(x):
            nonlocal boundary_request_queued
            if rank == 0 and len(x) < N_TRAIN and not boundary_request_queued:
                # Initial function evaluation is on all N_TRAIN points. This
                # first centroid evaluation runs after mid-level request polling.
                # With one batch, a new request will be handled at the boundary.
                world.Probe(source=real_mpi.ANY_SOURCE, tag=req)
                boundary_request_queued = True
            return rosenbrock(x, a=100.0)

        surrogate = AFINNISurrogate()
        train_surrogate(
            X, surrogate,
            f_fn=evaluate,
            atol=TOL,
            rtol=TOL,
            error_metric="Linf",
            num_initial_clusters=1,
            batch_size=10_000_000,
            profile_timing=True,
            min_pts_per_cluster=169,
            use_work_stealing=True,
        )

        # Everything the protocol sent must have been received by now.
        world.Barrier()
        stray = []
        if world.Iprobe(
            source=real_mpi.ANY_SOURCE, tag=real_mpi.ANY_TAG, status=status
        ):
            stray.append(status.Get_tag())
            # Report it; draining raw buffers safely is not this test's job.
        world.Barrier()

        per_rank = world.allgather((dict(counts), stray))
        if rank == 0:
            total = {k: sum(c[k] for c, _ in per_rank) for k in counts}
            centroids = np.asarray(src.to_host(surrogate.X_src))
            stats = surrogate.profiling_stats.get("per_rank", [])
            accepted = sum(
                e.get("good_clusters", 0)
                for r in stats
                for e in r.get("level_stats", [])
            )
            runs.append({
                "seed": seed,
                **total,
                "stray_tags": sorted({t for _, s in per_rank for t in s}),
                "n_clusters": int(surrogate.N_src),
                "unique_centroids": int(np.unique(np.round(centroids, 9), axis=0).shape[0]),
                "accepted_clusters": int(accepted),
            })  # fmt: skip
        if any(s for _, s in per_rank):
            break  # leave with the leak reported instead of risking a hang

    if rank == 0:
        checks = []
        for r in runs:
            checks += [
                r["req_sent"] == r["req_recv"] == r["grant_recv"] + r["deny_recv"],
                r["grant_sent"] == r["grant_recv"] and r["deny_sent"] == r["deny_recv"],
                not r["stray_tags"],
                r["boundary_grants"] > 0,
                r["n_clusters"] == r["unique_centroids"] == r["accepted_clusters"] > 0,
            ]
        print(
            RESULT_PREFIX
            + json.dumps(
                {
                    "ok": all(checks),
                    "grants": sum(r["grant_sent"] for r in runs),
                    "runs": runs,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
