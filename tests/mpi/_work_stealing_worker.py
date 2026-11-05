"""Conserve known clusters while a grant arrives after the victim goes idle."""

import json

import numpy as np

from tests.mpi._mpi_worker import RESULT_PREFIX


def main():
    import src.adaptive_clustering as ac
    from src.adaptive_clustering_utils import ClusteringConfig, Evaluator, SurrogateMode
    from src.backend import MPI

    world = MPI.COMM_WORLD
    rank, size = world.Get_rank(), world.Get_size()
    assert size > 1
    ac.MIN_LOAD_THRESHOLD = 1

    class Comm:
        def __init__(self):
            self.first_request_poll = True
            self.victim_idle = False
            self.requests = self.replies = self.grants = 0
            self.responses = 0
            self.delayed_grants = 0

        def __getattr__(self, name):
            return getattr(world, name)

        def Iprobe(self, source, tag, status=None):  # noqa: N802
            if rank == 0 and tag == ac.TAG_STEAL_REQ and self.first_request_poll:
                self.first_request_poll = False
                # Hold batch one until an empty rank has requested work.
                world.Probe(source=MPI.ANY_SOURCE, tag=ac.TAG_STEAL_REQ)
            if (
                delay_grant
                and rank != 0
                and source == 0
                and tag == ac.TAG_STEAL_GRANT
                and not self.victim_idle
            ):
                return False
            return world.Iprobe(source=source, tag=tag, status=status)

        def Recv(self, buf, source, tag):  # noqa: N802
            world.Recv(buf, source=source, tag=tag)
            if tag == ac.TAG_STATE_BROADCAST and source == 0 and buf[0] == 0:
                self.victim_idle = True

        def send(self, obj, dest, tag):
            if tag == ac.TAG_STEAL_REQ:
                self.requests += 1
            elif tag in (ac.TAG_STEAL_DENY, ac.TAG_STEAL_GRANT):
                self.replies += 1
            return world.send(obj, dest=dest, tag=tag)

        def recv(self, source, tag):
            obj = world.recv(source=source, tag=tag)
            if tag in (ac.TAG_STEAL_DENY, ac.TAG_STEAL_GRANT):
                self.responses += 1
            if tag == ac.TAG_STEAL_GRANT:
                self.grants += 1
                self.delayed_grants += int(source == 0 and self.victim_idle)
            return obj

    for has_g in (False, True):
        for stealing in (False, True):
            for repeat in range(3):
                # Also exercise payloads larger than MPI eager buffers. Delaying
                # these until the victim is idle would deadlock the test itself.
                delay_grant = repeat > 0
                points_per_cluster = 8 if delay_grant else 4096
                comm = Comm()
                x = (
                    np.arange(8 * points_per_cluster, dtype=float).reshape(-1, 1)
                    if rank == 0
                    else np.empty((0, 1))
                )
                sizes = (
                    np.full(8, points_per_cluster, dtype=np.int64)
                    if rank == 0
                    else np.empty(0, dtype=np.int64)
                )
                buffers, count, _, generated, _ = ac._run_local_refinement(
                    config=ClusteringConfig(
                        mode=SurrogateMode.NEAREST_NEIGHBOR,
                        error_metric="Linf-local",
                        atol=0.05,
                        rtol=0.05,
                        min_pts_per_cluster=1,
                        q_reg=1e-8,
                        max_levels=1000,
                        batch_size=points_per_cluster,
                        seed=0,
                        max_clusters_est=10,
                        use_work_stealing=stealing,
                        save_all_gen_training_data=False,
                    ),
                    CLUSTER_PTS=x,
                    CLUSTER_F=np.ones_like(x),
                    CLUSTER_G=np.ones((len(x), 1, 1)) if has_g else None,
                    CLUSTER_SIZES=sizes,
                    CLUSTER_STARTS=np.arange(len(sizes), dtype=np.int64)
                    * points_per_cluster,
                    F_max_global=np.ones(1),
                    d=1,
                    fd=1,
                    evaluator=Evaluator(f_fn=lambda x: np.ones((len(x), 1))),
                    rank=rank,
                    size=size,
                    profiling_stats={"level_stats": []},
                    comm=comm,
                )
                assert generated == 0
                results = world.allgather(
                    (
                        buffers["centroids"][:count, 0].tolist(),
                        buffers["counts"][:count].tolist(),
                        comm.requests,
                        comm.replies,
                        comm.responses,
                        comm.grants,
                        comm.delayed_grants,
                    )
                )
                centers = sorted(v for r in results for v in r[0])
                counts = [v for r in results for v in r[1]]
                assert (
                    centers
                    == (
                        np.arange(8) * points_per_cluster + (points_per_cluster - 1) / 2
                    ).tolist()
                ), results
                assert counts == [points_per_cluster] * 8, results
                assert (
                    sum(r[2] for r in results)
                    == sum(r[3] for r in results)
                    == sum(r[4] for r in results)
                )
                if stealing:
                    assert sum(r[5] for r in results) > 0, (
                        "test did not exercise a grant"
                    )
                    assert not delay_grant or sum(r[6] for r in results) > 0, (
                        "grant was not delayed until victim idle"
                    )
                else:
                    assert sum(r[2] for r in results) == 0
                world.Barrier()
                assert not world.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                # Do not let a faster rank start the next run's broadcasts
                # while another rank is still inspecting the previous run.
                world.Barrier()
    if rank == 0:
        print(RESULT_PREFIX + json.dumps({"conserved": True}), flush=True)


if __name__ == "__main__":
    main()
