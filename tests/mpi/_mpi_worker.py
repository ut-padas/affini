"""
MPI worker launched by tests/mpi/test_mpi.py as

    <launcher> -n N python -m tests.mpi._mpi_worker

Every rank checks the communication primitives AFINNI relies on, then runs the
shared smoke workload. Rank 0 prints one machine-readable result line.
"""

import json

import numpy as np

RESULT_PREFIX = "AFINNI_RESULT "


def check_primitives(MPI, comm, rank: int, size: int) -> None:  # noqa: N803
    obj = comm.bcast({"answer": 42} if rank == 0 else None, root=0)
    assert obj == {"answer": 42}, obj

    ranks = comm.gather(rank, root=0)
    if rank == 0:
        assert ranks == list(range(size)), ranks

    mine = comm.scatter([10 * r for r in range(size)] if rank == 0 else None, root=0)
    assert mine == 10 * rank, mine

    # Buffer-based point-to-point around a ring (as used by the work-stealing code).
    if size > 1:
        send_buf = np.full(4, rank, dtype=np.int64)
        recv_buf = np.empty(4, dtype=np.int64)
        comm.Sendrecv(
            send_buf, dest=(rank + 1) % size, sendtag=7,
            recvbuf=recv_buf, source=(rank - 1) % size, recvtag=7,
        )
        assert (recv_buf == (rank - 1) % size).all(), recv_buf

    comm.Barrier()
    # Nothing is in flight any more, so a probe must come back empty.
    # (Use the named constant: its value is implementation-specific, e.g. -2 in MPICH.)
    assert not comm.Iprobe(source=MPI.ANY_SOURCE, tag=12345)


def main() -> None:
    import src
    from src.infra.backend import MPI
    from tests._smoke import run_smoke

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size > 1:
        assert src.has_mpi(), "more than one rank but the serial stand-in is active"

    check_primitives(MPI, comm, rank, size)
    result = run_smoke()

    if rank == 0:
        print(RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
