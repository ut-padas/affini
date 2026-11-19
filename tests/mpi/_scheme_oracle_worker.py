"""
The schemes' distributed oracle under MPI, launched by tests/mpi/test_mpi.py (and serially, with
SURROGATE_MPI=off, as the reference): rank 0 trains a SparseCheb while every rank evaluates its chunk of
each batch, so the model must not depend on the rank count; then an oracle that fails on the last rank
must raise on rank 0 and release every rank, instead of leaving them waiting in a collective.
"""

import hashlib
import json

import numpy as np

from tests.mpi._mpi_worker import RESULT_PREFIX


def main() -> None:
    from src.infra.backend import MPI
    from src.problems import FUNCTIONS
    from src.schemes import sparsegrid
    from src.schemes.oracle import distributed

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    p = FUNCTIONS["genz_gaussian"]
    lo, hi = p.domain(3)
    result = {}
    with distributed(p.values) as oracle:
        if oracle is not None:
            model = sparsegrid.build(oracle, lo, hi, 0.01, seed=1)
            Q = np.random.default_rng(5).uniform(lo, hi, (200, 3))
            result["hash"] = hashlib.sha256(np.ascontiguousarray(model.predict(Q)).tobytes()).hexdigest()
            result["evals"] = model.evals

    def failing(X):
        if comm.Get_rank() == size - 1:
            raise ValueError("injected failure")
        return np.zeros(len(X))

    with distributed(failing) as oracle:
        if oracle is not None:
            try:
                oracle(np.zeros((4 * size, 3)))
                result["error"] = None
            except Exception as e:  # noqa: BLE001 (one rank: the oracle's own error; several: rank 0's report of it)
                result["error"] = f"{type(e).__name__}: {e}"
    if rank == 0:
        print(RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
