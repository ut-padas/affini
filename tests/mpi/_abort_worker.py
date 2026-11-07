"""Rank 1's function raises mid-training; the job must abort instead of leaving rank 0 blocked."""

import numpy as np


def main() -> None:
    import src
    from src.backend import MPI
    from src.problems.rosenbrock import rosenbrock
    from src.surrogate import AFINNISurrogate

    rank = MPI.COMM_WORLD.Get_rank()

    def f(x):
        if rank == 1 and len(x) < 1000:  # a centroid or augmentation evaluation on rank 1
            raise RuntimeError("injected failure on rank 1")
        return rosenbrock(x)

    X = src.xp.asarray(np.random.default_rng(0).uniform(-2, 2, (20_000, 2))) if rank == 0 else src.xp.empty((0, 2))
    AFINNISurrogate().train(X, f_fn=f, ATOL=0.01, RTOL=0.01, NUM_INITIAL_CLUSTERS=16, MIN_PTS_PER_CLUSTER=50)


if __name__ == "__main__":
    main()
