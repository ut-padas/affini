"""
ART training under MPI, launched by tests/mpi/test_mpi.py (and serially, with
AFINNI_MPI=off, as the reference).

ART draws every random number from a stream keyed by the node, so the tree
must not depend on the rank count or on which rank processed a node. Rank 0 prints
the sorted leaf keys, a hash of the predictions on fixed queries and the number of
steal grants, for stealing off and for stealing forced by tiny thresholds, with box and
Voronoi roots. One box root puts all initial work on rank 0, so the other ranks must steal.
"""

import hashlib
import json
import logging

import numpy as np

from tests.mpi._mpi_worker import RESULT_PREFIX


def main() -> None:
    from src import to_host, xp
    from src.config import ARTSettings, TrainingConfig
    from src.core.evaluation import Evaluator
    from src.infra.backend import MPI
    from src.surrogate import AFINNISurrogate

    rank = MPI.COMM_WORLD.Get_rank()
    grants = {"n": 0}

    class GrantLog(logging.Handler):
        def emit(self, record):
            if "Granted" in record.getMessage():
                grants["n"] += 1

    log = logging.getLogger("src.engine.work_stealing")
    log.setLevel(logging.INFO)
    log.addHandler(GrantLog())
    log.propagate = False

    def f(x):
        return xp.stack([xp.sin(3 * x[:, 0]) * xp.cos(2 * x[:, 1]), x[:, 0] * x[:, 2] ** 2], axis=1)

    X = np.random.default_rng(0).uniform(-1, 1, (300, 3))
    queries = np.random.default_rng(1).uniform(-1, 1, (500, 3))
    runs = {}
    forced = {"steal_min_load": 1, "steal_size": 1}
    for name, stealing, settings in (
        ("no_steal", False, {}),
        ("steal", True, forced),
        # The soft split search calls progress() inside its optimizer, so requests are served mid-batch.
        ("soft_no_steal", False, {"split_search": "soft", "soft_iters": 8}),
        ("soft_steal", True, {"split_search": "soft", "soft_iters": 8, **forced}),
        # Voronoi roots exercise the bisector chords, root assignment and walk recovery.
        ("voronoi_no_steal", False, {"roots": "voronoi"}),
        ("voronoi_steal", True, {"roots": "voronoi", **forced}),
    ):
        grants["n"] = 0
        config = TrainingConfig(
            atol=2e-2, rtol=2e-2, algorithm="art", domain=([-1] * 3, [1] * 3), max_levels=12,
            num_initial_clusters=8, batch_size=2_000, use_work_stealing=stealing,
            art=ARTSettings(**{"roots": "box", "sample_multiplier": 8, "validation_samples": 8, **settings}),
        )  # fmt: skip
        model = AFINNISurrogate()
        model.train(X, config, Evaluator(f_fn=f))
        pred, outside = model.predict(queries)
        total_grants = MPI.COMM_WORLD.allreduce(grants["n"])
        if rank == 0:
            keys = sorted(zip(model.art_leaves["root"].tolist(), model.art_leaves["path_code"].tolist(), strict=True))
            runs[name] = {
                "leaves": int(model.N_src),
                "keys_hash": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
                "pred_hash": hashlib.sha256(np.ascontiguousarray(to_host(pred)).tobytes()).hexdigest(),
                "evals": int(model.profiling_stats["total_f_evals"]),
                "statuses": model.leaf_status_counts(),
                "outside": int(to_host(outside).sum()),
                "grants": int(total_grants),
            }
    if rank == 0:
        print(RESULT_PREFIX + json.dumps(runs), flush=True)


if __name__ == "__main__":
    main()
