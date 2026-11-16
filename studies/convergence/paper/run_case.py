"""
Run one ``afinni_launcher.py`` experiment with its results under ``--results`` instead of the
repository's ``results/``, optionally with another seed.

Usage (from the AFINNI root, PYTHONPATH=.)::

    AFINNI_DEVICE=gpu AFINNI_MPI=off python studies/convergence/paper/run_case.py \\
        --results OUT/results/seed_0 [--seed 0] -- --experiment rosenbrock --dims 2 --tols 0.1

Everything after ``--`` is the launcher's own command line, so a case means exactly what the
same flags mean to ``afinni_launcher.py``. The launcher skips a result file that already exists,
which makes reruns resumable. ``--seed s`` (s != 0) replaces the launcher's data seed (its
training and test points) and ``TrainingConfig.seed`` (augmentation and the initial k-means), so
``seed_s`` replicates measure run-to-run spread; seed 0 runs the launcher unchanged. Each run
also writes ``<results>/<case>.peak.json`` with the peak resident memory of rank 0.
"""

import argparse
import json
import logging
import resource
import sys
import time
from functools import partial
from pathlib import Path


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: run_case.py --results DIR [--seed S] [--case ID] -- <afinni_launcher.py arguments>")
    split = argv.index("--")
    own = argparse.ArgumentParser()
    own.add_argument("--results", type=Path, required=True)
    own.add_argument("--seed", type=int, default=0)
    own.add_argument("--case", default=None, help="name of the peak-memory record (default: none written)")
    mine = own.parse_args(argv[:split])

    import afinni_launcher as launcher
    from src.config import TrainingConfig
    from src.infra.logging_utils import setup_logging

    mine.results.mkdir(parents=True, exist_ok=True)
    setup_logging(console_level=logging.INFO, log_dir=str(mine.results / ".logs"))
    # The launcher resolves RESULTS_DIR, SEED and TrainingConfig from its own module globals when a
    # run starts, so replacing them here redirects that run and nothing else.
    launcher.RESULTS_DIR = mine.results
    if mine.seed != 0:
        launcher.SEED = mine.seed
        launcher.TrainingConfig = partial(TrainingConfig, seed=mine.seed)
    t = time.perf_counter()
    launcher.run_experiment_from_args(launcher.setup_argparse(argv[split + 1 :]))
    if mine.case and launcher.get_mpi_rank() == 0:
        peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record = {"case": mine.case, "seed": mine.seed, "wall_s": time.perf_counter() - t,
                  "peak_rss_gib_rank0": peak_kib / 2**20, "argv": argv[split + 1 :]}  # fmt: skip
        (mine.results / f"{mine.case}.peak.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
