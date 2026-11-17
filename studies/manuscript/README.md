# The manuscript study

Every experiment of the AFINNI paper, from one specification, on any machine. The settings are the
technical companion's Appendix A (`docs/report/`), and [`spec.py`](spec.py) is the same
specification as code: a change to one is a change to the other.

## Reproduce the paper

From the repository root:

```bash
uv sync --locked                                     # once; add --extra gpu on a CUDA machine
M=studies/manuscript/machines/workstation.toml       # or vista-gh.toml / vista-gg.toml

python -m studies.manuscript.run run --smoke --machine $M --seeds 0   # minutes: checks this machine end to end
python -m studies.manuscript.run plan --machine $M                    # every case, with projected cost
python -m studies.manuscript.run run --machine $M                     # hours to days; resumable
python -m studies.manuscript.run status --machine $M                  # what is done, failed, skipped, left
python -m studies.manuscript.collect                                  # data files and tables
```

(`make study-smoke`, `study-plan`, `study-run`, `study-status` and `study-collect` run the same,
with `MACHINE=<profile>`.) On Vista, `sbatch -p gh studies/manuscript/machines/vista.sbatch gh` runs
every GPU case and `sbatch -p gg ... gg` the distributed experiment; resubmit the same command to
continue.

## How it runs

- **Cases.** A case is one training run and its predictions, named by its settings
  (`rosenbrock_d5_tol0.01_hessian_s3`). Experiments are tags on cases, so a run that serves two
  experiments runs once. `spec.cases()` lists all of them.
- **Order.** Seed 0 of every case first, then the other seeds; within a seed, small d and loose
  tolerances first (`spec.cost_order`).
- **One process per case,** with the machine's device and MPI launcher, an empty stdin and the
  machine's time limit. Finished and failed cases are not run again (`--retry-failed` reruns
  failures), so `run` resumes where it stopped.
- **Projection.** Before a case, `run` projects its time and host memory from the cases already
  measured: another seed of the same case, or the growth between the two next-looser tolerances of
  its series. A case projected over the machine's limit, or whose series already ran out of memory
  or time at a looser tolerance, is recorded as `skipped` with the reason (`--force` runs it). A
  later run on a larger machine tries skipped cases again.
- **Failures** end one case: the host memory guard, the GPU running out of memory, the time limit,
  a kill by the system, any error. Each is recorded with its class and the stage it reached, and
  the run continues.

## What it writes

Under `studies/manuscript/output/` (the profile's `results`):

| Path | Contents |
|---|---|
| `cases/<case>.json` | one result per case: settings, provenance (commit, machine, library versions), status and failure, counts, times, memory, the candidate search's and the exact search's metrics, per-level statistics, iterative rounds |
| `logs/<case>.log` | the case's output |
| `data/cases.csv` | one row per case: every scalar of its result (`cand.*` and `exact.*` metrics) |
| `data/levels.csv`, `data/rounds.csv` | per level, per iterative round |
| `data/provenance.json` | the commits, machines and libraries behind the data |
| `data/tables/*.csv`, `tables.json` | the paper's tables: median over seeds, with min, max and seed count |

The tables are declared in [`tables.py`](tables.py) (`TABLES`): a column is a `cases.csv` column
aggregated over seeds, so adding, dropping or renaming one is an edit there and a rerun of
`collect`, never of the study.

## Machine profiles

[`machines/`](machines) holds one TOML file per kind of machine: `device` (`gpu`, `cpu`, `auto`),
`mpi_launcher`, `max_ranks`, `case_timeout_h`, and optionally `host_memory_gib` (default 90% of the
machine's RAM, read at run time) and `exact_max_dim` (skip the exact search above this d). Copy one
for a new machine.

## For an agent

Asked to reproduce the paper: run the smoke check, then `plan`, `run` and `collect` as above, with
the profile of the machine at hand. Report `status` (how many cases are done, failed and skipped,
and why) and the tables' folder. Never edit `spec.py` to make a case fit: a skipped case is a result.
