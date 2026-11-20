# The benchmark

`src/benchmark` runs every method, AFINNI and ART (`AdaptiveSurrogate`) beside the schemes of
`src/schemes`, on two suites of problems, and keeps every run in one store:

- **functions** (`src/benchmark/functions.py`): the test functions of `src/problems` on their boxes,
  judged on a fixed test set (a 300 x 300 grid in 2-D, 50,000 uniform points above) by the worst error
  over the tolerance, at the tolerances of `TOLERANCES` (relative to max |f|);
- **step_maps** (`src/benchmark/step_maps.py`): the step maps of the ODE systems of `src/problems/ode`,
  trained on visited states and judged on held-out trajectories, one step and chained (rollout), with
  the speedup over the integrator the system would be solved with.

## The store

`src/benchmark/store.py` keeps one JSON record per run, its arrays in an `.npz` beside it, and each
problem's shared inputs:

```
results/benchmark/runs/<suite>/<setup>/<problem>/d<d>/<method>/tol<t>-s<seed>.json
results/benchmark/data/<suite>/<problem>/<name>          (slice directions, trajectories, solver costs)
results/benchmark/logs/<suite>-<setup>.log               (the sweeps' logs)
figures/benchmark/<suite>/<setup>/                       (every plot, redrawn from the runs alone)
```

Records are written through `src/infra/records.py`, the module every long run in the repository keeps its
cases with (the benchmark, both companions' convergence studies, the manuscript study). Each record is
written atomically the moment its run ends and carries:

- its identity, `suite`, `problem`, `d`, `method`, `tol`, `seed`, `setup` (also under `key`);
- the envelope: `schema` (the record format), `status` (`done` or `failed`), `recorded` (when it was
  written) and `provenance`: the commit and `git describe`, whether tracked files differed from it, a hash
  of `src`'s sources, the CPU, core count, memory and GPU (never the host name), the Python and library
  versions and the settings in the environment that change a run;
- what the run cost: `train_s`, `oracle_s` and `oracle_calls` (rank 0's time inside the oracle while
  training, to tell the method's own work from the function's), `peak_host_mb` (every rank's peak memory
  above its start, summed, and per rank in `peak_host_mb_ranks`), `peak_device_mb`, `predict_us` (inference
  per query), `oracle_us` (one evaluation per test point) and `model_bytes` (the saved model's size);
- what it achieved: `evals`, `worst_ratio` (the worst error over the tolerance), `errors` (the worst, 99th
  and 99.9th percentiles, mean and share within, in units of the tolerance), `within_percent`,
  `mean_error`, `max_error`, `n_test`, and each suite's own fields;
- for a run that did not finish, `failed` (the reason) and, when it raised, `failure` (type, message,
  traceback without this machine's paths).

A record of an older format is upgraded when it is read; `python -m src.benchmark.store migrate` rewrites
the store in the current one. Fields a record never measured are absent, and a migrated record's provenance
holds only the `describe` it recorded (null for the earliest runs).

## Running one suite


From the repository root, every method on the same 8 MPI ranks (AFINNI and ART train with work
stealing; the schemes spread their oracle calls):

```bash
SURROGATE_DEVICE=cpu SURROGATE_MPI=on OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PATH=$PWD/.venv/bin:$PATH PYTHONPATH=. .venv/bin/mpiexec -n 8 .venv/bin/python -m src.benchmark.functions \
  --problems rosenbrock,gaussian --d 2 --methods afinni,art,art_quad,chebtree,sparsecheb --plot </dev/null
```

- Runs and plots are kept per setup, the device and the rank count (`results/benchmark/runs/functions/cpu-8rank/`,
  `figures/benchmark/functions/cpu-8rank/`), so that one comparison never mixes them: times depend on
  both, and ART's split decisions differ between CPU and GPU.
- A run is trained once and recorded in the store (metrics for every seed; predictions, leaf ids and,
  above 2-D, the slice predictions for seed 1). A new scheme is plotted against the recorded
  baselines without retraining them. `--force` retrains.
- `--plot` writes to the setup's figures, per problem, after its runs:
  - `<problem>-d<d>-cost.png`: evaluations against the worst error / tolerance, the mean error,
    the % within tolerance, the training time and the inference time per query (with the oracle's
    time per point);
  - in 2-D, `<problem>-d2-tol<t>-maps.png`: each method's pieces over contours of f, and its error map;
  - above 2-D, `<problem>-d<d>-tol<t>-views.png`: the slice plane and the shadow coordinates spanned
    by the two directions in which f changes most (the leading eigenvectors of the gradient
    covariance), each method's error on both, and every method's error distribution;
  - `summary-d<d>.md`: every method's seed means;
  - `overview-d<d>.png`: every recorded problem (rows) and tolerance (column groups) against every
    method, coloured by the worst |error| / tolerance (worst seed; blue holds, red exceeds) and
    labelled with the oracle evaluations; the cheapest method that holds the tolerance is outlined,
    and the bottom rows count, per method, the cases held and won;
  - `overview-d<d>-cost.png`: the same grid for the training time, the inference time per query
    (beside the oracle's time per point), the pieces and the mean |error| / tolerance.
  `--overview` redraws both overviews from the store alone (`--setup` picks another setup's runs).
- `results/` and `figures/` are not committed (`.gitignore`).
- The new schemes are NumPy only. `SURROGATE_DEVICE=gpu` moves AFINNI and ART to the GPU; the
  schemes would need CuPy versions of their per-box arithmetic, which is milliseconds per run, so
  only their oracle calls matter. The baselines' runs here (up to about 10^6 evaluations, d <= 5)
  train no faster on one GPU than on 8 CPU ranks and predict several times slower, since their
  batches are too small to hide kernel launches; so the recipe uses 8 CPU ranks.
- AFINNI at d = 10 (10^6 training points) needs several GiB of host memory per rank: run it with
  fewer ranks, and watch the memory.

The step-map suite runs serially (`SURROGATE_MPI=off`; the solver's own pool of processes,
`SURROGATE_ODE_WORKERS`, serves the oracle):

```bash
SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python -m src.benchmark.step_maps \
  --problems robertson --methods slowtable,chebtt_box --plot
```

## Sweeping a grid

`python -m src.benchmark.sweep functions|step_maps` runs every case not yet in the store, or recorded before its
method's last change to its results (`RESULTS_CHANGED`, below), cheapest first:
problems by input dimension, each method's tolerances from the loosest, each job in its own process
group under `--limit` seconds. A job over the limit is recorded as failed and the method's tighter
tolerances on that problem are skipped; a method that fails a family's loosest tolerance (the same
function at a smaller d, or a smaller size of a step-map family) is not run on its larger sizes. After
every recorded case a background process redraws that problem's figures.

```bash
SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python -m src.benchmark.sweep functions --d 2,3,5,10,20 --limit 90
SURROGATE_DEVICE=cpu SURROGATE_MPI=off SURROGATE_ODE_WORKERS=12 PYTHONPATH=. .venv/bin/python -m src.benchmark.sweep step_maps --first lorenz96
```

## The summary table

`python -m src.benchmark table` reads the store and prints, per method and problem class, the cases it ran,
held (worst error within the tolerance on every seed) and won (held with the fewest evaluations; ties count
for each), with median evaluations and inference time over its held cases. `--write README.md` replaces the
table in the README's "Choosing a method". A method with records older than the last change to its
results (`RESULTS_CHANGED` in `src/benchmark/table.py`, one entry per deliberate numerical change) is
marked stale, and the sweep runs those records again (`--since <time>` reruns every record older than a time).
That is the one rule for reuse everywhere (`src.infra.records.stale`): an edit that does not change a method's
results never forces a rerun, so a deliberate change must get its entry.

`python -m src.benchmark <functions|step_maps|sweep|table> ...` is the same entry point for all four.
