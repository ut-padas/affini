# Using AFINNI

How to train and query surrogates on one CPU, a GPU and MPI ranks, how to profile,
plot and monitor runs, how to swap parts of the algorithm, and how to check your
changes. Installation and backend setup are in the [README](../README.md).

Run everything from the repository root (scripts as modules, e.g.
`python -m examples.example01`), so that `src` is importable.

- [Train and predict](#train-and-predict)
- [Running on CPU, GPU and MPI](#running-on-cpu-gpu-and-mpi)
- [Training settings](#training-settings)
- [Results, status and saving](#results-status-and-saving)
- [Profiling](#profiling)
- [Plots](#plots)
- [Live monitoring](#live-monitoring)
- [Paper experiments (launcher)](#paper-experiments-launcher)
- [Swapping parts of the algorithm](#swapping-parts-of-the-algorithm)
- [HPC clusters](#hpc-clusters)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Train and predict

```python
import numpy as np
from src import xp
from src.modes import SurrogateMode
from src.problems.rosenbrock import rosenbrock
from src.surrogate import AFINNISurrogate

X = xp.asarray(np.random.default_rng(0).uniform(-2, 3, (50_000, 2)))

surrogate = AFINNISurrogate()
surrogate.train(
    X, f_fn=rosenbrock, mode=SurrogateMode.AFFINE_COV_HESS,
    ATOL=0.02, RTOL=0.02, ERROR_METRIC="Linf-local",
    NUM_INITIAL_CLUSTERS=25, MIN_PTS_PER_CLUSTER=100,
)
Q = xp.asarray(np.random.default_rng(1).uniform(-2, 3, (10_000, 2)))
pred, outside = surrogate.predict(Q, k_nn=1)   # (N, fd) values, (N,) "outside every trust region"
```

- `f_fn(X)` returns `(N,)` or `(N, fd)`; vector outputs are supported throughout.
- Modes that need gradients take `g_fn` (returning `(N, d)` or `(N, fd, d)`), or a
  single `oracle_fn` returning `(F, G)`; `*_EXACT_HESS` modes also need `HESSIAN`
  (returning `(N, d, d)`).
- `Y_train=f(X)` skips the initial evaluations (they are still counted).
- `predict(..., k_nn=k)` blends the `k` best leaves by trust-region distance;
  `strict_trust_region=True` ignores leaves whose region does not contain the
  query. `coarse_factor` (default 10) sets how many Euclidean neighbours are reranked.
- [`example01.py`](example01.py) is a complete 2-D example with a plot.

## Running on CPU, GPU and MPI

The same script runs everywhere; the backend is chosen at import time.

| Run on | Command |
|---|---|
| One CPU process | `AFINNI_DEVICE=cpu AFINNI_MPI=off uv run python -m examples.example01` |
| One GPU | `AFINNI_DEVICE=gpu AFINNI_MPI=off uv run python -m examples.example01` |
| N CPU ranks | `AFINNI_DEVICE=cpu uv run mpiexec -n 4 python -m examples.example01` |
| N ranks sharing / owning GPUs | `AFINNI_DEVICE=gpu uv run mpiexec -n 4 python -m examples.example01` |
| TACC | `source scripts/hpc/tacc.sh; ibrun .venv/bin/python afinni_launcher.py ...` |

| Variable        | Values                         | Effect                                                                                                 |
| --------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `AFINNI_DEVICE` | `auto` (default), `cpu`, `gpu` | `cpu` forces NumPy even when a GPU is present. `gpu` raises an error instead of silently falling back. |
| `AFINNI_MPI`    | `auto` (default), `off`, `on`  | `off` ignores an installed mpi4py. `on` raises an error if MPI is unavailable.                         |

Set them explicitly for benchmarks. The same information is available in code
(`from src import IS_CUPY_BACKEND, HAS_CUML, HAS_MPI, xp`); detection lives in
[`src/backend.py`](../src/backend.py).

Without `mpiexec`, or when MPI isn't installed, the same code runs as a single rank.
If a script is started under `mpiexec` but MPI can't be loaded, AFINNI stops with an
error rather than letting every process quietly run the whole job as rank 0.

On Windows, Intel MPI installs `mpiexec.exe` into `.venv\Library\bin`, which `uv run`
does not put on `PATH`. Add it once per PowerShell session:

```powershell
$env:PATH = "$PWD\.venv\Library\bin;$env:PATH"
```

Under MPI:

- `train` and `predict` are collective: every rank calls them. Only rank 0's
  training data and queries are used (pass `xp.empty((0, d))` / `None` elsewhere);
  `predict` returns results on rank 0 and `(None, None)` on other ranks. Every
  rank ends up with the full model.
- Work stealing (`USE_WORK_STEALING=True`, the default) moves unprocessed clusters
  from busy to idle ranks. With more than one rank it raises `MAX_LEVELS` to 1000.
- Rank 0 should do all file writing and printing.
- Several ranks may share one GPU; on CPU, set `OMP_NUM_THREADS=1` so ranks do not
  oversubscribe the cores.

```python
from src import get_mpi_rank, xp
rank = get_mpi_rank()
X = xp.asarray(X_all) if rank == 0 else xp.empty((0, d))
surrogate.train(X, f_fn=f, ...)
pred, outside = surrogate.predict(Q_all if rank == 0 else None)
```

CPU and GPU runs are statistically equivalent but not bit-identical; each is
reproducible for a fixed `SEED` and rank count. The test problems in
`src/problems/` take their fixed constants (rotations, directions) from NumPy on
every backend, while sampling (training/test points, synthetic points, KMeans
initialisation) uses the active backend's random generator, and GPU kernels sum in
a different order.

## Training settings

`train` keyword arguments (upper case) map to `src.config.ClusteringConfig`.

| Setting | Meaning |
|---|---|
| `mode` | Preset pair (local model, split strategy); see the table below |
| `ATOL`, `RTOL`, `ERROR_METRIC` | Acceptance test: `"Linf-local"` (relative to the cluster's own max \|f\|, or absolute), `"L2"`, `"Linf-global"` (relative to max \|f\| over the initial data, compared with **ATOL**; RTOL unused). Exact formulas: [`src/acceptance.py`](../src/acceptance.py) |
| `NUM_INITIAL_CLUSTERS`, `INITIAL_CLUSTERING_METHOD` | Roots of the forest: `"kmeans"`, `"uniform_grid"` or `"knndbscan"` |
| `MIN_PTS_PER_CLUSTER` | Rejected clusters are augmented to at least twice this before splitting |
| `MAX_LEVELS` | Refinement depth limit; reaching it leaves `converged=False` |
| `SAMPLING_METHOD` | Where new points go when augmenting (`"convex_combination"`) |
| `Q_REG` | Floor on covariance eigenvalues in the trust-region shape |
| `STORE_HESSIAN` | Second-order Taylor leaves (TAYLOR_LSQ_HESS only), one Hessian per output |
| `BATCH_SIZE` | Points moved to the device at once; capped automatically to fit GPU memory |
| `SEED`, `USE_WORK_STEALING` | Reproducibility, MPI load balancing |
| `PROFILE_TIMING`, `PROFILE_EVENTS`, `PROFILE_DIR` | See [Profiling](#profiling) |

| Mode | Local model | Split metric | Needs |
|---|---|---|---|
| `NEAREST_NEIGHBOR` | constant f(c) | covariance (PCA) | f |
| `AFFINE_COV_NO_HESS` | affine least squares | covariance | f |
| `AFFINE_COV_HESS` (default) | affine least squares | covariance-estimated \|H\| | f |
| `AFFINE_COV_HESS_EXACT_GRAD` | affine least squares | \|H\| from low Taylor error, exact gradient | f, g |
| `AFFINE_EXACT_HESS` | affine least squares | exact \|H\| | f, Hessian |
| `TAYLOR_COV_NO_HESS` | first-order Taylor | covariance | f, g |
| `TAYLOR_COV_HESS` | first-order Taylor | covariance-estimated \|H\| | f, g |
| `TAYLOR_EXACT_HESS` | first-order Taylor | exact \|H\| | f, g, Hessian |
| `TAYLOR_LSQ_HESS` | Taylor (+ second order with `STORE_HESSIAN`) | \|H\| fitted to gradients | f, g at every point |

Host memory: queues hold about `rejected clusters x 2 x MIN_PTS_PER_CLUSTER`
points, which grows quickly in high dimension with tight tolerances. If a level
would not fit, training stops with a `MemoryError` explaining what to change
(looser tolerance, smaller `MIN_PTS_PER_CLUSTER`, `MAX_LEVELS`, more ranks/nodes).

## Results, status and saving

```python
surrogate.N_src, surrogate.X_src, surrogate.Qs_src    # leaves: count, centroids, trust shapes
surrogate.leaf_params                                  # the local model's parameters by name
surrogate.E_cluster_src, surrogate.counts_src          # accepted error and points per leaf
surrogate.converged                                    # False if MAX_LEVELS stopped refinement
surrogate.num_unresolved_clusters, surrogate.num_unresolved_pts
surrogate.config                                       # every effective setting and component
surrogate.profiling_stats                              # counters, level statistics, profile

path = surrogate.save("results/model")                 # -> results/model.npz (versioned, no pickle)
loaded = AFINNISurrogate.load_model(path)               # also reads files from older versions
```

A partial run (stopped by `MAX_LEVELS`) warns and reports the unresolved points;
a run that accepted nothing raises. Models with custom local models load if the
model's class is registered (import its module first) or passed as
`load_model(path, local_model=...)`.

Scoring predictions: `src.surrogate.compute_error_metrics(preds, truth, outside,
atol, rtol)` returns L2/Linf errors overall and inside/outside the trust regions.

## Profiling

Three levels, all off by default (level statistics are always recorded):

| Flag | Records | Cost |
|---|---|---|
| `PROFILE_TIMING=True` | Seconds per phase per rank (`profiling_stats["per_rank"][r]["time_*"]`) | Synchronizes the GPU around each phase |
| `PROFILE_EVENTS=True` | A timeline: phase intervals with level/batch, steal requests/grants/denials, queue loads, level summaries (`per_rank[r]["events"]`) | Small; repeats are coalesced |
| `PROFILE_DIR="runs/x"` | The same events, streamed live to `runs/x/rank_<r>.jsonl` | Same as events |

```python
surrogate.train(X, f_fn=f, PROFILE_TIMING=True, PROFILE_EVENTS=True, PROFILE_DIR="runs/rosen")
ps = surrogate.profiling_stats
ps["time_total"], ps["num_levels"], ps["total_f_evals"], ps["num_generated_pts"]
ps["level_stats"]          # one dict per (rank, level): clusters, accepted, points, runtime, ...
ps["per_rank"][0]["time_oracle"]   # time spent inside your f / g
```

Phases, their display groups and colours are defined once in
[`src/profiling.py`](../src/profiling.py) (`PHASES`); add a phase there and every
plot and the monitor pick it up. Profiles are saved with the model and, by the
launcher, as `<result>_profile.json` next to each result.

## Plots

All plots live in [`src/plotting/`](../src/plotting/) and accept a surrogate, its
`profiling_stats`, a saved profile JSON loaded with `json.load`, or (timeline plots)
the event streams. Each returns the figure and saves it when given a path.
[`example03_monitoring.py`](example03_monitoring.py) produces every profile plot
below from one multi-rank run.

```python
from src import plotting as P

P.plot_profile_dashboard(surrogate, "figs/dashboard.png")     # one-page summary
P.plot_rank_time_breakdown(surrogate, "figs/breakdown.png")   # needs PROFILE_TIMING
P.plot_level_heatmap(surrogate, "figs/levels.png", value="clusters_start")
P.plot_level_progress(surrogate, "figs/growth.png")
P.plot_timeline(surrogate, "figs/timeline.png")               # needs PROFILE_EVENTS / PROFILE_DIR
P.plot_load_over_time(surrogate, "figs/load.png")
P.plot_steal_matrix(surrogate, "figs/steals.png")             # needs steals (MPI)
```

| Plot | Shows | Needs |
|---|---|---|
| `plot_rank_time_breakdown` | Share (or seconds, `normalize=False`) of each rank's time per phase group, with total time per rank | `PROFILE_TIMING` |
| `plot_level_heatmap` | One level statistic per level and rank: `clusters_start`, `good_clusters`, `num_pts_next_level`, `generated_pts`, `size_mean`, `runtime_sec`, ... | always |
| `plot_level_progress` | Accepted vs split clusters per level, points queued and generated | always |
| `plot_timeline` | Gantt chart per rank coloured by phase, with victim to thief steal arrows | events |
| `plot_load_over_time` | Queued points per rank over time, steal grants marked | events |
| `plot_steal_matrix` | Points (or `value="n_clusters"`) moved from each victim to each thief | events, steals |
| `plot_profile_dashboard` | Breakdown, growth, load and steals, whichever are available | any |
| `plot_adaptive_clustering_results` | 2-D leaves and trust ellipsoids over the function | 2-D problems |
| `plot_error_histogram` | Pointwise relative errors | predictions |

The tables behind the profile plots are available for your own analysis:
`P.phase_breakdown_frame`, `P.level_stats_frame`, `P.events_frame`,
`P.steals_frame` (all `pandas.DataFrame`).

To plot a finished run from its files:

```python
import json
from src import plotting as P
from src.profiling import read_event_streams

P.plot_profile_dashboard(json.load(open("results/rosenbrock/<run>_profile.json")))
P.plot_timeline(read_event_streams("runs/rosen"))
```

Adding a plot: put it in `src/plotting/profiling.py` (training profiles) or
`src/plotting/surrogate.py` (surrogates and errors), accept the inputs above via
`profile_of`, draw inside `afinni_style()`, take an optional `path` and return the
figure; export it from `src/plotting/__init__.py`.

## Live monitoring

Train with `PROFILE_DIR`, then watch from another terminal.
[`example03_monitoring.py`](example03_monitoring.py) is a ready-made run: 8 MPI
ranks on the CPU train a 5-D Rosenbrock surrogate (2M points, tol 0.007; about 85 s
on a 22-core laptop). Ranks finish their own work at different times, and the idle
ones steal from the busy ones (~60 steals).

```bash
# terminal 1: train (8 ranks, CPU)
uv run mpiexec -n 8 python -m examples.example03_monitoring
# terminal 2: live view; exits when every rank is done
uv run python -m src.monitor scratch/example03_monitoring/stealing/stream
# afterwards (or any time): one snapshot
uv run python -m src.monitor scratch/example03_monitoring/stealing/stream --once

# the same run without work stealing, for comparison
uv run mpiexec -n 8 python -m examples.example03_monitoring --stealing off
uv run python -m src.monitor scratch/example03_monitoring/no_stealing/stream
```

Per rank the monitor shows the current phase (or idle / asking for work / retiring /
done), level, batch progress in the level (`current/planned`), queued points with a
load bar, clusters, accepted leaves, generated points, idle share, steals out/in
(count and points), and how long ago the rank last reported (`Seen`, relative to
the newest event of any rank), plus recent steals and level ends. Watch for ranks
that run out of work (idle, "asking R2 for work") and the steal counts and load bars
changing as work moves.

At the end the example writes `profile.json` and every profile plot (see
[Plots](#plots)) to `scratch/example03_monitoring/stealing/` or `.../no_stealing/`,
depending on `--stealing on|off`. Comparing the two timelines and time breakdowns
shows what work stealing buys: without it, ranks that finish their own share early
sit idle until the slowest rank is done. `--dim`, `--n-train`, `--tol`, `--init`
and `--metric` change the problem.

The monitor only reads the stream files, so it adds no MPI traffic and works from a
login node when the run directory is on a shared file system. The newest idle
stretch of a rank appears when that rank records its next event. For your own runs,
pass `PROFILE_DIR="<dir>"` to `train` and point the monitor at that directory.

## Paper experiments (launcher)

[`afinni_launcher.py`](../afinni_launcher.py) runs the benchmark problems and writes
one JSON result per run (plus error histogram, profile JSON and dashboard) under
`results/<problem>/`.

```bash
uv run python afinni_launcher.py --experiment rosenbrock --dims 2 5 --tols 0.1 0.05
uv run mpiexec -n 4 python afinni_launcher.py --experiment anisotropic_quadratic --dims 5 \
    --tols 0.02 --error_eval_mode taylor_cov_hess --sampling modified
```

Options: `--experiment` (`rosenbrock`, `anisotropic_quadratic`, `anisotropic_gaussian`,
`degenerate_quadratic`, `indef_degen_quadratic`, `lorenz96`, `wave`), `--dims`,
`--tols`, `--error_eval_mode` (a mode value), `--sampling uniform|modified` (initial
data), `--sampling_method` (augmentation sampler), `--N_train`, `--Q_reg`,
`--coarse_factor`, `--dt` (Lorenz flow map). An existing result file means "done"
and is skipped; non-default settings are part of the file name, so different
settings never reuse each other's results. `results/<problem>/_summary.csv`
collects all runs of a problem.

## Swapping parts of the algorithm

Training is a loop over batches of clusters: **fit** a local model on each cluster,
**accept** it if accurate enough (then build its **trust region**), otherwise
**augment** it with new points (**sampler**) and **split** it in two. Each bold part
is a component with a base class and a registry; a `mode` is only a preset pair
(local model, split strategy). The engine owns batching, GPU transfers, buffers,
MPI and work stealing, so a new component gets all of those for free.

| Component | Base class | Registry | Default |
|---|---|---|---|
| Local model | `src.local_models.LocalModel` | `LOCAL_MODELS` | from `mode` |
| Split strategy | `src.splitting.SplitStrategy` | `SPLIT_STRATEGIES` | from `mode` |
| Sampler | `src.sampling.Sampler` | `SAMPLERS` | `SAMPLING_METHOD` |
| Acceptance | `src.acceptance.AcceptanceCriterion` | `ACCEPTANCE_CRITERIA` | `ERROR_METRIC`, `ATOL`, `RTOL` |
| Trust region | `src.trust_regions.TrustRegion` | `TRUST_REGIONS` | `CovarianceEllipsoid(Q_REG)` |

Each can be given to `train` (lower-case keyword) or `ClusteringConfig` as an
instance, a registered name, or `(name, settings)`:

```python
from src.splitting import CovarianceSplit

surrogate.train(
    X, f_fn=f, mode=SurrogateMode.AFFINE_COV_HESS,
    split_strategy=CovarianceSplit(cut="sign"),          # instance: the paper's sign cut
    sampler="convex_combination",                         # registered name
    acceptance=("L2", {"atol": 1e-3, "rtol": 1e-2}),      # (name, settings)
)
```

Built-in settings: `TaylorModel(second_order=...)`,
`CovarianceHessianSplit(gradient="local_lsq"|"centroid", min_pts_per_cluster, cut)`,
every split's `cut="median"|"sign"`, `CovarianceEllipsoid(q_reg)`. Components built
by name receive matching configuration values (`min_pts_per_cluster`, `atol`/`rtol`,
`q_reg`) when their constructor takes them.

### Writing a component

Register the class so it can be named in configs, the launcher and saved models,
and implement the one abstract method. All methods work on a whole batch of
clusters at once (stored contiguously, `labels` giving each row's cluster) and
must stay vectorized on the active backend `xp`, with no Python loops over points
or clusters, so they run on the GPU too.

```python
from src import xp
from src.clusters import ParamSpec
from src.local_models import LOCAL_MODELS, AffineLSQModel, LocalModel


@LOCAL_MODELS.register
class OffsetAffine(LocalModel):
    """Affine least squares plus a constant offset, added only to accepted leaves."""

    name = "offset_affine"
    needs_point_grad = False     # True: fit/finalize receive data.G (gradients at every point)
    needs_centroid_grad = False  # True: data.g_c (gradients at the centroids)

    def __init__(self, offset: float = 0.0):
        self.offset, self._affine = offset, AffineLSQModel()

    def settings(self):          # how a saved model rebuilds it
        return {"offset": self.offset}

    def columns(self, d, fd):    # every stored per-leaf parameter and its shape
        return {**self._affine.columns(d, fd), "offset": ParamSpec((fd,))}

    def fit(self, data):         # used for the acceptance test
        return self._affine.fit(data)

    def finalize(self, data, params):   # optional: accepted clusters only
        return {"offset": xp.full((len(data.sizes), data.F.shape[1]), self.offset)}

    def evaluate(self, params, X, cluster_ids):   # cluster_ids (N,) or (n, k)
        out = self._affine.evaluate(params, X, cluster_ids)
        # The acceptance test calls this with fit() columns only; finalize() columns exist afterwards.
        return out + params["offset"][cluster_ids] if "offset" in params else out


surrogate.train(X, f_fn=f, local_model=("offset_affine", {"offset": 0.5}))
```

Contracts the engine checks (with a clear error):

- **Local model:** `fit` + `finalize` return exactly the declared `columns` with
  the declared shapes; the names `centroids`, `f`, `Q`, `E_cluster`, `counts`
  are reserved. `evaluate` may read `params["centroids"]` and `params["f"]`, and
  must also work without the `finalize` columns: the acceptance test evaluates the
  `fit` output, and only accepted leaves get `finalize` columns.
- **Sampler:** `sample(clusters, n_new, rng)` returns `sum(n_new)` points grouped by
  cluster; `clusters` holds `X`, `counts`, `starts`, `centroids`; use only `rng`
  for randomness.
- **Split strategy:** implement `metric(data) -> (M, M C M)` to change the split
  direction, `child_sizes(counts, sorted_scores)` to change where clusters are cut,
  or `split(data)` entirely; splits are binary and must keep each cluster's rows
  together and its two child sizes adding up to its size.
- **Acceptance:** `__call__(F, F_hat, starts, sizes, F_max_global) -> (is_good, E)`.
- **Trust region:** `shapes(X, centroids, labels, counts) -> Q` (K, d, d),
  symmetric positive semi-definite; inference treats `dx^T Q dx > 1` as outside.

Under MPI every rank builds its own components, so pass the same specs on every
rank (import the module that registers a custom class everywhere). The tests in
[`tests/integration/test_components_api.py`](../tests/integration/test_components_api.py)
are working examples of each kind of component.

## HPC clusters

On a cluster, use the `mpi-system` extra and **not** `mpi`: the pip-bundled MPICH
would take priority over the cluster's MPI and its fast interconnect. Installing
needs no modules loaded; running does, because mpi4py binds to the system MPI at
runtime.

On TACC (Vista, Stampede3):

```bash
# 1. Once per system, in ~/.bashrc: keep uv's downloads out of the small $HOME quota
#    (14 GB on Stampede3, 23 GB on Vista; the GPU packages are several GB).
export UV_CACHE_DIR=$WORK/.cache/uv
export UV_PYTHON_INSTALL_DIR=$WORK/uv-python

# 2. Get a compute node, and install there
idev -p gh -N 1 -n 4 -t 01:00:00           # Vista GH200 node; on Stampede3 use -p h100
uv sync --extra gpu --extra mpi-system     # `make install-hpc` is the same without the dev tools

# 3. Load the cluster's MPI, then run. Put the `source` line in your SLURM job scripts too.
source scripts/hpc/tacc.sh
ibrun .venv/bin/python afinni_launcher.py ...
```

For another cluster, copy [`scripts/hpc/tacc.sh`](../scripts/hpc/tacc.sh) to
`scripts/hpc/<site>.sh` and change the module lines.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error: Failed to spawn: mpiexec` on Windows | Intel MPI's launcher is not on `PATH`; see [Running on CPU, GPU and MPI](#running-on-cpu-gpu-and-mpi). |
| `AFINNI_DEVICE=gpu was requested but … no usable CUDA device was found` | No NVIDIA GPU or driver is visible to this process (on a cluster, probably a login node). `uv run python -m src` shows what was detected. |
| `UserWarning: CUDA path could not be detected` from CuPy | Harmless: the CUDA libraries come from pip. |
| `OFI EP enable failed` MPI abort (Intel MPI on a Stampede3 login node) | Run on a compute node, through `ibrun`, after `source scripts/hpc/tacc.sh`. |
| Scripts that use `multiprocessing` hang or re-run themselves on Windows and macOS | Those platforms start workers by re-importing your script; put the entry point under `if __name__ == "__main__":`, as in `examples/`. |
| `MemoryError: The next level's queue ... needs ... GiB more host memory` | The run would exhaust host RAM; see the host-memory note under [Training settings](#training-settings). |
| `RuntimeWarning: Training stopped at MAX_LEVELS=...` | Refinement hit the depth limit before every cluster met the tolerance; raise `MAX_LEVELS` or loosen the tolerance (see `surrogate.converged`). |

## Development

Conventions for code, docstrings and verification are in [`AGENTS.md`](../AGENTS.md).

```bash
uv sync --all-groups             # also the research/ and visualisation groups
make help                        # Linux/macOS wrappers, e.g. `make dev EXTRAS="gpu mpi"`
```

| Dependency group | Contents                                      | Installed by        |
| ---------------- | --------------------------------------------- | ------------------- |
| `dev`            | `lint` + `test` + `notebook`                  | `uv sync` (default) |
| `research`       | extra packages for the scripts in `research/` | `--group research`  |
| `viz`            | animation/plotting (manim, plotly)            | `--group viz`       |

Every `Makefile` target runs a single `uv` command, so on Windows run that command directly.

| Task | Command |
|---|---|
| Tests on CPU | `AFINNI_DEVICE=cpu AFINNI_MPI=off uv run pytest -q --no-cov -m "not mpi"` |
| Tests on GPU | `AFINNI_DEVICE=gpu AFINNI_MPI=off uv run pytest -q --no-cov -m "not mpi"` |
| Multi-rank tests | `AFINNI_MPI=on AFINNI_MPI_NP=4 uv run pytest tests/mpi -o addopts='' -q` |
| One area | `uv run pytest tests/unit`, `tests/integration`, `tests/problems` |
| Lint / format | `uv run ruff check .`, `uv run ruff format .` |

`uv run pytest` runs everything the current install supports; its header line shows
which backends were exercised (`afinni backend: array=NumPy | cuML=no | MPI=...`). A
missing backend is skipped unless an override makes it required:

| Checking  | Command |
| --------- | ------- |
| GPU       | `AFINNI_DEVICE=gpu uv run pytest tests/integration/test_backend.py` |
| MPI       | `AFINNI_MPI=on uv run pytest tests/integration/test_backend.py tests/mpi` (= `make test-mpi`) |
| GPU + MPI | `AFINNI_DEVICE=gpu AFINNI_MPI=on uv run pytest tests/integration/test_backend.py tests/mpi` |
| HPC       | `AFINNI_DEVICE=gpu AFINNI_MPI=on AFINNI_MPIEXEC=ibrun uv run pytest tests/mpi tests/integration/test_backend.py` on a GPU node after `source scripts/hpc/tacc.sh`, with at least `AFINNI_MPI_NP` tasks (default 2) |

Test layout: `tests/unit` (one component, no full training), `tests/integration`
(training and prediction through the public API), `tests/mpi` (real multi-rank
jobs, including work stealing with gradients), `tests/problems` (benchmark
functions against SymPy), `tests/experiments`. `tests/_helpers.py` has
`assert_backend_equal`, which is exact on CPU and allows only rounding-level
differences on GPU (`separate_runs=True` when comparing two trainings).

Checking that a change does not alter results, against another checkout (for
example a `git worktree` of `main`):

```bash
git worktree add ../affini-main main
(cd ../affini-main && AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. \
    python "$OLDPWD/scripts/regression_check.py" --save /tmp/ref)
AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. python scripts/regression_check.py \
    --save /tmp/new --compare /tmp/ref            # must be bit-identical on CPU
# on a GPU: AFINNI_DEVICE=gpu ... --rtol 1e-12     (leaf and evaluation counts still exact)
```

`regression_check.py` trains every mode, metric and output dimension plus extra
paths (oracle, precomputed Y, uniform grid, tiny batches, second order) and fails
on any difference in leaves, evaluations, trust masks or arrays. For timing,
`scripts/bench_predict.py` trains and times prediction on Rosenbrock
(`--save-predictions` / `--compare` exit non-zero on differences); preload heavy
imports or compare whole-process times, since the first `train()` imports
scikit-learn.
