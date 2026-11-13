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
- [ART (adaptive regression trees)](#art-adaptive-regression-trees)
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
from src.engine import ClusteringConfig, Evaluator, SurrogateMode
from src.problems.rosenbrock import rosenbrock
from src.surrogate import AFINNISurrogate

X = xp.asarray(np.random.default_rng(0).uniform(-2, 3, (50_000, 2)))

config = ClusteringConfig(
    mode=SurrogateMode.AFFINE_COV_HESS, atol=0.02, rtol=0.02, error_metric="Linf",
    num_initial_clusters=25, min_pts_per_cluster=100,
)
surrogate = AFINNISurrogate()
surrogate.train(X, config, Evaluator(f_fn=rosenbrock))
Q = xp.asarray(np.random.default_rng(1).uniform(-2, 3, (10_000, 2)))
pred, outside = surrogate.predict(Q, k_nn=1)   # (N, fd) values, (N,) "outside every trust region"
```

- `train(X, config, evaluator)` takes every algorithm setting from the
  `ClusteringConfig` ([Training settings](#training-settings)) and every function
  from the `Evaluator`.
- `Evaluator(f_fn=f)`: `f(X)` returns `(N,)` or `(N, fd)`; vector outputs are
  supported throughout. Modes that need gradients also take `g_fn` (returning
  `(N, d)` or `(N, fd, d)`), or a single `oracle_fn` returning `(F, G)`;
  `*_EXACT_HESS` modes also need `hessian_fn` (returning `(N, d, d)`). An evaluator
  can be reused; `profiling_stats` counts only the run's own evaluations.
- `train(..., Y_train=f(X))` skips the initial evaluations (they are still counted).
- `train(..., initial_labels=labels)` skips the initial clustering; compute the labels
  once with `initial_clustering(X, config)` (see [`example02_nelson.py`](example02_nelson.py)).
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
(`from src import IS_CUPY_BACKEND, HAS_MPI, xp`); detection lives in
[`src/infra/backend.py`](../src/infra/backend.py).

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
- Work stealing (`use_work_stealing=True`, the default) moves unprocessed clusters
  from busy to idle ranks.
- Rank 0 should do all file writing and printing.
- Several ranks may share one GPU; on CPU, set `OMP_NUM_THREADS=1` so ranks do not
  oversubscribe the cores.

```python
from src import get_mpi_rank, xp
rank = get_mpi_rank()
X = xp.asarray(X_all) if rank == 0 else xp.empty((0, d))
surrogate.train(X, config, Evaluator(f_fn=f))   # an equivalent config on every rank
pred, outside = surrogate.predict(Q_all if rank == 0 else None)
```

CPU and GPU runs are statistically equivalent but not bit-identical; each is
reproducible for a fixed `seed` and rank count. The test problems in
`src/problems/` take their fixed constants (rotations, directions) from NumPy on
every backend, while sampling (training/test points, synthetic points, KMeans
initialisation) uses the active backend's random generator, and GPU kernels sum in
a different order.

## Training settings

Fields of `src.config.ClusteringConfig` (all keyword-only); only a tolerance is required.

| Setting | Meaning |
|---|---|
| `mode` | Preset pair (local model, split strategy), `AFFINE_COV_HESS` by default; see the table below |
| `atol`, `rtol`, `error_metric` | Acceptance test: a cluster is accepted when every output component is within `rtol` relative **or** `atol` absolute error, measured as `"Linf"` (max error, relative to the cluster's own max \|f\|), `"L2"` (relative L2 / RMS) or `"pointwise"` (every point on its own: \|e\|/\|f\| < `rtol` or \|e\| < `atol`; ART's stopping rule). Give at least one tolerance; each is a scalar or one value per output component, and an omitted one never accepts. For an absolute test scaled to the function, pass `atol = tol * max\|f\|`. Exact formulas: [`src/components/acceptance.py`](../src/components/acceptance.py) |
| `num_initial_clusters`, `initial_clustering_method` | Roots of the forest: `"kmeans"` (custom k-means in `src/neighbors/kmeans.py` on both backends), `"uniform_grid"`, or `"knndbscan"` (needs `--group research_internal`, Linux). `src.engine.driver.initial_clustering(X, config)` returns the labels training starts from; pass them as `train(..., initial_labels=labels)` to reuse one clustering for several trainings on the same points |
| `min_pts_per_cluster` | Rejected clusters are augmented to at least twice this before splitting |
| `max_levels` | Refinement depth limit; reaching it leaves `converged=False` |
| `sampling_method` | Where new points go when augmenting (`"convex_combination"`) |
| `q_reg` | Floor on covariance eigenvalues in the trust-region shape |
| `store_hess` | Second-order Taylor leaves (TAYLOR_LSQ_HESS only), one Hessian per output |
| `batch_size` | Points moved to the device at once; capped automatically to fit GPU memory. A cluster larger than a batch is processed on its own |
| `algorithm`, `domain`, `art` | `"afinni"` (default: AFINNI's adaptive clustering) or `"art"`: see [ART](#art-adaptive-regression-trees) |
| `seed`, `use_work_stealing` | Reproducibility, MPI load balancing |
| `profile_timing`, `profile_events`, `profile_dir` | See [Profiling](#profiling) |

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
| `TAYLOR_LSQ_HESS` | Taylor (+ second order with `store_hess`) | \|H\| fitted to gradients | f, g at every point |

Host memory: queues hold about `rejected clusters x 2 x min_pts_per_cluster`
points, which grows quickly in high dimension with tight tolerances. If a level
would not fit, training stops with a `MemoryError` explaining what to change
(looser tolerance, smaller `min_pts_per_cluster`, `max_levels`, more ranks/nodes).

## Results, status and saving

```python
surrogate.N_src, surrogate.X_src, surrogate.Qs_src    # leaves: count, centroids, trust shapes
surrogate.leaf_params                                  # the local model's parameters by name
surrogate.E_cluster_abs_src, surrogate.E_cluster_rel_src  # accepted errors per leaf and output
surrogate.counts_src                                   # points per leaf
surrogate.converged                                    # False if max_levels stopped refinement
surrogate.num_unresolved_clusters, surrogate.num_unresolved_pts
surrogate.config                                       # every effective setting and component
surrogate.profiling_stats                              # counters, level statistics, profile

path = surrogate.save_model("results/model")           # -> results/model.npz (versioned, no pickle)
loaded = AFINNISurrogate.load_model("results/model")   # the same path forms as save_model
```

A partial run (stopped by `max_levels`) warns and reports the unresolved points;
a run that accepted nothing raises. Models with custom local models load if the
model's class is registered (import its module first) or passed as
`load_model(path, local_model=...)`. Under MPI, `save_model` writes on rank 0
only (call it on every rank or just rank 0), and `load_model` reads the file on
every rank, so it must be on a filesystem they all see.

Scoring predictions: `src.surrogate.compute_error_metrics(preds, truth, outside,
atol=..., rtol=...)` returns L2/Linf errors overall and inside/outside the trust
regions, and counts the inside points within tolerance under the same rule as training.

## ART (adaptive regression trees)

The repository has two algorithms on one refinement engine. `algorithm="afinni"` (the
default) is AFINNI's adaptive clustering: a forest of clusters, kNN plus trust-region
prediction. `algorithm="art"` is ART: an explicit oblique regression tree over a box
domain, built on the same engine (batched level-by-level loop, devices, MPI distribution,
work stealing, profiling and monitor).

`ClusteringConfig(algorithm="art", domain=(lower, upper), ...)` needs the domain
explicitly, because it is what routing covers and what `audit` samples
(`domain="data_bbox"` uses the training points' bounding box, and is recorded as such).
Settings are in `art=ARTSettings(...)`:

| Setting | Meaning |
|---|---|
| `roots` | `"box"` (one root, as in the reference ART; no initial clustering), `"grid"` (`num_initial_clusters` cells over the domain), or `"voronoi"` (the cells of the initial clustering's centres, for comparisons with AFINNI's forest) |
| `split_search` | how a rejected node's hyperplane is chosen: `"soft"` (default; threshold search, then the reference ART's soft-oblique optimizer in a consistent form, kept only if its hard fit is better), `"bins"` (threshold search over `n_directions` directions × `n_bins` cuts only), `"gap"` (the split strategy's direction, cut near the median) |
| `sampling` | `"hit_and_run"` (default), or `"convex_combination"`: AFINNI's convex-combination augmentation, to compare predictors on one tree |
| `store_Q` | keep the trust-region shape per leaf (needed only by `predictor="knn"`) |
| `sample_multiplier` | every node is topped up to `ceil(multiplier * (d + 1))` training points before it is fitted (reference ART: 50) |
| `validation_samples` | fresh points per node on which the deployed model is checked; a diagnostic |
| `min_side_margin` | a split leaves at least `d + 1 + margin` points on each side |
| `walk_steps`, `max_walk_steps`, `mixing_z` | hit-and-run steps per new point (box and grid roots are sampled exactly); chains that still drift between their halfway and last step are lengthened up to `max_walk_steps`; chains that did not move are rerun with coordinate directions; a node still drifting, stuck or with an infeasible step is marked `insufficient_sampling` (training warns) |
| `n_bins`, `n_directions`, `n_screened`, `soft_iters`, `soft_temperature` | the threshold search and the soft optimizer (defaults: 16, 6, 3, 40, 0.1) |
| `steal_min_load`, `steal_size` | work-stealing thresholds in work units: points after top-up and validation (default: a few nodes) |

Each rejected node is split by a hyperplane (see `split_search`), cut where the stored
predicate reproduces the children exactly. `error_metric="pointwise"` with `rtol` alone is
ART's stopping rule (every point within `rtol`). A node at `max_levels`, or one that cannot be split, keeps its fitted model as a
leaf, so every point of the domain has a leaf:

```python
surrogate.predict(X)                     # routed: one leaf per query; outside = outside the domain
surrogate.predict(X, predictor="knn")    # AFINNI's kNN / trust-region predictor still works
surrogate.leaf_status_counts()           # terminal_reason (accepted, max_depth, split_failed), validation_status
surrogate.route_details(X)               # per query: leaf, outside, terminal_reason, validation_status codes
surrogate.audit(Evaluator(f_fn=f), 3000) # violations (NaN/inf count as violations) on fresh uniform points, with a binomial upper bound
```

Leaves are affine (`local_model="affine_lsq"`, the default) or full quadratics
(`local_model="quadratic_lsq"`, the reference ART's `--leaf-degree 2`). Each node samples
`sample_multiplier` times the leaf model's coefficient count, as the reference ART does, and splits fit
the children in that model's basis. Quadratic leaves need far fewer nodes on smooth
targets; their `(d+1)(d+2)/2` coefficients suit low dimension.

Random numbers come from streams keyed by node and every training reduction sums each
node on its own, so an ART tree is the same for any batch size, rank count and
steal schedule, and the same in every run on one backend (CPU and GPU trees differ only at
rounding level). `scripts/art_comparison.py` trains the
reference ART code and this ART on the reference's benchmark problems and scores both on the
same test set; pass `--sampler parallel_numba` to time against the reference's fastest CPU
configuration, and add `--traversal bfs --optimizer cupy_batched_cholesky` for its fastest GPU
configuration. `--skip-art` / `--skip-afinni` run one side only; this ART alone also runs
under `mpiexec`. GPU results are in Sections 42.9-42.10 of `AFINNI_ART_DESIGN.md`: on the GPU,
this ART's B4 is 1-30x faster than the reference's GPU configuration, and faster than this
ART on the CPU whenever the CPU training takes more than about 1.5 s.

### Example: both algorithms on four benchmark problems

[`example04_art.py`](example04_art.py) compares the two algorithms on four
problems, each scored on the same uniform test points:

| `--problem` | What it tests | Domain |
|---|---|---|
| `spherical_piecewise` | a jump across a circle (the reference ART's TACC preset, `src/problems/spherical_piecewise.py`) | `[-3, 3]^d` |
| `anisotropic_gaussian` | smooth, rotated and strongly anisotropic | `[-3, 3]^d` |
| `wave` | a cosine along an oblique direction | `[-5, 5]^d` |
| `rosenbrock` | a narrow curved valley whose minimum is 0 (with `atol`) | `[-2, 3]^d` |

Each problem's depth limits (and, for `wave`, ART's training tolerance) are set so
both algorithms reach about the same relative L2 error on one CPU each; leaves, oracle
evaluations and training time then compare like for like. Both are scored with the
problem's tolerance: a point is above it when `|f - f_hat| > max(atol, rtol |f|)`.

```bash
uv run python -m examples.example04_art --problem all           # every problem + summary
uv run python -m examples.example04_art --problem wave          # one problem
uv run python -m examples.example04_art afinni --problem wave   # one stage (also: art, compare)
uv run python -m examples.example04_art --list                  # problems and their limits
uv run python -m src.infra.monitor scratch/example04_art/wave/stream/art   # or .../afinni
```

In 2-D, one process on one CPU (AFINNI / ART; about 30 s for all four):

| Problem | Relative L2 | % above tol | Leaves | Oracle evaluations | Training time |
|---|---|---|---|---|---|
| `spherical_piecewise` (AFINNI 7 levels, ART depth 12) | 0.095 / 0.049 | 1.7 / 0.47 | 416 / 433 | 363k / 66k | 0.59 / 1.7 s |
| `anisotropic_gaussian` (both converge) | 0.0018 / 0.0018 | 0.12 / 0.08 | 222 / 167 | 110k / 25k | 0.11 / 0.69 s |
| `wave` (both converge; ART at rtol 0.02) | 0.0043 / 0.0042 | 3.0 / 3.0 | 805 / 77 | 214k / 12k | 0.30 / 0.36 s |
| `rosenbrock` (AFINNI converges; ART depth 16) | 0.0024 / 0.0028 | 3.0 / 2.9 | 4,235 / 2,069 | 1.1M / 314k | 1.4 / 8.3 s |

ART needs 3.5-18x fewer oracle evaluations (AFINNI's include its 84,500 initial
points) at a similar or lower error, and fewer leaves on every problem but the spherical
jump, where it is twice as accurate; most of all it saves on the oblique `wave`. AFINNI
trains as fast or faster, since ART spends its time optimizing splits rather than
calling the oracle, so ART pays off when the oracle is expensive. Times are on a
laptop on battery. Notes per problem:

- `spherical_piecewise`: neither algorithm converges. A cluster straddling the jump fails
  AFINNI's test at every size, so its depth limit stops it and the unresolved clusters
  leave regions without leaves ("pout"); ART's remaining error comes from the few
  cells the jump still cuts.
- `wave`: converged, ART is about 2x more accurate (L2 0.0019 at rtol 0.01), so
  it trains at rtol 0.02 to match AFINNI's error.
- `rosenbrock`: the L2 errors agree for any ART depth from 10 on; depth 16 also
  matches the share above tolerance.

`--afinni-max-levels`, `--art-max-depth` and `--tol` override a problem's settings,
`--dim` its dimension (the limits are matched in 2-D only) and `--search` ART's
split search. The example runs in one process (`AFINNI_MPI` defaults to `off`), so the
training times compare directly. Outputs go to
`scratch/example04_art/<problem>/` (both models, `metrics.json`, the plots of
[Plots](#plots)); `--problem all` also writes `summary.json` and `summary.png`.

## Profiling

Three levels, all off by default (level statistics are always recorded):

| Flag | Records | Cost |
|---|---|---|
| `profile_timing=True` | Seconds per phase per rank (`profiling_stats["per_rank"][r]["time_*"]`) | Synchronizes the GPU around each phase |
| `profile_events=True` | A timeline: phase intervals with level/batch, steal requests/grants/denials, queue loads, level summaries (`per_rank[r]["events"]`) | Small; repeats are coalesced |
| `profile_dir="runs/x"` | The same events, streamed live to `runs/x/rank_<r>.jsonl` | Same as events |

```python
config = ClusteringConfig(atol=0.02, profile_timing=True, profile_events=True, profile_dir="runs/rosen")
surrogate.train(X, config, Evaluator(f_fn=f))
ps = surrogate.profiling_stats
ps["time_total"], ps["num_levels"], ps["total_f_evals"], ps["num_generated_pts"]
ps["level_stats"]          # one dict per (rank, level): clusters, accepted, points, runtime, ...
ps["per_rank"][0]["time_oracle"]   # time spent inside your f / g
```

Phases, their display groups and colours are defined once in
[`src/infra/profiling.py`](../src/infra/profiling.py) (`PHASES`); add a phase there and every
plot and the monitor pick it up. Profiles are saved with the model and, by the
launcher, under `profiling_stats` in each result JSON.

## Plots

All plots live in [`src/plotting/`](../src/plotting/) and accept a surrogate, its
`profiling_stats`, a saved profile JSON loaded with `json.load`, or (timeline plots)
the event streams. Each returns the figure and saves it when given a path.
[`example03_monitoring.py`](example03_monitoring.py) produces every profile plot
below from one multi-rank run.

```python
from src import plotting as P

P.plot_profile_dashboard(surrogate, "figs/dashboard.png")     # one-page summary
P.plot_rank_time_breakdown(surrogate, "figs/breakdown.png")   # needs profile_timing
P.plot_level_heatmap(surrogate, "figs/levels.png", value="clusters_start")
P.plot_level_progress(surrogate, "figs/growth.png")
P.plot_timeline(surrogate, "figs/timeline.png")               # needs profile_events / profile_dir
P.plot_load_over_time(surrogate, "figs/load.png")
P.plot_steal_matrix(surrogate, "figs/steals.png")             # needs steals (MPI)
```

| Plot | Shows | Needs |
|---|---|---|
| `plot_rank_time_breakdown` | Share (or seconds, `normalize=False`) of each rank's time per phase group, with total time per rank | `profile_timing` |
| `plot_level_heatmap` | One level statistic per level and rank: `clusters_start`, `good_clusters`, `num_pts_next_level`, `generated_pts`, `size_mean`, `runtime_sec`, ... | always |
| `plot_level_progress` | Accepted vs split clusters per level, points queued and generated | always |
| `plot_timeline` | Gantt chart per rank coloured by phase, with victim to thief steal arrows | events |
| `plot_load_over_time` | Queued points per rank over time, steal grants marked | events |
| `plot_steal_matrix` | Points (or `value="n_clusters"`) moved from each victim to each thief | events, steals |
| `plot_profile_dashboard` | Breakdown, growth, load and steals, whichever are available | any |
| `plot_adaptive_clustering_results` | 2-D leaves and trust ellipsoids over the function | 2-D problems |
| `plot_error_histogram` | Pointwise relative errors | predictions |
| `plot_art_partition` | A ART tree's cells coloured by depth, terminal reason and pointwise error, with the split boundaries | 2-D ART model |
| `plot_art_tree_stats` | Leaves per depth by terminal reason, and validation statuses | ART model |
| `plot_error_maps` | Pointwise relative error of several models side by side, with the tolerance contour | 2-D problems |
| `plot_error_cdf` | Distribution of pointwise relative errors of several models, tolerance marked | predictions |
| `plot_method_comparison` | Bar panels of errors, pout, leaves, evaluations and time for several methods | metric dicts |

Profile plots work for both algorithms: titles name the plan, and the forest-growth panels
show ART's fallback leaves (nodes stopped at the depth limit) separately.
[`example04_art.py`](example04_art.py) produces every comparison and
ART plot, for each of its problems.

The tables behind the profile plots are available for your own analysis:
`P.phase_breakdown_frame`, `P.level_stats_frame`, `P.events_frame`,
`P.steals_frame` (all `pandas.DataFrame`).

To plot a finished run from its files:

```python
import json
from src import plotting as P
from src.infra.profiling import read_event_streams

P.plot_profile_dashboard(json.load(open("results/rosenbrock/<run>.json")))
P.plot_timeline(read_event_streams("runs/rosen"))
```

Adding a plot: put it in `src/plotting/profiling.py` (training profiles),
`src/plotting/surrogate.py` (surrogates and errors) or `src/plotting/art.py` (ART
trees and comparisons of several surrogates), accept the inputs above via
`profile_of`, draw inside `afinni_style()`, take an optional `path` and return the
figure; export it from `src/plotting/__init__.py`.

## Live monitoring

Train with `profile_dir`, then watch from another terminal.
[`example03_monitoring.py`](example03_monitoring.py) is a ready-made run: 8 MPI
ranks on the CPU train a 5-D Rosenbrock surrogate (2M points, tol 0.007; about 85 s
on a 22-core laptop). Ranks finish their own work at different times, and the idle
ones steal from the busy ones (~60 steals).

```bash
# terminal 1: train (8 ranks, CPU)
uv run mpiexec -n 8 python -m examples.example03_monitoring
# terminal 2: live view; exits when every rank is done
uv run python -m src.infra.monitor scratch/example03_monitoring/stealing/stream
# afterwards (or any time): one snapshot
uv run python -m src.infra.monitor scratch/example03_monitoring/stealing/stream --once

# the same run without work stealing, for comparison
uv run mpiexec -n 8 python -m examples.example03_monitoring --stealing off
uv run python -m src.infra.monitor scratch/example03_monitoring/no_stealing/stream
```

Per rank the monitor shows the current phase (or idle / asking for work / retiring /
done), level, batch progress in the level (`current/planned`), queued points with a
load bar, clusters, accepted leaves, generated points, idle share, work given to
and taken from other ranks (`Gave work` / `Got work`: number of steals and points
moved), and how long ago the rank last reported (`Last updated`, relative to
the newest event of any rank), plus recent steals and level ends. Watch for ranks
that run out of work (idle, "asking R2 for work") and the steal counts and load bars
changing as work moves.

At the end the example writes `profile.json` and every profile plot (see
[Plots](#plots)) to `scratch/example03_monitoring/stealing/` or `.../no_stealing/`,
depending on `--stealing on|off`. Comparing the two timelines and time breakdowns
shows what work stealing buys: without it, ranks that finish their own share early
sit idle until the slowest rank is done. `--dim`, `--n-train`, `--tol`, `--init`
and `--metric` change the problem.

The header names the plan (`AFINNI`, or `ART (<split search> splits,
<roots> roots)`). In ART, nodes stopped at the depth limit keep a fallback leaf, so
they are counted as "in fallback leaves", not as queued work.

The monitor only reads the stream files, so it adds no MPI traffic and works from a
login node when the run directory is on a shared file system. The newest idle
stretch of a rank appears when that rank records its next event. For your own runs,
set `profile_dir="<dir>"` in the config and point the monitor at that directory.

## Paper experiments (launcher)

[`afinni_launcher.py`](../afinni_launcher.py) runs the benchmark problems and writes
one JSON result per run (plus error histogram and profile dashboard; the profile is
stored in the JSON) under `results/<problem>/`.

```bash
uv run python afinni_launcher.py --experiment rosenbrock --dims 2 5 --tols 0.1 0.05
uv run mpiexec -n 4 python afinni_launcher.py --experiment anisotropic_quadratic --dims 5 \
    --tols 0.02 --error_eval_mode taylor_cov_hess --sampling modified
uv run python afinni_launcher.py --experiment rosenbrock --dims 2 \
    --atols none 1e-3 --rtols 0.05 none      # 3 runs: every pair except none/none
```

Options: `--experiment` (`rosenbrock`, `anisotropic_quadratic`, `anisotropic_gaussian`,
`degenerate_quadratic`, `indef_degen_quadratic`, `lorenz96`, `wave`), `--dims`,
`--tols` (each run with atol = rtol), or `--atols` / `--rtols` (every combination; a
scalar applies to every output component, `none` leaves that test out; not allowed
together with `--tols`), `--error_eval_mode` (a mode value), `--sampling uniform|modified` (initial
data), `--sampling_method` (augmentation sampler), `--N_train`,
`--num_initial_clusters`, `--min_pts_per_cluster` (default: the dimension heuristics
`get_NUM_INITIAL_CLUSTERS` / `get_MIN_PTS_PER_CLUSTER`), `--Q_reg`,
`--coarse_factor`, `--dt` (Lorenz flow map). For lorenz96, atol is multiplied by
max|f_j| per output and `--tols` runs atol only. An existing result file means "done"
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
| Local model | `src.components.local_models.LocalModel` | `LOCAL_MODELS` | from `mode` |
| Split strategy | `src.components.splitting.SplitStrategy` | `SPLIT_STRATEGIES` | from `mode` |
| Sampler | `src.components.sampling.Sampler` | `SAMPLERS` | `sampling_method` |
| Acceptance | `src.components.acceptance.AcceptanceCriterion` | `ACCEPTANCE_CRITERIA` | `error_metric`, `atol`, `rtol` |
| Trust region | `src.components.trust_regions.TrustRegion` | `TRUST_REGIONS` | `CovarianceEllipsoid(q_reg)` |

Each is a `ClusteringConfig` field and accepts an
instance, a registered name, or `(name, settings)`:

```python
from src.components.splitting import CovarianceSplit

config = ClusteringConfig(
    mode=SurrogateMode.AFFINE_COV_HESS,
    split_strategy=CovarianceSplit(cut="sign"),          # instance: the paper's sign cut
    sampler="convex_combination",                         # registered name
    acceptance=("L2", {"atol": 1e-3, "rtol": 1e-2}),      # (name, settings)
)
surrogate.train(X, config, Evaluator(f_fn=f))
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
from src.core.clusters import ParamSpec
from src.components.local_models import LOCAL_MODELS, AffineLSQModel, LocalModel


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


surrogate.train(X, ClusteringConfig(atol=0.02, local_model=("offset_affine", {"offset": 0.5})), Evaluator(f_fn=f))
```

Contracts the engine checks (with a clear error):

- **Local model:** `fit` + `finalize` return exactly the declared `columns` with
  the declared shapes; the names `centroids`, `f`, `Q`, `E_cluster_abs`,
  `E_cluster_rel`, `counts` are reserved. `evaluate` may read `params["centroids"]` and `params["f"]`, and
  must also work without the `finalize` columns: the acceptance test evaluates the
  `fit` output, and only accepted leaves get `finalize` columns.
- **Sampler:** `sample(clusters, n_new, rng)` returns `sum(n_new)` points grouped by
  cluster; `clusters` holds `X`, `counts`, `starts`, `centroids`; use only `rng`
  for randomness.
- **Split strategy:** implement `metric(data) -> (M, M C M)` to change the split
  direction, `child_sizes(counts, sorted_scores)` to change where clusters are cut,
  or `split(data)` entirely; splits are binary and must keep each cluster's rows
  together and its two child sizes adding up to its size.
- **Acceptance:** `__call__(F, F_hat, starts, sizes) -> (is_good, E_abs, E_rel)`, the
  errors per cluster and output component (K, fd); `self.accept(E_abs, E_rel)` applies
  the built-in tolerance rule.
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
| `RuntimeWarning: Training stopped at max_levels=...` | Refinement hit the depth limit before every cluster met the tolerance; raise `max_levels` or loosen the tolerance (see `surrogate.converged`). |

## Development

Conventions for code, docstrings and verification are in [`AGENTS.md`](../AGENTS.md).

```bash
uv sync --all-groups             # also the research, research_internal, viz and profiling groups
make help                        # Linux/macOS wrappers, e.g. `make dev EXTRAS="gpu mpi"`
```

| Dependency group | Contents                                      | Installed by        |
| ---------------- | --------------------------------------------- | ------------------- |
| `dev`            | `lint` + `test` + `notebook`                  | `uv sync` (default) |
| `research`       | extra packages for the scripts in `research/`, incl. cuML for comparisons | `--group research`  |
| `research_internal` | `knndbscan` (TestPyPI, Linux) for `initial_clustering_method="knndbscan"` | `--group research_internal` |
| `profiling`      | py-spy, line_profiler, viztracer, memray      | `--group profiling` |
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
which backends were exercised (`afinni backend: array=NumPy | MPI=...`). A
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
