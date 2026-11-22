# Adaptive surrogates of expensive functions

This repository builds fast surrogates of expensive functions from as few evaluations of the function as
possible, in two families that share the problems, the benchmark and the infrastructure:

- **The adaptive surrogate** (`AdaptiveSurrogate`, `src/surrogate/`): AFINNI and ART refine a set of
  local models (affine or quadratic) until each meets a tolerance, on the refinement engine, on CPU or
  GPU and across MPI ranks. They differ in how they divide the input space and how a query finds its
  model; choose one with `TrainingConfig(algorithm=...)`.
- **The schemes** (`src/schemes/`): global approximants (ChebTT, ChebTT-M and ChebTT-A, ChebTree,
  SparseCheb, the hub expansion, MetricKernel, SlowTable) that sample the function where their own
  error estimate asks and fit one model, aiming at the worst pointwise error within the tolerance. They
  are NumPy on the CPU, standalone from the engine, and spread only their oracle calls over MPI ranks.
  See [docs/SCHEMES.md](docs/SCHEMES.md).

[Choosing a method](#choosing-a-method) compares them on the benchmark.

### AFINNI: adaptive affine nearest-neighbour interpolation

`algorithm="afinni"` (the default). The training points are clustered, and every cluster whose
local model misses the tolerance is augmented and bisected along a curvature-informed
direction, giving a forest of accepted clusters. Each accepted cluster carries a trust
ellipsoid; a query is answered by the nearest leaf in trust-region distance, and flagged when
it lies outside every trust region. Suited to fast training from a given sample.

### ART: adaptive regression trees

`algorithm="art"`. An oblique binary tree of convex polytopes over an explicit box domain:
every node is topped up with fresh points inside its polytope, and every node whose model
misses the tolerance is cut by an optimized plane. A query is routed down the tree, so
every point of the domain has exactly one leaf. Suited to expensive functions: fewer
evaluations for the same error, and a model that covers the whole domain.

AFINNI and ART run on one **refinement engine** (`src/engine/`): the partition is refined breadth-first,
level by level, with the queued nodes kept in host memory and streamed to the CPU or GPU in
batches, distributed across MPI ranks with work stealing. Local models, acceptance tests,
devices, MPI, profiling, the live monitor and model files are shared.

The code runs on **Linux, macOS and Windows**. It always works on the CPU, and can
optionally use **MPI** (multiple processes/nodes, with work stealing) and/or an
**NVIDIA GPU** (CUDA 12). Whatever is installed is detected at runtime, and
everything else falls back automatically.

## Installation

You need Git and [`uv`](https://docs.astral.sh/uv/getting-started/installation/), a
Python package and environment manager; `uv` installs a suitable Python if needed.

```bash
git clone https://github.com/ut-padas/affini.git
cd affini
uv sync                       # CPU backend + dev tools (pytest, ruff, Jupyter)
```

Optional backends are uv *extras*; combine them freely, e.g. `uv sync --extra gpu --extra mpi`.

| You want             | Command                                  | Notes                                                                                                   |
| -------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| CPU only             | `uv sync`                                | All platforms.                                                                                          |
| CPU + MPI            | `uv sync --extra mpi`                    | Ships its own MPI runtime (MPICH on Linux/macOS, Intel MPI on Windows), so no system install is needed. |
| NVIDIA GPU (CUDA 12) | `uv sync --extra gpu`                    | Linux and Windows. Only an NVIDIA driver is needed, because the CUDA libraries come from pip. |
| GPU + MPI            | `uv sync --extra gpu --extra mpi`        |                                                                                                         |
| HPC cluster          | `uv sync --extra gpu --extra mpi-system` | Uses the cluster's own MPI. See [HPC clusters](docs/USAGE.md#hpc-clusters).                        |

> [!IMPORTANT]
> `uv sync` makes `.venv` match exactly what you request: running it again without
> `--extra gpu` removes the GPU packages. `uv run` never removes packages. Add
> `--no-dev` for a runtime-only environment.

|             | Linux (incl. WSL2) | macOS (Apple Silicon, 14+)           | Windows                                                             |
| ----------- | ------------------ | ------------------------------------ | ------------------------------------------------------------------- |
| CPU         | ✅                  | ✅ verified on Apple Silicon          | ✅                                                                   |
| MPI (`mpi`) | ✅                  | ✅ bundled MPICH                     | ✅ Intel MPI ([note](docs/USAGE.md#running-on-cpu-gpu-and-mpi))  |
| GPU (`gpu`) | ✅ CuPy             | ❌ no CUDA, runs on CPU               | ✅ CuPy                                                              |

Verified so far: Linux x86_64 (CPU, MPICH), WSL2 and Windows with an NVIDIA laptop GPU
(CPU, GPU, MPI), TACC Vista (GH200, ARM64: CPU, GPU, Open MPI) and Stampede3 (H100,
Intel MPI).

## Quick start

```bash
uv run python -m src                                               # which backends (GPU / MPI) were detected
uv run python -m examples.example01_rosenbrock_afinni              # 2-D Rosenbrock surrogate: error + plot
uv run mpiexec -n 8 python -m examples.example03_monitoring_afinni  # multi-rank run with work stealing...
uv run python -m src.infra.monitor scratch/example03_monitoring_afinni/stealing/stream  # ...watched live
uv run python -m examples.example04_wave_art                       # ART on its own: train, route, audit, save
uv run python -m examples.example05_afinni_vs_art --problem all    # AFINNI vs ART on 4 problems
```

```python
import numpy as np

from src import xp
from src.config import TrainingConfig
from src.core.evaluation import Evaluator
from src.surrogate import AdaptiveSurrogate


def f(x):  # (N, d) -> (N, fd); x is a NumPy or CuPy array, matching the backend
    return (xp.sin(3 * x[:, 0]) * xp.cos(2 * x[:, 1]) + x[:, 0] ** 2)[:, None]


rng = np.random.default_rng(0)
X, X_query = rng.uniform(-1, 1, (2_000, 2)), rng.uniform(-1, 1, (500, 2))

# AFINNI: adaptive clustering, predicted by the nearest leaf in trust-region distance
afinni = AdaptiveSurrogate()
afinni.train(X, TrainingConfig(atol=1e-2, rtol=1e-2), Evaluator(f_fn=f))
prediction, outside_trust_region = afinni.predict(X_query)

# ART: an oblique regression tree over an explicit box domain, predicted by routing
art = AdaptiveSurrogate()
art.train(X, TrainingConfig(algorithm="art", domain=([-1, -1], [1, 1]), atol=1e-2, rtol=1e-2, max_levels=16),
          Evaluator(f_fn=f))
prediction, outside_domain = art.predict(X_query)
```

ART needs its `domain` explicitly: it is what routing covers. Its settings (split search,
roots, sampling) are in `art=ARTSettings(...)`; see
[ART in USAGE.md](docs/USAGE.md#art-adaptive-regression-trees).

A scheme is one call, on host arrays: `build(f, lo, hi, atol, **settings)`, with `f` returning (N,) values
(or (N, m) for a vector function), and optionally the oracle's Jacobian:

```python
from src.schemes import base, chebtt, hub

g = lambda x: np.sin(3 * x[:, 0]) * np.cos(2 * x[:, 1]) + x[:, 0] ** 2  # host (N, d) -> (N,)
model = chebtt.build(g, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), atol=1e-2, seed=0)
prediction = model.predict(X_query)
print(model.evals, model.report.summary())  # evaluations, time per phase, rounds
base.save(model, "chebtt-model.npz")  # base.load(...) predicts bit for bit as trained
ridge = hub.build(g, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), atol=1e-2, jacobian=None)  # or the oracle's (N, d) Jacobian
```

Run scripts from the repository root, as modules
(`python -m examples.example01_rosenbrock_afinni`), so that `src` is importable.

## Choosing a method

The benchmark (`src/benchmark/`, [docs/BENCHMARK.md](docs/BENCHMARK.md)) judges every method by its
worst error over the tolerance on a fixed test set, with the oracle evaluations it spent. What its
records show so far, per method (counts from the table below; the hub expansion's from the iteration
studies in [docs/SURROGATE_ITERATIONS.md](docs/SURROGATE_ITERATIONS.md), since it has no benchmark runs yet):

| Method | Use it for | Strengths | Weaknesses |
|---|---|---|---|
| ChebTT | smooth and vector-valued functions with low-rank coupling, any d | holds 90 of 91 smooth and 93 of 98 vector cases, 30+ of them at d >= 10; inference 0.05-0.3 us | oblique or curved features cost rank; its kink and jump records predate its piecewise axes |
| ChebTT-M | outputs that are ridges of shared oblique coordinates | wins 35 of 45 cases with many local features | the rotated box extends beyond the domain; on the axes it is ChebTT |
| ChebTT-A | a jump across an oblique plane or a quadric, a sharp front | holds the oblique jump at all three tolerances (2-D, 1.7k evaluations), where ChebTT held 3 of 16 non-smooth cases | one interface only (a plane or a diagonal quadric); run on that one problem only |
| SparseCheb | smooth functions of few interacting variables, low d | cheapest on smooth problems in 2-D and 3-D (25 of 42 wins) | inference 1-2 us, 23 us on jumps; not run above 3-D |
| ChebTree | kinks, jumps and fronts in low d | holds all 10 kink, jump and front cases it ran | tensor grids: 2-3-D only; 17k median evaluations there |
| Hub expansion | a few steep directions, weak dependence on the rest; uses Jacobians when given | 10-20-D latent-variable functions at 4-12k evaluations; a Jacobian about halves them | misses functions outside that class (Lorenz-96's state: 1.5 tol); no benchmark runs yet |
| MetricKernel | 2-D smooth functions | few evaluations where it holds (median 257) | holds 10 of 21 smooth cases; run in 2-D only |
| SlowTable | step maps queried on visited states (in-situ tabulation) | needs no box: tabulates the states a simulation visits | holds 9 of 20 stiff cases; inference 19 us; refuses what its ellipsoids miss |
| ART | expensive functions where every query must route to a model; GPU, MPI | covers the whole domain; fewest evaluations on easy flow maps (median 81, affine leaves) | holds 13 (affine leaves) and 33 (quadratic) of about 280 function cases at max-norm tolerances |
| AFINNI | training from a given sample; GPU, MPI with work stealing | trust regions flag queries outside the training data | holds 11 of 224 function cases at max-norm tolerances; 10^5 evaluations median |

The benchmark's counts, from its store (`python -m src.benchmark table --write README.md` regenerates them):

<!-- benchmark-table -->
**Function suite (8 CPU ranks, 3 seeds)**

| class | method | ran | held | won | held at d >= 10 | median evaluations (held) | median inference us (held) |
|---|---|---|---|---|---|---|---|
| smooth | ChebTT-A | 97 | 96 | 47 | 36 | 1.3k | 0.093 |
|  | ChebTT-M | 97 | 96 | 47 | 36 | 1.3k | 0.13 |
|  | SparseCheb | 42 | 42 | 27 | 0 | 337 | 1.3 |
|  | Hub | 105 | 87 | 14 | 35 | 3.0k | 1.7 |
|  | ChebTree | 42 | 42 | 9 | 0 | 757 | 0.24 |
|  | ART, quadratic leaves | 91 | 10 | 2 | 9 | 4.6k | 0.43 |
|  | MetricKernel | 21 | 10 | 2 | 0 | 257 | 1.4 |
|  | ChebTT | 97 | 96 | 1 | 36 | 1.5k | 0.096 |
|  | AFINNI | 63 | 2 | 0 | 0 | 110k | 0.43 |
|  | ART, affine leaves | 90 | 0 | 0 | 0 | - | - |
| many local features | ChebTT-M | 70 | 69 | 61 | 24 | 2.9k | 0.52 |
|  | ChebTT-A | 70 | 68 | 43 | 23 | 2.7k | 0.55 |
|  | SparseCheb | 9 | 9 | 4 | 0 | 433 | 2.5 |
|  | ChebTree | 30 | 29 | 3 | 0 | 17k | 0.34 |
|  | Hub | 75 | 14 | 1 | 4 | 2.6k | 4.7 |
|  | ChebTT | 70 | 69 | 0 | 24 | 2.9k | 0.54 |
|  | AFINNI | 38 | 0 | 0 | 0 | - | - |
|  | ART, affine leaves | 37 | 0 | 0 | 0 | - | - |
|  | ART, quadratic leaves | 37 | 0 | 0 | 0 | - | - |
| kinks, jumps, fronts | ChebTT-A | 66 | 61 | 55 | 16 | 4.3k | 0.16 |
|  | ChebTT-M | 39 | 34 | 28 | 10 | 4.3k | 0.23 |
|  | Hub | 72 | 6 | 6 | 6 | 5.0k | 3.8 |
|  | ChebTT | 39 | 34 | 0 | 10 | 4.7k | 0.31 |
|  | ChebTree | 26 | 24 | 0 | 0 | 48k | 0.23 |
|  | SparseCheb | 9 | 4 | 0 | 0 | 43k | 23 |
|  | AFINNI | 39 | 0 | 0 | 0 | - | - |
|  | ART, affine leaves | 36 | 0 | 0 | 0 | - | - |
|  | ART, quadratic leaves | 33 | 0 | 0 | 0 | - | - |
| ODE flow maps | ChebTree | 18 | 18 | 7 | 0 | 757 | 0.14 |
|  | ART, affine leaves | 27 | 7 | 6 | 0 | 81 | 0.077 |
|  | ChebTT | 39 | 37 | 5 | 12 | 1.6k | 0.12 |
|  | ChebTT-A | 37 | 34 | 5 | 10 | 1.2k | 0.1 |
|  | Hub | 45 | 28 | 5 | 7 | 2.6k | 3 |
|  | SparseCheb | 15 | 15 | 5 | 0 | 201 | 0.68 |
|  | ChebTT-M | 39 | 36 | 3 | 12 | 1.5k | 0.12 |
|  | ART, quadratic leaves | 27 | 10 | 3 | 2 | 201 | 0.12 |
|  | AFINNI | 21 | 6 | 0 | 0 | 110k | 0.3 |
|  | MetricKernel | 3 | 3 | 0 | 0 | 268 | 1.4 |
| vector-valued | ChebTT | 101 | 96 | 71 | 34 | 2.5k | 0.31 |
|  | ChebTT-A | 97 | 92 | 65 | 32 | 2.4k | 0.37 |
|  | ChebTT-M | 104 | 101 | 62 | 39 | 2.8k | 0.36 |
|  | Hub | 101 | 72 | 19 | 22 | 2.6k | 6.1 |
|  | ART, affine leaves | 92 | 6 | 6 | 3 | 251 | 0.29 |
|  | ART, quadratic leaves | 94 | 13 | 4 | 7 | 1.3k | 0.46 |
|  | AFINNI | 63 | 3 | 0 | 0 | 110k | 0.41 |

**Step-map suite (1 CPU core, rollouts of held-out trajectories)**

| class | method | ran | held | won | held at d >= 10 | median evaluations (held) | median inference us (held) |
|---|---|---|---|---|---|---|---|
| stiff | ChebTT, one basis per axis (no edge search) | 37 | 27 | 21 | 5 | 3.8k | 0.55 |
|  | ChebTT, piecewise axes | 37 | 27 | 18 | 5 | 3.8k | 0.63 |
|  | ChebTT-A | 37 | 27 | 16 | 5 | 3.8k | 0.64 |
|  | SlowTable | 20 | 9 | 5 | 0 | 2.6k | 19 |
|  | ChebTT, retrained until its rollouts hold | 35 | 25 | 3 | 4 | 5.1k | 0.63 |
|  | ART, affine leaves | 17 | 4 | 1 | 0 | 2.2k | 0.6 |
|  | ART, quadratic leaves | 19 | 8 | 0 | 0 | 2.4k | 0.83 |
|  | AFINNI | 15 | 0 | 0 | 0 | - | - |
| nonstiff | ChebTT, piecewise axes | 7 | 5 | 5 | 1 | 7.2k | 1.3 |
|  | ChebTT, one basis per axis (no edge search) | 7 | 5 | 5 | 1 | 7.2k | 1.1 |
|  | ChebTT-A | 7 | 5 | 4 | 1 | 7.2k | 0.96 |
|  | ChebTT, retrained until its rollouts hold | 7 | 5 | 1 | 1 | 7.2k | 1.4 |
|  | ART, quadratic leaves | 5 | 1 | 0 | 0 | 4.5k | 1.6 |
|  | SlowTable | 4 | 1 | 0 | 0 | 20k | 2.4e+02 |
|  | AFINNI | 3 | 0 | 0 | 0 | - | - |
|  | ART, affine leaves | 3 | 0 | 0 | 0 | - | - |

A case is a problem, dimension and tolerance; *held*: worst error within the tolerance on every seed; *won*: held with the fewest evaluations (ties count for each). Medians are over each method's own held cases, which differ in difficulty between methods. *Ran* differs between methods because the sweeps stop a method at its first failure on a family. No record is older than its method's last change to its results (`RESULTS_CHANGED` in `src/benchmark/table.py`).
<!-- /benchmark-table -->

## Documentation

- **[docs/reports/adaptive-surrogates/adaptive-surrogates.pdf](docs/reports/adaptive-surrogates/adaptive-surrogates.pdf)**: the technical
  companion. It explains AFINNI, ART and the refinement engine on one running example, each idea
  as a picture, in words and as a formal statement. Start here to understand the methods.
- **[docs/reports/global-approximants/global-approximants.pdf](docs/reports/global-approximants/global-approximants.pdf)**: the
  companion of the schemes. It starts from the theory of where a surrogate's error comes from (approximation,
  sampling, propagation, inputs off the data, and the refusal rate's cap on the speed-up), explains the strongest
  schemes it chose, ChebTT and its variants on a box and the chart recipe on trajectories, shows them on every class
  of the function suite and the step maps, and compares them with AFINNI and ART under one protocol.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: how a training run and a prediction move
  through the code, the contracts between the parts, and where to extend it.
- **[docs/USAGE.md](docs/USAGE.md)**: the user guide. Training and prediction
  on CPU, GPU and MPI; settings and modes; ART; saving models; profiling; plots; live
  monitoring; swapping parts of the algorithm (local models,
  split rules, samplers, acceptance tests, trust regions); HPC clusters;
  troubleshooting; development and testing.
- **[examples/](examples)**: runnable examples; `example04_wave_art.py` introduces ART and
  `example05_afinni_vs_art.py` compares AFINNI and ART on four problems.
- **[studies/manuscript/](studies/manuscript/README.md)**: the paper's experiments, reproducible
  on any machine with one command (plan, run, collect into data files and tables).
- **[docs/AFINNI_ART_DESIGN.md](docs/AFINNI_ART_DESIGN.md)**: the mathematics of both approaches, the
  refinement engine, and the design and implementation log.
- **[docs/SCHEMES.md](docs/SCHEMES.md)**: the schemes (ChebTT, ChebTree, SparseCheb, the hub expansion, ...): what
  each does, its contract, settings, profiling and model files.
- **[docs/BENCHMARK.md](docs/BENCHMARK.md)**: the benchmark of every method on test functions and ODE step
  maps: its store of runs, running a suite and sweeping a grid.
- **[docs/SURROGATE_RESEARCH.md](docs/SURROGATE_RESEARCH.md)** and
  **[docs/SURROGATE_ITERATIONS.md](docs/SURROGATE_ITERATIONS.md)**: the experiments behind the schemes, what each
  taught, and the literature reviews.
- **[AGENTS.md](AGENTS.md)**: code conventions and verification steps for contributors
  (human or AI).

## Code layout

`src/` is layered: each layer imports only from itself or the layers above it in this table
(`tests/unit/test_layering.py` enforces it).

| Layer | Packages / modules |
|---|---|
| infrastructure, maths, shared types | `infra/` (backend detection, profiling, monitor, logging, `io` for generated data, `records`: the records every long run keeps its cases in), `numerics/` (`segments`, `linalg`, `stats`, `grid`), `kernels/` (`.cu` sources), `core/` (`clusters`: queues and output columns; `evaluation`: the oracle `Evaluator`; `plan`: the contract between the refinement loop and an algorithm plan) |
| settings, components | `config.py` (settings and size heuristics), `components/` (registry, modes, local models, splitting, acceptance, trust regions, sampling), `neighbors/` (kNN, k-means) |
| algorithms | `afinni/` (`plan`: `AFINNIPlan`), `art/` (`ARTPlan` in `plan`; `constants`, `rng`, `geometry`, `sampling`, `split_search`, `routing`); `schemes/` (the global approximants: ChebTT, ChebTree, SparseCheb, the hub expansion, ...; standalone, importing only the infrastructure layer) |
| engine | `engine/` (`driver`: `refine`, collective training; `refinement`: `refine_rank`, the loop shared by both plans; `initial`, `work_stealing`) |
| public API | `surrogate/` (`model`, `persistence`, `metrics`); also `plotting/`, `problems/` (test functions, one module per family, registered in `registry`: `FUNCTIONS`; `ode/`: ODE systems and their `STEP_MAPS`) |
| benchmark | `benchmark/` (every method on the function and step-map suites, one store of runs, the sweep; see docs/BENCHMARK.md) |

Package `__init__` files re-export the names other modules import, so callers do not depend
on the file a name lives in (`from src.art import ARTPlan`). CUDA kernel sources live in
`src/kernels/*.cu` and are compiled at runtime (no build step).

## Development

```bash
uv run pytest -m "not mpi"      # tests (see docs/USAGE.md for GPU/MPI and the regression check)
uv run ruff check .             # lint
uv run ruff format .            # format
```

## Roadmap

- [x] Verify CPU and MPI on macOS (Apple Silicon).
- [ ] Rename the `src` package to `adaptive_surrogates` (src layout) and add a `[build-system]`, so it can be installed with `pip install adaptive-surrogates`.
- [ ] Publish `knndbscan` to PyPI. It is a core dependency but comes only from TestPyPI, so `adaptive-surrogates`
  cannot be published to PyPI before it.
- [ ] Set up GitHub Actions: `{ubuntu, macos, windows} × {cpu, mpi}` running `uv lock --check`, pytest and `mpiexec -n 2`.
- [ ] Publish `adaptive-surrogates` to PyPI.
