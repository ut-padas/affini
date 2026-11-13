# Adaptive surrogates of expensive functions: AFINNI and ART

This repository builds fast surrogates of smooth, expensive functions from as few evaluations
of the function as possible. It provides two approaches; both refine a set of local models
(affine or quadratic) until each meets a tolerance, and differ in how they divide the input
space and how a query finds its model. Choose one with `TrainingConfig(algorithm=...)`.

### AFINNI: adaptive affine nearest-neighbour interpolation

`algorithm="afinni"` (the default). The training points are clustered, and every cluster whose
local model misses the tolerance is augmented and bisected along a curvature-informed
direction, giving a forest of accepted clusters. Each accepted cluster carries a trust
ellipsoid; a query is answered by the nearest leaf in trust-region distance, and flagged when
it lies outside every trust region. Suited to fast training from a given sample.

### ART: adaptive regression trees

`algorithm="art"`. An oblique binary tree of convex polytopes over an explicit box domain:
every node is topped up with fresh points inside its polytope, and every node whose model
misses the tolerance is cut by an optimized hyperplane. A query is routed down the tree, so
every point of the domain has exactly one leaf. Suited to expensive functions: fewer
evaluations for the same error, and a model that covers the whole domain.

Both run on one **refinement engine** (`src/engine/`): the partition is refined breadth-first,
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
| NVIDIA GPU (CUDA 12) | `uv sync --extra gpu`                    | Linux and Windows. Only an NVIDIA driver is needed, because the CUDA libraries come from pip.           |
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
from src.surrogate import AFINNISurrogate


def f(x):  # (N, d) -> (N, fd); x is a NumPy or CuPy array, matching the backend
    return (xp.sin(3 * x[:, 0]) * xp.cos(2 * x[:, 1]) + x[:, 0] ** 2)[:, None]


rng = np.random.default_rng(0)
X, X_query = rng.uniform(-1, 1, (2_000, 2)), rng.uniform(-1, 1, (500, 2))

# AFINNI: adaptive clustering, predicted by the nearest leaf in trust-region distance
afinni = AFINNISurrogate()
afinni.train(X, TrainingConfig(atol=1e-2, rtol=1e-2), Evaluator(f_fn=f))
prediction, outside_trust_region = afinni.predict(X_query)

# ART: an oblique regression tree over an explicit box domain, predicted by routing
art = AFINNISurrogate()
art.train(X, TrainingConfig(algorithm="art", domain=([-1, -1], [1, 1]), atol=1e-2, rtol=1e-2, max_levels=16),
          Evaluator(f_fn=f))
prediction, outside_domain = art.predict(X_query)
```

ART needs its `domain` explicitly: it is what routing covers. Its settings (split search,
roots, sampling) are in `art=ARTSettings(...)`; see
[ART in USAGE.md](docs/USAGE.md#art-adaptive-regression-trees).

Run scripts from the repository root, as modules
(`python -m examples.example01_rosenbrock_afinni`), so that `src` is importable.

## Documentation

- **[docs/USAGE.md](docs/USAGE.md)**: the user guide. Training and prediction
  on CPU, GPU and MPI; settings and modes; ART; saving models; profiling; plots; live
  monitoring; the experiment launcher; swapping parts of the algorithm (local models,
  split rules, samplers, acceptance tests, trust regions); HPC clusters;
  troubleshooting; development and testing.
- **[examples/](examples)**: runnable examples; `example04_wave_art.py` introduces ART and
  `example05_afinni_vs_art.py` compares AFINNI and ART on four problems.
- **[docs/AFINNI_ART_DESIGN.md](docs/AFINNI_ART_DESIGN.md)**: the mathematics of both approaches, the
  refinement engine, and the design and implementation log.
- **[AGENTS.md](AGENTS.md)**: code conventions and verification steps for contributors
  (human or AI).

## Code layout

`src/` is layered: each layer imports only from itself or the layers above it in this table
(`tests/unit/test_layering.py` enforces it).

| Layer | Packages / modules |
|---|---|
| infrastructure, maths, shared types | `infra/` (backend detection, profiling, monitor, logging, `io` for generated data), `numerics/` (`segments`, `linalg`, `stats`, `grid`), `kernels/` (`.cu` sources), `core/` (`clusters`: queues and output columns; `evaluation`: the oracle `Evaluator`; `plan`: the contract between the refinement loop and an algorithm plan) |
| settings, components | `config.py` (settings and size heuristics), `components/` (registry, modes, local models, splitting, acceptance, trust regions, sampling), `neighbors/` (kNN, k-means) |
| algorithms | `afinni/` (`plan`: `AFINNIPlan`), `art/` (`ARTPlan` in `plan`; `constants`, `rng`, `geometry`, `sampling`, `split_search`, `routing`) |
| engine | `engine/` (`driver`: `refine`, collective training; `refinement`: `refine_rank`, the loop shared by both plans; `initial`, `work_stealing`) |
| public API | `surrogate/` (`model`, `persistence`, `metrics`); also `plotting/`, `problems/` |

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
- [ ] Rename the `src` package to `afinni` (src layout) and add a `[build-system]`, so it can be installed with `pip install`.
- [ ] Move `jax`/`diffrax` to an optional extra. `afinni_launcher.py` currently imports `jax` at top level.
- [ ] Publish `knndbscan` to PyPI. It currently comes only from TestPyPI and Linux.
- [ ] Set up GitHub Actions: `{ubuntu, macos, windows} × {cpu, mpi}` running `uv lock --check`, pytest and `mpiexec -n 2`.
- [ ] Publish `afinni` to PyPI.
