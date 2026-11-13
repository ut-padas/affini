# `AFINNI` - Adaptive Affine Nearest Neighbor Interpolation

`AFINNI` builds fast surrogates of smooth, expensive functions: it adaptively clusters
samples of the function, refines clusters until a local affine model meets a
tolerance, and answers queries with the best local model by trust-region distance.

The repository implements two algorithms on one **refinement engine** (`src/engine/`):
**AFINNI**, the adaptive clustering above (`algorithm="afinni"`, the default), and
**ART**, adaptive regression trees: an oblique tree of polytopes over a box domain,
queried by routing (`algorithm="art"`). The engine runs either one breadth-first, level by
level, streaming the queued nodes from host memory to the CPU or GPU in batches, across
MPI ranks with work stealing.

It runs on **Linux, macOS and Windows**. It always works on the CPU, and can
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
| HPC cluster          | `uv sync --extra gpu --extra mpi-system` | Uses the cluster's own MPI. See [HPC clusters](examples/USAGE.md#hpc-clusters).                        |

> [!IMPORTANT]
> `uv sync` makes `.venv` match exactly what you request: running it again without
> `--extra gpu` removes the GPU packages. `uv run` never removes packages. Add
> `--no-dev` for a runtime-only environment.

|             | Linux (incl. WSL2) | macOS (Apple Silicon, 14+)           | Windows                                                             |
| ----------- | ------------------ | ------------------------------------ | ------------------------------------------------------------------- |
| CPU         | ✅                  | ✅ verified on Apple Silicon          | ✅                                                                   |
| MPI (`mpi`) | ✅                  | ✅ bundled MPICH                     | ✅ Intel MPI ([note](examples/USAGE.md#running-on-cpu-gpu-and-mpi))  |
| GPU (`gpu`) | ✅ CuPy             | ❌ no CUDA, runs on CPU               | ✅ CuPy                                                              |

Verified so far: Linux x86_64 (CPU, MPICH), WSL2 and Windows with an NVIDIA laptop GPU
(CPU, GPU, MPI), TACC Vista (GH200, ARM64: CPU, GPU, Open MPI) and Stampede3 (H100,
Intel MPI).

## Quick start

```bash
uv run python -m src                                  # which backends (GPU / MPI) were detected
uv run python -m examples.example01                   # 2-D Rosenbrock surrogate: error + plot
uv run mpiexec -n 8 python -m examples.example03_monitoring   # multi-rank run with work stealing...
uv run python -m src.infra.monitor scratch/example03_monitoring/stream   # ...watched live from another terminal
uv run python -m examples.example04_art --problem all   # AFINNI vs ART on 4 problems
```

```python
from src.engine import ClusteringConfig, Evaluator
from src.surrogate import AFINNISurrogate

surrogate = AFINNISurrogate()
surrogate.train(X, ClusteringConfig(atol=1e-2, rtol=1e-2), Evaluator(f_fn=f))
prediction, outside_trust_region = surrogate.predict(X_query)
```

Run scripts from the repository root, as modules (`python -m examples.example01`), so
that `src` is importable.

## Documentation

- **[examples/USAGE.md](examples/USAGE.md)**: the user guide. Training and prediction
  on CPU, GPU and MPI; settings and modes; saving models; profiling; plots; live
  monitoring; the experiment launcher; swapping parts of the algorithm (local models,
  split rules, samplers, acceptance tests, trust regions); HPC clusters;
  troubleshooting; development and testing.
- **[examples/](examples/)**: runnable examples.
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
uv run pytest -m "not mpi"      # tests (see USAGE.md for GPU/MPI and the regression check)
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
