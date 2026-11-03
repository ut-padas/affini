# `AFINNI` - Adaptive Affine Nearest Neighbor Interpolation
Code for the `AFINNI` algorithm.

AFINNI runs on **Linux, macOS and Windows**. It always works on the CPU, and can optionally use **MPI** (multiple processes/nodes) and/or an **NVIDIA GPU** (CUDA 12). Whatever is installed is detected at runtime, and everything else falls back automatically.

## Quick start

You need [`uv`](https://docs.astral.sh/uv/getting-started/installation/), a Python package and environment manager. It installs a suitable Python for you if needed.

```bash
git clone https://github.com/ut-padas/affini.git
cd affini
uv sync                                 # creates .venv: CPU backend + dev tools (pytest, ruff, Jupyter)
uv run python -m src                    # shows which backends (GPU / MPI) were detected
uv run python -m examples.example01     # 2-D Rosenbrock surrogate, prints the error and saves a plot
```

Run scripts from the repository root, as modules (`python -m examples.example01`), so that `src` is importable.

## Choosing a backend

Optional backends are uv *extras*. Combine them freely, for example `uv sync --extra gpu --extra mpi`.

| You want             | Command                                  | Notes                                                                                                   |
| -------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| CPU only             | `uv sync`                                | All platforms.                                                                                          |
| CPU + MPI            | `uv sync --extra mpi`                    | Ships its own MPI runtime (MPICH on Linux/macOS, Intel MPI on Windows), so no system install is needed. |
| NVIDIA GPU (CUDA 12) | `uv sync --extra gpu`                    | Linux and Windows. Only an NVIDIA driver is needed, because the CUDA libraries come from pip.           |
| GPU + MPI            | `uv sync --extra gpu --extra mpi`        |                                                                                                         |
| HPC cluster          | `uv sync --extra gpu --extra mpi-system` | Uses the cluster's own MPI. See [HPC clusters](#hpc-clusters).                                          |

> [!IMPORTANT]
> `uv sync` makes `.venv` match exactly what you request.
> Running `uv sync` again without `--extra gpu` removes the GPU packages.

To get a runtime-only environment without the dev tools, add `--no-dev`.

### Platform support

|             | Linux         | macOS (Apple Silicon, 14+) | Windows                                                             |
| ----------- | ------------- | -------------------------- | ------------------------------------------------------------------- |
| CPU         | ✅             | ✅                          | ✅                                                                   |
| MPI (`mpi`) | ✅             | ✅                          | ✅ (untested)                                                        |
| GPU (`gpu`) | ✅ CuPy + cuML | ❌ no CUDA, runs on CPU     | ✅ CuPy only. KMeans/kNN fall back to scikit-learn/FAISS on the CPU. |

### Running with MPI

```bash
uv run mpiexec -n 4 python afinni_launcher.py --dims 2 --tols 0.1 --experiment rosenbrock
```
> [!IMPORTANT]
> Without `mpiexec`, or when MPI isn't installed, the same code runs as a single rank. If a script is started under `mpiexec` but MPI can't be loaded, AFINNI stops with an error. Otherwise every process would quietly run the whole job as rank 0.

### Overriding detection

| Variable        | Values                         | Effect                                                                                                 |
| --------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `AFINNI_DEVICE` | `auto` (default), `cpu`, `gpu` | `cpu` forces NumPy even when a GPU is present. `gpu` raises an error instead of silently falling back. |
| `AFINNI_MPI`    | `auto` (default), `off`, `on`  | `off` ignores an installed mpi4py. `on` raises an error if MPI is unavailable.                         |

For example: `AFINNI_DEVICE=cpu uv run python -m examples.example01`.

The same flags are available in code: `from src import IS_CUPY_BACKEND, HAS_CUML, HAS_MPI, xp`. Detection is implemented in [`src/backend.py`](src/backend.py).

## HPC clusters

On a cluster, use the `mpi-system` extra and **not** `mpi`. The pip-bundled MPICH would take priority over the cluster's MPI and its fast interconnect. Installing doesn't need any modules loaded. Running does, because mpi4py binds to the system MPI at runtime.

On TACC (Stampede3, Vista):

```bash
make install-hpc                 # = uv sync --no-dev --extra gpu --extra mpi-system
source scripts/hpc/tacc.sh       # loads gcc/cuda/MPI modules; add this line to your SLURM job scripts too
ibrun .venv/bin/python afinni_launcher.py ...
```

For another cluster, copy [`scripts/hpc/tacc.sh`](scripts/hpc/tacc.sh) to `scripts/hpc/<site>.sh` and change the `module load` lines.

## Development

```bash
uv run pytest                    # tests
uv run ruff check .              # lint
uv run ruff format .             # format
uv sync --all-groups             # also install research/ and visualisation extras (manim, plotly, optuna, ...)
```

On Linux and macOS, the [`Makefile`](Makefile) wraps these commands (`make help`). For example, `make dev EXTRAS="gpu mpi"` sets up a full GPU + MPI development environment. Every target runs a single `uv` command, so on Windows run that command directly.

| Dependency group | Contents                                      | Installed by        |
| ---------------- | --------------------------------------------- | ------------------- |
| `dev`            | `lint` + `test` + `notebook`                  | `uv sync` (default) |
| `research`       | extra packages for the scripts in `research/` | `--group research`  |
| `viz`            | animation/plotting (manim, plotly)            | `--group viz`       |

### Testing backends

`uv run pytest` runs everything that the current install supports. The header line shows which backends were actually exercised, for example `afinni backend: array=NumPy | cuML=no | MPI=...`.

- **[`tests/test_backend.py`](tests/test_backend.py)** checks backend detection, the overrides, the serial MPI stand-in and multiprocessing workers. It also includes an end-to-end smoke test that trains and predicts a small surrogate on whatever backend is active.
- **[`tests/mpi/`](tests/mpi/)** launches the same smoke test on several ranks and checks that it produces the same model as a serial run. These tests are skipped when MPI isn't installed.

By default, a missing backend is skipped. Set an override to make it required instead:

| Checking  | Command                                                                                                                                  |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| GPU       | `uv sync --extra gpu` then `AFINNI_DEVICE=gpu uv run pytest tests/test_backend.py`                                                       |
| MPI       | `uv sync --extra mpi` then `AFINNI_MPI=on uv run pytest tests/test_backend.py tests/mpi` (= `make test-mpi`)                             |
| GPU + MPI | `uv sync --extra gpu --extra mpi` then `AFINNI_DEVICE=gpu AFINNI_MPI=on uv run pytest tests/test_backend.py tests/mpi`                   |
| HPC       | `AFINNI_MPI=on AFINNI_MPIEXEC=ibrun uv run pytest tests/mpi` inside an allocation. `AFINNI_MPI_NP` sets the number of ranks (default 2). |

To exclude the multi-rank tests, add `-m "not mpi"`.

## Roadmap

- [ ] Rename the `src` package to `afinni` (src layout) and add a `[build-system]`, so it can be installed with `pip install`.
- [ ] Move `jax`/`diffrax` to an optional extra. `afinni_launcher.py` currently imports `jax` at top level.
- [ ] Publish `knndbscan` to PyPI. It currently comes only from TestPyPI and Linux.
- [ ] Set up GitHub Actions: `{ubuntu, macos, windows} × {cpu, mpi}` running `uv lock --check`, pytest and `mpiexec -n 2`.
- [ ] Publish `afinni` to PyPI.
