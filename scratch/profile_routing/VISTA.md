# Testing the GPU routing kernels on Vista

How to check the CUDA routing kernels (`src/kernels/art_routing.cu`) on a Vista GH200 node and
measure their speed. The kernels have never run on a GPU: they compiled and matched NumPy only as
host C++ with the CUDA intrinsics stubbed out. So run step 1 first, and if it fails, keep the
NVRTC error message.

Run everything from the repository root on the `inference` branch, on a GPU node (`gh` partition):
an interactive `idev` session or the job script in [Without an interactive node](#without-an-interactive-node).

## 0. Set up

```bash
git fetch origin && git switch inference && git pull
source scripts/hpc/tacc.sh
export UV_CACHE_DIR=$WORK/.cache/uv UV_PYTHON_INSTALL_DIR=$WORK/uv-python UV_CONCURRENT_INSTALLS=8
UV_PROJECT_ENVIRONMENT=.venv-vista-gpu uv sync --locked --extra gpu --extra mpi-system
PY=.venv-vista-gpu/bin/python
```

If `.venv-vista-gpu` already exists from the benchmark job, the sync only checks it.

## 1. The kernels compile and agree with NumPy (a minute)

```bash
SURROGATE_DEVICE=gpu SURROGATE_MPI=off $PY -m pytest -q --no-cov \
    tests/unit/test_art.py tests/unit/test_kernel_sources.py -k "walk or plane_scores or routing or kernel"
```

These tests check:
- `test_plane_scores_have_the_bits_of_numpy_row_sums_on_every_backend`: the `plane_scores_rows`
  kernel equals NumPy's row sum to the bit for d up to 300;
- `test_the_walk_routes_every_query_where_the_level_loop_does`: `route` (the `route_prepare` and
  `route_walk` kernels) sends every query to the leaf that the level-by-level definition, computed
  in NumPy on the host, gives, for d up to 130, with one query exactly on a plane;
- `test_routing_rejects_non_finite_or_misshaped_queries`: a NaN or infinity raises;
- `test_kernel_sources.py`: every kernel has a Python launcher.

A compile error appears here first, as a `cupy` `CompileException` with NVRTC's message.

## 2. The whole GPU suite (several minutes)

```bash
SURROGATE_DEVICE=gpu SURROGATE_MPI=off $PY -m pytest -q --no-cov -m "not mpi"
```

GPU training now computes every ART plane score with the `plane_scores_rows` kernel instead of
CuPy's `sum`. GPU-trained trees can change at rounding level, and should move closer to the CPU's.
The GPU suite has passed in full before, so treat any failure as new.

## 3. Regression against the previous commit (`7894c9b`)

```bash
git worktree add ../affini-ref 7894c9b
(cd ../affini-ref && SURROGATE_DEVICE=gpu SURROGATE_MPI=off PYTHONPATH=. ../affini/$PY \
    scripts/regression_check.py --save $SCRATCH/routing_gpu_ref)
SURROGATE_DEVICE=gpu SURROGATE_MPI=off PYTHONPATH=. $PY scripts/regression_check.py \
    --save $SCRATCH/routing_gpu_new --compare $SCRATCH/routing_gpu_ref --rtol 1e-12
git worktree remove ../affini-ref
```

Adjust `../affini` to your clone's folder name if it differs. What to expect:
- every AFINNI case is unchanged, since only ART uses `plane_scores`;
- the ART cases have the same leaf counts and evaluation counts;
- ART arrays differ at most at rounding level. A larger difference means a split decision changed,
  so send the `FAIL` lines.

## 4. Speed (submit as a job; about an hour)

```bash
BENCH_SCRIPT=scripts/profile_routing.py sbatch -p gh -N 1 -n 1 -t 01:00:00 \
    scripts/hpc/bench_inference_vista.sbatch gpu --problem lorenz96 --dim 5 \
    --queries 50000 1000000 10000000 --chunks 5000 50000 500000 5000000
```

The first run trains the ART model (32,768 leaves). Later runs reuse it from `scratch/profile_routing/gpu/`.
For each query count, the log `bench-inference-<jobid>.out` shows:
- `route (unstaged)`: the new kernels, from start to finish;
- `level loop (def.)`: the level-by-level definition on the same GPU, which is what `a94ccfe` ran,
  with the speedup over it. The script stops if any query lands in a different leaf;
- the stages `prepare`, `root_lookup`, `walk` and `leaf_lookup`, then `evaluate`;
- `predict_route_local` at each chunk size and at the default. On the GPU the default is still
  5,000 queries per chunk; these numbers show whether a larger default is worth it.

Send the result in `scratch/profile_routing/gpu/lorenz96_d5.json`, or the log. For Rosenbrock,
submit the same command with `--problem rosenbrock`.

## Without an interactive node

Steps 0–3 as one job (save as `routing_gpu_check.sbatch` in the repository root and run
`sbatch routing_gpu_check.sbatch`):

```bash
#!/bin/bash
#SBATCH -J routing-gpu-check
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o routing-gpu-check-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/hpc/tacc.sh
export UV_CACHE_DIR=$WORK/.cache/uv UV_PYTHON_INSTALL_DIR=$WORK/uv-python UV_CONCURRENT_INSTALLS=8
UV_PROJECT_ENVIRONMENT=.venv-vista-gpu uv sync --locked --extra gpu --extra mpi-system || exit 1
PY=$PWD/.venv-vista-gpu/bin/python
export SURROGATE_DEVICE=gpu SURROGATE_MPI=off
echo "== 1. kernels"; $PY -m pytest -q --no-cov tests/unit/test_art.py tests/unit/test_kernel_sources.py \
    -k "walk or plane_scores or routing or kernel"
echo "== 2. GPU suite"; $PY -m pytest -q --no-cov -m "not mpi"
echo "== 3. regression"
git worktree add ../routing-ref 7894c9b
(cd ../routing-ref && PYTHONPATH=. $PY scripts/regression_check.py --save $SCRATCH/routing_gpu_ref)
PYTHONPATH=. $PY scripts/regression_check.py --save $SCRATCH/routing_gpu_new --compare $SCRATCH/routing_gpu_ref --rtol 1e-12
git worktree remove --force ../routing-ref
```

`set -e` is left out on purpose, so a failing step doesn't stop the later ones. Each step's result
is in `routing-gpu-check-<jobid>.out`. Submit the speed job (step 4) separately.
