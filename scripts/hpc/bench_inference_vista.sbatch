#!/bin/bash
# The inference benchmarks on Vista: scripts/bench_inference.py, or the script named by
# BENCH_SCRIPT (e.g. scripts/profile_routing.py), on a GH200 node (gpu) or a Grace-Grace node
# (cpu), one process each. Submit from the repository root; extra arguments go to the script. The
# job installs its own environment first (uv sync, as in docs/USAGE.md, "HPC clusters"), so no
# interactive node is needed beforehand. Resubmitting the same command reuses the environment and
# the results already finished with the same settings.
#
#   sbatch -p gh -N 1 -n 1 -t 02:00:00 scripts/hpc/bench_inference_vista.sbatch gpu
#   sbatch -p gg -N 1 -n 1 -t 02:00:00 scripts/hpc/bench_inference_vista.sbatch cpu
#   sbatch -p gh -N 1 -n 1 -t 00:30:00 scripts/hpc/bench_inference_vista.sbatch gpu --problem rosenbrock --dim 2
#   BENCH_SCRIPT=scripts/profile_routing.py sbatch -p gh -N 1 -n 1 -t 01:00:00 \
#       scripts/hpc/bench_inference_vista.sbatch gpu --problem lorenz96 --dim 5
#SBATCH -J bench-inference
#SBATCH -o bench-inference-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
script="${BENCH_SCRIPT:-scripts/bench_inference.py}"
device="${1:-gpu}"
shift || true
case "$device" in
    gpu) extras=(--extra gpu --extra mpi-system) ;;
    cpu) extras=(--extra mpi-system) ;;
    *) echo "usage: [BENCH_SCRIPT=<script>] bench_inference_vista.sbatch gpu|cpu [script arguments]" >&2; exit 2 ;;
esac
[[ -f "$script" ]] || { echo "error: $script not found (run from the repository root)" >&2; exit 2; }

# The cluster's MPI first: mpi4py (the mpi-system extra) binds to it.
source scripts/hpc/tacc.sh

# uv's downloads and Python go to $WORK, not the small $HOME quota (unless already set).
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORK/uv-python}"
# uv sizes its install thread pool to the core count, which a per-user thread limit can refuse
# (uv then panics with "failed to initialize global rayon pool"); a few threads install as fast.
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-8}"
# Each device has its own environment, so a GPU and a CPU job running at once never install
# into the same directory; the shared uv cache keeps the second install fast.
export UV_PROJECT_ENVIRONMENT=".venv-vista-$device"
command -v uv >/dev/null || { echo "error: uv is not on PATH on $(hostname)" >&2; exit 1; }
echo "[bench-inference] uv sync into $UV_PROJECT_ENVIRONMENT on $(hostname) ($(date))"
uv sync --locked "${extras[@]}"
echo "[bench-inference] environment ready; running $script ($(date))"

# One process, no MPI: inference and training times are those of a single CPU or GPU.
SURROGATE_DEVICE="$device" SURROGATE_MPI=off "$UV_PROJECT_ENVIRONMENT/bin/python" "$script" "$@"
