#!/bin/bash
# scripts/bench_inference.py on Vista: the GPU run on a GH200 node (gh), the CPU run on a
# Grace-Grace node (gg), one process each. Submit from the repository root after the one-time
# install (docs/USAGE.md, "HPC clusters"); extra arguments go to the script. Resubmitting the same
# command reuses the cases already finished with the same settings.
#
#   sbatch -p gh -N 1 -n 1 -t 02:00:00 scripts/hpc/bench_inference_vista.sbatch gpu
#   sbatch -p gg -N 1 -n 1 -t 02:00:00 scripts/hpc/bench_inference_vista.sbatch cpu
#   sbatch -p gh -N 1 -n 1 -t 00:30:00 scripts/hpc/bench_inference_vista.sbatch gpu --problem rosenbrock --dim 2
#SBATCH -J bench-inference
#SBATCH -o bench-inference-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/hpc/tacc.sh
device="${1:-gpu}"
shift || true
case "$device" in
    gpu|cpu) ;;
    *) echo "usage: bench_inference_vista.sbatch gpu|cpu [bench_inference.py arguments]" >&2; exit 2 ;;
esac
# One process, no MPI: inference and training times are those of a single CPU or GPU.
SURROGATE_DEVICE="$device" SURROGATE_MPI=off .venv/bin/python scripts/bench_inference.py "$@"
