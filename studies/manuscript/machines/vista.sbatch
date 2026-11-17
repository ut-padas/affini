#!/bin/bash
# The manuscript study on Vista: GPU cases on a GH200 node, the distributed experiment on a
# Grace-Grace node. Resubmit the same command to continue; finished cases are not run again.
#
#   sbatch -p gh -t 48:00:00 studies/manuscript/machines/vista.sbatch gh
#   sbatch -p gg -t 48:00:00 studies/manuscript/machines/vista.sbatch gg
#SBATCH -N 1
#SBATCH -J manuscript-study
#SBATCH -o manuscript-study-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/hpc/tacc.sh
case "${1:-gh}" in
    gh) .venv/bin/python -m studies.manuscript.run run --machine studies/manuscript/machines/vista-gh.toml \
            --experiments convergence,dimension,absolute,gradient,progression,iterative ;;
    gg) .venv/bin/python -m studies.manuscript.run run --machine studies/manuscript/machines/vista-gg.toml \
            --experiments distributed ;;
    *) echo "usage: vista.sbatch gh|gg" >&2; exit 2 ;;
esac
.venv/bin/python -m studies.manuscript.collect
