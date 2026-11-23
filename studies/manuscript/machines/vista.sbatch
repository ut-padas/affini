#!/bin/bash
# The manuscript study on Vista: every GPU case on a GH200 node, the distributed experiment on a
# Grace-Grace node (up to 128 ranks, one thread each). Submit from the repository root, after the
# one-time setup in studies/manuscript/README.md. Resubmit the same command to continue: finished
# cases are not run again.
#
#   sbatch -p gh -N 1 -n 1   -t 48:00:00 studies/manuscript/machines/vista.sbatch gh
#   sbatch -p gg -N 1 -n 128 -t 48:00:00 studies/manuscript/machines/vista.sbatch gg
#   sbatch -p gh -N 1 -n 1   -t 00:30:00 studies/manuscript/machines/vista.sbatch gh-smoke   # first
#   sbatch -p gg -N 1 -n 4   -t 00:30:00 studies/manuscript/machines/vista.sbatch gg-smoke   # first
#
# EXPERIMENTS narrows a GPU job to some experiments; each case runs alone on the node's one GPU,
# without MPI. SHARD=i/n splits a GPU job's cases over n jobs (one GH200 node each, i = 0 .. n-1),
# all writing to the same results folder. Set both in front of sbatch, which passes the environment
# to the job: --export=EXPERIMENTS=a,b would split the list at its commas. The error experiments
# with AFINNI and ART, on two nodes:
#
#   for i in 0 1; do EXPERIMENTS=convergence,dimension,absolute SHARD=$i/2 \
#       sbatch -p gh -N 1 -n 1 -t 48:00:00 studies/manuscript/machines/vista.sbatch gh; done
#SBATCH -J manuscript-study
#SBATCH -o manuscript-study-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/hpc/tacc.sh
GPU=${EXPERIMENTS:-convergence,dimension,absolute,gradient,progression,iterative}
run() { .venv/bin/python -m studies.manuscript.run run "$@"; }
case "${1:-gh}" in
    gh)       run --machine studies/manuscript/machines/vista-gh.toml --experiments $GPU ${SHARD:+--shard $SHARD} ;;
    gg)       run --machine studies/manuscript/machines/vista-gg.toml --experiments distributed ;;
    gh-smoke) run --smoke --seeds 0 --machine studies/manuscript/machines/vista-gh.toml --experiments $GPU ;;
    gg-smoke) run --smoke --seeds 0 --machine studies/manuscript/machines/vista-gg.toml --experiments distributed ;;
    *) echo "usage: vista.sbatch gh|gg|gh-smoke|gg-smoke" >&2; exit 2 ;;
esac
.venv/bin/python -m studies.manuscript.run status --machine studies/manuscript/machines/vista-gh.toml | tail -5 || true
[[ "${1}" == *smoke ]] || .venv/bin/python -m studies.manuscript.collect
