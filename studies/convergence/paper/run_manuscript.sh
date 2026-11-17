#!/usr/bin/env bash
# Every experiment of the AFINNI manuscript with the current code, then a tracked record of the
# numbers with their provenance (record_results.py; CONVERGENCE_STUDY.md, "Recording the results").
#
#   MACHINE="GH200 node" studies/convergence/paper/run_manuscript.sh [run_paper.sh options]
#
# Defaults: --device gpu, --max-tier L, the manuscript's groups (conv, conv_nh, lorenz_dims,
# rosen_weak, wave_exact, iterative, mpi, seeds; not the conv_k10 diagnostic), --out
# $STUDY_OUT/manuscript, --exact-outside-max-d 40 (p_o_exact blank above d = 40; pass none to run it
# everywhere). Options given here override them. The tree must have no modified tracked
# files, so the recorded commit describes the code that ran (ALLOW_DIRTY=1 to override). Rerunning
# skips finished cases, so an interrupted run continues; the record is written at the end of each
# invocation, from whatever has finished.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

: "${MACHINE:?set MACHINE to a neutral description of the machine, e.g. MACHINE=\"GH200 node\"}"
if [[ "${ALLOW_DIRTY:-0}" != 1 ]] && [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]]; then
    log "error: modified tracked files; commit them first so the recorded commit describes the code (ALLOW_DIRTY=1 overrides)"
    exit 2
fi
OUT="$STUDY_OUT/manuscript"
for ((i = 1; i <= $#; i++)); do
    [[ "${!i}" == --out ]] && { j=$((i + 1)); OUT="${!j}"; }
done
"$STUDY_DIR/paper/run_paper.sh" --device gpu --max-tier L \
    --groups conv,conv_nh,lorenz_dims,rosen_weak,wave_exact,iterative,mpi,seeds --out "$OUT" --no-plots --exact-outside-max-d 40 "$@"
[[ " $* " == *" --dry-run "* ]] && exit 0
"$PYTHON" "$STUDY_DIR/paper/record_results.py" --out "$OUT" --machine "$MACHINE"
