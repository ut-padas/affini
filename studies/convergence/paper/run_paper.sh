#!/usr/bin/env bash
# Part A: re-run the AFINNI manuscript's experiments (see CONVERGENCE_STUDY.md, Part A).
#
#   studies/convergence/paper/run_paper.sh [--smoke] [--device gpu|cpu] [--max-tier S|M|L]
#       [--groups conv,conv_nh,conv_k10,lorenz_dims,rosen_weak,wave_exact,iterative,mpi,seeds]
#       [--seeds 0,1,2] [--nps 1,2,4,8] [--mpi-device cpu|gpu] [--lorenz-device auto|cpu|gpu]
#       [--only CASE_REGEX] [--out DIR] [--dry-run] [--no-plots]
#
# Defaults: --device gpu (cpu with --smoke), --max-tier S, every group, --mpi-device cpu (the
# manuscript's MPI runs were on Grace CPU nodes).
#
# Results: $OUT/results/seed_<s>/<problem>/*.json (the launcher's files; iterative runs under
# $OUT/results/iterative), logs in $OUT/logs, done-markers in $OUT/done. Rerunning skips finished
# cases, so an interrupted sweep continues where it stopped. Finish with
#   $PYTHON studies/convergence/paper/collect_paper.py --out $OUT
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SMOKE=0 DEVICE="" MAX_TIER=S RUN_GROUPS="" SEEDS="0,1,2" NPS="1,2,4,8" MPI_DEVICE=cpu LORENZ_DEVICE=auto DRY=0 OUT="" ONLY="" NO_PLOTS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1 ;;
        --device) DEVICE="$2"; shift ;;
        --max-tier) MAX_TIER="$2"; shift ;;
        --groups) RUN_GROUPS="$2"; shift ;;
        --seeds) SEEDS="$2"; shift ;;
        --nps) NPS="$2"; shift ;;
        --mpi-device) MPI_DEVICE="$2"; shift ;;
        --lorenz-device) LORENZ_DEVICE="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --only) ONLY="$2"; shift ;;
        --dry-run) DRY=1 ;;
        --no-plots) NO_PLOTS=1 ;;  # the launcher's diagnostic figures (records read only the JSON)
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) log "unknown argument $1"; exit 2 ;;
    esac
    shift
done
if [[ $SMOKE == 1 ]]; then
    OUT="${OUT:-$STUDY_OUT/paper_smoke}"; DEVICE="${DEVICE:-cpu}"; NPS="1,2"
else
    OUT="${OUT:-$STUDY_OUT/paper}"; DEVICE="${DEVICE:-gpu}"
fi
if [[ $DRY == 0 ]]; then
    check_device "$DEVICE"
    [[ "$MPI_DEVICE" != "$DEVICE" ]] && check_device "$MPI_DEVICE"
fi

# Lorenz-96 runs the batched ensemble integrator on either backend (the same algorithm, float64,
# rtol = atol = 1e-6); "auto" follows --device.
[[ "$LORENZ_DEVICE" == auto ]] && LORENZ_DEVICE="$DEVICE"

CASE_ARGS=(--max-tier "$MAX_TIER" --seeds "$SEEDS" --nps "$NPS")
[[ $SMOKE == 1 ]] && CASE_ARGS=(--smoke --nps "$NPS")
[[ -n "$RUN_GROUPS" ]] && CASE_ARGS+=(--groups "$RUN_GROUPS")
[[ -n "$ONLY" ]] && CASE_ARGS+=(--only "$ONLY")
[[ $DRY == 0 ]] && mkdir -p "$OUT"/{results,logs,done}
[[ $DRY == 0 ]] && record_provenance "$OUT"
log "Part A -> $OUT (device=$DEVICE, mpi-device=$MPI_DEVICE, lorenz-device=$LORENZ_DEVICE, max tier $MAX_TIER, smoke=$SMOKE)"

N_CASES=0
while IFS=$'\t' read -r case_id group tier np seed kind args_rest; do
    read -r -a args <<<"$args_rest"
    N_CASES=$((N_CASES + 1))
    dev="$DEVICE"
    [[ "$group" == mpi ]] && dev="$MPI_DEVICE"
    [[ " ${args[*]} " == *" lorenz96 "* ]] && dev="$LORENZ_DEVICE"
    if [[ "$kind" == iterative ]]; then
        cmd=("$PYTHON" "$STUDY_DIR/paper/run_iterative.py" --results "$OUT/results/iterative" "${args[@]}")
    else
        cmd=("$PYTHON" "$STUDY_DIR/paper/run_case.py" --results "$OUT/results/seed_$seed" --seed "$seed" --case "$case_id" -- "${args[@]}" $([[ $NO_PLOTS == 1 ]] && echo --no_plots))
    fi
    mpi=off
    if [[ "$np" -gt 1 ]]; then
        mpi=on
        # shellcheck disable=SC2206
        cmd=($MPIEXEC -n "$np" "${cmd[@]}")
    fi
    if [[ $DRY == 1 ]]; then
        echo "[$tier] SURROGATE_DEVICE=$dev SURROGATE_MPI=$mpi ${cmd[*]}"
        continue
    fi
    # Run from the repository root: the launcher and the iterative script write side files to the CWD.
    (FAILED_RUNS=0; cd "$REPO_ROOT" && SURROGATE_DEVICE="$dev" SURROGATE_MPI="$mpi" \
        run_logged "$OUT/done/$case_id.done" "$OUT/logs/$case_id.log" "${cmd[@]}"
     exit $FAILED_RUNS) || FAILED_RUNS=$((FAILED_RUNS + 1))
done < <("$PYTHON" "$STUDY_DIR/paper/cases.py" "${CASE_ARGS[@]}")

log "Part A: $N_CASES cases, $FAILED_RUNS failed"
[[ $DRY == 1 ]] && exit 0
"$PYTHON" "$STUDY_DIR/paper/collect_paper.py" --out "$OUT" || log "collection failed"
exit $(( FAILED_RUNS > 0 ))
