#!/usr/bin/env bash
# Part B: the reference ART vs this library's ART on the reference's problems (CONVERGENCE_STUDY.md, Part B).
#
#   studies/convergence/art/run_art.sh [--smoke] [--device cpu|gpu] [--ref-device cpu|gpu]
#       [--ref-sampler hit_and_run|parallel_numba] [--max-tier S|M|L]
#       [--groups conv,library_default,depth,exact_target,mpi,repeat,gpu_vs_cpu] [--seeds 1,2,3]
#       [--nps 1,2,4] [--only CASE_REGEX] [--no-exact] [--out DIR] [--dry-run]
#
# Defaults: --device gpu (cpu with --smoke) for this library's side; --ref-device cpu; the reference's
# default sampler (hit_and_run); --max-tier S. --ref-device gpu runs the reference with its
# fastest GPU configuration (--traversal bfs --optimizer cupy_batched_cholesky; sampling, oracle
# and leaf fits stay on its CPU). Results: $OUT/results/<group>/<case>.<side>.json, logs in
# $OUT/logs, the root-fit and slab checks in $OUT/results/exact_checks.json. Rerunning skips
# finished runs. Finish with $PYTHON studies/convergence/art/collect_art.py --out $OUT.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SMOKE=0 DEVICE="" REF_DEVICE=cpu REF_SAMPLER="" MAX_TIER=S RUN_GROUPS="" SEEDS="1,2,3" NPS="1,2,4" ONLY="" EXACT=1 OUT="" DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1 ;;
        --device) DEVICE="$2"; shift ;;
        --ref-device) REF_DEVICE="$2"; shift ;;
        --ref-sampler) REF_SAMPLER="$2"; shift ;;
        --max-tier) MAX_TIER="$2"; shift ;;
        --groups) RUN_GROUPS="$2"; shift ;;
        --seeds) SEEDS="$2"; shift ;;
        --nps) NPS="$2"; shift ;;
        --only) ONLY="$2"; shift ;;
        --no-exact) EXACT=0 ;;
        --out) OUT="$2"; shift ;;
        --dry-run) DRY=1 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) log "unknown argument $1"; exit 2 ;;
    esac
    shift
done
if [[ $SMOKE == 1 ]]; then
    OUT="${OUT:-$STUDY_OUT/art_smoke}"; DEVICE="${DEVICE:-cpu}"; NPS="1,2"
else
    OUT="${OUT:-$STUDY_OUT/art}"; DEVICE="${DEVICE:-gpu}"
fi
[[ -d "$ART_ROOT/art" ]] || { log "error: reference ART not found at $ART_ROOT (set ART_ROOT)"; exit 2; }
[[ $DRY == 0 ]] && check_device "$DEVICE"
REF_ARGS=()
[[ "$REF_DEVICE" == gpu ]] && { [[ $DRY == 0 ]] && check_device gpu; REF_ARGS+=(--traversal bfs --optimizer cupy_batched_cholesky); }
[[ -n "$REF_SAMPLER" ]] && REF_ARGS+=(--sampler "$REF_SAMPLER")

CASE_ARGS=(--max-tier "$MAX_TIER" --seeds "$SEEDS" --nps "$NPS")
[[ $SMOKE == 1 ]] && CASE_ARGS=(--smoke --nps "$NPS")
[[ -n "$RUN_GROUPS" ]] && CASE_ARGS+=(--groups "$RUN_GROUPS")
[[ -n "$ONLY" ]] && CASE_ARGS+=(--only "$ONLY")
[[ "$DEVICE" == gpu ]] && CASE_ARGS+=(--gpu)
[[ $DRY == 0 ]] && mkdir -p "$OUT"/{results,logs,done}
if [[ $DRY == 0 ]]; then
    record_provenance "$OUT"
    echo "reference: device=$REF_DEVICE extra=${REF_ARGS[*]:-none}" >>"$OUT/provenance.txt"
fi
log "Part B -> $OUT (this library's ART on $DEVICE, reference on $REF_DEVICE ${REF_ARGS[*]:-}, max tier $MAX_TIER, smoke=$SMOKE)"

if [[ $EXACT == 1 && $DRY == 0 ]]; then
    exact_args=(--art-path "$ART_ROOT" --out-json "$OUT/results/exact_checks.json")
    [[ $SMOKE == 1 ]] && exact_args+=(--smoke)
    # exact_checks.py exits non-zero when a check fails; the failure is a finding, so record it as done.
    (FAILED_RUNS=0; cd "$REPO_ROOT" && SURROGATE_DEVICE="$DEVICE" SURROGATE_MPI=off \
        run_logged "$OUT/done/exact_checks.done" "$OUT/logs/exact_checks.log" \
        bash -c '"$@"; s=$?; [[ $s -le 1 ]]' _ "$PYTHON" "$STUDY_DIR/art/exact_checks.py" "${exact_args[@]}")
fi

N_RUNS=0
while IFS=$'\t' read -r case_id group tier side np device out_json args_rest; do
    read -r -a args <<<"$args_rest"
    N_RUNS=$((N_RUNS + 1))
    dev="$DEVICE"
    [[ "$device" == cpu ]] && dev=cpu
    extra=()
    [[ "$side" == reference ]] && { dev=cpu; extra=("${REF_ARGS[@]}"); }  # the reference ignores SURROGATE_DEVICE
    name="$(basename "$out_json" .json)"
    cmd=("$PYTHON" "$STUDY_DIR/art/run_art_case.py" --side "$side" --art-path "$ART_ROOT"
         --out-json "$OUT/results/$out_json" --case "$case_id" -- "${args[@]}" "${extra[@]}")
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
    (FAILED_RUNS=0; cd "$REPO_ROOT" && SURROGATE_DEVICE="$dev" SURROGATE_MPI="$mpi" \
        run_logged "$OUT/done/$group.$name.done" "$OUT/logs/$group/$name.log" "${cmd[@]}"
     exit $FAILED_RUNS) || FAILED_RUNS=$((FAILED_RUNS + 1))
done < <("$PYTHON" "$STUDY_DIR/art/art_cases.py" "${CASE_ARGS[@]}")

log "Part B: $N_RUNS runs, $FAILED_RUNS failed"
[[ $DRY == 1 ]] && exit 0
"$PYTHON" "$STUDY_DIR/art/collect_art.py" --out "$OUT" || log "collection failed"
exit $(( FAILED_RUNS > 0 ))
