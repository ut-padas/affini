#!/bin/bash
set -Eeuo pipefail

OUTPUT="results/rosenbrock_pout_race/dim5"
SKIP_PREFLIGHT=0
EXTRA_ARGS=()

usage() {
    printf '%s\n' \
        "Usage: ./run_rosenbrock_pout_race.sh [options]" \
        "" \
        "Aggressive all-survivor race to a target P_out. The default run uses" \
        "dimension 5, the old random-design regime, and a 2% target." \
        "" \
        "Options:" \
        "  --output PATH       result directory" \
        "  --skip-preflight    skip GPU and import checks" \
        "  --help              show this message" \
        "" \
        "Useful driver options passed through unchanged:" \
        "  --methods LIST              comma-separated methods" \
        "  --target-pout FRACTION      0.02 means 2%" \
        "  --max-rounds N              default: 8" \
        "  --points-per-cluster N      default: 200" \
        "  --membership-k N            default: 10, matching the old script" \
        "  --design NAME               legacy_random or space_filling" \
        "  --max-new-points-per-round N  safety cap; 0 keeps every survivor" \
        "" \
        "Methods: uniform_gap,shell,gaussian,legacy_halo"
}

while (($#)); do
    case "$1" in
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --skip-preflight)
            SKIP_PREFLIGHT=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    printf 'uv is not available in the active idev environment.\n' >&2
    exit 69
fi

mkdir -p "$OUTPUT" "$OUTPUT/.matplotlib"
LOG_FILE="$OUTPUT/run.log"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$OUTPUT/.matplotlib"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

printf 'Aggressive Rosenbrock P_out race\n' | tee -a "$LOG_FILE"
printf '  output: %s\n' "$OUTPUT" | tee -a "$LOG_FILE"
printf '  host: %s\n' "$(hostname)" | tee -a "$LOG_FILE"
printf '  work stealing: disabled (one MPI rank)\n' | tee -a "$LOG_FILE"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    printf '  Slurm job: %s\n' "$SLURM_JOB_ID" | tee -a "$LOG_FILE"
else
    printf '  warning: SLURM_JOB_ID is not set; confirm this is a compute node.\n' | tee -a "$LOG_FILE"
fi

if ((SKIP_PREFLIGHT == 0)); then
    uv run python -m src.rosenbrock_sampling_study.preflight --require-gpu \
        2>&1 | tee -a "$LOG_FILE"
fi

set +e
uv run python -m src.rosenbrock_sampling_study.pout_race \
    --output "$OUTPUT" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
RACE_STATUS=${PIPESTATUS[0]}
set -e

if find "$OUTPUT" -mindepth 2 -maxdepth 2 -name metrics.csv -print -quit | grep -q .; then
    printf 'Generating race rankings and plots...\n' | tee -a "$LOG_FILE"
    uv run python -m src.rosenbrock_sampling_study.pout_race_analyze \
        --input "$OUTPUT" 2>&1 | tee -a "$LOG_FILE"
    printf '\nResults are ready:\n' | tee -a "$LOG_FILE"
    printf '  summary: %s/analysis/results_summary.md\n' "$OUTPUT" | tee -a "$LOG_FILE"
    printf '  ranking: %s/analysis/race_summary.csv\n' "$OUTPUT" | tee -a "$LOG_FILE"
    printf '  figures: %s/analysis/figures/*.png\n' "$OUTPUT" | tee -a "$LOG_FILE"
fi

exit "$RACE_STATUS"
