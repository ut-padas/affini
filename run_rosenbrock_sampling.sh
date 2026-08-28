#!/bin/bash
set -Eeuo pipefail

PRESET="idev"
OUTPUT=""
SKIP_PREFLIGHT=0
EXTRA_ARGS=()

usage() {
    printf '%s\n' \
        "Usage: ./run_rosenbrock_sampling.sh [options]" \
        "" \
        "Options:" \
        "  --preset NAME       smoke, idev, pilot, or full (default: idev)" \
        "  --output PATH       result directory (default: results/.../PRESET)" \
        "  --skip-preflight    skip GPU and import checks" \
        "  --help              show this message" \
        "" \
        "Any other option is passed to the Python experiment driver."
}

while (($#)); do
    case "$1" in
        --preset)
            PRESET="$2"
            shift 2
            ;;
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

case "$PRESET" in
    smoke|idev|pilot|full) ;;
    *)
        printf 'Unknown preset: %s\n' "$PRESET" >&2
        exit 64
        ;;
esac

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="results/rosenbrock_sampling/$PRESET"
fi

if ! command -v uv >/dev/null 2>&1; then
    printf 'uv is not available in the active idev environment.\n' >&2
    exit 69
fi

mkdir -p "$OUTPUT"
LOG_FILE="$OUTPUT/run.log"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

printf 'Rosenbrock sampling run\n' | tee -a "$LOG_FILE"
printf '  preset: %s\n' "$PRESET" | tee -a "$LOG_FILE"
printf '  output: %s\n' "$OUTPUT" | tee -a "$LOG_FILE"
printf '  host: %s\n' "$(hostname)" | tee -a "$LOG_FILE"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    printf '  Slurm job: %s\n' "$SLURM_JOB_ID" | tee -a "$LOG_FILE"
else
    printf '  warning: SLURM_JOB_ID is not set; confirm this is a compute node.\n' | tee -a "$LOG_FILE"
fi

if ((SKIP_PREFLIGHT == 0)); then
    uv run python -m src.rosenbrock_sampling_study.preflight --require-gpu 2>&1 | tee -a "$LOG_FILE"
fi

set +e
uv run python -m src.rosenbrock_sampling_study.experiment \
    --preset "$PRESET" \
    --output "$OUTPUT" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
EXPERIMENT_STATUS=${PIPESTATUS[0]}
set -e

printf 'Generating aggregate tables and plots...\n' | tee -a "$LOG_FILE"
uv run python -m src.rosenbrock_sampling_study.analyze \
    --input "$OUTPUT" 2>&1 | tee -a "$LOG_FILE"

printf '\nResults are ready:\n' | tee -a "$LOG_FILE"
printf '  summary: %s/analysis/results_summary.md\n' "$OUTPUT" | tee -a "$LOG_FILE"
printf '  tables:  %s/analysis/*.csv\n' "$OUTPUT" | tee -a "$LOG_FILE"
printf '  figures: %s/analysis/figures/*.png\n' "$OUTPUT" | tee -a "$LOG_FILE"

if ((EXPERIMENT_STATUS != 0)); then
    printf 'One or more acquisition arms stopped early. See STOPPED.json files.\n' >&2
fi
exit "$EXPERIMENT_STATUS"
