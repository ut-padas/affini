#!/bin/bash
set -Eeuo pipefail

OUTPUT="results/sc26_rosenbrock_pout_race/dim5"
SKIP_PREFLIGHT=0
EXTRA_ARGS=()

usage() {
    printf '%s\n' \
        "Usage: ./run_sc26_rosenbrock_pout_race.sh [options]" \
        "" \
        "SC26-matched 5D Rosenbrock P_out race." \
        "Default: Linf-local, rtol=0.05, N0=185646, Ntest=92823," \
        "four methods (Uniform, Halo, Cloud, Random chord), target P_out < 2%," \
        "10 acquisition rounds, at most 5,000,000 persistent training points, and" \
        "a post-checkpoint 8 GiB free-GPU-memory guard." \
        "" \
        "Options:" \
        "  --output PATH               result directory" \
        "  --skip-preflight            skip GPU/import checks" \
        "  --methods LIST              uniform,halo,cloud,random_chord" \
        "  --target-pout FRACTION      default 0.02" \
        "  --max-rounds N              default 10" \
        "  --max-training-points N     default 5000000; set 0 to remove the cap" \
        "  --min-free-gpu-gib N        default 8; set 0 to permit a caught OOM stop" \
        "  --points-per-cluster N      default 500" \
        "  --cluster-reference N       proposal budget begins scaling after this many clusters" \
        "  --help                      show this message"
}

while (($#)); do
    case "$1" in
        --output) OUTPUT="$2"; shift 2 ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

command -v uv >/dev/null 2>&1 || { printf 'uv is unavailable.\n' >&2; exit 69; }
mkdir -p "$OUTPUT" "$OUTPUT/.matplotlib"
LOG_FILE="$OUTPUT/run.log"
export MPLBACKEND=Agg MPLCONFIGDIR="$OUTPUT/.matplotlib" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

printf 'SC26-matched Rosenbrock P_out race\n  output: %s\n  work stealing: disabled for one idev rank\n' "$OUTPUT" | tee -a "$LOG_FILE"

if ((SKIP_PREFLIGHT == 0)); then
    uv run python -m src.rosenbrock_sampling_study.preflight --require-gpu 2>&1 | tee -a "$LOG_FILE"
fi

set +e
uv run python -m src.rosenbrock_sampling_study.sc26_pout_race --output "$OUTPUT" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
RACE_STATUS=${PIPESTATUS[0]}
set -e

if find "$OUTPUT" -mindepth 2 -maxdepth 2 -name metrics.csv -print -quit | grep -q .; then
    uv run python -m src.rosenbrock_sampling_study.sc26_pout_race_analyze --input "$OUTPUT" 2>&1 | tee -a "$LOG_FILE"
    printf 'Results:\n  %s/analysis/paper_style_iteration_table.md\n  %s/analysis/final_paper_style_table.csv\n  %s/analysis/figures/*.png\n' "$OUTPUT" "$OUTPUT" "$OUTPUT" | tee -a "$LOG_FILE"
fi

exit "$RACE_STATUS"
