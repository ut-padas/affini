#!/bin/bash
set -Eeuo pipefail

OUTPUT_ROOT="results/sc26_rosenbrock_pout_race/metric_sweep"
SKIP_PREFLIGHT=0
EXTRA_ARGS=()

usage() {
    printf '%s\n' \
        "Usage: ./run_sc26_rosenbrock_pout_metric_sweep.sh [options]" \
        "" \
        "Runs the SC26-matched 5D Rosenbrock P_out race for both L2 and" \
        "Linf-local surrogate acceptance criteria. Each criterion evaluates" \
        "Uniform, Halo, Cloud, and Random chord unless --methods is supplied." \
        "" \
        "Options:" \
        "  --output-root PATH          parent result directory" \
        "  --skip-preflight            skip GPU/import checks" \
        "  --methods LIST              uniform,halo,cloud,random_chord" \
        "  --error-metrics LIST        L2,Linf-local (default)" \
        "  --target-pout FRACTION      default 0.02" \
        "  --max-rounds N              default 10" \
        "  --max-training-points N     default 5000000; 0 means unlimited" \
        "  --min-free-gpu-gib N        default 8; 0 allows a caught OOM stop" \
        "  --points-per-cluster N      default 500" \
        "  --cluster-reference N       proposal scaling reference" \
        "  --help                      show this message"
}

while (($#)); do
    case "$1" in
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

command -v uv >/dev/null 2>&1 || { printf 'uv is unavailable.\n' >&2; exit 69; }
mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/.matplotlib"
LOG_FILE="$OUTPUT_ROOT/run.log"
export MPLBACKEND=Agg MPLCONFIGDIR="$OUTPUT_ROOT/.matplotlib" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

printf 'SC26-matched Rosenbrock P_out metric sweep\n  output: %s\n  acceptance metrics: L2, Linf-local\n  work stealing: disabled for one idev rank\n' "$OUTPUT_ROOT" | tee -a "$LOG_FILE"

if ((SKIP_PREFLIGHT == 0)); then
    uv run python -m src.rosenbrock_sampling_study.preflight --require-gpu 2>&1 | tee -a "$LOG_FILE"
fi

set +e
uv run python -m src.rosenbrock_sampling_study.sc26_pout_metric_sweep --output-root "$OUTPUT_ROOT" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
SWEEP_STATUS=${PIPESTATUS[0]}
set -e

if [[ -f "$OUTPUT_ROOT/analysis/metric_sweep_all_metrics.csv" ]]; then
    printf 'Combined results:\n  %s/analysis/metric_sweep_all_metrics.csv\n  %s/analysis/metric_sweep_final_table.csv\n  %s/analysis/figures/pout_by_error_metric.png\n' "$OUTPUT_ROOT" "$OUTPUT_ROOT" "$OUTPUT_ROOT" | tee -a "$LOG_FILE"
fi

exit "$SWEEP_STATUS"
