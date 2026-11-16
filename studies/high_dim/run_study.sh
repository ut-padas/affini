#!/usr/bin/env bash
# The high-dimensional scaling study on one GPU (see HIGH_DIM_STUDY.md, Part 3).
#
#   studies/high_dim/run_study.sh            # the full sweep on the GPU (resumable: rerun to continue)
#   studies/high_dim/run_study.sh --smoke    # d = 4, 8 on the CPU in a few minutes: checks the pipeline
#   studies/high_dim/run_study.sh --dry-run  # list the runs without running them
#
# Settings (environment variables, defaults in brackets):
#   AFINNI_DEVICE [gpu; cpu with --smoke]   AFINNI_MPI [off]
#   RESULTS       [studies/high_dim/results; .../results_smoke with --smoke]
#   DIMS          [20,40,60,80,100,120,160,200,250,300]   PROBLEMS [affine,quadratic,aniso_quadratic,rosenbrock]
#   VARIANTS      [afinni,art_affine,art_quad]            TIME_LIMIT [1800] seconds per run
#   EXTRA         extra arguments for sweep.py, e.g. "--library-min-pts" or "--tols 1,0.3,0.1"
# Any other argument is passed to sweep.py.
set -euo pipefail
cd "$(dirname "$0")/../.."

SMOKE=0
for arg in "$@"; do [[ "$arg" == "--smoke" ]] && SMOKE=1; done
if [[ $SMOKE == 1 ]]; then
  export AFINNI_DEVICE="${AFINNI_DEVICE:-cpu}"
  RESULTS="${RESULTS:-studies/high_dim/results_smoke}"
else
  export AFINNI_DEVICE="${AFINNI_DEVICE:-gpu}"
  RESULTS="${RESULTS:-studies/high_dim/results}"
fi
export AFINNI_MPI="${AFINNI_MPI:-off}"
PY=.venv/bin/python
mkdir -p "$RESULTS"

echo "== backends"
$PY -m src 2>&1 | tail -n 5 || true
if [[ "$AFINNI_DEVICE" == gpu ]]; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv | tee "$RESULTS/gpu.csv"
fi

echo "== predicted limits (CPU, seconds)"
AFINNI_DEVICE=cpu $PY -m studies.high_dim.predict_limits --out "$RESULTS/predicted.csv" \
  --dims "${DIMS:-4,8,10,20,40,60,80,100,120,160,200,250,300}" > "$RESULTS/predicted.md"
tail -n +1 "$RESULTS/predicted.md" | head -n 3

echo "== sweep"
ARGS=(--out "$RESULTS")
[[ -n "${DIMS:-}" ]] && ARGS+=(--dims "$DIMS")
[[ -n "${PROBLEMS:-}" ]] && ARGS+=(--problems "$PROBLEMS")
[[ -n "${VARIANTS:-}" ]] && ARGS+=(--variants "$VARIANTS")
[[ -n "${TIME_LIMIT:-}" ]] && ARGS+=(--time-limit "$TIME_LIMIT")
# shellcheck disable=SC2206
[[ -n "${EXTRA:-}" ]] && ARGS+=($EXTRA)
$PY -m studies.high_dim.sweep "${ARGS[@]}" "$@" 2>&1 | tee -a "$RESULTS/sweep.log"

for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && exit 0; done
echo "== collect"
$PY -m studies.high_dim.collect --results "$RESULTS"
