#!/usr/bin/env bash
# Shared settings and helpers for the convergence-study drivers (sourced, not executed).
#
# Every variable can be overridden from the environment before calling a run script:
#   AFINNI_ROOT   AFINNI checkout (default: two levels above this file)
#   ART_ROOT      reference ART checkout (default: ../Adaptive-Regression-Trees-ART- next to AFINNI_ROOT)
#   PYTHON        interpreter (default: $AFINNI_ROOT/.venv/bin/python)
#   MPIEXEC       MPI launcher (default: $AFINNI_ROOT/.venv/bin/mpiexec, else mpiexec on PATH);
#                 on a cluster set it to e.g. "ibrun" or "srun --mpi=pmix"
#   STUDY_OUT     where results/ and logs/ go (default: $AFINNI_ROOT/studies/convergence/output)
#   RUN_TIMEOUT   per-run wall-clock limit for `timeout` (default 6h; "0" disables it)

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFINNI_ROOT="${AFINNI_ROOT:-$(cd "$STUDY_DIR/../.." && pwd)}"
ART_ROOT="${ART_ROOT:-$(cd "$AFINNI_ROOT/.." && pwd)/Adaptive-Regression-Trees-ART-}"
PYTHON="${PYTHON:-$AFINNI_ROOT/.venv/bin/python}"
if [[ -z "${MPIEXEC:-}" ]]; then
    if [[ -x "$AFINNI_ROOT/.venv/bin/mpiexec" ]]; then MPIEXEC="$AFINNI_ROOT/.venv/bin/mpiexec"; else MPIEXEC="mpiexec"; fi
fi
STUDY_OUT="${STUDY_OUT:-$STUDY_DIR/output}"
RUN_TIMEOUT="${RUN_TIMEOUT:-6h}"

# Thread pools: one rank per core under MPI, all cores otherwise (callers may export their own).
export PYTHONPATH="$AFINNI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# run_logged <done-marker> <log-file> <command...>
# Skips when the marker exists (a finished run); otherwise runs the command with the timeout,
# appends stdout+stderr to the log, and writes the marker only on success. Failures are logged
# and counted but never stop the sweep, so one crashing case cannot hide the others.
FAILED_RUNS=0
run_logged() {
    local marker="$1" logfile="$2"; shift 2
    if [[ -f "$marker" ]]; then log "skip (done): $(basename "$marker" .done)"; return 0; fi
    mkdir -p "$(dirname "$marker")" "$(dirname "$logfile")"
    log "run: $(basename "$marker" .done)"
    local t0=$SECONDS status
    {
        echo "### $(date -Is) host=$(hostname) AFINNI_DEVICE=${AFINNI_DEVICE:-} AFINNI_MPI=${AFINNI_MPI:-}"
        echo "### $*"
    } >>"$logfile"
    if [[ "$RUN_TIMEOUT" == "0" ]]; then
        "$@" >>"$logfile" 2>&1; status=$?
    else
        timeout --signal=TERM --kill-after=60 "$RUN_TIMEOUT" "$@" >>"$logfile" 2>&1; status=$?
    fi
    echo "### exit=$status after $((SECONDS - t0)) s" >>"$logfile"
    if [[ $status -eq 0 ]]; then
        date -Is >"$marker"
    else
        FAILED_RUNS=$((FAILED_RUNS + 1))
        log "FAILED (exit $status, see $logfile)"
    fi
    return 0
}

# Record both checkouts' commits next to the results, so a result is never compared
# against a baseline from another revision by accident.
record_provenance() {
    local out="$1"; mkdir -p "$out"
    {
        echo "date: $(date -Is)"
        echo "host: $(hostname)"
        echo "afinni: $(git -C "$AFINNI_ROOT" rev-parse HEAD 2>/dev/null) ($(git -C "$AFINNI_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null))"
        echo "afinni_dirty: $(git -C "$AFINNI_ROOT" status --porcelain --untracked-files=no 2>/dev/null | wc -l) modified tracked files"
        echo "art: $(git -C "$ART_ROOT" rev-parse HEAD 2>/dev/null) ($(git -C "$ART_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null))"
        echo "python: $("$PYTHON" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null)"
        echo "AFINNI_DEVICE=${AFINNI_DEVICE:-} AFINNI_MPI=${AFINNI_MPI:-}"
        AFINNI_MPI=off "$PYTHON" -m src 2>/dev/null | sed 's/^/  /'
        command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu: /'
    } >>"$out/provenance.txt"
}

# The device the AFINNI side runs on; "gpu" needs CuPy and a CUDA device (uv sync --extra gpu).
check_device() {
    local dev="$1"
    [[ "$dev" == "cpu" ]] && return 0
    if ! AFINNI_DEVICE=gpu AFINNI_MPI=off "$PYTHON" -c "import src; assert src.IS_CUPY_BACKEND" 2>/dev/null; then
        log "error: AFINNI_DEVICE=gpu requested but CuPy/CUDA is not usable (uv sync --extra gpu, check nvidia-smi)"
        exit 2
    fi
}
