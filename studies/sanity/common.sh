#!/usr/bin/env bash
# Shared settings and helpers for the convergence-study drivers (sourced, not executed).
#
# Every variable can be overridden from the environment before calling a run script:
#   REPO_ROOT     this repository's checkout (default: two levels above this file)
#   ART_ROOT      reference ART checkout (default: ../Adaptive-Regression-Trees-ART- next to REPO_ROOT)
#   PYTHON        interpreter (default: $REPO_ROOT/.venv/bin/python)
#   MPIEXEC       MPI launcher (default: $REPO_ROOT/.venv/bin/mpiexec, else mpiexec on PATH);
#                 on a cluster set it to e.g. "ibrun" or "srun --mpi=pmix"
#   STUDY_OUT     where results/ and logs/ go (default: $REPO_ROOT/studies/sanity/output)
#   RUN_TIMEOUT   per-run wall-clock limit for `timeout` (default 6h; "0" disables it)

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$STUDY_DIR/../.." && pwd)}"
ART_ROOT="${ART_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Adaptive-Regression-Trees-ART-}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ -z "${MPIEXEC:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/mpiexec" ]]; then MPIEXEC="$REPO_ROOT/.venv/bin/mpiexec"; else MPIEXEC="mpiexec"; fi
fi
STUDY_OUT="${STUDY_OUT:-$STUDY_DIR/output}"
RUN_TIMEOUT="${RUN_TIMEOUT:-6h}"

# Thread pools: one rank per core under MPI, all cores otherwise (callers may export their own).
# The repository root for src, and this folder for the earlier draft's launcher (afinni_launcher.py)
# and iterative-refinement experiment, which the paper cases import.
export PYTHONPATH="$REPO_ROOT:$STUDY_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# run_logged <done-marker> <log-file> <command...>
# Skips when the marker exists (a finished run); otherwise runs the command with the timeout,
# appends stdout+stderr to the log, and writes the marker only on success. Failures are logged
# and counted but never stop the sweep, so one crashing case cannot hide the others. The command
# reads an empty stdin: the drivers loop over a case list on stdin, and a command that reads it
# (mpiexec forwards stdin to rank 0) would consume the remaining cases and end the sweep silently.
FAILED_RUNS=0
run_logged() {
    local marker="$1" logfile="$2"; shift 2
    if [[ -f "$marker" ]]; then log "skip (done): $(basename "$marker" .done)"; return 0; fi
    mkdir -p "$(dirname "$marker")" "$(dirname "$logfile")"
    log "run: $(basename "$marker" .done)"
    local t0=$SECONDS status
    {
        echo "### $(date -Is) host=$(hostname) SURROGATE_DEVICE=${SURROGATE_DEVICE:-} SURROGATE_MPI=${SURROGATE_MPI:-}"
        echo "### $*"
    } >>"$logfile"
    if [[ "$RUN_TIMEOUT" == "0" ]]; then
        "$@" </dev/null >>"$logfile" 2>&1; status=$?
    else
        timeout --signal=TERM --kill-after=60 "$RUN_TIMEOUT" "$@" </dev/null >>"$logfile" 2>&1; status=$?
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
        echo "repo: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null) ($(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null))"
        echo "repo_dirty: $(git -C "$REPO_ROOT" status --porcelain --untracked-files=no 2>/dev/null | wc -l) modified tracked files"
        echo "art: $(git -C "$ART_ROOT" rev-parse HEAD 2>/dev/null) ($(git -C "$ART_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null))"
        echo "python: $("$PYTHON" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null)"
        echo "SURROGATE_DEVICE=${SURROGATE_DEVICE:-} SURROGATE_MPI=${SURROGATE_MPI:-}"
        SURROGATE_MPI=off "$PYTHON" -m src 2>/dev/null | sed 's/^/  /'
        command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu: /'
    } >>"$out/provenance.txt"
}

# The device this library runs on; "gpu" needs CuPy and a CUDA device (uv sync --extra gpu).
check_device() {
    local dev="$1"
    [[ "$dev" == "cpu" ]] && return 0
    if ! SURROGATE_DEVICE=gpu SURROGATE_MPI=off "$PYTHON" -c "import src; assert src.IS_CUPY_BACKEND" 2>/dev/null; then
        log "error: SURROGATE_DEVICE=gpu requested but CuPy/CUDA is not usable (uv sync --extra gpu, check nvidia-smi)"
        exit 2
    fi
}
