#!/usr/bin/env bash
# Run one launcher case at a commit extracted with `git archive` (never checked out here).
#   run_at.sh COMMIT TAG [--launcher FILE] -- <launcher arguments>
# Extracts COMMIT under $BISECT_DIR/COMMIT once, clears that tree's results/ (launchers skip
# existing result files), and runs capture_launcher.py there. Output: $BISECT_DIR/runs/TAG.npz/.log.
# Environment:
#   BISECT_DIR  (required) where extractions and runs/ live
#   FAISS_SHIM  directories prepended to PYTHONPATH (the CPU stand-ins for faiss / cupy); empty on a
#               GPU with the commit's own environment
#   PYTHON      interpreter (default: this repo's .venv); for a paper-era GPU run, the extraction's
#               own `uv sync` venv, which has cupy, cuml-cu12 and faiss-cpu
#   DEVICE      cpu (default) or gpu: sets SURROGATE_DEVICE. Commits before d7ec052 ignore it and use the
#               GPU whenever cupy imports, so on a CPU-only run keep cupy out of the environment.
#   CAP         1 (default): run under a systemd scope (6 GB, 4 cores, nice 10, 900 s timeout); 0: plain
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
: "${BISECT_DIR:?set BISECT_DIR}"
: "${FAISS_SHIM:=}" "${PYTHON:=$repo/.venv/bin/python}" "${DEVICE:=cpu}" "${CAP:=1}"
commit=$1 tag=$2
shift 2
launcher=afinni_launcher.py
if [[ ${1:-} == --launcher ]]; then launcher=$2; shift 2; fi
[[ ${1:-} == -- ]] && shift
root=$BISECT_DIR/$commit
if [[ ! -d $root/src ]]; then
  mkdir -p "$root"
  git -C "$repo" archive "$commit" | tar -x -C "$root"
fi
rm -rf "$root/results"
mkdir -p "$BISECT_DIR/runs"
cd "$root"
cap=()
[[ $CAP == 1 ]] && cap=(systemd-run --user --scope -q -p MemoryMax=6G -p CPUQuota=400% nice -n 10 timeout 900)
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
SURROGATE_DEVICE=$DEVICE SURROGATE_MPI=off PYTHONPATH="$root${FAISS_SHIM:+:$FAISS_SHIM}" MPLBACKEND=Agg \
  "${cap[@]}" "$PYTHON" "$here/capture_launcher.py" --out "$BISECT_DIR/runs/$tag.npz" \
  --launcher "$launcher" -- "$@" >"$BISECT_DIR/runs/$tag.log" 2>&1 || { echo "FAILED $tag (see $BISECT_DIR/runs/$tag.log)"; tail -5 "$BISECT_DIR/runs/$tag.log"; exit 1; }
grep '^RESULT' "$BISECT_DIR/runs/$tag.log"
