# Thin convenience wrappers around `uv`. Every target is a single uv command, so
# on systems without `make` (e.g. Windows) just run the command shown in `make help`.
#
# Optional backends are uv "extras" and can be combined via EXTRAS, e.g.
#     make dev EXTRAS="gpu mpi"
# NOTE: `uv sync` makes the venv match exactly what you ask for, so switching
# targets also removes packages that the new target does not include.

.PHONY: help dev dev-all install-cpu install-mpi install-gpu install-gpu-mpi install-hpc \
        info format lint test test-mpi paper-experiments

EXTRAS ?=
_EXTRA_FLAGS := $(foreach e,$(EXTRAS),--extra $(e))

help:
	@echo "Setup (research & development):"
	@echo "  dev              - CPU + dev tools (lint, test, notebooks)    uv sync"
	@echo "  dev-all          - dev + research & viz groups               uv sync --all-groups"
	@echo "                     add backends with EXTRAS, e.g. make dev EXTRAS=\"gpu mpi\""
	@echo ""
	@echo "Runtime-only installs (no dev tools):"
	@echo "  install-cpu      - CPU only (Linux / macOS / Windows)       uv sync --no-dev"
	@echo "  install-mpi      - CPU + MPI, runtime bundled via pip        uv sync --no-dev --extra mpi"
	@echo "  install-gpu      - NVIDIA GPU (CUDA 12)                      uv sync --no-dev --extra gpu"
	@echo "  install-gpu-mpi  - GPU + bundled MPI                         uv sync --no-dev --extra gpu --extra mpi"
	@echo "  install-hpc      - GPU + the cluster's own MPI               uv sync --no-dev --extra gpu --extra mpi-system"
	@echo "                     (on TACC, first: source scripts/hpc/tacc.sh)"
	@echo ""
	@echo "Other:"
	@echo "  info             - Show detected backends (GPU / cuML / MPI)  uv run python -m src"
	@echo "  test             - Run pytest"
	@echo "  test-mpi         - Backend + multi-rank tests, MPI required (needs an MPI extra)"
	@echo "  format / lint    - ruff format / ruff check"
	@echo "  paper-experiments- Run run_paper_experiments.sh"

# --- Research & development --------------------------------------------------
dev:
	uv sync $(_EXTRA_FLAGS)

dev-all:
	uv sync --all-groups $(_EXTRA_FLAGS)

# --- Runtime-only installs ---------------------------------------------------
install-cpu:
	uv sync --no-dev $(_EXTRA_FLAGS)

install-mpi:
	uv sync --no-dev --extra mpi $(_EXTRA_FLAGS)

install-gpu:
	uv sync --no-dev --extra gpu $(_EXTRA_FLAGS)

install-gpu-mpi:
	uv sync --no-dev --extra gpu --extra mpi $(_EXTRA_FLAGS)

install-hpc:
	uv sync --no-dev --extra gpu --extra mpi-system $(_EXTRA_FLAGS)

# --- Utilities ---------------------------------------------------------------
info:
	uv run --no-sync python -m src

format:
	uv run --group lint ruff format .

lint:
	uv run --group lint ruff check .

test:
	uv run --group test pytest

test-mpi:
	AFINNI_MPI=on uv run --group test pytest tests/mpi tests/integration/test_backend.py

paper-experiments:
	bash run_paper_experiments.sh
