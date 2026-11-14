# Thin convenience wrappers around `uv`. Every target is a single uv command, so
# on systems without `make` (e.g. Windows) just run the command shown in `make help`.
#
# Optional backends are uv "extras" and can be combined via EXTRAS, e.g.
#     make dev EXTRAS="gpu mpi"
# NOTE: `uv sync` makes the venv match exactly what you ask for, so switching
# targets also removes packages that the new target does not include.

.PHONY: help dev dev-all install-cpu install-mpi install-gpu install-gpu-mpi install-hpc \
        info format lint test test-mpi coverage-cpu coverage-gpu coverage-mpi coverage paper-experiments

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
	@echo "  info             - Show detected backends (GPU / MPI)         uv run python -m src"
	@echo "  test             - Run pytest"
	@echo "  test-mpi         - Backend + multi-rank tests, MPI required (needs an MPI extra)"
	@echo "  coverage-cpu     - CPU suite with coverage data in .coverage.cpu"
	@echo "  coverage-gpu     - GPU suite with coverage data in .coverage.gpu (CUDA machine)"
	@echo "  coverage-mpi     - multi-rank suite, measured inside every rank (.coverage.mpi.*)"
	@echo "  coverage         - Combine the .coverage.* files present; report and htmlcov/"
	@echo "  format / lint    - ruff format / ruff check"
	@echo "  report           - Compile the technical companion to docs/report/adaptive-surrogates.pdf (needs typst)"
	@echo "  report-figures   - Regenerate the companion's panels and numbers (docs/report/figures/)"
	@echo "  report-steals    - Record the companion's 4-rank work-stealing run again (needs an MPI extra)"
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

# The CuPy branches (custom kernels and their launchers) only run in the GPU suite, and work
# stealing, scatter and gather only on several ranks, so measure each suite and combine:
# coverage-cpu, coverage-mpi, coverage-gpu on a CUDA machine, then coverage.
coverage-cpu:
	COVERAGE_FILE=.coverage.cpu AFINNI_DEVICE=cpu AFINNI_MPI=off uv run --group test pytest -q -m "not mpi" -o addopts="" --cov=src --cov-report=

coverage-gpu:
	COVERAGE_FILE=.coverage.gpu AFINNI_DEVICE=gpu AFINNI_MPI=off uv run --group test pytest -q -m "not mpi" -o addopts="" --cov=src --cov-report=

# The MPI tests start their ranks through mpiexec; coverage's subprocess patch (pyproject.toml)
# measures every rank, and parallel mode gives each process its own data file.
coverage-mpi:
	COVERAGE_FILE=.coverage.mpi AFINNI_DEVICE=cpu AFINNI_MPI=on AFINNI_MPI_NP=4 PATH="$(CURDIR)/.venv/bin:$$PATH" \
		uv run --group test coverage run --parallel-mode -m pytest -q tests/mpi -o addopts=""

coverage:
	uv run --group test coverage combine --keep
	uv run --group test coverage report --precision=1 --skip-covered
	uv run --group test coverage html

# The technical companion (docs/report/). Its panels are the CPU results of real runs.
report:
	typst compile docs/report/adaptive-surrogates.typ docs/report/adaptive-surrogates.pdf

report-figures:
	AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. uv run python docs/report/figures/make_figures.py

report-steals:
	AFINNI_DEVICE=cpu AFINNI_MPI=on PYTHONPATH=. uv run mpiexec -n 4 python docs/report/figures/make_figures.py --record-steals

paper-experiments:
	bash run_paper_experiments.sh
