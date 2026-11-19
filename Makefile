# Thin convenience wrappers around `uv`. Every target is a single uv command, so
# on systems without `make` (e.g. Windows) just run the command shown in `make help`.
#
# Optional backends are uv "extras" and can be combined via EXTRAS, e.g.
#     make dev EXTRAS="gpu mpi"
# NOTE: `uv sync` makes the venv match exactly what you ask for, so switching
# targets also removes packages that the new target does not include.

.PHONY: report-global-figures help dev dev-all install-cpu install-mpi install-gpu install-gpu-mpi install-hpc \
        info format lint test test-mpi coverage-cpu coverage-gpu coverage-mpi coverage

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
	@echo "  report           - Compile both technical companions (docs/reports/*/*.pdf; needs typst)"
	@echo "  report-figures   - Regenerate the adaptive-surrogates companion's panels and numbers (docs/reports/adaptive-surrogates/figures/)"
	@echo "  report-global-figures - Regenerate the global-approximants companion's panels and numbers (docs/reports/global-approximants/figures/)"
	@echo "  report-steals    - Record the companion's 6-rank work-stealing runs again (needs an MPI extra)"
	@echo "  study-smoke / study-plan / study-run / study-status / study-collect - The manuscript study (MACHINE=<profile>)"
	@echo "  report-tables    - Rewrite the companion appendix's manuscript tables (RECORD=\"<main record> [<supplementary> ...]\")"

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
	SURROGATE_MPI=on uv run --group test pytest tests/mpi tests/integration/test_backend.py

# The CuPy branches (custom kernels and their launchers) only run in the GPU suite, and work
# stealing, scatter and gather only on several ranks, so measure each suite and combine:
# coverage-cpu, coverage-mpi, coverage-gpu on a CUDA machine, then coverage.
coverage-cpu:
	COVERAGE_FILE=.coverage.cpu SURROGATE_DEVICE=cpu SURROGATE_MPI=off uv run --group test pytest -q -m "not mpi" -o addopts="" --cov=src --cov-report=

coverage-gpu:
	COVERAGE_FILE=.coverage.gpu SURROGATE_DEVICE=gpu SURROGATE_MPI=off uv run --group test pytest -q -m "not mpi" -o addopts="" --cov=src --cov-report=

# The MPI tests start their ranks through mpiexec; coverage's subprocess patch (pyproject.toml)
# measures every rank, and parallel mode gives each process its own data file.
coverage-mpi:
	COVERAGE_FILE=.coverage.mpi SURROGATE_DEVICE=cpu SURROGATE_MPI=on SURROGATE_MPI_NP=4 PATH="$(CURDIR)/.venv/bin:$$PATH" \
		uv run --group test coverage run --parallel-mode -m pytest -q tests/mpi -o addopts=""

coverage:
	uv run --group test coverage combine --keep
	uv run --group test coverage report --precision=1 --skip-covered
	uv run --group test coverage html

# The technical companions (docs/reports/): AFINNI, ART and the engine; the global approximants. Both import
# the shared companion package (docs/reports/companion), so they compile with docs/reports as Typst's root.
# Their panels are the CPU results of real runs.
REPORTS := adaptive-surrogates global-approximants
report:
	for r in $(REPORTS); do [ -f docs/reports/$$r/$$r.typ ] || continue; \
		typst compile --root docs/reports docs/reports/$$r/$$r.typ docs/reports/$$r/$$r.pdf || exit 1; done

report-figures:
	SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. uv run python docs/reports/adaptive-surrogates/figures/make_figures.py
	for f in docs/reports/adaptive-surrogates/figures/appendix/*.py; do \
		SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. uv run python $$f || exit 1; done

report-global-figures:
	SURROGATE_DEVICE=cpu SURROGATE_MPI=off OPENBLAS_NUM_THREADS=1 PYTHONPATH=. uv run python docs/reports/global-approximants/figures/make_figures.py

# STEAL_CPUS pins the ranks to cores of one speed (on a hybrid CPU, its performance cores),
# e.g. make report-steals STEAL_CPUS=0-5; the companion's run used six such cores.
STEAL_CPUS ?=
report-steals:
	SURROGATE_DEVICE=cpu SURROGATE_MPI=on PYTHONPATH=. OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
		$(if $(STEAL_CPUS),taskset -c $(STEAL_CPUS)) uv run mpiexec -n 6 python docs/reports/adaptive-surrogates/figures/make_figures.py --record-steals
	SURROGATE_DEVICE=cpu SURROGATE_MPI=on PYTHONPATH=. OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
		$(if $(STEAL_CPUS),taskset -c $(STEAL_CPUS)) uv run mpiexec -n 6 python docs/reports/adaptive-surrogates/figures/appendix/steal.py --record </dev/null

# The manuscript study (studies/manuscript/README.md): MACHINE is a profile in studies/manuscript/machines.
MACHINE ?= studies/manuscript/machines/workstation.toml
study-smoke:
	uv run python -m studies.manuscript.run run --smoke --seeds 0 --machine $(MACHINE)
study-plan study-run study-status:
	uv run python -m studies.manuscript.run $(subst study-,,$@) --machine $(MACHINE)
study-collect:
	uv run python -m studies.manuscript.collect

# The appendix's manuscript tables: the paper's numbers, and committed records' (none: every re-run
# cell "–"). RECORD lists record directories, the main record first; a later one replaces its values.
RECORD ?=
report-tables:
	PYTHONPATH=. uv run python docs/reports/adaptive-surrogates/figures/manuscript_tables.py $(foreach r,$(RECORD),--record $(r))

