.PHONY: help dev format install-cpu install-gpu lint load-modules

help:
	@echo "Available commands:"
	@echo "  benchmarks   - Run all the benchmarking scripts"
	@echo "  dev          - Install all development packages"
	@echo "  install-cpu  - Install packages relevant for runtime only"
	@echo "  install-gpu  - Install GPU-specific packages"
	@echo "  format       - Format code with ruff (Ref: https://ruff.rs/)"
	@echo "  lint         - Run linting with ruff (Ref: https://ruff.rs/)"
	@echo "  load-modules - Load required modules if running on TACC Vista"
	@echo "  test         - Run pytest tests"

benchmarks: load-modules
	uv run python -m research.benchmark_kmeans_gpu_vs_cpu
	uv run python -m research.benchmark_minibatch_vs_kmeans
	uv run python -m research.benchmark_two_means_qr_vs_kmeans
	uv run python -m research.benchmark_two_means_vs_kmeans

dev:
	uv sync --all-groups

install-cpu:
	uv sync

install-gpu:
	uv sync --group gpu

format:
	uv run --group lint ruff format .

lint:
	uv run --group lint ruff check .

test: load-modules
	uv run --group test pytest

load-modules:
	@echo "Checking if running on TACC Vista..."
	@if hostname | grep -q "vista.tacc.utexas.edu"; then \
		echo "Detected TACC Vista system. Loading required modules..."; \
		if command -v module >/dev/null 2>&1; then \
			module load gcc cuda; \
			echo "Modules loaded: gcc, cuda"; \
		else \
			echo "Warning: Module system not available. Skipping module loading."; \
		fi; \
	else \
		echo "Not running on TACC Vista. Skipping module loading."; \
	fi
