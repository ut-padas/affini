.PHONY: help dev format install-cpu install-gpu lint load-modules

help:
	@echo "Available commands:"
	@echo "  sc-experiments - Run all experiments for the SC26 paper"
	@echo "  dev            - Install development packages relevant for GPU, linting, and testing"
	@echo "  dev-and-viz    - Install all development and visualization packages"
	@echo "  install-cpu    - Install packages relevant for runtime only"
	@echo "  install-gpu    - Install GPU-specific packages"
	@echo "  format         - Format code with ruff (Ref: https://ruff.rs/)"
	@echo "  lint           - Run linting with ruff (Ref: https://ruff.rs/)"
	@echo "  load-modules   - Load required modules if running on TACC Vista"
	@echo "  test           - Run pytest tests"

sc-experiments: load-modules
	uv run python sc26_experiments.py

dev:
	uv sync --group gpu --group lint --group test

dev-and-viz:
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
			module load nvidia/25.9 gcc/15.1.0; \
			module load cuda/12.9; \
			echo "Modules loaded: nvidia/25.9, cuda/12.9, gcc/15.1.0"; \
		else \
			echo "Warning: Module system not available. Skipping module loading."; \
		fi; \
	else \
		echo "Not running on TACC Vista. Skipping module loading."; \
	fi
