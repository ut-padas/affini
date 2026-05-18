.PHONY: help dev dev-and-viz install-cpu install-gpu format lint test sc-experiments run run-parallel

# Helper to run commands inside the verified cluster environment wrapper
ENV_WRAPPER := . ./env_tacc.sh

help:
	@echo "Available commands:"
	@echo "  dev            - Install development packages relevant for GPU, linting, and testing"
	@echo "  dev-and-viz    - Install all development and visualization packages"
	@echo "  install-cpu    - Install packages relevant for runtime only"
	@echo "  install-gpu    - Install GPU-specific packages"
	@echo "  sc-experiments - Run all experiments for the SC26 paper"
	@echo "  format         - Format code with ruff (Ref: https://ruff.rs/)"
	@echo "  lint           - Run linting with ruff (Ref: https://ruff.rs/)"
	@echo "  test           - Run pytest tests"

dev:
	$(ENV_WRAPPER) && uv sync --group gpu --group lint --group test

dev-and-viz:
	$(ENV_WRAPPER) && uv sync --all-groups

install-cpu:
	uv sync

install-gpu:
	$(ENV_WRAPPER) && uv sync --group gpu

sc-experiments:
	$(ENV_WRAPPER) && uv run python sc26_experiments.py

format:
	uv run --group lint ruff format .

lint:
	uv run --group lint ruff check .

test:
	$(ENV_WRAPPER) && uv run --group test pytest