.PHONY: help dev format install-cpu install-gpu lint load-modules test sc-experiments dev-and-viz

# --- TACC Environment Auto-Detection & Setup ---
# Captures both short hostnames (c562-007) and FQDNs
HOST_INFO := $(shell hostname && hostname -f 2>/dev/null)

ifneq ($(filter %stampede3% c% g%,$(HOST_INFO)),)
    # Stampede3 Setup
    SYS_NAME   := Stampede3
    MODULE_CMD := module purge && module load gcc/15.1.0 cuda/12.8 impi
    export FI_PROVIDER=shm
else ifneq ($(filter %vista%,$(HOST_INFO)),)
    # Vista Setup
    SYS_NAME   := TACC Vista
    MODULE_CMD := module purge && module load nvidia/25.9 gcc/15.1.0 cuda/12.9
else
    # Local Workstation Fallback
    SYS_NAME   := Local/External System
    MODULE_CMD := true
endif

help:
	@echo "Available commands (Detected: $(SYS_NAME)):"
	@echo "  sc-experiments - Run all experiments for the SC26 paper"
	@echo "  dev            - Install development packages relevant for GPU, linting, and testing"
	@echo "  dev-and-viz    - Install all development and visualization packages"
	@echo "  install-cpu    - Install packages relevant for runtime only"
	@echo "  install-gpu    - Install GPU-specific packages"
	@echo "  format         - Format code with ruff (Ref: https://ruff.rs/)"
	@echo "  lint           - Run linting with ruff (Ref: https://ruff.rs/)"
	@echo "  load-modules   - Diagnostic target to inspect active module configuration"
	@echo "  test           - Run pytest tests"

sc-experiments:
	$(MODULE_CMD) && uv run python sc26_experiments.py

dev:
	$(MODULE_CMD) && uv sync --group gpu --group lint --group test

dev-and-viz:
	$(MODULE_CMD) && uv sync --all-groups

install-cpu:
	uv sync

install-gpu:
	$(MODULE_CMD) && uv sync --group gpu

format:
	uv run --group lint ruff format .

lint:
	uv run --group lint ruff check .

test:
	$(MODULE_CMD) && uv run --group test pytest

load-modules:
	@echo "===================================================="
	@echo "  System Detected : $(SYS_NAME)"
	@echo "  Module Engine   : $(MODULE_CMD)"
	@echo "  Network Fabric  : FI_PROVIDER=$(FI_PROVIDER)"
	@echo "===================================================="