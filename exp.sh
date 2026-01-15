#!/bin/bash
uv run python src/surrogate.py --tol 1e-1
uv run python src/surrogate.py --tol 1e-2