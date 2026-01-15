#!/bin/bash
uv run python -m src.surrogate --tol 1e-1
uv run python -m src.surrogate --tol 1e-2
uv run python -m src.surrogate --tol 1e-3
uv run python -m src.surrogate --tol 1e-4
uv run python -m src.surrogate --tol 1e-5
uv run python -m src.surrogate --tol 1e-6