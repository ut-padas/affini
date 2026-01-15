#!/bin/bash
uv run python -m src.surrogate --tol 1e-1 --dim 2
uv run python -m src.surrogate --tol 1e-2 --dim 2
uv run python -m src.surrogate --tol 1e-3 --dim 2
uv run python -m src.surrogate --tol 1e-4 --dim 2
uv run python -m src.surrogate --tol 1e-5 --dim 2
uv run python -m src.surrogate --tol 1e-6 --dim 2


uv run python -m src.surrogate --tol 1e-1 --dim 5
uv run python -m src.surrogate --tol 1e-2 --dim 5
uv run python -m src.surrogate --tol 1e-3 --dim 5
uv run python -m src.surrogate --tol 1e-4 --dim 5
uv run python -m src.surrogate --tol 1e-5 --dim 5