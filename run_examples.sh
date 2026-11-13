#!/usr/bin/env bash

# Run from the repository root. Examples are run as modules so that `src` is importable.
cd "$(dirname "$0")"

# Example 1: Rosenbrock in 2D
uv run python -m examples.example01_rosenbrock_afinni

# Example 2: Nelson flow map
uv run python -m examples.example02_nelson_afinni
