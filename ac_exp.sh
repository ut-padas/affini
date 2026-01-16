#!/bin/bash

uv run python -m src.adaptive_clustering --tol 1e-0 --dim 2 --n-train 100000 --n-init-clusters 20 --n-min-per-cluster 500 --rosenbrock-a 100
uv run python -m src.adaptive_clustering --tol 1e-1 --dim 2 --n-train 100000 --n-init-clusters 20 --n-min-per-cluster 500 --rosenbrock-a 100
uv run python -m src.adaptive_clustering --tol 1e-2 --dim 2 --n-train 100000 --n-init-clusters 20 --n-min-per-cluster 500 --rosenbrock-a 100 
uv run python -m src.adaptive_clustering --tol 1e-3 --dim 2 --n-train 100000 --n-init-clusters 20 --n-min-per-cluster 500 --rosenbrock-a 100 
uv run python -m src.adaptive_clustering --tol 1e-4 --dim 2 --n-train 100000 --n-init-clusters 20 --n-min-per-cluster 500 --rosenbrock-a 100

uv run python -m src.adaptive_clustering --tol 1e-3 --dim 2 --n-train 100000 --n-init-clusters 20 --n-min-per-cluster 500 --rosenbrock-a 0.001
uv run python -m src.adaptive_clustering --tol 1e-3 --dim 2 --n-train 100000 --n-init-clusters 20 --n-min-per-cluster 500 --rosenbrock-a 1
uv run python -m src.adaptive_clustering --tol 1e-3 --dim 2 --n-train 100000 --n-init-clusters 20 --n-min-per-cluster 500 --rosenbrock-a 100000