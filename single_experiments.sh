#!/bin/bash

## Adaptive Clustering Experiments ##
# Comparing global & local for min_pts_per_cluster = 5, 25, 125, 625, 1000 for rosenbrock - a = 100, tol = 0.1
uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 100 --error-scaling-method global --n-min-per-cluster 5 --output-subdir compare_global_local_vs_min_pts
uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 100 --error-scaling-method local --n-min-per-cluster 5 --output-subdir compare_global_local_vs_min_pts

uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 100 --error-scaling-method global --n-min-per-cluster 25 --output-subdir compare_global_local_vs_min_pts
uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 100 --error-scaling-method local --n-min-per-cluster 25 --output-subdir compare_global_local_vs_min_pts

uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 100 --error-scaling-method global --n-min-per-cluster 125 --output-subdir compare_global_local_vs_min_pts
uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 100 --error-scaling-method local --n-min-per-cluster 125 --output-subdir compare_global_local_vs_min_pts

uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 100 --error-scaling-method global --n-min-per-cluster 625 --output-subdir compare_global_local_vs_min_pts
uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 100 --error-scaling-method local --n-min-per-cluster 625 --output-subdir compare_global_local_vs_min_pts

uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 100 --error-scaling-method global --n-min-per-cluster 1000 --output-subdir compare_global_local_vs_min_pts
uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 100 --error-scaling-method local --n-min-per-cluster 1000 --output-subdir compare_global_local_vs_min_pts

# Comparing global (only) vs n_init-clusters = 1, 5, 25, 125, 625, 1000 for rosenbrock - a = 100, tol = 0.001
uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.001 --rosenbrock-a 100 --error-scaling-method global --output-subdir compare_global_local_vs_n_init
uv run python -m src.adaptive_clustering --n-init-clusters 5 --tol 0.001 --rosenbrock-a 100 --error-scaling-method global --output-subdir compare_global_local_vs_n_init
uv run python -m src.adaptive_clustering --n-init-clusters 25 --tol 0.001 --rosenbrock-a 100 --error-scaling-method global --output-subdir compare_global_local_vs_n_init
uv run python -m src.adaptive_clustering --n-init-clusters 125 --tol 0.001 --rosenbrock-a 100 --error-scaling-method global --output-subdir compare_global_local_vs_n_init
uv run python -m src.adaptive_clustering --n-init-clusters 625 --tol 0.001 --rosenbrock-a 100 --error-scaling-method global --output-subdir compare_global_local_vs_n_init
uv run python -m src.adaptive_clustering --n-init-clusters 1000 --tol 0.001 --rosenbrock-a 100 --error-scaling-method global --output-subdir compare_global_local_vs_n_init

# Comparing global & local for rosenbrock - a = 0.01, 1, 100, tol = 0.1
uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 0.01 --error-scaling-method global --output-subdir compare_global_local_vs_a
uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 0.01 --error-scaling-method local --output-subdir compare_global_local_vs_a

uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 1 --error-scaling-method global --output-subdir compare_global_local_vs_a
uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 1 --error-scaling-method local --output-subdir compare_global_local_vs_a

uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 100 --error-scaling-method global --output-subdir compare_global_local_vs_a
uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 100 --error-scaling-method local --output-subdir compare_global_local_vs_a

uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 10000 --error-scaling-method global --output-subdir compare_global_local_vs_a
uv run python -m src.adaptive_clustering --n-init-clusters 1 --tol 0.1 --rosenbrock-a 10000 --error-scaling-method local --output-subdir compare_global_local_vs_a
