#!/usr/bin/env bash


# Table 1: Convergence results for standard test functions
# Quadratic
uv run python afinni_launcher.py --dims 5 --tols 0.1 0.05 0.02 0.01 0.005 --experiment anisotropic_quadratic --error_eval_mode affine_cov_hess --coarse_factor 50
uv run python afinni_launcher.py --dims 10 --tols 0.1 0.05 0.02 0.01 --experiment anisotropic_quadratic --error_eval_mode affine_cov_hess --coarse_factor 200
# Gaussian
uv run python afinni_launcher.py --dims 2 --tols 0.1 0.05 0.02 0.01 0.005 --experiment anisotropic_gaussian --error_eval_mode affine_cov_hess --coarse_factor 10
uv run python afinni_launcher.py --dims 5 --tols 0.1 0.05 0.02 0.01 0.005 --experiment anisotropic_gaussian --error_eval_mode affine_cov_hess --coarse_factor 50
# Rosenbrock
uv run python afinni_launcher.py --dims 2 --tols 0.1 0.05 0.02 0.01 0.005 --experiment rosenbrock --error_eval_mode affine_cov_hess --coarse_factor 10
uv run python afinni_launcher.py --dims 5 --tols 0.1 0.05 0.02 0.01 0.005 --experiment rosenbrock --error_eval_mode affine_cov_hess --coarse_factor 50
# Lorenz96
uv run python afinni_launcher.py --dims 5 --tols 0.1 0.05 0.02 0.01 0.005 --experiment lorenz96 --error_eval_mode affine_cov_hess --coarse_factor 50
# Wave
uv run python afinni_launcher.py --dims 2 --tols 0.1 0.05 0.02 0.01 0.005 --experiment wave --error_eval_mode affine_cov_hess --coarse_factor 10
uv run python afinni_launcher.py --dims 5 --tols 0.1 0.05 0.02 0.01 0.005 --experiment wave --error_eval_mode affine_cov_hess --coarse_factor 50


# Table 2: Lorenz96 dimension scaling
uv run python afinni_launcher.py --dims 5 10 20 40 80 --tols 0.1 --experiment lorenz96 --error_eval_mode affine_cov_hess


# Figure 8: Wave 5D with exact gradients
uv run python afinni_launcher.py --dims 5 --tols 0.1 0.05 0.02 0.01 0.005 --experiment wave --error_eval_mode affine_cov_hess_exact_grad
