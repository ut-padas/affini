#!/usr/bin/env bash


# Table 1: Convergence results for standard test functions.
# Prediction checks the kappa = max(50, 2 d^2) nearest leaf centres for a trust region that
# contains the query (--coarse_factor): 50 at d = 2 and 5, 200 at d = 10. Fewer candidates
# miss elongated trust regions, which raises the share of queries flagged outside.
# Quadratic
uv run python afinni_launcher.py --dims 5 --tols 0.1 0.05 0.02 0.01 0.005 --experiment anisotropic_quadratic --error_eval_mode affine_cov_hess --coarse_factor 50
uv run python afinni_launcher.py --dims 10 --tols 0.1 0.05 0.02 0.01 --experiment anisotropic_quadratic --error_eval_mode affine_cov_hess --coarse_factor 200
# Gaussian
uv run python afinni_launcher.py --dims 2 --tols 0.1 0.05 0.02 0.01 0.005 --experiment anisotropic_gaussian --error_eval_mode affine_cov_hess --coarse_factor 50
uv run python afinni_launcher.py --dims 5 --tols 0.1 0.05 0.02 0.01 0.005 --experiment anisotropic_gaussian --error_eval_mode affine_cov_hess --coarse_factor 50
# Rosenbrock
uv run python afinni_launcher.py --dims 2 --tols 0.1 0.05 0.02 0.01 0.005 --experiment rosenbrock --error_eval_mode affine_cov_hess --coarse_factor 50
uv run python afinni_launcher.py --dims 5 --tols 0.1 0.05 0.02 0.01 0.005 --experiment rosenbrock --error_eval_mode affine_cov_hess --coarse_factor 50
# Lorenz96
uv run python afinni_launcher.py --dims 5 --tols 0.1 0.05 0.02 0.01 0.005 --experiment lorenz96 --error_eval_mode affine_cov_hess --coarse_factor 50
# Wave
uv run python afinni_launcher.py --dims 2 --tols 0.1 0.05 0.02 0.01 0.005 --experiment wave --error_eval_mode affine_cov_hess --coarse_factor 50
uv run python afinni_launcher.py --dims 5 --tols 0.1 0.05 0.02 0.01 0.005 --experiment wave --error_eval_mode affine_cov_hess --coarse_factor 50


# Table 2: Lorenz96 dimension scaling, with the launcher's default kappa = 10: max(50, 2 d^2)
# would gather kappa d x d trust matrices per query, about 120 MB per query at d = 80.
uv run python afinni_launcher.py --dims 5 10 20 40 80 --tols 0.1 --experiment lorenz96 --error_eval_mode affine_cov_hess


# Figure 8: Wave 5D with exact gradients
uv run python afinni_launcher.py --dims 5 --tols 0.1 0.05 0.02 0.01 0.005 --experiment wave --error_eval_mode affine_cov_hess_exact_grad
