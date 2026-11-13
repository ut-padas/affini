"""Named presets for ``ClusteringConfig.mode``."""

from enum import Enum


class SurrogateMode(Enum):
    """
    Named presets, each a pair (local model on the leaves, split strategy for
    rejected clusters): see ``src.components.local_models.local_model_for_mode`` and
    ``src.components.splitting.split_strategy_for_mode``.
    """

    NEAREST_NEIGHBOR = "nearest_neighbor"  #  No grads, f(mu) only, split via X^T X
    AFFINE_COV_NO_HESS = "affine_cov_no_hess"  # No grads, fit A/b, split via X^T X
    TAYLOR_COV_NO_HESS = (
        "taylor_cov_no_hess"  # Grads at centroids only, split via X^T X
    )
    AFFINE_COV_HESS = "affine_cov_hess"  # No grads, fit A/b, split via X^T H X, where H is estimated from covariance
    AFFINE_COV_HESS_EXACT_GRAD = "affine_cov_hess_exact_grad" # Same as above, but with Hessian estimated with exact gradient.
    AFFINE_EXACT_HESS = "affine_exact_hess" # No grads, fit A/b, split via X^T H X, where H is given by an exact formula
    TAYLOR_COV_HESS = "taylor_cov_hess"  # Grads at centroids only, split via X^T H X, where H is estimated from covariance
    TAYLOR_EXACT_HESS = "taylor_exact_hess" # Grads at centroids only, split via X^T H X, where H is given by an exact formula
    TAYLOR_LSQ_HESS = "taylor_lsq_hess"  # Grads everywhere, split via X^T H X, where H is estimated from LSQ fit to gradient residuals
