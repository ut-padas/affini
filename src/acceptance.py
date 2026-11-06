"""
Acceptance criteria: when is a cluster's local model accurate enough?

After the local model is fitted on every cluster of a batch, the refinement loop
evaluates it at the cluster's own points (``F_hat``) and asks an
``AcceptanceCriterion`` which clusters to accept. Accepted clusters become leaves;
the others are augmented and split.

Built-in criteria (``ClusteringConfig.error_metric``) and their exact semantics,
with ``e = F - F_hat`` and ``|.|`` the absolute value per output component:

``"L2"``
    Accept if ``||e||_2 / ||F||_2 < rtol`` (relative, over the cluster's points
    and components) or ``sqrt(sum(e^2) / n) < atol`` (RMS over points).
``"Linf-local"``
    Accept if ``max_j max_i |e_ij| / max_i |F_ij| < rtol`` (relative to the
    cluster's own largest value of each component) or ``max_ij |e_ij| < atol``.
``"Linf-global"``
    Accept if ``max_ij |e_ij| / max|F_j| < atol``, where ``max|F_j|`` is the
    largest absolute value of component ``j`` over the *initial* data
    (``F_max_global``). This is a relative error, but it is compared with
    **atol**; ``rtol`` is not used. (With ``atol == rtol``, the default and what the
    launcher uses, this is the same as comparing with ``rtol``.)

All comparisons are strict (``<``). A small epsilon (1e-12) guards each
denominator. ``E_cluster``, the error reported for each cluster, is the relative
error for all three.

Adding a criterion: subclass ``AcceptanceCriterion``, implement ``__call__``, and
register it in ``ACCEPTANCE_CRITERIA``. Criteria built by name receive the
configuration's ``atol`` and ``rtol`` as keyword arguments.
"""

from abc import ABC, abstractmethod

from src import Array, xp
from src.numerics import TINY_EPS, _reduce_max_1d, _reduce_sum_1d
from src.registry import Registry


class AcceptanceCriterion(ABC):
    """Decides which clusters of a batch are accepted."""

    name: str

    def __init__(self, atol: float, rtol: float):
        self.atol = float(atol)
        self.rtol = float(rtol)

    def settings(self) -> dict:
        return {"atol": self.atol, "rtol": self.rtol}

    @abstractmethod
    def __call__(
        self, F: Array, F_hat: Array, starts: Array, sizes: Array, F_max_global: Array
    ) -> tuple[Array, Array]:
        """
        Return ``(is_good, E_cluster)``, both (K,): whether each cluster is accepted,
        and the error reported for it.

        ``F`` and ``F_hat`` are (N, fd), the true and fitted values at the batch's
        points, stored contiguously by cluster (cluster ``k`` owns rows
        ``starts[k] : starts[k] + sizes[k]``). ``F_max_global`` (fd,) is max |F|
        per component over the initial data.
        """


ACCEPTANCE_CRITERIA: Registry[AcceptanceCriterion] = Registry(
    "acceptance criterion", AcceptanceCriterion
)


@ACCEPTANCE_CRITERIA.register
class L2Acceptance(AcceptanceCriterion):
    """Relative L2 error < rtol, or RMS error < atol."""

    name = "L2"

    def __call__(self, F, F_hat, starts, sizes, F_max_global):
        diff = F - F_hat
        # Calculate sum of squared errors
        err_pts = xp.sum(diff**2, axis=1)  # (N,)
        sq_err_cluster = _reduce_sum_1d(err_pts, starts)  # (K,)

        # Calculate sum of squared true values for relative scaling
        val_pts = xp.sum(F**2, axis=1)  # (N,)
        sq_val_cluster = _reduce_sum_1d(val_pts, starts)  # (K,)

        # Relative L2 Error
        E_cluster_rel = xp.sqrt(sq_err_cluster) / (xp.sqrt(sq_val_cluster) + TINY_EPS)

        # Absolute RMS Error (Average absolute error per point)
        E_cluster_abs = xp.sqrt(sq_err_cluster / sizes)

        # Pass if relative error is good, OR if absolute error is already negligible
        is_good = (E_cluster_rel < self.rtol) | (E_cluster_abs < self.atol)
        return is_good, E_cluster_rel


@ACCEPTANCE_CRITERIA.register
class LinfLocalAcceptance(AcceptanceCriterion):
    """Max error relative to the cluster's own max |F| (per component) < rtol, or max error < atol."""

    name = "Linf-local"

    def __call__(self, F, F_hat, starts, sizes, F_max_global):
        diff = F - F_hat
        K = len(sizes)
        E_cluster_rel = xp.zeros(K, dtype=xp.float64)
        E_cluster_abs = xp.zeros(K, dtype=xp.float64)
        for j in range(F.shape[1]):
            abs_diff_j = xp.abs(diff[:, j])  # (N,)
            local_max_F_j = _reduce_max_1d(xp.abs(F[:, j]), starts)  # (K,)
            max_err_j = _reduce_max_1d(abs_diff_j, starts)  # (K,)

            E_cluster_abs = xp.maximum(E_cluster_abs, max_err_j)  # (K,)
            E_cluster_rel = xp.maximum(
                E_cluster_rel, max_err_j / xp.maximum(local_max_F_j, TINY_EPS)
            )  # (K,)

        # Pass if relative error is good, OR if absolute error is already negligible
        is_good = (E_cluster_rel < self.rtol) | (E_cluster_abs < self.atol)
        return is_good, E_cluster_rel


@ACCEPTANCE_CRITERIA.register
class LinfGlobalAcceptance(AcceptanceCriterion):
    """
    Max error relative to max |F| over the initial data (per component) < **atol**.
    ``rtol`` is not used; see the module docstring.
    """

    name = "Linf-global"

    def __call__(self, F, F_hat, starts, sizes, F_max_global):
        diff = F - F_hat
        err_pts = xp.max(
            xp.abs(diff) / (F_max_global[None, :] + TINY_EPS), axis=1
        )  # (N,)
        E_cluster = _reduce_max_1d(err_pts, starts)  # (K,)
        is_good = E_cluster < self.atol  # (K,)
        return is_good, E_cluster


def cluster_errors(
    config,
    F_batch: Array,
    F_hat: Array,
    aug_starts: Array,
    aug_sizes: Array,
    F_max_global: Array,
) -> tuple[Array, Array]:
    """
    Error of the fitted values ``F_hat`` against the true values ``F_batch`` on
    each cluster (rows ``aug_starts[k] : aug_starts[k] + aug_sizes[k]``), using
    the configuration's acceptance criterion, and whether each cluster is within
    tolerance. Returns: (is_good_mask (K,), E_cluster (K,))
    """
    criterion = config.components().acceptance
    return criterion(F_batch, F_hat, aug_starts, aug_sizes, F_max_global)
