"""
Acceptance criteria: when is a cluster's local model accurate enough?

After the local model is fitted on every cluster of a batch, the refinement loop
evaluates it at the cluster's own points (``F_hat``) and asks an
``AcceptanceCriterion`` which clusters to accept. Accepted clusters become leaves;
the others are augmented and split.

Every built-in criterion measures, per cluster ``k`` and output component ``j``,
an absolute error ``abs_kj`` and a relative error ``rel_kj``, and accepts the
cluster if **every** component passes either test:

    rel_kj < rtol_j  or  abs_kj < atol_j   for all j

``atol`` and ``rtol`` are each a scalar (used for every component), a sequence of
one value per component, or ``None``; at least one must be given. An omitted
tolerance never accepts, so ``rtol`` alone is a purely relative test and ``atol``
alone a purely absolute one. To scale an absolute tolerance by the function's
magnitude, pass e.g. ``atol = tol * max|F_j|`` per component.

Built-in criteria (``ClusteringConfig.error_metric``), with ``e = F - F_hat`` over
the cluster's points ``i``:

``"L2"``
    ``rel_kj = ||e_.j||_2 / ||F_.j||_2`` and ``abs_kj = sqrt(sum_i e_ij^2 / n)``
    (RMS over the cluster's points).
``"Linf"``
    ``abs_kj = max_i |e_ij|`` and ``rel_kj = abs_kj / max_i |F_ij|`` (relative to
    the cluster's own largest value of that component).

All comparisons are strict (``<``). A small epsilon (1e-12) guards each
denominator. Both errors are returned, (K, fd), and stored per leaf as
``E_cluster_abs`` and ``E_cluster_rel``.

Adding a criterion: subclass ``AcceptanceCriterion``, implement ``__call__`` (usually
computing the two errors and returning ``self.accept(E_abs, E_rel)``), and register it
in ``ACCEPTANCE_CRITERIA``. Criteria built by name receive the configuration's
``atol`` and ``rtol`` as keyword arguments.
"""

import numbers
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from src import Array, _to_host, xp
from src.numerics import TINY_EPS, _reduce_max_1d, _reduce_sum_1d
from src.registry import Registry

Tolerance = float | Sequence[float] | Array | None


def normalize_tolerance(name: str, value: Tolerance) -> float | tuple[float, ...] | None:
    """Validate a tolerance and return it as a float or tuple of floats (serializable, hashable), or None."""
    if value is None:
        return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        values = np.array([value], dtype=np.float64)
        scalar = True
    else:
        values = np.asarray(_to_host(value), dtype=np.float64)
        scalar = values.ndim == 0
        if values.ndim > 1 or values.size == 0:
            raise ValueError(f"{name} must be a number or a 1-D sequence of numbers; got {value!r}.")
        values = values.reshape(-1)
    if not (np.all(np.isfinite(values)) and np.all(values >= 0)):
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}.")
    return float(values[0]) if scalar else tuple(float(v) for v in values)


def tolerance_vector(name: str, tol: float | tuple[float, ...] | None, fd: int) -> Array:
    """``tol`` as an (fd,) array; an omitted tolerance is -inf, so neither ``<`` nor ``<=`` ever accepts on it."""
    if tol is None:
        return xp.full(fd, -xp.inf)
    if isinstance(tol, tuple) and len(tol) != fd:
        raise ValueError(f"{name} has {len(tol)} components but the function has {fd} outputs.")
    return xp.broadcast_to(xp.asarray(tol, dtype=xp.float64), (fd,))


class AcceptanceCriterion(ABC):
    """Decides which clusters of a batch are accepted; ``atol``/``rtol`` are normalized and at least one is set."""

    name: str

    def __init__(self, atol: Tolerance = None, rtol: Tolerance = None):
        self.atol = normalize_tolerance("atol", atol)
        self.rtol = normalize_tolerance("rtol", rtol)
        if self.atol is None and self.rtol is None:
            raise ValueError("Give atol, rtol or both: an acceptance test needs a tolerance.")

    def settings(self) -> dict:
        return {"atol": self.atol, "rtol": self.rtol}

    def check_output_dim(self, fd: int) -> None:
        """Raise before training if a per-component tolerance does not match the ``fd`` outputs."""
        tolerance_vector("atol", self.atol, fd)
        tolerance_vector("rtol", self.rtol, fd)

    def accept(self, E_abs: Array, E_rel: Array) -> tuple[Array, Array, Array]:
        """Apply the per-component rule to (K, fd) errors; returns ``(is_good, E_abs, E_rel)``."""
        fd = E_abs.shape[1]
        atol, rtol = tolerance_vector("atol", self.atol, fd), tolerance_vector("rtol", self.rtol, fd)
        is_good = xp.all((E_rel < rtol[None, :]) | (E_abs < atol[None, :]), axis=1)
        return is_good, E_abs, E_rel

    @abstractmethod
    def __call__(
        self, F: Array, F_hat: Array, starts: Array, sizes: Array
    ) -> tuple[Array, Array, Array]:
        """
        Return ``(is_good, E_abs, E_rel)``: whether each cluster is accepted (K,), and
        its absolute and relative errors per output component (K, fd), which are
        stored with each accepted leaf.

        ``F`` and ``F_hat`` are (N, fd), the true and fitted values at the batch's
        points, stored contiguously by cluster (cluster ``k`` owns rows
        ``starts[k] : starts[k] + sizes[k]``).
        """


ACCEPTANCE_CRITERIA: Registry[AcceptanceCriterion] = Registry(
    "acceptance criterion", AcceptanceCriterion
)


@ACCEPTANCE_CRITERIA.register
class L2Acceptance(AcceptanceCriterion):
    """Per component: relative L2 error < rtol, or RMS error < atol."""

    name = "L2"

    def __call__(self, F, F_hat, starts, sizes):
        diff = F - F_hat
        # Per-component loop keeps each reduction 1-D, so a single output reduces exactly as before.
        sq_err = xp.stack([_reduce_sum_1d(diff[:, j] ** 2, starts) for j in range(F.shape[1])], axis=1)
        sq_val = xp.stack([_reduce_sum_1d(F[:, j] ** 2, starts) for j in range(F.shape[1])], axis=1)
        E_rel = xp.sqrt(sq_err) / (xp.sqrt(sq_val) + TINY_EPS)
        E_abs = xp.sqrt(sq_err / sizes[:, None])
        return self.accept(E_abs, E_rel)


@ACCEPTANCE_CRITERIA.register
class LinfAcceptance(AcceptanceCriterion):
    """Per component: max error relative to the cluster's own max |F| < rtol, or max error < atol."""

    name = "Linf"

    def __call__(self, F, F_hat, starts, sizes):
        diff = F - F_hat
        E_abs = xp.stack([_reduce_max_1d(xp.abs(diff[:, j]), starts) for j in range(F.shape[1])], axis=1)
        local_max_F = xp.stack([_reduce_max_1d(xp.abs(F[:, j]), starts) for j in range(F.shape[1])], axis=1)
        E_rel = E_abs / xp.maximum(local_max_F, TINY_EPS)
        return self.accept(E_abs, E_rel)


def cluster_errors(
    config,
    F_batch: Array,
    F_hat: Array,
    aug_starts: Array,
    aug_sizes: Array,
) -> tuple[Array, Array, Array]:
    """The configured criterion's ``(is_good, E_abs, E_rel)`` for each cluster (rows ``aug_starts[k] : + aug_sizes[k]``)."""
    criterion = config.components().acceptance
    return criterion(F_batch, F_hat, aug_starts, aug_sizes)
