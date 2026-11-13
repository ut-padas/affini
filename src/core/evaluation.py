"""The wrapper around the user's function (and its derivatives) used during training."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from src import Array
from src.numerics import convert_to_2d_array, convert_to_3d_array


@dataclass
class Evaluator:
    """
    Evaluates the target function, and optionally its Jacobian, and counts evaluations.

    Provide ``f_fn`` (with ``g_fn`` if gradients are needed), or ``oracle_fn``, which
    returns ``(F, G)`` in one call. When gradients are not needed and ``f_fn`` is
    available, only ``f_fn`` is called.

    ``hessian_fn`` (optional) returns the exact Hessian, shape (N, d, d); it is
    required by the *_EXACT_HESS modes and is not included in ``n_evals``.
    """

    f_fn: Optional[Callable[[Array], Array]] = None
    g_fn: Optional[Callable[[Array], Array]] = None
    oracle_fn: Optional[Callable[[Array], tuple[Array, Array]]] = None
    hessian_fn: Optional[Callable[[Array], Array]] = None
    n_evals: int = 0  # number of points evaluated so far
    # Wall time spent in the user's callables. Device work they launch
    # asynchronously is only counted once it is synchronized.
    seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.f_fn is None and self.oracle_fn is None:
            raise ValueError("Provide f_fn or oracle_fn.")

    @property
    def has_grad(self) -> bool:
        """True if gradients can be computed (g_fn or oracle_fn was given)."""
        return self.oracle_fn is not None or self.g_fn is not None

    def __call__(
        self, X: Array, need_grad: bool = False
    ) -> tuple[Array, Optional[Array]]:
        """
        Return ``F`` with shape (N, fd) and, if ``need_grad``, ``G`` with shape
        (N, fd, d); otherwise ``G`` is None.
        """
        if need_grad and not self.has_grad:
            raise ValueError("Gradients requested but no g_fn or oracle_fn was given.")
        self.n_evals += int(X.shape[0])
        start = time.perf_counter()
        try:
            return self._evaluate(X, need_grad)
        finally:
            self.seconds += time.perf_counter() - start

    def _evaluate(self, X: Array, need_grad: bool) -> tuple[Array, Optional[Array]]:
        if self.f_fn is not None and not need_grad:
            return convert_to_2d_array(self.f_fn(X)), None
        if self.oracle_fn is not None:
            F, G = self.oracle_fn(X)
            return convert_to_2d_array(F), (convert_to_3d_array(G) if need_grad else None)
        assert self.f_fn is not None and self.g_fn is not None
        return convert_to_2d_array(self.f_fn(X)), convert_to_3d_array(self.g_fn(X))
