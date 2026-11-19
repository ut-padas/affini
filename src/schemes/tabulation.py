"""
SlowTable: in-situ tabulation of a step map over the states a simulation visits (ISAT's design).

A record holds a visited state ``z_i`` (log10 abundances), the increment ``g_i = Phi(z_i) - z_i`` and
its Jacobian ``A_i``, and answers ``g_i + A_i (z - z_i)`` inside its ellipsoid of accuracy
``(z - z_i)^T B_i (z - z_i) <= 1``. The ellipsoid starts as the region where the linear change stays
within the tolerance, ``B_i = A_i^T A_i / atol^2`` plus a floor that caps its semi-axes at ``r_max``:
along directions the increment does not depend on (the fast modes, which have relaxed onto the slow
manifold, and conserved combinations) it is as long as the cap allows, so one record covers a whole
slice of states that differ only in quantities the step forgets. That is how the slow manifold enters.

Training streams the training states in order, as a simulation would query them:

- a query inside some record's ellipsoid is retrieved; a share ``check`` of retrievals is solved anyway
  and, where the answer misses the tolerance, the record's ellipsoid shrinks to put the query just
  outside (Lu and Pope's error checking);
- an uncovered query is solved. If the nearest record's linear answer is within the tolerance its
  ellipsoid grows to cover the query (the minimal growth that keeps the centre); otherwise the query
  becomes a new record, its Jacobian from forward differences.

After training the table is frozen: a query covered by an ellipsoid is answered by the record whose
scaled distance is smallest, and every other query is refused, to be answered by the solver. The
ellipsoids are estimates, not bounds: the error is second order in the distance and the growth trusts
one solved point, so answers can exceed the tolerance near an ellipsoid's edge, and the error checks
catch only a share of them.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SlowTable:
    atol: float
    r_max: float
    Z: list = field(default_factory=list)  # record states (d,)
    G: list = field(default_factory=list)  # increments (m,)
    A: list = field(default_factory=list)  # Jacobians (m, d)
    B: list = field(default_factory=list)  # ellipsoids (d, d)
    evals: int = 0
    retrieved: int = 0
    grown: int = 0
    shrunk: int = 0

    def _arrays(self):
        return np.array(self.Z), np.array(self.G), np.array(self.A), np.array(self.B)

    def scaled_distance(self, Q: np.ndarray) -> np.ndarray:
        """(N, records) ``(q - z_i)^T B_i (q - z_i)``: inside an ellipsoid below 1."""
        Z, _, _, B = self._arrays()
        D = Q[:, None, :] - Z[None]  # (N, n, d)
        return np.einsum("qnd,nde,qne->qn", D, B, D)

    def predict(self, Q: np.ndarray) -> np.ndarray:
        Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
        Z, G, A, _ = self._arrays()
        out = np.empty((len(Q), G.shape[1]))
        for s in range(0, len(Q), 2048):
            q = Q[s : s + 2048]
            i = np.argmin(self.scaled_distance(q), axis=1)
            out[s : s + 2048] = G[i] + np.einsum("qmd,qd->qm", A[i], q - Z[i])
        return out

    def refuses(self, Q: np.ndarray) -> np.ndarray:
        Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
        return np.concatenate([self.scaled_distance(Q[s : s + 2048]).min(axis=1) > 1.0 for s in range(0, len(Q), 2048)])

    def add(self, z, g, A) -> None:
        d = len(z)
        self.Z.append(z), self.G.append(g), self.A.append(A)
        self.B.append(A.T @ A / self.atol**2 + np.eye(d) / self.r_max**2)


def _rescale_along(B: np.ndarray, delta: np.ndarray, target: float) -> np.ndarray:
    """B changed so that the point ``delta`` has scaled distance ``target`` (1: on the surface), only along its direction in B's own metric."""
    L = np.linalg.cholesky(B)
    y = L.T @ delta
    r2 = float(y @ y)
    u = y / np.sqrt(r2)
    # in y-coordinates the ellipsoid is the unit ball; scale the axis along u so that |y|^2 -> target
    return L @ (np.eye(len(u)) + (target / r2 - 1.0) * np.outer(u, u)) @ L.T


def jacobian(step, z: np.ndarray, g: np.ndarray, h: float = 1e-4) -> tuple[np.ndarray, int]:
    """Forward-difference Jacobian of the increment at z (one batched call of d states) and its evaluation count."""
    d = len(z)
    Zp = z[None] + h * np.eye(d)
    Gp = step(Zp) - Zp
    return (Gp - g[None]).T / h, d


def build(step, states: np.ndarray, atol: float, safety: float = 0.5, r_max: float = 1.0, check: float = 0.05,
          seed: int = 0) -> SlowTable:  # fmt: skip
    """
    Train a SlowTable on ``states`` (N, d) in stream order; ``step(Z) -> Phi(Z)`` is the oracle in the
    same coordinates (the increment is ``Phi(z) - z``). Accepts at ``safety * atol``, so the frozen
    table has some margin.
    """
    rng = np.random.default_rng(seed)
    tol = safety * atol
    t = SlowTable(tol, r_max)
    for q in np.asarray(states, dtype=np.float64):
        if t.Z:
            dist = t.scaled_distance(q[None])[0]
            i = int(np.argmin(dist))
            if dist[i] <= 1.0:
                t.retrieved += 1
                if rng.random() >= check:
                    continue
                gq = step(q[None])[0] - q
                t.evals += 1
                if np.max(np.abs(t.G[i] + t.A[i] @ (q - t.Z[i]) - gq)) > tol:  # a wrong retrieval: shrink
                    t.B[i] = _rescale_along(t.B[i], q - t.Z[i], 1.05)
                    t.shrunk += 1
                continue
        gq = step(q[None])[0] - q
        t.evals += 1
        if t.Z:
            dist = t.scaled_distance(q[None])[0]
            i = int(np.argmin(dist))
            if np.max(np.abs(t.G[i] + t.A[i] @ (q - t.Z[i]) - gq)) <= tol:
                t.B[i] = _rescale_along(t.B[i], q - t.Z[i], 1.0)
                t.grown += 1
                continue
        A, n = jacobian(step, q, gq)
        t.evals += n
        t.add(q, gq, A)
    return t
