"""
Work stealing with per-point gradients, launched under MPI by tests/mpi/test_mpi.py.

Trains a second-order Taylor surrogate (TAYLOR_LSQ_HESS with STORE_HESSIAN), which
carries gradients at every point, so every steal must move them too. Thresholds are
lowered so steals happen at test size, and rank 0 holds the single initial cluster
until a steal request is pending, which forces a level-boundary grant. Checks:

- every grant was received, with its gradients, and nothing is left in flight;
- work is conserved (accepted + unresolved points = initial + generated points);
- predictions reproduce the quadratic exactly, whatever the schedule was, since
  every leaf's second-order model is exact for a quadratic.
"""

import json

import numpy as np

from tests.mpi._mpi_worker import RESULT_PREFIX

A = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, -0.5], [0.0, -0.5, 4.0]])
B = np.array([1.0, -2.0, 0.5])


def main() -> None:
    import src
    import src.engine.work_stealing as ws
    from src.components.modes import SurrogateMode
    from src.infra.backend import MPI
    from src.surrogate import AFINNISurrogate
    from tests._helpers import train_surrogate

    xp = src.xp
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    ws.MIN_LOAD_THRESHOLD, ws.STEAL_SIZE = 1_000, 2_000
    n_train = 40_000
    queued = False

    def f(x):
        nonlocal queued
        if rank == 0 and len(x) < n_train and not queued and world.Get_size() > 1:
            world.Probe(source=MPI.ANY_SOURCE, tag=ws.TAG_STEAL_REQ)  # see module docstring
            queued = True
        return x @ xp.asarray(B) + 0.5 * xp.einsum("nc,cd,nd->n", x, xp.asarray(A), x)

    def g(x):
        return xp.asarray(B) + x @ xp.asarray(A)

    rng = np.random.default_rng(0)
    X = xp.asarray(rng.uniform(-1, 1, (n_train, 3))) if rank == 0 else xp.empty((0, 3))
    s = AFINNISurrogate()
    train_surrogate(
        X, s, f_fn=f, g_fn=g, mode=SurrogateMode.TAYLOR_LSQ_HESS, store_hess=True,
        atol=0.01, rtol=0.01, error_metric="Linf", num_initial_clusters=1,
        min_pts_per_cluster=50, use_work_stealing=True, profile_events=True,
    )  # fmt: skip
    world.Barrier()
    stray = bool(world.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG))
    stray = world.allreduce(stray, op=MPI.LOR)

    Q = rng.uniform(-1, 1, (2_000, 3))
    pred, _ = s.predict(xp.asarray(Q) if rank == 0 else None, k_nn=4)
    if rank != 0:
        return
    ps = s.profiling_stats
    events = [e for r in ps["per_rank"] for e in r["events"]]
    grants = [e for e in events if e["kind"] == "steal_grant"]
    received = [e for e in events if e["kind"] == "steal_received"]
    truth = Q @ B + 0.5 * np.einsum("nc,cd,nd->n", Q, A, Q)
    err = float(np.max(np.abs(np.asarray(src._to_host(pred)).ravel() - truth)))
    accepted = int(np.asarray(s.counts_src).sum())
    result = {
        "grants": len(grants),
        "received": len(received),
        "all_with_gradients": all(e["has_g"] for e in received),
        "moved_pts": sum(e["n_pts"] for e in grants) == sum(e["n_pts"] for e in received),
        "stray": stray,
        "conserved": accepted + s.num_unresolved_pts == ps["N0"] + ps["num_generated_pts"],
        "converged": s.converged,
        "max_abs_err": err,
    }
    result["ok"] = bool(
        result["grants"] > 0 and result["grants"] == result["received"]
        and result["all_with_gradients"] and result["moved_pts"] and not stray
        and result["conserved"] and result["converged"] and err < 1e-8
    )  # fmt: skip
    print(RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
