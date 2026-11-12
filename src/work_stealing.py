"""
Decentralized work stealing between MPI ranks for adaptive clustering.

Each rank refines its own queue of clusters. Ranks gossip their load (number of
queued points) to every other rank. A rank that runs out of work asks the most
loaded rank for some; the victim answers every request with either a grant
(clusters taken from the unprocessed tail of its queue) or a denial. Once every
rank is idle, all ranks retire together through a non-blocking barrier, still
answering requests until every rank has retired.

The refinement loop only talks to this module through ``WorkStealer`` (or
``NoWorkStealing`` when stealing is off) and ``src.clusters.ClusterBatch``.
"""

import logging
import time

import numpy as np

from src.backend import MPI  # real mpi4py.MPI, or a single-rank stand-in without MPI
from src.clusters import TREE_FIELDS, ClusterBatch
from src.profiling import Profiler

logger = logging.getLogger(__name__)

# --- MPI tags ---
TAG_STATE_BROADCAST = 101
TAG_STEAL_REQ = 102
TAG_STEAL_DENY = 104
TAG_STEAL_GRANT = 110  # Payload buffers use tags 111 through 115 (118 with tree fields).

# --- Tuning parameters ---
MIN_LOAD_THRESHOLD = 1_000_000  # Min points to be eligible for stealing
STEAL_SIZE = 500_000  # Minimum points per grant (a grant gives about half the open work)
SLEEP_INTERVAL = 0.05  # Seconds to sleep when idle or after failed steal


def send_clusters(comm, batch: ClusterBatch, dest: int, tag: int = TAG_STEAL_GRANT):
    """Send ``batch`` to ``dest``: a metadata message on ``tag``, then buffers on tag+1..tag+5 (tag+8 with tree fields)."""
    meta = {
        "n_pts": batch.n_pts,
        "n_clusters": batch.n_clusters,
        "has_g": batch.G is not None,
        "has_tree": batch.has_tree,
    }
    comm.send(meta, dest=dest, tag=tag)
    comm.Send(np.ascontiguousarray(batch.X), dest=dest, tag=tag + 1)
    comm.Send(np.ascontiguousarray(batch.F), dest=dest, tag=tag + 2)
    if batch.G is not None:
        comm.Send(np.ascontiguousarray(batch.G), dest=dest, tag=tag + 3)
    comm.Send(np.ascontiguousarray(batch.sizes), dest=dest, tag=tag + 4)
    comm.Send(np.ascontiguousarray(batch.depth, dtype=np.int64), dest=dest, tag=tag + 5)
    for i, (name, dtype) in enumerate(TREE_FIELDS.items() if batch.has_tree else ()):
        comm.Send(np.ascontiguousarray(getattr(batch, name), dtype=dtype), dest=dest, tag=tag + 6 + i)


def recv_clusters(
    comm, source: int, d: int, fd: int, tag: int = TAG_STEAL_GRANT
) -> ClusterBatch:
    """Receive a batch sent with ``send_clusters`` from ``source``."""
    meta = comm.recv(source=source, tag=tag)
    n_pts = meta["n_pts"]
    X = np.empty((n_pts, d), dtype=np.float64)
    F = np.empty((n_pts, fd), dtype=np.float64)
    G = np.empty((n_pts, fd, d), dtype=np.float64) if meta["has_g"] else None
    sizes = np.empty(meta["n_clusters"], dtype=np.int64)
    comm.Recv(X, source=source, tag=tag + 1)
    comm.Recv(F, source=source, tag=tag + 2)
    if G is not None:
        comm.Recv(G, source=source, tag=tag + 3)
    comm.Recv(sizes, source=source, tag=tag + 4)
    depth = np.empty(meta["n_clusters"], dtype=np.int64)
    comm.Recv(depth, source=source, tag=tag + 5)
    tree = {}
    for i, (name, dtype) in enumerate(TREE_FIELDS.items() if meta["has_tree"] else ()):
        tree[name] = np.empty(meta["n_clusters"], dtype=dtype)
        comm.Recv(tree[name], source=source, tag=tag + 6 + i)
    return ClusterBatch(X, F, G, sizes, depth, **tree)


class WorkStealer:
    """
    One rank's side of the work-stealing protocol.

    Call order from the refinement loop:
      - ``exchange_initial_loads`` once, before refining;
      - ``poll_state`` / ``set_load`` at the top of each iteration;
      - ``idle_step`` whenever the queue is empty (or the level limit is hit);
      - ``serve_requests`` between batches and at each level boundary,
        after ``publish_load`` with the new queue size at the boundary.
    """

    enabled = True

    def __init__(self, comm, d: int, fd: int, profiler: Profiler):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.d, self.fd = d, fd
        self.profiler = profiler

        self.status = MPI.Status()
        self.local_load = 0
        self.global_loads = np.zeros(self.size, dtype=np.int64)
        # Pre-allocated 1-element buffers for C-level MPI load updates.
        self._send_buf = np.empty(1, dtype=np.int64)
        self._recv_buf = np.empty(1, dtype=np.int64)
        # Exact counts, so retirement can drain buffered updates (see _retire).
        self._broadcasts_sent = 0
        self._broadcasts_received = np.zeros(self.size, dtype=np.int64)

    def _timer(self, key: str):
        return self.profiler.phase(key)

    # --- Load gossip -------------------------------------------------------

    def exchange_initial_loads(self, load: int) -> None:
        """
        Learn every initial load before deciding whether to retire. Otherwise an
        initially empty rank could mistake missing updates for global idle.
        """
        self.set_load(load)
        self._broadcast()
        for src in range(self.size):
            if src != self.rank:
                self.comm.Recv(self._recv_buf, source=src, tag=TAG_STATE_BROADCAST)
                self.global_loads[src] = self._recv_buf[0]
                self._broadcasts_received[src] += 1

    def poll_state(self) -> None:
        """Process any pending load updates from other ranks."""
        with self._timer("time_state_update"):
            self._receive_state()

    def set_load(self, load: int) -> None:
        """Record this rank's load locally, without telling other ranks."""
        self.local_load = load
        self.global_loads[self.rank] = load

    def publish_load(self, load: int) -> None:
        """Record this rank's load and send it to every other rank."""
        self.set_load(load)
        self._broadcast()
        self.profiler.event("load", n_pts=int(load))

    def _receive_state(self) -> None:
        while self.comm.Iprobe(
            source=MPI.ANY_SOURCE, tag=TAG_STATE_BROADCAST, status=self.status
        ):
            src = self.status.Get_source()
            self.comm.Recv(self._recv_buf, source=src, tag=TAG_STATE_BROADCAST)
            self.global_loads[src] = self._recv_buf[0]
            self._broadcasts_received[src] += 1

    def _broadcast(self) -> None:
        """Send this rank's load to every peer with blocking 8-byte sends (assumes eager delivery)."""
        self._broadcasts_sent += 1
        self._send_buf[0] = self.local_load
        for r in range(self.size):
            if r != self.rank:
                # 8 bytes is well within the MPI eager limit; this returns instantly.
                self.comm.Send(self._send_buf, dest=r, tag=TAG_STATE_BROADCAST)

    # --- Victim side -------------------------------------------------------

    def serve_requests(
        self,
        queue: ClusterBatch,
        start_cluster: int,
        level: int,
        batch: int | None = None,
    ) -> ClusterBatch:
        """
        Answer every pending steal request and return what is left of ``queue``.

        Clusters before ``start_cluster`` are already being processed and are never
        given away. A grant takes clusters from the tail, about half the unprocessed points
        (at least STEAL_SIZE),
        and always leaves at least one unprocessed cluster. ``batch`` is the batch
        index for mid-level calls, or None at a level boundary.
        """
        where = f"R{self.rank:02} L{level:02}" + (
            f" B{batch + 1}" if batch is not None else ""
        )
        while self.comm.Iprobe(
            source=MPI.ANY_SOURCE, tag=TAG_STEAL_REQ, status=self.status
        ):
            thief = self.status.Get_source()
            self.comm.recv(source=thief, tag=TAG_STEAL_REQ)

            n_open = queue.n_clusters - start_cluster
            open_pts = int(np.sum(queue.sizes[start_cluster:])) if n_open > 0 else 0
            if n_open > 1 and open_pts > MIN_LOAD_THRESHOLD:
                t_send = time.perf_counter()
                # Give about half of the unprocessed work (at least STEAL_SIZE points),
                # so a heavily loaded victim sheds its surplus in a few grants.
                target = max(STEAL_SIZE, open_pts // 2)
                rev_cumsum = np.cumsum(queue.sizes[start_cluster:][::-1])
                n_give = min(int(np.searchsorted(rev_cumsum, target)) + 1, n_open - 1)
                queue, gift = queue.split(queue.n_clusters - n_give)
                send_clusters(self.comm, gift, thief)
                elapsed = time.perf_counter() - t_send
                self.profiler.event(
                    "steal_grant", thief=thief, n_pts=gift.n_pts,
                    n_clusters=gift.n_clusters, mid_level=batch is not None, dur=elapsed,
                )  # fmt: skip
                self.publish_load(queue.n_pts)

                kind = "mid-level steal" if batch is not None else "steal"
                logger.info(
                    f"{where}: Granted {kind} to R{thief:02} | "
                    f"gave {gift.n_pts:,} pts {gift.n_clusters:,} clusters | "
                    f"send={elapsed:.3f}s | "
                    f"remaining pts={queue.n_pts:,} clusters={queue.n_clusters:,}"
                )
                self.profiler.add_time("time_mpi_send", elapsed)
            else:
                if batch is not None and n_open <= 1:
                    reason = "MID_LEVEL_NO_CLUSTERS"
                elif batch is not None:
                    reason = f"MID_LEVEL_LOAD ({open_pts:,} <= threshold {MIN_LOAD_THRESHOLD:,})"
                elif open_pts <= MIN_LOAD_THRESHOLD:
                    reason = f"LOAD ({open_pts:,} <= threshold {MIN_LOAD_THRESHOLD:,})"
                else:
                    reason = (
                        f"STEAL_SIZE (only {queue.n_clusters} cluster(s), need > 1)"
                    )
                self.comm.send(reason, dest=thief, tag=TAG_STEAL_DENY)
                self.profiler.event("steal_deny", thief=thief, reason=reason)
                if batch is None:
                    logger.debug(f"{where}: Denied steal to R{thief:02}: {reason}")
        return queue

    def _deny_idle_requests(self) -> None:
        """Deny pending requests; idle ranks must answer even while awaiting their own reply."""
        while self.comm.Iprobe(
            source=MPI.ANY_SOURCE, tag=TAG_STEAL_REQ, status=self.status
        ):
            src = self.status.Get_source()
            self.comm.recv(source=src, tag=TAG_STEAL_REQ)
            self.comm.send("IDLE", dest=src, tag=TAG_STEAL_DENY)

    # --- Thief side and retirement -----------------------------------------

    def idle_step(self, level_limit_reached: bool) -> tuple[bool, ClusterBatch | None]:
        """
        One idle iteration, for a rank with nothing (more) to refine.

        Returns ``(retired, stolen)``: ``retired`` is True once every rank has
        finished, and ``stolen`` is the work received from another rank, if any.
        """
        with self._timer("time_idle"):
            # A rank that hit max_levels may still hold data but will never process
            # it. Advertise 0 so the termination condition works correctly.
            if self.local_load != 0:
                self.publish_load(0)
            self._deny_idle_requests()
            if np.sum(self.global_loads) == 0 or level_limit_reached:
                self.profiler.event("retire_start")
                self._retire()
                self.profiler.event("retired")
                logger.info(f"R{self.rank:02}: All ranks retired. Terminating.")
                return True, None

            target = int(np.argmax(self.global_loads))
            if self.global_loads[target] < MIN_LOAD_THRESHOLD:
                time.sleep(SLEEP_INTERVAL)
                return False, None

            logger.debug(
                f"R{self.rank:02}: Idle. Attempting steal from R{target:02} "
                f"(Load: {self.global_loads[target]:,}) [my load: {self.local_load:,}]"
            )
            self.comm.send(self.rank, dest=target, tag=TAG_STEAL_REQ)
            self.profiler.event("steal_request", victim=target, victim_load=int(self.global_loads[target]))
            t_wait = time.perf_counter()
            while True:
                self.poll_state()
                # A zero-load update is not a reply. The victim must explicitly
                # deny the request or grant work, which we keep. Answer crossed
                # requests from other idle thieves too.
                self._deny_idle_requests()

                if self.comm.Iprobe(source=target, tag=TAG_STEAL_GRANT):
                    t_recv = time.perf_counter()
                    stolen = recv_clusters(self.comm, target, self.d, self.fd)
                    waited = time.perf_counter() - t_wait
                    transfer = time.perf_counter() - t_recv
                    self.profiler.event(
                        "steal_received", victim=target, n_pts=stolen.n_pts,
                        n_clusters=stolen.n_clusters, has_g=stolen.G is not None,
                        waited=waited, dur=transfer,
                    )  # fmt: skip
                    self.publish_load(stolen.n_pts)
                    logger.info(
                        f"R{self.rank:02}: Steal OK from R{target:02} | "
                        f"pts={stolen.n_pts:,} clusters={stolen.n_clusters:,} | "
                        f"waited={waited:.3f}s transfer={transfer:.3f}s"
                    )
                    self.profiler.add_time("time_mpi_recv", transfer)
                    return False, stolen

                if self.comm.Iprobe(source=target, tag=TAG_STEAL_DENY):
                    reason = self.comm.recv(source=target, tag=TAG_STEAL_DENY)
                    waited = time.perf_counter() - t_wait
                    self.profiler.event("steal_denied", victim=target, reason=reason, waited=waited)
                    logger.debug(
                        f"R{self.rank:02}: Steal denied by R{target:02} "
                        f"(reason={reason}, waited {waited:.3f}s). Sleeping."
                    )
                    time.sleep(SLEEP_INTERVAL)
                    return False, None

    def _retire(self) -> None:
        """
        Leave stealing once every rank has: keep denying requests until a non-blocking
        barrier completes, then drain exactly the load updates still buffered.

        Enter only with no local work and no outstanding request; a load snapshot
        alone cannot detect work in transit.
        """
        if self.size == 1:
            return
        retirement = self.comm.Ibarrier()
        while not retirement.Test():
            self._receive_state()
            self._deny_idle_requests()
            time.sleep(0.001)

        # Every request now has its response and every transfer was consumed.
        # Blocking state sends can still be buffered. Receive their exact
        # remaining counts, rather than relying on a momentarily empty probe.
        sent_counts = self.comm.allgather(self._broadcasts_sent)
        for src in range(self.size):
            if src != self.rank:
                for _ in range(sent_counts[src] - self._broadcasts_received[src]):
                    self.comm.Recv(self._recv_buf, source=src, tag=TAG_STATE_BROADCAST)
                    self._broadcasts_received[src] += 1


class NoWorkStealing:
    """
    Drop-in for ``WorkStealer`` when stealing is disabled: every call is a no-op,
    except that a rank which runs out of work waits for the others as recorded idle
    time, so profiles show what stealing would recover.
    """

    enabled = False

    def __init__(self, comm, profiler: Profiler):
        self.comm, self.profiler = comm, profiler
        self.rank, self.size = comm.Get_rank(), comm.Get_size()

    def exchange_initial_loads(self, load: int) -> None:
        pass

    def poll_state(self) -> None:
        pass

    def set_load(self, load: int) -> None:
        pass

    def publish_load(self, load: int) -> None:
        pass

    def serve_requests(
        self,
        queue: ClusterBatch,
        start_cluster: int,
        level: int,
        batch: int | None = None,
    ) -> ClusterBatch:
        return queue

    def idle_step(self, level_limit_reached: bool) -> tuple[bool, ClusterBatch | None]:
        """Wait (as idle time) until every rank has finished, then retire."""
        logger.info(
            f"R{self.rank:02}: No clusters to process or exceeded max levels. Terminating."
        )
        if self.size > 1:
            # Short polled waits: the profiler streams them, so a monitor shows the rank as idle.
            done = self.comm.Ibarrier()
            while True:
                with self.profiler.phase("time_idle"):
                    if done.Test():
                        break
                    time.sleep(0.01)
        return True, None


def make_work_stealer(enabled: bool, comm, d: int, fd: int, profiler: Profiler):
    """Return a ``WorkStealer`` if ``enabled``, else a ``NoWorkStealing``."""
    if enabled:
        return WorkStealer(comm, d, fd, profiler)
    return NoWorkStealing(comm, profiler)
