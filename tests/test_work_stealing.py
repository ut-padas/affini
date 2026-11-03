"""Exercise work ownership and retirement under controlled message ordering."""

from collections import defaultdict

import numpy as np
import pytest

import src.adaptive_clustering as ac
from src.adaptive_clustering_utils import ClusteringConfig, SurrogateMode


class _Status:
    def Get_source(self):  # noqa: N802
        return 1


class _MPI:
    ANY_SOURCE = -1
    Status = _Status


class _Comm:
    def __init__(self, grant, has_g):
        self.grant = grant
        self.has_g = has_g
        self.states = [1001, 0]
        self.requested = False
        self.replied = False
        self.crossed_request = False
        self.polls = 0
        self.retiring = False
        self.sent = defaultdict(int)
        self.payloads = []

    def Send(self, buf, dest, tag):  # noqa: N802
        self.sent[tag] += 1

    def send(self, obj, dest, tag):
        self.sent[tag] += 1
        if tag == ac.TAG_STEAL_REQ:
            assert not self.requested
            self.requested = True
            self.crossed_request = True

    def Iprobe(self, source, tag, status=None):  # noqa: N802
        if tag == ac.TAG_STATE_BROADCAST:
            return bool(self.states) and (len(self.states) == 2 or self.requested)
        if tag == ac.TAG_STEAL_REQ:
            return self.crossed_request
        if tag == ac.TAG_STEAL_GRANT:
            self.polls += 1
            return self.grant and self.requested and not self.replied and self.polls > 1
        if tag == ac.TAG_STEAL_DENY:
            return not self.grant and self.requested and not self.replied
        return False

    def recv(self, source, tag):
        if tag == ac.TAG_STEAL_REQ:
            self.crossed_request = False
            return 1
        self.replied = True
        if tag == ac.TAG_STEAL_GRANT:
            return {"n_pts": 3, "n_clusters": 1, "has_g": self.has_g}
        assert tag == ac.TAG_STEAL_DENY
        return "IDLE"

    def Recv(self, buf, source, tag):  # noqa: N802
        if tag == ac.TAG_STATE_BROADCAST:
            buf[0] = self.states.pop(0)
        else:
            self.payloads.append(tag)
            if tag == ac.TAG_STEAL_GRANT + 1:
                buf[:] = np.arange(3)[:, None]
            elif tag == ac.TAG_STEAL_GRANT + 4:
                buf[:] = 3
            else:
                buf.fill(1)

    def Ibarrier(self):  # noqa: N802
        assert self.replied, "retired with a steal response outstanding"
        self.retiring = True
        # A late request must still be answered while retirement progresses.
        self.crossed_request = True
        comm = self

        class Retirement:
            def Test(self):  # noqa: N802
                return not comm.crossed_request

        return Retirement()

    def allgather(self, count):
        assert self.retiring and not self.crossed_request
        # A buffered state update invisible to Iprobe must also be drained.
        self.states.append(0)
        return [count, 3]


@pytest.mark.parametrize("grant,has_g", [(True, False), (True, True), (False, False)])
def test_late_steal_response_preserves_work_and_retires(monkeypatch, grant, has_g):
    monkeypatch.setattr(ac, "MPI", _MPI)
    monkeypatch.setattr(ac, "MIN_LOAD_THRESHOLD", 1000)
    comm = _Comm(grant, has_g)
    buffers, count, _, _, levels = ac._run_local_refinement(
        config=ClusteringConfig(SurrogateMode.NEAREST_NEIGHBOR, "Linf-local"),
        CLUSTER_PTS=np.empty((0, 1)),
        CLUSTER_F=np.empty((0, 1)),
        CLUSTER_G=None,
        CLUSTER_SIZES=np.empty(0, dtype=np.int64),
        CLUSTER_STARTS=np.empty(0, dtype=np.int64),
        F_max_global=np.ones(1),
        d=1,
        fd=1,
        f_fn=lambda x: np.ones((len(x), 1)),
        g_fn=None,
        oracle_fn=None,
        ATOL=0.05,
        RTOL=0.05,
        MIN_PTS_PER_CLUSTER=1,
        MAX_CLUSTERS_EST=10,
        SEED=0,
        Q_REG=1e-8,
        MAX_LEVELS=1000,
        BATCH_SIZE=100,
        USE_WORK_STEALING=True,
        LOG_MEM_STATS=False,
        PROFILE_TIMING=False,
        SAVE_ALL_GEN_CLUSTER_DATA=False,
        hessian=None,
        rank=0,
        size=2,
        profiling_stats={"level_stats": []},
        comm=comm,
    )
    assert not comm.states
    assert count == levels == int(grant)
    assert comm.sent[ac.TAG_STEAL_DENY] == 2
    if grant:
        assert buffers["counts"][0] == 3
        assert buffers["centroids"][0, 0] == 1
        assert len(comm.payloads) == 3 + int(has_g)
