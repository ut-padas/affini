import time

import cupy as cp

DTYPE = cp.float64


# ---------------- user batch ----------------
# shapes: [(n1,d),...,(nm,d)] with varying n_i and fixed d
def make_var_mats(sizes, d, dtype=DTYPE, seed=0):
    rs = cp.random.RandomState(seed)
    return [rs.standard_normal((n, d), dtype=dtype) for n in sizes]


def median_idx(w):  # w: (d,) on device
    # returns median threshold index (argpartition)
    k = w.size // 2
    # copy to host might be ok, but keep on device:
    thr = cp.partition(w, k)[k]
    return thr


# ------------- baseline: sequential -------------
def run_seq(Ms, ell=8):
    rs = cp.random.RandomState(1234)
    d = Ms[0].shape[1]
    Q1s = []

    t0 = time.perf_counter()
    for M in Ms:
        Omega = rs.standard_normal((d, ell), dtype=M.dtype)
        Y = cp.matmul(M, Omega)
        Q, _ = cp.linalg.qr(Y, mode="reduced")
        Yp = cp.matmul(M, cp.matmul(cp.transpose(M), Q))
        Q2, _ = cp.linalg.qr(Yp, mode="reduced")
        q1 = Q2[:, 0]
        w = cp.matmul(q1[cp.newaxis, :], M).ravel()
        thr = median_idx(w)
        # labels not needed for timing; just force compute:
        _ = cp.asarray(w <= thr, dtype=cp.int8).sum()
        Q1s.append(q1)
    cp.cuda.runtime.deviceSynchronize()
    return time.perf_counter() - t0, Q1s


# ------------- multi-stream pipeline -------------
def run_streams(Ms, ell=8, n_streams=4):
    rs = cp.random.RandomState(1234)
    d = Ms[0].shape[1]
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]
    events_done = []
    Q1s = [None] * len(Ms)

    t0 = time.perf_counter()
    for i, M in enumerate(Ms):
        s = streams[i % n_streams]
        with s.use():
            Omega = rs.standard_normal((d, ell), dtype=M.dtype)
            Y = cp.matmul(M, Omega)
            Q, _ = cp.linalg.qr(Y, mode="reduced")
            Yp = cp.matmul(M, cp.matmul(cp.transpose(M), Q))
            Q2, _ = cp.linalg.qr(Yp, mode="reduced")
            q1 = Q2[:, 0]
            w = cp.matmul(q1[cp.newaxis, :], M).ravel()
            thr = median_idx(w)
            _ = cp.asarray(w <= thr, dtype=cp.int8).sum()
            Q1s[i] = q1
            ev = cp.cuda.Event()
            ev.record(s)
            events_done.append(ev)

    # join
    for ev in events_done:
        ev.synchronize()
    cp.cuda.runtime.deviceSynchronize()
    return time.perf_counter() - t0, Q1s


# ------------- demo -------------
if __name__ == "__main__":
    cp.cuda.runtime.setDevice(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    n_sm = props["multiProcessorCount"]
    print(f"GPU: {props['name'].decode()} with {n_sm} SMs")

    d = 5
    ell = 8
    # build a realistic mixed batch
    sizes = [120000, 9000, 15000, 7000, 11000, 8000, 13000, 9500] * 1
    print(f"Total matrices: {len(sizes)}")
    Ms = make_var_mats(sizes, d)

    t_seq, _ = run_seq(Ms, ell)
    t_str, _ = run_streams(Ms, ell, n_streams=2)

    gops = (
        sum(n * d * ell for n in sizes) * 2 / 1e9
    )  # rough GEMM tall*skinny flop count
    print(f"matrices={len(Ms)}, d={d}, ell={ell}")
    print(f"Sequential : {t_seq:7.3f} s   (~{gops / t_seq:6.1f} GFLOP/s eqv.)")
    print(f"8-streams  : {t_str:7.3f} s   (~{gops / t_str:6.1f} GFLOP/s eqv.)")
    print(f"Speedup    : {t_seq / t_str:5.2f}×")
