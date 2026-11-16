"""
One d=2 Table 1 case with the pre-modularity API (`src.surrogate.AFINNISurrogate.train(X_train=...,
TOL=..., ...)`, commits up to the merge 935f7ae), with the settings of that era's launcher.

Run from the root of an extracted commit (PYTHONPATH=<that root>:<faiss shim dir>):

    python era_old.py --problem wave --tol 0.005 --mode affine_cov_hess --out run.npz

Data are drawn exactly as the launcher does (NumPy seed 0, X_train then X_test, both uniform), so
the random stream at the start of training is the launcher's; the script checks them against
`--ref-data` (saved once) so every commit sees the same points. It saves the leaf model and the
predictions on the test points, so `score.py` can score every commit with one metric.
"""

import argparse
import hashlib
import inspect
import time
from functools import partial

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--problem", choices=["wave", "rosenbrock", "gaussian"], required=True)
p.add_argument("--tol", type=float, required=True)
p.add_argument("--mode", default="affine_cov_hess")
p.add_argument("--coarse-factor", type=int, default=10)
p.add_argument("--metric", default="Linf-local", help="ERROR_METRIC: Linf-local (the launcher's), Linf-global or L2")
p.add_argument("--n-test", type=int, default=None, help="default: N_train // 2 (the paper-era launcher)")
p.add_argument("--ref-data", default=None)
p.add_argument("--out", required=True)
a = p.parse_args()

from src import xp  # noqa: E402
from src.adaptive_clustering_utils import SurrogateMode  # noqa: E402
from src.surrogate import AFINNISurrogate  # noqa: E402

d = 2
if a.problem == "wave":
    from src.problems.wave import wave as f, wave_grad as g

    dom = (-5.0, 5.0)
elif a.problem == "rosenbrock":
    from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

    f, g, dom = partial(rosenbrock, a=100.0), partial(rosenbrock_grad, a=100.0), (-2.0, 3.0)
else:
    from src.problems.anisotropic_gaussian import aniso_gaussian as f, aniso_gaussian_grad as g

    dom = (-3.0, 3.0)

n = int(50000 * 1.3**d)
nt = a.n_test or n // 2
xp.random.seed(0)
X = xp.random.uniform(dom[0], dom[1], size=(n, d))
Xt = xp.random.uniform(dom[0], dom[1], size=(nt, d))
digest = hashlib.sha256(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(Xt).tobytes()).hexdigest()[:16]
if a.ref_data:
    ref = np.load(a.ref_data)
    assert np.array_equal(ref["X"], X) and np.array_equal(ref["Xt"][: len(Xt)], Xt), "data differ"

kw = dict(X_train=X, f_fn=f, g_fn=g, mode=SurrogateMode(a.mode), TOL=a.tol, ERROR_METRIC=a.metric,
          INITIAL_CLUSTERING_METHOD="kmeans", NUM_INITIAL_CLUSTERS=int(25 * 1.3**d),
          MIN_PTS_PER_CLUSTER=int(100 * 1.3**d), USE_WORK_STEALING=True, MAX_LEVELS=1000,
          MAX_CLUSTERS_EST=100_000, STORE_AFFINE_FIT=True, Q_REG=1e-6, USE_V2=True)  # fmt: skip
params = inspect.signature(AFINNISurrogate.train).parameters
dropped = sorted(k for k in kw if k not in params)
kw = {k: v for k, v in kw.items() if k in params}
s = AFINNISurrogate()
t = time.perf_counter()
s.train(**kw)
t_train = time.perf_counter() - t
pred, out = s.predict(query_points=Xt, k_nn=1, strict_trust_region=False, coarse_factor=a.coarse_factor)
gt = f(Xt)
np.savez(a.out, X_src=np.asarray(s.X_src), Qs=np.asarray(s.Qs_src), A=np.asarray(s.A_fits_src),
         b=np.asarray(s.b_fits_src), pred=np.asarray(pred).reshape(len(Xt), -1), out=np.asarray(out),
         gt=np.asarray(gt).reshape(len(Xt), -1), N=s.profiling_stats.get("total_f_evals", -1),
         N_c=s.N_src, train_s=t_train, digest=digest, dropped=",".join(dropped))  # fmt: skip
print(f"RESULT era_old {a.problem} tol {a.tol} {a.mode}: N_c {s.N_src} N {s.profiling_stats.get('total_f_evals')} "
      f"p_o {100 * float(np.mean(np.asarray(out))):.3f} data {digest} dropped {dropped} ({t_train:.1f}s)")  # fmt: skip
