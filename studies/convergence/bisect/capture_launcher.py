"""
Run a commit's own experiment launcher unchanged and save what it trained and measured.

Run from the root of an extracted commit, with that root (and, for commits that import `faiss`, the
stand-in's directory) on PYTHONPATH:

    python capture_launcher.py --out run.npz --launcher afinni_launcher.py -- \\
        --experiment wave --dims 2 --tols 0.005 --error_eval_mode affine_cov_hess

`AFINNISurrogate.train`/`predict` and `compute_error_metrics` are wrapped before the launcher is
imported (the launcher binds `compute_error_metrics` at import, so it gets the wrapper), so the run
means exactly what the same flags meant at that commit: its data generation, settings, candidate
count, inference rule and metric. Saved: the test points, predictions, out-of-trust-region mask
and ground truth the launcher scored, the leaf centres and trust matrices when the surrogate
exposes them, `N_c`, `N`, and the launcher's own metric dictionary. One run per process.
"""

import argparse
import inspect
import json
import runpy
import sys
import time

import numpy as np

own, rest = sys.argv[1:], []
if "--" in own:
    i = own.index("--")
    own, rest = own[:i], own[i + 1 :]
p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--launcher", default="afinni_launcher.py")
a = p.parse_args(own)

import src.surrogate as S  # noqa: E402

cap: dict = {}
orig_train, orig_predict, orig_metrics = S.AFINNISurrogate.train, S.AFINNISurrogate.predict, S.compute_error_metrics


def host(x):
    return np.asarray(x.get() if hasattr(x, "get") else x)


def train(self, *args, **kw):
    t = time.perf_counter()
    r = orig_train(self, *args, **kw)
    cap["train_s"] = time.perf_counter() - t
    return r


def predict(self, *args, **kw):
    bound = inspect.signature(orig_predict).bind(self, *args, **kw)
    bound.apply_defaults()
    r = orig_predict(self, *args, **kw)
    if "pred" in cap:  # later calls (plots) are not the scored test-set prediction
        return r
    q = bound.arguments.get("query_points", bound.arguments.get("X"))
    cap.update(Xt=host(q), pred=host(r[0]), out=host(r[1]).astype(bool), surrogate=self,
               predict_args={k: v for k, v in bound.arguments.items() if isinstance(v, (int, float, bool, str))})  # fmt: skip
    return r


def metrics(*args, **kw):
    bound = inspect.signature(orig_metrics).bind(*args, **kw)
    r = orig_metrics(*args, **kw)
    if "metrics" in cap:  # only the first scoring is the test-set one
        return r
    names = bound.arguments
    cap["gt"] = host(names["gt"])
    cap["metric_out"] = host(names.get("ug_mask", names.get("out_mask"))).astype(bool)
    cap["metrics"] = r[0] if isinstance(r, tuple) else r
    return r


S.AFINNISurrogate.train, S.AFINNISurrogate.predict, S.compute_error_metrics = train, predict, metrics
sys.argv = [a.launcher, *rest]
try:
    runpy.run_path(a.launcher, run_name="__main__")
except Exception as exc:  # old launchers' plotting after the metrics can fail on this layout
    if "metrics" not in cap:
        raise
    print(f"launcher raised after scoring (ignored): {type(exc).__name__}: {exc}")

s = cap["surrogate"]
extra = {}
for name, attr in (("X_src", "X_src"), ("Qs", "Qs_src"), ("Qs", "trust_matrices_src")):
    if getattr(s, attr, None) is not None:
        extra[name] = host(getattr(s, attr))
prof = getattr(s, "profiling_stats", None) or {}
n_c = getattr(s, "N_src", None) or (len(extra["X_src"]) if "X_src" in extra else -1)
n_eval = prof.get("total_f_evals", -1)
assert np.array_equal(cap["out"], cap["metric_out"].reshape(cap["out"].shape)), "launcher scored another mask"


def plain(o):
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


np.savez(a.out, Xt=cap["Xt"], pred=cap["pred"].reshape(len(cap["Xt"]), -1), out=cap["out"].reshape(-1),
         gt=cap["gt"].reshape(len(cap["Xt"]), -1), N_c=n_c, N=n_eval, train_s=cap.get("train_s", -1),
         metrics=json.dumps(plain(cap["metrics"])), predict_args=json.dumps(cap["predict_args"]), **extra)  # fmt: skip
print(f"RESULT {' '.join(rest)}: N_c {n_c} N {n_eval} p_o {100 * cap['out'].mean():.3f} "
      f"predict {cap['predict_args']} train {cap.get('train_s', -1):.1f}s")  # fmt: skip
