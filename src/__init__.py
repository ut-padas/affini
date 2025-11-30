import warnings

import cupy as cp

try:
    _HAS_GPU = cp.is_available()
except Exception:
    _HAS_GPU = False
    msg = "CuPy not available; GPU acceleration will be disabled."
    warnings.warn(msg)
