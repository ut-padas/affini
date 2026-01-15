import warnings
from pathlib import Path

import cupy as cp

try:
    _HAS_GPU = cp.is_available()
except Exception:
    _HAS_GPU = False
    msg = "CuPy not available; GPU acceleration will be disabled."
    warnings.warn(msg)

_file_path = Path(__file__).parent.resolve()
PROJECT_DIR = _file_path.parent.resolve()
FIGURES_DIR = PROJECT_DIR / "figures"
