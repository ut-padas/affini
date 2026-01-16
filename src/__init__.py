import warnings
from pathlib import Path

import cupy as cp
import numpy as np
from array_api_compat import array_namespace
from array_api_compat.common._typing import Array

try:
    _HAS_GPU = cp.is_available()
except Exception:
    _HAS_GPU = False
    msg = "CuPy not available; GPU acceleration will be disabled."
    warnings.warn(msg)

if not _HAS_GPU:
    _dummy = np.array([0])
else:
    _dummy = cp.array([0])
xp = array_namespace(_dummy)

_file_path = Path(__file__).parent.resolve()
PROJECT_DIR = _file_path.parent.resolve()
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"
