import logging
import warnings
from pathlib import Path

import cupy as cp
import numpy as np
from array_api_compat import array_namespace
from array_api_compat.common._typing import Array

logger = logging.getLogger(__name__)

try:
    IS_CUPY_BACKEND = cp.is_available()
except Exception:
    IS_CUPY_BACKEND = False
    msg = "CuPy not available; GPU acceleration will be disabled."
    warnings.warn(msg)

# Set up the array namespace based on backend availability
if IS_CUPY_BACKEND:
    _dummy = cp.array([0])
    logging.info("Array backend: CuPy")
else:
    _dummy = np.array([0])
    logging.info("Array backend: NumPy")

xp = array_namespace(_dummy)

_file_path = Path(__file__).parent.resolve()
PROJECT_DIR = _file_path.parent.resolve()
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"
