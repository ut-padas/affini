"""
The public surrogate: ``AFINNISurrogate`` trains with the adaptive clustering engine, stores the
leaves' models and predicts, saves and loads them; ``compute_error_metrics`` scores predictions.

Submodules: ``model`` (``AFINNISurrogate``), ``persistence`` (file format), ``metrics`` (errors).
"""

from src.surrogate.metrics import compute_error_metrics
from src.surrogate.model import AFINNISurrogate
from src.surrogate.persistence import MODEL_FORMAT_VERSION

__all__ = ["MODEL_FORMAT_VERSION", "AFINNISurrogate", "compute_error_metrics"]
