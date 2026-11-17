"""
The public surrogate: ``AdaptiveSurrogate`` trains with the adaptive clustering engine, stores the
leaves' models and predicts, saves and loads them; ``compute_error_metrics`` scores predictions.

Submodules: ``model`` (``AdaptiveSurrogate``), ``persistence`` (file format), ``metrics`` (errors).
"""

from src.surrogate.metrics import compute_error_metrics
from src.surrogate.model import AdaptiveSurrogate
from src.surrogate.persistence import MODEL_FORMAT_VERSION

__all__ = ["MODEL_FORMAT_VERSION", "AdaptiveSurrogate", "compute_error_metrics"]
