"""Saving and loading trained surrogates: the file format version, file naming and JSON encoding of settings."""

from pathlib import Path
from typing import Any

import numpy as np

# Version of the file format written by AFINNISurrogate.save_model(); load_model()
# reads every version up to it.
# Version 2 adds the optional ART tree ("route:" arrays, "leaf:" statuses).
MODEL_FORMAT_VERSION = 2


DEFAULT_MODEL_FILENAME = "afinni_surrogate.npz"


PARAM_PREFIX = "param:"


ROUTE_PREFIX = "route:"


LEAF_PREFIX = "leaf:"


def model_path(path: str | Path) -> Path:
    """``path`` as an ``.npz`` file path; a directory gets the default file name."""
    path = Path(path)
    if path.is_dir():
        return path / DEFAULT_MODEL_FILENAME
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def json_default(obj: Any) -> Any:
    """Make NumPy scalars/arrays and enums JSON serializable."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "value"):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
