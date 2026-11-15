"""Saving and loading trained surrogates: the file format version, file naming and JSON encoding of settings."""

from pathlib import Path
from typing import Any

import numpy as np

# Version of the file format written by AFINNISurrogate.save_model(); load_model()
# reads every version up to it.
# Version 2 adds the optional ART tree ("route:" arrays, "leaf:" statuses).
# Version 3 names the trust matrices "trust_matrices_src" (see TRUST_MATRIX_NAMES_BEFORE_V3).
MODEL_FORMAT_VERSION = 3

# Names under which format versions 1 and 2 stored the trust matrices, the ART setting that
# keeps them and their profiling phase, mapped to the names version 3 uses.
TRUST_MATRIX_NAMES_BEFORE_V3 = {
    "Qs_src": "trust_matrices_src",
    "store_Q": "store_trust_matrix",
    "time_good_Q": "time_good_trust_matrix",
}


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


def rename_keys_before_v3(
    version: int, arrays: dict[str, Any], meta: dict[str, Any], profiling_stats: Any
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """
    ``arrays``, ``meta`` and ``profiling_stats`` of a file of format ``version`` with the
    trust-matrix names of version 3; a file of version 3 or later is returned unchanged.

    The leaf array is renamed among the arrays, the ART setting in the saved
    configuration, and the profiling phase both as a key of the phase totals and as the
    ``"phase"`` value of timeline events.
    """
    if version >= 3:
        return arrays, meta, profiling_stats
    arrays = {(TRUST_MATRIX_NAMES_BEFORE_V3[k] if k == "Qs_src" else k): v for k, v in arrays.items()}
    meta = {**meta, "config": _renamed(meta["config"], "store_Q")}
    return arrays, meta, _renamed(profiling_stats, "time_good_Q")


def _renamed(obj: Any, old: str) -> Any:
    """``obj`` (parsed JSON) with every dict key and string equal to ``old`` replaced by its version-3 name."""
    new = TRUST_MATRIX_NAMES_BEFORE_V3[old]
    if isinstance(obj, dict):
        return {(new if k == old else k): _renamed(v, old) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_renamed(v, old) for v in obj]
    return new if obj == old else obj
