"""
What every scheme shares: its settings (``Settings``), the model protocol (``Model``), and saving and
loading a trained model (``save``, ``load``).

A scheme is a module with a frozen settings dataclass and ``build(f, lo, hi, atol, settings=None, *,
jacobian=None, stream_dir=None, **overrides) -> model``: ``f`` is a callable on host (N, d) arrays or an
``Oracle``, ``overrides`` replace single settings (``build(f, lo, hi, atol, seed=1)``). The model predicts
on host arrays, counts its training evaluations in ``evals``, and carries its ``TrainingReport`` in
``report``.

A model is a dataclass whose fields are arrays, numbers, strings, None, tuples, lists or dicts of those,
and nested dataclasses of ``src.schemes``. ``save`` writes the arrays to one ``.npz`` and the rest, with the
report and the format version, to a JSON header inside it; ``load`` rebuilds the dataclasses. Properties
computed from the fields (``functools.cached_property``) are not saved and are recomputed on demand.
"""

import dataclasses
import importlib
import itertools
import json
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
from scipy.stats import qmc

FORMAT_VERSION = 1
_HEADER = "__header__"

S = TypeVar("S", bound="Settings")


@dataclasses.dataclass(frozen=True)
class Settings:
    """Base of every scheme's settings: frozen, so a model's settings cannot change after training."""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def resolve(cls: type[S], settings: S | None, overrides: dict) -> S:
        """``settings`` (default: the defaults) with ``overrides`` replacing single fields."""
        base = settings if settings is not None else cls()
        if not isinstance(base, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(base).__name__}")
        return dataclasses.replace(base, **overrides) if overrides else base


@runtime_checkable
class Model(Protocol):
    """A trained scheme: predictions on host (N, d) arrays, its training evaluations and report."""

    evals: int

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def check_points(lo, hi, n: int, seed: int, corners: bool = False) -> np.ndarray:
    """n scrambled Sobol points of [lo, hi], and with ``corners`` the 2^d corners for d <= 8 (where steep features hide)."""
    d = len(lo)
    S = qmc.Sobol(d, scramble=True, seed=seed).random(n)
    if corners and d <= 8:
        S = np.vstack([S, np.array(list(itertools.product((0.0, 1.0), repeat=d)))])
    return lo + S * (hi - lo)


def save(model, path: str | Path) -> Path:
    """Write ``model`` (a dataclass of ``src.schemes``) and its report to ``path`` (``.npz``)."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    arrays: dict[str, np.ndarray] = {}
    header = {
        "format_version": FORMAT_VERSION,
        "model": _encode(model, arrays),
        "report": _encode(getattr(model, "report", None), arrays),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{_HEADER: np.array(json.dumps(header))}, **arrays)
    return path


def load(path: str | Path):
    """The model ``save`` wrote to ``path``, with its ``report``."""
    with np.load(Path(path), allow_pickle=False) as data:
        header = json.loads(str(data[_HEADER]))
        if header.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: format version {header.get('format_version')}, this code reads {FORMAT_VERSION}")
        arrays = {k: data[k] for k in data.files if k != _HEADER}
    model = _decode(header["model"], arrays)
    report = _decode(header["report"], arrays)
    if report is not None:
        model.report = report
    return model


def _encode(value: Any, arrays: dict) -> Any:
    """A JSON-able description of ``value``; its arrays go to ``arrays`` under fresh keys."""
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise TypeError("object arrays cannot be saved")
        key = f"a{len(arrays)}"
        arrays[key] = value
        return {"__array__": key}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(v, arrays) for v in value]}
    if isinstance(value, list):
        return [_encode(v, arrays) for v in value]
    if isinstance(value, dict):
        return {"__dict__": [[_encode(k, arrays), _encode(v, arrays)] for k, v in value.items()]}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        if not cls.__module__.startswith("src.schemes"):
            raise TypeError(f"cannot save {cls.__module__}.{cls.__qualname__}: not a dataclass of src.schemes")
        fields = {f.name: _encode(getattr(value, f.name), arrays) for f in dataclasses.fields(value)}
        return {"__dataclass__": f"{cls.__module__}:{cls.__qualname__}", "fields": fields}
    raise TypeError(f"cannot save a {type(value).__name__}")


def _decode(value: Any, arrays: dict) -> Any:
    if isinstance(value, list):
        return [_decode(v, arrays) for v in value]
    if not isinstance(value, dict):
        return value
    if "__array__" in value:
        return arrays[value["__array__"]]
    if "__tuple__" in value:
        return tuple(_decode(v, arrays) for v in value["__tuple__"])
    if "__dict__" in value:
        return {_hashable(_decode(k, arrays)): _decode(v, arrays) for k, v in value["__dict__"]}
    module, qualname = value["__dataclass__"].split(":")
    if not module.startswith("src.schemes"):
        raise ValueError(f"refusing to load {module}.{qualname}")
    cls = importlib.import_module(module)
    for part in qualname.split("."):
        cls = getattr(cls, part)
    fields = {k: _decode(v, arrays) for k, v in value["fields"].items()}
    init = {f.name for f in dataclasses.fields(cls) if f.init}
    obj = cls(**{k: v for k, v in fields.items() if k in init})
    for k, v in fields.items():  # fields set after construction (init=False)
        if k not in init:
            object.__setattr__(obj, k, v)
    return obj


def _hashable(k: Any) -> Any:
    return tuple(_hashable(v) for v in k) if isinstance(k, (list, tuple)) else k
