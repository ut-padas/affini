"""
ChebTT: a Chebyshev interpolant in tensor-train form, built by cross approximation (``train``), with its
model and compression (``model``); ChebTT-M is ``build_vector(..., rotate=True)``; ChebTT-A, which adds an
interface term for an oblique or curved jump or a sharp front, is ``adaptive``.
"""

from src.schemes.chebtt import adaptive
from src.schemes.chebtt.model import ChebTT, VectorTT, compress, round_tt
from src.schemes.chebtt.train import (
    LATTICE,
    ChebTTSettings,
    KeyStore,
    ObliqueEdgeError,
    RankLimitError,
    VectorSettings,
    axis_edges,
    build,
    build_vector,
    edge_at,
    locate_edge,
    maxvol,
)

__all__ = ["LATTICE", "ChebTT", "ChebTTSettings", "KeyStore", "ObliqueEdgeError", "RankLimitError", "VectorSettings", "VectorTT",
           "adaptive", "axis_edges", "build", "build_vector", "compress", "edge_at", "locate_edge", "maxvol", "round_tt"]  # fmt: skip
