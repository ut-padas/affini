"""Rank-independent random-number construction."""

from __future__ import annotations

import hashlib
from collections.abc import Hashable

import numpy as np


def derived_seed(global_seed: int, *parts: Hashable) -> int:
    """Derive a stable 64-bit seed from semantic identifiers."""

    payload = "|".join([str(global_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def make_rng(global_seed: int, *parts: Hashable) -> np.random.Generator:
    return np.random.default_rng(derived_seed(global_seed, *parts))
