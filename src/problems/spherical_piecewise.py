"""
ART's spherical piecewise-affine benchmark: one affine function inside a sphere, another
outside, so the target jumps across a curved interface.

The constants are ART's recorded TACC preset (``tasmanian/oracle_presets.json`` and
``examples/benchmark_functions.py`` in the ART repository): domain ``[-3, 3]^d``, sphere
at the origin with the radius that holds 50% of the box's volume, output offset 4.15,
and the inside branch including the sphere itself. Supported for ``d = 2..5``.
"""

import numpy as np
from array_api_compat import array_namespace

from src import Array

DOMAIN = (-3.0, 3.0)
INSIDE_W = np.array([-0.25, 0.35, -0.15, 0.20, -0.10])
OUTSIDE_W = np.array([0.40, -0.30, 0.20, -0.10, 0.05])
INSIDE_BETA, OUTSIDE_BETA, OFFSET = 5.0, 0.0, 4.15
RADII = {2: 2.394139996923, 3: 2.954149913692, 4: 3.418315145003, 5: 3.834703297290}


def spherical_piecewise(X: Array) -> Array:
    """(N, 1) values: ``offset + 5 + w_in^T x`` where ``|x| <= r_d``, else ``offset + w_out^T x``."""
    xp = array_namespace(X)
    d = X.shape[1]
    if d not in RADII:
        raise ValueError(f"spherical_piecewise supports d in {sorted(RADII)}; got {d}.")
    inside = xp.sum(X**2, axis=1, keepdims=True) <= RADII[d] ** 2
    f_in = INSIDE_BETA + X @ xp.asarray(INSIDE_W[:d])[:, None]
    f_out = OUTSIDE_BETA + X @ xp.asarray(OUTSIDE_W[:d])[:, None]
    return OFFSET + xp.where(inside, f_in, f_out)
