"""Counter-based random numbers: streams keyed by node, identical on NumPy and CuPy."""

import math

from src import IS_CUPY_BACKEND, Array, kernels, xp

# --- Counter-based random numbers --------------------------------------------------

GOLDEN = 0x9E3779B97F4A7C15
MIX1, MIX2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def _mix64(z: Array) -> Array:
    """The splitmix64 finalizer on uint64 arrays (wrapping arithmetic, same on NumPy and CuPy)."""
    z = (z ^ (z >> xp.uint64(30))) * xp.uint64(MIX1)
    z = (z ^ (z >> xp.uint64(27))) * xp.uint64(MIX2)
    return z ^ (z >> xp.uint64(31))


def node_keys(seed: int, root: Array, locational_code: Array) -> Array:
    """(K,) uint64 stream keys, one per node; children's keys are unrelated to their parent's."""
    base = xp.uint64((int(seed) * GOLDEN + 1) % 2**64)
    k = _mix64(base + xp.asarray(root).astype(xp.uint64) * xp.uint64(GOLDEN))
    return _mix64(k ^ xp.asarray(locational_code).astype(xp.uint64))


def counter_uniform(keys: Array, purpose: int, counter: Array) -> Array:
    """Uniform draws in the open interval (0, 1): one per ``(keys[i], purpose, counter[i])``, broadcast."""
    tag = xp.uint64((purpose << 56) + GOLDEN % 2**56)
    z = _mix64(keys ^ _mix64(xp.asarray(counter).astype(xp.uint64) + tag))
    return ((z >> xp.uint64(11)).astype(xp.float64) + 0.5) * 2.0**-53


def counter_uniform_pairs(keys: Array, purpose: int, counter: Array) -> Array:
    """
    Two uniforms in (0, 1) per ``(keys[i], purpose, counter[i])``, from the high and low 32
    bits of one hash: shape ``counter.shape + (2,)``. Half the hashing of ``counter_uniform``,
    at 2^-32 resolution, which hit-and-run directions and chord positions do not resolve.
    """
    tag = xp.uint64((purpose << 56) + GOLDEN % 2**56)
    if IS_CUPY_BACKEND:  # one fused pass; integer hashing and exact conversions, so the same bits
        keys, counter = (xp.ascontiguousarray(a) for a in xp.broadcast_arrays(
            xp.asarray(keys).astype(xp.uint64), xp.asarray(counter).astype(xp.uint64)))  # fmt: skip
        out = xp.empty(counter.shape + (2,))
        kernels.module("hit_and_run").get_function("uniform_pairs")(
            ((counter.size + 255) // 256,), (256,), (keys, counter, tag, xp.int64(counter.size), out))
        return out
    z = _mix64(keys ^ _mix64(xp.asarray(counter).astype(xp.uint64) + tag))
    hi = (z >> xp.uint64(32)).astype(xp.float64)
    lo = (z & xp.uint64(0xFFFFFFFF)).astype(xp.float64)
    return (xp.stack([hi, lo], axis=-1) + 0.5) * 2.0**-32


def box_muller(u1: Array, u2: Array) -> Array:
    """Standard normals from two arrays of open-interval uniforms (Box-Muller)."""
    return xp.sqrt(-2.0 * xp.log(u1)) * xp.cos(2.0 * math.pi * u2)
