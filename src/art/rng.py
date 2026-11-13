"""Counter-based random numbers: streams keyed by node, identical on NumPy and CuPy."""

import math

from src import IS_CUPY_BACKEND, Array, xp

# --- Counter-based random numbers --------------------------------------------------

GOLDEN = 0x9E3779B97F4A7C15
MIX1, MIX2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def _mix64(z: Array) -> Array:
    """The splitmix64 finalizer on uint64 arrays (wrapping arithmetic, same on NumPy and CuPy)."""
    z = (z ^ (z >> xp.uint64(30))) * xp.uint64(MIX1)
    z = (z ^ (z >> xp.uint64(27))) * xp.uint64(MIX2)
    return z ^ (z >> xp.uint64(31))


def node_keys(seed: int, root: Array, path_code: Array) -> Array:
    """(K,) uint64 stream keys, one per node; children's keys are unrelated to their parent's."""
    base = xp.uint64((int(seed) * GOLDEN + 1) % 2**64)
    k = _mix64(base + xp.asarray(root).astype(xp.uint64) * xp.uint64(GOLDEN))
    return _mix64(k ^ xp.asarray(path_code).astype(xp.uint64))


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
        keys, counter = xp.broadcast_arrays(xp.asarray(keys).astype(xp.uint64), xp.asarray(counter).astype(xp.uint64))
        out = xp.empty(counter.shape + (2,))
        _uniform_pairs_kernel()(keys, counter, tag, out)
        return out
    z = _mix64(keys ^ _mix64(xp.asarray(counter).astype(xp.uint64) + tag))
    hi = (z >> xp.uint64(32)).astype(xp.float64)
    lo = (z & xp.uint64(0xFFFFFFFF)).astype(xp.float64)
    return (xp.stack([hi, lo], axis=-1) + 0.5) * 2.0**-32


_UNIFORM_PAIRS = None


def _uniform_pairs_kernel():
    """``counter_uniform_pairs`` as one CuPy elementwise kernel (compiled on first use)."""
    global _UNIFORM_PAIRS
    if _UNIFORM_PAIRS is None:
        import cupy as cp

        mix = f"z = (z ^ (z >> 30)) * 0x{MIX1:X}ULL; z = (z ^ (z >> 27)) * 0x{MIX2:X}ULL; z = z ^ (z >> 31);"
        _UNIFORM_PAIRS = cp.ElementwiseKernel(
            "uint64 key, uint64 ctr, uint64 tag", "raw float64 out",
            f"unsigned long long z = ctr + tag; {mix} z = key ^ z; {mix}"
            "out[2 * i] = ((double)(z >> 32) + 0.5) * 2.3283064365386963e-10;"
            "out[2 * i + 1] = ((double)(z & 0xFFFFFFFFULL) + 0.5) * 2.3283064365386963e-10;",
            "afinni_uniform_pairs",
        )
    return _UNIFORM_PAIRS


def box_muller(u1: Array, u2: Array) -> Array:
    """Standard normals from two arrays of open-interval uniforms (Box-Muller)."""
    return xp.sqrt(-2.0 * xp.log(u1)) * xp.cos(2.0 * math.pi * u2)
