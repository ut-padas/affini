"""The splitmix64 finalizer, for counter-based random streams and deterministic per-point keys."""

from src import Array, xp

GOLDEN = 0x9E3779B97F4A7C15
MIX1, MIX2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def mix64(z: Array) -> Array:
    """
    The splitmix64 finalizer on uint64 arrays (wrapping arithmetic, same on NumPy and CuPy);
    ``kernels/hit_and_run.cu`` repeats it bit for bit.
    """
    z = (z ^ (z >> xp.uint64(30))) * xp.uint64(MIX1)
    z = (z ^ (z >> xp.uint64(27))) * xp.uint64(MIX2)
    return z ^ (z >> xp.uint64(31))
