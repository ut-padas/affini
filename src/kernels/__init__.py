"""
CUDA sources of AFINNI's GPU kernels, one ``.cu`` file per area, compiled at runtime by
NVRTC through CuPy. CuPy caches each compiled binary on disk (``~/.cupy/kernel_cache``)
keyed by the source text, the options and the GPU architecture, so a process only
recompiles a file after it changes; there is no build step and no per-architecture binary.
The Python wrappers that launch a kernel document its contract; the ``.cu`` comments
document its implementation.
"""

from functools import cache
from pathlib import Path

_DIR = Path(__file__).parent


@cache
def source(name: str) -> str:
    """The text of ``src/kernels/<name>.cu``."""
    return (_DIR / f"{name}.cu").read_text()


@cache
def module(name: str, options: tuple[str, ...] = ()):
    """The compiled ``cupy.RawModule`` of ``<name>.cu`` (one per process and option set)."""
    import cupy as cp

    return cp.RawModule(code=source(name), options=options)
