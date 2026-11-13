"""
Static checks of the CUDA sources against their Python launchers. They run without a GPU, so a
kernel that loses its caller, or a launcher that names a kernel no ``.cu`` file defines, fails on
every machine rather than only on a CUDA one.
"""

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
# Calls that name the kernel they launch with a string literal: module.get_function("k"),
# cp.RawKernel(source, "k") and the ART split search's _split_kernel("k", ...).
LAUNCHERS = {"get_function", "RawKernel", "_split_kernel"}


def _defined_kernels() -> dict[str, list[str]]:
    """Every ``__global__`` kernel name, with the ``.cu`` files that define it."""
    out: dict[str, list[str]] = {}
    for path in sorted((SRC / "kernels").glob("*.cu")):
        for name in re.findall(r"__global__\s+void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?(\w+)\s*\(", path.read_text()):
            out.setdefault(name, []).append(path.name)
    return out


def _launched_kernels() -> dict[str, list[str]]:
    """Every kernel name a launcher call passes as a string literal, with the calling modules."""
    out: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if func in LAUNCHERS:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.isidentifier():
                        out.setdefault(arg.value, []).append(str(path.relative_to(SRC.parent)))
    return out


def test_kernel_names_are_unique_across_cu_files():
    duplicates = {k: v for k, v in _defined_kernels().items() if len(v) > 1}
    assert not duplicates, duplicates


def test_every_kernel_has_a_python_launcher():
    unused = sorted(set(_defined_kernels()) - set(_launched_kernels()))
    assert not unused, f"kernels without a caller (delete them or launch them): {unused}"


def test_every_launched_kernel_is_defined():
    defined = _defined_kernels()
    missing = {k: v for k, v in _launched_kernels().items() if k not in defined}
    assert not missing, f"launched but defined in no src/kernels/*.cu: {missing}"
