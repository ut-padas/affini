"""Enforce the layer order of README.md "Code layout": a module imports only from its own layer or lower ones."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

# Keep in sync with the table in README.md. ``src`` is the package root (``from src import xp``).
LAYERS = {
    "src": 0, "infra": 0, "numerics": 0, "kernels": 0, "core": 0,
    "config": 1, "components": 1, "neighbors": 1,
    "region": 2,
    "engine": 3,
    "surrogate": 4, "plotting": 4, "problems": 4,
}  # fmt: skip


def _part_of(module: str) -> str:
    """The top-level part of ``src`` a dotted module or ``src/`` path belongs to."""
    parts = module.split(".")
    if parts[0] != "src" or len(parts) == 1 or parts[1] in ("__init__", "__main__"):
        return "src"
    return parts[1]


def _imported_parts(tree: ast.Module) -> set[str]:
    """Parts of ``src`` imported anywhere in ``tree``, including imports inside functions."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {_part_of(a.name) for a in node.names if a.name.split(".")[0] == "src"}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "src":  # ``from src import config`` names a part; ``from src import xp`` the root
                found |= {a.name if a.name in LAYERS else "src" for a in node.names}
            elif node.module.split(".")[0] == "src":
                found.add(_part_of(node.module))
    return found


def test_every_part_of_src_has_a_layer():
    parts = {_part_of("src." + ".".join(p.relative_to(SRC).with_suffix("").parts)) for p in SRC.rglob("*.py")}
    assert parts <= LAYERS.keys(), f"add {sorted(parts - LAYERS.keys())} to LAYERS and README.md"


def test_no_module_imports_a_higher_layer():
    violations = []
    for path in sorted(SRC.rglob("*.py")):
        own = _part_of("src." + ".".join(path.relative_to(SRC).with_suffix("").parts))
        for part in _imported_parts(ast.parse(path.read_text(), filename=str(path))):
            if LAYERS[part] > LAYERS[own]:
                violations.append(f"{path.relative_to(SRC.parent)} ({own}) imports src.{part}")
    assert not violations, "\n".join(violations)
