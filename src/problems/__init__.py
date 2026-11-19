"""
Test problems: benchmark functions (one module per family, with their derivatives or flow maps where they
have them), registered with their domains and test sets in ``FUNCTIONS`` (``registry``), and ODE systems
with their step maps in ``ode`` (``STEP_MAPS``).
"""

from src.problems.registry import (
    FUNCTIONS,
    GRID,
    N_TEST,
    VECTOR_FUNCTIONS,
    FunctionProblem,
)

__all__ = ["FUNCTIONS", "GRID", "N_TEST", "VECTOR_FUNCTIONS", "FunctionProblem"]
