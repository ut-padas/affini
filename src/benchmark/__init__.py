"""
The benchmark: every method (AFINNI, ART and the schemes) on two suites of problems, recorded in one store.

- ``functions``: the test functions of ``src.problems`` on their boxes (``python -m src.benchmark.functions``);
- ``step_maps``: the ODE step maps of ``src.problems.ode`` on visited states (``python -m src.benchmark.step_maps``);
- ``sweep``: both suites' grids, cheapest case first, under a time limit (``python -m src.benchmark.sweep``);
- ``store``: where runs, data and figures are kept; ``methods``: the function suite's methods;
  ``plots``: what both suites' figures share.
"""
