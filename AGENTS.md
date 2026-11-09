# AGENTS.md

Guidance for coding agents working in this repository (AFINNI: adaptive clustering
surrogates with CPU/GPU backends, MPI and work stealing).

## Environment

- Python environment: `.venv/` (managed by `uv`; see `Makefile` for install targets).
- Backend selection: `AFINNI_DEVICE=cpu|gpu|auto`, `AFINNI_MPI=off|on|auto`.
  Always set them explicitly in tests and benchmarks, so a missing GPU/MPI cannot
  produce a misleading pass.
- MPI launcher: `.venv/bin/mpiexec` (bundled MPICH); MPI tests use `AFINNI_MPI_NP` ranks.

## Verification

Run from the repository root:

```bash
# Unit/integration tests, CPU and GPU (both must pass)
AFINNI_DEVICE=cpu AFINNI_MPI=off .venv/bin/python -m pytest -q --no-cov -m "not mpi"
AFINNI_DEVICE=gpu AFINNI_MPI=off .venv/bin/python -m pytest -q --no-cov -m "not mpi"

# Multi-rank tests (they launch their own mpiexec; never run pytest inside mpiexec)
AFINNI_DEVICE=cpu AFINNI_MPI=on AFINNI_MPI_NP=4 PATH=$PWD/.venv/bin:$PATH \
  .venv/bin/python -m pytest tests/mpi -o addopts='' -q

# Differential regression check against a reference run of another checkout
AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. .venv/bin/python scripts/regression_check.py \
  --save /tmp/new --compare /tmp/ref          # CPU refactors must be bit-identical
# on GPU add --rtol 1e-12 (decisions must still match exactly)
```

- A refactor must keep `scripts/regression_check.py` bit-identical on CPU. Deliberate
  numerical changes must be listed and justified separately. Compare against references
  saved from the current baseline (`--save` on the base commit), not older ones: the
  custom k-means/kNN deliberately changed results relative to the original PR head.
- GPU kernels accumulate with `atomicAdd`, so GPU results can differ at rounding level
  between runs. GPU tests compare with a tight tolerance (`tests/_helpers.py`:
  1e-12, or 1e-10 with `separate_runs=True` when comparing two trainings), never
  with a loose one that would hide an algorithmic change. Leaf counts and
  evaluation counts must still match exactly.
- Large runs can exhaust host RAM (queues grow with rejected clusters x
  2 * min_pts_per_cluster); watch RSS when trying big N / small tolerances.

## Code changes

- Before adding a helper, search for an existing one and reuse or extend it. Do not
  add small single-use functions, and do not duplicate logic that a module already
  owns (e.g. new components go through the registries in `src/registry.py`).
- When existing code gains new behaviour, rename the function, its docstring and
  related identifiers to describe what it now does, and update every caller, test
  and doc. Do not keep a name that only fits the old behaviour.
- Add or update tests with every behaviour change, so the reuse cannot quietly
  regress other callers.

Why: in a large codebase it is easy to miss the forest for the trees and add a
one-off function for each local problem. Reusing and extending shared modules,
with tests alongside, solves several use cases in fewer lines and makes the
codebase more robust and mature faster.

## Code documentation style

Apply to all Python code you create, and to existing code you edit once you
understand it. Do not go hunting for old code to fix when a request needs no edits.

Why not Google-style docstrings: type hints and descriptive names already state
the parameters, types and return values. Restating them adds length without
information. Docstrings should carry the context the code cannot express.

- Every module starts with a concise docstring stating its purpose. Add usage
  examples or parameter details only when the code does not make them obvious.
- Every class, function and method gets a concise docstring. Prefer one useful
  sentence; write more only when the contract needs it.
- Exception: skip the docstring when the name and a short body already say
  everything, e.g. a property returning a field, `__len__`, an override that only
  implements a documented abstract method, a test named after the behaviour it
  checks, or a trivial package `__init__`. Once an edit makes such code non-trivial,
  add the docstring in the same edit.
- Document what the signature and implementation do not say: contracts, invariants,
  design reasons, side effects, ownership, threading, coordinate conventions,
  performance limits, failure behaviour.
- Do not paraphrase the name, parameters, return annotation or visible code.
- Comment only where a local decision would otherwise be easy to misread or break,
  and explain why, not what the next line does.
- Keep documentation next to the behaviour it governs, and update it when that
  behaviour changes.

```python
# Bad
def trust_region_shapes(...):
    """Compute the trust region shapes."""

# Good
def trust_region_shapes(...):
    """Scale each Q so the farthest training point lies on dx^T Q dx = 1; inference treats > 1 as outside."""


# Bad
class WorkStealer:
    """Handles work stealing between ranks."""

# Good
class WorkStealer:
    """Answer every steal request with exactly one grant or denial; an unanswered request deadlocks retirement."""


# Bad
order = xp.lexsort(xp.stack([scores, labels]))  # sort the scores

# Good: explain the decision, not the call
order = xp.lexsort(xp.stack([scores, labels]))  # keeps clusters contiguous so children can be cut by position
```
