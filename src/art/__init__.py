"""
ART, the adaptive regression tree (``algorithm="art"``), as a plan of the refinement engine
(``src.engine``): ART's oblique tree of polytopes, built level by level in batches.

Every node of the tree is a convex polytope of the declared box domain (its region
``Omega_C`` in the companion): a root cell (the whole box, a uniform-grid cell, or the Voronoi
cell of a k-means centre) cut by its ancestors' planes. Per batch of nodes,
``ARTPlan.process_batch``:

1. tops every node up to ``ceil(sample_multiplier * p)`` training points (``n = ceil(m p)`` in
   the companion, with ``p = d + 1`` coefficients of an affine leaf) inside its polytope (``HitAndRunSampler``) and draws ``validation_samples`` fresh points;
   with ``sampling="convex_combination"`` it instead keeps AFINNI's convex-combination
   augmentation of rejected nodes;
2. fits the local model on the training points and screens it with the acceptance test;
3. deploys the screened models (``Q`` if stored, finalize, re-test) and checks them on
   the validation points;
4. splits every rejected node by a plane (``split_search``: ``"gap"`` along the
   split strategy's direction, ``"bins"`` the bins search over a bank of directions,
   ``"soft"`` that search polished by the consistent soft-oblique optimizer), cut so the
   stored predicate reproduces the children exactly; a node that cannot be split, or is
   at ``max_levels``, keeps its fitted model as a terminal leaf with a status.

Geometry lives in normalized coordinates ``u = (x - lower) / (upper - lower)`` of the
domain; every predicate is ``a^T u - z >= 0`` (right child) for a plane with unit normal ``a``
and offset ``z`` (``(a, z)`` in the companion; ``normal`` and ``offset`` in ``RoutingTable``
and the split records, ``w`` and ``z`` in the split search's arrays, its kernels and the model
file), evaluated by ``plane_scores`` both in training and in ``route``. Children carry their
ancestors' half-spaces (``constraints``), their node key ``(root, path_code)`` (``pi`` in the
companion: the root is 1, the children of ``pi`` are ``2 pi`` on the left and ``2 pi + 1`` on
the right) and depth, so a stolen node needs nothing from the rank that created it.

Random numbers come from a counter-based generator keyed by ``(seed, root, path_code,
purpose, draw index)``: a node's samples and random directions do not depend on the
batch it is processed in, the number of ranks or the steal schedule, so ART
results are the same for any batch size and rank count (except with
``sampling="convex_combination"``, whose augmentation uses a per-batch stream, as in AFINNI). A node draws
each purpose once (attempt 0); a second draw would need an attempt index.

Package layout: ``constants`` (statuses, purposes), ``rng`` (counter-based streams), ``geometry``
(domain, roots, plane predicates), ``sampling`` (hit-and-run), ``split_search`` (batched rows and the
soft objective), ``plan`` (``ARTPlan``) and ``routing`` (``RoutingTable``).
"""

from src.art.constants import (
    PURPOSE_TRAIN,
    PURPOSE_VALIDATION,
    TERMINAL_REASONS,
    VALIDATION_STATUSES,
)
from src.art.geometry import Domain, Roots, gap_cuts, plane_scores
from src.art.plan import ARTPlan
from src.art.rng import counter_uniform, counter_uniform_pairs, node_keys
from src.art.routing import RoutingTable
from src.art.sampling import HitAndRunSampler, NodePolytopes
from src.art.split_search import SplitRows, hard_sse, sigmoid, soft_objective

__all__ = ["Domain", "HitAndRunSampler", "PURPOSE_TRAIN", "PURPOSE_VALIDATION", "ARTPlan", "NodePolytopes", "Roots", "RoutingTable", "TERMINAL_REASONS", "VALIDATION_STATUSES", "SplitRows", "hard_sse", "sigmoid", "soft_objective", "counter_uniform", "counter_uniform_pairs", "gap_cuts", "node_keys", "plane_scores"]
