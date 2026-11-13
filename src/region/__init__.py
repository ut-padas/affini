"""
Region mode (``ClusteringConfig(algorithm="region")``): an ART-style region tree on
AFINNI's batched refinement loop.

Every node of the tree is a convex polytope of the declared box domain: a root cell
(the whole box, a uniform-grid cell, or the Voronoi cell of a k-means centre) cut by its
ancestors' hyperplanes. Per batch of nodes, ``RegionPlan.process_batch``:

1. tops every node up to ``ceil(sample_multiplier * (d + 1))`` training points inside
   its polytope (``HitAndRunSampler``) and draws ``validation_samples`` fresh points;
   the B1 variant (``sampling="convex_combination"``) instead keeps AFINNI's
   convex-combination augmentation of rejected nodes;
2. fits the local model on the training points and screens it with the acceptance test;
3. deploys the screened models (``Q`` if stored, finalize, re-test) and checks them on
   the validation points;
4. splits every rejected node by a hyperplane (``split_search``: ``"gap"`` along the
   split strategy's direction, ``"bins"`` a threshold search over a bank of directions,
   ``"soft"`` that search polished by the consistent soft-oblique optimizer), cut so the
   stored predicate reproduces the children exactly; a node that cannot be split, or is
   at ``max_levels``, keeps its fitted model as a terminal leaf with a status.

Geometry lives in normalized coordinates ``u = (x - lower) / (upper - lower)`` of the
domain; every predicate is ``w^T u - z >= 0`` (right child) with ``|w| = 1``, evaluated
by ``plane_scores`` both in training and in ``route``. Children carry their ancestors'
half-spaces (``constraints``), their node key ``(root, path_code)`` and depth, so a
stolen node needs nothing from the rank that created it.

Random numbers come from a counter-based generator keyed by ``(seed, root, path_code,
purpose, draw index)``: a node's samples and random directions do not depend on the
batch it is processed in, the number of ranks or the steal schedule, so region-mode
results are the same for any batch size and rank count (except the B1 variant, whose
augmentation uses a per-batch stream, as in AFINNI mode). A node draws
each purpose once (attempt 0); a second draw would need an attempt index.

Package layout: ``constants`` (statuses, purposes), ``rng`` (counter-based streams), ``geometry``
(domain, roots, plane predicates), ``sampling`` (hit-and-run), ``split_search`` (batched rows and the
B4 objective), ``plan`` (``RegionPlan``) and ``routing`` (``RoutingTable``).
"""

from src.region.constants import (
    PURPOSE_TRAIN,
    PURPOSE_VALIDATION,
    TERMINAL_REASONS,
    VALIDATION_STATUSES,
)
from src.region.geometry import Domain, Roots, gap_cuts, plane_scores
from src.region.plan import RegionPlan
from src.region.rng import counter_uniform, counter_uniform_pairs, node_keys
from src.region.routing import RoutingTable
from src.region.sampling import HitAndRunSampler, RegionPoints
from src.region.split_search import _hard_sse, _sigmoid, _soft_objective, _SplitRows

__all__ = ["Domain", "HitAndRunSampler", "PURPOSE_TRAIN", "PURPOSE_VALIDATION", "RegionPlan", "RegionPoints", "Roots", "RoutingTable", "TERMINAL_REASONS", "VALIDATION_STATUSES", "_SplitRows", "_hard_sse", "_sigmoid", "_soft_objective", "counter_uniform", "counter_uniform_pairs", "gap_cuts", "node_keys", "plane_scores"]
