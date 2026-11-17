"""
How far ART's hit-and-run chains move from their starts in high dimension, measured on the
library's own sampler (``HitAndRunSampler``, the Numba walk on the CPU and the ``hr_walk``
kernel on the GPU, which take identical steps up to rounding).

A node's new points are endpoints of chains of ``walk_steps`` steps started at the node's
inherited points. Those starts are already uniform in the node (they are its parent's uniform
points that fell inside it), so the endpoints are uniform too, whatever the chain length; what
a short chain loses is independence: an endpoint that has barely moved nearly duplicates its
start. The drift check (``mixing_z``) compares the chains' halfway and final means, which do
not differ when the starts are already uniform, so it cannot see this. The script reports, per
d and chain length tau, for nodes cut by ``--depth`` random planes through the cube:

- ``rho``: the correlation between start and endpoint of a random linear functional v^T u,
  averaged over 32 directions (1 for a chain that does not move, 0 for an independent point);
- ``disp``: the mean squared displacement over its value for two independent uniform points
  of the node (2 tr Cov), 1 when independent;
- ``flagged``: the share of nodes the drift check marks ``insufficient_sampling`` with the
  default doubling (``max_walk_steps = 4 tau``).

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off .venv/bin/python -m studies.high_dim.hit_and_run_mixing --dims 10,50,100

It calls the sampler's private ``_starts`` to know where each chain began, as a test would.
"""

import argparse
import json
import sys

import numpy as np


def node_points(d: int, depth: int, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """``depth`` random half-spaces through points near the cube's centre, as rows [a | z] (a^T u <= z), and n exact uniform points of the node."""
    rows = []
    for _ in range(depth):
        a = rng.standard_normal(d)
        a /= np.linalg.norm(a)
        rows.append(np.r_[a, a @ rng.uniform(0.4, 0.6, d)])
    C = np.array(rows).reshape(depth, d + 1)
    pts = []
    while sum(len(p) for p in pts) < n:  # rejection from the cube: exact, about 2^-depth accepted
        U = rng.uniform(0, 1, (4 * n, d))
        pts.append(U[np.all(C[:, d] >= U @ C[:, :d].T, axis=1)] if depth else U)
    return C, np.concatenate(pts)[:n]


def measure(d: int, tau: int, depth: int, n_nodes: int, n_per_node: int, seed: int) -> dict:
    from src import to_host, xp
    from src.art import Domain, HitAndRunSampler, NodePolytopes, Roots, node_keys
    from src.art.constants import PURPOSE_TRAIN
    from src.components.sampling import ClusterPoints
    from src.numerics import repeats

    rng = np.random.default_rng(seed + d)
    domain, roots = Domain(np.zeros(d), np.ones(d)), Roots.build("box", d, 1)
    nodes = [node_points(d, depth, n_per_node, rng) for _ in range(n_nodes)]
    X = xp.asarray(np.concatenate([p for _, p in nodes]))
    cons = xp.asarray(np.stack([c for c, _ in nodes]))
    K = n_nodes
    counts = xp.full(K, n_per_node, dtype=xp.int64)
    n_new = xp.full(K, n_per_node, dtype=xp.int64)
    root, depth_arr = xp.zeros(K, dtype=xp.int64), xp.full(K, depth, dtype=xp.int64)
    keys = node_keys(seed, root, xp.arange(K, dtype=xp.int64) + 2)

    def sample(max_steps):
        sampler = HitAndRunSampler(domain, roots, tau, max_steps)
        geo = NodePolytopes(root, depth_arr, keys, cons, PURPOSE_TRAIN)
        pts = ClusterPoints(X, counts, polytopes=geo)
        labels = repeats(xp.arange(K), n_new)
        draw = xp.arange(int(xp.sum(n_new)), dtype=xp.int64) - (xp.cumsum(n_new) - n_new)[labels]
        starts, _ = sampler._starts(pts, geo, labels, draw)
        end = sampler.sample(pts, n_new)
        return np.asarray(to_host(starts)), np.asarray(to_host(end)), np.asarray(to_host(geo.insufficient))

    U0, U1, _ = sample(tau)  # exactly tau steps: no doubling
    _, _, flagged = sample(4 * tau)
    V = np.random.default_rng(seed).standard_normal((d, 32))
    V /= np.linalg.norm(V, axis=0)
    rhos, disps = [], []
    for k in range(K):
        s, e = U0[k * n_per_node : (k + 1) * n_per_node], U1[k * n_per_node : (k + 1) * n_per_node]
        a, b = s @ V, e @ V
        rhos.append(np.mean([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(V.shape[1])]))
        node = nodes[k][1]
        disps.append(np.mean(np.sum((e - s) ** 2, axis=1)) / (2 * np.trace(np.cov(node.T))))
    return {"d": d, "tau": tau, "depth": depth, "rho": float(np.mean(rhos)), "disp": float(np.mean(disps)),
            "flagged": float(np.mean(flagged)), "nodes": K, "points_per_node": n_per_node}  # fmt: skip


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dims", default="10,20,50,100,200")
    parser.add_argument("--taus", default="20,80,320", help="chain lengths; 'dd' adds tau = d^2")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--points", type=int, default=400, help="points per node")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="JSON lines file")
    args = parser.parse_args(argv)
    rows = []
    for d in map(int, args.dims.split(",")):
        taus = [int(t) if t != "dd" else d * d for t in args.taus.split(",")]
        for tau in taus:
            r = measure(d, tau, args.depth, args.nodes, args.points, args.seed)
            rows.append(r)
            print(f"d={d:4d} tau={tau:6d}: rho={r['rho']:.3f} disp={r['disp']:.3f} flagged={r['flagged']:.2f}", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
