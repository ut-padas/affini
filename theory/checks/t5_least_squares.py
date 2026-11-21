"""
Proposition 5 (memo section T5): the stability of one quadratic least-squares leaf at ART's sampling rate.

ART fits a leaf with p = (d + 1)(d + 2) / 2 coefficients on N = 20 p + 1 points. For the space V of quadratics on
[-1, 1]^d with the uniform measure, in the orthonormal tensor Legendre basis:
  K(V) = sup_x sum_k phi_k(x)^2 = 1 + 8 d + 9 d (d - 1) / 2 (at a vertex) is the quantity in the
  Cohen-Davenport-Leviatan sufficient condition (FoCM 2013, Thm 1) K <= kappa N / ln N, kappa = (3 ln(3/2) - 1) / (2 + 2r);
  lambda_min of the empirical Gram matrix measures the discrete-continuous norm equivalence on V;
  the least-squares Lebesgue constant Lambda_N = sup_x sum_i |l_i(x)| gives the exact (deterministic) bound
  ||f - LS f||_inf <= (1 + Lambda_N) E_2(f), with E_2 the best sup-norm error by a quadratic.
Lambda_N's sup is estimated over random interior points and random vertices (a lower estimate of the sup).
"""

import math

import numpy as np
from common import write

DIMS = [2, 3, 5, 10, 20]
SEEDS = 5
R_CONF = 1.0  # probability 1 - 2 N^-r of ||G - I|| <= 1/2
KAPPA = (3 * math.log(1.5) - 1) / (2 + 2 * R_CONF)


def basis(X: np.ndarray) -> np.ndarray:
    """Orthonormal (uniform probability measure) Legendre features of total degree <= 2."""
    n, d = X.shape
    L1 = np.sqrt(3.0) * X
    L2 = np.sqrt(5.0) * (3 * X * X - 1) / 2
    cols = [np.ones((n, 1)), L1, L2]
    iu = np.triu_indices(d, 1)
    cols.append(L1[:, iu[0]] * L1[:, iu[1]])
    return np.concatenate(cols, axis=1)


def lebesgue(Xq: np.ndarray, Ginv_PhiT: np.ndarray) -> float:
    """max over the query points of sum_i |l_i(x)|, with l(x) = phi(x) @ Ginv_PhiT (in chunks of 500 points)."""
    return float(max(np.max(np.sum(np.abs(basis(Xq[i : i + 500]) @ Ginv_PhiT), axis=1)) for i in range(0, len(Xq), 500)))


def cdl_n(K: float) -> int:
    """Smallest N with K <= kappa N / ln N."""
    target = K / KAPPA
    N = int(target)
    while N / math.log(N) < target:
        N = int(N * 1.05) + 1
    return N


def main() -> None:
    rows = []
    for d in DIMS:
        p = (d + 1) * (d + 2) // 2
        N = 20 * p + 1
        K = 1 + 8 * d + 9 * d * (d - 1) / 2
        lam_min, leb_int, leb_vtx = [], [], []
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            Phi = basis(rng.uniform(-1, 1, (N, d)))
            Gram = Phi.T @ Phi / N
            lam_min.append(float(np.linalg.eigvalsh(Gram)[0]))
            Ginv_PhiT = np.linalg.solve(Phi.T @ Phi, Phi.T)  # (p, N): l(x) = phi(x) @ this

            leb_int.append(lebesgue(rng.uniform(-1, 1, (2000, d)), Ginv_PhiT))
            leb_vtx.append(lebesgue(rng.choice([-1.0, 1.0], (2000, d)), Ginv_PhiT))
        Kv = float(np.sum(basis(np.ones((1, d))) ** 2))
        rows.append({"d": d, "p": p, "N_art": N, "K": K, "K_check_at_vertex": Kv, "N_cdl_r1": cdl_n(K),
                     "lambda_min_median": float(np.median(lam_min)), "lebesgue_interior_median": float(np.median(leb_int)),
                     "lebesgue_vertex_median": float(np.median(leb_vtx)), "lebesgue_vertex_max": float(np.max(leb_vtx))})  # fmt: skip
    path = write("t5_least_squares", {"seeds": SEEDS, "kappa": KAPPA, "r": R_CONF, "rows": rows})
    for r in rows:
        print(f"d={r['d']:2d} p={r['p']:3d} N_art={r['N_art']:5d} K={r['K']:.0f} (check {r['K_check_at_vertex']:.0f}) N_cdl={r['N_cdl_r1']:,} "
              f"lambda_min={r['lambda_min_median']:.3f} Leb interior={r['lebesgue_interior_median']:.2f} vertex={r['lebesgue_vertex_median']:.2f} (max {r['lebesgue_vertex_max']:.2f})")  # fmt: skip
    print("wrote", path)


if __name__ == "__main__":
    main()
