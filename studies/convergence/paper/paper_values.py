"""
Numbers the AFINNI manuscript (review_artifacts/26acm_afinni.zip) reports, transcribed from its TeX
sources, with the location of each.

`collect_paper.py` matches launcher results against these records. A record's keys:
``ref`` (table/figure and row), ``loc`` (TeX file and label), ``group`` (the run group of
`cases.py`), ``problem`` (the launcher's result directory), ``dim``, ``tol`` (``--tols`` value:
atol = rtol, or atol only, scaled by max|f|, for Lorenz-96), and the reported quantities:
``N`` (oracle evaluations), ``N_c`` (leaf clusters), ``p_o`` (% of test points outside every trust
region), ``e_in`` / ``e`` (relative L2 error inside the trust regions / over all test points),
``e_nh`` (``e`` of the run without Hessian-informed splits), ``p_in_good`` (% of inside points
within the pointwise tolerance). Values are the manuscript's, typos included; inconsistencies
between its tables are listed in PAPER_NOTES.
"""

QUAD, GAUSS, ROSEN, LORENZ, WAVE = "anisotropic_quadratic_2", "anisotropic_gaussian", "rosenbrock", "lorenz96_dt_0.1", "wave"

# Tables 1a/1b (convergence.tex, tab:combined_convergence_A / _B): row, problem, d, rtol, N, N_c, p_o, e_in, e, e_nh
_CONVERGENCE = [
    (1, QUAD, 5, 0.1, 2.32e5, 390, 0.03, 2.17e-2, 2.18e-2, 7.43e-2),
    (2, QUAD, 5, 0.05, 4.93e5, 964, 0.42, 1.28e-2, 1.32e-2, 6.40e-1),
    (3, QUAD, 5, 0.02, 2.91e6, 5772, 6.15, 6.68e-3, 9.36e-3, 4.24e-2),
    (4, QUAD, 5, 0.01, 1.43e7, 28262, 16.9, 4.73e-3, 8.65e-3, 2.57e-2),
    (5, QUAD, 5, 0.005, 6.37e7, 1.25e5, 33.1, 3.22e-3, 7.55e-3, 1.69e-2),
    (6, QUAD, 10, 0.1, 6.50e6, 3229, 0.43, 2.95e-2, 3.02e-2, 1.42e-1),
    (7, QUAD, 10, 0.05, 1.90e7, 9422, 3.87, 2.06e-2, 2.85e-2, 1.34e-1),
    (8, QUAD, 10, 0.02, 1.51e8, 74969, 26.2, 1.53e-2, 3.92e-2, 1.14e-1),
    (9, QUAD, 10, 0.01, 9.85e8, 4.87e5, 58.9, 1.13e-2, 4.40e-2, 9.00e-2),
    (10, GAUSS, 2, 0.1, 8.46e4, 48, 0.02, 1.54e-2, 1.54e-2, 2.22e-2),
    (11, GAUSS, 2, 0.05, 8.49e4, 65, 0.02, 7.63e-3, 7.63e-3, 1.30e-2),
    (12, GAUSS, 2, 0.02, 8.67e4, 104, 0.04, 3.76e-3, 3.78e-3, 6.93e-3),
    (13, GAUSS, 2, 0.01, 9.92e4, 176, 0.15, 2.07e-3, 2.18e-3, 3.94e-3),
    (14, GAUSS, 2, 0.005, 1.40e5, 359, 0.21, 1.02e-3, 1.06e-3, 1.77e-3),
    (15, GAUSS, 5, 0.1, 2.28e5, 173, 0.05, 1.86e-3, 1.89e-3, 9.03e-3),
    (16, GAUSS, 5, 0.05, 5.64e5, 810, 1.1, 2.56e-4, 2.48e-2, 1.38e-2),
    (17, GAUSS, 5, 0.02, 1.64e6, 2847, 1.7, 1.73e-4, 1.16e-2, 1.73e-2),
    (18, GAUSS, 5, 0.01, 7.02e6, 13010, 2.6, 7.71e-5, 2.06e-2, 9.20e-3),
    (19, GAUSS, 5, 0.005, 1.73e7, 32471, 3.0, 2.88e-5, 2.86e-2, 6.98e-3),
    (20, ROSEN, 2, 0.1, 1.39e5, 344, 1.59, 3.40e-2, 3.40e-2, 4.21e-2),
    (21, ROSEN, 2, 0.05, 2.27e5, 746, 1.48, 1.58e-2, 1.58e-2, 4.06e-2),
    (22, ROSEN, 2, 0.02, 5.23e5, 2007, 1.45, 5.10e-3, 5.12e-3, 3.46e-2),
    (23, ROSEN, 2, 0.01, 1.08e6, 4232, 1.84, 2.32e-3, 2.34e-3, 1.11e-2),
    (24, ROSEN, 2, 0.005, 2.15e6, 8446, 2.36, 1.18e-3, 1.20e-3, 6.26e-3),
    (25, ROSEN, 5, 0.1, 1.42e7, 27916, 14.2, 4.21e-2, 5.15e-2, 7.38e-2),
    (26, ROSEN, 5, 0.05, 5.24e7, 1.03e5, 31.1, 2.60e-2, 4.02e-2, 5.62e-2),
    (27, ROSEN, 5, 0.02, 2.39e8, 4.69e5, 57.3, 1.44e-2, 3.29e-2, 4.23e-2),
    (28, ROSEN, 5, 0.01, 7.03e8, 1.38e6, 73.8, 8.80e-3, 2.89e-2, 3.53e-2),
    (29, ROSEN, 5, 0.005, 2.03e9, 3.97e6, 84.9, 5.13e-3, 2.56e-2, 3.03e-2),
    (30, LORENZ, 5, 0.1, 1.86e5, 139, 0.00, 5.09e-2, 5.09e-2, 5.50e-2),
    (31, LORENZ, 5, 0.05, 2.03e5, 355, 0.01, 2.62e-2, 2.62e-2, 4.07e-2),
    (32, LORENZ, 5, 0.02, 6.88e5, 1327, 0.03, 1.35e-2, 1.35e-2, 2.10e-2),
    (33, LORENZ, 5, 0.01, 2.23e6, 4285, 0.21, 8.88e-3, 8.93e-3, 1.34e-2),
    (34, LORENZ, 5, 0.005, 7.55e6, 14473, 2.04, 6.25e-3, 6.49e-3, 9.46e-3),
    (35, WAVE, 2, 0.1, 8.53e4, 162, 0.02, 1.95e-2, 1.95e-2, 4.91e-2),
    (36, WAVE, 2, 0.05, 9.00e4, 240, 0.04, 8.81e-3, 8.82e-2, 3.39e-2),
    (37, WAVE, 2, 0.02, 1.15e5, 393, 0.05, 4.21e-3, 4.22e-3, 2.13e-2),
    (38, WAVE, 2, 0.01, 1.77e5, 678, 0.10, 2.09e-3, 2.09e-3, 1.38e-2),
    (39, WAVE, 2, 0.005, 3.03e5, 1177, 0.18, 1.05e-3, 1.05e-3, 8.26e-3),
    (40, WAVE, 5, 0.1, 5.76e6, 15448, 2.49, 4.08e-2, 4.31e-2, 3.50e-1),
    (41, WAVE, 5, 0.05, 1.14e7, 30432, 6.85, 2.55e-2, 3.12e-2, 2.25e-1),
    (42, WAVE, 5, 0.02, 3.02e7, 81035, 22.4, 1.23e-2, 3.59e-2, 2.15e-1),
    (43, WAVE, 5, 0.01, 6.40e7, 1.72e5, 40.3, 6.49e-3, 5.20e-2, 1.98e-1),
    (44, WAVE, 5, 0.005, 1.37e8, 3.68e5, 57.8, 3.55e-3, 7.01e-2, 1.84e-1),
]

# Table 2 (convergence.tex, tab:lorenz_dim_scaling), rtol = 0.1: row, d, N, N_c, p_o, e_in, e
_LORENZ_DIMS = [
    (1, 5, 1.86e5, 117, 0.02, 5.45e-2, 5.45e-2),
    (2, 10, 2.97e6, 1400, 0.04, 4.91e-2, 4.90e-2),
    (3, 20, 4.55e7, 1801, 0.3, 5.42e-2, 5.38e-2),
    (4, 40, 8.40e7, 1775, 0.3, 6.12e-2, 6.09e-2),
    (5, 80, 1.19e8, 2335, 0.5, 7.34e-2, 7.31e-2),
]

# Table 3 (convergence.tex, tab:rosenbrock_weak_scaling), d = 5, atol only: atol, N_0, N_c, p_out, e_in, e, p_in_good
_ROSEN_WEAK = [
    (579.8, 50_000, 7443, 13.7, 4.49e-2, 6.05e-2, 96.7),
    (289.9, 282_842, 42125, 20.0, 2.26e-2, 3.49e-2, 96.8),
    (145.0, 1_600_889, 225786, 23.5, 1.16e-2, 1.98e-2, 96.4),
    (72.5, 9_061_035, 1121571, 25.0, 6.06e-3, 1.11e-2, 96.0),
]

# Figure 8 table (convergence.tex, fig:exact_grad_vs_approx_grad_comparison), wave d = 5:
# rtol, (LSQ gradient: N_c, p_o, e_in, e), (exact gradient: N_c, p_o, e_in, e)
_WAVE_EXACT = [
    (0.1, (15448, 2.48, 4.08e-2, 4.31e-2), (10351, 2.05, 4.28e-2, 4.38e-2)),
    (0.05, (30342, 6.85, 2.55e-2, 3.12e-2), (14721, 3.75, 2.18e-2, 2.36e-2)),
    (0.02, (81035, 22.4, 1.27e-2, 3.59e-2), (22710, 7.08, 9.13e-3, 1.29e-2)),
    (0.01, (171563, 40.2, 6.49e-3, 5.20e-2), (31120, 11.3, 4.75e-3, 1.47e-2)),
    (0.005, (368482, 57.8, 3.55e-3, 7.01e-2), (42626, 16.9, 2.75e-3, 1.60e-2)),
]

# Table 4 (comparison.tex, tab:other_surrogates): the AFINNI rows only; the other methods
# (Forest, GPR, Kernel, NN, SG) need libraries this repository does not ship.
_COMPARISON = [
    (1, ROSEN, 2, 0.05, 7.5e2, 1.6e-2),
    (9, ROSEN, 5, 0.05, 1.0e5, 4.0e-2),
    (18, LORENZ, 5, 0.05, 7.8e2, 1.9e-2),
]

# Section "Scaling with tolerance" (scalability.tex, fig:runtime_scalability_vs_rtol and
# fig:predict_scalability_vs_rtol): Rosenbrock d = 5 on one GH200; speed, informational only.
SCALABILITY_TEXT = {
    "loc": "scalability.tex, Sec. Scaling with Tolerance",
    "train_time_s": {1.0: 0.61, 0.005: 37.51},
    "N": {1.0: 1.9e5, 0.005: 1.0e9},  # "over 1E9" at 5E-3
    "N_c": {1.0: 1e2, 0.005: 2.7e6},
    "predict_time_s_46k_queries": {1.0: 0.01, 0.005: 0.41},
}

PAPER_NOTES = [
    "Table 1b row 31 (Lorenz d=5, rtol 0.05: N_c 355, e 2.62E-2) and Table 4 row 18 (AFINNI on Lorenz d=5: "
    "N_c 7.8E2, e 1.9E-2) disagree; Table 4 matches the commented-out older Lorenz rows (782, 1.87E-2).",
    "Lorenz d=5 at rtol 0.1 appears twice with the same settings: Table 1b row 30 (N_c 139, e_in 5.09E-2) and "
    "Table 2 row 1 (N_c 117, e_in 5.45E-2): an in-paper estimate of run-to-run spread (about 17% in N_c).",
    "Figure 8's LSQ column repeats Table 1b rows 40-44 with small differences (30,342 vs 30,432; 1.27E-2 vs 1.23E-2; "
    "171,563 vs 1.72E5; p_o 2.48 vs 2.49).",
    "Table 1a row 2 reports e_nh = 6.40E-1, ten times its neighbours: probably 6.40E-2.",
    "Table 1b row 36 reports e = 8.82E-2 with e_in = 8.81E-3 and p_o = 0.04%: almost certainly a typo for 8.82E-3.",
    "The text says e_in < rtol except rows 14 and 29; in Table 1 only row 29 (5.13E-3 > 5E-3) exceeds it.",
    "The text gives Lambda_ii = i^-3 for the quadratic; the code (anisotropic_quadratic_2, also at the paper-era "
    "commit 91ddd97) uses Lambda_ii = 5^-i.",
    "The text says rtol = alpha_tol in every experiment; the Lorenz-96 runs use an absolute test scaled by "
    "max|f| (paper-era 'Linf-global', now atol = tol * max|f| with rtol off).",
]


def records() -> list[dict]:
    """Every reported number as one record per table row (see the module docstring)."""
    out = []
    for row, problem, d, tol, N, Nc, po, e_in, e, e_nh in _CONVERGENCE:
        part = "a" if row <= 19 else "b"
        out.append(dict(
            ref=f"Table 1{part} #{row}", loc=f"convergence.tex tab:combined_convergence_{part.upper()}",
            group="conv", problem=problem,
            dim=d, tol=tol, mode="affine_cov_hess", N=N, N_c=Nc, p_o=po, e_in=e_in, e=e, e_nh=e_nh,
        ))  # fmt: skip
    for row, d, N, Nc, po, e_in, e in _LORENZ_DIMS:
        out.append(dict(
            ref=f"Table 2 #{row}", loc="convergence.tex tab:lorenz_dim_scaling", group="lorenz_dims",
            problem=LORENZ, dim=d, tol=0.1, mode="affine_cov_hess", N=N, N_c=Nc, p_o=po, e_in=e_in, e=e,
        ))  # fmt: skip
    for i, (atol, N0, Nc, po, e_in, e, good) in enumerate(_ROSEN_WEAK, 1):
        out.append(dict(
            ref=f"Table 3 #{i}", loc="convergence.tex tab:rosenbrock_weak_scaling", group="rosen_weak",
            problem=ROSEN, dim=5, atol=atol, N_0=N0, mode="affine_cov_hess", N_c=Nc, p_o=po, e_in=e_in, e=e,
            p_in_good=good,
        ))  # fmt: skip
    for tol, lsq, exact in _WAVE_EXACT:
        for mode, (Nc, po, e_in, e) in (("affine_cov_hess", lsq), ("affine_cov_hess_exact_grad", exact)):
            out.append(dict(
                ref=f"Figure 8 table, {'LSQ' if mode == 'affine_cov_hess' else 'exact'} gradient, rtol {tol}",
                loc="convergence.tex fig:exact_grad_vs_approx_grad_comparison", group="wave_exact",
                problem=WAVE, dim=5, tol=tol, mode=mode, N_c=Nc, p_o=po, e_in=e_in, e=e,
            ))  # fmt: skip
    for row, problem, d, tol, Nc, e in _COMPARISON:
        out.append(dict(
            ref=f"Table 4 #{row} (AFINNI)", loc="comparison.tex tab:other_surrogates", group="conv",
            problem=problem, dim=d, tol=tol, mode="affine_cov_hess", N_c=Nc, e=e,
        ))  # fmt: skip
    return out
