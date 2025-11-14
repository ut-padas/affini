from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

from src.nearode import ClusteringBackend, NearODESurrogate
from src.problems.borehole import borehole, grad_borehole
from src.problems.oned import grad_one_d_example, one_d_example
from src.problems.shcf import (
    _HAS_GPU,
    SHCF_cupy,
    SHCF_numpy,
    grad_SHCF_numpy,
    hessian_SHCF_numpy,
)

CWD = Path.cwd()
FIGURES_DIR = CWD / "figures" / "experiments"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def test_1D(N_init=2000, atol=0.01, make_plots=False):
    N = 4000
    domain = np.array([0, 10])
    interpolant = NearODESurrogate()
    interpolant.train_model_from_sampling(
        f=one_d_example,
        grad_f=grad_one_d_example,
        cupy_f=None,
        cupy_grad_f=None,
        clustering_backend=ClusteringBackend.NUMPY_CUPY_HYBRID,
        domain=domain,
        N=N_init,
        distribution="uniform",
        num_initial_clusters=20,
        atol=atol,
        alpha=100000,
    )

    xx = np.linspace(domain[0], domain[1], N)
    xx = np.atleast_2d(xx).T
    yy_interpolant, bad_points = interpolant.evaluate(
        targets=xx.reshape((N, 1)),
        k=1,
        clustering_backend=ClusteringBackend.NUMPY_CUPY_HYBRID,
    )
    yy_exact = one_d_example(xx)
    error = np.abs(yy_interpolant - yy_exact)
    rmse = np.sqrt(np.sum(error**2) / len(error))
    sources = interpolant.get_sources()

    if make_plots:
        plt.figure()
        plt.plot(xx, yy_exact, "-k", label="Exact")
        plt.plot(xx, yy_interpolant, "--m", label=f"Interpolant")
        plt.scatter(
            sources,
            one_d_example(sources),
            c="orange",
            marker="x",
            label=f"Sources ($N_s = {len(sources)}$)",
        )
        plt.scatter(
            xx[bad_points],
            one_d_example(xx[bad_points]),
            marker="o",
            label=f"Unguaranteed Points ($N = {bad_points.sum()}$)",
        )
        plt.legend()
        plt.xlabel("$x$")

        plt.figure()
        plt.plot(xx, error, "--r", label="Log Error")
        plt.plot(xx, np.full_like(xx, atol), "--k")
        plt.scatter(
            xx[bad_points],
            error[bad_points],
            marker="o",
            label=f"Unguaranteed Points ($N = {bad_points.sum()}$)",
        )
        plt.yscale("log")
        plt.legend()
        plt.xlabel("$x$")
        plt.savefig(FIGURES_DIR / "1D_example_error.png", dpi=400)

    print(f"Max l-infty error: {error.max()}")
    print(f"Root mean squared error: {rmse}")


def check_matrix_2norm(A, domain, make_plots=False):
    # A is a function of input x, which expects x with shape (n, 2)
    x = np.linspace(domain[0, 0], domain[0, 1], 300)
    y = np.linspace(domain[1, 0], domain[1, 1], 300)
    xv, yv = np.meshgrid(x, y)
    pts = np.column_stack((xv.flatten(), yv.flatten()))

    # Vectorized computation: A returns (n, 2, 2), compute 2-norm for each matrix
    hess_matrices = A(pts)  # Shape: (n, 2, 2)
    norms = np.linalg.norm(
        hess_matrices, ord=2, axis=(1, 2)
    )  # 2-norm of each (2, 2) matrix
    norms = norms.reshape((len(xv), len(yv)))

    if make_plots:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xv, yv, norms, cmap="viridis")
        ax.set_title("Hessian of SHCF: 2-norms")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        plt.savefig(FIGURES_DIR / "SHCF_hessian_2norms.png", dpi=400)

    max_ind = np.unravel_index(norms.argmax(), norms.shape)
    max_val = norms.max()
    print(f"Max achieved at ({xv[max_ind]}, {yv[max_ind]}), value = {max_val}")

    return max_val


def generate_latex_table(data):
    colstring = "|c|c|c||c|c|c|c||c|c|c|c|"
    headers = [
        "$N_{init}$",
        "atol",
        "rtol",
        "$N_{sources}$",
        "$p_{out}$",
        "$p_{in,good}$",
        "$p_{in,bad}$",
        "$\\|e\\|_\\infty$",
        "$e_{med}$",
        "$\\|e\\|_\\infty / \\|f\\|_\\infty$",
        "$\\|e\\|_2 / \\|f\\|_2$",
    ]
    headerstring = ""
    for j, header in enumerate(headers):
        headerstring += header
        if j < len(headers) - 1:
            headerstring += " & "
        else:
            headerstring += "\\\\"

    tablestr = f"""\
\\begin{{table}}[h!]
    \\centering
    \\begin{{tabular}}{{{colstring}}}
        \\hline
        {headerstring}
        \\hline\\hline
"""
    for row in data:
        rowstr = "\t\t"
        for j, entry in enumerate(row):
            if j == 0 or j == 3:
                rowstr += str(int(entry))
            elif j == 1:
                rowstr += str(entry)
            elif j == 2:
                rowstr += str(100.0 * entry) + "\\%"
            elif j >= 4 and j <= 6:
                rowstr += str(round(100 * entry, 2)) + "\\%"
            elif j >= 7:
                if entry > 0:
                    rowstr += str(round(entry, 4))
                else:
                    rowstr += "-"

            if j < len(row) - 1:
                rowstr += " & "
            else:
                rowstr += " \\\\\n"
        tablestr += rowstr

    tablestr += f"""\
        \\hline
    \\end{{tabular}}
    \\caption{{This is a caption!}}
    \\label{{tab:new_tab}}
\\end{{table}}\
"""

    print(tablestr)


def test_2D(
    atol=0.01, rtol=0.01, N_init=50000, domain=np.array([[-3, 3], [-2, 2]]), min_ppc=100
):
    L = check_matrix_2norm(hessian_SHCF_numpy, domain, make_plots=True)

    cupy_f = SHCF_cupy if _HAS_GPU else None

    interpolant = NearODESurrogate()
    interpolant.train_model_from_sampling(
        f=SHCF_numpy,
        grad_f=grad_SHCF_numpy,
        cupy_f=cupy_f,
        cupy_grad_f=None,
        clustering_backend=ClusteringBackend.NUMPY_CUPY_HYBRID,
        domain=domain,
        N=N_init,
        distribution="LHS",
        num_initial_clusters=100,
        min_pts_per_cluster=min_ppc,
        atol=atol,
        rtol=rtol,
        check_guarantee_N_times=0,
        L=L,
        seed=1234,
    )
    sources = interpolant.get_sources()

    x = np.linspace(domain[0, 0], domain[0, 1], 300)
    y = np.linspace(domain[1, 0], domain[1, 1], 300)
    xv, yv = np.meshgrid(x, y)
    pts = np.column_stack((xv.flatten(), yv.flatten()))

    zz_exact = SHCF_numpy(pts)  # Already has shape (n, 1)
    zz_interpolant, bad_pts_inds = interpolant.evaluate(
        pts,
        k=1,
        L=L,
        recluster=False,
        clustering_backend=ClusteringBackend.NUMPY_CUPY_HYBRID,
    )
    abs_error = np.linalg.norm(zz_exact - zz_interpolant, 2, axis=1)
    rel_error = abs_error / np.linalg.norm(zz_exact, 2, axis=1)
    good_abs_error = abs_error <= atol
    good_rel_error = rel_error <= rtol
    good_overall_error = np.logical_or(good_abs_error, good_rel_error)
    bad_overall_error = np.logical_not(good_overall_error)
    good_pts_inds = np.logical_not(bad_pts_inds)

    # sort the points outside of guarantee between those inside the error and those not
    no_g_good_err = bad_pts_inds & good_overall_error  # 1 (yellow)
    no_g_bad_err = bad_pts_inds & bad_overall_error  # 2 (orange)
    # sort the points inside of guarantee as well
    yes_g_good_err = good_pts_inds & good_overall_error  # 3 (green)
    yes_g_bad_err = good_pts_inds & bad_overall_error  # 4 (red)
    sorted_grid = (
        0 * no_g_good_err + 1 * no_g_bad_err + 2 * yes_g_good_err + 3 * yes_g_bad_err
    )

    zz_exact = zz_exact.reshape((len(xv), len(yv)))
    abs_error_reshaped = abs_error.reshape((len(xv), len(yv)))
    zz_interpolant = zz_interpolant.reshape((len(xv), len(yv)))
    sorted_grid = sorted_grid.reshape((len(xv), len(yv)))

    # CHECK ACCURACY AFTER ONE RECLUSTERING
    for j in range(0):
        zz_interpolant, bad_pts_inds = interpolant.evaluate(
            pts,
            k=1,
            L=L,
            recluster=True,
            clustering_backend=ClusteringBackend.NUMPY_CUPY_HYBRID,
        )
        good_pts_inds = np.logical_not(bad_pts_inds)
        sources = interpolant.get_sources()
        zz_exact = zz_exact.flatten()
        abs_error = np.abs(zz_exact - zz_interpolant)
        rel_error = abs_error / np.abs(zz_exact)
        good_abs_error = abs_error <= atol
        good_rel_error = rel_error <= rtol
        good_overall_error = np.logical_or(good_abs_error, good_rel_error)
        bad_overall_error = np.logical_not(good_overall_error)
        # sort the points outside of guarantee between those inside the error and those not
        no_g_good_err = (
            bad_pts_inds & good_overall_error
        )  # points outside guarantee, but have good error; 1
        no_g_bad_err = (
            bad_pts_inds & bad_overall_error
        )  # points outside guarantee and have bad error; 2
        # sort the points inside of guarantee as well
        yes_g_good_err = good_pts_inds & good_overall_error  # 3
        yes_g_bad_err = good_pts_inds & bad_overall_error  # 4
        sorted_grid = (
            0 * no_g_good_err
            + 1 * no_g_bad_err
            + 2 * yes_g_good_err
            + 3 * yes_g_bad_err
        )

        zz_exact = zz_exact.reshape((len(xv), len(yv)))
        abs_error_reshaped = abs_error.reshape((len(xv), len(yv)))
        zz_interpolant = zz_interpolant.reshape((len(xv), len(yv)))
        sorted_grid = sorted_grid.reshape((len(xv), len(yv)))

    # display stats about errors
    red_abs_errs = abs_error[yes_g_bad_err]
    red_rel_errs = rel_error[yes_g_bad_err]
    datavec = np.array(
        [
            N_init,
            atol,
            rtol,
            interpolant.N_sources,
            (np.sum(no_g_good_err) + np.sum(no_g_bad_err)) / len(pts),
            np.sum(yes_g_good_err) / len(pts),
            np.sum(yes_g_bad_err) / len(pts),
            np.max(red_abs_errs),
            np.median(red_abs_errs),
            np.sqrt(np.sum(red_abs_errs) / len(red_abs_errs)),
            np.max(red_rel_errs),
            np.median(red_rel_errs),
            np.sqrt(np.sum(red_rel_errs) / len(red_rel_errs)),
        ]
    )

    return datavec


def convergence_test_2D(tolmat, N_init, domain, min_ppc=40):
    datamat = np.empty((len(tolmat), 13))
    for j, tols in enumerate(tolmat):
        datamat[j] = test_2D(
            atol=tols[0], rtol=tols[1], N_init=N_init, domain=domain, min_ppc=min_ppc
        )

    print("\n")
    print(generate_latex_table(datamat))
    print("\n")


def convergence_test_a_function(test_function, tolmat, N_init, min_ppc=100):
    datamat = np.empty((len(tolmat), 11))
    for j, tols in tqdm(enumerate(tolmat), total=len(tolmat)):
        datamat[j] = test_function(
            atol=tols[0], rtol=tols[1], N_init=N_init, min_ppc=min_ppc
        )

    print("\n")
    print(generate_latex_table(datamat))
    print("\n")


def test_a_function(
    f, grad_f, m, domain, atol=0.01, rtol=0.01, N_init=10000, min_ppc=50
):
    interpolant = NearODESurrogate()
    interpolant.train_model_from_sampling(
        f=f,
        grad_f=grad_f,
        cupy_f=None,
        cupy_grad_f=None,
        clustering_backend=ClusteringBackend.NUMPY_CUPY_HYBRID,
        domain=domain,
        N=N_init,
        distribution="LHS",
        num_initial_clusters=100,
        atol=atol,
        rtol=rtol,
        min_pts_per_cluster=min_ppc,
        check_guarantee_N_times=0,
    )

    # evaluate exact
    # pick some evaluation points using LHS
    N_eval = 10000
    l = domain[:, 0]
    r = domain[:, 1]
    sampler = qmc.LatinHypercube(d=len(domain))
    pts = sampler.random(n=N_eval)
    pts = qmc.scale(pts, l, r)

    exact = f(pts)
    interp, bad_pts = interpolant.evaluate(
        pts, clustering_backend=ClusteringBackend.NUMPY_CUPY_HYBRID
    )

    abs_error = np.linalg.norm(exact - interp, 2, axis=1)
    rel_error = abs_error / np.linalg.norm(exact, 2, axis=1)
    good_abs_error = abs_error <= atol
    good_rel_error = rel_error <= rtol
    good_overall_error = np.logical_or(good_abs_error, good_rel_error)
    bad_overall_error = np.logical_not(good_overall_error)
    good_pts_inds = np.logical_not(bad_pts)

    # sort the points outside of guarantee between those inside the error and those not
    no_g_good_err = bad_pts & good_overall_error  # 1 (yellow)
    no_g_bad_err = bad_pts & bad_overall_error  # 2 (orange)
    # sort the points inside of guarantee as well
    yes_g_good_err = good_pts_inds & good_overall_error  # 3 (green)
    yes_g_bad_err = good_pts_inds & bad_overall_error  # 4 (red)

    # display stats about errors
    red_abs_errs = abs_error[yes_g_bad_err]
    red_rel_errs = rel_error[yes_g_bad_err]
    red_true_vals = interp[yes_g_bad_err]
    if len(red_abs_errs > 0):
        datavec = np.array(
            [
                # section 1: initial values
                N_init,
                atol,
                rtol,
                # section 2: point results
                interpolant.N_sources,
                (np.sum(no_g_good_err) + np.sum(no_g_bad_err)) / len(pts),
                np.sum(yes_g_good_err) / len(pts),
                np.sum(yes_g_bad_err) / len(pts),
                # section 3: errors for failure points
                np.max(red_abs_errs),  # max abs error
                np.median(red_abs_errs),  # median abs error
                np.linalg.norm(red_abs_errs, np.inf)
                / np.linalg.norm(red_true_vals, np.inf),  # l-inf rel error
                np.linalg.norm(red_abs_errs, 2)
                / np.linalg.norm(red_true_vals, 2),  # l2 rel error
            ]
        )
    else:
        datavec = np.array(
            [
                N_init,
                atol,
                rtol,
                interpolant.N_sources,
                (np.sum(no_g_good_err) + np.sum(no_g_bad_err)) / len(pts),
                np.sum(yes_g_good_err) / len(pts),
                np.sum(yes_g_bad_err) / len(pts),
                0,
                0,
                0,
                0,
            ]
        )

    return datavec


def test_8D(atol=0.2, rtol=0.01, N_init=10000, min_ppc=40):
    domain = np.array(
        [
            [0.05, 0.15],
            [100, 50000],
            [63070, 115600],
            [990, 1110],
            [63.1, 116],
            [700, 820],
            [1120, 1680],
            [9855, 12045],
        ]
    )
    test_a_function(
        f=borehole,
        grad_f=grad_borehole,
        m=8,
        domain=domain,
        atol=atol,
        rtol=rtol,
        N_init=N_init,
        min_ppc=min_ppc,
    )


def convergence_test_8D(tolmat, N_init, min_ppc=40):
    convergence_test_a_function(test_8D, tolmat, N_init, min_ppc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run convergence tests for NearODE surrogate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problem",
        "-p",
        type=str,
        default="borehole",
        choices=["1d", "2d-shcf", "borehole", "all"],
        help="Choose which problem to run.",
    )
    args = parser.parse_args()
    problem_name = str(args.problem).lower()
    print(f"Selected problem: {problem_name}")

    tolmat = np.array(
        [
            [0.1, 0.02],
            [0.05, 0.01],
            [0.025, 0.005],
            [0.01, 0.002],
            [0.005, 0.001],
        ],
    )

    if problem_name in ("1d", "all"):
        print("Running 1D test...")
        test_1D(atol=1e-3, make_plots=True)
    if problem_name in ("2d-shcf", "all"):
        print("Running 2D SHCF convergence test...")
        domain = np.array([[-3, 3], [-2, 2]])
        convergence_test_2D(tolmat, 10_000, domain)
    if problem_name in ("borehole", "all"):
        print("Running 8D Borehole convergence test...")
        convergence_test_8D(tolmat, N_init=1000, min_ppc=10)
