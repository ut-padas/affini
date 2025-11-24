import cupy as cp

from src.nearode import ClusteringBackend, NearODESurrogate
from src.problems.shcf import (
    six_hump_camelback,
    six_hump_camelback_grad,
    six_hump_camelback_hess,
)


def check_matrix_2norm(A, domain):
    # A is a function of input x, which expects x with shape (n, 2)
    x = cp.linspace(domain[0, 0], domain[0, 1], 300)
    y = cp.linspace(domain[1, 0], domain[1, 1], 300)
    xv, yv = cp.meshgrid(x, y)
    pts = cp.column_stack((xv.flatten(), yv.flatten()))

    # Vectorized computation: A returns (n, 2, 2), compute 2-norm for each matrix
    hess_matrices = A(pts)  # Shape: (n, 2, 2)
    norms = cp.linalg.norm(
        hess_matrices, ord=2, axis=(1, 2)
    )  # 2-norm of each (2, 2) matrix
    norms = norms.reshape((len(xv), len(yv)))

    max_ind = cp.unravel_index(norms.argmax(), norms.shape)
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
            entry = float(entry)
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
    atol=0.01, rtol=0.01, N_init=50000, domain=cp.array([[-3, 3], [-2, 2]]), min_ppc=100
):
    L = check_matrix_2norm(six_hump_camelback_hess, domain)

    interpolant = NearODESurrogate(data_dim=2, f_dim=1)
    interpolant.train_model_from_sampling(
        f=six_hump_camelback,
        grad_f=six_hump_camelback_grad,
        cupy_f=six_hump_camelback,
        cupy_grad_f=six_hump_camelback_grad,
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
        clustering_backend=ClusteringBackend.COMPLETE_CUPY,
    )
    sources = interpolant.get_sources()

    x = cp.linspace(domain[0, 0], domain[0, 1], 300)
    y = cp.linspace(domain[1, 0], domain[1, 1], 300)
    xv, yv = cp.meshgrid(x, y)
    pts = cp.column_stack((xv.flatten(), yv.flatten()))

    zz_exact = six_hump_camelback(pts)  # Already has shape (n, 1)
    zz_interpolant, bad_pts_inds = interpolant.evaluate(
        pts,
        k=1,
        L=L,
        recluster=False,
        clustering_backend=ClusteringBackend.COMPLETE_CUPY,
    )
    abs_error = cp.linalg.norm(zz_exact - zz_interpolant, 2, axis=1)
    rel_error = abs_error / cp.linalg.norm(zz_exact, 2, axis=1)
    good_abs_error = abs_error <= atol
    good_rel_error = rel_error <= rtol
    good_overall_error = cp.logical_or(good_abs_error, good_rel_error)
    bad_overall_error = cp.logical_not(good_overall_error)
    good_pts_inds = cp.logical_not(bad_pts_inds)

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
            clustering_backend=ClusteringBackend.COMPLETE_CUPY,
        )
        good_pts_inds = cp.logical_not(bad_pts_inds)
        sources = interpolant.get_sources()
        zz_exact = zz_exact.flatten()
        abs_error = cp.abs(zz_exact - zz_interpolant)
        rel_error = abs_error / cp.abs(zz_exact)
        good_abs_error = abs_error <= atol
        good_rel_error = rel_error <= rtol
        good_overall_error = cp.logical_or(good_abs_error, good_rel_error)
        bad_overall_error = cp.logical_not(good_overall_error)
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
    datavec = cp.array(
        [
            float(N_init),
            float(atol),
            float(rtol),
            float(interpolant.N_sources),
            float((cp.sum(no_g_good_err) + cp.sum(no_g_bad_err)) / len(pts)),
            float(cp.sum(yes_g_good_err) / len(pts)),
            float(cp.sum(yes_g_bad_err) / len(pts)),
            float(cp.max(red_abs_errs)) if len(red_abs_errs) > 0 else 0.0,
            float(cp.median(red_abs_errs)) if len(red_abs_errs) > 0 else 0.0,
            (
                float(cp.sqrt(cp.sum(red_abs_errs**2) / len(red_abs_errs)))
                if len(red_abs_errs) > 0
                else 0.0
            ),
            float(cp.max(red_rel_errs)) if len(red_rel_errs) > 0 else 0.0,
            float(cp.median(red_rel_errs)) if len(red_rel_errs) > 0 else 0.0,
            (
                float(cp.sqrt(cp.sum(red_rel_errs**2) / len(red_rel_errs)))
                if len(red_rel_errs) > 0
                else 0.0
            ),
        ]
    )

    return datavec


def convergence_test_2D(tolmat, N_init, domain, min_ppc=40):
    datamat = cp.empty((len(tolmat), 13))
    for j, tols in enumerate(tolmat):
        datamat[j] = test_2D(
            atol=tols[0], rtol=tols[1], N_init=N_init, domain=domain, min_ppc=min_ppc
        )

    print("\n")
    print(generate_latex_table(datamat))
    print("\n")


if __name__ == "__main__":
    # import cProfile
    from pathlib import Path

    FILE_NAME = Path(__file__).resolve().stem
    # cprofile = cProfile.Profile()
    # cprofile.enable()

    tolmat = cp.array(
        [
            [0.1, 0.02],
            [0.05, 0.01],
            [0.025, 0.005],
            [0.01, 0.002],
            [0.005, 0.001],
        ]
    )

    domain = cp.array([[-3, 3], [-2, 2]])
    convergence_test_2D(tolmat, 1_000_000_000, domain)

    # cprofile.disable()
    # cprofile.dump_stats(f"{FILE_NAME}_profile.prof")
