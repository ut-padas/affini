import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator

from src.nearode import NearODESurrogate

cp.cuda.set_allocator(rmm_cupy_allocator)


# Six-Hump Camelback function
def SHCF(x: cp.ndarray):
    """
    Vectorized Six-Hump Camelback function.

    Shape Convention:
        x.shape: (n, 2) where n=number of points, input is 2D
        Returns: (n, 1) function values
    """
    scale = 10
    x1, x2 = x[:, 0], x[:, 1]

    result = (
        (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
        + x1 * x2
        + (-4 + 4 * x2**2) * x2**2
        + scale
    )

    return result[:, cp.newaxis]  # Shape (n, 1)


def grad_SHCF(x: cp.ndarray):
    """
    Vectorized gradient of Six-Hump Camelback function.

    Shape Convention:
        x.shape: (n, 2) where n=number of points, input is 2D
        Returns: (n, 1, 2) gradients where each gradient is shape (1, 2)
    """
    x1, x2 = x[:, 0], x[:, 1]

    # Compute gradient components
    Dxf = 8 * x1 - 8.4 * x1**3 + 2 * x1**5 + x2
    Dyf = x1 - 8 * x2 + 16 * x2**3

    # Stack gradients: shape (n, 2)
    grads = cp.stack([Dxf, Dyf], axis=1)

    # Add new axis to get shape (n, 1, 2) to follow convention: (n, fd, d) where fd=1, d=2
    return grads[:, cp.newaxis, :]


def hessian_SHCF(x: cp.ndarray):
    """
    Vectorized hessian of Six-Hump Camelback function.

    Shape Convention:
        x.shape: (n, 2) where n=number of points, input is 2D
        Returns: (n, 2, 2) Hessian matrices
    """
    x1, x2 = x[:, 0], x[:, 1]

    # Compute hessian components for each point
    h11 = 8 - 25.2 * x1**2 + 10 * x1**4
    h12 = cp.ones_like(x1)  # Off-diagonal element
    h22 = -8 + 48 * x2**2

    # Stack into hessian matrices: shape (n, 2, 2)
    hessians = cp.stack(
        [cp.stack([h11, h12], axis=1), cp.stack([h12, h22], axis=1)], axis=1
    )

    return hessians


def SHCF_combined(x: cp.ndarray):
    """
    Combined function returning both function value and gradient.

    Shape Convention:
        x.shape: (n, 2) where n=number of points, input is 2D
        Returns:
            - values: (n, 1) function values
            - gradients: (n, 1, 2) gradients
    """
    return SHCF(x), grad_SHCF(x)


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
    L = check_matrix_2norm(hessian_SHCF, domain)

    interpolant = NearODESurrogate()
    interpolant.train_model_from_sampling(
        SHCF_combined,
        domain,
        1,
        N_init,
        "LHS",
        True,
        N_init_clusters=100,
        min_ppc=min_ppc,
        atol=atol,
        rtol=rtol,
        check_guarantee_N_times=0,
        L=L,
        seed=1234,
    )
    sources = interpolant.get_sources()

    x = cp.linspace(domain[0, 0], domain[0, 1], 300)
    y = cp.linspace(domain[1, 0], domain[1, 1], 300)
    xv, yv = cp.meshgrid(x, y)
    pts = cp.column_stack((xv.flatten(), yv.flatten()))

    zz_exact = SHCF(pts)  # Already has shape (n, 1)
    zz_interpolant, bad_pts_inds = interpolant.evaluate(pts, k=1, L=L, recluster=False)
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
            pts, k=1, L=L, recluster=True
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
    import cProfile
    from pathlib import Path

    FILE_NAME = Path(__file__).resolve().stem
    cprofile = cProfile.Profile()
    cprofile.enable()

    tolmat = cp.array(
        [[0.1, 0.02]],  # [0.05, 0.01], [0.025, 0.005], [0.01, 0.002], [0.005, 0.001]]
    )

    domain = cp.array([[-3, 3], [-2, 2]])
    convergence_test_2D(tolmat, 100_000_000, domain)

    cprofile.disable()
    cprofile.dump_stats(f"{FILE_NAME}_profile.prof")
