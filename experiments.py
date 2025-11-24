from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from nearode import NearODESurrogate, load_surrogate, save_surrogate
from scipy.stats import qmc
from scipy.integrate import solve_ivp


# One dimensional example
# x \in [0,10]
def one_d_example(x): 
    return np.sin(2*np.pi*x) / np.exp(x)

def grad_one_d_example(x):
    return (2*np.pi*np.cos(2*np.pi*x) - np.sin(2*np.pi*x)) / np.exp(x)

def one_d_combined(x):
    return one_d_example(x), grad_one_d_example(x)


def f_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    return helper


# Rosenbrock (2D)
a = 1
b = 100
def rosenbrock(x: np.ndarray):
    if x.ndim == 1:
        return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
    elif x.ndim == 2:
        return (a - x[:,0]) ** 2 + b * (x[:,1] - x[:,0] ** 2) ** 2

def grad_rosenbrock(x: np.ndarray):
    if x.ndim == 1:
        g = np.array([
            -2 * (a-x[0]) - 4*b*x[0]*(x[1] - x[0]**2),
            2*b*(x[1]-x[0]**2)
        ])
        return np.expand_dims(g, axis=1).T
    elif x.ndim == 2:
        Dxf = -2 * (a-x[:,0]) - 4*b*x[:,0]*(x[:,1] - x[:,0]**2)
        Dyf = 2*b*(x[:,1]-x[:,0]**2)
        g = np.column_stack((Dxf, Dyf))
        return np.expand_dims(g, axis=1).T

def rosenbrock_combined(x: np.ndarray):
    return rosenbrock(x), grad_rosenbrock(x)

# Six-Hump Camelback function
def SHCF(x: np.ndarray):
    if x.ndim == 1:
        return (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2
    elif x.ndim == 2:
        return (4 - 2.1*x[:,0]**2 + x[:,0]**4/3)*x[:,0]**2 + x[:,0]*x[:,1] + (-4 + 4*x[:,1]**2)*x[:,1]**2 + scale

def grad_SHCF(x: np.ndarray):

    if x.ndim == 1:
        g = np.array([
            8*x[0] - 8.4 * x[0]**3 + 2*x[0]**5 + x[1],
            x[0] - 8*x[1] + 16*x[1]**3
        ])
        return np.expand_dims(g, axis=1).T
    elif x.ndim == 2:
        Dxf = 8*x[:,0] - 8.4 * x[:,0]**3 + 2*x[:,0]**5 + x[:,1]
        Dyf = x[:,0] - 8*x[:,1] + 16*x[:,1]**3
        g = np.column_stack((Dxf, Dyf))
        return np.expand_dims(g, axis=1).T
    
def hessian_SHCF(x: np.ndarray):
    if x.ndim == 1:
        return np.array([
            [8 - 25.2*x[0]**2 + 10*x[0]**4, 1],
            [1, -8 + 48*x[1]**2]
        ])

@f_counter
def SHCF_combined(x: np.ndarray):
    return SHCF(x), grad_SHCF(x)

def check_matrix_2norm(A, domain):
    # A is a function of input x, which is a 2-vector
    x = np.linspace(domain[0,0], domain[0,1], 300)
    y = np.linspace(domain[1,0], domain[1,1], 300)
    xv, yv = np.meshgrid(x, y)
    pts = np.column_stack((xv.flatten(), yv.flatten()))
    norms = np.empty(len(pts))

    for j, pt in enumerate(pts):
        norms[j] = np.linalg.norm(A(pt), 2)
    norms = norms.reshape((len(xv),len(yv)))
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xv, yv, norms, cmap='viridis')
    # ax.set_title('Hessian of SHCF: 2-norms')
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    # plt.show()

    max_ind = np.unravel_index(norms.argmax(), norms.shape)
    max_val = norms.max()
    print(f'Max achieved at ({xv[max_ind]}, {yv[max_ind]}), value = {max_val}')

    return max_val

# Eight dimensional borehole function (from Kriging paper)
# Code produced via Mathematica
def borehole(x):
    r_w = x[0]
    r   = x[1]
    T_u = x[2]
    H_u = x[3]
    T_l = x[4]
    H_l = x[5]
    L   = x[6]
    K_w = x[7]

    return 5*(-1*H_l+H_u)*T_u*np.power((np.power(T_l,-1)*T_u+np.log(np.power(r_w,-1)*r)*(1.5+2*L*\
    np.power(K_w,-1)*np.power(r_w,-2)*T_u*np.power(np.log(np.power(r_w,-1)*r),-1))),-1)

def grad_borehole(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    x8 = x[7]

    g = np.zeros(8)

    g[0] = -5*x3*(x4+-1*x6)*(-1*np.power(x1,-1)*(1.5+2*np.power(x1,-2)*np.power(x8,-1)*x3*x7*\
    np.power(np.log(np.power(x1,-1)*x2),-1))+np.log(np.power(x1,-1)*x2)*(-4*np.power(x1,-\
    3)*np.power(x8,-1)*x3*x7*np.power(np.log(np.power(x1,-1)*x2),-1)+2*np.power(x1,-3)*np.power(\
    x8,-1)*x3*x7*np.power(np.log(np.power(x1,-1)*x2),-2)))*np.power((np.power(x5,-1)*x3+\
    np.log(np.power(x1,-1)*x2)*(1.5+2*np.power(x1,-2)*np.power(x8,-1)*x3*x7*np.power(np.\
    log(np.power(x1,-1)*x2),-1))),-2)

    g[1] = -5*x3*(x4+-1*x6)*(-2*np.power(x1,-2)*np.power(x2,-1)*np.power(x8,-1)*x3*x7*np.power(np.\
    log(np.power(x1,-1)*x2),-1)+np.power(x2,-1)*(1.5+2*np.power(x1,-2)*np.power(x8,-1)*x3*x7*\
    np.power(np.log(np.power(x1,-1)*x2),-1)))*np.power((np.power(x5,-1)*x3+np.log(np.power(x1,-\
    1)*x2)*(1.5+2*np.power(x1,-2)*np.power(x8,-1)*x3*x7*np.power(np.log(pow(x1,-1)*x2),\
    -1))),-2)

    g[2] = (5*(x4+-1*x6)*np.power((np.power(x5,-1)*x3+np.log(np.power(x1,-1)*x2)*(1.5+2*np.power(\
    x1,-2)*np.power(x8,-1)*x3*x7*np.power(np.log(np.power(x1,-1)*x2),-1))),-1)+-5*x3*(\
    x4+-1*x6)*(np.power(x5,-1)+2*np.power(x1,-2)*np.power(x8,-1)*x7)*np.power((np.power(x5,-1)*x3+\
    np.log(np.power(x1,-1)*x2)*(1.5+2*np.power(x1,-2)*np.power(x8,-1)*x3*x7*np.power(np.\
    log(np.power(x1,-1)*x2),-1))),-2))

    g[3] = 5*x3*np.power((np.power(x5,-1)*x3+np.log(np.power(x1,-1)*x2)*(1.5+2*np.power(x1,-2)*\
    np.power(x8,-1)*x3*x7*np.power(np.log(pow(x1,-1)*x2),-1))),-1)

    g[4] = 5*np.power(x3,2)*np.power(x5,-2)*(x4+-1*x6)*np.power((np.power(x5,-1)*x3+np.log(np.power(x1,-\
    1)*x2)*(1.5+2*np.power(x1,-2)*np.power(x8,-1)*x3*x7*np.power(np.log(pow(x1,-1)*x2),-1))),-2)

    g[5] = -5*x3*np.power((np.power(x5,-1)*x3+np.log(np.power(x1,-1)*x2)*(1.5+2*np.power(x1,-2)*\
    np.power(x8,-1)*x3*x7*np.power(np.log(np.power(x1,-1)*x2),-1))),-1)

    g[6] = -10*np.power(x1,-2)*np.power(x3,2)*np.power(x8,-1)*(x4+-1*x6)*np.power((np.power(x5,-1)*x3+\
    np.log(np.power(x1,-1)*x2)*(1.5+2*np.power(x1,-2)*np.power(x8,-1)*x3*x7*np.power(np.\
    log(np.power(x1,-1)*x2),-1))),-2)

    g[7] = 10*np.power(x1,-2)*np.power(x3,2)*np.power(x8,-2)*(x4+-1*x6)*x7*np.power((np.power(x5,-1)*x3+\
    np.log(np.power(x1,-1)*x2)*(1.5+2*np.power(x1,-2)*np.power(x8,-1)*x3*x7*np.power(np.\
    log(np.power(x1,-1)*x2),-1))),-2)

    return np.expand_dims(g, axis=1).T

def borehole_combined(x):
    return borehole(x), grad_borehole(x)

# Arbitrary dimension Dixon-Price function
def dixon_price(x):
    n = len(x)
    total = (x[0] - 1) ** 2
    for j in range(1,n):
        total += (j+1) * (2 * x[j]**2 - x[j-1]) ** 2
    return total

def grad_dixon_price(x):
    n = len(x)
    g = np.zeros(n)
    g[0] = 2 * (x[0] - 1) - 4 * (2 * x[1] ** 2 - x[0])
    g[-1] = 2 * n * (2 * x[-1] ** 2 - x[-2]) * (4 * x[-1])
    for j in range(1,n-1):
        g[j] = 2 * (j+1) * (2 * x[j] ** 2 - x[j-1]) * (4 * x[j]) \
                    - 2 * (j+2) * (2 * x[j+1] ** 2 - x[j])
    return np.expand_dims(g, axis=1).T

def dixon_price_combined(x):
    return dixon_price(x), grad_dixon_price(x)


# vector function (affine)
A = np.random.uniform(0, 1, size=(10,20))
c = np.random.uniform(0, 1, size=(10))
def affine(x):
    return A @ x + c

def grad_affine():
    return A

def affine_combined(x):
    return affine(x), grad_affine()


# quadratic vector function (Lorenz 96), arbitrary dimensions
def L96(x,p):
    """Lorenz 96 model with constant forcing"""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + p[0]

# Thank you ChatGPT for this
def dx_L96(x,p):
    """
    Compute the Jacobian matrix of the Lorenz-96 system at state x.
    
    Parameters
    ----------
    x : array_like, shape (N,)
        State vector [x1, x2, ..., xN].
    
    Returns
    -------
    J : ndarray, shape (N, N)
        Jacobian matrix evaluated at x.
    """
    x = np.asarray(x)
    N = x.size
    J = np.zeros((N, N))
    
    # Cyclic indices
    im2 = np.arange(N) - 2
    im1 = np.arange(N) - 1
    ip1 = np.arange(N) + 1
    
    # Wrap around modulo N
    im2 %= N
    im1 %= N
    ip1 %= N
    
    # Fill nonzero entries
    J[np.arange(N), im2] = -x[im1]              # ∂f_i / ∂x_{i-2}
    J[np.arange(N), im1] = x[ip1] - x[im2]      # ∂f_i / ∂x_{i-1}
    J[np.arange(N), np.arange(N)] = -1          # ∂f_i / ∂x_i
    J[np.arange(N), ip1] = x[im1]               # ∂f_i / ∂x_{i+1}
    
    return J

def dp_L96(x,p):
    return np.ones(shape=len(x))


# solving ODEs!
def sensitivities_solver(f, J, DfDp, x0, p, tf):
    # x0 is the input
    # tf is the final time

    # number of parameters
    Np = len(p)
    n = len(x0)

    new_ic = np.zeros(len(x0) * (1 + Np + n))
    new_ic[:n] = x0
    new_ic[2*n:2*n+n**2] = np.eye(n).flatten()

    def new_rhs(x, f, DfDx, DfDp):
        rhsvec = np.zeros_like(x)
        rhsvec[:n] = f(x[:n],p)

        J = DfDx(x[:n],p)
        for j in range(1,n+2):
            if j == 1:
                rhsvec[n:2*n] = J @ x[n:2*n] + DfDp(x[:n],p)
            else:
                rhsvec[j*n:(j+1)*n] = J @ x[j*n:(j+1)*n]
        return rhsvec
    
    sol = solve_ivp(lambda t,x: new_rhs(x,f,J,DfDp), [0,tf], new_ic, max_step=0.1)

    return sol

def L96_combined(x,p):
    return L96(x,p), dx_L96(x,p)

def solve_L96_ode(input, F, tf):
    p = np.array([F])
    x0 = input[0:]
    sol = solve_ivp(lambda t,x: L96(x,p), [0,tf], x0, max_step=0.1, atol=1e-6, rtol=1e-4)
    xf = sol.y[:,-1]
    return xf

def solve_L96_ode_return_all(input, F, tf):
    p = np.array([F])
    x0 = input[0:]
    sol = solve_ivp(lambda t,x: L96(x,p), [0,tf], x0, max_step=0.1, atol=1e-6, rtol=1e-4)
    return sol

def test_and_plot_L96_ode_solve(input, F, tf):
    sol = solve_L96_ode_return_all(input, F, tf)
    x = sol.y
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x[0,:], x[1,:], x[2,:])
    ax.set_title(f'Lorenz 96 ODE plot (first 3 variables), d={len(input)}, tf={tf}, F={F}')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    plt.show()

# this is the version we supply to the nearest neighbors algorithm
# input is a length (N+1) vector, where the first entry is the value F (forcing)
# output is a length N vector (the solve at a time tf) and the gradient (size N*(N+1))
def solve_L96_ode_with_sensitivities(input, F, tf):
    p = np.array([F])
    x0 = input[0:]
    sol = sensitivities_solver(L96, dx_L96, dp_L96, x0, p, tf)
    xf = sol.y[:,-1]
    n = len(x0)
    return xf[:n], np.reshape(xf[2*n:], shape=(n,n)).T


def SVD_analysis(points):
    # assume points is a p*n matrix (each column corresponds to a point)
    # first, normalize data
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    points_normalized = (points - mean) / std

    # compute SVD
    return np.linalg.svd(points_normalized, compute_uv=False)



def generate_latex_table(data):

    colstring = '|c|c|c||c|c|c|c||c|c|c||c|c|'
    headers = ['$N_{init}$', 'atol', 'rtol',
        '$N_{sources}$', '$p_{out}$', '$p_{in,good}$', '$p_{in,bad}$',
        '$\\|e\\|_2 / \\|f\\|_2$', '$\\|e\\|_\\infty / \\|f\\|_\\infty$', '$\\|e\\|_\\infty$',
        '$\\|e\\|_2 / \\|f\\|_2$', '$\\|e\\|_\\infty / \\|f\\|_\\infty$']
    headerstring = ''
    for j, header in enumerate(headers):
        headerstring += header
        if j < len(headers) - 1:
            headerstring += ' & '
        else:
            headerstring += '\\\\'

    tablestr = f"""\
\\begin{{table}}
    \\centering
    \\begin{{tabular}}{{{colstring}}}
        \\hline
        \\multicolumn{{3}}{{|c|}}{{Inputs}} &\
        \\multicolumn{{4}}{{|c|}}{{Statistics}} &\
        \\multicolumn{{3}}{{|c|}}{{Failure Region Errors}} &\
        \\multicolumn{{2}}{{|c|}}{{Overall Errors}} \\\\
        \\hline
        {headerstring}
        \\hline\\hline
"""
    for row in data:
        rowstr = '\t\t'
        for j, entry in enumerate(row):
            if j == 0 or j == 3:
                rowstr += str(int(entry))
            elif j == 1:
                rowstr += str(entry)
            elif j == 2:
                rowstr += str(100.0 * entry) + '\\%'
            elif j >= 4 and j <= 6:
                rowstr += str(round(100 * entry, 2)) + '\\%'
            elif j >= 7:
                if entry > 0:
                    rowstr += str(round(entry, 4))
                else:
                    rowstr += '-'

            if j < len(row) - 1:
                rowstr += ' & '
            else:
                rowstr += ' \\\\\n'
        tablestr += rowstr

    tablestr += \
"""\
        \\hline
    \\end{tabular}
    \\caption{This is a caption!}
    \\label{tab:new_tab}
\\end{table}\
"""

    print(tablestr)






# test these functions
def test_1D(N_init = 2000, atol=0.01):

    N = 4000
    domain = np.array([0, 8])
    interpolant = NearODESurrogate()
    interpolant.train_model_from_sampling(one_d_combined, domain, 1,
            N_init, 'uniform', True, N_init_clusters=20, atol=atol, alpha=100000)

    xx = np.linspace(domain[0], domain[1], N)
    yy_interpolant, bad_points = interpolant.evaluate(xx.reshape((N, 1)), k=1, disp_time=True)
    yy_interpolant = yy_interpolant.flatten()
    yy_exact = one_d_example(xx)
    error = np.abs(yy_interpolant - yy_exact)
    rmse = np.sqrt(np.sum(error ** 2) / len(error))
    sources = interpolant.get_sources()

    fig, ax = plt.subplots(1,2)

    ax[0].plot(xx, yy_exact, '-k', label='Exact')
    ax[0].plot(xx, yy_interpolant, '--m', label='Interpolant')
    ax[0].scatter(sources, one_d_example(sources), c='blue', marker='x', label=f'Sources ($N_s = {len(sources)}$)')
    ax[0].scatter(xx[bad_points], one_d_example(xx[bad_points]),
                marker='o', label=f'Unguaranteed Points ($N = {bad_points.sum()}$)', c='orange')
    ax[0].legend()
    ax[0].set_xlabel('$x$')
    
    ax[1].plot(xx, error, '--r', label='Log Error')
    ax[1].plot(xx, np.full_like(xx, atol), '--k')
    ax[1].scatter(xx[bad_points], error[bad_points], marker='o', label=f'Unguaranteed Points ($N = {bad_points.sum()}$)')
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_xlabel('$x$')

    plt.show()

    print(f'Max l-infty error: {error.max()}')
    print(f'Root mean squared error: {rmse}')



def test_2D_SHCF(domain=np.array([[-3,3],[-2,2]]), atol=0.01, rtol=0.01, N_init = 50000, min_ppc=100):

    L = check_matrix_2norm(hessian_SHCF, domain)

    interpolant = NearODESurrogate()
    SHCF_combined.calls = 0
    interpolant.train_model_from_sampling(
            SHCF_combined, domain, 1, N_init, 'LHS', True, N_init_clusters = 100,
            min_ppc = min_ppc, atol = atol, rtol = rtol,
            check_guarantee_N_times=0, L=L, seed=1234,
            use_gradient = True)
    sources = interpolant.get_sources()
    print(f'Total number of function evals for training: {SHCF_combined.calls - N_init}')

    x = np.linspace(domain[0,0], domain[0,1], 300)
    y = np.linspace(domain[1,0], domain[1,1], 300)
    xv, yv = np.meshgrid(x, y)
    pts = np.column_stack((xv.flatten(), yv.flatten()))

    zz_exact = np.expand_dims(SHCF(pts), axis=1)
    zz_interpolant, bad_pts_inds = interpolant.evaluate(pts, k=1, disp_time=True, L=L, recluster=False, alpha=1)
    abs_error = np.linalg.norm(zz_exact - zz_interpolant, 2, axis=1)
    rel_error = abs_error / np.linalg.norm(zz_exact, 2, axis=1)
    good_abs_error = abs_error <= atol
    good_rel_error = rel_error <= rtol
    good_overall_error = np.logical_or(good_abs_error, good_rel_error)
    bad_overall_error = np.logical_not(good_overall_error)
    good_pts_inds = np.logical_not(bad_pts_inds)

    # sort the points outside of guarantee between those inside the error and those not
    no_g_good_err = bad_pts_inds & good_overall_error # 1 (yellow)
    no_g_bad_err = bad_pts_inds & bad_overall_error # 2 (orange)
    # sort the points inside of guarantee as well
    yes_g_good_err = good_pts_inds & good_overall_error # 3 (green)
    yes_g_bad_err = good_pts_inds & bad_overall_error # 4 (red)
    sorted_grid = 0 * no_g_good_err + 1 * no_g_bad_err + 2 * yes_g_good_err + 3 * yes_g_bad_err

    zz_exact_reshaped = zz_exact.reshape((len(xv),len(yv)))
    abs_error_reshaped = abs_error.reshape((len(xv),len(yv)))


    ## PLOTS

    # Plot of function (for completeness)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xv, yv, zz_exact_reshaped, cmap='viridis')
    ax.set_title('Six-Hump Camelback Function')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    # Plot of absolute errors
    plt.figure()
    plt.title('Absolute Errors')
    colors = ['yellow', 'orange', 'green', 'red']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]  # one more than len(colors)
    norm = BoundaryNorm(bounds, ncolors=cmap.N)
    plt.scatter(pts[yes_g_good_err][:,0], pts[yes_g_good_err][:,1], c='green', marker='o', s=2, label=f'Pts. inside theoretical guarantee with good error (N={np.sum(yes_g_good_err)})')
    plt.scatter(pts[yes_g_bad_err][:,0], pts[yes_g_bad_err][:,1], c='red', marker='o', s=2, label=f'Pts. inside theoretical guarantee with bad error (N={np.sum(yes_g_bad_err)})')
    plt.scatter(pts[no_g_good_err][:,0], pts[no_g_good_err][:,1], c='yellow', marker='o', s=2, label=f'Pts. outside theoretical guarantee with good error (N={np.sum(no_g_good_err)})')
    plt.scatter(pts[no_g_bad_err][:,0], pts[no_g_bad_err][:,1], c='orange', marker='o', s=2, label=f'Pts. outside theoretical guarantee with bad error (N={np.sum(no_g_bad_err)})')
    plt.pcolormesh(xv, yv, sorted_grid, cmap=cmap, norm=norm)
    plt.scatter(sources[:,0], sources[:,1], c='blue', marker='o', s=2, label=f'Sources (N={len(sources)})')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(loc='upper right')
    plt.show()


    # display stats about errors
    red_abs_errs = abs_error[yes_g_bad_err]
    red_fvals = zz_exact[yes_g_bad_err]
    if len(red_abs_errs > 0):
        datavec = np.array([
            # section 1: initial values
            N_init,
            atol,
            rtol,
            # section 2: point results
            interpolant.N_sources,
            (np.sum(no_g_good_err)+np.sum(no_g_bad_err)) / len(pts),
            np.sum(yes_g_good_err) / len(pts), 
            np.sum(yes_g_bad_err) / len(pts),
            # section 3: errors for failure points
            np.linalg.norm(red_abs_errs, 2) / np.linalg.norm(red_fvals, 2), # relative l2 error
            np.linalg.norm(red_abs_errs, np.inf) / np.linalg.norm(red_fvals, np.inf), # relative l-inf error
            np.linalg.norm(red_abs_errs, 2), # absolute l2 error
            np.linalg.norm(red_abs_errs, np.inf), # absolute l-inf error
            # section 4: overall errors
            np.linalg.norm(abs_error, 2) / np.linalg.norm(zz_exact, 2), # relative l2 error (entire region)
            np.linalg.norm(abs_error, np.inf) / np.linalg.norm(zz_exact, np.inf), # relative l-inf error
            np.linalg.norm(abs_error, 2) # absolute l2 error
        ])
    else:
        datavec = np.array([
            N_init,
            atol,
            rtol,
            interpolant.N_sources,
            (np.sum(no_g_good_err)+np.sum(no_g_bad_err)) / len(pts),
            np.sum(yes_g_good_err) / len(pts), 
            np.sum(yes_g_bad_err) / len(pts),
            0,
            0,
            0,
            0,
        ])



    # Why are there STILL points that don't achieve error?
    # For points outside of tolerance, plot the function on the line
    #   segment between the point and its nearest neighbor.
    if 0:
        bad_ind = np.unravel_index(abs_error_reshaped.argmax(), abs_error_reshaped.shape)
        bad_pt = np.array([xv[bad_ind], yv[bad_ind]])
        D, I = interpolant.KNN_search._faiss_index.search(bad_pt.reshape(1,2), 1)  # noqa: E741
        bad_nn = interpolant.KNN_search.features[I.flatten()].flatten()
        print(f'Nearest neighbor: {bad_nn}')
        print(f'Point with large error: {bad_pt}')
        print(f'Global distance tolerance: {np.sqrt(2*atol/L)}')

        lambdagrid = np.linspace(0, 1, 1000)
        convexgrid = np.outer(lambdagrid, bad_pt) + np.outer((1-lambdagrid), bad_nn)
        f = SHCF(convexgrid)
        eta, _ = interpolant.evaluate(convexgrid, k=1, disp_time=False)
        plt.figure()
        plt.plot(lambdagrid, f, '-', label='Exact')
        plt.plot(lambdagrid, eta, '--', label='Interpolant')
        plt.xlim((-0.1,1.1))


    # SVD of clusters
    # Take all initial data and find nearest neighbor
    # For each cluster, put points into matrix, compute SVD
    # first, sort data
    if 0:
        grid = np.random.uniform(domain[:,0], domain[:,1], (N_init*10, 2))
        near_source_inds = interpolant.KNN_search.find_nearest_sources(grid)
        S_ratio = np.zeros(interpolant.N_sources)
        for j in range(interpolant.N_sources):
            points_in_cluster_j = grid[near_source_inds == j]
            if len(points_in_cluster_j) > 2:
                S = SVD_analysis(points_in_cluster_j)
                S_ratio[j] = S[0] / S[1]
                if S_ratio[j] > 1e6:
                    print(points_in_cluster_j)
            else:
                S_ratio[j] = -1
        plt.figure()
        plt.hist(S_ratio)
        plt.title('$\\sigma_1 / \\sigma_2$ (SVD of Clusters)')
        plt.show()


    return datavec


def convergence_test_2D_SHCF(tolmat, N_init, domain, min_ppc=40):
    convergence_test_a_function(lambda a,b,c,d: test_2D_SHCF(domain,a,b,c,d), tolmat, N_init, min_ppc)




def test_8D(atol=0.2, rtol=0.01, N_init = 10000, min_ppc = 40):
    domain = np.array([
        [0.05, 0.15],
        [100, 50000],
        [63070, 115600],
        [990, 1110],
        [63.1, 116],
        [700, 820],
        [1120, 1680],
        [9855, 12045]
    ])
    test_a_function(borehole, borehole_combined, 1, domain, atol, rtol, N_init, min_ppc)


def convergence_test_8D(tolmat, N_init, min_ppc=40):
    convergence_test_a_function(test_8D, tolmat, N_init, min_ppc)




def test_nd_dixon(d, atol=0.01, rtol=0.01, N_init = 1000000, min_ppc = 50):

    domain = np.column_stack((np.zeros(d), np.ones(d)))
    return test_a_function(dixon_price, dixon_price_combined, 1, domain, atol, rtol, N_init, min_ppc)


def convergence_test_nd_dixon(tolmat, d, N_init, min_ppc=100):
    convergence_test_a_function( \
        lambda arg1,arg2,arg3,arg4: test_nd_dixon(d,arg1,arg2,arg3,arg4), tolmat, N_init, min_ppc)



def test_2d_rosenbrock(atol=0.01, rtol=0.01, N_init = 100000, min_ppc = 50):

    domain = np.array([
        [-2, 2],
        [-1, 3]
    ])
    return test_a_function(rosenbrock, rosenbrock_combined, 1, domain, atol, rtol, N_init, min_ppc)


def convergence_test_2d_rosenbrock(tolmat, N_init, min_ppc=100):
    convergence_test_a_function( \
        lambda arg1,arg2,arg3,arg4: test_2d_rosenbrock(arg1,arg2,arg3,arg4), tolmat, N_init, min_ppc)




def test_affine(atol=0.01, rtol=0.01, N_init = 10000, min_ppc = 50):
    n = 20
    m = 10
    domain = np.column_stack((np.zeros(n), np.ones(n)))
    return test_a_function(affine, affine_combined, m, domain, atol, rtol, N_init, min_ppc)

def convergence_test_affine(tolmat, N_init, min_ppc=100):
    convergence_test_a_function(test_affine, tolmat, N_init, min_ppc)


def test_a_function(f, f_combined, m, domain, atol=0.01, rtol=0.01, N_init=10000, min_ppc=50, save_surrogates=False):

    if 1:
        interpolant = NearODESurrogate()
        interpolant.train_model_from_sampling(
                f_combined, domain, m, N_init, 'LHS', True,
                N_init_clusters=100, atol = atol, rtol = rtol, min_ppc = min_ppc,
                check_guarantee_N_times=0, disp_time = True, use_gradient=True)
    else:
        training_data = generate_lorenz_sample_data(domain, 2, 0.25)
        # TODO: need to do solves and fill in arguments below
        interpolant = NearODESurrogate()
        interpolant.train_model_from_data(training_data)
        

    
    datavec = test_surrogate_error(interpolant, f, m, domain, atol, rtol, N_init)

    if save_surrogates:
        save_surrogate(f'lorenz_surrogates/lorenz_surrogate_dt_0.25_atol_{atol}_rtol_{rtol}_F_2.pkl', interpolant)

    return datavec


def test_surrogate_error(surrogate: NearODESurrogate, f, m, domain, atol, rtol, N_init):

    # evaluate exact
    # pick some evaluation points using LHS
    N_eval = 10000
    l = domain[:,0]  # noqa: E741
    r = domain[:,1]
    sampler = qmc.LatinHypercube(d=len(domain))
    pts = sampler.random(n=N_eval)
    pts = qmc.scale(pts, l, r)

    exact = np.zeros((len(pts),m))
    for j in range(len(exact)):
        exact[j] = f(pts[j])
    interp, bad_pts = surrogate.evaluate(pts, k=1, disp_time=True)

    abs_error = np.linalg.norm(exact - interp, 2, axis=1)
    rel_error = abs_error / np.linalg.norm(exact, 2, axis=1)
    good_abs_error = abs_error <= atol
    good_rel_error = rel_error <= rtol
    good_overall_error = np.logical_or(good_abs_error, good_rel_error)
    bad_overall_error = np.logical_not(good_overall_error)
    good_pts_inds = np.logical_not(bad_pts)

    # sort the points outside of guarantee between those inside the error and those not
    no_g_good_err = bad_pts & good_overall_error # 1 (yellow)
    no_g_bad_err = bad_pts & bad_overall_error # 2 (orange)
    # sort the points inside of guarantee as well
    yes_g_good_err = good_pts_inds & good_overall_error # 3 (green)
    yes_g_bad_err = good_pts_inds & bad_overall_error # 4 (red)

    # display stats about errors
    red_abs_errs = abs_error[yes_g_bad_err]
    red_fvals = exact[yes_g_bad_err]
    if len(red_abs_errs > 0):
        datavec = np.array([
            # section 1: initial values
            N_init,
            atol,
            rtol,
            # section 2: point results
            surrogate.N_sources,
            (np.sum(no_g_good_err)+np.sum(no_g_bad_err)) / len(pts),
            np.sum(yes_g_good_err) / len(pts), 
            np.sum(yes_g_bad_err) / len(pts),
            # section 3: errors for failure points
            np.linalg.norm(red_abs_errs, 2) / np.linalg.norm(red_fvals, 2), # relative l2 error
            np.linalg.norm(red_abs_errs, np.inf) / np.linalg.norm(red_fvals, np.inf), # relative l-inf error
            np.linalg.norm(red_abs_errs, np.inf), # absolute l-inf error
            # section 4: overall errors
            np.linalg.norm(abs_error, 2) / np.linalg.norm(exact, 2), # relative l2 error (entire region)
            np.linalg.norm(abs_error, np.inf) / np.linalg.norm(exact, np.inf), # relative l-inf error
        ])
    else:
        datavec = np.array([
            N_init,
            atol,
            rtol,
            interpolant.N_sources,
            (np.sum(no_g_good_err)+np.sum(no_g_bad_err)) / len(pts),
            np.sum(yes_g_good_err) / len(pts), 
            np.sum(yes_g_bad_err) / len(pts),
            0,
            0,
            0,
            np.linalg.norm(abs_error, 2) / np.linalg.norm(exact, 2), # relative l2 error (entire region)
            np.linalg.norm(abs_error, np.inf) / np.linalg.norm(exact, np.inf), # relative l-inf error
        ])
    
    return datavec


def convergence_test_a_function(test_function, tolmat, N_init, min_ppc=100):
    datamat = np.empty((len(tolmat), 12))
    for j, tols in enumerate(tolmat):
        datamat[j] = test_function(tols[0], tols[1], N_init, min_ppc)
    
    print('\n')
    print(generate_latex_table(datamat))
    print('\n')


def test_quadratic(atol=0.01, rtol=0.01, N_init = 10000, min_ppc = 50):
    n = 5
    F = np.array([8])
    domain = np.column_stack((0.5*np.ones(n), 1.5*np.ones(n)))
    return test_a_function(lambda x: L96(x,F), lambda x: L96_combined(x,F),
            n, domain, atol, rtol, N_init, min_ppc)

def convergence_test_quadratic(tolmat, N_init, min_ppc=100):
    convergence_test_a_function(test_quadratic, tolmat, N_init, min_ppc)


def test_lorenz(n, tf, atol=0.01, rtol=0.01, N_init=100000, min_ppc=50, save_surrogates=False):
    F = 2
    domain = np.column_stack((-7*np.ones(n),7*np.ones(n)))
    return test_a_function(lambda x: solve_L96_ode(x,F,tf), lambda x: solve_L96_ode_with_sensitivities(x,F,tf),
            n, domain, atol, rtol, N_init, min_ppc, save_surrogates)

def convergence_test_lorenz(n, tf, tolmat, N_init, min_ppc=100, save_surrogates=False):
    convergence_test_a_function(lambda y1,y2,y3,y4: test_lorenz(n,tf,y1,y2,y3,y4,save_surrogates), tolmat, N_init, min_ppc, )


def generate_lorenz_sample_data(domain, F, dt):
    # pick 1000 random points
    n = len(domain)
    N = 1000
    l = domain[:,0]  # noqa: E741
    r = domain[:,1]
    sampler = qmc.LatinHypercube(d=n)
    features = sampler.random(n=N)
    features = qmc.scale(features, l, r)

    # solve ODEs for each of these on uniform time stepping grid (i.e. every 0.2 seconds)
    # take 1000 * 50 = 50000 training points, which lie on "manifolds" of solutions
    dt = 0.2
    tf = 10
    tvec = np.arange(0, tf+1e-6, dt)
    p = np.array([F])
    training_data = np.zeros(n, N * (tf / dt))
    for j, feature in enumerate(features):
        sol = solve_ivp(lambda t,x: L96(x,p), [0,tf], x0, max_step=0.1, atol=1e-6, rtol=1e-4, t_eval=tvec)
        training_data[:,j*N:(j+1)*N] = sol.x[:,1:]
    
    return training_data.T


def train_lorenz_surrogate(N_init, d, atol, F, dt):

    interpolant = NearODESurrogate()
    domain = np.column_stack((-5*np.ones(d), 5*np.ones(d)))

    # train surrogate model
    interpolant.train_model_from_sampling(
            lambda x: solve_L96_ode_with_sensitivities(x,F,dt),
            domain, d, N_init, 'LHS', True, N_init_clusters = 100,
            min_ppc = 100, atol = atol, seed=1234,
            use_gradient = True)
    
    return interpolant


def lorenz_time_stepping_test(interpolant: NearODESurrogate, d, F, dt, x0, tf):

    # Exact ODE solution
    tvec = np.arange(0, tf+1e-6, dt)
    p = np.array([F])
    sol = solve_ivp(lambda t,x: L96(x,p), [0,tf], x0, max_step=0.1, atol=1e-6, rtol=1e-4, t_eval=tvec)
    x_exact = sol.y

    # Surrogate ODE solution
    x_surrogate = np.zeros_like(x_exact)
    in_trust_region = np.full(len(tvec), False)
    x_surrogate[:,0] = x0
    
    for j in range(len(tvec)-1):
        eval, not_trust = interpolant.evaluate(np.expand_dims(x_surrogate[:,j], axis=1).T)
        in_trust_region[j+1] = np.logical_not(not_trust[0])
        if not_trust[0]:
            # solve ODE exactly, for time horizon dt with ic x_surrogate[:,j]
            sol = solve_L96_ode_return_all(x_surrogate[:,j], F, dt)
            x_surrogate[:,j+1] = sol.y[:,-1]
        else:
            # use surrogate
            x_surrogate[:,j+1] = eval.flatten()
            

    
    # Compare solutions
    print(f'Number of evaluations in trust region: {np.sum(in_trust_region)}/{len(in_trust_region)-1}')
    two_error = np.linalg.norm(x_surrogate - x_exact, 2, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(tvec, two_error)
    ax.set(xlabel='$t$', ylabel='$L^2$ distance from exact')
    ax.scatter(tvec[in_trust_region], two_error[in_trust_region], s=5, label='Evaluations in Trust Region', c='red')

    # Plot on top of each other
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x_exact[0,:], x_exact[1,:], x_exact[2,:], label='Exact Solution')
    ax.plot(x_surrogate[0,:], x_surrogate[1,:], x_surrogate[2,:], label='Surrogate Solution')
    
    plt.legend()
    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$',
           title=f'Lorenz 96 ODE predicted via surrogate, dt={dt}, tf={tf}, F={F}, d={d}')
    plt.show()






if __name__ == '__main__':
    
    tolmat = np.array([
        [0.1, 0.02],
        [0.05, 0.01],
        [0.025, 0.005],
        [0.01, 0.002],
        [0.005, 0.001]
    ])

    # 1D test
    if 0:
        test_1D(atol=0.05)

    # 2D test (SHCF)
    elif 0:
        domain = np.array([[-3, 3], [-2, 2]])
        convergence_test_2D_SHCF(tolmat, 400000, domain)

    # Rosenbrock 2D
    elif 0:
        convergence_test_2d_rosenbrock(tolmat, 400000, 100)

    # Borehole function (d=8)
    elif 0:
        # test_8D(N_init=1000000, atol=5, rtol=0.01)
        convergence_test_8D(tolmat, N_init=1000000)

    # Dixon-Price test
    elif 0:
        # domain = np.column_stack((np.zeros(d), np.ones(d)))
        # test_nd(domain, N_init = 10000, atol=0.05, rtol=0.01, min_ppc=50)
        d = 3
        convergence_test_nd_dixon(tolmat, d=d, N_init = 100000)

    # Affine function (R^n to R^m)
    elif 0:
        convergence_test_affine(tolmat, N_init=10000)

    # Quadratic function (Lorenz 96 RHS)
    elif 0:
        convergence_test_quadratic(tolmat, N_init=100000)

    # Lorenz 96 ODE solve
    elif 0:
        dt = 0.25
        tolmat = np.array([
            [0.5,  0.1],
            [0.2,  0.05],
            [0.1,  0.02],
            [0.05, 0.01],
            [0.025,0.005]
        ])
        convergence_test_lorenz(4, dt, tolmat, 20000, 100, True)


    elif 0:
        tf = 0.5
        N = 100
        nsteps = 0
        F = 0
        params = np.random.uniform(5, 10, size=(N,5))
        for row in params:
            solution = solve_L96_ode_return_all(row, F, tf)
            nsteps += solution.t.size - 1
        print(float(nsteps) / N)
        print(solution.t)
        print(solution.y)
        x = solution.y

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot(x[0,:], x[1,:], x[2,:])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")
        plt.show()



    elif 0:

        n = 4
        m = 4
        N = 100000
        F = 0
        l = np.ones(5) * 5  # noqa: E741
        r = np.ones(5) * 10

        # draw parameters
        sampler = qmc.LatinHypercube(d=n)
        features = sampler.random(n=N)
        features = qmc.scale(features, l, r)

        # do solves (maybe this isn't necessary but it's certainly easier)
        labels = np.empty((len(features),m))
        gradients = np.empty((len(features),m,n))
        for jj, feature in enumerate(features):
            labels[jj], gradients[jj] = solve_L96_ode_with_sensitivities(feature, F, 0.2)

        np.save('data/input_dt4.npy', features)
        np.save('data/output_dt4.npy', labels)
        np.save('data/gradients_dt4.npy', gradients)

    elif 1:
        d = 4
        # N_init = 100000
        atol = 0.025
        rtol = 0.005
        F = 2
        dt = 0.25
        tf = 50
        # x0 = 0.9 * np.ones(4)
        # x0[0] += 0.01
        # x0 = np.arange(4)
        x0 = np.random.uniform(-5, 5, 4)
        # interpolant = train_lorenz_surrogate(N_init, d, atol, F, dt)
        # save_surrogate(f'lorenz_surrogates/lorenz_surrogate_dt_{dt}_atol_{atol}_F_{F}.pkl', interpolant)
        interpolant: NearODESurrogate = load_surrogate(f'lorenz_surrogates/lorenz_surrogate_dt_{dt}_atol_{atol}_rtol_{rtol}_F_{F}.pkl')
        lorenz_time_stepping_test(interpolant, d, F, dt, x0, tf)

    elif 0:
        interpolant: NearODESurrogate = load_surrogate('new_folder/surrogate_dt_0.2_atol_0.1.pkl')
        KNN = interpolant.KNN_search

        hessian_order = np.argsort(KNN.hessians)

        plt.figure()
        plt.hist(KNN.trust_radii)
        plt.title('Trust radii size')

        plt.figure()
        plt.hist(KNN.n_pts_per_cluster)
        plt.title('N pts per cluster')

        fig, ax = plt.subplots()
        plt.hist(KNN.hessians, bins=np.arange(0,1.6,0.05))
        plt.title('Lipschitz constant magnitudes')
        plt.text(0.95, 0.95,
                 f'Outliers:\n{KNN.hessians[hessian_order[-1]]}\n\
                 {KNN.hessians[hessian_order[-2]]}\n\
                 {KNN.hessians[hessian_order[-3]]}\n\
                 {KNN.hessians[hessian_order[-4]]}\n',
                 ha='right', va='top', transform = ax.transAxes)

        plt.figure()
        plt.hist(KNN.tree_levels, bins=np.arange(25))
        plt.title('Tree Levels')

        plt.show()




