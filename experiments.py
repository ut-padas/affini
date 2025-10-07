from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from nearode import NearODESurrogate
from scipy.stats import qmc

scale = -10



# One dimensional example
# x \in [0,10]
def one_d_example(x): 
    return np.sin(2*np.pi*x) / np.exp(x)

def grad_one_d_example(x):
    return (2*np.pi*np.cos(2*np.pi*x) - np.sin(2*np.pi*x)) / np.exp(x)

def one_d_combined(x):
    return one_d_example(x), grad_one_d_example(x)



# Six-Hump Camelback function
def SHCF(x: np.ndarray):
    if x.ndim == 1:
        return (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2 + scale
    elif x.ndim == 2:
        return (4 - 2.1*x[:,0]**2 + x[:,0]**4/3)*x[:,0]**2 + x[:,0]*x[:,1] + (-4 + 4*x[:,1]**2)*x[:,1]**2 + scale

def grad_SHCF(x: np.ndarray):
    if x.ndim == 1:
        return np.array([
            8*x[0] - 8.4 * x[0]**3 + 2*x[0]**5 + x[1],
            x[0] - 8*x[1] + 16*x[1]**3
        ])
    elif x.ndim == 2:
        Dxf = 8*x[:,0] - 8.4 * x[:,0]**3 + 2*x[:,0]**5 + x[:,1]
        Dyf = x[:,0] - 8*x[:,1] + 16*x[:,1]**3
        return np.column_stack((Dxf, Dyf))
    
def hessian_SHCF(x: np.ndarray):
    if x.ndim == 1:
        return np.array([
            [8 - 25.2*x[0]**2 + 10*x[0]**4, 1],
            [1, -8 + 48*x[1]**2]
        ])

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

    return g

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
    return g

def dixon_price_combined(x):
    return dixon_price(x), grad_dixon_price(x)






# test these functions
def test_1D(N_init = 2000, atol=0.01):

    N = 4000
    domain = np.array([0, 10])
    interpolant = NearODESurrogate()
    interpolant.train_model_from_sampling(one_d_combined, domain, N_init, 'uniform', True,
                                          N_init_clusters=20, atol=atol, alpha=100000)

    xx = np.linspace(domain[0], domain[1], N)
    yy_interpolant, bad_points = interpolant.evaluate(xx.reshape((N, 1)), k=1, disp_time=True)
    yy_exact = one_d_example(xx)
    error = np.abs(yy_interpolant - yy_exact)
    rmse = np.sqrt(np.sum(error ** 2) / len(error))
    sources = interpolant.get_sources()

    plt.plot(xx, yy_exact, '-k', label='Exact')
    plt.plot(xx, yy_interpolant, '--m', label=f'Interpolant')
    plt.scatter(sources, one_d_example(sources), c='orange', marker='x', label=f'Sources ($N_s = {len(sources)}$)')
    plt.scatter(xx[bad_points], one_d_example(xx[bad_points]), marker='o', label=f'Unguaranteed Points ($N = {bad_points.sum()}$)')
    plt.legend()
    plt.xlabel('$x$')
    
    plt.figure()
    plt.plot(xx, error, '--r', label='Log Error')
    plt.plot(xx, np.full_like(xx, atol), '--k')
    plt.scatter(xx[bad_points], error[bad_points], marker='o', label=f'Unguaranteed Points ($N = {bad_points.sum()}$)')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('$x$')
    plt.show()

    print(f'Max l-infty error: {error.max()}')
    print(f'Root mean squared error: {rmse}')



def test_2D(atol=0.01, rtol=0.01, N_init = 50000, domain=np.array([[-3,3],[-2,2]]), min_ppc=100):

    L = check_matrix_2norm(hessian_SHCF, domain)

    interpolant = NearODESurrogate()
    interpolant.train_model_from_sampling(
            SHCF_combined, domain, N_init, 'LHS', True, N_init_clusters = 100,
            min_ppc = min_ppc, atol = atol, rtol = rtol,
            check_guarantee_N_times=0, L=L)
    sources = interpolant.get_sources()

    x = np.linspace(domain[0,0], domain[0,1], 300)
    y = np.linspace(domain[1,0], domain[1,1], 300)
    xv, yv = np.meshgrid(x, y)
    pts = np.column_stack((xv.flatten(), yv.flatten()))

    zz_exact = SHCF(pts)
    zz_interpolant, bad_pts_inds = interpolant.evaluate(pts, k=1, disp_time=True, L=L, recluster=False, alpha=1)
    abs_error = np.abs(zz_exact - zz_interpolant)
    rel_error = abs_error / np.abs(zz_exact)

    # sort the points outside of guarantee between those inside the error and those not
    no_g_good_err = bad_pts_inds & (abs_error <= atol) # points outside guarantee, but have good error; 1
    no_g_bad_err = bad_pts_inds & (abs_error > atol) # points outside guarantee and have bad error; 2
    # sort the points inside of guarantee as well
    yes_g_good_err = np.logical_not(bad_pts_inds) & (abs_error <= atol) # 3
    yes_g_bad_err = np.logical_not(bad_pts_inds) & (abs_error > atol) # 4
    sorted_grid = 0 * no_g_good_err + 1 * no_g_bad_err + 2 * yes_g_good_err + 3 * yes_g_bad_err

    zz_exact = zz_exact.reshape((len(xv),len(yv)))
    abs_error_reshaped = abs_error.reshape((len(xv),len(yv)))
    zz_interpolant = zz_interpolant.reshape((len(xv),len(yv)))
    sorted_grid = sorted_grid.reshape((len(xv),len(yv)))


    ## PLOTS

    # Plot of function (for completeness)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xv, yv, zz_exact, cmap='viridis')
    ax.set_title('Six-Hump Camelback Function')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    # plt.show()

    # # Plot of interpolant function (for completeness)
    # plt.figure()
    # plt.pcolormesh(xv, yv, zz_interpolant, cmap='viridis')
    # plt.title(f'Interpolant, $N_s = {interpolant.N_sources}$')
    # plt.colorbar()
    # plt.scatter(sources[:,0], sources[:,1], c='red', marker='o', s=2)
    # plt.scatter(bad_pts[:,0], bad_pts[:,1], c='magenta', marker='o', s=2)
    # plt.xlabel('$x_1$')
    # plt.ylabel('$x_2$')
    # plt.show()

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
    # plt.show()
    # rel_error = abs_error / np.abs(zz_exact)

    # Plot of relative errors
    if 0:
        rel_error = abs_error / zz_exact
        plt.figure()
        plt.pcolormesh(xv, yv, rel_error, cmap='viridis')
        plt.title('Relative Errors')
        plt.colorbar()
        plt.scatter(sources[:,0], sources[:,1], c='red', marker='o', s=2)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.show()

        # Plot of relative vs absolute errors
        plt.figure()
        plt.scatter(rel_error, np.log10(abs_error), c='black', marker='x', s=2)
        plt.title('Relative vs. log Absolute Errors')
        plt.xlabel('Rel err.')
        plt.ylabel('log Abs err.')
        plt.show()




    # CHECK ACCURACY AFTER ONE RECLUSTERING
    for j in range(0):
        zz_interpolant, bad_pts_inds = interpolant.evaluate(pts, k=1, disp_time=True, L=L, recluster=True)
        sources = interpolant.get_sources()
        zz_exact = zz_exact.flatten()
        abs_error = np.abs(zz_exact - zz_interpolant)
        # sort the points outside of guarantee between those inside the error and those not
        no_g_good_err = bad_pts_inds & (abs_error <= atol) # points outside guarantee, but have good error; 1
        no_g_bad_err = bad_pts_inds & (abs_error > atol) # points outside guarantee and have bad error; 2
        # sort the points inside of guarantee as well
        yes_g_good_err = np.logical_not(bad_pts_inds) & (abs_error <= atol) # 3
        yes_g_bad_err = np.logical_not(bad_pts_inds) & (abs_error > atol) # 4
        sorted_grid = 0 * no_g_good_err + 1 * no_g_bad_err + 2 * yes_g_good_err + 3 * yes_g_bad_err
        
        zz_exact = zz_exact.reshape((len(xv),len(yv)))
        abs_error_reshaped = abs_error.reshape((len(xv),len(yv)))
        zz_interpolant = zz_interpolant.reshape((len(xv),len(yv)))
        sorted_grid = sorted_grid.reshape((len(xv),len(yv)))

        # Plot of absolute errors
        plt.figure()
        plt.title(f'Iteration {j}')
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
    print(f'# of red points: {np.sum(yes_g_bad_err)}')
    red_abs_errs = abs_error[yes_g_bad_err]
    print(f'max abs error in red points: {np.max(red_abs_errs)}')


    # Why are there STILL points that don't achieve error?
    # For points outside of tolerance, plot the function on the line
    #   segment between the point and its nearest neighbor.
    bad_ind = np.unravel_index(abs_error_reshaped.argmax(), abs_error_reshaped.shape)
    bad_pt = np.array([xv[bad_ind], yv[bad_ind]])
    D, I = interpolant.KNN_search._faiss_index.search(bad_pt.reshape(1,2), 1)
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
    plt.show()



def test_8D(atol=0.2, rtol=0.01, N_init = 10000):

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

    interpolant = NearODESurrogate()
    interpolant.train_model_from_sampling(
            borehole_combined, domain, N_init, 'LHS', True,
            N_init_clusters=30, atol = atol, rtol = rtol,
            check_guarantee_N_times=0, disp_time = True)
    
    # evaluate exact
    # pick some evaluation points using LHS
    N_eval = 10000
    l = domain[:,0]
    r = domain[:,1]
    sampler = qmc.LatinHypercube(d=len(domain))
    pts = sampler.random(n=N_eval)
    pts = qmc.scale(pts, l, r)

    exact = np.zeros(len(pts))
    for j in range(len(exact)):
        exact[j] = borehole(pts[j])
    interp, _ = interpolant.evaluate(pts, k=1, disp_time=True)

    abs_error = np.abs(exact - interp)
    print(f'Number of sources: {interpolant.N_sources}')
    print(f'Max abs. error: {abs_error.max()}')
    print(f'Avg abs. error: {abs_error.mean()}')




def test_nd(domain, atol=0.01, rtol=0.01, N_init = 1000000, min_ppc = 50):

    interpolant = NearODESurrogate()
    interpolant.train_model_from_sampling(
            dixon_price_combined, domain, N_init, 'LHS', True,
            N_init_clusters=100, atol = atol, rtol = rtol, min_ppc = min_ppc,
            check_guarantee_N_times=0, disp_time = True)
    
    # evaluate exact
    # pick some evaluation points using LHS
    N_eval = 1000
    l = domain[:,0]
    r = domain[:,1]
    sampler = qmc.LatinHypercube(d=len(domain))
    pts = sampler.random(n=N_eval)
    pts = qmc.scale(pts, l, r)

    exact = np.zeros(len(pts))
    for j in range(len(exact)):
        exact[j] = dixon_price(pts[j])
    interp, _ = interpolant.evaluate(pts, k=1, disp_time=True)

    abs_error = np.abs(exact - interp)
    print(f'Number of sources: {interpolant.N_sources}')
    print(f'Max abs. error: {abs_error.max()}')
    print(f'Avg abs. error: {abs_error.mean()}')



if __name__ == '__main__':
    
    d = 2

    # 1D test
    if d == 1:
        test_1D(atol=1e-3)

    # 2D test
    elif d == 2:
        domain = np.array([[-2, 2], [-2, 2]])
        test_2D(atol=0.05, rtol=None, N_init = 100000, domain=domain, min_ppc=40)
    
    # Borehole function (d=8)
    elif d == 8:
        test_8D(N_init=1000000, atol=1, rtol=0.1)

    # Dixon-Price test
    else:
        domain = np.column_stack((np.zeros(d), np.ones(d)))
        test_nd(domain, N_init = 1000000, atol=0.05, rtol=0.01, min_ppc=100)

    