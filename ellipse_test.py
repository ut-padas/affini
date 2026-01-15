import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.cluster import KMeans
import time
import functools
import scipy.linalg as la
from scipy.spatial import ConvexHull
import pandas as pd


#### FUNCTIONS TO GENERATE POINTS


# Generate "num_points" random points in "dimension" that have uniform
# probability over the unit ball scaled by "radius" (length of points
# are in range [0, "radius"]).
def generate_random_ball(num_points, dimension, radius=1):
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = np.random.normal(size=(dimension,num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = np.random.random(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T



def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper


def generate_random_ball_boundary(num_points, dimension):
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = np.random.normal(size=(dimension,num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    # Return the list of random (direction & length) points.
    return random_directions.T


def generate_random_ellipse_points(num_points, dimension, size):

    # generate random spd matrix A and center c
    R = np.random.uniform(-1, 1, (dimension,dimension))
    A = R.T @ R
    c = np.random.uniform(-4, 4, dimension)
    
    # compute unique positive square root
    D, U = np.linalg.eigh(A)
    B = (U * (D ** 0.5)) @ U.T

    # use this to generate samples
    # first generate points in unit ball
    points = generate_random_ball(num_points, dimension, radius=size)
    # map points to ellipse
    points = (points - c) @ B.T

    return points, c, B


def generate_random_box_points_randloc(num_points, dimension, size):

    box_points = np.random.uniform(-size, size, (num_points, dimension))
    center = np.random.uniform(-2, 2, dimension)
    return box_points + center


def generate_random_box_points(num_points, domain):

    box_points = np.random.uniform(domain[:,0], domain[:,1], (num_points, len(domain)))
    return box_points


'''
V: number of vertices of the polytope
N: number of points to use rejection sampling
d: dimension
'''
def generate_random_polytope_points(V, N, d, size):

    # Generate a bunch of points inside an ellipse
    ellipse_interior_points, c, B = generate_random_ellipse_points(N, d, size)
    # B takes points on ball to ellipse via B @ (x_ball - c)

    # Generate V points on the boundary of a ball
    ball_points = size * generate_random_ball_boundary(V, d)

    # Map these boundary points to ellipse def. by B; these will be vertices
    vertices = (ball_points - c) @ B.T

    # Compute the bounding planes of the polytope (convex hull)
    convex_hull = ConvexHull(vertices)
    equations = convex_hull.equations
    A = equations[:,:-1]
    b = equations[:,-1]

    # point is in convex hull if Ax + b < 0
    in_half_plane = ((ellipse_interior_points @ A.T) + b) < 0
    in_convex_hull = in_half_plane.all(axis=1)

    volume = convex_hull.volume
    points_in_polytope = ellipse_interior_points[in_convex_hull]
    density = len(points_in_polytope) / volume

    return points_in_polytope, volume, density
    







##### ELLIPSE SOLVERS


# @timeit
def fit_ellipse_PCA(points):

    num_points, dimension = points.shape
    # points is all the cluster points stacked into rows.
    # first normalize
    c = np.mean(points, axis=0)
    # # Do we need to also scale by standard deviation?
    # sigma = np.std(points, axis=1)
    # normalized_points = ((points.T - c) / sigma).T
    normalized_points = points - c

    # points are stacked as columns. Take SVD, a la PCA
    U, Sigma, _ = np.linalg.svd(normalized_points.T)
    # Ideally, U contains the directions of the ellipse and
    # the singular values relate to the radii.

    return (U * (Sigma ** -2)) @ U.T


# Solve for the smallest SPD matrix A such that all points reside inside the ellipse.
# In order words, solve the semidefinite programming problem:
#   min_X <I, X>
#   s.t.  <x_i * x_i^T, X> <= 1 for each x_j,
#         X >= 0.
# We use the package cvxpy to do this.
# Assume points are normalized
@timeit
def fit_ellipse_SDP(points):

    # Each row corresponds to a point of data
    num_points, dimension = points.shape
    
    # build constraints
    A = []
    for j in range(num_points):
        A.append(np.outer(points[j], points[j]))

    # define and solve CVXPY problem
    X = cp.Variable((dimension,dimension), PSD=True)
    constraints = [X >> 0]
    constraints += [
        cp.trace(A[i] @ X) <= 1 for i in range(num_points)
    ]
    prob = cp.Problem(cp.Minimize(-cp.log_det(X)), constraints)
    prob.solve()

    return X.value



# Thanks ChatGPT :D
# @timeit
# @jit(nopython=False)
def fit_ellipse_SDP_khachiyan(X, tol=1e-2, max_iter=1000):
    """
    Fast MVEE centered at zero using Cholesky-based Khachiyan updates.

    Solves:
        min_A log det(A^{-1})
        s.t. x_i^T A x_i <= 1

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Input points (assumed centered)
    """

    X = np.asarray(X)
    N, d = X.shape

    # Initialize weights
    u = np.ones(N) / N

    for i in range(max_iter):
        # M = X^T diag(u) X
        M = (X.T * u) @ X

        # Cholesky factorization
        L = np.linalg.cholesky(M)

        # Solve L y = X^T  → y = L^{-1} X^T
        # Y = lower_triangular_solve(L, X.T)
        Y = la.solve_triangular(L, X.T, lower=True)

        # Compute squared norms ||y_i||^2 = x_i^T M^{-1} x_i
        vals = np.sum(Y**2, axis=0)

        j = np.argmax(vals)
        max_val = vals[j]

        if max_val - d < tol:
            break

        step = (max_val - d) / (d * (max_val - 1))
        u *= (1 - step)
        u[j] += step

    # Final ellipsoid matrix
    M = (X.T * u) @ X
    A = np.linalg.inv(M) / d
    return A




# Assume points are normalized to the center of the ellipse
# @timeit
def fit_ellipse_PCA_scaled(points):

    num_points, dimension = points.shape
    # points is all the cluster points stacked into rows.

    U, Sigma, _ = np.linalg.svd(points.T)
    # U contains the directions of the ellipse.

    # Figure out scaling: for each direction j, project onto u_j.
    # From the collection of projected vectors, the norm of the largest vector
    # is the length of the axis.
    D = np.zeros(dimension)
    for j in range(dimension):
        scalings = U[:,j] @ (points.T)
        D[j] = np.linalg.norm(scalings, np.inf)

    return (U * (D ** -2)) @ U.T



# Assume points are normalized
@timeit
def bound_hessian_SDP(x_diffs, errors):

    num_points, dimension = x_diffs.shape

    # Normalize points

    # formulate and solve linear optimization problem
    # build constraints and variables
    A = []
    for j in range(num_points):
        A.append(np.outer(x_diffs[j], x_diffs[j]))
    Q = cp.Variable((dimension,dimension), PSD=True)
    constraints = [
        cp.trace(A[j] @ Q) >= 2 * errors[j] for j in range(num_points)
    ]
    # solve
    prob = cp.Problem(cp.Minimize(cp.trace(Q)), constraints)
    prob.solve()

    return Q.value


def solve_hessian_least_squares(points, df_points):

    # compute centroid and function value
    centroid = np.mean(points, axis=0)
    df_centroid = df(centroid)
    x_diffs = points - centroid
    df_diffs = df_points - df_centroid

    # solve for Hessian directly using Moore-Penrose psuedoinverse
    X = x_diffs.T
    G = df_diffs.T
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    Sigma_modified = np.maximum(Sigma, 1e-12)
    H = G @ (VT.T / Sigma_modified) @ U.T
    H_sym = 0.5 * (H + H.T)

    return H_sym

@timeit
def solve_hessian_least_squares_with_solves(points, df, absolute_value=False):
    # assume each row is a point
    # f takes each row to a value
    # df takes row to row

    # for simplicity, do solves here
    df_points = np.zeros_like(points)
    for j, point in enumerate(points):
        df_points[j] = df(point)

    return solve_hessian_least_squares(points, df_points, absolute_value)


def solve_hessian_least_squares_symm(x_diffs, df_diffs, absolute_value=False):
    # assume each row corresponds to a point
    # need to loop through each output variable
    if df_diffs.ndim == 2:

        XT = x_diffs
        GT = df_diffs

        # df_centroid is the gradient at mu
        C = XT.T @ XT # X*X^T
        D = XT.T @ GT # X*G^T

        # solve CH + HC = D+D^T
        H = la.solve_continuous_lyapunov(C, D + D.T)
        return H
    
    elif df_diffs.ndim == 3:

        # N is number of points, n is output dim, m is input dim
        # diffs should be size N*m
        N, n, m = df_diffs.shape
        Htensor = np.zeros((n,m,m))
        for j in range(n):
            XT = x_diffs
            GT = df_diffs[:,j,:]

            # df_centroid is the gradient at mu
            C = XT.T @ XT # X*X^T
            D = XT.T @ GT # X*G^T

            # solve CH + HC = D+D^T
            Htensor[j] = la.solve_continuous_lyapunov(C, D + D.T)
        
        H = np.mean(Htensor, axis=0)
        
    if absolute_value:
        return compute_absolute_value(H)
    else:
        return H



@timeit
def solve_hessian_least_squares_symm_with_solves(points, df):

    centroid = np.mean(points, axis=0)
    df_points = np.zeros_like(points)
    for j, point in enumerate(points):
        df_points[j] = df(point)
    df_centroid = df(centroid)

    x_diffs = points - centroid
    df_diffs = df_points - df_centroid

    return solve_hessian_least_squares_symm(x_diffs, df_diffs)



##### FUNCTIONS USED FOR PLOTTING



def scatter_points(points, color, ax, label=''):

    _, dim = points.shape

    if dim == 3:
        ax.scatter(points[:,0], points[:,1], points[:,2], marker='x', color=color, label=label)

    if dim == 2:
        ax.scatter(points[:,0], points[:,1], marker='x', color=color, label=label)
    
    ax.set_aspect('equal')


def compute_domain_of_plot(points):

    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    range = maximum - minimum
    padding = 0.2 * range
    return np.column_stack((minimum - padding, maximum + padding))


def get_domain_of_plot(ax):

    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    x_padding = 0.1 * (x_right - x_left)
    y_padding = 0.1 * (y_top - y_bottom)

    return np.array([
        [x_left - x_padding, x_right + x_padding],
        [y_bottom - y_padding, y_top + y_padding]
    ])

# only supports 2d of course
def plot_contours(domain, f, ax):
    
    xmin = domain[0,0]
    xmax = domain[0,1]
    ymin = domain[1,0]
    ymax = domain[1,1]

    # Make it a bit bigger
    factor = 0.5
    xrange = xmax - xmin
    xmax += xrange * factor
    xmin -= xrange * factor
    yrange = ymax - ymin
    ymax += yrange * factor
    ymin -= yrange * factor

    # Define grid
    x = np.linspace(xmin, xmax, 400)
    y = np.linspace(ymin, ymax, 400)
    X, Y = np.meshgrid(x, y)
    XY = np.stack((X,Y), axis=-1)
    Z = f(XY)

    # 2. Plot contour lines
    cs = ax.contourf(X, Y, Z, levels=50)
    plt.gcf().colorbar(cs, ax=ax)
    ax.axis('equal')


# Plot the ellipse defined by x^T A x <= r on the plot ax
def plot_ellipse(A, c, r, ax, color, label='', plot_axes=False, use_absolute_value=False):

    D, U = np.linalg.eigh(A)
    dim = len(D)
    if use_absolute_value:
        D = np.abs(D)
    B = U @ np.diag(D ** (-1/2)) # inverse square root

    if dim == 3:

        # plot line from c to c + U[:,j]
        if plot_axes:
            l1 = np.column_stack((c, c+(U[:,0]*D[0]**(-1/2))))
            l2 = np.column_stack((c, c+(U[:,1]*D[1]**(-1/2))))
            l3 = np.column_stack((c, c+(U[:,2]*D[2]**(-1/2))))
            ax.plot(l1[0,:], l1[1,:], l1[2,:], '-', color=color)
            ax.plot(l2[0,:], l2[1,:], l2[2,:], '-', color=color)
            ax.plot(l3[0,:], l3[1,:], l3[2,:], '-', color=color)

        # plot an ellipse in 3d (I don't care right now)

    elif dim == 2:

        # plot line from c to c + U[:,j]
        if plot_axes:
            l1 = np.column_stack((c, c+(U[:,0]*D[0]**(-1/2))))
            l2 = np.column_stack((c, c+(U[:,1]*D[1]**(-1/2))))
            ax.plot(l1[0,:], l1[1,:], '-', color=color)
            ax.plot(l2[0,:], l2[1,:], '-', color=color)

        # for 2d: plot boundary of ellipse
        ball_pts_complex = r * np.exp(1j * np.linspace(0, 2*np.pi, 200))
        ball_pts = np.vstack((np.real(ball_pts_complex), np.imag(ball_pts_complex)))
        # map to ellipse via square root
        ellipse_pts = (B @ ball_pts).T + c
        ax.plot(ellipse_pts[:,0], ellipse_pts[:,1], '--', label=label, color=color)


def plot_text_in_br_corner(string, ax):

    ax.text(0.95, 0.05, string,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform = ax.transAxes)


##### HELPER FUNCTIONS


def linear_interpolant_errors_one_cluster(points, centroid, f, df):

    # first get all the f data (for simplicity, do solves here)
    f_points = f(points)

    # compute centroid and function value (and gradient)
    f_centroid = f(centroid)
    df_centroid = df(centroid)
    x_diffs = points - centroid

    # compute exact errors from linear interpolant
    errors = np.abs(f_points - (f_centroid + (x_diffs @ df_centroid)))
    return errors



def measure_L2_linear_interpolant_error(domain, centroids, f, df):
    '''
    Approximates the L2 error of a linear interpolant of
    nearest neighbors, defined by the centroids.

    Arguments:
        - domain: where we sample points to compute the error
        - centroids: the centers for the Taylor expansions and nearest neighbor sources
        - f: the underlying function
        - df: the underlying gradient
    '''

    # make sure centroids is a matrix of points.
    # If it a single point we need to turn it into a matrix
    if centroids.ndim == 1:
        centroids = centroids[np.newaxis, ...]

    # generate points from uniformly sampled grid
    N = 1000
    test_points = generate_random_box_points(N, domain)

    # compute L2 distances squared
    diff = centroids[:, None, :] - test_points[None, :, :]
    distances = np.sum(diff**2, axis=-1)

    # take minimum row-wise to determine index of nearest neighbor
    nn_indices = np.argmin(distances, axis=0)

    # compute error for each local linear interpolant
    errorsum = 0
    for i, centroid in enumerate(centroids):
        errors = linear_interpolant_errors_one_cluster(test_points[nn_indices==i], centroid, f, df)
        errorsum += np.sum(errors ** 2)
    
    # last, use this to approximate L2 error via Monte Carlo
    error = np.sqrt(errorsum / N)
    return error



# Assumes points are centered at ellipse centroid defined by Q
def metric_adaptive_clustering(points, Q, n_clusters=2):

    # eigenvalue decomposition to compute square root
    D, U = np.linalg.eigh(Q)
    D = np.maximum(D, 1e-10)
    # Q should be psd, so eigenvalues should be positive.
    # But if points fall approximately in a line, then one singular value is zero.
    # This causes major problems when computing D ** -0.5.
    # TODO: how to fix clusters that are approximately lines?
    B = U * (D ** 0.5) @ U.T 
    B_inv = U * (D ** -0.5) @ U.T

    # rotate points
    rotated_points = points @ B

    # cluster in new space
    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(rotated_points)
    centroids = km.cluster_centers_

    # rotate clusters back
    centroids_original_frame = centroids @ B_inv
    return centroids_original_frame, labels


# Includes a "soft factor" which determines what percentage to blur points
def metric_adaptive_two_means_PCA_clustering(points, Q, soft_factor=0):
    
    # eigenvalue decomposition to compute square root
    D, U = np.linalg.eigh(Q)
    D = np.maximum(D, 1e-10)
    # Q should be psd, so eigenvalues should be positive.
    # But if points fall approximately in a line, then one singular value is zero.
    # This causes major problems when computing D ** -0.5.
    # TODO: how to fix clusters that are approximately lines?
    B = U * (D ** 0.5) @ U.T 

    # rotate points
    rotated_points = points @ B

    # cluster in new space; use PCA (Ding et.al 2004)
    U, _, _ = np.linalg.svd(rotated_points.T)
    u1 = U[0]

    # project points onto v1
    indvec_continuous = rotated_points @ u1

    # normalize lengths so that values are in [-1,1]
    max_indvec = np.linalg.norm(indvec_continuous, ord=np.inf)

    indvec_continuous = indvec_continuous / max_indvec

    # determine membership
    if soft_factor == 0:
        indvec_discrete = np.astype(indvec_continuous >= 0, int)
    else:
        in_first_cluster = indvec_continuous > soft_factor
        in_second_cluster = indvec_continuous < -soft_factor
        in_both_clusters = np.logical_and((indvec_continuous >= -soft_factor), (indvec_continuous <= soft_factor))
        indvec_discrete = 0.0 * in_first_cluster + 0.5 * in_both_clusters + 1.0 * in_second_cluster

    # determine new centroids
    c0 = np.mean(points[indvec_discrete <= 0.5], axis=0)
    c1 = np.mean(points[indvec_discrete >= 0.5], axis=0)

    centroids_original_frame = np.array([c0, c1])
    return centroids_original_frame, indvec_discrete




def normal_clustering(points, n_clusters=2):

    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(points)
    centroids = km.cluster_centers_
    return centroids, labels



def lower_triangular_solve(A: np.ndarray, B: np.ndarray):

    # Sizes of matrices A (lower triangular) and B
    mA, nA = np.shape(A)
    mB, nB = np.shape(B)

    # Output
    X = np.empty((mB, nB))

    # Loop over rows
    for j in range(mA):
        X[j,:] = (B[j,:] - (A[j,1:j] @ X[1:j,:])) / A[j,j]

    return X


def compute_absolute_value(Q: np.ndarray):
    D, U = np.linalg.eigh(Q)
    return (U * np.abs(D)) @ U.T





##### FUNCTIONS TO GENERATE FIGURES

def ellipse_comparison_figure(points, atol, f, df, hf):

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots(2,2)

    n_pts, dim = points.shape
    centroid = points.mean(axis=0)
    diffs = points - centroid
    errors = linear_interpolant_errors_one_cluster(points, centroid, f, df)

    # sort points
    good_error = errors <= atol
    bad_error = errors > atol

    Qvec = np.zeros((4,dim,dim))

    # a) No new metric (Q=I)
    Qvec[0] = np.eye(dim)

    # b) Least squares
    hessian_lsq = solve_hessian_least_squares_symm_with_solves(points, df)
    Qvec[1] = compute_absolute_value(hessian_lsq)

    # c) Solve SDP PCA version: uses SVD
    Qvec[2] = bound_hessian_SDP(diffs, errors)

    # d) Shape-aligned ellipse via PCA
    Qvec[3] = fit_ellipse_PCA(diffs[good_error])

    # e) Sanity check: compute true abs hessian
    true_hessian = compute_absolute_value(hf(centroid))
    print(f'True abs. Hessian:\n{true_hessian}')
    print(f'SDP abs. Hessian:\n{Qvec[2]}')
    print(f'Least Squares abs. Hessian:\n{Qvec[1]}')


    # figure (a): plot each ellipse in one figure. Write down linear interpolant error
    domain_of_plot = compute_domain_of_plot(points)
    plot_contours(domain_of_plot, f, ax1)
    scatter_points(points[good_error], 'black', ax1)
    scatter_points(points[bad_error], 'gray', ax1)

    plot_ellipse(Qvec[1], centroid, 1, ax1, 'orange', '(b) LSQ', plot_axes=False, use_absolute_value=True)
    plot_ellipse(Qvec[2], centroid, 1, ax1, 'cyan', '(c) SDP', plot_axes=False)
    plot_ellipse(Qvec[3], centroid, 1, ax1, 'lime', '(d) PCA', plot_axes=False)
    precluster_errors = np.sum(errors)
    plot_text_in_br_corner(f'$E_1 = {precluster_errors:.3f}$', ax1)
    ax1.legend()

    # figure (b): plot the result of clustering with a matrix Q
    for i in range(4):

        ax_cur = ax2[i//2,i%2]

        # cluster
        new_centroids_old_frame, labels = metric_adaptive_two_means_PCA_clustering(diffs, Qvec[i], 0)
        centroids = new_centroids_old_frame + centroid

        # plot points
        plot_contours(domain_of_plot, f, ax_cur)
        scatter_points(points[labels==0], 'pink', ax_cur)
        scatter_points(points[labels==1], 'lime', ax_cur)
        scatter_points(centroids, 'red', ax_cur)

        # compute and display error
        error_c1 = linear_interpolant_errors_one_cluster(points[labels==0], centroids[0], f, df)
        error_c2 = linear_interpolant_errors_one_cluster(points[labels==1], centroids[1], f, df)
        total_error = error_c1.sum() + error_c2.sum()
        plot_text_in_br_corner(f'$E = {total_error:.3f}$', ax_cur)
        
        # set title
        if i == 0:
            ax_cur.set_title('(a) No metric change')
        elif i == 1:
            ax_cur.set_title('(b) Least Squares')
        elif i == 2:
            ax_cur.set_title('(c) Semi-definite program')
        elif i == 3:
            ax_cur.set_title('(d) Shape-aligned ellipse via PCA')

    plt.show()




##### FUNCTIONS TO RUN TESTS


def test_trust_region_via_hessian_lsq(points, f, df, hf):
    
    H = solve_hessian_least_squares_with_solves(points, df)
    c = points.mean(axis=0)
    H_exact = hf(c)

    print('Approximated Hessian:')
    print(H)
    print('\n')

    print(f'Absolute error ||H-\\hat{{H}}||_2 = {np.linalg.norm(H - H_exact)}')
    print(f'Relative error: {np.linalg.norm(H - H_exact) / np.linalg.norm(H_exact)}')



def test_trust_region_via_hessian_sdp(points, atol, f, df):
    '''
    Fit psd matrix Q for trust region in cluster
    We assume that points is a matrix whose points form a cluster
    Do linear fit at centroid, measure error. Then fit MVEE over
    the points where the error is acceptable.

    Arguments:
        - points: the matrix of N points in cluster
            (size N * d, each row is point)
        - eps: the absolute tolerance
        - f: the underlying function (maps R^n to R)
        - df: the gradient function (maps R^n to R^n)
        - hf: the Hessian (for true computation)
    '''

    centroid = points.mean(axis=0)
    x_diffs = points - centroid
    Q = bound_hessian_SDP(x_diffs, f, df)
    dim = len(Q)
    errors = linear_interpolant_errors_one_cluster(points, f, df)
    
    # For comparison, compute old trust radius as well
    L_old = np.max(2 * errors / (np.linalg.norm(x_diffs, axis=1) ** 2))

    # if d=2, plot contour
    if dim == 2:
        
        fig, ax = plt.subplots()

        # Contour plot
        domain = compute_domain_of_plot(points)
        plot_contours(domain, f, ax)

        # Plot points
        # Filter points if the error is good for the linear interpolant at centroid
        within_error = errors <= atol
        not_within_error = np.logical_not(within_error)
        scatter_points(points[within_error], 'black', ax)
        scatter_points(points[not_within_error], 'grey', ax)
        scatter_points(centroid, 'red', ax)

        # Plot scaled ellipse in aqua (trust region)
        r = np.sqrt(2 * atol)
        plot_ellipse(
            Q,
            centroid,
            r,
            ax,
            'cyan',
            label='New trust region, SDP (absolute)'
        )

        # Plot old ball trust region
        plot_ellipse(
            np.diag(L_old * np.ones(dim)),
            centroid,
            r,
            ax,
            'orange',
            label='Old trust region (absolute)'
        )

        fig.legend()
        plt.show()



def test_clustering_rotating_frame(points, atol, f, df, hf):

    n_pts, dim = points.shape
    centroid = points.mean(axis=0)
    diffs = points - centroid
    errors = linear_interpolant_errors_one_cluster(points, centroid, f, df)

    # sort points
    good_error = errors <= atol
    bad_error = errors > atol

    # MVEE version: fit ellipse around acceptable points
    Q1 = fit_ellipse_SDP(diffs[good_error])
    Q1 = fit_ellipse_SDP_khachiyan(diffs[good_error])

    # SDP version (bounds the Hessian)
    Q2 = bound_hessian_SDP(diffs, errors)

    # PCA version: uses SVD
    Q3 = fit_ellipse_PCA(diffs[good_error])

    # Least squares version
    Q4 = solve_hessian_least_squares_symm_with_solves(points, df)
    Q4_abs = compute_absolute_value(Q4)

    # How close are Q2 and Q4 to the true Hessian?
    Q_true_abs = compute_absolute_value(hf(centroid))
    print(f'True abs. Hessian:\n{Q_true_abs}')
    print(f'SDP abs. Hessian:\n{Q2}')
    print(f'Least Squares abs. Hessian:\n{Q4_abs}')

    # plot everything
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    # plot_ellipse(Q1, centroid, 1, ax1, 'cyan', 'MVEE', plot_axes=True)
    plot_ellipse(Q2, centroid, 1, ax1, 'orange', 'SDP (Hessian bound)', plot_axes=True)
    plot_ellipse(Q3, centroid, 1, ax1, 'cyan', 'PCA', plot_axes=True)
    plot_ellipse(Q4, centroid, 1, ax1, 'lime', 'Least Squares', plot_axes=True, use_absolute_value=True)

    domain_of_plot = get_domain_of_plot(ax1)
    plot_contours(domain_of_plot, f, ax1)
    scatter_points(points[good_error], 'black', ax1)
    scatter_points(points[bad_error], 'gray', ax1)

    Q = Q3 # the specific psd matrix used to rotate

    # rotate coordinate frame
    D, U = np.linalg.eigh(Q)
    # go from ellipse to unit ball via y = B @ (x-c)
    B = U * (D ** 0.5) @ U.T 
    rotated_points = diffs @ B
    # go from unit ball to ellipse via (B_inv @ y) + c
    B_inv = U * (D ** -0.5) @ U.T

    scatter_points(rotated_points[good_error], 'black', ax2)
    scatter_points(rotated_points[bad_error], 'grey', ax2)
    plot_ellipse(np.eye(dim), np.zeros(2), 1, ax2, 'cyan')


    # cluster in rotated coordinates
    n_clusters = 2
    km = KMeans(n_clusters=n_clusters)
    km.fit(rotated_points)
    cluster_centroids = km.cluster_centers_
    scatter_points(cluster_centroids, 'red', ax2, 'Clusters')


    # rotate centroids back
    cluster_centroids_original_frame = cluster_centroids @ B_inv + centroid
    scatter_points(cluster_centroids_original_frame, 'red', ax1, 'New')


    # compare to clustering straight out of the box
    cluster_centroids_2, _ = normal_clustering(points, n_clusters=2)
    scatter_points(cluster_centroids_2, 'magenta', ax1, 'Old')


    ax1.set_title('Metric Adaptive Clustering Test: Rosenbrock')
    ax1.legend()
    ax2.set_title('Transformed Coordinate Space')


    # I wrote a function to do all of this nicely (to plug into algorithm). Here is how to use:
    # Although probably instead of taking Q, it should take U and D so we don't have to decompose each time
    # centroids_unscaled, _ = metric_adaptive_clustering(diffs, Q2, n_clusters=2)
    # cluster_centroids_original_frame_2 = centroids_unscaled + centroid


    # Last, approximate l2 error over domain (Monte-Carlo)
    domain = np.column_stack((np.min(points, axis=0), np.max(points, axis=0)))
    error_one_cluster = measure_L2_linear_interpolant_error(domain, centroid, f, df)
    error_two_clusters_old = measure_L2_linear_interpolant_error(domain, cluster_centroids_2, f, df)
    error_two_clusters_new = measure_L2_linear_interpolant_error(domain, cluster_centroids_original_frame, f, df)

    # display results
    print(f'\nL2 error from one cluster: {error_one_cluster}')
    print(f'L2 error from two clusters, old scheme: {error_two_clusters_old}')
    print(f'L2 error from two clusters, new scheme: {error_two_clusters_new}')
    plt.show()




'''
Compare using PCA (shape Hessian) to using least squares to approximate Hessian
Do not worry about figures as we would like these tests to be valid in high dimensions
'''
def compare_pca_vs_lsq_hessian(f, df, hf, d):

    N_tests = 10000
    atol = 0.05

    table_data = np.empty((N_tests, 7))
    # first column is # points
    # second column is point density (#points/unit volume)
    # third column is original cluster linear errors
    # fourth column is pca linear errors
    # fifth column is lsq linear errors
    # sixth column is theoretically optimal linear errors

    # do a bunch of tests at once
    for j in range(N_tests):

        # generate points in a polytope
        V = np.random.randint(15, 25)
        N_points = np.random.randint(20000, 30000)

        points, volume, density = generate_random_polytope_points(V, N_points, d, size)
        centroid = points.mean(axis=0)
        df_centroid = df(centroid)

        # solves for points
        f_points = f(points)
        df_points = df(points)
        for k in range(len(points)):
            f_points[k] = f(points[k])
            df_points[k] = df(points[k])

        diffs = points - centroid
        df_diffs = df_points - df_centroid

        # with these points, compute the linearization error
        errors = linear_interpolant_errors_one_cluster(points, centroid, f, df)
        good_error = errors <= atol

        # If there are not enough points to measure, then pca fails.
        while np.sum(good_error) <= d:
            atol *= 2
            good_error = errors <= atol

        # compute PCA and LSQ ellipses
        Q_pca = fit_ellipse_PCA(diffs[good_error])
        Q_lsq = solve_hessian_least_squares_symm(diffs, df_diffs, True)
        Q_exact = compute_absolute_value(hf(centroid))

        # do clustering and measure errors

        # first for pca
        centroids_scaled, labels_PCA = metric_adaptive_two_means_PCA_clustering(diffs, Q_pca, 0)
        centroids_PCA = centroids_scaled + centroid

        error_c1 = linear_interpolant_errors_one_cluster(points[labels_PCA==0], centroids_PCA[0], f, df)
        error_c2 = linear_interpolant_errors_one_cluster(points[labels_PCA==1], centroids_PCA[1], f, df)
        total_error_PCA = error_c1.sum() + error_c2.sum()

        # second for lsq
        centroids_scaled, labels_LSQ = metric_adaptive_two_means_PCA_clustering(diffs, Q_lsq, 0)
        centroids_LSQ = centroids_scaled + centroid

        error_c1 = linear_interpolant_errors_one_cluster(points[labels_LSQ==0], centroids_LSQ[0], f, df)
        error_c2 = linear_interpolant_errors_one_cluster(points[labels_LSQ==1], centroids_LSQ[1], f, df)
        total_error_LSQ = error_c1.sum() + error_c2.sum()

        # third for exact hessian
        centroids_scaled, labels_exact = metric_adaptive_two_means_PCA_clustering(diffs, Q_exact, 0)
        centroids_exact = centroids_scaled + centroid

        error_c1 = linear_interpolant_errors_one_cluster(points[labels_exact==0], centroids_exact[0], f, df)
        error_c2 = linear_interpolant_errors_one_cluster(points[labels_exact==1], centroids_exact[1], f, df)
        total_error_exact = error_c1.sum() + error_c2.sum()

        # save results in table
        table_data[j, 0] = len(points)
        table_data[j, 1] = volume
        table_data[j, 2] = density
        table_data[j, 3] = errors.sum()
        table_data[j, 4] = total_error_PCA
        table_data[j, 5] = total_error_LSQ
        table_data[j, 6] = total_error_exact

    # Create pandas table for visualization
    error_table = pd.DataFrame(table_data, columns=[
        '# Points',
        'Volume',
        'Density',
        'Before Splitting Error',
        'PCA Errors',
        'LSQ Errors',
        '"Optimal" Errors'
    ])
    print(error_table)

    # What other statistics can we compute?

    # Count the number of instances where LSQ is less than PCA
    num_LSQ_better = np.sum(table_data[:,5] < table_data[:,4])
    num_PCA_better = np.sum(table_data[:,4] < table_data[:,5])
    num_same = np.sum(table_data[:,4] == table_data[:,5])
    print(f'Number of instances least squares performs better: {num_LSQ_better}/{N_tests}')
    print(f'Number of instances PCA performs better: {num_PCA_better}/{N_tests}')
    print(f'Number of instances where they perform the same: {num_same}')

    # Plot density of points vs which error is better
    amount_LSQ_better = table_data[:,4] - table_data[:,5]

    # figure 1: density vs errors
    fig, ax = plt.subplots()
    ax.scatter(table_data[:,2], amount_LSQ_better, s=5, c='k', marker='x')
    ax.set_xlabel('Density (#Points/unit volume)')
    ax.set_ylabel('$E\_\{pca\} - E\_\{lsq\}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Rosenbrock 4D')

    # figure 2: volume vs errors
    fig, ax = plt.subplots()
    ax.scatter(table_data[:,1], amount_LSQ_better, s=5, c='k', marker='x')
    ax.set_xlabel('Volume')
    ax.set_ylabel('$E\_\{pca\} - E\_\{lsq\}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Rosenbrock 4D')

    plt.show()



##### FUNCTIONS



# FUNCTION 1: just a parabola
# assumes everything is fine, last axis is variables
def parabola(X):
    return a * np.sum(X ** 2, axis=-1)

def grad_parabola(X):
    return 2 * a * X

def hessian_parabola(X):
    shape = X.shape
    return a * np.diag(2.0 * np.ones(shape[-1]))


# FUNCTION 2: Rosenbrock
def rosenbrock(X):
    x = X[...,0]
    y = X[...,1]
    return (a-x) ** 2 + b * (y - x**2)**2

def grad_rosenbrock(X):
    x = X[...,0]
    y = X[...,1]
    z = np.zeros_like(X)
    z[...,0] = -2*(a-x) - 4*b*x*(y-x**2)
    z[...,1] = 2*b*(y-x**2)
    return z

def hessian_rosenbrock(X):
    shape = np.array(X.shape)
    input_dim = shape[:-1]
    x = X[...,0]
    y = X[...,1]
    z = np.zeros(np.concat((input_dim, (2,2))))
    z[...,0,0] = -4*b*(y-x**2) + 8*b*x**2 + 2
    z[...,0,1] = -4*b*x
    z[...,1,0] = -4*b*x
    z[...,1,1] = 2*b
    return z


# FUNCTION 3: higher dimensional rosenbrock
# taken from Prajwal's code
"""Rosenbrock function and its derivatives."""
def rosenbrock_high_d(X):
    """
    Computes the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix

    Returns:
        F: (n, 1) function values at X
    """
    x_i = X[..., :-1]
    x_next = X[..., 1:]
    F = np.sum(
        100.0 * (x_next - x_i**2.0) ** 2.0 + (1.0 - x_i) ** 2.0, axis=-1
    )
    return F


def grad_rosenbrock_high_d(X):
    """
    Computes the gradient of the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix

    Returns:
        J: (n, 1, d) Jacobian at X
    """
    J = np.zeros(X.shape, dtype=X.dtype)

    # The term (1 - x_i)^2 only exists for i = 0 to d-2
    J[..., :-1] += 2 * (X[..., :-1] - 1)

    # Term from 100*(x_{i+1} - x_i^2)^2 for i < d-1 (derivative w.r.t x_i)
    # This affects 0 to d-2
    J[..., :-1] -= 400 * X[..., :-1] * (X[..., 1:] - X[..., :-1] ** 2)

    # Term from 100*(x_i - x_{i-1}^2)^2 for i > 0 (derivative w.r.t x_{i+1})
    # This affects 1 to d-1
    J[..., 1:] += 200 * (X[..., 1:] - X[..., :-1] ** 2)

    return J


def hessian_rosenbrock_high_d(X: np.ndarray):
    """
    Computes the Hessian of the Rosenbrock function for given input points.

    Args:
        X: (n, d) data matrix

    Returns:
        H: (n, d, d) Hessian at X
    """
    n_dim = X.shape[-1]
    H = np.zeros(np.concat((X.shape, np.array([n_dim]))))

    for i in range(n_dim):
        if i < n_dim - 1:
            H[..., i, i] += 1200 * X[..., i] ** 2 - 400 * X[..., i + 1] + 2
            H[..., i, i + 1] += -400 * X[..., i]
            H[..., i + 1, i] += -400 * X[..., i]
        else:
            H[..., i, i] += 200

    return H


# FUNCTION 4: Beale function
def beale(X):
    x = X[...,0]
    y = X[...,1]
    return (1.5 - x + x*y)**2 + (2.25 - x * x*(y**2))**2 + (2.625 - x + x*(y**3))**2

def grad_beale(X):
    x = X[...,0]
    y = X[...,1]
    z = np.zeros_like(X)
    z[...,0] = 2*(1.5-x+x*y)*(y-1) + 2*(2.25-x+x*(y**2))*(y**2-1) + 2*(2.625-x+x*y**3)*(y**3-1)
    z[...,1] = 2*(1.5-x+x*y)*x     + 2*(2.25-x+x*(y**2))*(2*x*y)  + 2*(2.625-x+x*y**3)*(3*x*y**2)
    return z

def hessian_beale(X):
    shape = X.shape
    input_dim = shape[:-1]
    x = X[...,0]
    y = X[...,1]
    z = np.zeros(np.append((input_dim, (2,2))))
    z[...,0,0] = 2*(y-1)**2 + 2*(y**2-1)**2 + 2*(y**3-1)**2
    z[...,0,1] = 4*x*(y**2-1)*y + 4*y*(x*y**2-x+2.25) + 6*x*(y**3-1)*y**2 + 6*y**2*(x*y**3-x+2.625) + 2*x*(y-1) + 2*(x*y-x+1.5)
    z[...,1,0] = 4*x*(y**2-1)*y + 4*y*(x*y**2-x+2.25) + 6*x*(y**3-1)*y**2 + 6*y**2*(x*y**3-x+2.625) + 2*x*(y-1) + 2*(x*y-x+1.5)
    z[...,1,1] = 18*x**2*y**4 + 8*x**2*y**2 + 2*x**2 + 12*x*y*(x*y**3-x+2.625) + 4*x*(x*y**2-x+2.25)
    return z


# FUNCTION 5: Goldstein-Price
def goldstein_price(X):
    x = X[...,0]
    y = X[...,1]
    return (1 + (x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))





if __name__ == '__main__':

    N = 5000
    d = 2
    size = 0.2
    # points = generate_random_box_points_randloc(N, d, size)
    # points = generate_random_box_points_randloc(N, d, size)
    # domain = np.array([
    #     [-0.6, -0.55],
    #     [-0.2, -0.1]
    # ])
    # points = generate_random_box_points(N, domain)
    V = 5
    points = generate_random_polytope_points(V, N, d, size)

    test = 'rosenbrock_high_d'

    if test == 'parabola':
        a = 100
        f = parabola
        df = grad_parabola
        hf = hessian_parabola
    elif test == 'rosenbrock':
        a = 1
        b = 100
        f = rosenbrock
        df = grad_rosenbrock
        hf = hessian_rosenbrock
    elif test == 'rosenbrock_high_d':
        f = rosenbrock_high_d
        df = grad_rosenbrock_high_d
        hf = hessian_rosenbrock_high_d
    elif test == 'beale':
        f = beale
        df = grad_beale
        hf = hessian_beale
    
    # fit_ellipsoid_trust_region(ellipse_points, 0.01, 0.01, f, df, hf)
    # test_clustering_rotating_frame(points, 0.05, f, df, hf)
    # ellipse_comparison_figure(points, 0.05, f, df, hf)
    compare_pca_vs_lsq_hessian(f, df, hf, 6)
