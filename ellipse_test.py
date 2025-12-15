import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.cluster import KMeans


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
    c = np.random.uniform(-2, 2, dimension)
    
    # compute unique positive square root
    D, U = np.linalg.eigh(A)
    B = U @ np.diag(np.sqrt(D))

    # use this to generate samples
    # first generate points in unit ball
    points = generate_random_ball(num_points, dimension, radius=size)
    # map points to ellipse
    points = (points - c) @ B.T

    return points


def generate_random_box_points_randloc(num_points, dimension, size):

    box_points = np.random.uniform(-size, size, (num_points, dimension))
    center = np.random.uniform(-2, 2, dimension)
    return box_points + center


def generate_random_box_points(num_points, domain):

    box_points = np.random.uniform(domain[:,0], domain[:,1], (num_points, len(domain)))
    return box_points






##### ELLIPSE SOLVERS



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


# Assume points are normalized
def bound_hessian_SDP(points, f, df):

    num_points, dimension = points.shape

    errors = linear_interpolant_errors_one_cluster(points, f, df)

    # formulate and solve linear optimization problem
    # build constraints and variables
    A = []
    for j in range(num_points):
        A.append(np.outer(points[j], points[j]))
    Q = cp.Variable((dimension,dimension), PSD=True)
    constraints = [
        cp.trace(A[j] @ Q) >= 2 * errors[j] for j in range(num_points)
    ]
    # solve
    prob = cp.Problem(cp.Minimize(cp.trace(Q)), constraints)
    prob.solve()

    return Q.value



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
    padding = 0.1 * range
    return np.column_stack((minimum - padding, maximum + padding))


# only supports 2d of course
def plot_contours(domain, f, ax):
    
    xmin = domain[0,0]
    xmax = domain[0,1]
    ymin = domain[1,0]
    ymax = domain[1,1]

    # 1. Define grid
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
def plot_ellipse(A, c, r, ax, color, label='', plot_axes=False):

    D, U = np.linalg.eigh(A)
    dim = len(D)
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



def normal_clustering(points, n_clusters=2):

    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(points)
    centroids = km.cluster_centers_
    return centroids, labels






##### FUNCTIONS TO RUN TESTS



def test_trust_region_via_hessian_lsq(points, f, df, hf):
    # assume each row is a point
    # f takes each row to a value
    # df takes row to row

    # first get all the f and df/dx data (for simplicity, do solves in function)
    f_points = np.zeros(len(points))
    df_points = np.zeros_like(points)
    for j, point in enumerate(points):
        f_points[j] = f(point)
        df_points[j] = df(point)

    # compute centroid and function value
    centroid = np.mean(points, axis=0)
    df_centroid = df(centroid)
    x_diffs = points - centroid
    df_diffs = df_points - df_centroid
    H_exact = hf(centroid)
    print(f'Centroid: {centroid}')
    print('True Hessian at Centroid:')
    print(H_exact)

    # solve for Hessian directly using least squares (uses SVD for pseudoinverse)
    X = x_diffs.T
    G = df_diffs.T
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    H = G @ (VT.T / Sigma) @ U.T
    H_sym = 0.5 * (H + H.T)

    print('Approximated Hessian:')
    print(H)
    print('\n')

    print('Enforcing symmetry:')
    print(H_sym)
    print('\n')

    print(f'Absolute error ||H-\\hat{{H}}||_2 = {np.linalg.norm(H_sym - H_exact)}')
    print(f'Relative error: {np.linalg.norm(H_sym - H_exact) / np.linalg.norm(H_exact)}')



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



def test_clustering_rotating_frame(points, atol, f, df):

    n_pts, dim = points.shape
    centroid = points.mean(axis=0)
    diffs = points - centroid
    errors = linear_interpolant_errors_one_cluster(points, centroid, f, df)

    # sort points
    good_error = errors <= atol
    bad_error = errors > atol

    # MVEE version: fit ellipse around acceptable points
    Q1 = fit_ellipse_SDP(diffs[good_error])

    # PCA version: uses SVD
    Q2 = fit_ellipse_PCA(diffs[good_error])

    # plot everything
    fig, ax = plt.subplots(1, 2)
    ax1 = ax[0]
    ax2 = ax[1]
    domain_of_plot = compute_domain_of_plot(points)
    plot_contours(domain_of_plot, f, ax1)
    scatter_points(points[good_error], 'black', ax1)
    scatter_points(points[bad_error], 'gray', ax1)
    plot_ellipse(Q1, centroid, 1, ax1, 'cyan', 'MVEE', plot_axes=True)
    plot_ellipse(Q2, centroid, 0.1, ax1, 'orange', 'PCA')

    # rotate coordinate frame
    D, U = np.linalg.eigh(Q2)
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
    scatter_points(cluster_centroids, 'red', ax2)


    # rotate centroids back
    cluster_centroids_original_frame = cluster_centroids @ B_inv + centroid
    scatter_points(cluster_centroids_original_frame, 'red', ax1, 'New')


    # compare to clustering straight out of the box
    # (kmeans, not QR thing Prajwal is doing, which might turn out different)
    cluster_centroids_2, _ = normal_clustering(points, n_clusters=2)
    scatter_points(cluster_centroids_2, 'magenta', ax1, 'Old')


    ax1.set_title('Metric Adaptive Clustering Test: Rosenbrock')
    ax1.legend()
    ax2.set_title('Transformed Coordinate Space (SVD)')


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
    shape = X.shape
    input_dim = shape[:-1]
    x = X[...,0]
    y = X[...,1]
    z = np.zeros(np.append((input_dim, (2,2))))
    z[...,0,0] = -4*b*(y-x**2) + 8*b*x**2 + 2
    z[...,0,1] = -4*b*x
    z[...,1,0] = -4*b*X
    z[...,1,1] = 2*b
    return z


# FUNCTION 3: higher dimensional rosenbrock
# taken from Prajwal's code
def rosenbrock_high_d(point):
    x_i = point[:-1]
    x_next = point[1:]
    return np.sum(100.0 * (x_next - x_i**2.0) ** 2.0 + (1.0 - x_i) ** 2.0)

def grad_rosenbrock_high_d(point):
    n_dim = len(point)
    J = np.zeros(n_dim)

    # The term (1 - x_i)^2 only exists for i = 0 to d-2
    J[:-1] += 2 * (point[:-1] - 1)

    # Term from 100*(x_{i+1} - x_i^2)^2 for i < d-1 (derivative w.r.t x_i)
    # This affects 0 to d-2
    J[:-1] -= 400 * point[:-1] * (point[1:] - point[:-1] ** 2)

    # Term from 100*(x_i - x_{i-1}^2)^2 for i > 0 (derivative w.r.t x_{i+1})
    # This affects 1 to d-1
    J[1:] += 200 * (point[1:] - point[:-1] ** 2)

    return J

def hessian_rosenbrock_high_d(point):
    n_dim = len(point)
    H = np.zeros((n_dim, n_dim))

    for i in range(n_dim):
        if i < n_dim - 1:
            H[i, i] += 1200 * point[i] ** 2 - 400 * point[i + 1] + 2
            H[i, i + 1] += -400 * point[i]
            H[i + 1, i] += -400 * point[i]
        else:
            H[i, i] += 200

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
    z[...,0,0] = 2*(y-1)**2 + 2*(y**2-1)**2 + 2(y**3-1)**2
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


    N = 200
    d = 2
    size = 0.1
    points = generate_random_ellipse_points(N, d, size)
    # points = generate_random_box_points_randloc(N, d, size)
    # domain = np.array([
    #     [-0.6, -0.55],
    #     [-0.2, -0.1]
    # ])
    # points = generate_random_box_points(N, domain)

    test = 'rosenbrock'

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
    test_clustering_rotating_frame(points, 0.05, f, df)
    

