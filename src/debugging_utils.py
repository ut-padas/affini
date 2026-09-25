from src import IS_CUPY_BACKEND, Array, xp, _to_host
from typing import Callable
from src.adaptive_clustering_utils import \
    get_n_smallest_per_block, \
    run_fused_stats, \
    run_fused_cross_cov, \
    solve_regularized_covariance_inplace
import matplotlib.pyplot as plt
import numpy as np


def compute_and_plot_relative_errors_per_row(
    exact_fn: Callable,
    eval_pts: Array,
    approx: Array,
):

    exact = exact_fn(eval_pts)
    rel_err = xp.linalg.norm(exact - approx, axis=1) / xp.linalg.norm(exact, axis=1)
    plt.figure()
    minimum = _to_host(min(rel_err))
    maximum = _to_host(max(rel_err))
    width = 0.05
    bins = np.arange(minimum, maximum + width, width)
    plt.hist(_to_host(rel_err), bins=bins, edgecolor='black')
    plt.savefig("rel_err_histogram.png", dpi=200)


def compute_and_plot_hessian_errors(exact_H_fn, eval_pts, approx_H, fn_name, method):
    
    # Exact evaluation
    exact_H = _to_host(exact_H_fn(eval_pts)) # (N,d,d)
    approx_H = _to_host(approx_H)

    # Fix scaling (since Hessian can be scaled multiple of true value)
    eigvals_exact, _ = np.linalg.eigh(exact_H)
    eigvals_approx, _ = np.linalg.eigh(approx_H)
    exact_scaling = eigvals_exact[:,-1]
    approx_scaling = eigvals_approx[:,-1]
    exact_H = exact_H / exact_scaling[:,None,None]
    approx_H = approx_H / approx_scaling[:,None,None]
    
    rel_err = xp.linalg.norm(exact_H - approx_H, axis=(1,2)) / (xp.linalg.norm(exact_H, axis=(1,2)) + 1e-2) # frobenius by default
    plt.figure()
    minimum = _to_host(min(rel_err))
    maximum = _to_host(max(rel_err))
    width = 0.03
    bins = np.arange(minimum, maximum + width, width)
    plt.hist(_to_host(rel_err), bins=bins, edgecolor='black')
    plt.xlabel("Relative Error $\\|H - \\hat{H}\\|_F / \\|H\\|_F$")
    plt.ylabel("Frequency")
    plt.title(f"Relative errors of Hessian; function {fn_name}, method {method}")
    plt.savefig(f"hessian_rel_err_{fn_name}_{method}.png", dpi=200)



def plot_cluster_points(points, center, g_approx, f_fn, g_fn, H_fn):

    f_center = f_fn(center[None,:]).flatten()
    g_exact = g_fn(center[None,:]).flatten()
    f_pts = f_fn(points).flatten()
    dx = points - center

    # Compute errors
    inexact_taylor = f_center + dx @ g_approx
    exact_taylor = f_center + dx @ g_exact
    inexact_taylor_error = np.abs(f_pts - inexact_taylor)
    exact_taylor_error = np.abs(f_pts - exact_taylor)

    # Sort
    inexact_order = np.argsort(inexact_taylor_error)
    inexact_mask = inexact_order[:50]
    exact_order = np.argsort(exact_taylor_error)
    exact_mask = exact_order[:50]

    # Plot on two axes
    fig, ax = plt.subplots(1, 2, figsize=(12,8))
    # Background grid
    xmin = min(points[:,0]); xmax = max(points[:,0])
    ymin = min(points[:,1]); ymax = max(points[:,1])
    xpad = 0.05 * (xmax-xmin); ypad = 0.05 * (ymax-ymin)
    _n = 400
    x = np.linspace(xmin-xpad, xmax+xpad, _n)
    y = np.linspace(ymin-ypad, ymax+ypad, _n)
    X_grid, Y_grid = np.meshgrid(x, y)
    grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    Z_grid = f_fn(grid_pts).reshape(X_grid.shape)
    ax[0].contourf(
        X_grid,
        Y_grid,
        Z_grid,
        cmap="viridis",
    )
    ax[1].contourf(
        X_grid,
        Y_grid,
        Z_grid,
        cmap="viridis",
    )
    # Plot points
    ax[0].scatter(points[:,0], points[:,1], c='black', s=20)
    ax[1].scatter(points[:,0], points[:,1], c='black', s=20)
    ax[0].scatter(points[inexact_mask,0], points[inexact_mask,1], c='red', s=30)
    ax[1].scatter(points[exact_mask,0], points[exact_mask, 1], c='blue', s=30)
    ax[0].scatter(center[0], center[1], c='pink', s=50, marker='x')
    ax[1].scatter(center[0], center[1], c='pink', s=50, marker='x')

    # Labels
    ax[0].set_title("Inexact Taylor ellipse (gradient fit)")
    ax[1].set_title("Exact Taylor ellipse (exact gradient)")

    # Save figure
    fig.savefig("cluster.png", dpi=300)



'''
index = 69
points = X_bad[bad_starts[index]:bad_starts[index+1],:]
center = mus_bad[index]
g_approx = A_tangent[index,0,:]
from src.problems.wave import wave, wave_grad, wave_hess
import matplotlib.pyplot as plt
f_fn = wave; g_fn = wave_grad; H_fn = wave_hess

f_center = f_fn(center[None,:]).flatten()
g_exact = g_fn(center[None,:]).flatten()
f_pts = f_fn(points).flatten()
dx = points - center

# Compute errors
inexact_taylor = f_center + dx @ g_approx
exact_taylor = f_center + dx @ g_exact
inexact_taylor_error = np.abs(f_pts - inexact_taylor)
exact_taylor_error = np.abs(f_pts - exact_taylor)

# Sort
inexact_order = np.argsort(inexact_taylor_error)
inexact_mask = inexact_order[:50]
exact_order = np.argsort(exact_taylor_error)
exact_mask = exact_order[:50]

# Plot on two axes
fig, ax = plt.subplots(1, 2, figsize=(12,8))
# Background grid
xmin = min(points[:,0]); xmax = max(points[:,0])
ymin = min(points[:,1]); ymax = max(points[:,1])
xpad = 0.05 * (xmax-xmin); ypad = 0.05 * (ymax-ymin)
_n = 400
x = np.linspace(xmin-xpad, xmax+xpad, _n)
y = np.linspace(ymin-ypad, ymax+ypad, _n)
X_grid, Y_grid = np.meshgrid(x, y)
grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
Z_grid = f_fn(grid_pts).reshape(X_grid.shape)
ax[0].contourf(
    X_grid,
    Y_grid,
    Z_grid,
    cmap="viridis",
)
ax[1].contourf(
    X_grid,
    Y_grid,
    Z_grid,
    cmap="viridis",
)
# Plot points
ax[0].scatter(points[:,0], points[:,1], c='black', s=20)
ax[1].scatter(points[:,0], points[:,1], c='black', s=20)
ax[0].scatter(points[inexact_mask,0], points[inexact_mask,1], c='red', s=30)
ax[1].scatter(points[exact_mask,0], points[exact_mask, 1], c='blue', s=30)
ax[0].scatter(center[0], center[1], c='pink', s=40)
ax[1].scatter(center[0], center[1], c='pink', s=40)

# Labels
ax[0].set_title("Inexact Taylor ellipse (gradient fit)")
ax[1].set_title("Exact Taylor ellipse (exact gradient)")

# Save figure
fig.savefig("cluster.png", dpi=300)
'''