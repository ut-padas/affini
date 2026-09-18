import csv
import time
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

import argparse
from src import RESULTS_DIR, _to_device, _to_host, xp, Array
from src.adaptive_clustering_utils import SurrogateMode
from src.logging_utils import setup_logging
from src.problems.rosenbrock import rosenbrock, rosenbrock_hess
from src.problems.anisotropic_quadratic import aniso_quadratic, _get_params
from src.problems.isotropic_gaussian import gaussian
from src.problems.wave import wave
from src.problems.lorenz96_flowmap import lorenz96_flowmap
from src.surrogate import AFINNISurrogate, compute_error_metrics

setup_logging()
SEED = 0
    

# ==========================================
# PLOTTING TEMPLATE
# ==========================================
def plot_adaptive_clustering_results(
    centroids: np.ndarray,
    Q: np.ndarray,
    radii: np.ndarray,
    domain: list[float],
    f_fn: Callable,
    title: str,
    X_new: Optional[Array] = None,
    img_fpath: Optional[Path] = None,
):
    if centroids.shape[1] != 2:
        return

    _n = 400
    x = np.linspace(domain[0], domain[1], _n)
    y = np.linspace(domain[0], domain[1], _n)
    X_grid, Y_grid = np.meshgrid(x, y)

    grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    Z_grid = _to_host(f_fn(_to_device(grid_pts)).reshape(X_grid.shape))
    cbar_norm = LogNorm()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    cs = axs[0].contourf(
        X_grid, Y_grid, Z_grid, levels=50, cmap="viridis", norm=cbar_norm
    )
    fig.colorbar(cs, ax=axs[0])
    axs[0].set_title("Function Contour")
    axs[0].set_aspect("equal")
    axs[0].scatter(centroids[:, 0], centroids[:, 1], c="red", s=5, label="Centroids")

    cs2 = axs[1].contourf(
        X_grid, Y_grid, Z_grid, levels=50, cmap="viridis", norm=cbar_norm
    )
    fig.colorbar(cs2, ax=axs[1])
    axs[1].set_title("Trust Ellipsoids & Halo Expansions")
    axs[1].set_aspect("equal")

    for i in range(centroids.shape[0]):
        _centroid = centroids[i]
        _Q = Q[i]
        eigvals, eigvecs = np.linalg.eigh(_Q)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        a, b = 2.0 * np.sqrt(1.0 / np.maximum(eigvals, 1e-12))
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ellipse = Ellipse(
            xy=_centroid,
            width=a,
            height=b,
            angle=angle,
            edgecolor="blue",
            facecolor="none",
            lw=0.5,
            alpha=0.7,
        )
        axs[1].add_patch(ellipse)

    axs[1].scatter(centroids[:, 0], centroids[:, 1], c="red", s=5, zorder=10)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Centroids",
            markerfacecolor="red",
            markersize=5,
        ),
        Line2D([0], [0], color="blue", lw=0.5, label="Trust Ellipsoid"),
    ]

    if X_new is not None and len(X_new) > 0:
        X_new_host = _to_host(X_new)
        axs[1].scatter(
            X_new_host[:, 0], X_new_host[:, 1], c="black", s=5, alpha=0.8, zorder=5
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="New Halo Points",
                markerfacecolor="black",
                markersize=5,
            )
        )

    axs[1].set_xlim(domain); axs[1].set_ylim(domain)
    axs[1].legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.suptitle(title, fontsize=11)
    plt.subplots_adjust(top=0.88)

    img_fpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_fpath, dpi=300)
    plt.close(fig)
    print(f"Saved visualization to {img_fpath}")



def plot_test_points(
    X_test: np.ndarray,
    domain: list[float],
    f_fn: Callable,
    title: str,
    img_fpath: Path
):
    """
    For the paper, plots the test point distribution (on a scatter plot) with the background contours.
    """
    if X_test.shape[1] != 2:
        return

    _n = 400
    x = np.linspace(domain[0], domain[1], _n)
    y = np.linspace(domain[0], domain[1], _n)
    X_grid, Y_grid = np.meshgrid(x, y)

    grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    Z_grid = _to_host(f_fn(_to_device(grid_pts)).reshape(X_grid.shape))
    cbar_norm = LogNorm()

    fig, axs = plt.subplots(figsize=(9, 7.5))

    cs = axs.contourf(
        X_grid, Y_grid, Z_grid, levels=50, cmap="viridis", norm=cbar_norm
    )
    fig.colorbar(cs, ax=axs)
    axs.set_title(title)
    axs.set_aspect("equal")

    N_scatter = 1000
    axs.scatter(X_test[:N_scatter, 0], X_test[:N_scatter, 1], c="red", s=5, label="Uniform Points")
    axs.scatter(X_test[-N_scatter:, 0], X_test[-N_scatter:, 1], c="pink", s=5, label="Function Dist. Points")

    axs.set_xlim(domain[0], domain[1]); axs.set_ylim(domain[0], domain[1])
    axs.legend(loc="lower right")

    img_fpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_fpath, dpi=300)
    plt.close(fig)
    print(f"Saved test points plot to {img_fpath}")



# ==========================================
# MANIFOLD INFLATION & FILTERING
# ==========================================
def halo_sampling(
    surrogate: AFINNISurrogate,
    pts_per_cluster: int,
    gamma: float,
    left:float,
    right:float
) -> Array:
    """
    Each MPI rank generates candidates ONLY for its assigned fraction of clusters,
    filters them locally using the fast geometric filter, and Rank 0 gathers the survivors.
    """

    Nc, d = surrogate.X_src.shape

    centroids = surrogate.X_src
    Qs = surrogate.Qs_src

    # 2. Generate candidates only for MY clusters
    eigenvals, eigenvecs = xp.linalg.eigh(Qs)
    lambda_inv_sqrt = 1.0 / xp.sqrt(xp.maximum(eigenvals, 1e-12))

    xp.random.seed(SEED)
    z = xp.random.randn(Nc, pts_per_cluster, d)
    z_normalized = z / xp.linalg.norm(z, axis=2)[:,:,None]

    scaled_z = z_normalized * lambda_inv_sqrt[:, None, :]
    rotated_scaled_z = xp.einsum("nij,npj->npi", eigenvecs, scaled_z)

    X_cand = centroids[:, None, :] + (gamma * rotated_scaled_z)
    X_cand = X_cand.reshape(-1, d)
    # Remove points that are outside the domain
    in_box_mask = verify_in_domain(X_cand, left, right)
    X_cand = X_cand[in_box_mask]

    # 3. Fast Filter Locally
    ug_mask = surrogate.filter_unguaranteed_local(X_cand, k_coarse=10, batch_size=50000)
    valid_points = X_cand[ug_mask]

    return valid_points



def cloud_sampling(
    surrogate: AFINNISurrogate,
    pts_per_cluster: int,
    left: float,
    right: float
) -> Array:

    Nc, d = surrogate.X_src.shape

    centroids = surrogate.X_src
    Qs = surrogate.Qs_src

    # 2. Generate candidates only for MY clusters
    eigenvals, eigenvecs = xp.linalg.eigh(Qs)
    lambda_inv_sqrt = 1.0 / xp.sqrt(xp.maximum(eigenvals, 1e-12))

    xp.random.seed(SEED)
    z = xp.random.randn(Nc, pts_per_cluster, d)

    scaled_z = z * lambda_inv_sqrt[:, None, :]
    rotated_scaled_z = xp.einsum("nij,npj->npi", eigenvecs, scaled_z)

    X_cand = centroids[:, None, :] + rotated_scaled_z
    X_cand = X_cand.reshape(-1, d)
    # Remove points that are outside the domain
    in_box_mask = verify_in_domain(X_cand, left, right)
    X_cand = X_cand[in_box_mask]

    # 3. Fast Filter Locally
    ug_mask = surrogate.filter_unguaranteed_local(X_cand, k_coarse=10, batch_size=50000)
    valid_points = X_cand[ug_mask]

    return valid_points


def uniform_sampling(
    surrogate: AFINNISurrogate,
    pts_per_cluster: int,
    left:float,
    right:float
) -> Array:
    
    Nc, d = surrogate.X_src.shape
    N_new = Nc * pts_per_cluster
    X_cand = xp.random.uniform(left, right, size=(N_new, d))
    ug_mask = surrogate.filter_unguaranteed_local(X_cand, k_coarse=10, batch_size=50000)
    valid_points = X_cand[ug_mask]
    return valid_points


# Don't need left/right for this because it is convex combinations and these are guaranteed to be in the box
def chord_sampling(
    surrogate: AFINNISurrogate,
    pts_per_cluster: int
):
    N_c, d = surrogate.X_src.shape
    centroids = surrogate.X_src

    # Find five nearest neighbors of each centroid (minus itself)
    # Slow but I don't want to use a library
    diff = centroids[:, None, :] - centroids[None, :, :]
    sq_dists = xp.einsum('ijk,ijk->ij', diff, diff)
    xp.fill_diagonal(sq_dists, xp.inf)
    k = 5; idxs = xp.argpartition(sq_dists, k, axis=1)[:, :k]

    # Sample uniformly between 0 and 4 (these will be the clusters we use to build chords)
    N_sample = pts_per_cluster * N_c
    random_ind = xp.random.randint(0, 5, size=N_sample)

    # Build vectors of indices to sample
    src_cluster_idx = xp.repeat(xp.arange(N_c), pts_per_cluster)    # (N_c * pts_per_cluster)
    dest_cluster_idx = idxs[src_cluster_idx,random_ind]             # (N_c * pts_per_cluster)

    # Build new points
    src_cluster = centroids[src_cluster_idx]
    dest_cluster = centroids[dest_cluster_idx]
    rho = xp.random.uniform(size=N_sample)[:,None]
    X_cand = rho * src_cluster + (1-rho) * dest_cluster

    ug_mask = surrogate.filter_unguaranteed_local(X_cand, k_coarse=10, batch_size=50000)
    valid_points = X_cand[ug_mask]
    return valid_points



"""
Rosenbrock test distribution: sample half the points from the valley distribution
    and sample the other half from a uniform distribution.
"""
def sample_rosenbrock_test_distribution(N_test:int, d:int, left=-2.0, right=3.0):

    N_valley = N_test // 2; N_uniform = N_test - N_valley
    uniform_pts = xp.random.uniform(left, right, (N_uniform, d))

    a = 1.0; b = 5.0; mu = 0.5;

    # Use rejection sampling to ensure all points are in the bounding box.
    in_domain = xp.full(N_valley, False)
    valley_pts = xp.zeros((N_valley, d))
    print("Sampling Rosenbrock distribution.")
    counter = 1
    while sum(in_domain) != N_valley:
        N_out = int(N_valley - sum(in_domain))
        print(f"Iteration {counter}: current points inside: {sum(in_domain)}.")
        counter += 1
        if d == 2:
            x1 = mu + xp.random.randn(N_out) / xp.sqrt(2*a)
            x2 = xp.random.randn(N_out) / np.sqrt(2*b) + (x1**2)
            new_points = xp.column_stack((x1, x2))
        elif d == 3:
            x1 = mu + xp.random.randn(N_out) / xp.sqrt(2*a)
            x2_mean = (2*b*x1**2 + 2*a*mu)**2 / (2*b + 2*a)
            x2_var = 1 / (2*b + 2*a)
            x2 = x2_mean + xp.random.randn(N_out) * xp.sqrt(x2_var)
            x3 = (x2**2) + xp.random.randn(N_out) / xp.sqrt(2*b)
            new_points = xp.column_stack((x1, x2, x3))
        elif d > 3:
            # This distribution doesn't work well; just use uniform for now.
            # mean = xp.ones(d)
            # cov_inv = rosenbrock_hess(xp.reshape(mean, (1,d))).reshape(d,d)
            # cov = xp.linalg.inv(cov_inv)
            # new_points = xp.random.multivariate_normal(mean, cov / (d**2), N_out)
            new_points = xp.random.uniform(left, right, (N_out, d))
        else:
            print(f"Error: d={d} is not valid")
            return None
            
        valley_pts[~in_domain] = new_points
        in_domain = verify_in_domain(valley_pts, left, right)

    return xp.vstack((uniform_pts, valley_pts))


"""
Quadratic test distribution: sample half the points from a Gaussian,
    with covariance equal to the Hessian inverse. Other half comes from uniform distribution.
"""
def sample_quadratic_test_distribution(N_test:int, d:int, left=-3.0, right=3.0):

    N_valley = N_test // 2; N_uniform = N_test - N_valley
    uniform_pts = xp.random.uniform(left, right, (N_uniform, d))

    H = _get_params(d, xp.float64)
    mean = xp.zeros(d)
    cov = 0.5 * np.linalg.inv(H) / (d**2)

    # Use rejection sampling to ensure all points are in the bounding box.
    in_domain = xp.full(N_valley, False)
    valley_pts = xp.random.multivariate_normal(mean, cov, size=N_valley)
    print("Sampling quadratic distribution.")
    counter = 1
    while sum(in_domain) != N_valley:
        N_out = int(N_valley - sum(in_domain))
        print(f"Iteration {counter}: current points inside: {sum(in_domain)}.")
        counter += 1
        new_pts = xp.random.multivariate_normal(mean, cov, size=N_out)
        valley_pts[~in_domain] = new_pts
        in_domain = verify_in_domain(valley_pts, left, right)

    return np.vstack((uniform_pts, valley_pts))


def verify_in_domain(X, left, right):
    X_min = xp.min(X, 1); X_max = xp.max(X, 1)
    out_box_mask = (X_min < left) | (X_max > right)
    return xp.logical_not(out_box_mask)


def generate_new_points(
    surrogate: AFINNISurrogate,
    N_train: int,
    strategy: str,
    pct_to_add: float,
    domain: Array,
    gamma: float
):
    # We want to generate pct_to_add * Nc new points.
    # To do this, we can use rejection sampling.
    Nc, d = surrogate.X_src.shape
    N_target = int(N_train * pct_to_add)
    X_new = xp.zeros((N_target, d))
    N_current = 0
    pts_per_cluster = max(int(N_target / Nc) + 1, 5)
    
    # Sample batches of points
    while N_current < N_target:
        if strategy == "halo":
            X_new_current = halo_sampling(surrogate, pts_per_cluster, gamma, domain[0], domain[1])
        elif strategy == "uniform":
            X_new_current = uniform_sampling(surrogate, pts_per_cluster, domain[0], domain[1])
        elif strategy == "chord":
            X_new_current = chord_sampling(surrogate, pts_per_cluster)
        elif strategy == "cloud":
            X_new_current = cloud_sampling(surrogate, pts_per_cluster, domain[0], domain[1])
        else:
            print(f"Error: sampling strategy \"{strategy}\" is not recognized.")
            return

        if N_current + len(X_new_current) > N_target:
            N_remaining = N_target - N_current
            xp.random.shuffle(X_new_current)
            X_new[N_current:] = X_new_current[:N_remaining]
        else:
            start = N_current; end = N_current + len(X_new_current)
            X_new[start:end] = X_new_current
        N_current += len(X_new_current)

    return X_new


def save_halo_points_for_testing_surrogates(X_train, function_name:str, d:int, strategy:str, N:int=50000):
    numpy_data = _to_device(X_train)
    np.random.shuffle(numpy_data)
    np.save(f"{function_name}_d{d}_{strategy}.npy", numpy_data[:N])



def run_iterative_refinement(
    function:str,
    sampling:str,
    d:int,
    tol:float = 0.05,
    n_iter:int = 5,
    gamma:float = 1.5,
    save_halo_pts:bool = False,
):

    # Hyperparameters
    N_INITIAL_TRAIN = int(20_000 * (1.3 ** d))
    N_TEST = 20_000

    NUM_INITIAL_CLUSTERS = 20 * d
    MIN_PTS_PER_CLUSTER = 50 * d
    ERROR_METRIC = "Linf-local"
    INITIAL_CLUSTERING_METHOD = "kmeans"

    PTS_PER_CLUSTER_CANDIDATE = 10

    # Define parameters based on function selection
    if function == "rosenbrock":
        DOMAIN = [-2.0, 3.0] 
        f_fn = lambda x: rosenbrock(x, a=100.0)
    elif function == "quadratic":
        DOMAIN = [-3.0, 3.0]
        f_fn = lambda x: aniso_quadratic(x)
    elif function == "lorenz96":
        DOMAIN = [-5.0, 5.0]
        f_fn = lambda x: lorenz96_flowmap(x)
    elif function == "wave":
        DOMAIN = [-4.0, 4.0]
        f_fn = lambda x: wave(x)
    elif function == "gaussian":
        DOMAIN = [-3.0, 3.0]
        f_fn = lambda x: gaussian(x)
    else:
        print(f"Function \"{function}\" not implemented; aborting.")
        return

    xp.random.seed(SEED)

    OUTPUT_DIR = RESULTS_DIR / "active_learning_halo"
    csv_path = OUTPUT_DIR / f"halo_metrics_dim{d}_f_{function}_tol{tol}_sampling_{sampling}.csv"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Iteration",
                "N_Train",
                "N_c",
                "e (All)",
                "e_in (In)",
                "e_out (Out)",
                "e_in_inf (In)",
                "e_in_ptwise (In)",
                "P_Out (%)",
                "Train_Time_s",
                "Gen_Filter_Time_s",
                "Eval_Time_s",
            ]
        )

    X_train = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_INITIAL_TRAIN, d))
    p_out_vec = []
    
    # Sample X_test from distribution defined by function
    if function == "rosenbrock":
        X_test = sample_rosenbrock_test_distribution(N_TEST, d, DOMAIN[0], DOMAIN[1])
    elif function == "quadratic":
        X_test = sample_quadratic_test_distribution(N_TEST, d, DOMAIN[0], DOMAIN[1])
    else:
        print(f"No modified test distribution for selected function; using uniform points.")
        X_test = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TEST, d))
    plot_test_points(_to_host(X_test), DOMAIN, f_fn, "Test Distribution", OUTPUT_DIR / "test_points.png")


    ground_truth = f_fn(X_test)

    surrogate = AFINNISurrogate()

    for iteration in range(n_iter + 1):
        t_train, t_gen_filter, t_eval = 0.0, 0.0, 0.0

        print(
            f"\n{'=' * 50}\nITERATION {iteration}: Training on {len(X_train):,} points\n{'=' * 50}"
        )

        # 1. Train Surrogate
        _t_start = time.perf_counter()
        surrogate.train(
            X_train=X_train,
            f_fn=f_fn,
            g_fn=None,
            mode=SurrogateMode.AFFINE_COV_HESS,
            TOL=tol,
            ERROR_METRIC=ERROR_METRIC,
            INITIAL_CLUSTERING_METHOD=INITIAL_CLUSTERING_METHOD,
            NUM_INITIAL_CLUSTERS=NUM_INITIAL_CLUSTERS,
            MIN_PTS_PER_CLUSTER=MIN_PTS_PER_CLUSTER,
            USE_WORK_STEALING=False,
            USE_V2=True,
            STORE_AFFINE_FIT=True,
            save_model=False,
            PROFILE_TIMING=False,
        )
        t_train = time.perf_counter() - _t_start

        # 2. Evaluate Performance
        _t_start = time.perf_counter()
        preds, ug_mask = surrogate.predict(
            query_points=X_test,
            k_nn=1,
            strict_trust_region=False,
        )
        t_eval = time.perf_counter() - _t_start

        p_out = 0.0
        metrics = compute_error_metrics(preds, ground_truth, ug_mask, tol=tol)
        e_all = metrics["all"]["l2_rel_error"]
        e_in = metrics["in"]["l2_rel_error"]
        e_out = metrics["out"]["l2_rel_error"]
        e_in_inf = metrics["in"]["linf_rel_error"]
        e_in_ptwise_max = metrics["in"]["ptwise_max_rel_error_linf"]
        p_out = metrics["perct_out"]
        N_c = surrogate.N_src

        print(f"--> RESULTS: P_out = {p_out:.2f}% | e = {e_all:.4e}")

        dynamic_pts_per_cluster = PTS_PER_CLUSTER_CANDIDATE
        if N_c > 10000:
            dynamic_pts_per_cluster = max(
                10, int(PTS_PER_CLUSTER_CANDIDATE * (10000 / N_c))
            )

        # 3. Generate & Filter New Points via new sampling method
        _t_start = time.perf_counter()
        X_new = generate_new_points(surrogate, len(X_train), sampling, 0.4, DOMAIN, gamma)

        t_gen_filter = time.perf_counter() - _t_start

        # Log metrics to CSV (Rank 0 only)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    iteration,
                    len(X_train),
                    N_c,
                    f"{e_all:.4e}",
                    f"{e_in:.4e}",
                    f"{e_out:.4e}",
                    f"{e_in_inf:.4e}",
                    f"{e_in_ptwise_max:.4e}",
                    f"{p_out:.2f}",
                    f"{t_train:.3f}",
                    f"{t_gen_filter:.3f}",
                    f"{t_eval:.3f}",
                ]
            )

        p_out_vec.append(p_out)
        if X_new is not None and len(X_new) > 0:
            print(f"Retained {len(X_new):,} verified gap points.")
            X_train = xp.vstack([X_train, X_new])
        else:
            print("Could not find any new gap points. Domain is fully packed.")

        # Plot current state
        plot_title = (
            f"Iter {iteration} | N_c={N_c:,} | P_out={p_out:.2f}% | e={e_all:.2e}"
        )
        img_path = OUTPUT_DIR / f"halo_iter_{iteration}.png"
        plot_adaptive_clustering_results(
            centroids=_to_host(surrogate.X_src),
            Q=_to_host(surrogate.Qs_src),
            radii=_to_host(surrogate.trust_radii_src),
            domain=DOMAIN,
            f_fn=f_fn,
            title=plot_title,
            X_new=X_new,
            img_fpath=img_path,
        )

    print(f"\nActive Learning Complete! Metrics saved to {csv_path}")
    if save_halo_pts:
        save_halo_points_for_testing_surrogates(X_train, function, d, sampling)
    return p_out_vec



def plot_iterates_p_out(p_out_halo, p_out_cloud, p_out_uniform, p_out_chord):

    fig_p_out, ax_p_out = plt.subplots()
    fig_Nc, ax_Nc = plt.subplots()
    fig_queries, ax_queries = plt.subplots()

    xrange = np.arange(start=0, stop=10+1, step=1)
    if p_out_halo is not None:
        ax_p_out.plot(xrange, p_out_halo, '-o', label="Halo")
    if p_out_cloud is not None:
        ax_p_out.plot(xrange, p_out_cloud, '-o', label="Cloud")
    if p_out_uniform is not None:
        ax_p_out.plot(xrange, p_out_uniform, '-o', label="Uniform")
    if p_out_chord is not None:
        ax_p_out.plot(xrange, p_out_chord, '-o', label="Chord")
    OUTPUT_DIR = RESULTS_DIR / "active_learning_halo"
    ax_p_out.legend()
    fig_p_out.savefig(OUTPUT_DIR/"sampling_comparison.png")



def plot_halo_various_gamma(p_out_mat, gammavec):

    fig, ax = plt.subplots()
    N_experiment, N_iter = p_out_mat.shape
    xrange = np.arange(N_iter)
    for j in range(N_experiment):
        ax.plot(xrange, p_out_mat[j], '-o', label=f"$\\gamma={gammavec[j]}$")
    OUTPUT_DIR = RESULTS_DIR / "active_learning_halo"
    ax.legend()
    fig.savefig(OUTPUT_DIR/"halo_gamma_comparison.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run p_out iterative experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--function", type=str, default="rosenbrock")
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--exp-num", type=int, default=1)
    return parser.parse_args()




if __name__ == "__main__":

    args = parse_args()
    d = args.dimension
    function = args.function
    n_iter = args.n_iter
    tol = args.tol
    exp_num = args.exp_num

    # Experiment 1: comparing sampling schemes
    if exp_num == 1:
        p_out_halo = run_iterative_refinement(function=function, sampling="halo", d=d, tol=tol, n_iter=n_iter, save_halo_pts=True)
        p_out_cloud = run_iterative_refinement(function=function, sampling="cloud", d=d, tol=tol, n_iter=n_iter)
        p_out_uniform = run_iterative_refinement(function=function, sampling="uniform", d=d, tol=tol, n_iter= n_iter)
        plot_iterates_p_out(p_out_halo, p_out_cloud, p_out_uniform, None)

    # Experiment 2: comparing values of gamma for halo sampling
    elif exp_num == 2:
        p_out_mat = np.zeros((5, n_iter+1))
        gammavec = [1.1, 1.3, 1.5, 1.75, 2.0]
        for j in range(5):
            p_out_mat[j] = run_iterative_refinement(
                function=function,
                sampling="halo",
                d=d,
                tol=tol,
                n_iter=n_iter,
                gamma=gammavec[j]
            )
        plot_halo_various_gamma(p_out_mat, gammavec)
