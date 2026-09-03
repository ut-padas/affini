import csv
import time
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

from src import RESULTS_DIR, _to_device, _to_host, xp, Array
from src.adaptive_clustering_utils import SurrogateMode
from src.logging_utils import setup_logging
from src.problems.rosenbrock import rosenbrock, rosenbrock_grad
from src.surrogate import AFINNISurrogate, compute_error_metrics

setup_logging()

# ==========================================
# HYPERPARAMETERS
# ==========================================
DIM = 5
TOL = 0.05
DOMAIN = [-2.0, 3.0]
SEED = 0

N_INITIAL_TRAIN = 5000
N_TEST = 20_000

NUM_INITIAL_CLUSTERS = 20 * DIM
MIN_PTS_PER_CLUSTER = 50 * DIM
ERROR_METRIC = "Linf-local"
INITIAL_CLUSTERING_METHOD = "kmeans"

# Active Learning Configuration
MAX_ITERATIONS = 5
HALO_FACTOR = 1.5
PTS_PER_CLUSTER_CANDIDATE = 10

A_ROSENBROCK = 100.0


def f_fn(x):
    return rosenbrock(x, a=A_ROSENBROCK)


def g_fn(x):
    return rosenbrock_grad(x, a=A_ROSENBROCK)


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

    axs[1].legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.suptitle(title, fontsize=11)
    plt.subplots_adjust(top=0.88)

    img_fpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_fpath, dpi=300)
    plt.close(fig)
    print(f"Saved visualization to {img_fpath}")



def plot_test_points(X_test: Array):
    """
    For the paper, plots the test point distribution (on a scatter plot) with the background contours.
    """
    



# ==========================================
# DISTRIBUTED MANIFOLD INFLATION & FILTERING
# ==========================================
def generate_and_filter(
    surrogate: AFINNISurrogate, pts_per_cluster: int, gamma: float
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
    X_cand_min = xp.min(X_cand, 1); X_cand_max = xp.max(X_cand, 1)
    out_box_mask = (X_cand_min < DOMAIN[0]) | (X_cand_max > DOMAIN[1])
    X_cand = X_cand[~out_box_mask]

    # 3. Fast Filter Locally
    ug_mask = surrogate.filter_unguaranteed_local(X_cand, k_coarse=10, batch_size=50000)
    valid_points = X_cand[ug_mask]

    return valid_points



def run_halo_sampling():
    xp.random.seed(SEED)

    OUTPUT_DIR = RESULTS_DIR / "active_learning_halo"
    csv_path = OUTPUT_DIR / f"halo_metrics_dim{DIM}_tol{TOL}.csv"

    X_test, ground_truth, X_train = None, None, None
    X_new_halo = None

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
                "P_Out (%)",
                "Train_Time_s",
                "Gen_Filter_Time_s",
                "Eval_Time_s",
            ]
        )

        X_test = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TEST, DIM))
        ground_truth = f_fn(X_test)
        X_train = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_INITIAL_TRAIN, DIM))

    surrogate = AFINNISurrogate()

    for iteration in range(MAX_ITERATIONS + 1):
        t_train, t_gen_filter, t_eval = 0.0, 0.0, 0.0

        print(
            f"\n{'=' * 50}\nITERATION {iteration}: Training on {len(X_train):,} points\n{'=' * 50}"
        )

        # 1. Train Surrogate
        _t_start = time.perf_counter()
        surrogate.train(
            X_train=X_train,
            f_fn=f_fn,
            g_fn=g_fn,
            mode=SurrogateMode.AFFINE_COV_HESS,
            TOL=TOL,
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
        metrics = compute_error_metrics(preds, ground_truth, ug_mask, tol=TOL)
        e_all = metrics["all"]["l2_rel_error"]
        e_in = metrics["in"]["l2_rel_error"]
        e_out = metrics["out"]["l2_rel_error"]
        p_out = metrics["perct_out"]
        N_c = surrogate.N_src

        print(f"--> RESULTS: P_out = {p_out:.2f}% | e = {e_all:.4e}")

        dynamic_pts_per_cluster = PTS_PER_CLUSTER_CANDIDATE
        if N_c > 10000:
            dynamic_pts_per_cluster = max(
                10, int(PTS_PER_CLUSTER_CANDIDATE * (10000 / N_c))
            )

        # 3. Generate & Filter New Points via Distributed Halo Expansion
        _t_start = time.perf_counter()
        X_new_halo = generate_and_filter(surrogate, dynamic_pts_per_cluster, HALO_FACTOR)
        t_gen_filter = time.perf_counter() - _t_start

        if X_new_halo is not None and len(X_new_halo) > 0:
            print(f"Retained {len(X_new_halo):,} verified gap points.")
            X_train = xp.vstack([X_train, X_new_halo])
        else:
            print("Could not find any new gap points. Domain is fully packed.")

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
                    f"{p_out:.2f}",
                    f"{t_train:.3f}",
                    f"{t_gen_filter:.3f}",
                    f"{t_eval:.3f}",
                ]
            )

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
            X_new=X_new_halo,
            img_fpath=img_path,
        )

    print(f"\nActive Learning Complete! Metrics saved to {csv_path}")



if __name__ == "__main__":
    run_halo_sampling()
