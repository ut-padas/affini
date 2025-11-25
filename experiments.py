import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.surrogate import DataSampler, NearODESurrogate

logger = logging.getLogger(__name__)

CWD = Path.cwd()
FIGURES = CWD / "figures" / "experiments"
FIGURES.mkdir(parents=True, exist_ok=True)


def run_1D_experiment(N_train=2000, N_test=4000, ATOL=1e-2, RTOL=None):
    from src.problems.oned import one_d_example, one_d_example_grad

    domain = np.array([0.0, 10.0])
    X_train = DataSampler.sample_latin_hypercube(domain, N=N_train, seed=0)

    surrogate = NearODESurrogate(data_dim=1, f_dim=1)
    surrogate.train(
        X_train=X_train,
        f=one_d_example,
        grad=one_d_example_grad,
        ATOL=ATOL,
        RTOL=RTOL,
        NUM_INITIAL_CLUSTERS=10,
        MIN_PTS_PER_CLUSTER=1,
    )

    X_test = DataSampler.sample_grid(domain, num_points_per_dim=N_test)
    Y_pred, outside_mask = surrogate.predict(
        X_test,
        ATOL=ATOL,
        RTOL=RTOL,
        k_NEIGHBOURS=1,
    )
    Y_true = one_d_example(X_test)

    error = np.abs(Y_pred - Y_true)
    rmse = np.sqrt(np.mean(error**2))

    source_centroids = surrogate.get_source_centroids()

    plt.subplots(1, 2, figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X_test, Y_true, "-k", label="Exact")
    plt.plot(X_test, Y_pred, "--m", label="Surrogate")
    plt.scatter(
        source_centroids,
        one_d_example(source_centroids),
        c="blue",
        marker="x",
        label=f"Sources $(N_{{sources}}={len(source_centroids)})$",
    )
    plt.scatter(
        X_test[outside_mask],
        one_d_example(X_test[outside_mask]),
        marker="o",
        c="orange",
        label=f"Unguaranteed Points $(N={np.sum(outside_mask)})$",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")

    plt.subplot(1, 2, 2)
    plt.semilogy(X_test, error, "-k", label="Absolute Error")
    plt.axhline(ATOL, color="r", linestyle="--", label="Absolute Tolerance")
    plt.scatter(
        X_test[outside_mask],
        error[outside_mask],
        marker="o",
        c="orange",
        label=f"Unguaranteed Points $(N={np.sum(outside_mask)})$",
    )
    above_tol_mask = error > ATOL
    plt.scatter(
        X_test[above_tol_mask],
        error[above_tol_mask],
        c="r",
        marker="x",
        label=f"Points Above ATOL $(N={np.sum(above_tol_mask)})$",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("Absolute Error $|f_{surrogate} - f_{exact}|$")

    img_path = FIGURES / f"1D_experiment_ATOL{ATOL}_RTOL{RTOL}.png"
    plt.savefig(img_path, dpi=400)
    logger.info(f"Saved 1D experiment figure to {img_path}")

    print(f"Max Error: {np.max(error):.4e}")
    print(f"RMSE: {rmse:.4e}")


if __name__ == "__main__":
    from src.logging_config import setup_logging

    setup_logging(console_level=logging.INFO)

    tolmat = np.array(
        [
            [0.1, 0.02],
            [0.05, 0.01],
            [0.025, 0.005],
            [0.01, 0.002],
            [0.005, 0.001],
        ]
    )

    run_1D_experiment()
