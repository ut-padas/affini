import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.adaptive_clustering import _HAS_GPU
from surrogate import DataSampler, NearODESurrogate

logger = logging.getLogger(__name__)

CWD = Path.cwd()
FIGURES = CWD / "figures" / "experiments"
FIGURES.mkdir(parents=True, exist_ok=True)


def run_1D_experiment(N_train=2000, ATOL=1e-2):
    from src.problems.oned import one_d_example, one_d_example_grad

    domain = np.array([0.0, 10.0])
    X_train = DataSampler.sample_uniform(domain, N=N_train, seed=0)

    surrogate = NearODESurrogate(data_dim=1, f_dim=1)
    surrogate.train(
        X_train=X_train,
        f=one_d_example,
        grad=one_d_example_grad,
        ATOL=ATOL,
        RTOL=1e99,  # No relative tolerance
        NUM_INITIAL_CLUSTERS=20,
        MIN_PTS_PER_CLUSTER=1,
    )

    N_test = 4000
    X_test = DataSampler.sample_grid(domain, num_points_per_dim=N_test)
    Y_pred, outside_mask = surrogate.predict(
        X_test, ATOL=ATOL, RTOL=1e99, k_NEIGHBOURS=1
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
        c="orange",
        marker="x",
        label="Sources $(N_{src}={})$".format(len(source_centroids)),
    )
    plt.scatter(
        X_test[outside_mask],
        one_d_example(X_test[outside_mask]),
        marker="o",
        label="Unguaranteed Points $(N={})$".format(np.sum(outside_mask)),
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")

    plt.subplot(1, 2, 2)
    plt.semilogy(X_test, error, "-b", label="Absolute Error")
    plt.axhline(ATOL, color="r", linestyle="--", label="Absolute Tolerance")
    plt.scatter(
        X_test[outside_mask],
        error[outside_mask],
        marker="o",
        label="Unguaranteed Points $(N={})$".format(np.sum(outside_mask)),
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("Absolute Error $|f_{surrogate} - f_{exact}|$")

    plt.savefig(FIGURES / "1D_experiment.png", dpi=400)

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
