"""Benchmark sklearn KMeans vs MiniBatchKMeans performance."""

import time
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans as skMBKMeans
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from research.benchmark_kmeans_gpu_vs_cpu import (
    REPEATS,
    SEED,
    K,
    _gen_data,
    _make_sklearn_kmeans,
    _sse_original,
)

D = 10
SIZES = np.unique((np.logspace(3, 6, num=15)).astype(int)).tolist()

CWD = Path(__file__).parent.parent.resolve()
FIGURES = Path(CWD) / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
OUT_CSV = FIGURES / "minibatch_vs_kmeans_results.csv"


def _make_minibatch_kmeans(n_samples: int) -> skMBKMeans:
    batch = int(min(8192, max(256, np.sqrt(n_samples))))
    return skMBKMeans(
        n_clusters=K,
        random_state=SEED,
        n_init=1,
        max_iter=50,
        batch_size=batch,
        init="k-means++",
    )


def _time_sklearn(X: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float]:
    """Returns (time, SSE, ARI)."""
    t0 = time.perf_counter()
    km = _make_sklearn_kmeans()
    km.fit(X)
    elapsed = time.perf_counter() - t0

    sse = _sse_original(X, km.cluster_centers_)
    ari = adjusted_rand_score(y_true, km.labels_)

    return elapsed, sse, ari


def _time_minibatch(X: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float]:
    """Returns (time, SSE, ARI)."""
    t0 = time.perf_counter()
    mb = _make_minibatch_kmeans(X.shape[0])
    mb.fit(X)
    elapsed = time.perf_counter() - t0

    sse = _sse_original(X, mb.cluster_centers_)
    ari = adjusted_rand_score(y_true, mb.labels_)

    return elapsed, sse, ari


def _bench_sizes(sizes: Iterable[int]) -> pd.DataFrame:
    results = []

    for n in tqdm(list(sizes), desc="sizes", unit="N"):
        X, y_true = _gen_data(n, D, SEED, K)

        sklearn_times, sklearn_sses, sklearn_aris = [], [], []
        for _ in tqdm(range(REPEATS), desc="sklearn", unit="run", leave=False):
            time_sklearn, sse_sklearn, ari_sklearn = _time_sklearn(X, y_true)
            sklearn_times.append(time_sklearn)
            sklearn_sses.append(sse_sklearn)
            sklearn_aris.append(ari_sklearn)

        minibatch_times, minibatch_sses, minibatch_aris = [], [], []
        for _ in tqdm(range(REPEATS), desc="minibatch", unit="run", leave=False):
            time_minibatch, sse_minibatch, ari_minibatch = _time_minibatch(X, y_true)
            minibatch_times.append(time_minibatch)
            minibatch_sses.append(sse_minibatch)
            minibatch_aris.append(ari_minibatch)

        result = {
            "n_samples": n,
            "dimensions": D,
            "n_clusters": K,
            "sklearn_time_mean": np.mean(sklearn_times),
            "sklearn_time_std": np.std(sklearn_times),
            "sklearn_sse_mean": np.mean(sklearn_sses),
            "sklearn_sse_std": np.std(sklearn_sses),
            "sklearn_ari_mean": np.mean(sklearn_aris),
            "sklearn_ari_std": np.std(sklearn_aris),
            "minibatch_time_mean": np.mean(minibatch_times),
            "minibatch_time_std": np.std(minibatch_times),
            "minibatch_sse_mean": np.mean(minibatch_sses),
            "minibatch_sse_std": np.std(minibatch_sses),
            "minibatch_ari_mean": np.mean(minibatch_aris),
            "minibatch_ari_std": np.std(minibatch_aris),
            "speedup": np.mean(sklearn_times) / np.mean(minibatch_times),
        }

        results.append(result)

    return pd.DataFrame(results)


def _plot(df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))

    second_line_alpha = 0.5
    grid_alpha = 0.4

    # Plot runtime vs N
    sizes = df["n_samples"].tolist()
    sklearn_times = df["sklearn_time_mean"].tolist()
    minibatch_times = df["minibatch_time_mean"].tolist()

    # Plot MinibatchKmeans with circle marker and normal color
    (line,) = plt.plot(sizes, minibatch_times, marker="o", label="MinibatchKMeans")
    color = line.get_color()

    # Plot KMeans with square marker, same color, smaller alpha, and dashed line
    plt.plot(
        sizes,
        sklearn_times,
        marker="s",
        linestyle="--",
        alpha=second_line_alpha,
        color=color,
        label="KMeans",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of points (N)")
    plt.ylabel("Wall time (seconds)")
    plt.title(f"MiniBatchKMeans vs KMeans runtime (D={D}, K={K}, repeats={REPEATS})")
    plt.grid(True, which="both", ls=":", alpha=grid_alpha)
    plt.legend()
    plt.tight_layout()
    fname = figures_dir / "minibatch_vs_kmeans_runtime.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    # Plot SSE vs N and ARI vs N in separate subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # SSE subplot
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of points (N)")
    ax1.set_ylabel("SSE (sum of squared errors)")

    # Plot MiniBatchKMeans with circle marker and normal color
    (line,) = ax1.plot(
        sizes, df["minibatch_sse_mean"], marker="o", label="SSE (MiniBatchKMeans)"
    )
    color = line.get_color()

    # Plot KMeans with square marker, same color, smaller alpha, and dashed line
    ax1.plot(
        sizes,
        df["sklearn_sse_mean"],
        marker="s",
        linestyle="--",
        alpha=second_line_alpha,
        color=color,
        label="SSE (KMeans)",
    )

    ax1.legend(loc="best")
    ax1.set_title(
        f"MiniBatchKMeans vs KMeans SSE vs N (D={D}, K={K}, repeats={REPEATS})"
    )
    ax1.grid(True, which="both", ls=":", alpha=grid_alpha)

    # ARI subplot
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of points (N)")
    ax2.set_ylabel("ARI (Adjusted Rand Index)")

    # Plot MiniBatchKMeans with circle marker and normal color
    (line,) = ax2.plot(
        sizes, df["minibatch_ari_mean"], marker="o", label="ARI (MiniBatchKMeans)"
    )
    color = line.get_color()

    # Plot KMeans with square marker, same color, smaller alpha, and dashed line
    ax2.plot(
        sizes,
        df["sklearn_ari_mean"],
        marker="s",
        linestyle="--",
        alpha=second_line_alpha,
        color=color,
        label="ARI (KMeans)",
    )

    ax2.legend(loc="best")
    ax2.set_title(
        f"MiniBatchKMeans vs KMeans ARI vs N (D={D}, K={K}, repeats={REPEATS})"
    )
    ax2.grid(True, which="both", ls=":", alpha=grid_alpha)

    plt.tight_layout()
    fname = figures_dir / "minibatch_vs_kmeans_sse_ari.png"
    plt.savefig(fname, dpi=150)
    plt.close()


def main():
    if OUT_CSV.exists():
        print(f"Output files: {OUT_CSV} already exist; skipping benchmark.")
        return

    print("Running MiniBatchKMeans vs KMeans benchmark...")
    df_results = _bench_sizes(SIZES)

    # Save results to CSV
    df_results.to_csv(OUT_CSV, index=False)
    print(f"Results saved to {OUT_CSV}")

    # Create plot
    _plot(df_results, FIGURES)


if __name__ == "__main__":
    main()
