"""Benchmark fast two-means (PCA-threshold) vs sklearn KMeans performance."""

import time
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from research.benchmark_kmeans_cpu_vs_gpu import REPEATS, SEED, _gen_data, _sse_original
from src.two_means import two_means_pca_threshold

D = 10  # Number of dimensions
K = 2  # Number of clusters has to be 2 for two-means
SIZES = np.unique((np.logspace(1, 6, num=20)).astype(int)).tolist()

CWD = Path(__file__).parent.parent.resolve()
FIGURES = Path(CWD) / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
OUT_CSV = FIGURES / f"two_means_vs_kmeans_results_D{D}_K{K}.csv"


def _normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def _time_two_means(X: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float]:
    """Returns (time, SSE, ARI)."""
    t0 = time.perf_counter()
    Xn, mean, std = _normalize(X)
    labels, centers = two_means_pca_threshold(Xn, mean, std)
    elapsed = time.perf_counter() - t0

    sse = _sse_original(X, centers)
    ari = adjusted_rand_score(y_true, labels)

    return elapsed, sse, ari


def _time_kmeans(X: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float]:
    """Returns (time, SSE, ARI)."""
    t0 = time.perf_counter()
    Xn, mean, std = _normalize(X)
    km = KMeans(n_clusters=K, random_state=SEED)
    labels = km.fit_predict(Xn)
    centers = km.cluster_centers_ * std + mean
    elapsed = time.perf_counter() - t0

    sse = _sse_original(X, centers)
    ari = adjusted_rand_score(y_true, labels)

    return elapsed, sse, ari


def _bench_sizes(sizes: Iterable[int]) -> pd.DataFrame:
    results = []

    for n in tqdm(list(sizes), desc="sizes", unit="N"):
        X, y_true = _gen_data(n, D, SEED, K)

        two_means_times, two_means_sses, two_means_aris = [], [], []
        for _ in tqdm(range(REPEATS), desc="two_means", unit="run", leave=False):
            time_two_means, sse_two_means, ari_two_means = _time_two_means(X, y_true)
            two_means_times.append(time_two_means)
            two_means_sses.append(sse_two_means)
            two_means_aris.append(ari_two_means)

        kmeans_times, kmeans_sses, kmeans_aris = [], [], []
        for _ in tqdm(range(REPEATS), desc="kmeans", unit="run", leave=False):
            time_kmeans, sse_kmeans, ari_kmeans = _time_kmeans(X, y_true)
            kmeans_times.append(time_kmeans)
            kmeans_sses.append(sse_kmeans)
            kmeans_aris.append(ari_kmeans)

        result = {
            "n_samples": n,
            "dimensions": D,
            "n_clusters": K,
            "two_means_time_mean": np.mean(two_means_times),
            "two_means_time_std": np.std(two_means_times),
            "two_means_sse_mean": np.mean(two_means_sses),
            "two_means_sse_std": np.std(two_means_sses),
            "two_means_ari_mean": np.mean(two_means_aris),
            "two_means_ari_std": np.std(two_means_aris),
            "kmeans_time_mean": np.mean(kmeans_times),
            "kmeans_time_std": np.std(kmeans_times),
            "kmeans_sse_mean": np.mean(kmeans_sses),
            "kmeans_sse_std": np.std(kmeans_sses),
            "kmeans_ari_mean": np.mean(kmeans_aris),
            "kmeans_ari_std": np.std(kmeans_aris),
            "speedup": np.mean(kmeans_times) / np.mean(two_means_times),
        }

        results.append(result)

    return pd.DataFrame(results)


def _plot(df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot runtime vs N
    plt.figure(figsize=(8, 5))

    sizes = df["n_samples"].tolist()
    two_means_times = df["two_means_time_mean"].tolist()
    kmeans_times = df["kmeans_time_mean"].tolist()

    plt.plot(sizes, two_means_times, marker="o", label="Two-means (PCA-threshold)")
    plt.plot(sizes, kmeans_times, marker="s", label="sklearn KMeans")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of points (N)")
    plt.ylabel("Wall time (seconds)")
    plt.title(f"Two-means vs KMeans runtime (D={D}, K={K}, repeats={REPEATS})")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    fname = figures_dir / f"two_means_vs_kmeans_runtime_D{D}_K{K}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    # Plot SSE vs N and ARI vs N in separate subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # SSE subplot
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of points (N)")
    ax1.set_ylabel("SSE (sum of squared errors)")
    ax1.plot(
        sizes,
        df["two_means_sse_mean"],
        marker="o",
        color="blue",
        label="SSE (Two-means)",
    )
    ax1.plot(
        sizes,
        df["kmeans_sse_mean"],
        marker="s",
        color="cyan",
        label="SSE (KMeans)",
    )

    ax1.legend(loc="best")
    ax1.set_title(f"Two-means vs KMeans SSE vs N (D={D}, K={K}, repeats={REPEATS})")
    ax1.grid(True, which="both", ls=":", alpha=0.6)

    # ARI subplot
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of points (N)")
    ax2.set_ylabel("ARI (Adjusted Rand Index)")
    ax2.set_ylim(0, 1)
    ax2.plot(
        sizes,
        df["two_means_ari_mean"],
        marker="o",
        color="red",
        label="ARI (Two-means)",
    )
    ax2.plot(
        sizes,
        df["kmeans_ari_mean"],
        marker="s",
        color="orange",
        label="ARI (KMeans)",
    )

    ax2.legend(loc="best")
    ax2.set_title(f"Two-means vs KMeans ARI vs N (D={D}, K={K}, repeats={REPEATS})")
    ax2.grid(True, which="both", ls=":", alpha=0.6)

    plt.tight_layout()
    fname = figures_dir / f"two_means_vs_kmeans_sse_ari_D{D}_K{K}.png"
    plt.savefig(fname, dpi=150)
    plt.close()


if __name__ == "__main__":
    if OUT_CSV.exists():
        print(f"Output file {OUT_CSV} already exists; skipping benchmark.")
    else:
        print("Running Two-means vs KMeans benchmark...")

        # Warm up JIT
        Xw = np.random.randn(50, D).astype(np.float64)
        Xnw, meanw, stdw = _normalize(Xw)
        two_means_pca_threshold(Xnw, meanw, stdw)

        df_results = _bench_sizes(SIZES)

        # Save results to CSV
        df_results.to_csv(OUT_CSV, index=False)
        print(f"Results saved to {OUT_CSV}")

        # Create plots
        _plot(df_results, FIGURES)
