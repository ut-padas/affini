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

from research.benchmark_kmeans_gpu_vs_cpu import REPEATS, SEED, _gen_data, _sse_original
from src.two_means import numba_two_means_warmup
from src.two_means_random_qr import two_means_random_qr_threshold

DK_PAIRS = [(5, 2), (25, 2), (125, 2)]  # (smallD, medD, largeD) with K=2
SIZES = np.unique((np.logspace(1, 6, num=20)).astype(int)).tolist()

CWD = Path(__file__).parent.parent.resolve()
FIGURES = Path(CWD) / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
OUT_CSV = FIGURES / "two_means_qr_vs_kmeans_results.csv"


def _normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    return (X - mean), mean


def _time_two_means(X: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float]:
    """Returns (time, SSE, ARI)."""
    t0 = time.perf_counter()
    Xn, mean = _normalize(X)
    labels, centers = two_means_random_qr_threshold(Xn, mean)
    elapsed = time.perf_counter() - t0

    sse = _sse_original(X, centers)
    ari = adjusted_rand_score(y_true, labels)

    return elapsed, sse, ari


def _time_kmeans(
    X: np.ndarray, y_true: np.ndarray, K: int
) -> tuple[float, float, float]:
    """Returns (time, SSE, ARI)."""
    t0 = time.perf_counter()
    Xn, mean = _normalize(X)
    km = KMeans(n_clusters=K, random_state=SEED)
    labels = km.fit_predict(Xn)
    centers = km.cluster_centers_ + mean
    elapsed = time.perf_counter() - t0

    sse = _sse_original(X, centers)
    ari = adjusted_rand_score(y_true, labels)

    return elapsed, sse, ari


def _bench_sizes(sizes: Iterable[int], D: int, K: int) -> pd.DataFrame:
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
            time_kmeans, sse_kmeans, ari_kmeans = _time_kmeans(X, y_true, K)
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

    # Plot runtime vs N for all dimensions
    plt.figure(figsize=(10, 6))

    second_line_alpha = 0.5
    grid_alpha = 0.4

    for D in sorted(df["dimensions"].unique()):
        df_d = df[df["dimensions"] == D]
        sizes = df_d["n_samples"].tolist()
        two_means_times = df_d["two_means_time_mean"].tolist()
        kmeans_times = df_d["kmeans_time_mean"].tolist()

        # Plot two-means with circle marker and normal color
        (line,) = plt.plot(
            sizes, two_means_times, marker="o", label=f"Two-means (QR) D={D}"
        )
        color = line.get_color()

        # Plot k-means with square marker, same color, smaller alpha, and dashed line
        plt.plot(
            sizes,
            kmeans_times,
            marker="s",
            linestyle="--",
            alpha=second_line_alpha,
            color=color,
            label=f"KMeans D={D}",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of points (N)")
    plt.ylabel("Wall time (seconds)")
    plt.title(f"Two-means (QR) vs KMeans runtime (repeats={REPEATS})")
    plt.grid(True, which="both", ls=":", alpha=grid_alpha)
    plt.legend()
    plt.tight_layout()
    fname = figures_dir / "two_means_qr_vs_kmeans_runtime.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    # Plot SSE and ARI for all dimensions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # SSE subplot
    for D in sorted(df["dimensions"].unique()):
        df_d = df[df["dimensions"] == D]
        sizes = df_d["n_samples"].tolist()

        # Plot two-means with circle marker and normal color
        (line,) = ax1.plot(
            sizes, df_d["two_means_sse_mean"], marker="o", label=f"Two-means (QR) D={D}"
        )
        color = line.get_color()

        # Plot k-means with square marker, same color, smaller alpha, and dashed line
        ax1.plot(
            sizes,
            df_d["kmeans_sse_mean"],
            marker="s",
            linestyle="--",
            alpha=second_line_alpha,
            color=color,
            label=f"KMeans D={D}",
        )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of points (N)")
    ax1.set_ylabel("SSE (sum of squared errors)")
    ax1.set_title(f"Two-means (QR) vs KMeans SSE (repeats={REPEATS})")
    ax1.grid(True, which="both", ls=":", alpha=grid_alpha)
    ax1.legend()

    # ARI subplot
    for D in sorted(df["dimensions"].unique()):
        df_d = df[df["dimensions"] == D]
        sizes = df_d["n_samples"].tolist()

        # Plot two-means with circle marker and normal color
        (line,) = ax2.plot(
            sizes, df_d["two_means_ari_mean"], marker="o", label=f"Two-means (QR) D={D}"
        )
        color = line.get_color()

        # Plot k-means with square marker, same color, smaller alpha, and dashed line
        ax2.plot(
            sizes,
            df_d["kmeans_ari_mean"],
            marker="s",
            linestyle="--",
            alpha=second_line_alpha,
            color=color,
            label=f"KMeans D={D}",
        )

    ax2.set_xscale("log")
    ax2.set_xlabel("Number of points (N)")
    ax2.set_ylabel("ARI (Adjusted Rand Index)")
    ax2.set_title(f"Two-means (QR) vs KMeans ARI (repeats={REPEATS})")
    ax2.grid(True, which="both", ls=":", alpha=grid_alpha)
    ax2.legend()

    plt.tight_layout()
    fname = figures_dir / "two_means_qr_vs_kmeans_sse_ari.png"
    plt.savefig(fname, dpi=150)
    plt.close()


def main():
    if OUT_CSV.exists():
        print(f"Output file {OUT_CSV} already exists; skipping benchmark.")
        return
    numba_two_means_warmup()
    print("Running Two-means (QR) vs KMeans benchmark for all dimensions...")

    all_results = []

    for D, K in DK_PAIRS:
        print(f"Processing D={D}, K={K}...")

        df_results = _bench_sizes(SIZES, D, K)
        all_results.append(df_results)

    # Combine all results
    df_combined = pd.concat(all_results, ignore_index=True)

    # Save combined results to CSV
    df_combined.to_csv(OUT_CSV, index=False)
    print(f"Combined results saved to {OUT_CSV}")

    # Create combined plots
    _plot(df_combined, FIGURES)


if __name__ == "__main__":
    main()
