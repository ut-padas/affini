"""Benchmark sklearn KMeans vs cuML KMeans performance."""

import time
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans as skKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

try:
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans

    HAS_GPU = cp.is_available()
except Exception:
    HAS_GPU = False
    warnings.warn("CuPy or cuML not available; GPU benchmarks will be skipped.")

SEED = 0
D = 15  # Number of dimensions
K = 10  # Number of clusters
REPEATS = 3  # average across runs to reduce noise
SIZES = np.unique((np.logspace(3, 6, num=15)).astype(int)).tolist()

CWD = Path(__file__).parent.parent.resolve()
FIGURES = Path(CWD) / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
OUT_CSV = FIGURES / "kmeans_gpu_vs_cpu_results.csv"


def _make_sklearn_kmeans() -> skKMeans:
    return skKMeans(
        n_clusters=K,
        random_state=SEED,
        n_init=1,
        max_iter=50,
        algorithm="elkan",
        init="k-means++",
    )


def _make_cuml_kmeans():
    return cuKMeans(
        n_clusters=K,
        random_state=SEED,
        max_iter=50,
        init="k-means||",
    )


def _gen_data(
    n: int, d: int, seed: int, num_centers: int, std: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(  # type: ignore
        n_samples=n,
        centers=num_centers,
        n_features=d,
        random_state=seed,
        cluster_std=std,
    )
    return X.astype(np.float32), y


def _sse_original(X: np.ndarray, centers: np.ndarray) -> float:
    """SSE in original space: sum over points of min squared distance to any center."""
    if len(centers) == 0:
        return float("inf")

    distances = np.zeros((len(X), len(centers)))
    for i, center in enumerate(centers):
        diffs = X - center
        distances[:, i] = np.einsum("ij,ij->i", diffs, diffs)

    return float(distances.min(axis=1).sum())


def _time_sklearn(X: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float]:
    """Returns (time, SSE, ARI)."""
    t0 = time.perf_counter()
    km = _make_sklearn_kmeans()
    km.fit(X)
    elapsed = time.perf_counter() - t0

    sse = _sse_original(X, km.cluster_centers_)
    ari = adjusted_rand_score(y_true, km.labels_)

    return elapsed, sse, ari


def _time_cuml(X: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float]:
    """Returns (time, SSE, ARI). Include host->device transfer in timing."""
    Xg = cp.asarray(X)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    km = _make_cuml_kmeans()
    km.fit(Xg)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0

    # Convert centers back to CPU for SSE calculation
    centers_cpu = cp.asnumpy(km.cluster_centers_)
    labels_cpu = cp.asnumpy(km.labels_)

    sse = _sse_original(X, centers_cpu)
    ari = adjusted_rand_score(y_true, labels_cpu)

    return elapsed, sse, ari


def _bench_sizes(sizes: Iterable[int]) -> pd.DataFrame:
    results = []

    for n in tqdm(list(sizes), desc="sizes", unit="N"):
        # Generate new data for each size
        X, y_true = _gen_data(n, D, SEED, K)

        # Time CPU implementation
        cpu_times, cpu_sses, cpu_aris = [], [], []
        for _ in tqdm(range(REPEATS), desc="cpu", unit="run", leave=False):
            time_cpu, sse_cpu, ari_cpu = _time_sklearn(X, y_true)
            cpu_times.append(time_cpu)
            cpu_sses.append(sse_cpu)
            cpu_aris.append(ari_cpu)

        # Time GPU implementation if available
        gpu_times, gpu_sses, gpu_aris = [], [], []
        if HAS_GPU:
            for _ in tqdm(range(REPEATS), desc="gpu", unit="run", leave=False):
                time_gpu, sse_gpu, ari_gpu = _time_cuml(X, y_true)
                gpu_times.append(time_gpu)
                gpu_sses.append(sse_gpu)
                gpu_aris.append(ari_gpu)

        # Store results
        result = {
            "n_samples": n,
            "dimensions": D,
            "n_clusters": K,
            "cpu_time_mean": np.mean(cpu_times),
            "cpu_time_std": np.std(cpu_times),
            "cpu_sse_mean": np.mean(cpu_sses),
            "cpu_sse_std": np.std(cpu_sses),
            "cpu_ari_mean": np.mean(cpu_aris),
            "cpu_ari_std": np.std(cpu_aris),
        }

        if HAS_GPU and gpu_times:
            result.update(
                {
                    "gpu_time_mean": np.mean(gpu_times),
                    "gpu_time_std": np.std(gpu_times),
                    "gpu_sse_mean": np.mean(gpu_sses),
                    "gpu_sse_std": np.std(gpu_sses),
                    "gpu_ari_mean": np.mean(gpu_aris),
                    "gpu_ari_std": np.std(gpu_aris),
                    "speedup": np.mean(cpu_times) / np.mean(gpu_times),
                }
            )
        else:
            result.update(
                {
                    "gpu_time_mean": np.nan,
                    "gpu_time_std": np.nan,
                    "gpu_sse_mean": np.nan,
                    "gpu_sse_std": np.nan,
                    "gpu_ari_mean": np.nan,
                    "gpu_ari_std": np.nan,
                    "speedup": np.nan,
                }
            )

        results.append(result)

    return pd.DataFrame(results)


def _plot(df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.parent.mkdir(parents=True, exist_ok=True)

    # Plot runtime vs N
    plt.figure(figsize=(8, 5))

    second_line_alpha = 0.5
    grid_alpha = 0.4

    sizes = df["n_samples"].tolist()
    cpu_times = df["cpu_time_mean"].tolist()
    gpu_times = df["gpu_time_mean"].tolist()

    # Plot GPU with circle marker and normal color (main algorithm)
    if not df["gpu_time_mean"].isna().all():
        (line,) = plt.plot(sizes, gpu_times, marker="o", label="cuML KMeans (GPU)")
        color = line.get_color()
    else:
        # If GPU unavailable
        (line,) = plt.plot([], [], marker="o", label="cuML KMeans (GPU) [unavailable]")
        color = line.get_color()

    # Plot CPU with square marker, same color, smaller alpha, and dashed line
    plt.plot(
        sizes,
        cpu_times,
        marker="s",
        linestyle="--",
        color=color,
        alpha=second_line_alpha,
        label="sklearn KMeans (CPU)",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of points (N)")
    plt.ylabel("Wall time (seconds)")
    plt.title(f"KMeans runtime vs N (D={D}, K={K}, repeats={REPEATS})")
    plt.grid(True, which="both", ls=":", alpha=grid_alpha)
    plt.legend()
    plt.tight_layout()
    fname = figures_dir / "kmeans_gpu_vs_cpu_runtime.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    # Plot SSE vs N and ARI vs N in separate subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # SSE subplot
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of points (N)")
    ax1.set_ylabel("SSE (sum of squared errors)")

    # Plot GPU with circle marker and normal color (main algorithm)
    if not df["gpu_sse_mean"].isna().all():
        (line,) = ax1.plot(sizes, df["gpu_sse_mean"], marker="o", label="SSE (GPU)")
        color = line.get_color()
    else:
        # If GPU unavailable
        (line,) = ax1.plot([], [], marker="o", label="SSE (GPU) [unavailable]")
        color = line.get_color()

    # Plot CPU with square marker, same color, smaller alpha, and dashed line
    ax1.plot(
        sizes,
        df["cpu_sse_mean"],
        marker="s",
        linestyle="--",
        alpha=second_line_alpha,
        color=color,
        label="SSE (CPU)",
    )

    ax1.legend(loc="best")
    ax1.set_title(f"KMeans SSE vs N (D={D}, K={K}, repeats={REPEATS})")
    ax1.grid(True, which="both", ls=":", alpha=grid_alpha)

    # ARI subplot
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of points (N)")
    ax2.set_ylabel("ARI (Adjusted Rand Index)")
    ax2.set_ylim(0, 1)  # since ARI ranges from 0 to 1

    # Plot GPU with circle marker and normal color (main algorithm)
    if not df["gpu_ari_mean"].isna().all():
        (line,) = ax2.plot(sizes, df["gpu_ari_mean"], marker="o", label="ARI (GPU)")
        color = line.get_color()
    else:
        # If GPU unavailable
        (line,) = ax2.plot([], [], marker="o", label="ARI (GPU) [unavailable]")
        color = line.get_color()
    # Plot CPU with square marker, same color, smaller alpha, and dashed line
    ax2.plot(
        sizes,
        df["cpu_ari_mean"],
        marker="s",
        linestyle="--",
        alpha=second_line_alpha,
        color=color,
        label="ARI (CPU)",
    )

    ax2.legend(loc="best")
    ax2.set_title(f"KMeans ARI vs N (D={D}, K={K}, repeats={REPEATS})")
    ax2.grid(True, which="both", ls=":", alpha=grid_alpha)

    plt.tight_layout()
    fname = figures_dir / "kmeans_gpu_vs_cpu_sse_ari.png"
    plt.savefig(fname, dpi=150)
    plt.close()


if __name__ == "__main__":
    if not HAS_GPU:
        print("cuML not available; GPU benchmark will be skipped.")
    if OUT_CSV.exists():
        print(f"Output files : {OUT_CSV} already exist; skipping benchmark.")
    else:
        print("Running KMeans GPU vs CPU benchmark...")
        df_results = _bench_sizes(SIZES)

        # Save results to CSV
        df_results.to_csv(OUT_CSV, index=False)
        print(f"Results saved to {OUT_CSV}")

        # Create plot
        _plot(df_results, FIGURES)
