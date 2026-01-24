import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ics26_rosenbrock import ICS26_DIR
from src.adaptive_clustering import calculate_Q

np.random.seed(0)

SHRK_RES_DIR = ICS26_DIR / "shrinkage_analysis"
SHRK_RES_DIR.mkdir(parents=True, exist_ok=True)

# Publication-Quality Plot Settings
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (6.5, 4.5),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    }
)


# Common Constants
SHRINK_VALS = [0.0, 0.01, 0.05, 0.1]
PALETTE = "deep"
COLORS = sns.color_palette(PALETTE, n_colors=len(SHRINK_VALS))
N_MULTIPLIER = 2


def get_target_N(d):
    """Calculates the Vershynin threshold: 10 * d * log(d)"""
    if d <= 1:
        return 10
    return int(N_MULTIPLIER * d * np.log(d))


def generate_degenerate_data(
    dim, n_points, main_scale=1.0, small_scale=0.1, num_degen_dims=1
):
    """
    Generates data where 'num_degen_dims' dimensions have 'small_scale' variance,
    and the rest have 'main_scale'. The data is then randomly rotated.
    """
    # 1. Create diagonal scales
    n_normal = dim - num_degen_dims
    scales = np.array([main_scale] * n_normal + [small_scale] * num_degen_dims)
    np.random.shuffle(scales)

    # 2. Generate axis-aligned data
    data = np.random.randn(n_points, dim) * scales  # Dim: (n_points, dim)

    # 3. Random rotation (QR decomposition of random matrix)
    Q_rot, _ = np.linalg.qr(np.random.randn(dim, dim))  # Dim: (dim, dim)
    return data @ Q_rot.T


def run_covariance_experiment(X, shrinkage_list):
    """Helper to compute condition numbers for a given dataset X and list of shrinkages."""
    results = {}
    N, D = X.shape

    starts = np.array([0])
    counts = np.array([N])
    K = 1

    for s_val in shrinkage_list:
        _, Cov = calculate_Q(X, starts, counts, K, Q_SHRINK=s_val)

        # Calculate Condition Number
        evals = np.linalg.eigvalsh(Cov[0])
        # Avoid division by zero
        min_eig = max(evals.min(), 1e-15)
        cond_num = evals.max() / min_eig
        results[s_val] = cond_num

    return results


# =============================================================================
# EXPERIMENT 1: Varying Degenerate Dimensions (Fixed D, Fixed N)
# =============================================================================
def experiment_1_varying_degeneracy():
    print("\n--- Running Experiment 1: Varying Degeneracy ---")
    D = 50
    N = get_target_N(D)
    main_scale = 100.0
    small_scale = 1.0
    ratio = int(main_scale / small_scale)

    # Vary degenerate dimensions from 1 to D-1 (increment by 2)
    degen_range = list(range(1, D, 2))

    records = []

    for n_deg in tqdm(degen_range, desc="Degenerate Dimensions"):
        X = generate_degenerate_data(
            D, N, main_scale, small_scale, num_degen_dims=n_deg
        )
        res = run_covariance_experiment(X, SHRINK_VALS)

        for s_val, cond in res.items():
            records.append(
                {
                    "Degenerate_Dims": n_deg,
                    "Shrinkage": s_val,
                    "CondNumber": cond,
                    "LogCond": np.log10(cond),
                }
            )

    df = pd.DataFrame(records)

    # Dump Table
    csv_path = SHRK_RES_DIR / "exp1_varying_degeneracy.csv"
    df.to_csv(csv_path, index=False)
    print(f"Exp 1 Data saved to {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.lineplot(
        data=df,
        x="Degenerate_Dims",
        y="CondNumber",
        hue="Shrinkage",
        palette=PALETTE,
        marker="o",
        ax=ax,
        linewidth=1.5,
        markersize=5,
    )
    ax.set_title(
        rf"Condition Number vs.Degeneracy ($D={D}$, $N={N}$, Ratio={ratio:.0f})",
        fontsize=12,
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of Degenerate Dimensions")
    ax.set_ylabel(r"Condition Number $\kappa$")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(title=r"Shrinkage $\tau$", frameon=True)
    plt.tight_layout()
    plt.savefig(
        SHRK_RES_DIR / "exp1_cond_vs_degeneracy.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


# =============================================================================
# EXPERIMENT 2: Varying Sample Size N (Fixed D, Fixed Degeneracy)
# =============================================================================
def experiment_2_varying_N():
    print("\n--- Running Experiment 2: Varying Sample Size N ---")
    D = 50
    n_deg = 10
    target_N = get_target_N(D)

    # Scale Ratios to test: 10 and 100
    scale_configs = [
        {"ratio": 10, "main": 100.0, "small": 1.0},
        {"ratio": 100, "main": 1000.0, "small": 1.0},
    ]

    # Sample sizes: From D+2 up to 2*Target_N
    n_steps = np.unique(np.linspace(D + 5, target_N * 2.5, 20).astype(int))
    n_steps = np.sort(n_steps)

    records = []

    for config in tqdm(scale_configs, desc="Scale Ratios"):
        ratio = config["ratio"]
        for n_points in tqdm(n_steps, desc="Sample Sizes", leave=False):
            X = generate_degenerate_data(
                D, n_points, config["main"], config["small"], num_degen_dims=n_deg
            )
            res = run_covariance_experiment(X, SHRINK_VALS)

            for s_val, cond in res.items():
                records.append(
                    {
                        "N": n_points,
                        "Scale_Ratio": ratio,
                        "Shrinkage": s_val,
                        "CondNumber": cond,
                    }
                )

    df = pd.DataFrame(records)

    # Dump Table
    csv_path = SHRK_RES_DIR / "exp2_varying_sample_size.csv"
    df.to_csv(csv_path, index=False)
    print(f"Exp 2 Data saved to {csv_path}")

    # Plot
    g = sns.FacetGrid(
        df,
        col="Scale_Ratio",
        hue="Shrinkage",
        height=4.5,
        aspect=1.35,
        palette=PALETTE,
        despine=True,
    )
    g.map(
        sns.lineplot,
        "N",
        "CondNumber",
        marker="o",
        linewidth=1.5,
        markersize=5,
    )

    # Add vertical line at Target N and format axes
    for ax, ratio in zip(g.axes.flat, sorted(df["Scale_Ratio"].unique())):
        ax.axvline(
            target_N,
            color="red",
            linestyle="--",
            linewidth=1,
            label=rf"Target $N$ ({target_N})",
            alpha=0.7,
        )
        ax.set_yscale("log")
        ax.set_xlabel(r"Sample Size $N$")
        ax.set_ylabel(r"Condition Number $\kappa$")
        ax.grid(True, which="both", ls="-", alpha=0.3)
        ax.set_title(rf"Scale Ratio = {int(ratio)}")

    g.add_legend(title=r"Shrinkage $\tau$", frameon=True)

    g.figure.suptitle(
        rf"Condition Number vs.\ Sample Size ($D={D}$, Degenerate Dims={n_deg})",
        y=0.98,
        fontsize=12,
    )
    g.figure.subplots_adjust(top=0.86)

    plt.savefig(
        SHRK_RES_DIR / "exp2_cond_vs_samplesize.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


# =============================================================================
# EXPERIMENT 3: Varying Dimensions D (N = 10 d log d)
# =============================================================================
def experiment_3_varying_D():
    print("\n--- Running Experiment 3: Varying Dimensions D ---")
    dimensions = [2, 4, 8, 16, 32, 64, 128, 256]
    main_scale = 100.0
    small_scale = 1.0
    ratio = int(main_scale / small_scale)

    records = []

    for D in tqdm(dimensions, desc="Dimensions"):
        N = get_target_N(D)
        # Use a proportional number of degenerate dims, e.g., 20% of D, minimum 1
        n_deg = max(1, int(D * 0.2))

        X = generate_degenerate_data(
            D, N, main_scale, small_scale, num_degen_dims=n_deg
        )
        res = run_covariance_experiment(X, SHRINK_VALS)

        for s_val, cond in res.items():
            records.append(
                {"Dimension": D, "N_used": N, "Shrinkage": s_val, "CondNumber": cond}
            )

    df = pd.DataFrame(records)

    # Dump Table
    csv_path = SHRK_RES_DIR / "exp3_varying_dimension.csv"
    df.to_csv(csv_path, index=False)
    print(f"Exp 3 Data saved to {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.lineplot(
        data=df,
        x="Dimension",
        y="CondNumber",
        hue="Shrinkage",
        palette=PALETTE,
        marker="o",
        ax=ax,
        linewidth=1.5,
        markersize=5,
    )
    ax.set_title(
        rf"Condition Number vs.\ Dimension ($N = {N_MULTIPLIER}d\log d$, Ratio={ratio})",
        fontsize=12,
    )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel(r"Condition Number $\kappa$")
    ax.set_xlabel(r"Dimension $d$")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(title=r"Shrinkage $\tau$", frameon=True)
    plt.tight_layout()
    plt.savefig(
        SHRK_RES_DIR / "exp3_cond_vs_dimension.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    experiment_1_varying_degeneracy()
    experiment_2_varying_N()
    experiment_3_varying_D()
    print("\nAll experiments completed.")
