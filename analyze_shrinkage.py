from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ics26_experiments import ICS26_DIR
from src.adaptive_clustering import calculate_Q

np.random.seed(0)

SHRK_RES_DIR = ICS26_DIR / "shrinkage_analysis"
SHRK_RES_DIR.mkdir(parents=True, exist_ok=True)

SHRINK_VALS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def generate_degenerate_data(
    dim,
    n_points,
    main_scale=1.0,
    small_scale=0.01,
    num_degen_dims: int | Literal["random", "max"] = "random",
):
    """
    Generates data that is highly stretched (high correlation).
    Almost all variance is in the first dimension.
    """
    rand_mat = np.random.randn(dim, dim)
    Q_rot, _ = np.linalg.qr(rand_mat)

    if num_degen_dims == "random":
        # pick a random number of degenerate dims between 1 and dim-1
        _n_dd = np.random.randint(1, dim)
    elif num_degen_dims == "max":
        _n_dd = dim - 1
    else:
        _n_dd = int(num_degen_dims)
    print(f"Generating data in {dim}D with {_n_dd} degenerate dims.")
    major_dims = dim - _n_dd
    scales = np.array([main_scale] * major_dims + [small_scale] * _n_dd)
    np.random.shuffle(scales)

    data_aligned = np.random.randn(n_points, dim) * scales
    return data_aligned @ Q_rot.T


def analyze_shrinkage_dataframe():
    dimensions = [5, 10, 50, 100]

    records = []

    print("Running simulations across dimensions and shrinkage values...")

    for d in dimensions:
        # Use low sample count (N = d + 2) to force instability
        N = d + 2
        X = generate_degenerate_data(d, N)

        starts = np.array([0])
        counts = np.array([N])
        K = 1

        for s_val in SHRINK_VALS:
            _, Cov = calculate_Q(X, starts, counts, K, Q_SHRINK=s_val)

            # Calculate Condition Number: max_eig / min_eig
            # Add tiny epsilon to avoid div-by-zero if singular
            evals = np.linalg.eigvalsh(Cov[0])
            cond_num = evals.max() / (evals.min() + 1e-15)

            records.append(
                {
                    "Dimension": d,
                    "Shrinkage": s_val,
                    "CondNumber": cond_num,
                    "LogCond": np.log10(
                        cond_num
                    ),  # Log scale is easier to read/correlate
                }
            )

    # Create DataFrame
    df = pd.DataFrame(records)

    # --- 1. DISPLAY TABLE ---
    print("\n" + "=" * 20 + " CONDITION NUMBER TABLE " + "=" * 20)
    # Pivot to show Shrinkage as rows, Dimensions as columns
    pivot_df = df.pivot(index="Shrinkage", columns="Dimension", values="CondNumber")

    # Format the float output for cleaner reading
    pd.set_option("display.float_format", "{:.2e}".format)
    print(pivot_df)

    # --- 2. STATISTICAL TEST ---
    print("\n" + "=" * 20 + " STATISTICAL CORRELATION TEST " + "=" * 20)
    print(
        "Hypothesis: Higher Shrinkage is correlated with lower Condition Number (Better Stability)."
    )
    print("Using Spearman Rank Correlation (non-parametric).\n")

    for d in dimensions:
        sub_df = df[df["Dimension"] == d]

        # We test correlation between Shrinkage and Condition Number
        # We expect a strong NEGATIVE correlation (more shrink = lower condition num)
        corr, p_val = stats.spearmanr(sub_df["Shrinkage"], sub_df["CondNumber"])

        significance = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"  # type: ignore
        print(
            f"Dim {d:<3} | Spearman R: {corr:>6.3f} | p-value: {p_val:.2e} -> {significance}"
        )

    # --- 3. PLOT VISUALIZATION ---
    print("\nGenerating visual plot...")
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="Shrinkage",
        y="LogCond",
        hue="Dimension",
        marker="o",
        palette="tab10",
    )
    plt.title("Effect of Shrinkage on Condition Number of Covariance Matrix")
    plt.xlabel("Shrinkage Value")
    plt.ylabel("Log10 Condition Number")
    plt.yscale("linear")
    plt.grid(True)
    plt.legend(title="Dimension")
    plt.tight_layout()
    plt.savefig(SHRK_RES_DIR / "shrinkage_condition_number_analysis.png", dpi=300)
    plt.close()


def analyze_shrinkage_for_fixed_d_but_varying_degen(
    dim=10, main_scale=1, small_scale=0.01, n_points=None
):
    if n_points is None:
        n_points = dim + 2  # minimal points to force instability
    else:
        n_points = int(n_points)

    degen_dims_options = np.linspace(1, dim - 1, num=3, dtype=int)

    records = []

    print("Running simulations for fixed dimension with varying degenerate dims...")

    for n_dd in degen_dims_options:
        X = generate_degenerate_data(
            dim,
            n_points,
            num_degen_dims=n_dd,
            main_scale=main_scale,
            small_scale=small_scale,
        )

        starts = np.array([0])
        counts = np.array([n_points])
        K = 1

        for s_val in SHRINK_VALS:
            _, Cov = calculate_Q(X, starts, counts, K, Q_SHRINK=s_val)

            evals = np.linalg.eigvalsh(Cov[0])
            cond_num = evals.max() / (evals.min() + 1e-15)

            records.append(
                {
                    "N": n_points,
                    "D": dim,
                    "Min/Max Scales": main_scale / small_scale,
                    "DegenerateDims": n_dd,
                    "Shrinkage": s_val,
                    "CondNumber": cond_num,
                    "LogCond": np.log10(cond_num),
                }
            )

    df = pd.DataFrame(records)
    print(df.head())

    # --- 3. PLOT VISUALIZATION ---
    print("\nGenerating visual plot...")
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="Shrinkage",
        y="LogCond",
        hue="DegenerateDims",
        marker="o",
        palette="tab10",
    )
    plt.title(
        f"Shrinkage on cond(COV) (Fixed Dim={dim}, Scales: {main_scale}/{small_scale}, N={n_points})"
    )
    plt.xlabel("Shrinkage Value")
    plt.ylabel("Log10 Condition Number")
    plt.yscale("linear")
    plt.grid(True)
    plt.legend(title="Degenerate Dimensions")
    plt.tight_layout()
    fname = f"shrinkage_condition_number_npoints{n_points}_fixedD{dim}_main{main_scale}_small{small_scale}_analysis.png"
    plt.savefig(SHRK_RES_DIR / fname, dpi=300)
    plt.close()

    return df


if __name__ == "__main__":
    analyze_shrinkage_dataframe()

    # Large fixed dim, varying scales of degeneracy
    dim = 50
    ms = 1
    df_d_vs_ss_1 = analyze_shrinkage_for_fixed_d_but_varying_degen(
        dim=dim, main_scale=ms, small_scale=0.001
    )
    df_d_vs_ss_2 = analyze_shrinkage_for_fixed_d_but_varying_degen(
        dim=dim, main_scale=ms, small_scale=0.01
    )
    df_d_vs_ss_3 = analyze_shrinkage_for_fixed_d_but_varying_degen(
        dim=dim, main_scale=ms, small_scale=0.1
    )
    # merge these dataframes for comparison lineplot sns
    df_merged = pd.concat(
        [df_d_vs_ss_1, df_d_vs_ss_2, df_d_vs_ss_3],
        ignore_index=True,
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_merged,
        x="Shrinkage",
        y="CondNumber",
        hue="Min/Max Scales",
        style="DegenerateDims",
        marker="o",
        palette="tab10",
    )
    plt.title(f"Shrinkage on cond(COV) (Fixed Dim={dim}, Varying Scales of Degeneracy)")
    plt.xlabel("Shrinkage Value")
    plt.ylabel("Condition Number")
    plt.yscale("log")
    ymin = df_merged["CondNumber"].min()
    ymax = min(1e5, df_merged["CondNumber"].max())
    plt.ylim(ymin * 0.9, ymax * 1.1)

    plt.grid(True)
    plt.legend(title="Min/Max Scales, DegenerateDims")
    plt.tight_layout()
    fname = f"shrinkage_condition_number_varying_scales_fixedD{dim}_analysis.png"
    plt.savefig(SHRK_RES_DIR / fname, dpi=300)
    plt.close()

    # Large fixed dims, varying n_points
    ms, ss = 1, 0.01
    dim = 100
    df_d_vs_n_1 = analyze_shrinkage_for_fixed_d_but_varying_degen(
        dim=dim, main_scale=ms, small_scale=ss, n_points=None
    )
    df_d_vs_n_2 = analyze_shrinkage_for_fixed_d_but_varying_degen(
        dim=dim, main_scale=ms, small_scale=ss, n_points=200
    )
    df_d_vs_n_3 = analyze_shrinkage_for_fixed_d_but_varying_degen(
        dim=dim, main_scale=ms, small_scale=ss, n_points=1000
    )

    # merge these dataframes for comparison lineplot sns
    df_merged = pd.concat(
        [df_d_vs_n_1, df_d_vs_n_2, df_d_vs_n_3],
        ignore_index=True,
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_merged,
        x="Shrinkage",
        y="CondNumber",
        hue="N",
        style="DegenerateDims",
        marker="o",
        palette="tab10",
    )
    plt.title(f"Shrinkage on cond(COV) (Fixed Dim={dim}, Scales: {ms}/{ss}, Varying N)")
    plt.xlabel("Shrinkage Value")
    plt.ylabel("Condition Number")
    plt.yscale("log")
    ymin = df_merged["CondNumber"].min()
    ymax = min(1e5, df_merged["CondNumber"].max())
    plt.ylim(ymin * 0.9, ymax * 1.1)
    plt.grid(True)
    plt.legend(title="N, DegenerateDims")
    plt.tight_layout()
    fname = f"shrinkage_condition_number_npoints_varying_fixedD{dim}_main{ms}_small{ss}_analysis.png"
    plt.savefig(SHRK_RES_DIR / fname, dpi=300)
    plt.close()
