import json
from pathlib import Path

import pandas as pd


def json_to_csv(target_dirs):
    """
    Convert JSON files from multiple directories to CSV files.

    Args:
        target_dirs: List of Path objects or directory paths
    """
    for target_dir in target_dirs:
        target_path = Path(target_dir)

        if not target_path.exists():
            print(f"Directory not found: {target_path}")
            continue

        # 1. Glob all JSON files in the specified directory
        json_files = list(target_path.glob("*.json"))

        if not json_files:
            print(f"No JSON files found in {target_path}")
            continue

        data_list = []

        # 2. Iterate and parse
        for file_path in json_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    # Add filename for traceability
                    data["source_file"] = file_path.name
                    data_list.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {file_path}")

        # 3. Create DataFrame and sort
        df = pd.DataFrame(data_list)
        if "mpi_size" in df.columns:
            df = df.sort_values("mpi_size")

        # Keep only dim==5 and tol==5e-3
        if "dim" in df.columns and "tol" in df.columns:
            df = df[(df["dim"] >= 5) & (df["tol"] <= 5e-3)]

        if "gather_time_sec" not in df.columns:
            df["gather_time_sec"] = pd.NA

        if "train_time_sec" in df.columns and "init_time_sec" in df.columns:
            df["compute_time_sec"] = df["train_time_sec"] - df["init_time_sec"]

        # Rename & Reorder columns
        column_mapping = {
            "mpi_size": "MPI Size",
            "dim": "Dim",
            "tol": "Tol",
            "num_clusters": "N_c",
            "compute_time_sec": "Compute Time (s)",
            "train_time_sec": "Train Time (s)",
            "init_time_sec": "Init Time (s)",
            "gather_time_sec": "Gather Time (s)",
            "n_train": "N_0",
            "total_f_evals": "N",
            "max_tree_level": "Max Tree Level",
            "num_initial_clusters": "N_init_clusters",
        }
        df = df.rename(columns=column_mapping)
        desired_order = list(column_mapping.values())
        df = df[[col for col in desired_order if col in df.columns]]

        # 4. Save CSV with subdirectory name
        output_path = target_path.parent / f"{target_path.name}.csv"
        df.to_csv(output_path, index=False)

        print(f"Successfully saved {len(df)} records to {output_path}")


def json_to_csv_cumrf(target_dir):
    """
    Convert cuML RF JSON result files in one directory to a flattened CSV.

    Args:
        target_dir: Path object or directory path
    """
    target_path = Path(target_dir)

    if not target_path.exists():
        print(f"Directory not found: {target_path}")
        return

    json_files = sorted(target_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {target_path}")
        return

    data_list = []

    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {file_path}")
            continue

        row = {k: v for k, v in data.items() if k != "rmse_history"}

        best_params = row.pop("best_params", {})
        if isinstance(best_params, dict):
            row.update(best_params)

        row["source_file"] = file_path.name
        data_list.append(row)

    if not data_list:
        print(f"No valid JSON records found in {target_path}")
        return

    df = pd.DataFrame(data_list)
    output_path = target_path.parent / f"{target_path.name}.csv"
    df.to_csv(output_path, index=False)

    print(f"Successfully saved {len(df)} records to {output_path}")


if __name__ == "__main__":
    result_dir = Path("results")
    dirs = [
        result_dir / "adaptive_clustering_mpi_static",
        result_dir / "adaptive_clustering_mpi_rebalancing",
        result_dir / "adaptive_clustering_mpi_static_gpu",
        result_dir / "adaptive_clustering_mpi_rebalancing_gpu",
    ]
    json_to_csv(dirs)
    json_to_csv_cumrf(result_dir / "cuml_rf")
