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

        # 4. Save CSV with subdirectory name
        output_path = target_path.parent / f"{target_path.name}.csv"
        df.to_csv(output_path, index=False)

        print(f"Successfully saved {len(df)} records to {output_path}")


if __name__ == "__main__":
    result_dir = Path("results")
    dirs = [
        result_dir / "adaptive_clustering_mpi_static",
        result_dir / "adaptive_clustering_mpi_rebalancing",
    ]
    json_to_csv(dirs)
