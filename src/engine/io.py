"""Saving the points generated during training (per rank) and combining them afterwards."""

import glob
import os
import shutil
import uuid
from pathlib import Path

import numpy as np

from src import xp


def save_generated_data(X, filename):
    """Write ``X`` to a uniquely named file in ``saved_training_data/`` (safe for concurrent ranks)."""
    path = Path("saved_training_data/")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    xp.save(f"saved_training_data/{filename}_{uuid.uuid4()}.npy", X)


def combine_generated_data(filename):
    """Stack the files written by ``save_generated_data`` into ``<filename>.npy`` and delete the folder."""
    path = Path("saved_training_data/")
    input_dir = "saved_training_data"

    # Grab all .npy files in the folder with the correct beginning
    files = glob.glob(os.path.join(input_dir, f"{filename}_*.npy"))

    # Load all files into a list
    data_list = [np.load(f) for f in files]

    if data_list:
        # Stack them into one large matrix and save
        final_matrix = np.vstack(data_list)
        if path.exists():
            shutil.rmtree(input_dir)
        np.save(f"{filename}.npy", final_matrix)
    else:
        print("ERROR: No files found to stack.")
