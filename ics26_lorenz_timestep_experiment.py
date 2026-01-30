from functools import partial
from itertools import product
from pathlib import Path
from tqdm import tqdm
from experiments_utils import (
    ICS26_DIR,
    Problem_experiment_setup,
    create_summary_dataframe,
    del_all_error_files_in_dir,
)
from src.problems.lorenz96_flowmap import lorenz96_flowmap, lorenz96_flowmap_grad






if __name__ == '__main__':

    # Chaining together multiple time steps.

    # Parameters for surrogate training
    
    
