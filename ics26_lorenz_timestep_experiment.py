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
from ics26_experiments import get_N_TRAIN
from src import IS_CUPY_BACKEND, RESULTS_DIR, Array, xp
from src.surrogate import HATANNSurrogate


# Helper functions

def generate_training_data_along_trajectories(seed, f, N_train, N_timesteps):

    xp.random.seed(SEED)
    N_train_init = N_train // N_timesteps

    # Save as tensor, then flatten at the end
    X_train = xp.zeros(shape=(N_timesteps, N_train_init, DIM))

    # Take time steps
    X_train[0] = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_train_init, DIM))
    for j in range(1, N_timesteps):
        X_train[j] = f(X_train[j - 1])

    X_train = X_train.reshape((N_train_init * N_timesteps, DIM))

    return X_train




if __name__ == '__main__':

    # Chaining together multiple time steps.

    # Parameters for surrogate training
    DIM = 10
    TOL = 0.01
    ERROR_METRIC = 'Linf_local'
    DOMAIN = [-5.0, 5.0]
    SEED = 0
    FORCING = 1
    FINAL_TIME = 0.2
    N_TRAIN = get_N_TRAIN(DIM)
    N_TIMESTEPS = 20

    # Modified functions for output variables of choice
    def f(x, output_var: int):
        f = lorenz96_flowmap(x, FORCING, FINAL_TIME)  # (N, d) vector
        N, d = f.shape
        return f[:, output_var].reshape((N, 1))

    def f_full(x):
        return lorenz96_flowmap(x, FORCING, FINAL_TIME)  # (N, d) vector

    def grad_f(x, output_var: int):
        g = lorenz96_flowmap_grad(x, FORCING, FINAL_TIME)
        N, d, _ = g.shape
        return g[:, output_var, :].reshape((N, 1, d))

    def grad_f_full(x):
        return lorenz96_flowmap_grad(x, FORCING, FINAL_TIME)


    # 1. Generate training data for surrogate (uniform along trajectories)
    X_train = generate_training_data_along_trajectories(SEED, f_full, N_TRAIN, N_TIMESTEPS)

    # 2. Train surrogate
    # We choose to train d one-d output surrogates, as it seems to scale better in practice.
    surrogates = [HATANNSurrogate(version=CLUSTERING_VERSION)]
    
    
