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
from src.problems.lorenz96_flowmap import (
    lorenz96_flowmap,
    lorenz96_flowmap_grad
)
from ics26_experiments import (
    get_N_TRAIN,
    get_N_INIT_CLUSTERS,
    get_N_MIN_PER_CLUSTER
)
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
    from src.logging_utils import setup_logging
    setup_logging()

    # Parameters for surrogate training
    DIM = 5
    TOL = 0.02
    ERROR_METRIC = 'Linf-global'
    DOMAIN = [-5.0, 5.0]
    SEED = 0
    FORCING = 1
    DT = 0.1
    N_TRAIN = get_N_TRAIN(DIM)
    N_TIMESTEPS = 20
    N_INIT_CLUSTERS = get_N_INIT_CLUSTERS(DIM)
    N_MIN_PER_CLUSTER = get_N_MIN_PER_CLUSTER(DIM)

    # Modified functions for output variables of choice
    def f(x, output_var: int):
        f = lorenz96_flowmap(x, FORCING, DT)  # (N, d) vector
        N, d = f.shape
        return f[:, output_var].reshape((N, 1))

    def f_full(x):
        return lorenz96_flowmap(x, FORCING, DT)  # (N, d) vector

    def grad_f(x, output_var: int):
        g = lorenz96_flowmap_grad(x, FORCING, DT)
        N, d, _ = g.shape
        return g[:, output_var, :].reshape((N, 1, d))

    def grad_f_full(x):
        return lorenz96_flowmap_grad(x, FORCING, DT)


    # 1. Generate training data for surrogate (uniform along trajectories)
    X_train = generate_training_data_along_trajectories(SEED, f_full, N_TRAIN, N_TIMESTEPS)

    # 2. Train surrogate
    # We choose to train d one-d output surrogates, as it seems to scale better in practice.
    surrogates = [HATANNSurrogate(version='v2') for _ in range(DIM)]
    for i in range(DIM):
        lorenz_path = Path.cwd() / 'lorenz_surrogates'
        model_path = lorenz_path / f"lorenz96_dim{DIM}_F{FORCING}_tol{TOL}_tf{DT}_outputvar_{i}.npz"
        surrogates[i].train(
            X_train=X_train,
            f_fn=lambda x: f(x, i),
            g_fn=lambda x: grad_f(x, i),
            TOL=TOL,
            ERROR_METRIC=ERROR_METRIC,
            NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
            MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=20_000,
            SEED=SEED,
            USE_HESSIAN_INFORMED_SPLIT=True,
            IGNORE_GRADIENT=False,
            save_model=model_path,
            load_if_exists=True
        )

    '''
    # 3. Each surrogate is trained. Pick some initial points.
    # First, parameters for testing
    N_TRAJECTORIES = 100
    TF = 3.0

    # We take a slightly smaller cube because trajectories can go outside the domain
    x0 = xp.random.uniform(-5, 5, (N_TRAJECTORIES, DIM)) + FORCING

    # Initialize time stepping
    tvec = xp.arange(0, TF+DT, DT)
    yvec_surrogate = xp.empty((len(tvec), N_TRAJECTORIES, DIM))
    # dim 0: time steps, dim 1: trajectories, dim 2: dimension.
    yvec_surrogate[0] = x0
    yvec_exact = xp.empty((len(tvec), N_TRAJECTORIES, DIM))
    yvec_exact[0] = x0

    # Step in time
    for k in range(1, len(tvec)):

        # Surrogates evaluation
        for i in range(DIM):
            pred, unguar_mask = surrogates[i].predict(yvec_surrogate[k-1,:,:])
            yvec_surrogate[k,:,i] = pred.flatten()
            # We should definitely evaluate exactly the unguaranteed points here
            # But for now, ignoring for simplicity

        # Exact evaluation
        yvec_exact[k,:,:] = f_full(yvec_exact[k-1,:,:])

    # Save data (to analyze in notebook)
    directory = ICS26_DIR / 'lorenz_successive_timestep_data'
    directory.mkdir(parents=True, exist_ok=True)
    xp.savez(directory / 'trajectory_data.npz', exact=yvec_exact, surrogate=yvec_surrogate)
    '''
