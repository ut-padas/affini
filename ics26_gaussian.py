import json
import time
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np

from ics26_experiments import (
    ICS26_DIR,
    SEED,
    create_problem_result_dict,
    create_summary_dataframe,
    plot_error_histogram,
)
from src import xp
from src.problems.shifted_gaussians import (
    CENTERS_2D,
    CENTERS_3D,
    shifted_gaussians,
    shifted_gaussians_grad,
)
from src.surrogate import NearODESurrogate


def SGaussian_experiment_setup(
    json_fpath: Path,
    DIM: int,
    N_TRAIN: int,
    N_INIT_CLUSTERS: int,
    TOL: float,
    ERROR_SCALING_METHOD: str,
    N_MIN_PER_CLUSTER: int,
    MAX_CLUSTERS_EST: int,
    N_TEST: int,
    K_NEIGHBOURS: int,
    CONTAINMENT_LEVEL: float,
    ENFORCE_CONTAINMENT: bool,
    Q_SRINK_FLAG: bool,
    SYNTH_BUDGET_FACTOR: int = 10,
):
    if DIM == 2:
        centers = CENTERS_2D
        sg_f = partial(shifted_gaussians, centers=centers)
        sg_grad_f = partial(shifted_gaussians_grad, centers=centers)
    elif DIM == 3:
        centers = CENTERS_3D
        sg_f = partial(shifted_gaussians, centers=centers)
        sg_grad_f = partial(shifted_gaussians_grad, centers=centers)
    else:
        raise ValueError("DIM must be 2 or 3")

    # Sample training data according to N(0, 2I)
    rng = xp.random.default_rng(SEED)
    X_train = rng.normal(0, np.sqrt(2), size=(N_TRAIN, DIM))

    surrogate = NearODESurrogate()
    error_tag = ""

    FAILED = False
    _TRAINED, _PREDICTED = False, False
    try:
        Q_SHRINK = 0.05 if Q_SRINK_FLAG else -1.0

        # Train
        print("\n--- Training ---")
        _start_train_time = time.perf_counter()
        surrogate.train(
            X_train=X_train,
            f_fn=sg_f,
            g_fn=sg_grad_f,
            TOL=TOL,
            ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,  # type: ignore
            NUM_INITIAL_CLUSTERS=N_INIT_CLUSTERS,
            MIN_PTS_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            SEED=SEED,
            SYNTH_BUDGET_FACTOR=SYNTH_BUDGET_FACTOR,
            Q_SHRINK=Q_SHRINK,
        )
        _end_train_time = time.perf_counter()
        _TRAINED = True

        # Evaluate
        print("--- Evaluation ---")
        # Sample test data according to N(0, 2I)
        X_test = rng.normal(0, np.sqrt(2), size=(N_TEST, DIM))
        ground_truth = sg_f(X_test)

        # Example utilizing the new signature with Guarantee checks
        _start_predict_time = time.perf_counter()
        preds, ungauranteed_mask = surrogate.predict(
            X_test,
            k_NEIGHBOURS=K_NEIGHBOURS,
            enforce_containment=ENFORCE_CONTAINMENT,
            containment_level=CONTAINMENT_LEVEL,
        )
        _end_predict_time = time.perf_counter()
        _PREDICTED = True

        train_time = _end_train_time - _start_train_time
        predict_time = _end_predict_time - _start_predict_time

        N_CLUSTERS = surrogate.get_number_of_sources()
        num_levels = surrogate.training_num_levels
    except Exception as e:
        FAILED = True
        error_tag = f"Experiment failed with exception: {e} | TRAINED={_TRAINED}, PREDICTED={_PREDICTED}"
        print(error_tag)

        error_path = json_fpath.with_suffix(".error.txt")
        with open(error_path, "w") as ef:
            ef.write(error_tag)

        ground_truth = None
        preds = None
        ungauranteed_mask = None
        train_time = None
        predict_time = None
        N_CLUSTERS = None
        num_levels = None

    result_dict = create_problem_result_dict(
        problem_name="ShiftedGaussians",
        problem_tag=f"{DIM}D_N{N_TRAIN}_InitClust{N_INIT_CLUSTERS}_Tol{TOL}_ErrScale{ERROR_SCALING_METHOD}_MinPts{N_MIN_PER_CLUSTER}_MaxClustEst{MAX_CLUSTERS_EST}_K{K_NEIGHBOURS}_ContainLvl{CONTAINMENT_LEVEL}_EnfContain{ENFORCE_CONTAINMENT}_QShrink{Q_SRINK_FLAG}",
        error_tag=error_tag,
        train_time=train_time,
        predict_time=predict_time,
        ground_truth=ground_truth,
        predictions=preds,
        ungauranteed_mask=ungauranteed_mask,
        num_levels=num_levels,
        TOL=TOL,
        DIM=DIM,
        N_TRAIN=N_TRAIN,
        N_CLUSTERS=N_CLUSTERS,
        ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
        N_INIT_CLUSTERS=N_INIT_CLUSTERS,
        N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
        K_NEIGHBOURS=K_NEIGHBOURS,
        ENFORCE_CONTAINMENT=ENFORCE_CONTAINMENT,
        CONTAINMENT_LEVEL=CONTAINMENT_LEVEL,
        Q_SRINK_FLAG=Q_SRINK_FLAG,
        SYNTH_BUDGET_FACTOR=SYNTH_BUDGET_FACTOR,
    )

    with open(json_fpath, "w") as jf:
        json.dump(result_dict, jf, indent=4)

    if FAILED:
        return None

    plot_error_histogram(
        ground_truth=ground_truth, predictions=preds, json_fpath=json_fpath, TOL=TOL
    )


def experiment_no_q_shrink() -> Path:
    N_INIT_CLUSTERS = 25
    N_MIN_PER_CLUSTER = 100
    MAX_CLUSTERS_EST = 10_000
    K_NEIGHBOURS = 1
    CONTAINMENT_LEVEL = 1.0
    ENFORCE_CONTAINMENT = False
    ERROR_SCALING_METHOD = "global"

    EXP_DIR = ICS26_DIR / "shifted_gaussians_no_q_shrink_global_error_scaling"
    TOLs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4]
    DIMs = [2, 3]

    for DIM, TOL in product(DIMs, TOLs):
        json_path = EXP_DIR / f"ics26_sgaussian_dim{DIM}_tol{TOL}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            print(f"Skipping existing experiment at {json_path}")
            continue

        if DIM == 2:
            N_TRAIN = 200_000
            N_TEST = 20_000
        elif DIM == 3:
            N_TRAIN = 300_000
            N_TEST = 30_000
        else:
            raise ValueError("DIM must be 2 or 3")

        SGaussian_experiment_setup(
            json_fpath=json_path,
            DIM=DIM,
            N_TRAIN=N_TRAIN,
            N_INIT_CLUSTERS=N_INIT_CLUSTERS,
            TOL=TOL,
            ERROR_SCALING_METHOD=ERROR_SCALING_METHOD,
            N_MIN_PER_CLUSTER=N_MIN_PER_CLUSTER,
            MAX_CLUSTERS_EST=MAX_CLUSTERS_EST,
            N_TEST=N_TEST,
            K_NEIGHBOURS=K_NEIGHBOURS,
            CONTAINMENT_LEVEL=CONTAINMENT_LEVEL,
            ENFORCE_CONTAINMENT=ENFORCE_CONTAINMENT,
            Q_SRINK_FLAG=False,
        )

    return EXP_DIR


if __name__ == "__main__":
    from src.logging_config import setup_logging

    setup_logging()
    dir1 = experiment_no_q_shrink()
    create_summary_dataframe(dir1)
