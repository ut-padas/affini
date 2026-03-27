import json
import logging
import time
from typing import Any

import matplotlib.pyplot as plt
import optuna
from cuml.ensemble import RandomForestRegressor as CumlRF
from sklearn.ensemble import RandomForestRegressor as SklearnRF

from src import IS_CUPY_BACKEND, RESULTS_DIR, xp
from src.problems.rosenbrock import rosenbrock

logger = logging.getLogger(__name__)

SEED = 0
DOMAIN = [-2.0, 3.0]
N_TRAIN = int(1.74e5)
N_TRAIN = int(1.12e7)
N_VAL = int(5e4)
DIM = 5
EPS = 1e-10

N_TRIALS = 3
TUNING_FEATURE_BUDGET = int(8e5)
MIN_TUNING_SAMPLES = 100_000


def create_dataset():
    xp.random.seed(SEED)
    # Generate the training data (will be split 80/20 internally)
    X_train_full = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_TRAIN, DIM))
    Y_train_full = rosenbrock(X_train_full)

    # Generate the unseen validation data
    X_val = xp.random.uniform(DOMAIN[0], DOMAIN[1], size=(N_VAL, DIM))
    Y_val = rosenbrock(X_val)

    return X_train_full, Y_train_full, X_val, Y_val


def _as_column(y):
    """Ensure outputs are shaped as (n_samples, 1) for consistent metrics."""
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y


def compute_metrics(Y_pred, Y_val, n_samples):
    """Helper to compute metrics consistently for all models."""
    Y_pred = _as_column(Y_pred)
    Y_val = _as_column(Y_val)
    diff = Y_pred - Y_val

    norm_diff = xp.linalg.norm(diff)
    norm_gt = xp.linalg.norm(Y_val)
    l2_rel_error = float(norm_diff / (norm_gt + EPS))
    l2_abs_error = float(norm_diff / xp.sqrt(n_samples))

    element_rel_error = xp.abs(diff) / (xp.abs(Y_val) + EPS)
    ptwise_max_rel_error_linf = float(xp.nanmax(element_rel_error))
    ptwise_median_rel_error_linf = float(xp.nanmedian(element_rel_error))

    return {
        "l2_rel_error": l2_rel_error,
        "l2_abs_error": l2_abs_error,
        "ptwise_max_rel_error_linf": ptwise_max_rel_error_linf,
        "ptwise_median_rel_error_linf": ptwise_median_rel_error_linf,
    }


def get_tuning_sample_count(n_full: int, dim: int) -> int:
    """Cap tuning rows based on dimensionality while respecting dataset size."""
    dim_based_cap = TUNING_FEATURE_BUDGET // dim
    if n_full <= MIN_TUNING_SAMPLES:
        return n_full
    return max(MIN_TUNING_SAMPLES, min(n_full, dim_based_cap))


def train_and_evaluate_rf(X_train_full, Y_train_full, X_val, Y_val) -> dict[str, Any]:
    n_tune = get_tuning_sample_count(X_train_full.shape[0], X_train_full.shape[1])
    X_tune_full = X_train_full[:n_tune]
    Y_tune_full = Y_train_full[:n_tune]

    split_idx = int(0.8 * X_tune_full.shape[0])
    X_train, Y_train = X_tune_full[:split_idx], Y_tune_full[:split_idx]
    X_test, Y_test = X_tune_full[split_idx:], Y_tune_full[split_idx:]

    def objective(trial):
        if DIM == 2:
            max_depth = trial.suggest_int("max_depth", 2, 4)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 1000)
            n_estimators = trial.suggest_categorical("n_estimators", [50])
        elif DIM == 5:
            max_depth = trial.suggest_int("max_depth", 2, 9)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 1000)
            n_estimators = trial.suggest_categorical("n_estimators", [58])
        else:
            raise ValueError(f"Unsupported DIM={DIM} for RF tuning")

        if IS_CUPY_BACKEND:
            model = CumlRF(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=SEED,
            )
        else:
            model = SklearnRF(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=SEED,
                n_jobs=-1,
            )

        model.fit(X_train, Y_train.ravel())
        preds = _as_column(model.predict(X_test))
        rmse = float(xp.sqrt(xp.mean((preds - _as_column(Y_test)) ** 2)))
        return rmse

    rmse_history: list[float] = []

    def record_rmse_callback(study, trial):
        if trial.value is not None:
            rmse_history.append(float(trial.value))

    if DIM == 2:
        RF_START_PARAMS = {
            "max_depth": 4,
            "min_samples_split": 792,
            "n_estimators": 50,
        }
    elif DIM == 5:
        RF_START_PARAMS = {
            "max_depth": 9,
            "min_samples_split": 11,
            "n_estimators": 58,
        }
    else:
        raise ValueError(f"Unsupported DIM={DIM} for RF tuning")

    logger.info(
        "Starting RF Optuna Search (%d trials) on %d/%d rows (dim=%d)...",
        N_TRIALS,
        n_tune,
        X_train_full.shape[0],
        X_train_full.shape[1],
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.enqueue_trial(RF_START_PARAMS)
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        callbacks=[record_rmse_callback],
    )

    best_params = study.best_params
    print(f"\n--- RF Optuna Tuning Complete ---")
    print(f"Best RF Parameters: {best_params}\n")

    if IS_CUPY_BACKEND:
        final_model = CumlRF(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            random_state=SEED,
        )
    else:
        final_model = SklearnRF(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            random_state=SEED,
            n_jobs=-1,
        )

    logger.info("Training final RF model on full dataset...")
    start_train_time = time.perf_counter()
    final_model.fit(X_train_full, Y_train_full.ravel())
    train_time = time.perf_counter() - start_train_time

    logger.info("Evaluating final RF model on validation set...")
    start_eval_time = time.perf_counter()
    Y_pred = final_model.predict(X_val)
    eval_time = time.perf_counter() - start_eval_time

    metrics = compute_metrics(Y_pred, Y_val, X_val.shape[0])

    rf_leaves_per_tree: list[int] = []
    rf_total_leaves: int | None = None
    rf_avg_leaves_per_tree: float | None = None
    if hasattr(final_model, "estimators_"):
        for estimator in final_model.estimators_:
            if hasattr(estimator, "tree_"):
                n_leaves = getattr(estimator.tree_, "n_leaves", None)
                if n_leaves is not None:
                    rf_leaves_per_tree.append(int(n_leaves))

        if rf_leaves_per_tree:
            rf_total_leaves = int(sum(rf_leaves_per_tree))
            rf_avg_leaves_per_tree = float(rf_total_leaves / len(rf_leaves_per_tree))

    data_dict = {
        "model": "RandomForest",
        "IS_CUPY_BACKEND": IS_CUPY_BACKEND,
        "dim": int(X_train_full.shape[1]),
        "best_params": best_params,
        "rmse_history": rmse_history,
        "train_time_sec": train_time,
        "eval_time_sec": eval_time,
        "N_tune": n_tune,
        "N_train": int(X_train_full.shape[0]),
        "N_test": int(X_val.shape[0]),
        "rf_total_leaves": rf_total_leaves,
        "rf_avg_leaves_per_tree": rf_avg_leaves_per_tree,
        **metrics,
    }
    return data_dict


def save_results(data_dict):
    from pprint import pprint

    print(f"\n--- Final Metrics for {data_dict['model']} ---")
    pprint({k: v for k, v in data_dict.items() if k != "rmse_history"})

    OUTPUT_DIR = RESULTS_DIR / f"cuml_rf"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    f_prefix = f"{DIM}D_{data_dict['model']}_ntrain{int(data_dict['N_train'])}"

    rmse_history = data_dict.get("rmse_history", [])
    if rmse_history:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(
            range(1, len(rmse_history) + 1), rmse_history, marker="o", linewidth=1.5
        )
        ax.set_title(f"Optuna RMSE History - {data_dict['model']}")
        ax.set_xlabel("Trial")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        rmse_plot_file = OUTPUT_DIR / f"{f_prefix}_rmse_history.png"
        fig.savefig(rmse_plot_file, dpi=160)
        plt.close(fig)
        print(f"RMSE history plot saved to {rmse_plot_file}")

    output_file = OUTPUT_DIR / f"{f_prefix}_results.json"
    with open(output_file, "w") as f:
        json.dump(data_dict, f, indent=4)
    print(f"Results saved to {output_file}\n")


if __name__ == "__main__":
    from src.logging_utils import setup_logging

    setup_logging()

    logger.info("Creating datasets...")
    X_train_full, Y_train_full, X_val, Y_val = create_dataset()

    # 1. Random Forest
    rf_results = train_and_evaluate_rf(X_train_full, Y_train_full, X_val, Y_val)
    save_results(rf_results)

    # 2. XGBoost
    # xgb_results = train_and_evaluate_xgb(X_train_full, Y_train_full, X_val, Y_val)
    # save_results(xgb_results)
