import numpy as np
import torch

from falkon import Falkon, FalkonOptions
from falkon.kernels import GaussianKernel, PolynomialKernel
import time # Timing not built for GPUs




# ============================================================
# Rosenbrock function
# ============================================================
def rosenbrock_nd(X, a=1.0, b=100.0):
    """
    X : (n, d) numpy array
    Returns : (n, 1) numpy array
    Standard d-dimensional Rosenbrock:
        f(x) = sum_{i=1}^{d-1} [(a - x_i)^2 + b (x_{i+1} - x_i^2)^2]
    """
    X = np.asarray(X, dtype=np.float64)
    term1 = (a - X[:, :-1]) ** 2
    term2 = b * (X[:, 1:] - X[:, :-1] ** 2) ** 2
    y = np.sum(term1 + term2, axis=1, keepdims=True)
    return y


# ============================================================
# Sampling utilities
# ============================================================
def sample_uniform_box(n, d, low=-2.0, high=3.0, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, d))


def prepare_train_test_data(
    d=2,
    n_train=5000,
    n_test=5000,
    X_train=None,
    X_test=None,
    low=-2.0,
    high=3.0,
    seed=0,
    a=1.0,
    b=100.0,
):
    """
    Two modes:
      1) if X_train / X_test are provided, use them
      2) otherwise sample uniformly from [low, high]^d
    Returns:
      X_train, y_train, X_test, y_test   all as numpy arrays
    """
    if X_train is None:
        X_train = sample_uniform_box(n_train, d, low=low, high=high, seed=seed)
    else:
        X_train = np.asarray(X_train, dtype=np.float64)
        d = X_train.shape[1]

    if X_test is None:
        X_test = sample_uniform_box(n_test, d, low=low, high=high, seed=seed + 1)
    else:
        X_test = np.asarray(X_test, dtype=np.float64)

    y_train = rosenbrock_nd(X_train, a=a, b=b)
    y_test = rosenbrock_nd(X_test, a=a, b=b)

    return X_train, y_train, X_test, y_test


# ============================================================
# Metrics
# ============================================================
def regression_metrics(y_true, y_pred, eps=1e-12):
    """
    y_true, y_pred : shape (n,1) or (n,)
    Returns a dictionary with:
      - rel_l2
      - median_pointwise_rel
      - max_pointwise_rel
      - mse
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    diff = y_pred - y_true

    rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(y_true) + eps)
    pointwise_rel = np.abs(diff) / (np.abs(y_true) + eps)

    return {
        "rel_l2": rel_l2,
        "median_pointwise_rel": np.median(pointwise_rel),
        "max_pointwise_rel": np.max(pointwise_rel),
        "mse": np.mean(diff ** 2),
    }


# ============================================================
# Torch conversion
# ============================================================
def to_torch(X, device=None, dtype=torch.float32):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.tensor(X, dtype=dtype, device=device)


# ============================================================
# Falkon fit / predict
# ============================================================
def fit_falkon_regression(
    X_train,
    y_train,
    kernel=None,
    penalty=1e-6,
    M=1000,
    device=None,
    dtype=torch.float32,
    options=None,
):
    """
    Fits a Falkon regressor and returns:
      model, device_used
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if kernel is None:
        kernel = GaussianKernel(sigma=1.0)

    if options is None:
        options = FalkonOptions(
            use_cpu=(device == "cpu"),
            keops_active="auto",
        )

    Xtr_t = to_torch(X_train, device=device, dtype=dtype)
    ytr_t = to_torch(y_train, device=device, dtype=dtype)

    model = Falkon(
        kernel=kernel,
        penalty=penalty,
        M=M,
        options=options,
    )
    
    if device.startswith("cuda"):
        torch.cuda.synchronize()


    start = time.perf_counter()
    model.fit(Xtr_t, ytr_t)
    
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    end = time.perf_counter()

    train_time = end - start
    print("Training time:", train_time)

    return model, device


def predict_falkon(model, X, device=None, dtype=torch.float32):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    Xt = to_torch(X, device=device, dtype=dtype)
    pred_t = model.predict(Xt)
    print("Prediction shape:", pred_t.shape)

    if isinstance(pred_t, torch.Tensor):
        pred = pred_t.detach().cpu().numpy()
    else:
        pred = np.asarray(pred_t)
    return pred


# ============================================================
# Full experiment wrapper
# ============================================================
def run_rosenbrock_experiment(
    d=2,
    n_train=5000,
    n_test=2000,
    X_train=None,
    X_test=None,
    low=-2.0,
    high=3.0,
    seed=0,
    kernel=None,
    penalty=1e-6,
    M=1000,
    device=None,
    dtype=torch.float32,
    a=1.0,
    b=100.0,
):
    """
    Full pipeline:
      1) generate or accept train/test numpy arrays
      2) evaluate Rosenbrock on them
      3) fit Falkon
      4) predict
      5) compute train/test metrics

    Returns:
      results dict
    """
    X_train, y_train, X_test, y_test = prepare_train_test_data(
        d=d,
        n_train=n_train,
        n_test=n_test,
        X_train=X_train,
        X_test=X_test,
        low=low,
        high=high,
        seed=seed,
        a=a,
        b=b,
    )

    model, used_device = fit_falkon_regression(
        X_train=X_train,
        y_train=y_train,
        kernel=kernel,
        penalty=penalty,
        M=M,
        device=device,
        dtype=dtype,
    )

    yhat_train = predict_falkon(model, X_train, device=used_device, dtype=dtype)
    yhat_test = predict_falkon(model, X_test, device=used_device, dtype=dtype)

    train_metrics = regression_metrics(y_train, yhat_train)
    test_metrics = regression_metrics(y_test, yhat_test)

    return {
        "model": model,
        "device": used_device,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "yhat_train": yhat_train,
        "yhat_test": yhat_test,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    # -----------------------------------
    # Option 2: provide X_train, X_test manually
    # -----------------------------------
    dim = 5
    X_train_np = sample_uniform_box(int(1e7), dim, low=-2, high=3, seed=123)
    X_test_np = sample_uniform_box(50000, dim, low=-2, high=3, seed=456)

    print("Train Dims:", X_train_np.shape)
    print("Test Dims:", X_test_np.shape)
    res2 = run_rosenbrock_experiment(
        d=dim,
        X_train=X_train_np,
        X_test=X_test_np,
        kernel=GaussianKernel(sigma=1.0),
        penalty=1e-6,
        M=int(1e5),
        dtype=torch.float32,
    )

 
    print("\nProvided NumPy arrays")
    print("Device:", res2["device"])
    print("Train metrics:", res2["train_metrics"])
    print("Test metrics:", res2["test_metrics"])