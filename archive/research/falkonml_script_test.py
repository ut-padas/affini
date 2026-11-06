from falkon import Falkon, kernels
import torch
import time


def sample_uniform_nd(n, d, range, device="cpu"):
    """
    Samples N points in a d-dimensional hypercube [low, high]^d.
    """
    # Generate points in [0, 1) and scale to [low, high]
    low = range[0]
    high = range[1]
    return (high - low) * torch.rand(n, d, device=device) + low


def rosenbrock(x):
    """
    Computes the Rosenbrock function.

    Args:
        x (torch.Tensor): A tensor of shape (..., D), where D is the dimensionality.
                          The function is evaluated over the last dimension.

    Returns:
        torch.Tensor: The value of the Rosenbrock function at each point.
    """
    # x_i
    x_i = x[..., :-1]
    # x_{i+1}
    x_next = x[..., 1:]

    # Summation terms
    term1 = 100 * (x_next - x_i**2) ** 2
    term2 = (x_i - 1) ** 2

    return torch.sum(term1 + term2, dim=-1)


def rosenbrock_ridge_regression_example(n_train=1000, n_test=1000, d=2):
    # Generate training data data
    torch.manual_seed(67)
    train_x = sample_uniform_nd(n_train, d, [-2, 3])
    train_y = rosenbrock(train_x)

    # Generate testing data
    test_x = sample_uniform_nd(n_test, d, [-2, 3])
    test_y = rosenbrock(test_x)

    # Put everything on the GPU
    # if torch.cuda.is_available():
    #     train_x = train_x.cuda()
    #     train_y = train_y.cuda()
    #     test_x = test_x.cuda()
    #     test_y = test_y.cuda()

    # Parameters
    sigma = 1.0
    k = kernels.GaussianKernel(sigma)

    def mean_squared_error(true, pred):
        return torch.mean((true - pred) ** 2)

    model = Falkon(kernel=k, penalty=1e-6, M=5000, seed=67, error_fn=mean_squared_error)

    # Train
    start_train = time.perf_counter()
    model.fit(train_x, train_y)
    end_train = time.perf_counter()
    time_train = end_train - start_train

    # Predict
    start_pred = time.perf_counter()
    preds = model.predict(test_x).flatten()
    end_pred = time.perf_counter()
    time_pred = end_pred - start_pred
    rel_err = torch.linalg.norm(test_y - preds, 2) / torch.linalg.norm(test_y, 2)
    print(f"Training time: {time_train:.4f} seconds")
    print(f"Inference time: {time_pred:.4f} seconds")
    print(f"Relative L2 Error: {rel_err}")


if __name__ == "__main__":
    rosenbrock_ridge_regression_example(n_train=776, n_test=50000, d=2)
