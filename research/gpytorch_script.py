import numpy as np
import math
import torch
import gpytorch
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from src.problems.lorenz96_flowmap import lorenz96_flowmap_oracle_first_component


"""
GP Models
"""


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


"""
GP functions
"""


def train(model, likelihood, train_x, train_y, training_iter=50):
    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        # print("Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f" % (
        #     i + 1, training_iter, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.squeeze()[0],
        #     model.covar_module.base_kernel.lengthscale.squeeze()[1],
        #     model.likelihood.noise.item()
        # ))
        optimizer.step()


def predict(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with (
        torch.no_grad(),
        gpytorch.settings.fast_pred_var(),
        gpytorch.settings.max_root_decomposition_size(50),
    ):
        return likelihood(model(test_x))


def plot(observed_pred, train_x, train_y, test_x=torch.linspace(0, 1, 51)):
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), "k*")
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), "b")
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(["Observed Data", "Mean", "Confidence"])
        f.savefig("gp_plot.png")


"""
Other Rosenbrock-related helper functions
"""


def sample_uniform_nd(n, d, range, device="cpu"):
    """
    Samples N points in a d-dimensional hypercube [low, high]^d.
    """
    # Generate points in [0, 1) and scale to [low, high]
    low = range[0]
    high = range[1]
    return (high - low) * torch.rand(n, d, dtype=torch.float, device=device) + low


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


def rosenbrock_grad(x):
    """
    Computes the analytical gradient of the Rosenbrock function.

    Args:
        x (torch.Tensor): Input tensor of shape (..., D)
    Returns:
        torch.Tensor: Gradient tensor of the same shape as x.
    """
    xm = x[..., :-1]  # x_i
    xn = x[..., 1:]  # x_{i+1}
    grad = torch.zeros_like(x)

    # Common term used in the partial derivatives
    common = -400 * xm * (xn - xm**2)

    # Gradient for i=1 to d-1
    grad[..., :-1] += common + 2 * (xm - 1)

    # Gradient for i=2 to d
    grad[..., 1:] += 200 * (xn - xm**2)

    return grad


def rosenbrock_2d_combined(X):
    f = rosenbrock(X)
    grad_f = rosenbrock_grad(X)
    return f, grad_f[:, 0], grad_f[:, 1]


def rosenbrock_5d_combined(X):
    f = rosenbrock(X)
    grad_f = rosenbrock_grad(X)
    return torch.column_stack((f, grad_f))


"""
Examples
"""


def simple_gp_regression_example():
    """
    Implements the most basic notebook that is available on the GPyTorch GitHub page.
    """

    # Generate training data (y = sin(2*pi*x) + eps_noise)
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(
        train_x.size()
    ) * math.sqrt(0.04)

    # Setup the GP regression model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # Train the GP regression model: go into train mode
    model.train()
    likelihood.train()
    # Then train
    train(model, likelihood, train_x, train_y)

    # Make predictions: go into evaluation mode
    model.eval()
    likelihood.eval()

    # Test points are evenly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    observed_pred = predict(model, likelihood)

    # Plot the model fit
    plot(observed_pred, train_x, train_y)


def simple_gp_with_derivatives_example():
    # Define function
    def franke(X, Y):
        term1 = 0.75 * torch.exp(-((9 * X - 2).pow(2) + (9 * Y - 2).pow(2)) / 4)
        term2 = 0.75 * torch.exp(-((9 * X + 1).pow(2)) / 49 - (9 * Y + 1) / 10)
        term3 = 0.5 * torch.exp(-((9 * X - 7).pow(2) + (9 * Y - 3).pow(2)) / 4)
        term4 = 0.2 * torch.exp(-(9 * X - 4).pow(2) - (9 * Y - 7).pow(2))

        f = term1 + term2 + term3 - term4
        dfx = (
            -2 * (9 * X - 2) * 9 / 4 * term1
            - 2 * (9 * X + 1) * 9 / 49 * term2
            + -2 * (9 * X - 7) * 9 / 4 * term3
            + 2 * (9 * X - 4) * 9 * term4
        )
        dfy = (
            -2 * (9 * Y - 2) * 9 / 4 * term1
            - 9 / 10 * term2
            + -2 * (9 * Y - 3) * 9 / 4 * term3
            + 2 * (9 * Y - 7) * 9 * term4
        )

        return f, dfx, dfy

    # Define training data
    xv, yv = torch.meshgrid(
        torch.linspace(0, 1, 10), torch.linspace(0, 1, 10), indexing="ij"
    )
    train_x = torch.cat(
        (xv.contiguous().view(xv.numel(), 1), yv.contiguous().view(yv.numel(), 1)),
        dim=1,
    )

    f, dfx, dfy = franke(train_x[:, 0], train_x[:, 1])
    train_y = torch.stack([f, dfx, dfy], -1).squeeze(1)

    train_y += 0.05 * torch.randn(
        train_y.size()
    )  # Add noise to both values and gradients

    # Set up model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=3
    )  # Value + x-derivative + y-derivative
    model = GPModelWithDerivatives(train_x, train_y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.05
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print(
            "Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f"
            % (
                i + 1,
                training_iter,
                loss.item(),
                model.covar_module.base_kernel.lengthscale.squeeze()[0],
                model.covar_module.base_kernel.lengthscale.squeeze()[1],
                model.likelihood.noise.item(),
            )
        )
        optimizer.step()

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Initialize plots
    fig, ax = plt.subplots(2, 3, figsize=(14, 10))

    # Test points
    n1, n2 = 50, 50
    xv, yv = torch.meshgrid(
        torch.linspace(0, 1, n1), torch.linspace(0, 1, n2), indexing="ij"
    )
    f, dfx, dfy = franke(xv, yv)

    # Make predictions
    with (
        torch.no_grad(),
        gpytorch.settings.fast_computations(
            log_prob=False, covar_root_decomposition=False
        ),
    ):
        test_x = torch.stack(
            [xv.reshape(n1 * n2, 1), yv.reshape(n1 * n2, 1)], -1
        ).squeeze(1)
        predictions = likelihood(model(test_x))
        mean = predictions.mean

    extent = (xv.min(), xv.max(), yv.max(), yv.min())
    ax[0, 0].imshow(f, extent=extent, cmap=cm.get_cmap("jet"))
    ax[0, 0].set_title("True values")
    ax[0, 1].imshow(dfx, extent=extent, cmap=cm.get_cmap("jet"))
    ax[0, 1].set_title("True x-derivatives")
    ax[0, 2].imshow(dfy, extent=extent, cmap=cm.get_cmap("jet"))
    ax[0, 2].set_title("True y-derivatives")

    ax[1, 0].imshow(
        mean[:, 0].detach().numpy().reshape(n1, n2),
        extent=extent,
        cmap=cm.get_cmap("jet"),
    )
    ax[1, 0].set_title("Predicted values")
    ax[1, 1].imshow(
        mean[:, 1].detach().numpy().reshape(n1, n2),
        extent=extent,
        cmap=cm.get_cmap("jet"),
    )
    ax[1, 1].set_title("Predicted x-derivatives")
    ax[1, 2].imshow(
        mean[:, 2].detach().numpy().reshape(n1, n2),
        extent=extent,
        cmap=cm.get_cmap("jet"),
    )
    ax[1, 2].set_title("Predicted y-derivatives")

    fig.savefig("simple_gp_with_derivatives_test.png")

    # EXTRA: test L2 error
    diff = f.flatten() - mean[:, 0]
    rel_err = torch.linalg.norm(diff) / torch.linalg.norm(f.flatten())
    print(f"GP Surrogate Relative L2 Error: {rel_err}")


def rosenbrock_gp_regression_example(n_train=10000, n_test=1000, d=2):
    """
    Adapts the simple notebook to the 2D Rosenbrock function.
    """

    # Generate training data
    # Do not add any noise to the model! Maybe we should but let's see how it works

    train_x = sample_uniform_nd(n_train, d, [-2, 3])
    train_y = rosenbrock(train_x)

    # Setup the GP regression model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # CUDA!
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    likelihood = likelihood.cuda()
    model = model.cuda()

    # Train the model
    model.train()
    likelihood.train()
    start_train = time.perf_counter()
    train(model, likelihood, train_x, train_y, training_iter=50)
    end_train = time.perf_counter()
    time_train = end_train - start_train

    # Predict
    test_x = sample_uniform_nd(n_test, d, [-2, 3])
    exact_y = rosenbrock(test_x)
    test_x_cuda = test_x.cuda()
    start_pred = time.perf_counter()
    observed_pred = predict(model, likelihood, test_x_cuda)
    pred_y = observed_pred.mean.cpu().numpy()
    end_pred = time.perf_counter()
    time_pred = end_pred - start_pred

    # Compare to exact
    error = exact_y - pred_y
    abs_err = np.linalg.norm(error, 2)
    rel_err = abs_err / np.linalg.norm(exact_y, 2)

    print(f"Time to train: {time_train:.4f} seconds")
    print(f"Relative L2 error of Gaussian process model: {rel_err:.4f}")
    print(f"Time for inference: {time_pred:.4f} seconds")

    return torch.tensor([n_train, rel_err, time_train, time_pred])


def rosenbrock_2d_gp_with_derivatives(n_train=1000):
    """
    Implements the GP regression with derivative information, for Rosenbrock.
    This is in 08_Advanced_Usage.
    """

    # Generate training data
    torch.manual_seed(0)
    d = 2
    train_x = sample_uniform_nd(n_train, d, [-2, 3])

    f, dfx, dfy = rosenbrock_2d_combined(train_x)
    train_y = torch.stack([f, dfx, dfy], -1).squeeze(1)

    # Model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=3
    )  # Value + x-derivative + y-derivative
    model = GPModelWithDerivatives(train_x, train_y, likelihood)

    # Train

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.05
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    start_train = time.perf_counter()
    training_iter = 1000
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print(
            "Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f"
            % (
                i + 1,
                training_iter,
                loss.item(),
                model.covar_module.base_kernel.lengthscale.squeeze()[0],
                model.covar_module.base_kernel.lengthscale.squeeze()[1],
                model.likelihood.noise.item(),
            )
        )
        optimizer.step()
    end_train = time.perf_counter()

    # Predict

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Initialize plots
    fig, ax = plt.subplots(2, 3, figsize=(14, 10))

    # Test points
    n1, n2 = 50, 50
    xv, yv = torch.meshgrid(
        torch.linspace(-2, 3, n1), torch.linspace(-2, 3, n2), indexing="ij"
    )
    test_x = torch.column_stack((xv.flatten(), yv.flatten()))
    f, dfx, dfy = rosenbrock_2d_combined(test_x)

    # Make predictions
    start_pred = time.perf_counter()
    with (
        torch.no_grad(),
        gpytorch.settings.fast_computations(
            log_prob=False, covar_root_decomposition=False
        ),
    ):
        predictions = likelihood(model(test_x))
        mean = predictions.mean
    end_pred = time.perf_counter()

    extent = (-2, 3, 3, -2)
    ax[0, 0].imshow(f.reshape(n1, n2), extent=extent, cmap=cm.get_cmap("jet"))
    ax[0, 0].set_title("True values")
    ax[0, 1].imshow(dfx.reshape(n1, n2), extent=extent, cmap=cm.get_cmap("jet"))
    ax[0, 1].set_title("True x-derivatives")
    ax[0, 2].imshow(dfy.reshape(n1, n2), extent=extent, cmap=cm.get_cmap("jet"))
    ax[0, 2].set_title("True y-derivatives")

    ax[1, 0].imshow(
        mean[:, 0].detach().numpy().reshape(n1, n2),
        extent=extent,
        cmap=cm.get_cmap("jet"),
    )
    ax[1, 0].set_title("Predicted values")
    ax[1, 1].imshow(
        mean[:, 1].detach().numpy().reshape(n1, n2),
        extent=extent,
        cmap=cm.get_cmap("jet"),
    )
    ax[1, 1].set_title("Predicted x-derivatives")
    ax[1, 2].imshow(
        mean[:, 2].detach().numpy().reshape(n1, n2),
        extent=extent,
        cmap=cm.get_cmap("jet"),
    )
    ax[1, 2].set_title("Predicted y-derivatives")

    fig.tight_layout()
    # fig.savefig('rosenbrock_grad_gp.png')

    # Compute surrogate error
    abs_err = torch.linalg.norm(f - mean[:, 0], 2)
    rel_err = abs_err / torch.linalg.norm(f, 2)
    print(f"Time to train model: {(end_train - start_train):.4f}")
    print(f"Inference time: {(end_pred - start_pred):.4f}")
    print(f"GP Surrogate Rel. L2 Error: {rel_err:.4f}")


def rosenbrock_5d_gp_with_derivatives(n_train=1000, n_test=2000):
    """
    Implements the GP regression with derivative information, for Rosenbrock.
    This is in 08_Advanced_Usage.
    """

    # Generate training data
    torch.manual_seed(0)
    d = 5
    train_x = sample_uniform_nd(n_train, d, [-2, 3])
    train_y = rosenbrock_5d_combined(train_x)

    # Model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=(d + 1)
    )  # Value + x-derivative + y-derivative
    model = GPModelWithDerivatives(train_x, train_y, likelihood)

    # Train

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.05
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 100
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print(
            "Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f"
            % (
                i + 1,
                training_iter,
                loss.item(),
                model.covar_module.base_kernel.lengthscale.squeeze()[0],
                model.covar_module.base_kernel.lengthscale.squeeze()[1],
                model.likelihood.noise.item(),
            )
        )
        optimizer.step()

    # Predict

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Test points
    test_x = sample_uniform_nd(n_test, d, [-2, 3])
    test_y = rosenbrock_2d_combined(test_x)
    f = test_y[:, 0]

    # Make predictions
    with (
        torch.no_grad(),
        gpytorch.settings.fast_computations(
            log_prob=False, covar_root_decomposition=False
        ),
    ):
        predictions = likelihood(model(test_x))
        mean = predictions.mean

    # Compute surrogate error
    abs_err = torch.linalg.norm(f - mean[:, 0], 2)
    rel_err = abs_err / torch.linalg.norm(f, 2)
    print(f"GP Surrogate Rel. L2 Error: {rel_err:.4f}")


def lorenz_gp_regression(n_train=50000, n_test=50000, d=5):
    """
    Adapts the simple notebook to the 2D Rosenbrock function.
    """

    # Generate training data
    # Do not add any noise to the model! Maybe we should but let's see how it works

    train_x = sample_uniform_nd(n_train, d, [-5, 5], device="cpu")
    train_y, _ = lorenz96_flowmap_oracle_first_component(
        train_x.numpy(), forcing=1, tf=0.1
    )
    train_y = torch.from_numpy(train_y.flatten().get()).to(torch.float)

    # Setup the GP regression model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # CUDA!
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    likelihood = likelihood.cuda()
    model = model.cuda()

    # Train the model
    model.train()
    likelihood.train()
    start_train = time.perf_counter()
    train(model, likelihood, train_x, train_y, training_iter=50)
    end_train = time.perf_counter()
    time_train = end_train - start_train

    # Predict
    test_x = sample_uniform_nd(n_test, d, [-2, 3], device="cpu")
    exact_y, _ = lorenz96_flowmap_oracle_first_component(test_x.numpy())
    exact_y = torch.from_numpy(exact_y.flatten().get()).to(torch.float)
    test_x_cuda = test_x.cuda()
    start_pred = time.perf_counter()
    observed_pred = predict(model, likelihood, test_x_cuda)
    pred_y = observed_pred.mean.cpu().numpy()
    end_pred = time.perf_counter()
    time_pred = end_pred - start_pred

    # Compare to exact
    error = exact_y - pred_y
    abs_err = np.linalg.norm(error, 2)
    rel_err = abs_err / np.linalg.norm(exact_y, 2)

    print(f"Time to train: {time_train:.4f} seconds")
    print(f"Relative L2 error of Gaussian process model: {rel_err:.4f}")
    print(f"Time for inference: {time_pred:.4f} seconds")

    return torch.tensor([n_train, rel_err, time_train, time_pred])


if __name__ == "__main__":
    # print('\n----- d = 2 -----')
    # gpytorch.settings.memory_efficient(True)
    # n_train_vec = np.array([60000])
    # on_first = True
    # for n_train in n_train_vec:
    #     print(f'Training with {n_train} data points')
    #     table_row = rosenbrock_gp_regression_example(n_train=n_train, n_test=50000, d=2)

    table_row = lorenz_gp_regression(n_train=1000, n_test=50000)
    print(table_row)
