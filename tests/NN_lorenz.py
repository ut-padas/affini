import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import random

from NN_utils import (
    CENTERS_2D,
    CENTERS_3D,
    CENTERS_4D,
    CENTERS_5D,
    shifted_gaussians,
    rosenbrock,
)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Global PARAMS
adam_lr = 1e-3
epochs = 30
log_every = 1
lambda_grad = 0.0
batch_size = 256
act = nn.SiLU
in_dim = 5
out_dim = 5
hidden_dim = 128
depth = 4
device = "cpu"
SEED = 42
set_global_seed(SEED)

print(f"Using device: {device}")

# Loading Training Data
f_path = r"/work/11328/skandan_subramanian/vista/hatann/ics_neural_network_train_evals/lorenz5D/lorenz5D_f.npz"
g_path = r"/work/11328/skandan_subramanian/vista/hatann/ics_neural_network_train_evals/lorenz5D/lorenz5D_g_npz"
f_data = np.load(f_path)
g_data = np.load(g_path)

# Test data
test_path = r"/work/11328/skandan_subramanian/vista/hatann/ics_neural_network_train_evals/TestData/LOR_5D.npz"
test_pts = np.load(test_path)
test_x = test_pts['X_test']
test_f = np.load("/work/11328/skandan_subramanian/vista/hatann/ics_neural_network_train_evals/TestData/LOR_5D_test_f.npz")['f']
print(f"Test X shape: {test_x.shape}, Test f shape: {test_f.shape}")

# Data Stats
x = np.array(f_data['X'])
f_vals = np.array(f_data['result'])
g_vals = np.array(g_data['result'])

#g_vals = g_vals.reshape(g_vals.shape[0], g_vals.shape[2])
print("X Data Points, dim: ", x.shape)
print("F Data Points, dim: ", f_vals.shape)
print("G Data Points, dim: ", g_vals.shape)

# 2d gaussian [-2, 2] x [-2, 2]
print(f"X range: {np.min(x[:, 0])}, {np.max(x[:, 0])}") 
print(f"Y range: {np.min(x[:, 1])}, {np.max(x[:, 1])}")

# ---- torch tensors ----
dtype = torch.float32

X_train = torch.tensor(x, dtype=dtype, device=device) # torch.float32 device = cpu
Y_train = torch.tensor(f_vals, dtype=dtype, device=device)
dY_train = torch.tensor(g_vals, dtype=dtype, device=device)
print("Y min/max:", Y_train.min().item(), Y_train.max().item())
print("dY min/max:", dY_train.abs().min().item(), dY_train.abs().max().item()) #this might be doing elementwise abs

# Setting TensorDataSet and DataLoader
g = torch.Generator()
g.manual_seed(SEED)

train_dataset = TensorDataset(X_train, Y_train, dY_train)

# DataLoader: mini-batches
# seed for shuffle
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=g,
    num_workers=0,   # important on Windows/Jupyter
    drop_last=False
)

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, depth=4, out_dim=1, activation=nn.SiLU): #sigmoid linear unit
        """
        depth = number of hidden layers
        """
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(activation())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Instantiate
# seed for weights inst.
model = MLP(in_dim=in_dim, hidden_dim=hidden_dim, depth=depth, out_dim=out_dim, activation=act).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable params:", n_params)

# Sanity check: one forward pass
with torch.no_grad(): # don't build graph of computation
    xb, yb, yb_grad = next(iter(train_loader))
    pred = model(xb.to(device))
    print("pred shape:", pred.shape)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)


def train(
    model,
    train_loader,
    optimizer,
    loss_fn,
    epochs=200,
    device="cpu",
    log_every=10,
    lam=0.0,
):
    model.to(device)
    model.train()
    history = []

    with_grad = lam > 0.0
    if with_grad:
        print("Jacobian loss is ON!")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_count = 0

        for Xb, Yb, dYb in train_loader:
            # Xb: (B,5), Yb: (B,5), dYb: (B,5,5)
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            dYb = dYb.to(device)

            if with_grad:
                # Need gradients w.r.t inputs to form Jacobian
                Xb = Xb.requires_grad_(True)

            pred = model(Xb)                  # (B,5)
            loss_val = loss_fn(pred, Yb)      # scalar

            loss_grad = 0.0
            if with_grad:
                B, d_in = Xb.shape            # d_in = 5
                d_out = pred.shape[1]         # d_out = 5

                # Build Jacobian: J[b, i, j] = d pred[b,i] / d Xb[b,j]
                J = torch.empty(B, d_out, d_in, device=device, dtype=pred.dtype)

                # Loop over output dims (5) and take grads
                for i in range(d_out):
                    # grad_i: (B, d_in) where grad_i[b,j] = d pred[b,i] / d Xb[b,j]
                    grad_i = torch.autograd.grad(
                        outputs=pred[:, i].sum(),
                        inputs=Xb,
                        create_graph=True,
                        retain_graph=True,   # needed because we call grad multiple times
                        only_inputs=True,
                    )[0]
                    J[:, i, :] = grad_i

                loss_grad = loss_fn(J, dYb)

            # Weight Jacobian loss (your old scaling used input dim; keep that pattern)
            loss = loss_val + Xb.shape[1] * lam * loss_grad

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = Xb.shape[0]
            total_loss += loss.item() * bs
            total_count += bs

        epoch_loss = total_loss / total_count
        history.append(epoch_loss)

        if epoch % log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs}  |  Loss: {epoch_loss:.6e}")

    return history

history = train(model, train_loader, optimizer, loss_fn, epochs=epochs, device=device, log_every=log_every, lam = lambda_grad)

# Error Plot
plt.figure(figsize=(6,4))
plt.plot(history)
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("train MSE (log scale)")
plt.title("Training loss")
plt.show()


def errors(model, test_x, test_f, device="cpu", dtype=torch.float32, eps=1e-12):
    """
    model: torch nn expects (N,5) -> (N,5)
    test_x: np.ndarray, shape (N,5)
    test_f: np.ndarray, shape (N,5)

    Pointwise relative error:
        ||pred_i - f_i||_2 / (||f_i||_2 + eps)

    Global L2 error:
        ||pred - f||_F
    """
    model.eval()

    # --- to torch ---
    X = torch.as_tensor(test_x, dtype=dtype, device=device)
    F = torch.as_tensor(test_f, dtype=dtype, device=device)

    # safety checks
    assert X.ndim == 2 and X.shape[1] == 5
    assert F.shape == X.shape

    with torch.no_grad():
        pred = model(X)
        assert pred.shape == F.shape  # (N,5)

    # --- compute errors ---
    diff = pred - F  # (N,5)

    # pointwise l2 norm per sample
    diff_norm = torch.linalg.norm(diff, dim=1)   # (N,)
    F_norm = torch.linalg.norm(F, dim=1)         # (N,)

    # pointwise relative l2 error
    rel = diff_norm / (F_norm + eps)              # (N,)

    max_rel = rel.max().item()
    median_rel = rel.median().item()

    # global l2 (Frobenius) error
    l2 = torch.linalg.norm(diff).item()
    l2_truth = torch.linalg.norm(F).item()
    rel_l2 = l2 / (l2_truth + eps)

    return {
        "max_relative_error": max_rel,
        "median_relative_error": median_rel,
        "l2_error": l2,
        "relative_l2_error": rel_l2,
    }

error_dict = errors(model, test_x, test_f, device=device, dtype=torch.float32, eps=1e-12)
print(f"Model Results: {error_dict}")

