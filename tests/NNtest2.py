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
    shifted_gaussians_grad,
    rosenbrock,
    rosenbrock_grad,
)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Global PARAMS
adam_lr = 1e-3
epochs = 50
log_every = 1
lambda_grad = 1.0
lam_w = 0.0 # weight deacay on all trainable params
batch_size = 256
act = nn.SiLU
in_dim = 5
out_dim = 1
hidden_dim = 128
depth = 4
device = "cpu"
SEED = 42
set_global_seed(SEED)

print(f"Using device: {device}")

# Loading Training Data
f_path = r"/work/11328/skandan_subramanian/vista/hatann/ics_neural_network_train_evals/lorenz5D/Lorenz96_d5_NEW.npz"
# g_path = r"/work/11328/skandan_subramanian/vista/hatann/ics_neural_network_train_evals/rosenbrock_5d_tol0.05/rosenbrock_5d_tol0.05_g.npz"
f_data = np.load(f_path)
# g_data = np.load(g_path)
print(list(f_data.keys()))

# Test data
test_path = r"/work/11328/skandan_subramanian/vista/hatann/ics_neural_network_train_evals/TestData/Lorenz96_d5_NEW_testdata.npz"
test_pts = np.load(test_path)
print(list(test_pts.keys())) 
test_x = test_pts['X_test']
test_f = test_pts['F_test']
test_f = test_f.reshape(test_f.shape[0], 1)
print(f"Test X shape: {test_x.shape}, Test f shape: {test_f.shape}")

# Data Stats
x = np.array(f_data['all'])
# x = np.random.uniform(low=-2.0, high=2.0, size=x_praj.shape)  # Using uniform samples in [-2, 2]^d
f_vals = f_data['F_all']
g_vals = f_data['G_all']
print(g_vals.shape)

# f_vals = np.array(f_data['F_centroids'])
# g_vals = np.array(f_data['G_centroids'])

#g_vals = g_vals.reshape(g_vals.shape[0], g_vals.shape[2])
f_vals = f_vals.reshape(f_vals.shape[0], 1) # ensure (N,1) for consistent training
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

# class MLP(nn.Module):
#     def __init__(self, in_dim=2, hidden_dim=128, depth=4, out_dim=1, activation=nn.SiLU): #sigmoid linear unit
#         """
#         depth = number of hidden layers
#         """
#         super().__init__()
#         layers = []
#         d = in_dim
#         for _ in range(depth):
#             layers.append(nn.Linear(d, hidden_dim))
#             layers.append(activation())
#             d = hidden_dim
#         layers.append(nn.Linear(d, out_dim))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, depth=4, out_dim=1, activation=nn.SiLU):
        super().__init__()
        self.act = activation()

        # main hidden transforms
        self.hidden = nn.ModuleList()
        # affine skips from input
        self.skip = nn.ModuleList()

        # biases for the nonlinear branch and skip branch
        self.b_nl = nn.ParameterList()
        self.b_skip = nn.ParameterList()

        d = in_dim
        for _ in range(depth):
            self.hidden.append(nn.Linear(d, hidden_dim, bias=False))   # we'll use b_nl manually
            self.skip.append(nn.Linear(in_dim, hidden_dim, bias=False)) # A_l x

            self.b_nl.append(nn.Parameter(torch.zeros(hidden_dim)))     # b_0
            self.b_skip.append(nn.Parameter(torch.zeros(hidden_dim)))   # b_1

            d = hidden_dim

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):

        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        h = x

        for W, A, b0, b1 in zip(self.hidden, self.skip, self.b_nl, self.b_skip):
            h = self.act(W(h) + b0) + A(x) + b1

        out = self.out(h)

        return out.squeeze(0) if out.shape[0] == 1 else out




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
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=lam_w)

def train(model, train_loader, optimizer, loss_fn, epochs=200, device="cpu", log_every=10, lam = 0.0):
    model.train()
    history = []

    with_grad = False
    if lam > 0.0:
        with_grad = True
        print("Grad is ON!")
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_count = 0

        for Xb, Yb, dYb in train_loader:
            
            Xb = Xb.to(device).requires_grad_(with_grad)
            Yb = Yb.to(device)
            dYb = dYb.to(device)
            
            pred = model(Xb)
            loss_val = loss_fn(pred, Yb)
            
            # Gradient of prediction wrt inputs: (B,2)
            if(with_grad):
                grad_pred = torch.autograd.grad(
                    outputs=pred.sum(),     # we take  d/dx_j (sum_i f(x_i))
                    inputs=Xb,
                    create_graph=True
                )[0]
            
            # Gradient loss
            loss_grad = 0.0
            if(with_grad):
                loss_grad = loss_fn(grad_pred, dYb) # nn.MSELoss just does element-wise MSE so be wary of denominator

            loss = loss_val + Xb.shape[1]*lam * loss_grad

            optimizer.zero_grad() #gradients accumulate between batches by def
            loss.backward()
            optimizer.step()

            bs = Xb.shape[0] #batch size
            total_loss += loss.item() * bs
            total_count += bs

        epoch_loss = total_loss / total_count
        history.append(epoch_loss)

        if epoch % log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs}  |  MSE: {epoch_loss:.6e}")

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
    model: torch nn expects (N,d) -> (N,1)
    test_x: np.ndarray, shape (N,d)
    test_f: np.ndarray, shape (N,1) (or (N,))
    Returns a dict of error metrics.
    """
    model.eval()

    # --- to torch ---
    X = torch.as_tensor(test_x, dtype=dtype, device=device)

    # ensure (N,1)
    F = torch.as_tensor(test_f, dtype=dtype, device=device)
    if F.ndim == 1:
        F = F.unsqueeze(1)
    else:
        F = F.reshape(-1, 1)

    with torch.no_grad():
        pred = model(X)
        # ensure (N,1) for consistent broadcasting
        pred = pred.reshape(-1, 1)

    # --- compute errors in torch ---
    diff = pred - F  # (N,1)
    
    
    # relative error per sample: |diff| / (|truth| + eps)
    rel = diff.abs() / (F.abs() + eps)  # (N,1)

    max_rel = rel.max().item()
    median_rel = rel.median().item()

    l2 = torch.linalg.norm(diff).item()
    l2_truth = torch.linalg.norm(F).item()
    rel_l2 = (l2 / (l2_truth + eps))
   
    return {
        "max_relative_error": max_rel,
        "median_relative_error": median_rel,
        "l2_error": l2,
        "relative_l2_error": rel_l2,
    }

error_dict = errors(model, test_x, test_f, device=device, dtype=torch.float32, eps=1e-12)
print(f"Model Results: {error_dict}")

