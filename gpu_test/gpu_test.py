import gc
import logging
import time

import cupy as xp
import numpy as np
import pandas as pd

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Problem Definition (Rosenbrock)
# -----------------------------------------------------------------------------
ROSENBROCK_SCALE = 100.0


def rosenbrock(X: xp.ndarray, a: float = ROSENBROCK_SCALE) -> xp.ndarray:
    x_i = X[:, :-1]
    x_next = X[:, 1:]
    F = xp.sum(
        a * (x_next - x_i**2.0) ** 2.0 + (1.0 - x_i) ** 2.0, axis=1, keepdims=True
    )
    return F


def rosenbrock_grad(X: xp.ndarray, a: float = ROSENBROCK_SCALE) -> xp.ndarray:
    N, d = X.shape
    J = xp.zeros((N, 1, d), dtype=X.dtype)
    J[:, 0, :-1] += 2 * (X[:, :-1] - 1) - 4 * a * X[:, :-1] * (
        X[:, 1:] - X[:, :-1] ** 2
    )
    J[:, 0, 1:] += 2 * a * (X[:, 1:] - X[:, :-1] ** 2)
    return J


# -----------------------------------------------------------------------------
# 1. Baseline Implementation (Broadcasting & Temporary Arrays)
# -----------------------------------------------------------------------------
def baseline_taylor_error(X, mus, labels, F_vals, f_mus, g_mus):
    """
    Computes Taylor error using standard CuPy broadcasting.
    Memory Bottleneck: Allocates 'dx' (N, D).
    """
    # 1. Expand centroids to (N, d) - Implicitly or Explicitly
    # This line below is the killer: X_pool[idx] - mus[bad_idx][map]
    # We simulate it exactly:
    mus_expanded = mus[labels]

    # Allocating dx (Huge)
    dx = X - mus_expanded

    # Compute Linear approx: f_mu + dx @ g_mu
    # We have to reshape g_mus to match
    g_mus_expanded = g_mus[labels]

    # Dot product (sum over D)
    linear_term = xp.sum(dx * g_mus_expanded, axis=1)

    f_approx = f_mus[labels] + linear_term

    # Error
    E = xp.abs(F_vals.ravel() - f_approx)
    return E


def baseline_covariance(X, mus, labels, K, D):
    """
    Computes Covariance using standard broadcasting.
    Memory Bottleneck: Allocates (N, D) for dx.
    And potentially (N, D, D) for the outer product term before reduction.
    """
    # Recalculate dx (typically code does this again if not cached)
    dx = X - mus[labels]

    # Outer product for every point: (N, D, 1) * (N, 1, D) -> (N, D, D)
    # This is an even BIGGER bottleneck than dx.
    # To be fair to the baseline, we'll assume we use a slightly smarter loop
    # if N is huge, but standard vectorization tries to do it all at once.

    # We will try a batched approach for the baseline to give it a fighting chance,
    # otherwise it crashes instantly on small N.
    Cov = xp.zeros((K, D, D), dtype=X.dtype)

    # Using ufunc.at or scatter_add on the flattened outer product
    # Constructing term:
    # term = dx[:, :, None] * dx[:, None, :] # (N, D, D) <--- INSTANT OOM for large N

    # We will compute term and add in one go? No, standard numpy semantics don't allow
    # reduction without materialization easily unless using einsum.
    # xp.einsum('ni,nj->nij', dx, dx) materializes (N,D,D).

    # Optimistic Baseline: We iterate chunks to survive, but it's slow.
    CHUNK = 10000
    N = X.shape[0]
    for i in range(0, N, CHUNK):
        end = min(N, i + CHUNK)
        dx_chunk = dx[i:end]
        # Outer product
        term = dx_chunk[:, :, None] * dx_chunk[:, None, :]
        # Add
        chunk_labels = labels[i:end]
        # This part is tricky in pure cupy without loop over K.
        # We'll use scatter_add
        import cupyx

        cupyx.scatter_add(Cov, chunk_labels, term)

    return Cov


# -----------------------------------------------------------------------------
# 2. Fused Kernel Implementation (In-Place / Register Math)
# -----------------------------------------------------------------------------

# Kernel 1: Fused Taylor Error
# Calculates dx = X - mus[labels] on the fly, computes dot product, finds error.
# Never stores dx.
fused_taylor_kernel = xp.ElementwiseKernel(
    "raw T X, raw T mus, int64 labels, raw T g_mus, raw T f_mus, T F_val, int32 D",
    "T error",
    """
    T f_lin = f_mus[labels];
    
    // Dot product loop
    int mu_idx = labels * D;
    int x_idx = i * D;
    
    T dot = 0;
    for (int j = 0; j < D; j++) {
        T diff = X[x_idx + j] - mus[mu_idx + j];
        dot += diff * g_mus[mu_idx + j];
    }
    
    f_lin += dot;
    error = abs(F_val - f_lin);
    """,
    "fused_taylor_error",
)


def fused_taylor_error(X, mus, labels, F_vals, f_mus, g_mus):
    N, D = X.shape
    return fused_taylor_kernel(X, mus, labels, g_mus, f_mus, F_vals.ravel(), D)


# Kernel 2: Fused Covariance
# Calculates dx = X - mus[labels], computes outer prod, atomic adds to Cov.
# Never stores dx or (N, D, D) term.
cov_kernel_code = r"""
extern "C" __global__
void fused_covariance(const double* X, const double* mus, const long long* labels, 
                      double* Cov, int N, int D, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    long long k = labels[idx];
    
    // We need to iterate over the DxD upper triangle
    // To avoid allocating local arrays for dx, we compute dx[i] and dx[j] on demand
    // or cache small D in registers. For D=5, registers are fine. 
    // For D=50, we might need shared mem. 
    // For this simple kernel, we just re-read global mem which is cached L2.
    
    int mu_offset = k * D;
    int x_offset = idx * D;
    
    for (int r = 0; r < D; r++) {
        double dx_r = X[x_offset + r] - mus[mu_offset + r];
        
        for (int c = r; c < D; c++) { // Upper triangle
            double dx_c = X[x_offset + c] - mus[mu_offset + c];
            double val = dx_r * dx_c;
            
            // Atomic Add to Cov[k, r, c] and Cov[k, c, r] if distinct
            // Cov shape: (K, D, D) -> Flat index: k*D*D + r*D + c
            
            unsigned long long* address = (unsigned long long*)&Cov[k * D * D + r * D + c];
            atomicAdd(address, (unsigned long long)__double_as_longlong(val));
            
            if (r != c) {
                address = (unsigned long long*)&Cov[k * D * D + c * D + r];
                atomicAdd(address, (unsigned long long)__double_as_longlong(val));
            }
        }
    }
}
"""
cov_module = xp.RawModule(code=cov_kernel_code)
cov_kernel = cov_module.get_function("fused_covariance")


def fused_covariance(X, mus, labels, K, D):
    Cov = xp.zeros((K, D, D), dtype=xp.float64)  # Use float64 for atomics
    block_size = 256
    grid_size = (X.shape[0] + block_size - 1) // block_size
    cov_kernel((grid_size,), (block_size,), (X, mus, labels, Cov, X.shape[0], D, K))
    return Cov


# -----------------------------------------------------------------------------
# Experiment Runner
# -----------------------------------------------------------------------------
def get_mem_gb():
    mempool = xp.get_default_memory_pool()
    return mempool.used_bytes() / 1e9


def run_experiment():
    dims = [5, 20, 30]
    ns = [1e7, 5e7, 1e8, 2e8, 3e8, 4e8, 8e8]
    results = []

    mempool = xp.get_default_memory_pool()
    K = 100

    for d in dims:
        for n_float in ns:
            n = int(n_float)
            if n * d * 8 > 60e9:
                continue  # Skip if data > 60GB (Physical limit)

            print(f"--- Running: D={d}, N={n:,} ---")

            try:
                # 1. Alloc Data
                # Free everything to start fresh
                mempool.free_all_blocks()
                gc.collect()

                X = xp.random.random((n, d), dtype=xp.float64)
                mus = xp.random.random((K, d), dtype=xp.float64)
                labels = xp.random.randint(
                    0, K, size=(n,), dtype=xp.int64  # type: ignore
                )

                # Precompute F/G (Simulating existing state)
                F_vals = rosenbrock(X)
                f_mus = rosenbrock(mus).ravel()
                g_mus = rosenbrock_grad(mus).reshape(K, d)

                xp.cuda.Device().synchronize()

                # --- Test Baseline ---
                mempool.free_all_blocks()
                baseline_start_size = mempool.total_bytes()

                try:
                    t0 = time.perf_counter()
                    # 1. Taylor
                    E_base = baseline_taylor_error(X, mus, labels, F_vals, f_mus, g_mus)
                    xp.cuda.Device().synchronize()
                    t1 = time.perf_counter()

                    # 2. Covariance
                    C_base = baseline_covariance(X, mus, labels, K, d)
                    xp.cuda.Device().synchronize()
                    t2 = time.perf_counter()

                    # CAPTURE PEAK: The pool grew to handle the temps.
                    # total_bytes() is the size of the arena, effectively our Peak.
                    peak_baseline = mempool.total_bytes()
                    mem_overhead_base = (peak_baseline - baseline_start_size) / 1e9

                    time_base_total = t2 - t0
                    status_base = "Success"

                    del E_base, C_base

                except xp.cuda.memory.OutOfMemoryError:
                    status_base = "OOM"
                    time_base_total = np.nan
                    mem_overhead_base = np.nan
                    print("  Baseline: OOM")
                except Exception as e:
                    status_base = f"Err: {str(e)}"
                    time_base_total = np.nan
                    mem_overhead_base = np.nan

                # --- Test Fused ---
                mempool.free_all_blocks()
                fused_start_size = mempool.total_bytes()

                try:
                    t0 = time.perf_counter()
                    # 1. Taylor
                    E_fused = fused_taylor_error(X, mus, labels, F_vals, f_mus, g_mus)
                    xp.cuda.Device().synchronize()
                    t1 = time.perf_counter()

                    # 2. Covariance
                    C_fused = fused_covariance(X, mus, labels, K, d)
                    xp.cuda.Device().synchronize()
                    t2 = time.perf_counter()

                    # CAPTURE PEAK
                    peak_fused = mempool.total_bytes()
                    mem_overhead_fused = (peak_fused - fused_start_size) / 1e9

                    time_fused_total = t2 - t0
                    status_fused = "Success"

                    del E_fused, C_fused

                except xp.cuda.memory.OutOfMemoryError:
                    status_fused = "OOM"
                    time_fused_total = np.nan
                    mem_overhead_fused = np.nan
                    print("  Fused: OOM")
                except Exception as e:
                    status_fused = f"Err: {str(e)}"
                    time_fused_total = np.nan
                    mem_overhead_fused = np.nan

                results.append(
                    {
                        "dim": d,
                        "N": n,
                        "base_status": status_base,
                        "base_time": time_base_total,
                        "base_mem_gb": mem_overhead_base,
                        "fused_status": status_fused,
                        "fused_time": time_fused_total,
                        "fused_mem_gb": mem_overhead_fused,
                        "speedup": (
                            (time_base_total / time_fused_total)
                            if (status_base == "Success" and status_fused == "Success")
                            else np.nan
                        ),
                        "mem_reduction_gb": (
                            (mem_overhead_base - mem_overhead_fused)
                            if (status_base == "Success" and status_fused == "Success")
                            else np.nan
                        ),
                    }
                )

            except Exception as e:
                print(f"Setup Failed: {e}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_experiment()
    print("\n\n=== BENCHMARK RESULTS ===")
    print(df.to_string())
    df.to_csv("fused_vs_base_benchmark.csv", index=False)
