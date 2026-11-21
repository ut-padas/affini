"""Independent recomputation of section F's Bernstein parameters, and the true eps-ranks of the corner peak at d = 2 (SVD), against the memo's bound and a per-axis bound."""
import numpy as np


def rho(z, lo, hi):
    """Bernstein parameter of the ellipse with foci lo, hi through z: sum of the distances to the foci over the focal half-distance, rho = (sum + sqrt(sum^2 - 4 h^2)) / (2 h)."""
    h = (hi - lo) / 2
    s = abs(z - lo) + abs(z - hi)  # = h (rho + 1/rho)
    return (s + np.sqrt(s * s - 4 * h * h)) / (2 * h)


def params(b, d):
    rng = np.random.default_rng(7000 + d)
    a = rng.uniform(0, 1, d)
    return a * b / a.sum(), rng.uniform(0, 1, d)


a, w = params(7.25, 2)
r = [rho(w[i] + 1j / a[i], 0, 1) for i in range(2)]
print("product peak d=2: a", a, "w", w, "rho", r, "deg/decade", [1 / np.log10(x) for x in r])
rs = rho(-1 + 0j, 0, 1.85)
print("corner peak: rho_s", rs, "1/log10", 1 / np.log10(rs))

# corner peak d = 2: eps-rank of the matrix f(x1, x2) on a fine Chebyshev grid, relative to max |f| = 1
a, _ = params(1.85, 2)
n = 200
t = 0.5 + 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)
F = (1 + a[0] * t[:, None] + a[1] * t[None, :]) ** -3.0
U, S, Vt = np.linalg.svd(F)
print("corner peak d=2: a =", a)
# per-axis bound: as a function of x_i alone (the others anywhere in the box) the nearest pole is at x_i = -1/a_i
ri = [rho(-1 / a[i] + 0j, 0, 1) for i in range(2)]
print("  per-axis rho_i", ri, " rank/decade bound from the better axis", 1 / np.log10(max(ri)), " memo's bound", 1 / np.log10(rs))
for tol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
    # smallest rank whose truncated SVD is within tol in the sup norm on the grid
    for k in range(1, 30):
        if np.abs((U[:, :k] * S[:k]) @ Vt[:k] - F).max() <= tol:
            break
    print(f"  tol {tol:.0e}: sup-norm eps-rank {k}   (memo's ChebTT max rank at 1e-2..1e-8: 2, 4, 6, 6)")
ks = []
tols = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
for tol in tols:
    for k in range(1, 30):
        if np.abs((U[:, :k] * S[:k]) @ Vt[:k] - F).max() <= tol:
            break
    ks.append(k)
print("  eps-rank per decade (fit):", np.polyfit(np.log10(1 / np.array(tols)), ks, 1)[0])
