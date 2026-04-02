# Iterative $p_\mathrm{out}$ Improvement Experiment
## Setup
1. **Training**
   1. $X_\mathrm{train}$ is sampled uniformly from domain of apt dimensions.
   2. Surrogate is trained on $X_\mathrm{train}$.
2. **Prediction & Unguranteed Point Identification**
   1. $X_\mathrm{test}$ is sampled uniformly from domain of apt dimensions.
   2. Run `surrogate.predict(X_test)` to get predictions and unguaranteed point mask.
   3. Compute the error metrics and percentages of unguaranteed points, and log them.
   4. Append the unguaranteed points to $X_\mathrm{train}$.
3. **Repeat** steps 1 and 2 for a fixed number of iterations.

```python
def sample_chord(X, num_synth, rng):
    """Current Approach: Interpolate between 2 random points."""
    m, d = X.shape
    idx1 = rng.integers(0, m, num_synth)
    idx2 = (idx1 + rng.integers(1, m, num_synth)) % m
    alpha = rng.random((num_synth, 1))

    return X[idx1] * alpha + X[idx2] * (1.0 - alpha)


def sample_dirichlet(X, num_synth, rng):
    """Proposed Approach: Dirichlet weights over d+1 points."""
    m, d = X.shape
    # If the cluster has fewer points than d+1, we cap the sample size
    num_pts = min(d + 1, m)

    # Create permutation/weight matrix B of size (m, num_synth)
    B = np.zeros((m, num_synth))
    for i in range(num_synth):
        # Choose d+1 random points
        idxs = rng.choice(m, num_pts, replace=False)
        # Generate Dirichlet weights
        weights = rng.dirichlet(np.ones(num_pts))
        B[idxs, i] = weights

    # X is (m, d). B is (m, num_synth).
    # X_new = B^T @ X -> results in (num_synth, d)
    return B.T @ X
```


# Sampling Comparison Experiment
## Metrics
1. Variance Retained
$$ \text{Variance Retained} = \frac{\text{Tr}(\Sigma_{\text{synth}})}{\text{Tr}(\Sigma_{\text{orig}})} $$
2. Mean Radius Ratio
$$ \text{Radius Ratio} = \frac{\frac{1}{N_\mathrm{synth}} \sum_{i=1}^{N_\mathrm{synth}} \|x_i - \mu_{\text{synth}}\|}{\frac{1}{N_\mathrm{orig}} \sum_{i=1}^{N_\mathrm{orig}} \|x_i - \mu_{\text{orig}}\|} $$
3. Mean Pairwise Distance Ratio
$$ \text{Pairwise Ratio} = \frac{\text{Mean}(\|x_i - x_j\|)_{\text{synth}}}{\text{Mean}(\|x_i - x_j\|)_{\text{orig}}} $$
4. Relative Covariance Error
$$ \text{Cov Error} = \frac{\|\Sigma_{\text{synth}} - \Sigma_{\text{orig}}\|_F}{\|\Sigma_{\text{orig}}\|_F} $$

## Sampling Methods
1. **Random Chord**: Existing random convex combination method.
$$ x_{\text{synth}} = \alpha x_i + (1 - \alpha) x_j, \quad \alpha \sim \text{Uniform}(0, 1) $$
1. **Dirichlet Sampling**: New method using Dirichlet weights over $d+1$ random points.
$$ X_{\text{synth}} = B^T X, \quad B_{ij} = \begin{cases} w_k & \text{if } j \in \text{idxs}_i \\ 0 & \text{otherwise} \end{cases}, \quad w \sim \text{Dirichlet}(\mathbf{1}) $$


## Original Data

- Original data: Sampled from a multivariate normal distribution with a diagonal covariance matrix that has decreasing variance across dimensions.
$$ \Sigma_{\text{orig}} = \text{diag}(\text{np.linspace(10, 1, d)}) $$

## Observations
- Random Chord preserves original variance across dimensions but not Dirichlet Sampling which tends to 0.
- Even the spread of the data (Mean Radius & Pairwise Distance) is not preserved by Dirichlet Sampling, but kept well by Random Chord.
- The covariance error grows to 100% for Dirichlet Sampling, but is much smaller for Random Chord.