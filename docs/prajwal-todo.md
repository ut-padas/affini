- [Iterative P\_out Improvement Experiment](#iterative-p_out-improvement-experiment)
  - [Setup](#setup)
- [Sampling Comparison Experiment](#sampling-comparison-experiment)
  - [Metrics](#metrics)
  - [Sampling Methods](#sampling-methods)
  - [Original Data](#original-data)
  - [Observations](#observations)
- [Affine Surrogates](#affine-surrogates)
  - [Training](#training)
  - [Inference](#inference)
  - [Estimation of $H^{-1}$ from the Covariance of the Cluster - Taylor Surrogaters](#estimation-of-h-1-from-the-covariance-of-the-cluster---taylor-surrogaters)
  - [Issues with Affine Surrogates](#issues-with-affine-surrogates)
  - [Fix - Anchoring the Affine Surrogate to the Cluster Center](#fix---anchoring-the-affine-surrogate-to-the-cluster-center)
- [Analysis: Large errors for Anisotropic Gaussian](#analysis-large-errors-for-anisotropic-gaussian)
  - [Table - $5D$ Anisotropic Gaussian](#table---5d-anisotropic-gaussian)






# Iterative P_out Improvement Experiment
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



# Affine Surrogates
## Training
Given function $f: \mathbb{R}^d \to \mathbb{R}$, and a cluster of points $X \in \mathbb{R}^{N \times d}$, we define centered points $X_c = X - \mu$ and centered function values $F_c = f(X) - f_\mu$ where $\mu$ is the cluster center and $f_\mu$ is the average of function values in the cluster.

A linear model is fit, given by:
$$ \hat{f}(x) = A x + b $$
where $A \in \mathbb{R}^{1 \times d}$ and $b \in \mathbb{R}$ are found by solving the normal equations (with Tikhonov regularization), using batched `np.linalg.solve`:
$$ \left( X_c^T X_c + \lambda I \right) A^T = X_c^T F_c $$
$$ b = f_\mu - A \mu $$

$\lambda$ is defined as $\lambda = 10^{-8} \cdot \text{max\_eigenvalue}(X_c^T X_c)$ to ensure numerical stability. Here, since $X_c^T X_c$ is a positive semi-definite matrix, its eigenvalues are non-negative, which implies $\lambda_\text{max} = O(\sum_i \lambda_i) = O(\text{Tr}(X_c^T X_c))$. Hence, instead of computing the eigenvalues of the symmetric matrix $X_c^T X_c$, we can compute the trace which is much cheaper and still provides a good scaling for $\lambda$.

## Inference
For a query point $x_q$, we find the cluster center closest to $x_q$ in $\| \cdot \|_{H}$ Hessian norm. Then using the linear model of that cluster, we compute the predicted function:
$$ \hat{f}(x_q; A_{k}, b_{k}) = A_{k} x_q + b_{k} $$


## Estimation of $H^{-1}$ from the Covariance of the Cluster - Taylor Surrogaters
When we were using the Taylor surrogates, we had access to oteh grads, and we estimate the Hessian inverse $H^{-1}$ using the covariance of the cluster points, whose errors are small w.r.t the Taylor expansion:
$$ e = f(x) - \hat{f}(x; \mu) = f(x) - f(\mu) - \nabla f(\mu)^T (x - \mu) $$
where $\hat{f}(x; \mu)$ is the first-order Taylor expansion of $f$ around $\mu$, the cluster center. 
We do so by finding the $n$ points with the smallest errors, and computing their covariance:
$$ H^{-1} \approx \text{Cov}(x_i \text{ with smallest } e_i). $$

- MAKE THIS SECTION TECHNICAL, eplxianing why it is necessary to pick the smallest points, and how this covariance captures the local geometry of the function around the cluster center, which is crucial for estimating HEss later to split the cluster along dominant eigenvector of the Hessian.


## Issues with Affine Surrogates
With affine surrogates, we do not have access to the gradients. Hence when we tried to estimate $H^{-1}$ using the same approach as with Taylor surrogates, with error defined as:
$$ e = f(x) - \hat{f}(x; A, b) = f(x) - A x - b $$
we found that the covariance of the points with the smallest errors did not capture the local geometry of the function around the cluster center, and hence we were getting more clusters.
- Make this section technical again

## Fix - Anchoring the Affine Surrogate to the Cluster Center
To fix the issue with estimating $H^{-1}$ from the covariance of the points with the smallest errors, we propose to anchor the affine surrogate to the cluster center:
$$ eqn $$
- Explain the tehncial details and what it is and how it fits the region, and essentially got us what we wanted.


Fill up the above sections with the technical details and equations, and make it more rigorous, while being mathematically precise and grounded. Be concise, use the same nottaions I have already used/defined. To make things easy to describe block equations, you can use $X \in \mathbb{R}^{N \times d}$ to denote the cluster points, and $f(x) : \mathbb{R}^d \to \mathbb{R}$ to denote the function we are trying to approximate, and $F = f(X) \in \mathbb{R}^N$ to denote the function values at the cluster points, $\mu$ to denote the cluster center.


---
