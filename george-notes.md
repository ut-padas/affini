
# HATANN: High dimensional approximate Taylor approximation using approximate nearest neighbors


## 2/10 TODO
- Create absollute error histrogram curves that shows for each test point the performance over baseline (say no Hessian); straigh line is baseline and then points above and below if error is larger or smaller.
- Support for multiGPU runs and easy MPI with DBSCAN; on a single node with multiple GPUs we can just do it in an embarrasingly parallel way.
- Support for computing metric by error acceptance (without need to compute gradients at each point
- Rewrite parallel section, it is unclear what BFS etc exactly means
- Rewrite results section and discuss parallel complexity of the algorithm (under assumptions)
- Nvsight measurements to check performance of Raw kernels
- Retrain all networks with double the size of weights and repeat experiment


### Unresolved issues
- Probabilistic error estimates
- Estimating local intrinsic dimensionality
- Can we reduce function evaluations and gradient? for example we could fit quadratic and  gradient using fFisher estimates and randomized approximation of gradients and Hessians. 
- Shrinkage should not be fixed but relative to the scaling of the convariance matrix (proportional to the singular viector)
- Fix the write-up on cluster splitting: Sequential has sign
- Proof that filtering points by error acceptance and computing covariance is approximating Hessian (at least for scalar functions).

### Future 

### HPC
Distributed memory; load migration, task based (Dask, Parla) to deal with load imbalance

### Analysis
Assume high-dimensional fourier nad gaussians; assume uniform distribution; what is the convergence of the scheme in terms of number of points.

### ML
- ML Convergence analysis for ML function, increase weights and check convergence. Fix the vector regression
