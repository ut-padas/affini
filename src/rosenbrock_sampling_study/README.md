# Rosenbrock iterative-sampling study

This package runs paired comparisons of four outer-loop acquisition methods:

- unconditional uniform sampling;
- global uniform sampling filtered to the uncovered set;
- a normalized ellipsoidal shell;
- the unnormalized Gaussian cloud used by the earlier halo script.

The driver uses one MPI rank. Every arm within a dimension and seed receives the same initial design, calibration design, evaluation design, acquisition count, model settings, and error tolerance. It checkpoints the accumulated training design after every acquisition.

From an active Vista `idev` session, run:

```bash
./run_rosenbrock_sampling.sh
```

The default `idev` preset uses all five dimensions, one paired seed, four samplers, and five acquisitions of 500 new Rosenbrock evaluations. It is meant to finish inside a development allocation and produces descriptive comparisons. It does not provide enough replicates for inferential claims.

Other presets are available:

```bash
./run_rosenbrock_sampling.sh --preset smoke
./run_rosenbrock_sampling.sh --preset pilot
./run_rosenbrock_sampling.sh --preset full
```

The pilot and full presets use the dimension-dependent initialization heuristic. Run those in a production `gh` allocation with enough wall time. Repeating a command resumes incomplete arms and skips completed arms.
