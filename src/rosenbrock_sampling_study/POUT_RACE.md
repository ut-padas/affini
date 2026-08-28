# Aggressive P_out race

This experiment answers a different question from the fixed-budget sampling
study: which proposal scheme reaches a target uncovered volume first when it
may append every surviving proposal?

Run it from an active one-GPU Vista idev session:

```bash
./run_rosenbrock_pout_race.sh
```

The default run compares:

- uniform proposals filtered to the uncovered set;
- normalized ellipsoidal shells;
- unnormalized Gaussian clouds with out-of-domain rejection;
- the legacy Gaussian halo with boundary clipping.

All methods start from the same 5,000 random points and use the same 20,000
test points. Each round gives every cluster the same number of raw proposals.
Every proposal classified as uncovered is appended. The resulting batch sizes
are deliberately unequal.

The default membership search uses `k=10` to match the old script. For the more
conservative setting used by the fixed-budget study, run:

```bash
./run_rosenbrock_pout_race.sh \
  --output results/rosenbrock_pout_race/dim5_k50 \
  --membership-k 50
```

To recreate only the old halo path:

```bash
./run_rosenbrock_pout_race.sh \
  --output results/rosenbrock_pout_race/legacy_only \
  --methods legacy_halo
```

The default target is 2%. To stop at 3%, close to the old 2.64% result:

```bash
./run_rosenbrock_pout_race.sh \
  --output results/rosenbrock_pout_race/dim5_target3 \
  --target-pout 0.03
```

The analysis ranks methods by measured compute time to target and also reports
rounds, unique added points, cumulative fit points, and final P_out. Rosenbrock
is cheap, so the fastest wall-clock method may be wasteful for an expensive
scientific model. Keep both rankings.

The driver saves each arm after every acquisition. Repeating the same command
resumes incomplete arms and skips completed arms. Use a new output directory
when changing any scientific setting.
