```sh
c608-142[gh](636)$ time uv run python src/gpu_adaptive_clustering_optimized.py 
ABS_TOL=0.1, REL_TOL=0.1
Running on GPU. N_SAMPLES=10,000
  --> 0/100 clusters need splitting
Total final clusters formed: 100
Adaptive clustering completed in 0.36 seconds.
Total clusters violating tolerance on test points: (0 / 100) = 0.00%
Average cluster size: 100.00 points per cluster

real    0m3.736s
user    0m1.681s
sys     0m0.674s
c608-142[gh](637)$ time uv run python src/gpu_adaptive_clustering_optimized.py 
ABS_TOL=0.01, REL_TOL=0.01
Running on GPU. N_SAMPLES=10,000
  --> 98/100 clusters need splitting
Level 1: processing 196 clusters (prev level took 0.46s)
  --> 177/196 clusters need splitting
Level 2: processing 354 clusters (prev level took 0.58s)
  --> 179/354 clusters need splitting
Level 3: processing 358 clusters (prev level took 0.74s)
  --> 51/358 clusters need splitting
Level 4: processing 102 clusters (prev level took 0.44s)
  --> 7/102 clusters need splitting
Level 5: processing 14 clusters (prev level took 0.12s)
  --> 1/14 clusters need splitting
Level 6: processing 2 clusters (prev level took 0.03s)
  --> 0/2 clusters need splitting
Total final clusters formed: 613
Adaptive clustering completed in 2.60 seconds.
Total clusters violating tolerance on test points: (190 / 613) = 31.00%
Average cluster size: 26.67 points per cluster

real    0m7.579s
user    0m4.916s
sys     0m1.612s
c608-142[gh](638)$ time uv run python src/gpu_adaptive_clustering_optimized.py 
ABS_TOL=0.001, REL_TOL=0.001
Running on GPU. N_SAMPLES=10,000
  --> 100/100 clusters need splitting
Level 1: processing 200 clusters (prev level took 0.45s)
  --> 200/200 clusters need splitting
Level 2: processing 400 clusters (prev level took 0.64s)
  --> 400/400 clusters need splitting
Level 3: processing 800 clusters (prev level took 1.37s)
  --> 799/800 clusters need splitting
Level 4: processing 1598 clusters (prev level took 2.72s)
  --> 1504/1598 clusters need splitting
Level 5: processing 3008 clusters (prev level took 5.26s)
  --> 2061/3008 clusters need splitting
Level 6: processing 4122 clusters (prev level took 8.37s)
  --> 1568/4122 clusters need splitting
Level 7: processing 3136 clusters (prev level took 8.36s)
  --> 694/3136 clusters need splitting
Level 8: processing 1388 clusters (prev level took 4.94s)
  --> 274/1388 clusters need splitting
Level 9: processing 548 clusters (prev level took 2.04s)
  --> 81/548 clusters need splitting
Level 10: processing 162 clusters (prev level took 0.74s)
  --> 20/162 clusters need splitting
Level 11: processing 40 clusters (prev level took 0.21s)
  --> 5/40 clusters need splitting
Level 12: processing 10 clusters (prev level took 0.06s)
  --> 1/10 clusters need splitting
Level 13: processing 2 clusters (prev level took 0.02s)
  --> 1/2 clusters need splitting
Level 14: processing 2 clusters (prev level took 0.01s)
  --> 0/2 clusters need splitting
Total final clusters formed: 7808
Adaptive clustering completed in 35.43 seconds.
Total clusters violating tolerance on test points: (1913 / 7808) = 24.50%
Average cluster size: 25.50 points per cluster

real    1m7.706s
user    0m54.870s
sys     0m12.002s
c608-142[gh](639)$ time uv run python src/gpu_adaptive_clustering_optimized.py 
ABS_TOL=0.0001, REL_TOL=0.0001
Running on GPU. N_SAMPLES=10,000
  --> 100/100 clusters need splitting
Level 1: processing 200 clusters (prev level took 0.44s)
  --> 200/200 clusters need splitting
Level 2: processing 400 clusters (prev level took 0.64s)
  --> 400/400 clusters need splitting
Level 3: processing 800 clusters (prev level took 1.35s)
  --> 800/800 clusters need splitting
Level 4: processing 1600 clusters (prev level took 2.69s)
  --> 1600/1600 clusters need splitting
Level 5: processing 3200 clusters (prev level took 5.47s)
  --> 3199/3200 clusters need splitting
Level 6: processing 6398 clusters (prev level took 11.28s)
  --> 6223/6398 clusters need splitting
Level 7: processing 12446 clusters (prev level took 22.05s)
  --> 10566/12446 clusters need splitting
Level 8: processing 21132 clusters (prev level took 39.31s)
  --> 12988/21132 clusters need splitting
Level 9: processing 25976 clusters (prev level took 53.66s)
  --> 10877/25976 clusters need splitting
Level 10: processing 21754 clusters (prev level took 57.05s)
  --> 6355/21754 clusters need splitting
Level 11: processing 12710 clusters (prev level took 40.64s)
  --> 2922/12710 clusters need splitting
Level 12: processing 5844 clusters (prev level took 21.32s)
  --> 1160/5844 clusters need splitting
Level 13: processing 2320 clusters (prev level took 9.37s)
  --> 390/2320 clusters need splitting
Level 14: processing 780 clusters (prev level took 3.46s)
  --> 122/780 clusters need splitting
Level 15: processing 244 clusters (prev level took 1.12s)
  --> 31/244 clusters need splitting
Level 16: processing 62 clusters (prev level took 0.33s)
  --> 13/62 clusters need splitting
Level 17: processing 26 clusters (prev level took 0.14s)
  --> 1/26 clusters need splitting
Level 18: processing 2 clusters (prev level took 0.03s)
  --> 0/2 clusters need splitting
Total final clusters formed: 58047
Adaptive clustering completed in 270.56 seconds.
Total clusters violating tolerance on test points: (13084 / 58047) = 22.54%
Average cluster size: 25.49 points per cluster

real    7m51.938s
user    6m33.818s
sys     1m15.819s
```