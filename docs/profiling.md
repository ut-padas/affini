# Version 2 (3.2M points)
```sh
c608-091[gh](404)$ time uv run python adaptive_clustering_accelerated.py 
CUML available: True
Generated 3200000 sample data points.
Total clusters formed: 17512
Clusters by CUML: 474
Clusters by SKLearn KMeans: 7384
Clusters by SKLearn MiniBatchKMeans: 0
Clusters by TwoMeans: 9687
Total clustering calls: 17545
Time taken: 35.9510 seconds # V4: 16.8780 seconds

real    0m38.511s
user    0m36.553s
sys     0m1.011s
```

# Version 3 (10M points)
```sh
c608-102[gh](429)$ time uv run python adaptive_clustering_accelerated.py 
CUML available: True
Generated 10000000 sample data points.
Using fast two-means length threshold: 8192 for dimension: 3
Total clusters formed: 30382
Clusters by CUML: 1471
Clusters by SKLearn KMeans: 308
Clusters by SKLearn MiniBatchKMeans: 0
Clusters by TwoMeans: 28730
Total clustering calls: 30509
Time taken: 59.2800 seconds

real    1m3.325s
user    0m59.868s
sys     0m1.480s
```

# Version 4 (10M points)
```sh
c608-092[gh](456)$ time uv run python adaptive_clustering_accelerated.py 
CUML available: True
Generated 10000000 sample data points.
Using fast two-means length threshold: 8192 for dimension: 3
Total clusters formed: 30382
Clusters by CUML: 1471
Clusters by SKLearn KMeans: 308
Clusters by SKLearn MiniBatchKMeans: 0
Clusters by TwoMeans: 28730
Total clustering calls: 30509
Time taken: 37.0057 seconds

real    0m40.464s
user    0m37.611s
sys     0m1.728s
```