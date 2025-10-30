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
Time taken: 35.9510 seconds

real    0m38.511s
user    0m36.553s
sys     0m1.011s
```