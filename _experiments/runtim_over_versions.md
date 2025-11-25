# 2D (SHCF) Test: n = 4000
## Version 1 (Time: 24s)
```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                4000 & 0.1 & 2.0\% & 1016 & 27.19\% & 70.98\% & 1.83\% & 3.5151 & 0.4981 & 0.8215 & 0.0654 & 0.0249 & 0.1642 \\
                4000 & 0.05 & 1.0\% & 1664 & 33.34\% & 63.44\% & 3.23\% & 3.961 & 0.3379 & 0.6955 & 0.0479 & 0.0139 & 0.1256 \\
                4000 & 0.025 & 0.5\% & 2719 & 42.59\% & 53.91\% & 3.5\% & 1.5815 & 0.1826 & 0.5093 & 0.034 & 0.0073 & 0.0932 \\
                4000 & 0.01 & 0.2\% & 4855 & 53.41\% & 42.25\% & 4.34\% & 1.9812 & 0.088 & 0.3645 & 0.0276 & 0.0033 & 0.0653 \\
                4000 & 0.005 & 0.1\% & 7613 & 60.74\% & 34.58\% & 4.68\% & 0.8856 & 0.0469 & 0.2798 & 0.0276 & 0.0018 & 0.05 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 2 (Accelerated) (Time: 14.8s)
```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                4000 & 0.1 & 2.0\% & 971 & 27.46\% & 70.97\% & 1.57\% & 4.2424 & 0.4853 & 0.8023 & 0.0511 & 0.0247 & 0.1627 \\
                4000 & 0.05 & 1.0\% & 1551 & 34.72\% & 62.91\% & 2.37\% & 3.1473 & 0.2814 & 0.6087 & 0.0407 & 0.013 & 0.1209 \\
                4000 & 0.025 & 0.5\% & 2441 & 43.37\% & 53.19\% & 3.44\% & 1.9408 & 0.159 & 0.4654 & 0.029 & 0.0074 & 0.0916 \\
                4000 & 0.01 & 0.2\% & 4491 & 54.89\% & 41.08\% & 4.03\% & 1.3877 & 0.0757 & 0.3458 & 0.0225 & 0.0032 & 0.064 \\
                4000 & 0.005 & 0.1\% & 7031 & 62.93\% & 32.98\% & 4.09\% & 1.2247 & 0.0391 & 0.2567 & 0.0157 & 0.0017 & 0.0475 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```



# 2D (SHCF) Test: n = 400,000
## Version 1 (Time: 1m 29s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 6.73 seconds
Time elapsed for predictions: 0.04989327595103532 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 8.57 seconds
Time elapsed for predictions: 0.08672645402839407 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 11.81 seconds
Time elapsed for predictions: 0.1518221179721877 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 20.48 seconds
/home1/10973/prajwal_prathiksh/github-repos/forked-knnsur/nn_inference.py:203: RuntimeWarning: divide by zero encountered in divide
  2 * atol / (alpha * self.hessians[I[:, 0]]),
/home1/10973/prajwal_prathiksh/github-repos/forked-knnsur/nn_inference.py:207: RuntimeWarning: divide by zero encountered in divide
  2
Time elapsed for predictions: 0.3159333019866608 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 32.57 seconds
Time elapsed for predictions: 0.5439730540383607 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                400000 & 0.1 & 2.0\% & 2812 & 1.64\% & 98.14\% & 0.22\% & 2.8277 & 0.5371 & 0.8877 & 0.0448 & 0.0237 & 0.1576 \\
                400000 & 0.05 & 1.0\% & 5126 & 1.79\% & 97.82\% & 0.39\% & 1.5883 & 0.4064 & 0.6971 & 0.0278 & 0.0118 & 0.1148 \\
                400000 & 0.025 & 0.5\% & 9209 & 2.28\% & 97.23\% & 0.49\% & 0.9754 & 0.1678 & 0.482 & 0.0195 & 0.0059 & 0.0806 \\
                400000 & 0.01 & 0.2\% & 19572 & 4.14\% & 94.99\% & 0.87\% & 0.5997 & 0.0702 & 0.3181 & 0.0117 & 0.0024 & 0.0522 \\
                400000 & 0.005 & 0.1\% & 34124 & 7.0\% & 91.63\% & 1.37\% & 0.5157 & 0.0328 & 0.2248 & 0.0043 & 0.0012 & 0.0372 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```


## Version 2 (Accelerated) (Time: 52s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 4.13 seconds
Time elapsed for predictions: 0.05067253496963531 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 5.00 seconds
Time elapsed for predictions: 0.0853774729766883 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 6.82 seconds
Time elapsed for predictions: 0.14745725400280207 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 10.79 seconds
Time elapsed for predictions: 0.3012487340020016 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 16.36 seconds
Time elapsed for predictions: 0.516799102013465 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                400000 & 0.1 & 2.0\% & 2831 & 1.58\% & 98.23\% & 0.19\% & 3.4508 & 0.3445 & 0.8129 & 0.047 & 0.023 & 0.1564 \\
                400000 & 0.05 & 1.0\% & 5045 & 1.71\% & 98.05\% & 0.24\% & 1.3716 & 0.3266 & 0.6259 & 0.0216 & 0.0117 & 0.1107 \\
                400000 & 0.025 & 0.5\% & 8980 & 1.96\% & 97.61\% & 0.43\% & 0.7413 & 0.1591 & 0.4619 & 0.0115 & 0.006 & 0.0798 \\
                400000 & 0.01 & 0.2\% & 18685 & 3.6\% & 95.57\% & 0.83\% & 0.568 & 0.063 & 0.2961 & 0.0073 & 0.0024 & 0.0512 \\
                400000 & 0.005 & 0.1\% & 32316 & 6.52\% & 92.31\% & 1.18\% & 0.2779 & 0.0333 & 0.2121 & 0.0045 & 0.0012 & 0.0366 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 3 (Time: 34.7s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 2.71 seconds
Time elapsed for predictions: 0.04481420700903982 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 2.74 seconds
Time elapsed for predictions: 0.07426815299550071 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 3.64 seconds
Time elapsed for predictions: 0.13056770304683596 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 6.17 seconds
Time elapsed for predictions: 0.26651790499454364 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 10.25 seconds
Time elapsed for predictions: 0.46388948400272056 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                400000 & 0.1 & 2.0\% & 2494 & 1.2\% & 98.77\% & 0.02\% & 1.197 & 0.5109 & 0.7374 & 0.0247 & 0.0211 & 0.1461 \\
                400000 & 0.05 & 1.0\% & 4376 & 1.52\% & 98.31\% & 0.17\% & 1.1672 & 0.2563 & 0.6258 & 0.0202 & 0.0114 & 0.1099 \\
                400000 & 0.025 & 0.5\% & 7976 & 1.79\% & 97.98\% & 0.23\% & 0.7179 & 0.1526 & 0.4505 & 0.0108 & 0.0056 & 0.0768 \\
                400000 & 0.01 & 0.2\% & 16646 & 3.66\% & 95.75\% & 0.59\% & 0.7179 & 0.0528 & 0.2857 & 0.0061 & 0.0024 & 0.0502 \\
                400000 & 0.005 & 0.1\% & 29228 & 6.35\% & 92.74\% & 0.91\% & 0.3161 & 0.0291 & 0.2142 & 0.0055 & 0.0012 & 0.0364 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 4 (Time: 22.6s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 1.21 seconds
Time elapsed for predictions: 0.0443276499863714 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 1.54 seconds
Time elapsed for predictions: 0.07355414796620607 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 2.35 seconds
Time elapsed for predictions: 0.12917017401196063 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 4.59 seconds
Time elapsed for predictions: 0.26404133497271687 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 7.92 seconds
Time elapsed for predictions: 0.45727067603729665 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                400000 & 0.1 & 2.0\% & 2494 & 1.2\% & 98.77\% & 0.02\% & 1.197 & 0.5109 & 0.7374 & 0.0247 & 0.0211 & 0.1461 \\
                400000 & 0.05 & 1.0\% & 4376 & 1.52\% & 98.31\% & 0.17\% & 1.1672 & 0.2563 & 0.6258 & 0.0202 & 0.0114 & 0.1099 \\
                400000 & 0.025 & 0.5\% & 7976 & 1.79\% & 97.98\% & 0.23\% & 0.7179 & 0.1526 & 0.4505 & 0.0108 & 0.0056 & 0.0768 \\
                400000 & 0.01 & 0.2\% & 16646 & 3.65\% & 95.75\% & 0.59\% & 0.7179 & 0.0528 & 0.2857 & 0.0061 & 0.0024 & 0.0502 \\
                400000 & 0.005 & 0.1\% & 29228 & 6.35\% & 92.74\% & 0.91\% & 0.3161 & 0.0291 & 0.2142 & 0.0055 & 0.0012 & 0.0364 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 5 (Complete CuPy) (Time: 4m 24s)
```sh
c608-081[gh](732)$ time uv run python shcf_gpu.py 
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 7.56 seconds
Time elapsed for predictions: 0.03634042199701071 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 14.09 seconds
Time elapsed for predictions: 0.062202931847423315 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 28.56 seconds
Time elapsed for predictions: 0.11463687405921519 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 70.26 seconds
Time elapsed for predictions: 0.24824676918797195 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 137.64 seconds
Time elapsed for predictions: 0.45381303201429546 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                400000 & 0.1 & 2.0\% & 1913 & 2.86\% & 97.11\% & 0.03\% & 1.4807 & 0.5176 & 0.7574 & 0.0275 & 0.0221 & 0.0222 \\
                400000 & 0.05 & 1.0\% & 3560 & 2.59\% & 97.37\% & 0.03\% & 0.9197 & 0.1717 & 0.3663 & 0.0177 & 0.011 & 0.0116 \\
                400000 & 0.025 & 0.5\% & 6843 & 3.51\% & 96.45\% & 0.03\% & 0.6079 & 0.1206 & 0.2796 & 0.0077 & 0.0054 & 0.0056 \\
                400000 & 0.01 & 0.2\% & 15294 & 5.31\% & 94.61\% & 0.08\% & 0.2896 & 0.0505 & 0.0865 & 0.0033 & 0.0022 & 0.0023 \\
                400000 & 0.005 & 0.1\% & 28341 & 8.04\% & 91.79\% & 0.17\% & 0.1957 & 0.0341 & 0.0604 & 0.0023 & 0.0011 & 0.0012 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    4m24.873s
user    4m2.482s
sys     0m18.642s
```

## Version 6 (Hybrid) (Time: 19.1s)
```sh
c608-081[gh](720)$ time uv run python shcf_test.py 
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 1.56 seconds
Time elapsed for predictions: 0.03387067397125065 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 1.04 seconds
Time elapsed for predictions: 0.05754775297828019 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 1.63 seconds
Time elapsed for predictions: 0.1025037569925189 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 3.31 seconds
Time elapsed for predictions: 0.21758841490373015 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 5.79 seconds
Time elapsed for predictions: 0.38225826108828187 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                400000 & 0.1 & 2.0\% & 1813 & 1.86\% & 98.14\% & 0.0\% & 1.5447 & 0.2907 & 0.7702 & 0.0242 & 0.0215 & 0.1478 \\
                400000 & 0.05 & 1.0\% & 3289 & 2.45\% & 97.55\% & 0.0\% & 0.1458 & 0.1153 & 0.354 & 0.0109 & 0.0108 & 0.1032 \\
                400000 & 0.025 & 0.5\% & 6134 & 3.33\% & 96.64\% & 0.02\% & 0.6651 & 0.0822 & 0.3854 & 0.007 & 0.0053 & 0.0753 \\
                400000 & 0.01 & 0.2\% & 13399 & 6.08\% & 93.9\% & 0.03\% & 0.2111 & 0.0555 & 0.2629 & 0.0023 & 0.0021 & 0.0464 \\
                400000 & 0.005 & 0.1\% & 23843 & 10.03\% & 89.84\% & 0.14\% & 0.1554 & 0.0267 & 0.1979 & 0.0015 & 0.0011 & 0.0337 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    0m19.079s
user    0m15.459s
sys     0m0.639s
```

## Version 7 (CuPy v2) (Time: 12.2s)
```sh
CUML available: True
2025-11-23 18:41:56 - root - INFO - Logging to file: /work/10973/prajwal_prathiksh/vista/github-repos/knnsur/.logs/20251123_184156.log
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:41:56 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (400000, 2)
2025-11-23 18:41:56 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.1460s
2025-11-23 18:41:57 - src.gpu_adaptive_clustering - INFO - Finished in 1.3142s
2025-11-23 18:41:57 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 1784
Time to train: 1.33 seconds
Time elapsed for predictions: 0.03434985800413415 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:41:57 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (400000, 2)
2025-11-23 18:41:57 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.0767s
2025-11-23 18:41:59 - src.gpu_adaptive_clustering - INFO - Finished in 1.1832s
2025-11-23 18:41:59 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 3411
Time to train: 1.19 seconds
Time elapsed for predictions: 0.05984075099695474 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:41:59 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (400000, 2)
2025-11-23 18:41:59 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.0769s
2025-11-23 18:42:00 - src.gpu_adaptive_clustering - INFO - Finished in 1.4404s
2025-11-23 18:42:00 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 6315
Time to train: 1.45 seconds
Time elapsed for predictions: 0.10543355502886698 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:42:00 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (400000, 2)
2025-11-23 18:42:00 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.0756s
2025-11-23 18:42:02 - src.gpu_adaptive_clustering - INFO - Finished in 1.8647s
2025-11-23 18:42:02 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 14586
Time to train: 1.88 seconds
Time elapsed for predictions: 0.23576232901541516 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:42:02 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (400000, 2)
2025-11-23 18:42:02 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.0753s
2025-11-23 18:42:05 - src.gpu_adaptive_clustering - INFO - Finished in 2.1496s
2025-11-23 18:42:05 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 27030
Time to train: 2.16 seconds
Time elapsed for predictions: 0.4292452320223674 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                400000 & 0.1 & 2.0\% & 1784 & 2.2\% & 97.79\% & 0.01\% & 1.0611 & 0.257 & 0.4209 & 0.0242 & 0.0218 & 0.022 \\
                400000 & 0.05 & 1.0\% & 3411 & 2.39\% & 97.6\% & 0.01\% & 0.5158 & 0.2617 & 0.335 & 0.0119 & 0.0105 & 0.0107 \\
                400000 & 0.025 & 0.5\% & 6315 & 3.02\% & 96.95\% & 0.03\% & 0.8072 & 0.0853 & 0.2441 & 0.0071 & 0.0054 & 0.0057 \\
                400000 & 0.01 & 0.2\% & 14586 & 5.15\% & 94.79\% & 0.06\% & 0.3281 & 0.061 & 0.1175 & 0.0032 & 0.0022 & 0.0023 \\
                400000 & 0.005 & 0.1\% & 27030 & 7.68\% & 92.15\% & 0.17\% & 0.1661 & 0.0305 & 0.0568 & 0.0015 & 0.0011 & 0.0011 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    0m12.265s
user    0m6.728s
sys     0m2.675s
```

# 2D (SHCF) Test: n = 800,000
## Version 1 (Time: 2m 4s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 12.82 seconds
Time elapsed for predictions: 0.04918995400657877 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 14.75 seconds
Time elapsed for predictions: 0.08846727397758514 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 18.16 seconds
Time elapsed for predictions: 0.15688921901164576 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 27.56 seconds
Time elapsed for predictions: 0.3391438249964267 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 40.98 seconds
Time elapsed for predictions: 0.6009737669955939 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                800000 & 0.1 & 2.0\% & 2801 & 1.27\% & 98.63\% & 0.1\% & 2.3446 & 0.5649 & 0.8803 & 0.0334 & 0.0231 & 0.1539 \\
                800000 & 0.05 & 1.0\% & 5225 & 1.41\% & 98.35\% & 0.24\% & 2.0089 & 0.4134 & 0.7133 & 0.0348 & 0.012 & 0.1155 \\
                800000 & 0.025 & 0.5\% & 9535 & 1.51\% & 98.11\% & 0.38\% & 1.0307 & 0.2393 & 0.528 & 0.0163 & 0.0059 & 0.0812 \\
                800000 & 0.01 & 0.2\% & 21096 & 2.31\% & 97.13\% & 0.56\% & 0.8368 & 0.0747 & 0.3282 & 0.0072 & 0.0024 & 0.052 \\
                800000 & 0.005 & 0.1\% & 37593 & 3.82\% & 95.36\% & 0.82\% & 0.315 & 0.0369 & 0.2272 & 0.0042 & 0.0012 & 0.0367 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 2 (Accelerated) (Time: 1m 12s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 6.71 seconds
Time elapsed for predictions: 0.05417023797053844 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 8.13 seconds
Time elapsed for predictions: 0.09407168201869354 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 10.32 seconds
Time elapsed for predictions: 0.1647661229944788 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 15.23 seconds
Time elapsed for predictions: 0.3460207669995725 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 22.04 seconds
Time elapsed for predictions: 0.6038463549921289 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                800000 & 0.1 & 2.0\% & 3082 & 1.19\% & 98.68\% & 0.13\% & 2.7823 & 0.3666 & 0.7926 & 0.0466 & 0.023 & 0.1554 \\
                800000 & 0.05 & 1.0\% & 5566 & 1.21\% & 98.58\% & 0.21\% & 1.8738 & 0.227 & 0.6394 & 0.024 & 0.0118 & 0.1118 \\
                800000 & 0.025 & 0.5\% & 10045 & 1.36\% & 98.27\% & 0.37\% & 0.8902 & 0.1492 & 0.4839 & 0.0136 & 0.0059 & 0.0813 \\
                800000 & 0.01 & 0.2\% & 21519 & 1.91\% & 97.48\% & 0.62\% & 0.7809 & 0.065 & 0.3303 & 0.0088 & 0.0025 & 0.0528 \\
                800000 & 0.005 & 0.1\% & 37816 & 3.47\% & 95.69\% & 0.83\% & 0.3753 & 0.0268 & 0.2243 & 0.005 & 0.0012 & 0.0373 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 3 (Time: 47.6s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 4.62 seconds
Time elapsed for predictions: 0.04901405604323372 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 4.90 seconds
Time elapsed for predictions: 0.08377609896706417 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 5.81 seconds
Time elapsed for predictions: 0.14906654198421165 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 8.61 seconds
Time elapsed for predictions: 0.3132875579758547 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 13.34 seconds
Time elapsed for predictions: 0.5548449970083311 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                800000 & 0.1 & 2.0\% & 2764 & 0.76\% & 99.2\% & 0.04\% & 1.2736 & 0.3501 & 0.6327 & 0.0293 & 0.0213 & 0.1492 \\
                800000 & 0.05 & 1.0\% & 4983 & 0.93\% & 98.96\% & 0.11\% & 1.0773 & 0.1982 & 0.5766 & 0.022 & 0.0115 & 0.1097 \\
                800000 & 0.025 & 0.5\% & 9158 & 1.04\% & 98.78\% & 0.18\% & 0.8072 & 0.1479 & 0.454 & 0.0108 & 0.0059 & 0.0788 \\
                800000 & 0.01 & 0.2\% & 19665 & 1.75\% & 97.83\% & 0.42\% & 0.3397 & 0.0536 & 0.2847 & 0.0055 & 0.0023 & 0.051 \\
                800000 & 0.005 & 0.1\% & 35090 & 3.15\% & 96.21\% & 0.64\% & 0.2538 & 0.0272 & 0.2041 & 0.0036 & 0.0012 & 0.0361 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 4 (Time: 28.1s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 1.90 seconds
Time elapsed for predictions: 0.048375364975072443 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 2.31 seconds
Time elapsed for predictions: 0.0825596550712362 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 3.23 seconds
Time elapsed for predictions: 0.1471255380893126 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 5.85 seconds
Time elapsed for predictions: 0.30898429302033037 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 9.91 seconds
Time elapsed for predictions: 0.5475936199072748 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                800000 & 0.1 & 2.0\% & 2764 & 0.76\% & 99.2\% & 0.04\% & 1.2736 & 0.3501 & 0.6327 & 0.0293 & 0.0213 & 0.1492 \\
                800000 & 0.05 & 1.0\% & 4983 & 0.93\% & 98.96\% & 0.11\% & 1.0773 & 0.1982 & 0.5766 & 0.022 & 0.0115 & 0.1097 \\
                800000 & 0.025 & 0.5\% & 9158 & 1.04\% & 98.78\% & 0.18\% & 0.8072 & 0.1479 & 0.454 & 0.0108 & 0.0059 & 0.0788 \\
                800000 & 0.01 & 0.2\% & 19665 & 1.75\% & 97.83\% & 0.42\% & 0.3397 & 0.0536 & 0.2847 & 0.0055 & 0.0023 & 0.051 \\
                800000 & 0.005 & 0.1\% & 35090 & 3.14\% & 96.21\% & 0.64\% & 0.2538 & 0.0272 & 0.2041 & 0.0036 & 0.0012 & 0.0361 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 5 (Complete CuPy) (Time: 4m 16s)
```sh
c608-081[gh](730)$ time uv run python shcf_gpu.py 
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 7.36 seconds
Time elapsed for predictions: 0.035443756030872464 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 14.09 seconds
Time elapsed for predictions: 0.06604527402669191 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 27.05 seconds
Time elapsed for predictions: 0.12081580818630755 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 67.51 seconds
Time elapsed for predictions: 0.2686100220307708 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 135.58 seconds
Time elapsed for predictions: 0.4951401411090046 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                800000 & 0.1 & 2.0\% & 1904 & 2.4\% & 97.55\% & 0.05\% & 0.6587 & 0.256 & 0.2888 & 0.0266 & 0.0221 & 0.0223 \\
                800000 & 0.05 & 1.0\% & 3787 & 2.22\% & 97.76\% & 0.02\% & 1.1899 & 0.2362 & 0.5494 & 0.0171 & 0.0105 & 0.0121 \\
                800000 & 0.025 & 0.5\% & 7259 & 2.46\% & 97.5\% & 0.03\% & 0.3977 & 0.0922 & 0.124 & 0.0088 & 0.0055 & 0.0061 \\
                800000 & 0.01 & 0.2\% & 16636 & 3.66\% & 96.31\% & 0.04\% & 0.3019 & 0.0772 & 0.137 & 0.004 & 0.0022 & 0.0024 \\
                800000 & 0.005 & 0.1\% & 30976 & 5.27\% & 94.67\% & 0.06\% & 0.1821 & 0.0263 & 0.0582 & 0.0016 & 0.0011 & 0.0011 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    4m16.943s
user    3m55.389s
sys     0m19.490s
```

## Version 6 (Hybrid) (Time: 24.1s)
```sh
c608-081[gh](722)$ time uv run python shcf_test.py 
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 2.16 seconds
Time elapsed for predictions: 0.034446988021954894 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 1.57 seconds
Time elapsed for predictions: 0.06019938411191106 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 2.17 seconds
Time elapsed for predictions: 0.1074980259872973 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 3.90 seconds
Time elapsed for predictions: 0.23716642195358872 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 6.70 seconds
Time elapsed for predictions: 0.4251497979275882 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                800000 & 0.1 & 2.0\% & 1844 & 1.59\% & 98.41\% & 0.0\% & 0.5189 & 0.5189 & 0.7203 & 0.02 & 0.02 & 0.1416 \\
                800000 & 0.05 & 1.0\% & 3504 & 1.75\% & 98.25\% & 0.0\% & 0.1072 & 0.1072 & 0.3275 & 0.0101 & 0.0101 & 0.1007 \\
                800000 & 0.025 & 0.5\% & 6437 & 2.24\% & 97.75\% & 0.02\% & 0.3978 & 0.1089 & 0.4171 & 0.0061 & 0.0052 & 0.0736 \\
                800000 & 0.01 & 0.2\% & 14631 & 3.35\% & 96.64\% & 0.01\% & 0.1547 & 0.0451 & 0.2511 & 0.0027 & 0.0021 & 0.0472 \\
                800000 & 0.005 & 0.1\% & 26564 & 5.85\% & 94.11\% & 0.04\% & 0.0943 & 0.0191 & 0.1737 & 0.0016 & 0.0011 & 0.0333 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    0m24.165s
user    0m18.655s
sys     0m0.654s
```

## Version 7 (CuPy v2) (Time: 13.3s)
```sh
CUML available: True
2025-11-23 18:43:05 - root - INFO - Logging to file: /work/10973/prajwal_prathiksh/vista/github-repos/knnsur/.logs/20251123_184305.log
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:43:05 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (800000, 2)
2025-11-23 18:43:06 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.1671s
2025-11-23 18:43:07 - src.gpu_adaptive_clustering - INFO - Finished in 1.3935s
2025-11-23 18:43:07 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 1893
Time to train: 1.41 seconds
Time elapsed for predictions: 0.036650896014180034 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:43:07 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (800000, 2)
2025-11-23 18:43:07 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.1074s
2025-11-23 18:43:08 - src.gpu_adaptive_clustering - INFO - Finished in 1.4162s
2025-11-23 18:43:08 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 3549
Time to train: 1.43 seconds
Time elapsed for predictions: 0.062260730017442256 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:43:09 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (800000, 2)
2025-11-23 18:43:09 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.1074s
2025-11-23 18:43:10 - src.gpu_adaptive_clustering - INFO - Finished in 1.4111s
2025-11-23 18:43:10 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 6755
Time to train: 1.42 seconds
Time elapsed for predictions: 0.11229810700751841 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:43:10 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (800000, 2)
2025-11-23 18:43:10 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.1074s
2025-11-23 18:43:12 - src.gpu_adaptive_clustering - INFO - Finished in 1.6950s
2025-11-23 18:43:12 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 15613
Time to train: 1.71 seconds
Time elapsed for predictions: 0.251224217005074 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:43:12 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (800000, 2)
2025-11-23 18:43:12 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.1069s
2025-11-23 18:43:15 - src.gpu_adaptive_clustering - INFO - Finished in 2.8839s
2025-11-23 18:43:15 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 29182
Time to train: 2.89 seconds
Time elapsed for predictions: 0.46338501694845036 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                800000 & 0.1 & 2.0\% & 1893 & 1.94\% & 98.04\% & 0.01\% & 0.6791 & 0.2501 & 0.3132 & 0.0226 & 0.0214 & 0.0214 \\
                800000 & 0.05 & 1.0\% & 3549 & 1.89\% & 98.09\% & 0.02\% & 0.5052 & 0.1362 & 0.275 & 0.0124 & 0.0109 & 0.011 \\
                800000 & 0.025 & 0.5\% & 6755 & 2.3\% & 97.69\% & 0.01\% & 0.3206 & 0.1561 & 0.2181 & 0.0058 & 0.0051 & 0.0054 \\
                800000 & 0.01 & 0.2\% & 15613 & 3.25\% & 96.71\% & 0.04\% & 0.184 & 0.076 & 0.0983 & 0.0027 & 0.0021 & 0.0022 \\
                800000 & 0.005 & 0.1\% & 29182 & 4.85\% & 95.08\% & 0.07\% & 0.1608 & 0.0382 & 0.0607 & 0.0019 & 0.0011 & 0.0011 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    0m13.317s
user    0m7.023s
sys     0m2.848s
```

# 2D (SHCF) Test: n = 3,200,000
## Version 1 (Time: 4m 34s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 39.94 seconds
Time elapsed for predictions: 0.05475225701229647 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 42.89 seconds
Time elapsed for predictions: 0.10039516899269074 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 47.58 seconds
Time elapsed for predictions: 0.1837062650010921 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 59.16 seconds
Time elapsed for predictions: 0.4109925450175069 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 75.70 seconds
Time elapsed for predictions: 0.7478364929556847 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                3200000 & 0.1 & 2.0\% & 3116 & 0.97\% & 98.98\% & 0.05\% & 3.4642 & 0.7256 & 1.0143 & 0.0335 & 0.0236 & 0.1542 \\
                3200000 & 0.05 & 1.0\% & 5965 & 0.8\% & 99.1\% & 0.1\% & 0.9079 & 0.3867 & 0.6316 & 0.0257 & 0.0123 & 0.1161 \\
                3200000 & 0.025 & 0.5\% & 11215 & 0.76\% & 99.09\% & 0.15\% & 0.9876 & 0.1624 & 0.4674 & 0.0091 & 0.0057 & 0.0772 \\
                3200000 & 0.01 & 0.2\% & 25439 & 1.01\% & 98.72\% & 0.27\% & 0.4563 & 0.0728 & 0.3122 & 0.0062 & 0.0024 & 0.051 \\
                3200000 & 0.005 & 0.1\% & 46898 & 1.18\% & 98.48\% & 0.34\% & 0.3654 & 0.0319 & 0.2161 & 0.0033 & 0.0012 & 0.0359 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 2 (Accelerated) (Time: 2m 36s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 19.90 seconds
Time elapsed for predictions: 0.05250216502463445 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 22.40 seconds
Time elapsed for predictions: 0.09410497499629855 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 26.29 seconds
Time elapsed for predictions: 0.1717532220063731 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 34.21 seconds
Time elapsed for predictions: 0.38198837998788804 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 43.67 seconds
Time elapsed for predictions: 0.6947888839640655 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                3200000 & 0.1 & 2.0\% & 2987 & 1.4\% & 98.4\% & 0.2\% & 2.4203 & 0.5528 & 0.7803 & 0.0445 & 0.0237 & 0.1575 \\
                3200000 & 0.05 & 1.0\% & 5620 & 1.11\% & 98.6\% & 0.29\% & 2.573 & 0.3916 & 0.7216 & 0.0354 & 0.0123 & 0.1178 \\
                3200000 & 0.025 & 0.5\% & 10560 & 1.11\% & 98.51\% & 0.38\% & 2.0012 & 0.1564 & 0.5007 & 0.0275 & 0.0058 & 0.0812 \\
                3200000 & 0.01 & 0.2\% & 23906 & 1.01\% & 98.42\% & 0.57\% & 0.5877 & 0.0806 & 0.3379 & 0.0101 & 0.0024 & 0.0525 \\
                3200000 & 0.005 & 0.1\% & 43739 & 1.08\% & 98.16\% & 0.76\% & 0.511 & 0.0422 & 0.2534 & 0.0045 & 0.0013 & 0.0376 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 3 (Time: 1m 54s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 17.53 seconds
Time elapsed for predictions: 0.051398105977568775 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 17.95 seconds
Time elapsed for predictions: 0.09137935098260641 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 19.02 seconds
Time elapsed for predictions: 0.16563432000111789 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 22.00 seconds
Time elapsed for predictions: 0.36827575601637363 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 26.79 seconds
Time elapsed for predictions: 0.6693715450237505 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                3200000 & 0.1 & 2.0\% & 2950 & 1.26\% & 98.6\% & 0.14\% & 2.0113 & 0.4159 & 0.7483 & 0.0445 & 0.0239 & 0.1584 \\
                3200000 & 0.05 & 1.0\% & 5504 & 0.92\% & 98.84\% & 0.23\% & 1.9776 & 0.3805 & 0.6881 & 0.0288 & 0.0117 & 0.1147 \\
                3200000 & 0.025 & 0.5\% & 10203 & 0.8\% & 98.88\% & 0.33\% & 1.5027 & 0.1659 & 0.5257 & 0.0207 & 0.006 & 0.0835 \\
                3200000 & 0.01 & 0.2\% & 23160 & 0.69\% & 98.87\% & 0.43\% & 0.6215 & 0.077 & 0.3219 & 0.0074 & 0.0025 & 0.0518 \\
                3200000 & 0.005 & 0.1\% & 42356 & 0.86\% & 98.48\% & 0.66\% & 0.2907 & 0.0392 & 0.2336 & 0.0047 & 0.0012 & 0.0373 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 4 (Time: 55s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 6.36 seconds
Time elapsed for predictions: 0.05083717405796051 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 7.02 seconds
Time elapsed for predictions: 0.09027685201726854 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 8.40 seconds
Time elapsed for predictions: 0.16340424900408834 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 11.57 seconds
Time elapsed for predictions: 0.36357142601627856 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 16.48 seconds
Time elapsed for predictions: 0.659658906981349 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                3200000 & 0.1 & 2.0\% & 2950 & 1.26\% & 98.6\% & 0.14\% & 2.0113 & 0.4159 & 0.7483 & 0.0445 & 0.0239 & 0.1584 \\
                3200000 & 0.05 & 1.0\% & 5504 & 0.92\% & 98.84\% & 0.23\% & 1.9776 & 0.3805 & 0.6881 & 0.0288 & 0.0117 & 0.1147 \\
                3200000 & 0.025 & 0.5\% & 10203 & 0.8\% & 98.88\% & 0.33\% & 1.5027 & 0.1659 & 0.5257 & 0.0207 & 0.006 & 0.0835 \\
                3200000 & 0.01 & 0.2\% & 23160 & 0.69\% & 98.87\% & 0.43\% & 0.6215 & 0.077 & 0.3219 & 0.0074 & 0.0025 & 0.0518 \\
                3200000 & 0.005 & 0.1\% & 42356 & 0.86\% & 98.49\% & 0.66\% & 0.2907 & 0.0392 & 0.2336 & 0.0047 & 0.0012 & 0.0373 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 5 (Complete CuPy) (Time: 4m 52s)
```sh
c608-082[gh](697)$ time uv run python shcf_gpu.py 
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 8.86 seconds
Time elapsed for predictions: 0.038804734824225307 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 16.75 seconds
Time elapsed for predictions: 0.06870165793225169 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 32.32 seconds
Time elapsed for predictions: 0.1280414902139455 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 75.74 seconds
Time elapsed for predictions: 0.29357613902539015 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 150.06 seconds
Time elapsed for predictions: 0.5560964839532971 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                3200000 & 0.1 & 2.0\% & 2029 & 1.66\% & 98.33\% & 0.01\% & 0.493 & 0.2192 & 0.308 & 0.0222 & 0.0207 & 0.0211 \\
                3200000 & 0.05 & 1.0\% & 3971 & 1.66\% & 98.33\% & 0.01\% & 0.9441 & 0.2704 & 0.5688 & 0.0141 & 0.0111 & 0.0122 \\
                3200000 & 0.025 & 0.5\% & 7773 & 1.67\% & 98.3\% & 0.03\% & 0.8254 & 0.1464 & 0.4155 & 0.0074 & 0.0055 & 0.0057 \\
                3200000 & 0.01 & 0.2\% & 18423 & 2.0\% & 97.98\% & 0.02\% & 0.2533 & 0.0542 & 0.0989 & 0.0035 & 0.0021 & 0.0023 \\
                3200000 & 0.005 & 0.1\% & 35561 & 2.46\% & 97.5\% & 0.03\% & 0.1276 & 0.0172 & 0.0526 & 0.0015 & 0.0011 & 0.0011 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    4m52.092s
user    4m27.910s
sys     0m19.947s
```

## Version 6 (Hybrid) (Time: 42s)
```sh
c608-081[gh](724)$ time uv run python shcf_test.py 
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 5.41 seconds
Time elapsed for predictions: 0.03443178907036781 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 5.09 seconds
Time elapsed for predictions: 0.061198575887829065 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 6.07 seconds
Time elapsed for predictions: 0.11253018118441105 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 8.14 seconds
Time elapsed for predictions: 0.25834278482943773 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 11.17 seconds
Time elapsed for predictions: 0.48002614779397845 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                3200000 & 0.1 & 2.0\% & 1846 & 1.3\% & 98.69\% & 0.01\% & 0.5245 & 0.2554 & 0.5319 & 0.0236 & 0.0211 & 0.1467 \\
                3200000 & 0.05 & 1.0\% & 3555 & 1.38\% & 98.62\% & 0.0\% & 0.4858 & 0.2992 & 0.547 & 0.0108 & 0.0105 & 0.1024 \\
                3200000 & 0.025 & 0.5\% & 6802 & 1.28\% & 98.71\% & 0.01\% & 0.2231 & 0.1035 & 0.3675 & 0.0053 & 0.0051 & 0.0716 \\
                3200000 & 0.01 & 0.2\% & 16010 & 1.63\% & 98.36\% & 0.01\% & 0.1111 & 0.081 & 0.2576 & 0.0023 & 0.0022 & 0.0467 \\
                3200000 & 0.005 & 0.1\% & 30086 & 2.19\% & 97.8\% & 0.01\% & 0.0608 & 0.0189 & 0.1605 & 0.0011 & 0.001 & 0.0324 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    0m42.359s
user    0m38.501s
sys     0m1.111s
```

## Version 7 (CuPy v2) (Time: 16.9s)
```sh
CUML available: True
2025-11-23 18:44:38 - root - INFO - Logging to file: /work/10973/prajwal_prathiksh/vista/github-repos/knnsur/.logs/20251123_184438.log
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:44:39 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (3200000, 2)
2025-11-23 18:44:39 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.6763s
2025-11-23 18:44:41 - src.gpu_adaptive_clustering - INFO - Finished in 2.1963s
2025-11-23 18:44:41 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 1973
Time to train: 2.21 seconds
Time elapsed for predictions: 0.037045539007522166 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:44:41 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (3200000, 2)
2025-11-23 18:44:42 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.5556s
2025-11-23 18:44:43 - src.gpu_adaptive_clustering - INFO - Finished in 2.0126s
2025-11-23 18:44:43 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 3796
Time to train: 2.02 seconds
Time elapsed for predictions: 0.06532641500234604 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:44:43 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (3200000, 2)
2025-11-23 18:44:44 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.5554s
2025-11-23 18:44:46 - src.gpu_adaptive_clustering - INFO - Finished in 2.1867s
2025-11-23 18:44:46 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 7378
Time to train: 2.20 seconds
Time elapsed for predictions: 0.12147191900294274 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:44:46 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (3200000, 2)
2025-11-23 18:44:46 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.5552s
2025-11-23 18:44:48 - src.gpu_adaptive_clustering - INFO - Finished in 2.4100s
2025-11-23 18:44:48 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 17392
Time to train: 2.42 seconds
Time elapsed for predictions: 0.2780115669593215 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:44:49 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (3200000, 2)
2025-11-23 18:44:49 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 0.5556s
2025-11-23 18:44:52 - src.gpu_adaptive_clustering - INFO - Finished in 2.7054s
2025-11-23 18:44:52 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 33087
Time to train: 2.72 seconds
Time elapsed for predictions: 0.5244578699930571 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                3200000 & 0.1 & 2.0\% & 1973 & 1.65\% & 98.34\% & 0.01\% & 0.2893 & 0.2405 & 0.2439 & 0.0241 & 0.0222 & 0.022 \\
                3200000 & 0.05 & 1.0\% & 3796 & 1.32\% & 98.68\% & 0.0\% & 0.5165 & 0.1624 & 0.2929 & 0.0136 & 0.0114 & 0.0117 \\
                3200000 & 0.025 & 0.5\% & 7378 & 1.51\% & 98.48\% & 0.0\% & 0.1431 & 0.1178 & 0.1191 & 0.006 & 0.0057 & 0.0056 \\
                3200000 & 0.01 & 0.2\% & 17392 & 1.81\% & 98.18\% & 0.01\% & 0.1363 & 0.0332 & 0.0646 & 0.0028 & 0.0021 & 0.0022 \\
                3200000 & 0.005 & 0.1\% & 33087 & 2.25\% & 97.74\% & 0.01\% & 0.1238 & 0.0238 & 0.0558 & 0.0013 & 0.0011 & 0.0011 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    0m16.965s
user    0m10.955s
sys     0m3.091s
```

# 2D (SHCF) Test: n = 10,000,000
## Version 1 (Time: 12m 32s)

## Version 3 (Time: 4m 56s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 51.49 seconds
Time elapsed for predictions: 0.058705562027171254 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 53.46 seconds
Time elapsed for predictions: 0.10605836601462215 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 56.17 seconds
Time elapsed for predictions: 0.19361822900827974 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 60.50 seconds
Time elapsed for predictions: 0.43674639996606857 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold: 8192 for dimension: 2
Time to train: 66.05 seconds
Time elapsed for predictions: 0.8112662389175966 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                10000000 & 0.1 & 2.0\% & 3446 & 0.84\% & 99.05\% & 0.12\% & 2.5486 & 1.2264 & 1.0376 & 0.0336 & 0.0231 & 0.1548 \\
                10000000 & 0.05 & 1.0\% & 6501 & 0.75\% & 99.13\% & 0.12\% & 1.716 & 0.4904 & 0.7108 & 0.0176 & 0.0107 & 0.1073 \\
                10000000 & 0.025 & 0.5\% & 12152 & 0.59\% & 99.18\% & 0.24\% & 1.1398 & 0.2817 & 0.5567 & 0.0239 & 0.0063 & 0.0847 \\
                10000000 & 0.01 & 0.2\% & 27839 & 0.47\% & 99.27\% & 0.26\% & 0.4302 & 0.0935 & 0.3323 & 0.007 & 0.0024 & 0.0511 \\
                10000000 & 0.005 & 0.1\% & 52070 & 0.46\% & 99.16\% & 0.38\% & 0.2096 & 0.053 & 0.2421 & 0.0035 & 0.0012 & 0.0366 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 4 (Time: 1m 44s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold (dim=2): 8192
Time to train: 16.03 seconds
Time elapsed for predictions: 0.034881183062680066 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold (dim=2): 8192
Time to train: 16.66 seconds
Time elapsed for predictions: 0.061885106028057635 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold (dim=2): 8192
Time to train: 18.90 seconds
Time elapsed for predictions: 0.11492149892728776 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold (dim=2): 8192
Time to train: 22.57 seconds
Time elapsed for predictions: 0.26930444000754505 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold (dim=2): 8192
Time to train: 26.47 seconds
Time elapsed for predictions: 0.5075021359371021 seconds
```

```latex
\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                10000000 & 0.1 & 2.0\% & 1854 & 1.23\% & 98.76\% & 0.01\% & 0.278 & 0.2385 & 0.4921 & 0.0228 & 0.0222 & 0.1477 \\
                10000000 & 0.05 & 1.0\% & 3617 & 1.22\% & 98.77\% & 0.01\% & 0.1363 & 0.1351 & 0.367 & 0.0128 & 0.0109 & 0.1067 \\
                10000000 & 0.025 & 0.5\% & 6964 & 1.12\% & 98.88\% & 0.0\% & 0.0883 & 0.0879 & 0.2965 & 0.0057 & 0.0057 & 0.0755 \\
                10000000 & 0.01 & 0.2\% & 16788 & 1.1\% & 98.9\% & 0.0\% & 0.0286 & 0.0252 & 0.1588 & 0.0024 & 0.0022 & 0.0473 \\
                10000000 & 0.005 & 0.1\% & 31869 & 1.29\% & 98.7\% & 0.01\% & 0.1174 & 0.0586 & 0.2439 & 0.0013 & 0.0011 & 0.0331 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
```

## Version 5 (Complete CuPy) (Time: 5m 8s)
```sh
c608-081[gh](711)$ time uv run python shcf_gpu.py 
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 10.16 seconds
Time elapsed for predictions: 0.03906202712096274 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 17.83 seconds
Time elapsed for predictions: 0.07139645400457084 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 33.89 seconds
Time elapsed for predictions: 0.13308478496037424 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 81.74 seconds
Time elapsed for predictions: 0.31831784290261567 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 153.80 seconds
Time elapsed for predictions: 0.6089395999442786 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                10000000 & 0.1 & 2.0\% & 2115 & 1.53\% & 98.44\% & 0.03\% & 1.2802 & 0.3391 & 0.5716 & 0.0382 & 0.0226 & 0.0257 \\
                10000000 & 0.05 & 1.0\% & 4110 & 1.54\% & 98.44\% & 0.02\% & 0.5795 & 0.1454 & 0.2526 & 0.0148 & 0.0112 & 0.0118 \\
                10000000 & 0.025 & 0.5\% & 8080 & 1.51\% & 98.48\% & 0.01\% & 0.2723 & 0.1051 & 0.1278 & 0.0059 & 0.0056 & 0.0056 \\
                10000000 & 0.01 & 0.2\% & 19683 & 1.51\% & 98.48\% & 0.01\% & 0.4269 & 0.1231 & 0.2347 & 0.0036 & 0.0023 & 0.0025 \\
                10000000 & 0.005 & 0.1\% & 38127 & 1.73\% & 98.25\% & 0.02\% & 0.087 & 0.0222 & 0.037 & 0.0015 & 0.0011 & 0.0012 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    5m8.762s
user    4m40.216s
sys     0m24.486s
```

## Version 6 (Hybrid) (Time: 1m 38s)
```sh
c608-081[gh](726)$ time uv run python shcf_test.py 
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 13.81 seconds
Time elapsed for predictions: 0.03458791598677635 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 14.49 seconds
Time elapsed for predictions: 0.06279053306207061 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 16.67 seconds
Time elapsed for predictions: 0.11642161104828119 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 20.21 seconds
Time elapsed for predictions: 0.2712896061129868 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 24.13 seconds
Time elapsed for predictions: 0.5105098360218108 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                10000000 & 0.1 & 2.0\% & 1859 & 1.23\% & 98.75\% & 0.01\% & 0.278 & 0.2385 & 0.4921 & 0.0228 & 0.0222 & 0.1477 \\
                10000000 & 0.05 & 1.0\% & 3626 & 1.23\% & 98.76\% & 0.01\% & 0.1363 & 0.1351 & 0.367 & 0.0128 & 0.0109 & 0.1067 \\
                10000000 & 0.025 & 0.5\% & 6974 & 1.14\% & 98.86\% & 0.0\% & 0.0883 & 0.0879 & 0.2965 & 0.0057 & 0.0057 & 0.0755 \\
                10000000 & 0.01 & 0.2\% & 16820 & 1.09\% & 98.91\% & 0.0\% & 0.0219 & 0.0219 & 0.148 & 0.002 & 0.002 & 0.045 \\
                10000000 & 0.005 & 0.1\% & 31900 & 1.27\% & 98.71\% & 0.01\% & 0.1174 & 0.0581 & 0.2445 & 0.0014 & 0.0011 & 0.0336 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    1m38.019s
user    1m33.432s
sys     0m2.566s
```


## Version 7 (CuPy v2) (Time: 26s)
```sh
CUML available: True
2025-11-23 18:45:23 - root - INFO - Logging to file: /work/10973/prajwal_prathiksh/vista/github-repos/knnsur/.logs/20251123_184523.log
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:45:24 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (10000000, 2)
2025-11-23 18:45:25 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 1.4437s
2025-11-23 18:45:27 - src.gpu_adaptive_clustering - INFO - Finished in 3.5899s
2025-11-23 18:45:27 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 2019
Time to train: 3.61 seconds
Time elapsed for predictions: 0.037756700010504574 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:45:28 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (10000000, 2)
2025-11-23 18:45:29 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 1.2133s
2025-11-23 18:45:31 - src.gpu_adaptive_clustering - INFO - Finished in 3.3630s
2025-11-23 18:45:31 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 3888
Time to train: 3.37 seconds
Time elapsed for predictions: 0.06671702698804438 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:45:32 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (10000000, 2)
2025-11-23 18:45:33 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 1.2129s
2025-11-23 18:45:36 - src.gpu_adaptive_clustering - INFO - Finished in 3.6268s
2025-11-23 18:45:36 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 7560
Time to train: 3.64 seconds
Time elapsed for predictions: 0.12474136298988014 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:45:37 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (10000000, 2)
2025-11-23 18:45:38 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 1.2192s
2025-11-23 18:45:41 - src.gpu_adaptive_clustering - INFO - Finished in 3.9125s
2025-11-23 18:45:41 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 18293
Time to train: 3.92 seconds
Time elapsed for predictions: 0.29237179696792737 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:45:42 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (10000000, 2)
2025-11-23 18:45:43 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 1.2191s
2025-11-23 18:45:46 - src.gpu_adaptive_clustering - INFO - Finished in 4.1415s
2025-11-23 18:45:46 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 35528
Time to train: 4.15 seconds
Time elapsed for predictions: 0.5616304139839485 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                10000000 & 0.1 & 2.0\% & 2019 & 1.26\% & 98.73\% & 0.01\% & 0.3037 & 0.2562 & 0.2712 & 0.0272 & 0.022 & 0.0228 \\
                10000000 & 0.05 & 1.0\% & 3888 & 1.19\% & 98.81\% & 0.0\% & 0.3831 & 0.1411 & 0.2249 & 0.0121 & 0.0109 & 0.011 \\
                10000000 & 0.025 & 0.5\% & 7560 & 1.22\% & 98.77\% & 0.01\% & 0.4029 & 0.0688 & 0.2198 & 0.0064 & 0.006 & 0.0058 \\
                10000000 & 0.01 & 0.2\% & 18293 & 1.3\% & 98.68\% & 0.01\% & 0.2536 & 0.0412 & 0.1006 & 0.0033 & 0.0023 & 0.0024 \\
                10000000 & 0.005 & 0.1\% & 35528 & 1.51\% & 98.48\% & 0.01\% & 0.1186 & 0.0424 & 0.0712 & 0.0013 & 0.0011 & 0.0011 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    0m26.705s
user    0m19.740s
sys     0m4.116s
```

# 2D (SHCF) Test: n = 100,000,000
## Version 4 (Time: 13m 31s)
```sh
c608-052[gh](572)$ time uv run python shcf.py 
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold (dim=2): 8192
Time to train: 122.77 seconds
Time elapsed for predictions: 0.04202492709737271 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold (dim=2): 8192
Time to train: 137.37 seconds
Time elapsed for predictions: 0.06527890788856894 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold (dim=2): 8192
Time to train: 149.81 seconds
Time elapsed for predictions: 0.11941940395627171 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold (dim=2): 8192
Time to train: 178.95 seconds
Time elapsed for predictions: 0.28549521300010383 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold (dim=2): 8192
Time to train: 200.79 seconds
Time elapsed for predictions: 0.5369986499426886 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                100000000 & 0.1 & 2.0\% & 1924 & 1.4\% & 98.59\% & 0.01\% & 1.9581 & 0.2246 & 0.7155 & 0.0206 & 0.0202 & 0.1423 \\
                100000000 & 0.05 & 1.0\% & 3815 & 1.26\% & 98.74\% & 0.0\% & 0.1414 & 0.1345 & 0.3692 & 0.011 & 0.0106 & 0.1035 \\
                100000000 & 0.025 & 0.5\% & 7271 & 1.12\% & 98.87\% & 0.01\% & 0.5457 & 0.0663 & 0.3997 & 0.0055 & 0.0053 & 0.0727 \\
                100000000 & 0.01 & 0.2\% & 17859 & 0.96\% & 99.03\% & 0.01\% & 0.1303 & 0.0584 & 0.2695 & 0.0025 & 0.0022 & 0.0473 \\
                100000000 & 0.005 & 0.1\% & 34005 & 0.92\% & 99.07\% & 0.01\% & 0.1442 & 0.0126 & 0.2089 & 0.0013 & 0.0011 & 0.0337 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    13m31.647s
user    12m38.538s
sys     0m32.616s
```


## Version 5 (Complete CuPy) (Time: 7m 42s)
```sh
c608-081[gh](713)$ time uv run python shcf_gpu.py 
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 24.48 seconds
Time elapsed for predictions: 0.04039616486988962 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 34.09 seconds
Time elapsed for predictions: 0.07363599492236972 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 52.58 seconds
Time elapsed for predictions: 0.1402705041691661 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 104.03 seconds
Time elapsed for predictions: 0.3382324599660933 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 186.74 seconds
Time elapsed for predictions: 0.6615750398486853 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                100000000 & 0.1 & 2.0\% & 2156 & 1.4\% & 98.57\% & 0.04\% & 0.7473 & 0.2623 & 0.3076 & 0.0343 & 0.0235 & 0.0247 \\
                100000000 & 0.05 & 1.0\% & 4255 & 1.29\% & 98.7\% & 0.01\% & 0.7131 & 0.5568 & 0.4962 & 0.0133 & 0.0115 & 0.0118 \\
                100000000 & 0.025 & 0.5\% & 8491 & 1.16\% & 98.83\% & 0.0\% & 0.0892 & 0.0684 & 0.0753 & 0.0056 & 0.0051 & 0.0053 \\
                100000000 & 0.01 & 0.2\% & 20942 & 1.13\% & 98.86\% & 0.01\% & 0.2386 & 0.0545 & 0.1143 & 0.0024 & 0.0021 & 0.0022 \\
                100000000 & 0.005 & 0.1\% & 41315 & 1.19\% & 98.8\% & 0.01\% & 0.124 & 0.0146 & 0.0606 & 0.0013 & 0.0011 & 0.0011 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    7m42.043s
user    6m54.928s
sys     0m44.626s
```

## Version 6 (Hybrid) (Time: 11m 35s)
```sh
c608-081[gh](728)$ time uv run python shcf_test.py 
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 97.17 seconds
Time elapsed for predictions: 0.035746979992836714 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 110.23 seconds
Time elapsed for predictions: 0.06597985606640577 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 122.70 seconds
Time elapsed for predictions: 0.12054600194096565 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 145.45 seconds
Time elapsed for predictions: 0.28884219587780535 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 162.40 seconds
Time elapsed for predictions: 0.5482421130873263 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                100000000 & 0.1 & 2.0\% & 1924 & 1.4\% & 98.59\% & 0.01\% & 1.9581 & 0.2246 & 0.7155 & 0.0206 & 0.0202 & 0.1423 \\
                100000000 & 0.05 & 1.0\% & 3815 & 1.26\% & 98.74\% & 0.0\% & 0.1414 & 0.1345 & 0.3692 & 0.011 & 0.0106 & 0.1035 \\
                100000000 & 0.025 & 0.5\% & 7271 & 1.12\% & 98.87\% & 0.01\% & 0.5457 & 0.0663 & 0.3997 & 0.0055 & 0.0053 & 0.0727 \\
                100000000 & 0.01 & 0.2\% & 17915 & 0.93\% & 99.06\% & 0.01\% & 0.1303 & 0.0584 & 0.2695 & 0.0025 & 0.0022 & 0.0473 \\
                100000000 & 0.005 & 0.1\% & 34299 & 0.89\% & 99.1\% & 0.01\% & 0.1442 & 0.0126 & 0.2089 & 0.0013 & 0.0011 & 0.0337 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    11m35.685s
user    11m4.493s
sys     0m29.064s
```

## Version 7 (CuPy v2) (Time: 3m 17s)
```sh
CUML available: True
2025-11-23 18:46:45 - root - INFO - Logging to file: /work/10973/prajwal_prathiksh/vista/github-repos/knnsur/.logs/20251123_184645.log
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:46:56 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (100000000, 2)
2025-11-23 18:47:09 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 13.1544s
2025-11-23 18:47:20 - src.gpu_adaptive_clustering - INFO - Finished in 24.7790s
2025-11-23 18:47:20 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 2006
Time to train: 24.80 seconds
Time elapsed for predictions: 0.037457214028108865 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:47:31 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (100000000, 2)
2025-11-23 18:47:44 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 13.0030s
2025-11-23 18:47:57 - src.gpu_adaptive_clustering - INFO - Finished in 26.2551s
2025-11-23 18:47:57 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 4030
Time to train: 26.27 seconds
Time elapsed for predictions: 0.0699945809901692 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:48:08 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (100000000, 2)
2025-11-23 18:48:21 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 13.0076s
2025-11-23 18:48:36 - src.gpu_adaptive_clustering - INFO - Finished in 27.3434s
2025-11-23 18:48:36 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 7836
Time to train: 27.35 seconds
Time elapsed for predictions: 0.12948909000260755 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:48:46 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (100000000, 2)
2025-11-23 18:48:59 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 13.0052s
2025-11-23 18:49:16 - src.gpu_adaptive_clustering - INFO - Finished in 29.4774s
2025-11-23 18:49:16 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 19326
Time to train: 29.49 seconds
Time elapsed for predictions: 0.3093927009613253 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
2025-11-23 18:49:27 - src.gpu_adaptive_clustering - INFO - Starting GPU Clustering on (100000000, 2)
2025-11-23 18:49:40 - src.gpu_adaptive_clustering - INFO - Initial KMeans done in 13.0018s
2025-11-23 18:49:59 - src.gpu_adaptive_clustering - INFO - Finished in 31.8777s
2025-11-23 18:49:59 - src.gpu_adaptive_clustering - INFO - Final Cluster Count: 38460
Time to train: 31.89 seconds
Time elapsed for predictions: 0.6083428340498358 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                100000000 & 0.1 & 2.0\% & 2006 & 1.26\% & 98.73\% & 0.01\% & 0.4166 & 0.3001 & 0.3077 & 0.0256 & 0.0232 & 0.0231 \\
                100000000 & 0.05 & 1.0\% & 4030 & 1.08\% & 98.92\% & 0.0\% & 0.3638 & 0.1236 & 0.2307 & 0.0109 & 0.0105 & 0.0105 \\
                100000000 & 0.025 & 0.5\% & 7836 & 1.08\% & 98.92\% & 0.0\% & 0.1281 & 0.0941 & 0.1001 & 0.0065 & 0.006 & 0.006 \\
                100000000 & 0.01 & 0.2\% & 19326 & 0.91\% & 99.09\% & 0.01\% & 0.1701 & 0.0293 & 0.0751 & 0.0024 & 0.0022 & 0.0022 \\
                100000000 & 0.005 & 0.1\% & 38460 & 0.87\% & 99.12\% & 0.01\% & 0.0528 & 0.0178 & 0.027 & 0.0013 & 0.001 & 0.0011 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    3m17.655s
user    2m59.962s
sys     0m14.883s
```