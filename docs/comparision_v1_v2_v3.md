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

# 2D (SHCF) Test: n = 800,000
## Version 1 (Time: 2m 4s)
```sh
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
^[      Time to train: 12.82 seconds
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

## Version 6 (Hybrid CuPy + Numpy + Numba) (Time: 49s)
```sh
c608-141[gh](583)$ time uv run python shcf_test.py 
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using GPU for function evaluation.
Time to train: 6.68 seconds
Time elapsed for predictions: 0.03489174181595445 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using GPU for function evaluation.
Time to train: 5.63 seconds
Time elapsed for predictions: 0.06201182911172509 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using GPU for function evaluation.
Time to train: 7.00 seconds
Time elapsed for predictions: 0.11392427701503038 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using GPU for function evaluation.
Time to train: 10.11 seconds
Time elapsed for predictions: 0.26153256092220545 seconds
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using GPU for function evaluation.
Time to train: 14.83 seconds
Time elapsed for predictions: 0.4859354440122843 seconds


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



real    0m49.330s
user    0m46.656s
sys     0m1.000s
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

# 2D (SHCF) Test: n = 1,000,000,000
## Version 5 (Complete CuPy) (Time: 
```sh
```