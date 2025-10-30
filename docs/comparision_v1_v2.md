# 2D (SHCF) Test: n = 4000
## Version 1
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

## Version 2 (Accelerated)
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