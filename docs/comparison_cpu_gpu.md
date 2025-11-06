# 2D (SHCF) Test: n = 100,000,000
## CPU Version (Time: 2m 5s)
```sh
c608-091[gh](586)$ time uv run python shcf.py 
CUML available: True
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Using fast two-means length threshold (dim=2): 8192
Time to train: 122.13 seconds
Time elapsed for predictions: 0.03542057110462338 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                100000000 & 0.1 & 2.0\% & 1924 & 1.4\% & 98.59\% & 0.01\% & 1.9581 & 0.2246 & 0.7155 & 0.0206 & 0.0202 & 0.1423 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    2m5.591s
user    1m57.680s
sys     0m6.177s
```

## GPU Version (Time: 57s)
```sh
c608-091[gh](585)$ time uv run python shcf_gpu.py 
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 42.84 seconds
Time to train: 53.71 seconds
Time elapsed for predictions: 0.04006835096515715 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                100000000 & 0.1 & 2.0\% & 2156 & 1.4\% & 98.57\% & 0.04\% & 0.7473 & 0.2623 & 0.3076 & 0.0343 & 0.0235 & 0.0247 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    0m57.060s
user    0m47.523s
sys     0m8.044s
```

# 2D (SHCF) Test: n = 1,000,000,000
## GPU Version (Time: 4m 8s)
```sh
c608-061[gh](564)$ time uv run python shcf_gpu.py 
Max achieved at (-3.0, -2.0), value = 591.2024557808674
Time to train: 123.10 seconds
Time to train: 245.17 seconds
Time elapsed for predictions: 0.04081753198988736 seconds


\begin{table}[h!]
    \centering
    \begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
        \hline
        $N_{init}$ & atol & rtol & $N_{sources}$ & $p_{out}$ & $p_{in,good}$ & $p_{in,bad}$ & $\|e\|_\infty$ & $e_{med}$ & $\|e\|_\infty / \|f\|_\infty$ & $\|e\|_2 / \|f\|_2$\\
        \hline\hline
                1000000000 & 0.1 & 2.0\% & 2191 & 1.14\% & 98.85\% & 0.01\% & 1.9874 & 0.2542 & 0.9148 & 0.0213 & 0.0208 & 0.0208 \\
        \hline
    \end{tabular}
    \caption{This is a caption!}
    \label{tab:new_tab}
\end{table}
None



real    4m8.910s
user    3m28.272s
sys     0m37.991s
```