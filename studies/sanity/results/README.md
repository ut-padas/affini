# Sanity-check records

The records behind the technical companion's Appendix B (the rewritten code against the earlier
draft's numbers). Each folder holds the numbers a GH200 node recorded with
`studies/sanity/paper/run_manuscript.sh`, as printed there: `manuscript_results.csv` (table, row,
quantity, seed-0 value), `cost.csv` (case wall times) and `provenance.json` (commit, date, machine
label). The machine's full record (`RESULTS.md`, libraries, run settings) stayed on that machine.

| Record | Commit | Holds |
|---|---|---|
| `2026-11-17_b4bac64_gh200-node-96-gb-hbm` | `b4bac64` | every group but mpi, taken before Table 2 d = 80 and Table 3 #3-4 finished |
| `2026-11-17_d2537a9_gh200-node-96-gb-hbm` | `d2537a9` | Table 2 with the paper-era cluster sizes |
| `2026-11-17_ffbd39e_gh200-node-96-gb-hbm` | `ffbd39e` | Figure 8's exact-gradient runs with 50 candidates |

Rebuild the appendix's tables from them, the main record first:

    make report-tables RECORD="studies/sanity/results/2026-11-17_b4bac64_gh200-node-96-gb-hbm \
      studies/sanity/results/2026-11-17_d2537a9_gh200-node-96-gb-hbm \
      studies/sanity/results/2026-11-17_ffbd39e_gh200-node-96-gb-hbm" && make report
