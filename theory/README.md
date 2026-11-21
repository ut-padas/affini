# theory/

Candidate theorems for a surrogate scheme for the step maps of stiff ODEs, each checked numerically against its
predictions and against the benchmark's records.

| File | Contents |
|---|---|
| [MEMO.md](MEMO.md) | The theory memo: the observations to explain (section 0), seven candidate theorems T1–T7 with assumptions, proofs, predictions and checks, a synthesis and a ranking of foundations for a scheme |
| [LITERATURE.md](LITERATURE.md) | Prior art per theorem (IDs `TH-*`): what is known, what would be new |
| [REVIEWS.md](REVIEWS.md) | The external reviews of the memo, round by round, and how each point was resolved |
| `checks/t*.py` | One numerical check per theorem (NumPy and SciPy only, each under two minutes); `checks/data/t*.json` holds their output, the only source of the memo's numbers |

Run a check from this folder's `checks/` directory with the repository's environment:

```bash
cd theory/checks && ../../.venv/bin/python t1_scalar.py
```

The checks are independent of the library (`src/`), so they run on any machine with NumPy and SciPy, and their
results do not change when the library does.
