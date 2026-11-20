"""Draw the panel of the companion's appendix on deterministic segmented sums (apx-sums-*.svg, apx-sums.json).

Usage, from the repository root:

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/report/figures/appendix/sums.py

The GPU kernels cannot run here, so the script restates them in NumPy. One cluster of the running
example's first AFINNI level is placed after m rows of other data, and its sum is taken three ways:

- as ``segment_cross`` takes it on the GPU (``segment_cross_partial`` then ``segment_cross_reduce``):
  chunks of ``SEGMENT_CHUNK`` rows counted from the cluster's first row, each summed in row order,
  then the chunks in order;
- as CuPy's ``add.reduceat`` takes it: a cumulative sum of the whole batch, then the difference of
  two of its entries (``_add_reduceat`` in CuPy's ``_routines_indexing.pyx``). The GPU scan's own
  order cannot be reproduced here, so the script takes NumPy's sequential scan and, as the best any
  scan can do, the correctly rounded prefixes (``math.fsum``);
- as the library takes it on the CPU (``reduce_sum_1d``, NumPy's ``add.reduceat``).

The restated kernel is checked against the library: its chunk boundaries against the layout of
``segment_layout`` and the chunk formula of the companion's lemma, and its two passes against two
calls of the library's CPU ``segment_sum``, whose CSR product adds each group's rows in input order
starting from zero, exactly as one kernel thread does. The script stops if any of them differs.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_figures as mf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src import IS_CUPY_BACKEND  # noqa: E402
from src.numerics import count_groups, grouped_cross, reduce_sum_1d, segment_layout  # noqa: E402
from src.numerics.linalg import SPD_COOP_MIN_P  # noqa: E402
from src.numerics.segments import (  # noqa: E402
    SEGMENT_CHUNK,
    SEGMENT_TILED2D_MIN,
    SEGMENT_TILED_MIN_E,
    segment_sum,
)

CH = SEGMENT_CHUNK
U = 2.0**-53  # unit roundoff of float64; eps = 2 U
PREFIX_EXPONENTS = range(4, 23)  # m = 2^4 ... 2^22 rows before the cluster
# CuPy's CUB_REDUCE_SIZE_THRESHOLD (cupy/_core/_cub_reduction.pyx): shorter axis reductions take its
# generic kernel. Quoted by the text; CuPy is not installed on the CPU, so it is recorded here.
CUPY_CUB_MIN_LENGTH = 128


def kernel_chunks(starts: np.ndarray, counts: np.ndarray) -> list[tuple[int, int, int]]:
    """``(segment, first row, end row)`` of every chunk as ``segment_cross_partial`` maps its threads, from ``segment_layout``."""
    st, ct, cum = (np.asarray(v) for v in segment_layout(starts, counts))
    out = []
    for j in range(int(cum[-1])):
        k = int(np.searchsorted(cum, j, side="right"))  # the kernel's binary search: first k with cum[k] > j
        first = int(cum[k]) - (int(ct[k]) + CH - 1) // CH
        r0 = int(st[k]) + (j - first) * CH
        out.append((k, r0, min(r0 + CH, int(st[k] + ct[k]))))
    return out


def lemma_chunks(starts: np.ndarray, counts: np.ndarray) -> list[tuple[int, int, int]]:
    """chunk_j(k) = [s_k + CH j, min(s_k + CH (j + 1), s_k + n_k)), as the companion's lemma states it."""
    return [(k, int(s) + CH * j, min(int(s) + CH * (j + 1), int(s + n)))
            for k, (s, n) in enumerate(zip(starts, counts, strict=True)) for j in range(-(-int(n) // CH))]  # fmt: skip


def sequential_rows(M: np.ndarray) -> np.ndarray:
    """Row-order sums of each row of M (zero-padded), one addition per column, from 0.0: a kernel thread's loop.

    Adding a padding 0.0 leaves every partial sum's bits unchanged (only -0.0 would become +0.0).
    """
    s = np.zeros(M.shape[0])
    for c in range(M.shape[1]):
        s = s + M[:, c]
    return s


def chunked_sums(a: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The two passes of ``segment_cross`` on the GPU for a plain sum (B = None, no offsets or weights), restated."""
    chunks = kernel_chunks(starts, counts)
    tile = np.zeros((len(chunks), CH))
    for j, (_, r0, r1) in enumerate(chunks):
        tile[j, : r1 - r0] = a[r0:r1]
    partial = sequential_rows(tile)
    owner = np.array([k for k, _, _ in chunks])
    per = np.bincount(owner, minlength=len(counts))
    grid = np.zeros((len(counts), max(1, int(per.max()))))
    for k in range(len(counts)):
        grid[k, : per[k]] = partial[owner == k]
    return sequential_rows(grid), partial, owner


def library_two_pass(a: np.ndarray, starts: np.ndarray, counts: np.ndarray, chunks) -> np.ndarray:
    """The same two passes from the library's CPU ``segment_sum`` (rows of a group added in order from zero)."""
    chunk_of_row = np.empty(len(a), dtype=np.int64)
    for j, (_, r0, r1) in enumerate(chunks):
        chunk_of_row[r0:r1] = j
    partial = segment_sum(a, chunk_of_row, len(chunks))
    return segment_sum(partial, np.array([k for k, _, _ in chunks]), len(counts))


def python_sequential(values) -> float:
    s = 0.0
    for v in values:
        s += float(v)
    return s


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    if IS_CUPY_BACKEND:
        raise SystemExit("Run on the CPU: the script restates the GPU kernels in NumPy.")
    s, levels, _ = mf.record_afinni()
    nodes = mf.afinni_forest(levels)
    level0 = [n for n in nodes if n.level == 0]
    which = int(np.argmax([len(n.X) for n in level0]))
    cluster = level0[which]
    values = mf.f_np(cluster.X)  # the cluster's f, from which the affine fit takes its mean (reduce_sum_1d)
    n = len(values)
    exact = math.fsum(values)

    # 1. The kernel's chunk mapping is the lemma's formula, on the real level-0 layout, on the whole
    #    level as one segment, and on a layout with long and empty segments.
    lv0 = levels[0]
    layouts = [(np.r_[0, np.cumsum(lv0.counts)[:-1]], lv0.counts), (np.array([0]), np.array([lv0.counts.sum()])),
               (np.array([0, 1000, 1000, 1513]), np.array([1000, 0, 513, 257]))]  # fmt: skip
    for st, ct in layouts:
        nz = ct > 0
        got = kernel_chunks(st[nz], ct[nz])
        check(got == lemma_chunks(st[nz], ct[nz]), "The kernel's chunk mapping is not the lemma's formula.")
        a = mf.f_np(np.random.default_rng(int(ct.sum())).uniform(mf.LOWER, mf.UPPER, (int((st + ct).max()), 2)))
        mine, _, _ = chunked_sums(a, st[nz], ct[nz])
        check(np.array_equal(mine, library_two_pass(a, st[nz], ct[nz], got)),
              "The restated kernel differs from the library's two-pass segment_sum.")  # fmt: skip

    # 2. The cluster after m rows of f at uniform points of the domain (other clusters' rows).
    rng = np.random.default_rng(mf.SEED)
    before = mf.f_np(rng.uniform(mf.LOWER, mf.UPPER, (2 ** max(PREFIX_EXPONENTS), 2)))
    ms = [0] + [2**e for e in PREFIX_EXPONENTS]
    rows = {"chunked": [], "cpu": [], "scan": [], "rounded": [], "bound": []}
    chunked_bits, cpu_bits = set(), set()
    for m in ms:
        batch = np.concatenate([before[:m], values])
        st, ct = (np.array([0, m]), np.array([m, n])) if m else (np.array([0]), np.array([n]))
        k = len(st) - 1
        chunks = kernel_chunks(st, ct)
        gpu, _, _ = chunked_sums(batch, st, ct)
        check(np.array_equal(gpu, library_two_pass(batch, st, ct, chunks)),
              "The restated kernel differs from the library's two-pass segment_sum.")  # fmt: skip
        check(gpu[k] == python_sequential(values) if n <= CH else True, "A one-chunk sum is not the row-order sum.")
        cpu = float(reduce_sum_1d(batch, st)[k])
        acc = np.concatenate([[0.0], np.cumsum(batch)])  # CuPy's _add_reduceat: zero, then the inclusive scan
        scan = acc[m + n] - acc[m]
        rounded = math.fsum(batch) - math.fsum(batch[:m])
        bound = 2 * U * math.fsum(np.abs(batch)) + U * abs(exact)
        check(abs(rounded - exact) <= bound, "A correctly rounded prefix difference exceeds the lemma's bound.")
        chunked_bits.add(float(gpu[k]).hex())
        cpu_bits.add(cpu.hex())
        for key, v in (("chunked", gpu[k]), ("cpu", cpu), ("scan", scan), ("rounded", rounded)):
            rows[key].append(abs(float(v) - exact) / abs(exact))
        rows["bound"].append(bound / abs(exact))
    check(len(chunked_bits) == 1, "The chunked sum changed with the rows before the cluster.")
    check(len(cpu_bits) == 1, "The CPU reduce_sum_1d changed with the rows before the cluster.")

    # 3. The CPU's grouped products (count_groups, grouped_cross): one BLAS product per block, so a
    #    block's bits do not change when other blocks, of its size or another, leave the batch.
    #    Level 2 of the example has several blocks of each size; its affine features are [1, x - c].
    lv2 = levels[2]
    feats = np.concatenate([np.ones((len(lv2.X), 1)), lv2.X - np.repeat(lv2.centroids, lv2.counts, axis=0)], axis=1)
    st2 = np.r_[0, np.cumsum(lv2.counts)[:-1]]
    groups = count_groups(lv2.counts, st2)
    whole = grouped_cross(groups, *([[feats[r] for _, r in groups]] * 2))
    for keep in ([0], [0, 1, 2], [1, len(lv2.counts) - 1], list(range(len(lv2.counts)))[::-1]):
        sub = np.concatenate([feats[st2[k] : st2[k] + lv2.counts[k]] for k in keep])
        cs = lv2.counts[keep]
        g = count_groups(cs, np.r_[0, np.cumsum(cs)[:-1]])
        part = grouped_cross(g, *([[sub[r] for _, r in g]] * 2))
        check(np.array_equal(part, whole[keep]), "grouped_cross changed a block with the other blocks of its batch.")

    # 4. NumPy's add.reduceat, the CPU path: the segment's first row, plus NumPy's pairwise sum of the
    #    rest (not a row-order sum); either way only the segment's own rows enter.
    probe = mf.f_np(np.random.default_rng(7).uniform(mf.LOWER, mf.UPPER, (3000, 2))) * 10 ** np.linspace(-6, 6, 3000)
    cuts = np.array([0, 700, 1900])
    red = np.add.reduceat(probe, cuts)
    ends = np.r_[cuts[1:], len(probe)]
    check(all(red[i] == probe[c] + np.add.reduce(probe[c + 1 : e]) for i, (c, e) in enumerate(zip(cuts, ends, strict=True))),
          "NumPy's add.reduceat is not first row + pairwise rest on this NumPy.")  # fmt: skip
    reduceat_sequential = bool(all(red[i] == python_sequential(probe[c:e]) for i, (c, e) in enumerate(zip(cuts, ends, strict=True))))

    # 5. The panel: relative error of the cluster's sum against the rows before it.
    floor = 1e-18  # an exact sum (zero error) is drawn here, on the tick labelled 0
    x = np.array(ms[1:], dtype=float)
    fig, ax = plt.subplots(figsize=(4.4, 2.35))
    ax.loglog(x, np.array(rows["bound"][1:]), color=mf.LIGHT, lw=1.4, ls="--", label=r"bound $\varepsilon\,\sum_{i<s+n}|a_i|$")
    ax.loglog(x, np.maximum(rows["scan"][1:], floor), color=mf.GREY, lw=0.9, marker="o", ms=3.2,
              label="prefix difference, row-order scan")  # fmt: skip
    ax.loglog(x, np.maximum(rows["rounded"][1:], floor), color=mf.GREY, lw=0.9, ls=":", marker="o", ms=3.2, mfc="white",
              label="prefix difference, correctly rounded prefixes")  # fmt: skip
    ax.loglog(x, np.maximum(rows["chunked"][1:], floor), color=mf.DARK, lw=1.6, label="chunked (segment_cross)")
    ax.axhline(U, color=mf.LIGHT, lw=0.6)
    ax.text(x[-1], U * 0.75, "unit roundoff $u$", fontsize=6, color=mf.GREY, va="top", ha="right")
    ax.set_xlabel("rows $m$ before the cluster in the batch")
    ax.set_ylabel("relative error of the cluster's sum")
    ax.set_xscale("log", base=2)
    ax.set_ylim(floor / 2, max(rows["scan"]) * 4)
    ticks = [floor] + [10.0**e for e in range(-17, int(np.ceil(np.log10(max(rows["scan"])))) + 1, 2)]
    ax.set_yticks(ticks, ["0"] + [f"$10^{{{int(round(np.log10(t)))}}}$" for t in ticks[1:]])
    ax.minorticks_off()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    mf.save(fig, "apx-sums-prefix")

    last = len(ms) - 1
    out = {
        "chunk": CH, "tiled_min_e": SEGMENT_TILED_MIN_E, "tiled2d_min": SEGMENT_TILED2D_MIN, "coop_min_p": SPD_COOP_MIN_P, "cub_min_length": CUPY_CUB_MIN_LENGTH,
        "cluster": {"level": 0, "index": which, "points": n, "colour": cluster.colour, "sum": round(exact, 3),
                    "chunks": -(-n // CH), "level_counts": [int(c) for c in lv0.counts]},
        "m_max": ms[-1],
        "m_max_log2": max(PREFIX_EXPONENTS),
        "chunked_rel": f"{rows['chunked'][0]:.1e}",
        "cpu_rel": f"{rows['cpu'][0]:.1e}" if rows["cpu"][0] else "0",
        "scan_rel_max": f"{rows['scan'][last]:.1e}",
        "rounded_rel_max": f"{rows['rounded'][last]:.1e}",
        "bound_rel_max": f"{rows['bound'][last]:.1e}",
        "scan_over_chunked": f"{rows['scan'][last] / rows['chunked'][0]:.0f}" if rows["chunked"][0] else "inf",
        "same_bits_chunked": len(chunked_bits) == 1,
        "same_bits_cpu": len(cpu_bits) == 1,
        "reduceat_sequential": reduceat_sequential,
        "numpy": np.__version__,
        "unit_roundoff": f"{U:.2e}",
    }
    (mf.OUT / "apx-sums.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
