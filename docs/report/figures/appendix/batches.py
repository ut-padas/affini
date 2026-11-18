"""The appendix section on where a batch ends: AFINNI's grid rule against ART's packing rule, on one real queue.

Usage, from the repository root:

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/report/figures/appendix/batches.py

The queue is the one the companion's queue-layout figure shows (the running example's AFINNI
level and batch size of ``make_figures.queue_layout``), cut by both plans' own ``batch_end``; a
second frame cuts the queue of ``tests/unit/test_plans.py``, whose oversized cluster re-anchors
AFINNI's grid. The script restates both rules as the formulas the appendix prints, and stops
unless they give the library's boundaries on these queues and on many random ones, and unless
every batch obeys the size bounds the appendix states. Writes apx-batches-*.svg and
apx-batches.json.
"""

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_figures as mf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.afinni.plan import AFINNIPlan  # noqa: E402
from src.art.plan import ARTPlan  # noqa: E402
from src.config import TrainingConfig  # noqa: E402
from src.engine import driver  # noqa: E402

TEST_QUEUE = [30, 30, 200, 10, 10]  # the queue and batch size of test_plans.py's AFINNI grid test
TEST_BATCH = 50
RANDOM_QUEUES = 2000  # random queues (and four batch sizes each) the restated rules are checked on


def library_ends(plan_class, counts: np.ndarray, batch_size: int) -> list[int]:
    """The batch ends the plan's own batch_end gives, batch after batch, as refine_rank calls it."""
    cutter = object.__new__(plan_class)  # batch_end needs no other member (AFINNI keeps its grid offset)
    cumsum = np.cumsum(counts)
    ends, start, index = [], 0, 0
    while start < len(cumsum):
        end = cutter.batch_end(cumsum, start, index, batch_size)
        ends.append(end)
        start, index = end, index + 1
    return ends


def afinni_rule(counts, B: int) -> tuple[list[int], list[int]]:
    """
    e_b = min(K, min{k : S_k >= (b + 1) B + o}); if that is not past e_(b-1), the next cluster
    goes alone and o becomes S_(e_(b-1)+1) - (b + 1) B. Returns the ends and the grid line of each batch.
    """
    S = np.r_[0, np.cumsum(counts)]
    K, ends, lines, o, prev = len(counts), [], [], 0, 0
    for b in range(K):
        if prev == K:
            break
        line = (b + 1) * B + o
        e = next((k for k in range(1, K + 1) if S[k] >= line), K)
        if e <= prev:
            o = int(S[prev + 1]) - (b + 1) * B
            line, e = (b + 1) * B + o, prev + 1
        ends.append(e)
        lines.append(line)
        prev = e
    return ends, lines


def art_rule(counts, B: int) -> list[int]:
    """e_b = min(K, max(e_(b-1) + 1, max{k : S_k <= S_(e_(b-1)) + B}))."""
    S = np.r_[0, np.cumsum(counts)]
    K, ends, prev = len(counts), [], 0
    while prev < K:
        fit = max(k for k in range(K + 1) if S[k] <= S[prev] + B)
        prev = min(K, max(prev + 1, fit))
        ends.append(prev)
    return ends


def check(counts, B: int) -> None:
    """The restated rules give the library's ends, and every batch obeys the bounds the appendix states."""
    counts = np.asarray(counts, dtype=np.int64)
    a_ends, _ = afinni_rule(counts, B)
    if a_ends != library_ends(AFINNIPlan, counts, B) or art_rule(counts, B) != library_ends(ARTPlan, counts, B):
        raise SystemExit(f"The restated batch rules differ from the library's on {counts.tolist()} (B = {B}).")
    for ends, bound in ((a_ends, lambda pts, last: pts < B + last), (art_rule(counts, B), lambda pts, last: pts <= B)):
        lo = 0
        for e in ends:
            pts, last = int(counts[lo:e].sum()), int(counts[e - 1])
            if e - lo > 1 and not bound(pts, last):
                raise SystemExit(f"A batch of {e - lo} clusters and {pts} points breaks its bound (B = {B}).")
            lo = e


def strip(ax, y: float, counts, colours, B: int, label: str, show_grid: bool) -> dict:
    """One queue: clusters back to back, AFINNI's batches (and grid) above, ART's below."""
    S = np.r_[0, np.cumsum(counts)]
    h = 0.42
    for k, n in enumerate(counts):
        ax.add_patch(plt.Rectangle((S[k], y - h / 2), n, h, fc=colours[k], ec="white", lw=0.8, zorder=2))
        ax.text(S[k] + n / 2, y, str(int(n)), ha="center", va="center", fontsize=5.4, zorder=3,
                color="white" if colours[k] not in (mf.LIGHT,) else mf.DARK)  # fmt: skip
    a_ends, lines = afinni_rule(counts, B)
    art_ends = art_rule(counts, B)
    if show_grid:  # the fixed multiples of B, faint, and the line each batch was cut at, blue
        for g in range(1, int(S[-1] // B) + 1):
            ax.plot([g * B] * 2, [y + h / 2, y + h / 2 + 0.5], color=mf.LIGHT, lw=0.7, ls=":", zorder=1)
    for b, line in enumerate(lines):
        if line <= S[-1]:
            ax.plot([line] * 2, [y - h / 2, y + h / 2 + 0.5], color=mf.BLUE, lw=0.8, ls=":", zorder=4)
            ax.plot(line, y + h / 2 + 0.03, marker="v", ms=3.0, color=mf.BLUE, zorder=5)
        if line - (b + 1) * B != (lines[b - 1] - b * B if b else 0):  # the grid was re-anchored for this batch
            ax.text(line, y + h / 2 + 0.62, f"line {(b + 1) * B} is behind: re-anchored to {line}", ha="right",
                    va="bottom", fontsize=5.4, color=mf.BLUE)  # fmt: skip
    for ends, yy, colour, va in ((a_ends, y + h / 2 + 0.3, mf.BLUE, "bottom"), (art_ends, y - h / 2 - 0.3, mf.DARK, "top")):
        lo = 0
        for e in ends:
            a, z = S[lo], S[e]
            ax.annotate("", (a + 0.6, yy), (z - 0.6, yy),
                        arrowprops={"arrowstyle": "|-|", "color": colour, "lw": 0.8, "mutation_scale": 2.3})  # fmt: skip
            ax.text((a + z) / 2, yy + (0.06 if va == "bottom" else -0.06), f"{int(z - a)}", ha="center", va=va,
                    fontsize=5.4, color=colour)  # fmt: skip
            lo = e
    ax.text(-0.012 * S[-1], y + h / 2 + 0.3, "AFINNI", ha="right", va="center", fontsize=6, color=mf.BLUE)
    ax.text(-0.012 * S[-1], y - h / 2 - 0.3, "ART", ha="right", va="center", fontsize=6, color=mf.DARK)
    ax.text(-0.012 * S[-1], y, label, ha="right", va="center", fontsize=6, color=mf.DARK)
    return {"afinni": a_ends, "art": art_ends, "lines": [int(v) for v in lines]}


def board(name: str, counts, colours, B: int, label: str, show_grid: bool) -> dict:
    total = int(np.sum(counts))
    fig, ax = plt.subplots(figsize=(6.2, 1.15))
    out = strip(ax, 0.0, counts, colours, B, label, show_grid)
    ax.set_xlim(-0.12 * total, total * 1.01)
    ax.set_ylim(-0.95, 1.2)
    ax.axis("off")
    mf.save(fig, name)
    return out


def sizes(counts, ends) -> list[int]:
    S = np.r_[0, np.cumsum(counts)]
    return [int(S[e] - S[a]) for a, e in zip([0, *ends[:-1]], ends, strict=True)]


def main() -> None:
    afinni, levels, _ = mf.record_afinni()
    nodes = mf.afinni_forest(levels)
    defaults = inspect.signature(mf.queue_layout).parameters
    level, B = defaults["level"].default, defaults["batch_size"].default
    here = [n for n in nodes if n.level == level]
    counts = np.array([len(n.X) for n in here], dtype=np.int64)
    layout = json.loads((mf.OUT / "example.json").read_text())["afinni"]["layout"]
    if int(counts.sum()) != layout["points"] or layout["batch_size"] != B:
        raise SystemExit("This level is not the one the queue-layout figure shows; rerun make report-figures.")

    rng = np.random.default_rng(mf.SEED)
    for counts_ in [counts, TEST_QUEUE, *(rng.integers(1, 300, rng.integers(1, 40)) for _ in range(RANDOM_QUEUES))]:
        for B_ in (TEST_BATCH, B, 97, 250):
            check(counts_, B_)

    example = board("apx-batches-example", counts, [n.colour for n in here], B, f"level {level}", False)
    test = board("apx-batches-anchor", np.array(TEST_QUEUE), [mf.LIGHT, mf.GREY] * 3, TEST_BATCH, "test queue", True)

    # The cap the driver puts on batch_size for the running example's AFINNI plan on this (CPU) backend:
    # once for the default batch size, once for one far above every cap, which shows the upper clip.
    afinni_plan = AFINNIPlan(TrainingConfig(atol=1.0), 2, 1, 0)
    capped, huge = TrainingConfig(atol=1.0), TrainingConfig(atol=1.0, batch_size=10**12)
    driver._cap_batch_size(capped, afinni_plan, 2, 1)
    driver._cap_batch_size(huge, afinni_plan, 2, 1)
    numbers = {
        "level": level, "batch_size": B, "counts": counts.tolist(), "points": int(counts.sum()),
        "example": example | {"afinni_sizes": sizes(counts, example["afinni"]), "art_sizes": sizes(counts, example["art"])},
        "test": test | {"counts": TEST_QUEUE, "batch_size": TEST_BATCH, "afinni_sizes": sizes(TEST_QUEUE, test["afinni"]),
                        "art_sizes": sizes(TEST_QUEUE, test["art"])},
        # An oversized cluster that a batch reaches before its line joins that batch.
        "joined": {"counts": [TEST_QUEUE[0], TEST_QUEUE[2]], "afinni_sizes": sizes(
            [TEST_QUEUE[0], TEST_QUEUE[2]], library_ends(AFINNIPlan, np.array([TEST_QUEUE[0], TEST_QUEUE[2]]), TEST_BATCH))},
        "random_checked": RANDOM_QUEUES,
        "floor": {"afinni": AFINNIPlan.batch_floor, "art": ARTPlan.batch_floor},
        "default_batch_size": TrainingConfig(atol=1.0).batch_size,
        "afinni_bytes_per_point": afinni_plan.bytes_per_point(), "afinni_cpu_batch_size": capped.batch_size,
        "cap_max": huge.batch_size,
    }  # fmt: skip
    # The large numbers as the text prints them, with thousands separators.
    numbers["text"] = {k: f"{v:,}" for k, v in (("floor_afinni", AFINNIPlan.batch_floor), ("floor_art", ARTPlan.batch_floor),
                                                 ("cap_max", huge.batch_size), ("default", numbers["default_batch_size"]),
                                                 ("cpu", capped.batch_size))}  # fmt: skip
    (mf.OUT / "apx-batches.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps(numbers))


if __name__ == "__main__":
    main()
