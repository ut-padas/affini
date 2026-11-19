"""The appendix section on why work stealing cannot deadlock or retire early: a message-sequence chart of one recorded run.

Record the run (6 ranks, one root, the companion's steal example), from the repository root:

    SURROGATE_DEVICE=cpu SURROGATE_MPI=on OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PATH=$PWD/.venv/bin:$PATH \
      PYTHONPATH=. .venv/bin/mpiexec -n 6 .venv/bin/python docs/reports/adaptive-surrogates/figures/appendix/steal.py --record </dev/null

It trains the example once as a warm-up, then once without and once with work stealing, stops
unless both build the same tree, and stores every protocol event of the stealing run in
apx-steal-events.json. Steal timing differs between runs, so the chart is of that one recording.
Then draw the chart and write apx-steal.json (no MPI):

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/reports/adaptive-surrogates/figures/appendix/steal.py

Drawing first checks the protocol's properties on the recording itself: every request has exactly
one reply, a rank never has two requests out, every grant is received by the thief that asked,
a rank that asks holds no work, and no rank retires before the last grant has arrived.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_figures as mf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

EVENTS = mf.OUT / "apx-steal-events.json"
# The protocol's events (src/engine/work_stealing.py), the level and run markers that place them,
# and the idle intervals; batch phases are left out, they would only fill the file.
KINDS = ("run_start", "level_start", "level_end", "batch", "load", "steal_request", "steal_grant", "steal_deny",
         "steal_received", "steal_denied", "retire_start", "retired", "done")  # fmt: skip
FIELDS = ("kind", "t", "t_last", "count", "rank", "level", "batch", "victim", "thief", "reason", "n_pts",
          "n_clusters", "mid_level", "dur", "waited", "phase", "clusters_level_end")  # fmt: skip


def train(stealing: bool) -> "mf.AdaptiveSurrogate":
    """The companion's single-root steal example (make_figures.STEAL), on every rank."""
    s = mf.AdaptiveSurrogate()
    st = mf.STEAL
    s.train(mf.art_initial_points(), mf.art_config(
        atol=st["atol"], max_levels=st["max_levels"], batch_size=st["batch_size"], num_initial_clusters=1,
        use_work_stealing=stealing, profile_events=True,
        art=mf.ARTSettings(sample_multiplier=mf.ART_MULTIPLIER, steal_min_load=st["steal_min_load"],
                           steal_size=st["steal_size"], roots="box")),
            mf.Evaluator(f_fn=mf.f_xp))  # fmt: skip
    return s


def tree_of(model) -> tuple[dict, list]:
    """Every split plane by (root, locational code), and the sorted leaves: the tree, whatever the row order."""
    a = {k: mf.to_host(v) for k, v in model.routing.arrays().items()}
    planes = {(int(r), int(c)): (w.tobytes(), float(z)) for r, c, w, z in
              zip(a["split_root"], a["split_locational_code"], a["normal"], a["offset"], strict=True)}  # fmt: skip
    return planes, sorted(zip(a["leaf_root"].tolist(), a["leaf_locational_code"].tolist(), strict=True))


def record() -> None:
    train(True)  # warm-up: compiles every kernel on every rank
    static, stealing = train(False), train(True)
    if mf.get_mpi_rank() != 0:
        return
    ranks = len(stealing.profiling_stats["per_rank"])
    if ranks != mf.STEAL["ranks"]:
        raise SystemExit(f"Run under mpiexec -n {mf.STEAL['ranks']}.")
    if tree_of(static) != tree_of(stealing):
        raise SystemExit("The tree built with work stealing differs from the one built without it.")
    keep = [[{k: (round(v, 5) if isinstance(v, float) else v) for k, v in e.items() if k in FIELDS}
             for e in r["events"] if e["kind"] in KINDS or e.get("phase") == "time_idle"]
            for r in stealing.profiling_stats["per_rank"]]  # fmt: skip
    EVENTS.write_text(json.dumps({"settings": mf.STEAL, "leaves": int(len(tree_of(stealing)[1])),
                                  "leaves_per_rank": [int(r["num_final_clusters"]) for r in stealing.profiling_stats["per_rank"]],
                                  "events": keep}) + "\n")  # fmt: skip
    print("recorded", sum(len(e) for e in keep), "events; the tree equals the one built without stealing")


# ----------------------------------------------------------------------------- the recording, checked
def asks_of(events: list[list[dict]]) -> list[tuple[int, dict, dict]]:
    """Every (thief, request, reply) of the run; stops if a rank ever has two requests out or a reply comes from another rank."""
    asks = []
    for r, ev in enumerate(events):
        out = None
        for e in ev:
            if e["kind"] == "steal_request":
                if out is not None:
                    raise SystemExit(f"Rank {r} sent a second request before its first was answered.")
                out = e
            elif e["kind"] in ("steal_received", "steal_denied"):
                if out is None or out["victim"] != e["victim"]:
                    raise SystemExit(f"Rank {r} got a reply from rank {e['victim']} it did not ask.")
                asks.append((r, out, e))
                out = None
        if out is not None:
            raise SystemExit(f"Rank {r} finished with a request unanswered.")
    return asks


def busy_intervals(ev: list[dict]) -> list[tuple[float, float]]:
    """When a rank held work: from a level's start (or the grant that fed it) to the level end that left its queue empty."""
    spans, start, last_received = [], None, None
    for e in ev:
        if e["kind"] == "steal_received":
            last_received = e["t"]
        elif e["kind"] == "level_start" and start is None:
            start = last_received if last_received is not None else e["t"]
        elif e["kind"] == "level_end" and e.get("clusters_level_end") == 0 and start is not None:
            spans.append((start, e["t"]))
            start, last_received = None, None
    if start is not None:
        raise SystemExit("A rank still held work when the run ended.")
    return spans


def inside(t: float, spans) -> bool:
    return any(a <= t <= b for a, b in spans)


def check_protocol(events: list[list[dict]], asks) -> dict:
    """The properties the appendix proves, checked on the recording; returns the counts the text quotes."""
    expand = lambda e: [e] * int(e.get("count", 1))  # noqa: E731 - coalesced repeats of one denial
    victim_side = [(e["rank"], e["thief"], e["kind"], e.get("reason"), e.get("n_clusters"))
                   for ev in events for e in ev if e["kind"] in ("steal_grant", "steal_deny") for e in expand(e)]  # fmt: skip
    thief_side = [(e["victim"], r, "steal_grant" if e["kind"] == "steal_received" else "steal_deny", e.get("reason"),
                   e.get("n_clusters")) for r, _, e in asks if e.get("reason") != "IDLE"]  # fmt: skip
    if sorted(victim_side, key=str) != sorted(thief_side, key=str):
        raise SystemExit("A grant or denial a victim sent is not the reply its thief received.")
    busy = [busy_intervals(ev) for ev in events]
    for r, ev in enumerate(events):
        for e in ev:
            if e["kind"] in ("steal_request", "retire_start") and inside(e["t"], busy[r]):
                raise SystemExit(f"Rank {r} asked for work or joined the barrier while holding some.")
            if e["kind"] in ("steal_grant", "steal_deny") and not inside(e["t"], busy[r]):
                raise SystemExit(f"Rank {r} answered as a victim while idle.")
    one = lambda r, k: [e["t"] for e in events[r] if e["kind"] == k]  # noqa: E731
    joined, retired = [one(r, "retire_start") for r in range(len(events))], [one(r, "retired") for r in range(len(events))]
    if any(len(j) != 1 or len(x) != 1 for j, x in zip(joined, retired, strict=True)):
        raise SystemExit("Every rank must join the barrier once and retire once.")
    last_join, first_retired = max(j[0] for j in joined), min(x[0] for x in retired)
    last_reply = max(e["t"] for _, _, e in asks)
    if not last_reply < first_retired or not last_join <= first_retired:
        raise SystemExit("A rank retired before every rank had joined the barrier or before the last reply.")
    for r, ev in enumerate(events):
        if any(e["kind"] in ("steal_request", "steal_grant") and e["t"] > joined[r][0] for e in ev):
            raise SystemExit(f"Rank {r} asked for or gave work after joining the barrier.")
    reasons = {}
    for _, _, e in asks:
        key = "grant" if e["kind"] == "steal_received" else e["reason"].split(" ")[0].lower()
        reasons[key] = reasons.get(key, 0) + 1
    return {"requests": len(asks), "replies": reasons, "busy": busy}


# ----------------------------------------------------------------------------- the chart
SHORT = {"MID_LEVEL_LOAD": "too little left", "LOAD": "too little left", "MID_LEVEL_NO_CLUSTERS": "one node left",
         "STEAL_SIZE": "one node left", "LEASED": "leased", "IDLE": "idle"}  # fmt: skip
MERGE_S = 0.0015  # events of one kind on different ranks this close share a row


def rows_of(window: list[dict]) -> dict[int, float]:
    """One row per event, in time order, except that same-kind events on different ranks close in time share one."""
    row, rows, first, kind, ranks = -1, {}, None, None, set()
    for e in window:
        if first is None or e["kind"] != kind or e["t"] - first > MERGE_S or e["rank"] in ranks:
            row, first, kind, ranks = row + 1, e["t"], e["kind"], set()
        ranks.add(e["rank"])
        rows[id(e)] = float(row)
    return rows


def chart(name: str, events, asks, busy, t0: float, t1: float, row_h: float = 0.15) -> dict:
    """A message-sequence chart of the run between t0 and t1: ranks as lifelines, time downward, one row per event."""
    shown = ("steal_request", "steal_grant", "steal_deny", "steal_received", "steal_denied", "load", "retire_start",
             "retired")  # fmt: skip
    idle_now = [e for ev in events for e in ev if e["kind"] == "level_end" and e.get("clusters_level_end") == 0]
    window = sorted((e for ev in events for e in ev if t0 <= e["t"] <= t1 and e["kind"] in shown), key=lambda e: e["t"])
    window += [e for e in idle_now if t0 <= e["t"] <= t1]
    window.sort(key=lambda e: (e["t"], e["kind"] != "level_end"))
    y = rows_of(window)
    n_rows = int(max(y.values())) + 1
    P = len(events)
    x = lambda r: 1.0 + r  # noqa: E731
    fig, ax = plt.subplots(figsize=(6.3, row_h * n_rows + 0.55))
    ax.set_xlim(-0.35, P + 1.9)
    ax.set_ylim(n_rows - 0.2, -1.15)
    ax.axis("off")

    def row_t(t: float) -> float:
        """The row of time t, between the rows of the events either side of it."""
        ts = [(e["t"], y[id(e)]) for e in window]
        before = [r for s, r in ts if s <= t]
        after = [r for s, r in ts if s >= t]
        if not before:
            return -0.6
        if not after:
            return n_rows - 0.4
        return (max(before) + min(after)) / 2 if max(before) != min(after) else max(before)

    # 1. Lifelines: coloured while the rank holds work, thin and grey while it is idle, a grey bar in the barrier.
    joined = {e["rank"]: e for e in window if e["kind"] == "retire_start"}
    end = {e["rank"]: y[id(e)] for e in window if e["kind"] == "retired"}
    for r in range(P):
        ax.text(x(r), -0.85, f"rank {r}", ha="center", va="bottom", fontsize=6.3, color=mf.RANKS[r], weight="bold")
        ax.plot([x(r)] * 2, [-0.6, n_rows - 0.4], color=mf.LIGHT, lw=0.8, zorder=1)
        for a, b in busy[r]:
            if b >= t0 and a <= t1:
                ax.plot([x(r)] * 2, [row_t(a), row_t(b)], color=mf.RANKS[r], lw=2.4, solid_capstyle="butt", zorder=1.2)
            if t0 <= b <= t1:
                ax.text(x(r) + 0.1, row_t(b), "queue empty", ha="left", va="center", fontsize=4.6, color=mf.GREY)
        if r in joined:
            ax.add_patch(plt.Rectangle((x(r) - 0.09, y[id(joined[r])]), 0.18, end.get(r, n_rows - 0.4) - y[id(joined[r])],
                                       fc="#e4e4e4", ec=mf.GREY, lw=0.4, zorder=1.5))  # fmt: skip

    # 2. Requests (grey), then each reply: a grant (dark, with what it carried) or a denial (red, with its reason).
    def served_row(r_thief, req, rep) -> float:
        """Where the reply leaves the victim: its own grant or denial, or just before an idle rank's denial arrives."""
        if rep.get("reason") == "IDLE":
            return y[id(rep)] - 0.55 if id(rep) in y else row_t(rep["t"]) - 0.55
        kind = "steal_grant" if rep["kind"] == "steal_received" else "steal_deny"
        sent = [e for e in events[rep["victim"]] if e["kind"] == kind and e["thief"] == r_thief and req["t"] <= e["t"] <= rep["t"]]
        return y[id(sent[0])] if sent and id(sent[0]) in y else row_t(sent[0]["t"] if sent else rep["t"])

    labels = []
    for r_thief, req, rep in asks:
        if rep["t"] < t0 or req["t"] > t1:
            continue
        v = rep["victim"]
        y_req = y.get(id(req), -0.6)
        y_srv = served_row(r_thief, req, rep)
        y_rep = y.get(id(rep), n_rows - 0.4)
        mf.arrow(ax, (x(r_thief), y_req), (x(v), y_srv), colour=mf.GREY, lw=0.5, scale=4, zorder=3)
        grant = rep["kind"] == "steal_received"
        colour = mf.DARK if grant else mf.RED
        mf.arrow(ax, (x(v), y_srv), (x(r_thief), y_rep), colour=colour, lw=1.0 if grant else 0.7, scale=5, zorder=4)
        text = f"{rep['n_clusters']} nodes" if grant else SHORT[rep["reason"].split(" ")[0]]
        labels.append((x(r_thief), y_rep, text, colour, x(v) < x(r_thief)))
    for xr, yr, text, colour, from_left in labels:
        ax.text(xr + (0.07 if from_left else -0.07), yr - 0.05, text, ha="left" if from_left else "right", va="bottom",
                fontsize=4.9, color=colour, zorder=6,
                path_effects=[mf.withStroke(linewidth=1.6, foreground="white")])  # fmt: skip

    # 3. Load gossip (an open diamond: sent to every rank, never waited for) and the barrier.
    for e in window:
        if e["kind"] == "load":
            ax.plot(x(e["rank"]), y[id(e)], marker="D", ms=2.6, mfc="white", mec=mf.RANKS[e["rank"]], mew=0.7, zorder=5)
            ax.text(x(e["rank"]) + 0.1, y[id(e)], f"load {e['n_pts']}", ha="left", va="center", fontsize=4.6,
                    color=mf.GREY, zorder=5)  # fmt: skip
        elif e["kind"] == "retire_start":
            ax.plot(x(e["rank"]), y[id(e)], marker="_", ms=6, mew=1.0, color=mf.DARK, zorder=5)
    if end:
        y_end = max(end.values())
        ax.plot([x(0) - 0.25, x(P - 1) + 0.25], [y_end, y_end], color=mf.DARK, lw=1.0, zorder=5)
        ax.text(x(P - 1) + 0.35, y_end, "barrier completes:\nevery rank retires", ha="left", va="center", fontsize=5.3,
                color=mf.DARK, linespacing=1.1)  # fmt: skip
        first_join = min(y[id(e)] for e in joined.values())
        ax.text(x(P - 1) + 0.35, (first_join + y_end) / 2, "grey: in the barrier,\nonly denying", ha="left",
                va="center", fontsize=5.3, color=mf.GREY, linespacing=1.1)  # fmt: skip

    # 4. Time on the left, in milliseconds from the window's first event, at the first event of each row.
    seen = set()
    for e in window:
        if y[id(e)] not in seen:
            seen.add(y[id(e)])
            ax.text(0.25, y[id(e)], f"{1000 * (e['t'] - t0):.2f}", ha="right", va="center", fontsize=4.6, color=mf.GREY)
    ax.text(0.25, -0.85, "ms", ha="right", va="bottom", fontsize=5.0, color=mf.GREY)
    mf.save(fig, name)
    return {"rows": n_rows, "ms": round(1000 * (t1 - t0), 1)}


def draw() -> None:
    data = json.loads(EVENTS.read_text())
    events = data["events"]
    asks = asks_of(events)
    checked = check_protocol(events, asks)
    busy = checked.pop("busy")

    # The first round in which a victim grants: every request its serving of that round answers.
    grant = min((e for ev in events for e in ev if e["kind"] == "steal_grant"), key=lambda e: e["t"])
    victim = events[grant["rank"]]
    k = victim.index(grant)
    lo = hi = k
    while lo > 0 and victim[lo - 1]["kind"] in ("steal_grant", "steal_deny", "load"):
        lo -= 1
    while hi + 1 < len(victim) and victim[hi + 1]["kind"] in ("steal_grant", "steal_deny", "load"):
        hi += 1
    served = {id(e) for e in victim[lo : hi + 1]}
    round_asks = [(r, q, a) for r, q, a in asks if q["victim"] == grant["rank"]
                  and any(id(e) in served and e.get("thief") == r and q["t"] <= e["t"] <= a["t"] for e in victim)]  # fmt: skip
    t0 = min(q["t"] for _, q, _ in round_asks)
    # The thieves publish their new load right after the grant arrives: keep that in the window.
    t1 = max(e["t"] for r, _, a in round_asks for e in events[r] if e["kind"] == "load" and a["t"] <= e["t"] <= a["t"] + 0.001)
    first = chart("apx-steal-grant", events, asks, busy, t0, t1)
    first |= {"victim": grant["rank"], "thieves": len(round_asks),
              "grants": [{"thief": e["thief"], "nodes": e["n_clusters"], "points": e["n_pts"]}
                         for e in victim[lo : hi + 1] if e["kind"] == "steal_grant"],  # in the order they were sent
              "denied": sum(a["kind"] == "steal_denied" for _, _, a in round_asks)}  # fmt: skip
    first["left"] = int(next(e for e in victim[lo : hi + 1] if e["kind"] == "steal_deny")["reason"].split("(")[1].split(" ")[0].replace(",", ""))

    # The end of the run: everything after the last reply a working victim sent.
    last_victim_reply = max(e["t"] for ev in events for e in ev if e["kind"] in ("steal_grant", "steal_deny"))
    t0 = min(q["t"] for _, q, _ in asks if q["t"] > last_victim_reply)
    t1 = max(e["t"] for ev in events for e in ev if e["kind"] == "retired")
    last = chart("apx-steal-retire", events, asks, busy, t0, t1)
    joins = sorted((e["t"], e["rank"]) for ev in events for e in ev if e["kind"] == "retire_start")
    # The first request sent to a rank whose queue emptied within the window, before the request: a stale load.
    emptied = {r: [s[1] for s in busy[r]] for r in range(len(events))}
    stale = next((r, q, a) for r, q, a in sorted(asks, key=lambda x: x[1]["t"])
                 if q["t"] >= t0 and any(t0 <= b < q["t"] for b in emptied[q["victim"]]) and a.get("reason") == "IDLE")  # fmt: skip
    waiting_denied = sorted(r for r, q, a in asks if q["victim"] == stale[0] and a.get("reason") == "IDLE"
                            and stale[1]["t"] <= a["t"] <= stale[2]["t"])  # fmt: skip
    out_at_join = sorted(r for r, q, a in asks if q["t"] <= joins[0][0] <= a["t"])
    if not waiting_denied or not out_at_join:
        raise SystemExit("The end of this recording shows no waiting thief that denies, or no early join.")
    last |= {"first_join": joins[0][1], "last_join": joins[-1][1],
             "early_ms": round(1000 * (joins[-1][0] - joins[0][0]), 1),
             "emptied": stale[1]["victim"], "stale_thief": stale[0], "waiting_denied": waiting_denied,
             "out_at_join": out_at_join}  # fmt: skip

    st = data["settings"]
    run = {"ranks": len(events), "atol": st["atol"], "batch_size": st["batch_size"], "min_load": st["steal_min_load"],
           "steal_size": st["steal_size"], "leaves": data["leaves"], "leaves_per_rank": data["leaves_per_rank"],
           "wall": round(max(e["t"] for ev in events for e in ev if e["kind"] == "done"), 2)}  # fmt: skip
    (mf.OUT / "apx-steal.json").write_text(json.dumps({"run": run, **checked, "first": first, "last": last}, indent=2) + "\n")
    print(json.dumps({"run": run, **checked, "first": first, "last": last}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="store_true", help="record apx-steal-events.json (under mpiexec -n 6)")
    if parser.parse_args().record:
        record()
    else:
        draw()


if __name__ == "__main__":
    main()
