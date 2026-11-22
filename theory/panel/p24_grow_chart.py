"""
Growing the chart under coupling, and refusal-driven data on the reduced charts (council round 5: Fable's 2.4, Astra's POLLU refit). p20's
three rounds of refused (or uniformly drawn) closed-column inputs, audited on independent closed columns, on:
  pollu, hires   with the full chart (every free coordinate, no reduction), the start fitted here: on R4e's reduced charts their coupled
                 inputs lie off the chart and refused data collapsed the fit to degree 2 (p18);
  bruss_d20, burgers_d20, fisher_kpp_d10   on R4e's reduced charts, which p20 did not cover.
One record per (problem, arm) under data/records/p24/; a case is limited to 3,600 s. Usage: p24_grow_chart.py [problem ...].

Predictions, committed before the run:
  G1  on the full chart pollu is served at 340 trajectories, and after three refused rounds its audit solver share is below 20% (p17 on its
      reduced chart: 96-100%), every audit column within tol / 2;
  G2  hires on the full chart stays above 50% audit solver share after three refused rounds, or exceeds tol / 2 in an audit column (its
      error amplification, A_n = 6,669, not its chart, is the limit);
  G3  on the three reduced-chart lattices the refused arm's audit solver share after three rounds is below the uniform arm's, every audit
      within tol / 2.
"""

import sys
from pathlib import Path

from p20_closed_active import OUT, ROUNDS, TOL, case

from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import recorded  # noqa: E402

CHARTS = {"pollu": "full", "hires": "full", "bruss_d20": "r4e", "burgers_d20": "r4e", "fisher_kpp_d10": "r4e"}


def main() -> None:
    import json

    names = sys.argv[1:] or list(CHARTS)
    dstore = RecordStore(OUT / "records" / "p13_data", path_of=lambda k: OUT / "records" / "p13_data" / f"{k['problem']}_{k['chunk']}.json")
    store = RecordStore(OUT / "records" / "p24", path_of=lambda k: OUT / "records" / "p24" / f"{k['problem']}_{k['arm']}.json")
    for name in names:
        for arm in ("refused", "uniform"):
            recorded(store, {"problem": name, "arm": arm, "chart": CHARTS[name], "rounds": ROUNDS, "tol": TOL}, case, name, arm, dstore, CHARTS[name], limit=3600)
    summary = {f"{r['key']['problem']}/{r['key']['arm']}": {"status": r.get("status"), "chart": r.get("chart"), "k": r.get("k"), "rounds": [
        {k: v for k, v in x.items() if k != "audit"} | {"audit_worst": x["audit"]["worst"], "audit_solver_share": x["audit"]["solver_share"],
                                                         "audit_speedup": x["audit"]["median_speedup"]} for x in r.get("rounds", [])]}
        for r in store.records()}  # fmt: skip
    (OUT / "p24_grow_chart.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
