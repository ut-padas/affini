"""
KIDA's coverage against its data (council round 2: Fable's step 6d, Astra's learning curve against independent driver histories). The
local arm of p5 (ridge on the 64 nearest fitted states) and the degree-2 chart arm, refitted on nested subsets of the dark range's 160
fit tracers (20, 40, 80, 160; whole tracers, the first n of a fixed permutation), scored on the same 40 test tracers' plain inputs: the
median and worst key-species error and the oracle coverage (the share of test states within 0.01 dex and within half of it). No solves.

Prediction, committed before the run: the local arm's oracle coverage at 0.01 rises with the tracers but by less than a factor 2 per
doubling, and its extrapolation (a power law in the tracers through 80 and 160) reaches 50% only beyond 1,000 tracers; the chart arm's
coverage changes by less than 3 percentage points from 40 to 160 tracers (its error is the basis's, not the data's).
"""

import json

import numpy as np
from p5_kida_screen import OUT, TOL, Data, LinearArm, LocalArm


def main() -> None:
    D = Data()
    Xt, Yt = D.pairs(D.test)
    order = np.random.default_rng(5).permutation(len(D.fit))
    out = {}
    for n in (20, 40, 80, 160):
        Xf, Yf = D.pairs(D.fit[order[:n]])
        for arm, build in (("local", lambda X=Xf, Y=Yf: LocalArm(D, X, Y)), ("chart2", lambda X=Xf, Y=Yf: LinearArm(D, "chart2", X, Y))):
            if arm == "chart2" and n < 40:
                continue  # 276 features: fewer than 40 tracers (920 pairs) leave the fit barely determined
            key, _ = D.errors(build().predict(Xt), Yt)
            out[f"{arm}/{n}"] = {"tracers": n, "median": float(np.median(key)), "worst": float(key.max()),
                                 "coverage_percent": float(100 * np.mean(key <= TOL)), "coverage_half_percent": float(100 * np.mean(key <= TOL / 2))}  # fmt: skip
            r = out[f"{arm}/{n}"]
            print(f"{arm} {n} tracers: median {r['median']:.2e}, worst {r['worst']:.2f}, coverage {r['coverage_percent']:.1f}% (half {r['coverage_half_percent']:.1f}%)")
    loc = [out[f"local/{n}"]["coverage_percent"] for n in (80, 160)]
    if min(loc) > 0:
        slope = np.log(loc[1] / loc[0]) / np.log(2)
        out["local_extrapolation"] = {"exponent": float(slope), "tracers_for_50_percent": float(160 * (50 / loc[1]) ** (1 / slope)) if slope > 0 else None}
        print("local extrapolation:", out["local_extrapolation"])
    (OUT / "p8_kida_coverage.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
