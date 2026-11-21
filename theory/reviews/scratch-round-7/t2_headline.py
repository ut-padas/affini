"""Is the corrected rollout of 7.3e-3 robust? Degrees 4-8, three draws of the 300 sensitivity states, sensitivities against central
differences, 100 against 300 states, 20 against all test trajectories; and the selection applied to the corrected surrogate."""

import json
import sys

import numpy as np
from common import P, fit, jac_of, rollout, setup, spec

from src.benchmark.step_maps import ROLLOUTS

TOL = 0.01
rows = []
cases = [(3, 300, "sens"), (3, 300, "fd"), (11, 300, "sens"), (12, 300, "sens"), (13, 300, "fd"), (14, 300, "fd"), (3, 100, "sens"), (11, 100, "fd"), (3, 600, "fd")]
for seed, n, jac in cases:
    s = setup(seed, n, jac)
    train, test = s["train"], s["test"]
    Xe, Ye = test[:, :-1].reshape(-1, 8), test[:, 1:].reshape(-1, 8)
    Zs = s["X"][s["js"][:200]]
    dyn = float(spec(s["Jd"][:200]).max())
    chart_gain = float(spec(np.einsum("nij,njk,lk->nil", s["Jd"], s["dD"], s["V"])).max())  # ||J DD V^T||: the dynamics along the chart
    for q in range(4, 9):
        G = fit(s, "gradients", q)
        plain, corr = s["mk"](G), s["mk"](G, s["Bcoef"])
        w = lambda sur: float(np.max(np.where(P.scored(Ye), np.abs(Xe + sur.predict(Xe) - Ye), 0.0)))  # noqa: E731
        r = lambda sur, tr: rollout(P, sur, tr)  # noqa: E731
        rc20 = r(corr, test[:ROLLOUTS])
        row = {"seed": seed, "n_sens": n, "jacobian": jac, "degree": q, "dynamics_norm": dyn, "chart_gain": chart_gain,
               "train_chain_plain": r(plain, train[:40])["rollout_worst"], "train_chain_corrected": r(corr, train[:40])["rollout_worst"],
               "train_chain_corrected_all": r(corr, train)["rollout_worst"],
               "dpsi_plain": float(spec(jac_of(plain, Zs)).max()), "dpsi_corrected": float(spec(jac_of(corr, Zs)).max()),
               "one_step_corrected": w(corr), "test_rollout_plain_20": r(plain, test[:ROLLOUTS])["rollout_worst"],
               "test_rollout_corrected_20": rc20["rollout_worst"], "test_rollout_corrected_all": r(corr, test)["rollout_worst"],
               "test_rollout_corrected_20_final": rc20["rollout_error_by_step"][-1],
               "test_rollout_corrected_20_argmax_step": int(np.argmax(rc20["rollout_error_by_step"]))}  # fmt: skip
        rows.append(row)
        print(f"seed {seed} n {n} {jac} q={q}: train chain plain {row['train_chain_plain']:.2e} corr(40) {row['train_chain_corrected']:.2e} corr(all) {row['train_chain_corrected_all']:.2e}; "
              f"|DPsi| plain {row['dpsi_plain']:.3f} corr {row['dpsi_corrected']:.3f} (dyn {dyn:.3f}, chart {chart_gain:.3f}); one-step {row['one_step_corrected']:.2e}; "
              f"test rollout plain {row['test_rollout_plain_20']:.2e}, corr 20: {row['test_rollout_corrected_20']:.2e} (step {row['test_rollout_corrected_20_argmax_step']}), all {len(test)}: {row['test_rollout_corrected_all']:.2e}", flush=True)  # fmt: skip
    json.dump(rows, open("t2_headline.json", "w"), indent=1)
