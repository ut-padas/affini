"""The error of robertson_leaf.Curve.time (linear interpolation of the table z1 -> t), which both of the script's references share, so their difference cannot see it."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("rl", Path(__file__).resolve().parents[2] / "demos/robertson_leaf.py")
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)
ref = rl.Curve(1e-12)
ug = np.linspace(-0.29486, -0.02015, 20_001)
t = ref.time(ug)
back = np.log10(ref.sol.sol(t)[0])
print("sup |z1(time(u)) - u| on the defect grid:", np.abs(back - ug).max(), " (script's reference-error estimate: 1.1e-11)")
# the same inversion refined by two Newton steps on the dense output, and the step map's change
tn = t.copy()
for _ in range(3):
    y = ref.sol.sol(tn)
    dz = (-rl.K1 * y[0] + rl.K3 * y[1] * y[2]) / (y[0] * np.log(10))
    tn = tn - (np.log10(y[0]) - ug) / dz
print("after Newton: sup |z1(t) - u|:", np.abs(np.log10(ref.sol.sol(tn)[0]) - ug).max())
print("change of the step map's values (state at time + 1) from the inversion error, per component:", np.abs(ref.state(t + 1.0) - ref.state(tn + 1.0)).max(axis=0))
