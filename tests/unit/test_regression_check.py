"""scripts/regression_check.py must fail on non-finite values (NaN compares as no difference)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import regression_check  # noqa: E402


def _save(prefix, pred):
    np.savez(f"{prefix}.npz", **{"case|pred_1_False": pred})
    Path(f"{prefix}.json").write_text(json.dumps({"cases": {"case": {"leaves": 1, "evals": 1, "generated": 0}}}))


def test_nan_predictions_fail_the_comparison(tmp_path):
    ref, new = str(tmp_path / "ref"), str(tmp_path / "new")
    _save(ref, np.array([1.0, 2.0]))
    _save(new, np.array([1.0, 2.0]))
    assert regression_check.compare(new, ref, 0.0)
    _save(new, np.array([1.0, np.nan]))
    assert not regression_check.compare(new, ref, 0.0)
