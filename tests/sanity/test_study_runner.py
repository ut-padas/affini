"""The convergence-study drivers' case runner (studies/sanity/common.sh, run_logged)."""

import subprocess
from pathlib import Path

COMMON = Path(__file__).resolve().parents[2] / "studies" / "sanity" / "common.sh"


def test_a_case_that_reads_stdin_does_not_consume_the_remaining_cases(tmp_path):
    # The drivers read their case list from stdin; mpiexec forwards stdin to rank 0, so a case
    # launched through it would otherwise swallow every case after it.
    script = (
        f"source {COMMON} 2>/dev/null; RUN_TIMEOUT=0; n=0; "
        'while read -r c; do n=$((n+1)); run_logged "done/$c" "log/$c" cat; done < <(printf "a\\nb\\nc\\n"); '
        'echo "$n"'
    )
    out = subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert out.stdout.split()[-1] == "3"
    assert sorted(p.name for p in (tmp_path / "done").iterdir()) == ["a", "b", "c"]
