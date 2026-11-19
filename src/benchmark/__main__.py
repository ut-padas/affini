"""``python -m src.benchmark <functions|step_maps|sweep|table> [options]``: one entry point for the suites, the sweep and the summary table."""

import sys

USAGE = "usage: python -m src.benchmark {functions,step_maps,sweep,table} [options]  (each takes --help)"


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("functions", "step_maps", "sweep", "table"):
        sys.exit(USAGE)
    if argv[0] == "functions":
        from src.benchmark.functions import main as run
    elif argv[0] == "step_maps":
        from src.benchmark.step_maps import main as run
    elif argv[0] == "sweep":
        from src.benchmark.sweep import main as run
    else:
        from src.benchmark.table import main as run
    run(argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
