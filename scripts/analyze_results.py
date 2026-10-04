"""Summarize saved simulation replications without rerunning models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from simulation.statistics import analyze  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPOSITORY / "results" / "raw")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "results" / "aggregated")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be between 0 and 1.")
    result = analyze(
        args.input_dir,
        args.output_dir,
        make_plots=not args.no_plots,
        alpha=args.alpha,
    )
    print(f"Analyzed {result['replication_rows']} core test rows.")
    print(f"Summarized {result['counterfactual_rows']} separate FCRM counterfactual test rows.")
    print(f"Summary: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
