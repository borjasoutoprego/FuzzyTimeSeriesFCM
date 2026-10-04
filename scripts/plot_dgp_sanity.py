"""Plot example series and latent regimes/transitions for E1--E4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from simulation.config import DEFAULT_BURN_IN, Scenario  # noqa: E402
from simulation.dgp import generate_series  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, choices=(50, 100, 500), default=500)
    parser.add_argument("--sigma", type=float, choices=(0.25, 0.50), default=0.25)
    parser.add_argument("--gamma", type=float, choices=(1.0, 5.0, 20.0), default=5.0)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--burn-in", type=int, default=DEFAULT_BURN_IN)
    parser.add_argument("--output", type=Path, default=REPOSITORY / "results" / "figures" / "dgp_sanity.png")
    args = parser.parse_args(argv)
    if args.burn_in < 0 or args.seed < 0:
        parser.error("--burn-in and --seed must be non-negative.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        parser.error(f"matplotlib is needed to create the sanity plot: {error}")

    data = {}
    for index, dgp in enumerate(("E1", "E2", "E3", "E4")):
        gamma = args.gamma if dgp == "E4" else None
        scenario = Scenario(dgp, args.T, args.sigma, gamma)
        data[dgp] = generate_series(
            scenario,
            seed=args.seed + index,
            burn_in=args.burn_in,
        )

    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    indices = range(args.T)
    for axis, dgp in zip(axes, ("E1", "E2", "E3", "E4")):
        generated = data[dgp]
        axis.plot(indices, generated.series, linewidth=1.0, label="X_t")
        if dgp == "E3":
            axis.step(indices, generated.true_regime, where="post", alpha=0.65, label="true regime")
        if dgp == "E4":
            axis.plot(indices, generated.G_true, alpha=0.8, label="G_true")
        axis.set_title(dgp + (f" (gamma={args.gamma:g})" if dgp == "E4" else ""))
        axis.legend(loc="upper right")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time index after burn-in")
    figure.tight_layout()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
