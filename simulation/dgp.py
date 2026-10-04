"""Data generating processes E.1--E.4 from the methodology PDF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .config import Scenario


@dataclass(frozen=True)
class DGPResult:
    """Final T observations and aligned latent regime diagnostics."""

    series: np.ndarray
    true_regime: np.ndarray | None = None
    G_true: np.ndarray | None = None


def generate_series(
    scenario: Scenario,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    burn_in: int = 200,
) -> DGPResult:
    """Generate a reproducible series, dropping burn-in from every output.

    The normal draw uses ``sigma`` as the standard deviation, as stated in the
    methodology PDF. Initial conditions are zero. ``burn_in`` is configurable
    because the PDF does not prescribe an initialization or burn-in length.
    """
    if not isinstance(burn_in, (int, np.integer)) or burn_in < 0:
        raise ValueError("burn_in must be a non-negative integer.")
    if (rng is None) == (seed is None):
        raise ValueError("provide exactly one of seed or rng.")
    generator = np.random.default_rng(seed) if rng is None else rng
    if not isinstance(generator, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    size = scenario.T + int(burn_in)
    values = np.zeros(size, dtype=float)
    regimes = np.zeros(size, dtype=np.int8) if scenario.dgp == "E3" else None
    transition = np.zeros(size, dtype=float) if scenario.dgp == "E4" else None
    innovations = generator.normal(0.0, scenario.sigma, size=size)
    if regimes is not None:
        # Zero initial conditions imply the first two threshold states are low.
        regimes[:min(2, size)] = 0
    if transition is not None:
        # The initial lag values are zero, so their logistic transitions are 0.5.
        transition[:min(2, size)] = 0.5

    if scenario.dgp == "E1":
        for t in range(1, size):
            values[t] = 0.7 * values[t - 1] + innovations[t]
    else:
        for t in range(2, size):
            lag_1, lag_2 = values[t - 1], values[t - 2]
            if scenario.dgp == "E2":
                values[t] = 1.5 * lag_1 - 0.7 * lag_2 + innovations[t]
            elif scenario.dgp == "E3":
                regime = 0 if lag_1 <= 0.0 else 1
                regimes[t] = regime
                if regime == 0:
                    values[t] = 0.82 * lag_1 + 0.05 * lag_2 + innovations[t]
                else:
                    values[t] = -0.45 * lag_1 + 0.30 * lag_2 + innovations[t]
            else:
                g_t = float(expit(float(scenario.gamma) * lag_1))
                transition[t] = g_t
                values[t] = (
                    (1.0 - g_t) * (0.82 * lag_1 + 0.05 * lag_2)
                    + g_t * (-0.45 * lag_1 + 0.30 * lag_2)
                    + innovations[t]
                )

    start = int(burn_in)
    return DGPResult(
        series=values[start:].copy(),
        true_regime=regimes[start:].copy() if regimes is not None else None,
        G_true=transition[start:].copy() if transition is not None else None,
    )


def derive_seed(
    master_seed: int,
    scenario: Scenario,
    replication: int,
    stream: int,
    candidate: int = 0,
) -> int:
    """Derive independent stable seeds without touching NumPy's global RNG."""
    if not isinstance(master_seed, (int, np.integer)) or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer.")
    if not isinstance(replication, (int, np.integer)) or replication < 1:
        raise ValueError("replication must be a positive one-based index.")
    gamma_code = 0 if scenario.gamma is None else int(scenario.gamma)
    spawn_key = (
        (int(scenario.dgp[1:]) - 1),
        int(scenario.T),
        int(round(scenario.sigma * 100)),
        gamma_code,
        int(replication),
        int(stream),
        int(candidate),
    )
    sequence = np.random.SeedSequence(int(master_seed), spawn_key=spawn_key)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])
