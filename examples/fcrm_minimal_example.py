"""Show FCRM fitting, both PDF gating methods, and both forecasts."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fcrm_timeseries import FuzzyCRegression


def main():
    rng = np.random.default_rng(4)
    series = np.zeros(120)
    for t in range(2, series.size):
        if t < 65:
            series[t] = 0.2 + 0.7 * series[t - 1] - 0.1 * series[t - 2]
        else:
            series[t] = -0.1 + 0.3 * series[t - 1] + 0.25 * series[t - 2]
        series[t] += rng.normal(0.0, 0.03)

    train = series[:100]
    history = series[:100]
    model = FuzzyCRegression(
        n_clusters=2, n_lags=2, m=2.0, n_init=10, random_state=0
    ).fit_series(train)

    print("FCRM regression coefficients [intercept, lag_1, lag_2]:")
    print(model.coef_)
    print("FCRM memberships shape:", model.memberships_.shape)
    print("FCRM objective:", model.objective_)

    for gating in ("fuzzy_centers", "multinomial"):
        model.fit_gating(gating)
        lag_vector = history[-1:-model.n_lags - 1:-1]
        omega = model.predict_gating_memberships(lag_vector, method=gating)
        cluster_forecasts = model.predict_by_cluster(lag_vector)
        weighted = model.predict_next(history, gating=gating, method="weighted")
        dominant = model.predict_next(history, gating=gating, method="dominant")
        regime = model.predict_regime(lag_vector, gating=gating)[0]

        print(f"\nGating method: {gating}")
        print("Gating weights omega:", omega[0])
        print("Dominant estimated regime:", regime)
        print("Per-regime forecasts:", cluster_forecasts[0])
        print("Weighted forecast:", weighted)
        print("Dominant forecast:", dominant)


if __name__ == "__main__":
    main()
