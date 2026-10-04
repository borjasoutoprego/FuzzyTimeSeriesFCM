# Proyecto TFM — Predicción de series de tiempo basada en cluster fuzzy

## Descripción

Este repositorio contiene el código y los recursos utilizados para el Trabajo Fin de Máster (TFM).

El objetivo principal del proyecto es estudiar y aplicar técnicas estadísticas y de aprendizaje automático para el análisis y predicción de series temporales basado en cluster fuzzy.

El proyecto se desarrolla principalmente en Python.

## Fuzzy C-Regression Models (FCRM)

`fcrm_timeseries.py` implements the procedure in `FCRM.pdf` as
`FuzzyCRegression`. FCRM jointly estimates `K` autoregressive dynamics of order
`p` and fuzzy memberships for each observed target. For each target `X_t`, the
regression vector is `[1, X_(t-1), ..., X_(t-p)]`; the cluster-specific
coefficients are found by weighted least squares with weights `u_tk**m`, with
no regression regularization.

The FCRM memberships (`memberships_`) are obtained from the absolute regression
residuals using the update formula in the PDF. They are used to fit the
regressions and as soft targets for gating. `residuals_` has one row per target
and one column per dynamic regime. The fitted attributes also include
`coef_` (intercept followed by lag 1 through lag `p`), `labels_`, `objective_`,
`objective_history_`, `converged_`, and `n_iter_`. Multiple independent random
initializations are compared; `random_state` makes them reproducible. Here
`labels_` is the argmax of the residual-based FCRM memberships; it is separate
from the dominant regime predicted by gating.

Gating estimates regime weights from observed lag vectors, without needing the
future target or its residual. `fit_gating("fuzzy_centers")` computes
membership-weighted lag centers and fuzzy distance memberships.
`fit_gating("multinomial")` fits a reference-category softmax by minimizing
cross-entropy against the soft FCRM memberships. Both can be fitted on the same
FCRM solution. `predict_gating_memberships(X, method=...)` returns these gating
weights (`omega`), distinct from `memberships_`; `predict_regime(X)` returns the
dominant gating regime, and `predict_by_cluster(X)` returns all `K` individual
regression forecasts.

For prediction, `predict(X, gating=..., method=...)` combines the regression
forecasts with either `method="weighted"` (the sum of each forecast times its
gating weight) or `method="dominant"` (the forecast from the regime with the
largest gating weight). The gating choice is independently either
`"fuzzy_centers"` or `"multinomial"`. `predict_next(history, gating=...,
method=...)` builds the newest-to-oldest lag vector from the observed history,
obtains its gating weights, and forecasts the next target. It never calculates
memberships from a future target residual.

```python
model.fit_series(train_series)
model.fit_gating("fuzzy_centers")
forecast = model.predict_next(
    history, gating="fuzzy_centers", method="weighted"
)
model.fit_gating("multinomial")
forecast_multinomial = model.predict_next(
    history, gating="multinomial", method="dominant"
)
```

## Examples and tests

Each implemented method has a minimal executable example in `examples/` and
unit tests in `tests/`: FCM, Cheng FTS, Egrioglu FTS-NN, FCM information
granules, and FCRM. Run the suite with:

```bash
python -m unittest discover -s tests -v
```

## Temporal evaluation

For FCM/Cheng/Egrioglu, fit `FuzzyTimeSeriesFCM` only with the training
partition. `fuzzify_new` assigns validation or test observations to the fixed
training centres; it does not refit FCM. `grid_search_fcm(train, validation,
...)` performs rolling one-step validation and deliberately has no test-data
argument. After selecting parameters, create a new model fitted on
train+validation and evaluate test one step at a time.

`FuzzyInformationGranules.fit(train)` uses only windows whose next target is
inside `train`; `predict(history)` assigns its final observed window to the
already fitted granular clusters. FCRM keeps residual-based memberships for
fitting and uses a separate gating model for forecasts, so
`predict_next(history, gating=..., method=...)` does not need the future target.

## Objetivos

- Preparar y explorar los datos.
- Analizar las características de las series temporales.
- Aplicar técnicas de análisis estadístico.
- Entrenar modelos de predicción.
- Evaluar y comparar los resultados obtenidos.
- Generar visualizaciones que permitan interpretar los resultados.

## Estructura del proyecto

La estructura puede variar durante el desarrollo. Como referencia:

```text
.
├── data/               # Datos utilizados por el proyecto
├── notebooks/          # Jupyter notebooks para análisis y experimentación
├── src/                # Código fuente
├── results/            # Resultados, tablas y gráficos
├── tests/              # Pruebas
├── README.md
└── AGENTS.md
