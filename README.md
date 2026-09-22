# Proyecto TFM — Predicción de series de tiempo basada en cluster fuzzy

## Descripción

Este repositorio contiene el código y los recursos utilizados para el Trabajo Fin de Máster (TFM).

El objetivo principal del proyecto es estudiar y aplicar técnicas estadísticas y de aprendizaje automático para el análisis y predicción de series temporales basado en cluster fuzzy.

El proyecto se desarrolla principalmente en Python.

## Fuzzy C-Regression Models (PF.4)

`fcrm_timeseries.py` implements PF.4 as `FuzzyCRegression`. Unlike FCM, which
uses distances from observations to centroids, FCRM obtains memberships from
the residual of each cluster-specific regression. Each cluster is therefore an
estimated dynamic regime. The public attributes after fitting are `coef_`
(intercept followed by lag coefficients), `memberships_`, `labels_`,
`residuals_`, `objective_` and `objective_history_`.
Lag columns are ordered as `lag_n_lags, ..., lag_1`; the same names are
available in `feature_names_in_`.

Use `make_lagged_supervised(series, n_lags)` to form train/validation/test
matrices, then fit only the training partition. `predict_by_cluster(X)` returns
all regression forecasts. For leakage-safe temporal one-step forecasting, use
`predict_next(history)`: it derives the current regime from already observed
history and applies it to the next forecast. `predict(X, memberships, method)`
also supports explicit dominant-regime (`"dominant"`) and weighted (`"weighted"`)
combinations when memberships are already available.

## Examples and tests

Each implemented method has a minimal executable example in `examples/` and
unit tests in `tests/`: FCM, Cheng FTS, Egrioglu FTS-NN, FCM information
granules, and FCRM. Run the suite with:

```bash
python -m unittest discover -s tests -v
```

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
