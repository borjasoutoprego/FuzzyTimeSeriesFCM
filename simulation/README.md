# Simulation framework

This package implements the TFM experiment in `PLAN_SIMULACION.pdf` and uses
the existing Cheng, Egrioglu, FCM-Granular, and FCRM implementations.

## Design and evaluation

- `config.py` records E1--E4, `T={50,100,500}`, `sigma={0.25,0.50}` as the
  innovation **standard deviation**, E4 `gamma={1,5,20}`, `m=2`, candidate
  cluster counts, and the chronological split.
- `dgp.py` generates exactly T retained observations after a configurable
  burn-in and preserves the SETAR regime or LSTAR `G_true` at every retained
  time index. Random generators are local to a derived seed.
- `evaluation.py` has no test argument in any selection function. Cheng,
  Egrioglu, and FCRM choose `c` by validation MAE; FCM-Granular jointly chooses
  `(c,L)` by validation MAE. Validation forecasts are rolling one-step: the
  forecast is made before the actual validation value is appended.
- Each selected model is refit using train+validation. The test is then
  forecast one step at a time, appending actual observations after each
  forecast. AR uses the DGP's structural order (E1: 1; E2--E4: 2), as in the
  PDF's “real significant lags” benchmark.
- FCRM candidates share one regression/membership fit across the two gates;
  each gate receives its own validation score and may select a different `c`.
  For E3/E4, both gates also get a separate fixed `c=2` model.
- `statistics.py` summarizes test results within each scenario, ranks
  procedures, runs paired Friedman tests, runs Nemenyi comparisons only after a
  significant Friedman result, and saves scenario-specific boxplots.
  E3's `FCRM_TRUE`, `FCRM_RV`, and `FCRM` error decomposition is summarized in
  a separate `fcrm_counterfactual_summary.csv`, outside the core procedure
  rankings and Friedman/Nemenyi tests.

## FCRM conventions and diagnostics

All FCRM predictions fit `fit_gating(method=...)` and use `method="dominant"`.
This retains the hard regime forecast used in the PDF while honoring the
instruction that forecasting must use predictor-based gating. Residual-based
`memberships_` are not used as future prediction weights. Both
`fuzzy_centers` and `multinomial` are stored as distinct `method_variant`s.

For E3, the JSON result stores fitted coefficients, residual memberships,
true regimes, fit and post-hoc test residual-membership ARI, test gating
weights, and raw gating ARI. The test membership ARI uses observed responses
only after test forecasts are complete; it does not affect prediction.
For fixed `c=2`, estimated coefficient rows are matched to the two true
coefficient vectors by minimum total Euclidean distance; this mapping is used
for the coefficient-matched gate labels and the coefficient diagnostic. For
`c>2`, only the raw gate ARI is a direct comparison; nearest-coefficient regime
grouping is stored under an explicitly auxiliary field.

The three E3 counterfactual forecasts are saved for each gate: oracle
coefficients and regime (`FCRM_TRUE`), estimated coefficients and true regime
(`FCRM_RV`), and estimated coefficients with predictor-based dominant gating
(`FCRM`). The PDF defines its third decomposition forecast using
defuzzification of response-residual memberships. The implementation retains
predictor-based gating for forecasts as separately requested, so that
decomposition forecast is a known methodological divergence to resolve before
the full run. Residual memberships do not use test responses for prediction.

For E4, the PDF-defined transition diagnostic uses the fixed `c=2` model. Its
coefficient rows are matched one-to-one to the true dynamics by minimum total
Euclidean distance, then test response-residual membership `u_t2` is correlated
with test `G_true`. The memberships are computed post hoc after forecasts have
been produced; they are not used in validation, selection, or prediction. For
`c>2`, a nearest-coefficient membership aggregate is retained only as an
explicitly labeled auxiliary extension; it is not reported as the PDF-defined
diagnostic.

The PDF describes defuzzifying FCRM's response-residual memberships for its
regime ARI and E3 decomposition forecast. The experiment request requires
predictor-based gating for forecasts and prohibits using `memberships_for()`
as a substitute for gating. Therefore all out-of-sample predictions, including
the E3 `FCRM` decomposition forecast, use the selected gate. `memberships_for()`
is used only for the explicitly post-hoc E3 regime ARI and E4
membership-versus-`G_true` diagnostics, after predictions have been recorded.
This E3 decomposition choice still requires methodological approval before a
full run.

## Reproducibility and saved files

`numpy.random.SeedSequence` derives one seed per DGP, T, sigma, gamma,
replication, and model/candidate stream from the master seed. Both gates share
the same FCRM candidate/refit, so their only model difference is the gating
method. Every replication is one JSON file containing
metrics, validation/test predictions, and diagnostics. A file is written to a
temporary sibling and atomically replaced only after the replication finishes.
Existing files are skipped, enabling restart after interruption; resuming with
a different master seed or burn-in in the same output directory is rejected.
Metadata is
written to `results/experiment_metadata.json` and records the design, package
versions, Python/platform, CPU count, Git SHA, burn-in, seeds, and run history.

Each replication JSON's `metrics` array holds one row per evaluated variant and
split; its `predictions` array stores each individual forecast with time index,
split, true value, predicted value, and the SETAR/LSTAR latent value when
applicable. `diagnostics` retains model memberships, coefficients, latent
series, candidate scores, and FCRM-specific arrays.

## Commands

Pilot 1 (5 replications):

```powershell
python scripts/run_simulation.py --dgp E1 --T 50 --sigma 0.25 --replications 5 --output-dir results/pilot_5/raw
```

Pilot 2 (10 replications, separate output for timing):

```powershell
python scripts/run_simulation.py --dgp E1 --T 50 --sigma 0.25 --replications 10 --output-dir results/pilot_10/raw
```

Create DGP sanity plots:

```powershell
python scripts/plot_dgp_sanity.py --output results/figures/dgp_sanity.png
```

Analyze saved results:

```powershell
python scripts/analyze_results.py --input-dir results/raw --output-dir results/aggregated
```

Launch 200 replications for the full design (36 scenario combinations,
including all three E4 gamma values):

```powershell
python scripts/run_simulation.py --dgp all --T all --sigma all --gamma all --replications 200 --start-replicate 1 --n-jobs 1 --seed 20261004 --burn-in 200 --output-dir results/raw
```

`--end-replicate` is an inclusive alternative to `--replications` for a
replication range. Increase `--n-jobs` to run independent replications in
separate CPU processes. Result files are isolated by scenario and replication.

## Methodology choices left explicit

The PDF does not specify initial values or a burn-in. The request proposed a
configurable burn-in such as 200; 200 is the default and is stored in metadata.
Initial DGP values are zero, and burn-in is discarded before splitting the T
retained observations. Selection uses MAE because the PDF explicitly gives MAE
as an example validation criterion. sMAPE is reported as a percentage, with a
zero contribution when both the target and forecast are zero.
