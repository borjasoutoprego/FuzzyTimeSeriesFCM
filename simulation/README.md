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

## FCRM conventions and diagnostics

All FCRM predictions fit `fit_gating(method=...)` and use `method="dominant"`.
This retains the hard regime forecast used in the PDF while honoring the
instruction that forecasting must use predictor-based gating. Residual-based
`memberships_` are not used as future prediction weights. Both
`fuzzy_centers` and `multinomial` are stored as distinct `method_variant`s.

For E3, the JSON result stores the fitted coefficients, residual memberships,
true regimes, residual-membership ARI, test gating weights, and gating-regime
ARI. With fixed `c=2`, estimated coefficient rows are matched to the two true
coefficient vectors by minimum total Euclidean distance. The three PDF error
decomposition forecasts are also saved for each gate: oracle coefficients and
regime (`FCRM_TRUE`), estimated coefficients and true regime (`FCRM_RV`), and
estimated coefficients and dominant gating regime (`FCRM`). For `c>2`, an
ARI can still be computed label-invariantly, but a one-to-one coefficient
matching to two true regimes is not defined; exact coefficient matching and
the error decomposition therefore use the requested fixed `c=2` models.

For E4, each gate's estimated transition is the total gating weight assigned
to estimated coefficient vectors closer to the second true dynamic. This is a
direct two-cluster correspondence for `c=2`; for `c>2`, it aggregates clusters
by their nearest true coefficient vector. Pearson and Spearman correlations
with test `G_true`, plus the values and gating weights, are stored.

The PDF describes defuzzifying FCRM's response-residual memberships for its
regime ARI and forecast. The request separately requires predictor-based
gating for forecasts. The implementation preserves both ideas as separate
diagnostics: fit-membership ARI follows the PDF; actual forecasts and test
regime diagnostics use the selected gating. `memberships_for()` is not used.

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
