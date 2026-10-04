"""One-replication simulation workflow and incremental result persistence."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fuzzy_information_granules import FuzzyInformationGranules
from fcrm_timeseries import FuzzyCRegression

from .config import (
    AR_ORDERS,
    CLUSTER_CANDIDATES,
    DEFAULT_BURN_IN,
    FCRM_GATING_METHODS,
    FUZZIFIER,
    GRANULAR_WINDOWS,
    Scenario,
    scenario_slug,
    split_series,
)
from .diagnostics import (
    TRUE_REGIME_COEFFICIENTS,
    adjusted_rand,
    correlations,
    gating_transition_estimate,
    match_two_regime_coefficients,
    nearest_true_regime_by_coefficients,
)
from .dgp import derive_seed, generate_series
from .evaluation import (
    fit_ar,
    load_fcm_class,
    predict_ar,
    rolling_lag_vectors,
    rolling_one_step,
    select_fcm_clusters,
    select_fcrm_clusters,
    select_granular_parameters,
)
from .metrics import calculate_metrics


def _plain(value):
    """Convert NumPy scalar/container values into JSON-compatible objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _fcrm_variant(gating, dgp, *, selected):
    name = "FCRM_fuzzy_centers" if gating == "fuzzy_centers" else "FCRM_multinomial"
    if dgp in {"E3", "E4"}:
        name += "_c_selected" if selected else "_c2"
    return name


def evaluate_replication(
    scenario: Scenario,
    replication: int,
    *,
    master_seed: int,
    burn_in: int = DEFAULT_BURN_IN,
):
    """Evaluate every procedure for one scenario and one independent series.

    Selection helpers receive only train and validation arrays. The test series
    is first passed to a forecaster after all choices and final refits are done.
    """
    seed_ledger = {}

    def model_seed(stream, candidate=0):
        key = f"{int(stream)}:{int(candidate)}"
        seed_ledger[key] = derive_seed(
            master_seed, scenario, replication, stream=stream, candidate=candidate
        )
        return seed_ledger[key]

    dgp_seed = model_seed(0)
    generated = generate_series(scenario, seed=dgp_seed, burn_in=burn_in)
    split = split_series(generated.series)
    train = split.train
    validation = split.validation
    test = split.test
    train_validation = np.concatenate((train, validation))
    validation_start = split.validation_range[0]
    test_start = split.test_range[0]

    base = {
        "dgp": scenario.dgp,
        "gamma": scenario.gamma,
        "T": scenario.T,
        "sigma": scenario.sigma,
        "replication": int(replication),
        "seed": int(dgp_seed),
    }
    metric_rows = []
    prediction_rows = []
    diagnostics = {
        "scenario": _plain(base),
        "master_seed": int(master_seed),
        "burn_in": int(burn_in),
        "derived_seeds": seed_ledger,
        "split_indices": {
            "train": list(split.train_range),
            "validation": list(split.validation_range),
            "test": list(split.test_range),
        },
        "true_regime": _plain(generated.true_regime),
        "G_true": _plain(generated.G_true),
        "selection": {},
        "FCRM": {},
    }

    def record(method, variant, gating, group, evaluation_split, y_true, y_pred,
               *, selected_c=None, selected_L=None, start_index=0):
        scores = calculate_metrics(y_true, y_pred)
        metric_rows.append({
            **base,
            "method": method,
            "method_variant": variant,
            "gating": gating,
            "selected_c": selected_c,
            "selected_L": selected_L,
            "evaluation_split": evaluation_split,
            "result_group": group,
            **scores,
        })
        for offset, (truth, prediction) in enumerate(zip(y_true, y_pred)):
            index = int(start_index + offset)
            row = {
                **base,
                "method": method,
                "method_variant": variant,
                "gating": gating,
                "selected_c": selected_c,
                "selected_L": selected_L,
                "time_index": index,
                "y_true": float(truth),
                "y_pred": float(prediction),
                "split": evaluation_split,
                "result_group": group,
            }
            if generated.true_regime is not None:
                row["true_regime"] = int(generated.true_regime[index])
            else:
                row["true_regime"] = None
            if generated.G_true is not None:
                row["G_true"] = float(generated.G_true[index])
            else:
                row["G_true"] = None
            prediction_rows.append(row)

    # Benchmarks do not require hyperparameter selection.
    naive = rolling_one_step(train_validation, test, lambda history: history[-1])
    record("Naive", "Naive", None, "core", "test", test, naive, start_index=test_start)

    ar_order = AR_ORDERS[scenario.dgp]
    ar_coefficients = fit_ar(train_validation, ar_order)
    ar_predictions = rolling_one_step(
        train_validation,
        test,
        lambda history: predict_ar(history, ar_coefficients),
    )
    record("AR", "AR", None, "core", "test", test, ar_predictions, start_index=test_start)
    diagnostics["AR"] = {"order": ar_order, "coefficients": ar_coefficients.tolist()}

    # Cheng FTS and Egrioglu FTS-NN use the existing FCM implementation.
    fcm_selections = {}
    FCM = load_fcm_class()
    for method, variant, stream in (
        ("cheng", "Cheng_FTS", 10),
        ("egrioglu", "Egrioglu_FTS_NN", 11),
    ):
        selection = select_fcm_clusters(
            train,
            validation,
            method=method,
            seed_for_candidate=lambda c, stream=stream: model_seed(stream, c),
        )
        selected_c = selection["selected_c"]
        record(
            variant,
            variant,
            None,
            "validation_selection",
            "validation",
            validation,
            selection["validation_predictions"],
            selected_c=selected_c,
            start_index=validation_start,
        )
        model = FCM(
            selected_c,
            m=FUZZIFIER,
            random_state=model_seed(30 + stream, selected_c),
        ).fit_fcm(train_validation)
        labels = model.fuzzify()
        if method == "cheng":
            model.build_rules(labels)
        else:
            model.train_nn(labels)
        final_predictions = rolling_one_step(
            train_validation,
            test,
            lambda history, model=model, method=method: model.predict_one_step(
                history, method=method
            ),
        )
        record(
            variant,
            variant,
            None,
            "core",
            "test",
            test,
            final_predictions,
            selected_c=selected_c,
            start_index=test_start,
        )
        fcm_selections[variant] = {
            "selected_c": selected_c,
            "validation_mae_by_c": selection["scores"],
            "failed_candidates": selection["failures"],
        }
    diagnostics["selection"].update(fcm_selections)

    # Joint selection of FCM-Granular (c, L), followed by a train+validation refit.
    granular_selection = select_granular_parameters(
        train,
        validation,
        window_candidates=GRANULAR_WINDOWS[scenario.T],
        seed_for_candidate=lambda c, window: model_seed(12, int(c) * 100 + int(window)),
    )
    granular_c = granular_selection["selected_c"]
    granular_L = granular_selection["selected_L"]
    record(
        "FCM_Granular",
        "FCM_Granular",
        None,
        "validation_selection",
        "validation",
        validation,
        granular_selection["validation_predictions"],
        selected_c=granular_c,
        selected_L=granular_L,
        start_index=validation_start,
    )
    granular_model = FuzzyInformationGranules(
        granular_L,
        granular_c,
        m=FUZZIFIER,
        random_state=model_seed(32, granular_c * 100 + granular_L),
    ).fit(train_validation)
    granular_predictions = rolling_one_step(
        train_validation, test, lambda history: granular_model.predict(history)
    )
    record(
        "FCM_Granular",
        "FCM_Granular",
        None,
        "core",
        "test",
        test,
        granular_predictions,
        selected_c=granular_c,
        selected_L=granular_L,
        start_index=test_start,
    )
    diagnostics["selection"]["FCM_Granular"] = {
        "selected_c": granular_c,
        "selected_L": granular_L,
        "validation_mae_by_c_L": {
            f"{c},{window}": mae for (c, window), mae in granular_selection["scores"].items()
        },
        "failed_candidates": granular_selection["failures"],
    }

    # FCRM candidates are fit once and scored separately through each gating.
    fcrm_selection = select_fcrm_clusters(
        train,
        validation,
        n_lags=2,
        seed_for_candidate=lambda c: model_seed(20, c),
    )
    selected_by_gate = {
        gate: details["selected_c"]
        for gate, details in fcrm_selection["selected"].items()
    }
    diagnostics["selection"]["FCRM"] = {
        gate: {
            "selected_c": item["selected_c"],
            "validation_mae_by_c": fcrm_selection["candidate_scores"][gate],
            "failed_candidates": fcrm_selection["candidate_failures"][gate],
        }
        for gate, item in fcrm_selection["selected"].items()
    }
    for gate, item in fcrm_selection["selected"].items():
        record(
            "FCRM",
            _fcrm_variant(gate, scenario.dgp, selected=True),
            gate,
            "validation_selection",
            "validation",
            validation,
            item["validation_predictions"],
            selected_c=item["selected_c"],
            start_index=validation_start,
        )
        if scenario.dgp in {"E3", "E4"} and 2 in fcrm_selection["candidate_forecasts"][gate]:
            record(
                "FCRM",
                _fcrm_variant(gate, scenario.dgp, selected=False),
                gate,
                "validation_fixed_c2",
                "validation",
                validation,
                fcrm_selection["candidate_forecasts"][gate][2],
                selected_c=2,
                start_index=validation_start,
            )

    # Refit each required c only on train+validation; both gates use this same fit.
    final_cluster_counts = set(selected_by_gate.values())
    if scenario.dgp in {"E3", "E4"}:
        final_cluster_counts.add(2)
    final_models = {}
    for c in sorted(final_cluster_counts):
        model = FuzzyCRegression(
            n_clusters=c,
            n_lags=2,
            m=FUZZIFIER,
            random_state=model_seed(40, c),
        ).fit_series(train_validation)
        for gate in FCRM_GATING_METHODS:
            model.fit_gating(method=gate)
        final_models[c] = model

    fcrm_test_predictions = {}
    for gate in FCRM_GATING_METHODS:
        selected_c = selected_by_gate[gate]
        selected_model = final_models[selected_c]
        predictions = rolling_one_step(
            train_validation,
            test,
            lambda history, model=selected_model, gate=gate: model.predict_next(
                history, gating=gate, method="dominant"
            ),
        )
        fcrm_test_predictions[(gate, selected_c)] = predictions
        record(
            "FCRM",
            _fcrm_variant(gate, scenario.dgp, selected=True),
            gate,
            "core",
            "test",
            test,
            predictions,
            selected_c=selected_c,
            start_index=test_start,
        )

        if scenario.dgp in {"E3", "E4"}:
            fixed_model = final_models[2]
            fixed_predictions = rolling_one_step(
                train_validation,
                test,
                lambda history, model=fixed_model, gate=gate: model.predict_next(
                    history, gating=gate, method="dominant"
                ),
            )
            fcrm_test_predictions[(gate, 2)] = fixed_predictions
            record(
                "FCRM",
                _fcrm_variant(gate, scenario.dgp, selected=False),
                gate,
                "core",
                "test",
                test,
                fixed_predictions,
                selected_c=2,
                start_index=test_start,
            )

    if scenario.dgp == "E3":
        diagnostics["FCRM"]["E3"] = _setar_diagnostics(
            final_models,
            selected_by_gate,
            generated.true_regime,
            train_validation,
            test,
            test_start,
            fcrm_test_predictions,
            record,
        )
    elif scenario.dgp == "E4":
        diagnostics["FCRM"]["E4"] = _lstar_diagnostics(
            final_models,
            selected_by_gate,
            generated.G_true,
            train_validation,
            test,
            test_start,
        )

    return {
        "metrics": metric_rows,
        "predictions": prediction_rows,
        "diagnostics": _plain(diagnostics),
    }


def _setar_diagnostics(
    models,
    selected_by_gate,
    true_regime,
    train_validation,
    test,
    test_start,
    predictions_by_gate_and_c,
    record,
):
    """Compute PDF coefficient matching, ARI, and c=2 error decomposition."""
    if true_regime is None:
        raise ValueError("E3 must provide true regime labels.")
    true_test = np.asarray(true_regime[test_start:], dtype=int)
    lag_vectors = rolling_lag_vectors(train_validation, test, n_lags=2)
    output = {"true_coefficients": TRUE_REGIME_COEFFICIENTS.tolist(), "models": {}}

    fixed = models[2]
    matching = match_two_regime_coefficients(fixed.coef_)
    cluster_to_regime = np.asarray(matching["estimated_cluster_to_true_regime"], dtype=int)
    regime_to_cluster = np.argsort(cluster_to_regime)
    cluster_predictions = fixed.predict_by_cluster(lag_vectors)
    true_setar = np.where(
        true_test == 0,
        0.82 * lag_vectors[:, 0] + 0.05 * lag_vectors[:, 1],
        -0.45 * lag_vectors[:, 0] + 0.30 * lag_vectors[:, 1],
    )

    for gate in FCRM_GATING_METHODS:
        for label, c in (("c_selected", selected_by_gate[gate]), ("c2", 2)):
            model = models[c]
            fit_truth = np.asarray(true_regime[2:train_validation.size], dtype=int)
            if fit_truth.size != model.memberships_.shape[0]:
                raise RuntimeError("FCRM fitted membership rows do not align with E3 targets.")
            residual_labels = model.labels_
            weights = model.predict_gating_memberships(lag_vectors, method=gate)
            raw_gate_labels = np.argmax(weights, axis=1)
            mapped_gate_labels = nearest_true_regime_by_coefficients(model.coef_)[0][raw_gate_labels]
            output["models"][f"{gate}_{label}"] = {
                "n_clusters": c,
                "coefficients": model.coef_.tolist(),
                "fit_memberships": model.memberships_.tolist(),
                "fit_true_regime": fit_truth.tolist(),
                "fit_estimated_residual_regime": residual_labels.tolist(),
                "fit_membership_ari": adjusted_rand(fit_truth, residual_labels),
                "test_true_regime": true_test.tolist(),
                "test_gating_weights": weights.tolist(),
                "test_estimated_gating_regime_raw": raw_gate_labels.tolist(),
                "test_estimated_gating_regime_nearest_true_coefficient": mapped_gate_labels.tolist(),
                "test_gating_ari_raw": adjusted_rand(true_test, raw_gate_labels),
                "test_gating_ari_mapped": adjusted_rand(true_test, mapped_gate_labels),
            }
            if c == 2:
                output["models"][f"{gate}_{label}"].update({
                    "true_coefficients": TRUE_REGIME_COEFFICIENTS.tolist(),
                    "coefficient_distance_matrix": matching["distance_matrix"].tolist(),
                    "coefficient_total_distance": matching["total_distance"],
                    "estimated_cluster_to_true_regime": cluster_to_regime.tolist(),
                })

        # Three c=2 counterfactual predictors from the PDF, duplicated by gate
        # label so the two required FCRM variants remain explicitly traceable.
        gate_weights = fixed.predict_gating_memberships(lag_vectors, method=gate)
        dominant_clusters = np.argmax(gate_weights, axis=1)
        fcrm_predictions = cluster_predictions[np.arange(test.size), dominant_clusters]
        fcrm_rv = cluster_predictions[np.arange(test.size), regime_to_cluster[true_test]]
        variants = (
            (f"FCRM_TRUE_{gate}_c2", true_setar),
            (f"FCRM_RV_{gate}_c2", fcrm_rv),
            (f"FCRM_{gate}_c2", fcrm_predictions),
        )
        for variant, forecast in variants:
            record(
                "FCRM",
                variant,
                gate,
                "counterfactual",
                "test",
                test,
                forecast,
                selected_c=2,
                start_index=test_start,
            )
        output.setdefault("counterfactual", {})[gate] = {
            "test_true_regime": true_test.tolist(),
            "FCRM_TRUE": true_setar.tolist(),
            "FCRM_RV": fcrm_rv.tolist(),
            "FCRM": fcrm_predictions.tolist(),
            "estimated_gate_regime": dominant_clusters.tolist(),
            "coefficient_matching": matching,
        }
    return output


def _lstar_diagnostics(models, selected_by_gate, G_true, train_validation, test, test_start):
    """Compare gating-derived soft regime-2 weight with true LSTAR G_t."""
    if G_true is None:
        raise ValueError("E4 must preserve G_true.")
    lag_vectors = rolling_lag_vectors(train_validation, test, n_lags=2)
    true_test = np.asarray(G_true[test_start:], dtype=float)
    output = {"models": {}}
    for gate in FCRM_GATING_METHODS:
        for label, c in (("c_selected", selected_by_gate[gate]), ("c2", 2)):
            model = models[c]
            weights = model.predict_gating_memberships(lag_vectors, method=gate)
            estimated, mapping, distances = gating_transition_estimate(weights, model.coef_)
            output["models"][f"{gate}_{label}"] = {
                "n_clusters": c,
                "coefficients": model.coef_.tolist(),
                "gating_weights": weights.tolist(),
                "estimated_cluster_to_nearest_true_regime": mapping.tolist(),
                "coefficient_distances_to_true_regimes": distances.tolist(),
                "G_true_test": true_test.tolist(),
                "G_estimated_test": estimated.tolist(),
                "correlations": correlations(true_test, estimated),
            }
    return output


def result_path(output_dir, scenario, replication):
    return Path(output_dir) / f"{scenario_slug(scenario)}_rep{replication:04d}.json"


def run_and_save_replication(
    scenario,
    replication,
    *,
    master_seed,
    burn_in,
    output_dir,
    overwrite=False,
):
    """Run one replication and atomically persist it for interruption recovery."""
    output_path = result_path(output_dir, scenario, replication)
    if output_path.exists() and not overwrite:
        with output_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        prior = existing.get("diagnostics", {})
        if prior.get("master_seed") != int(master_seed) or prior.get("burn_in") != int(burn_in):
            raise ValueError(
                f"{output_path} was created with a different master seed or burn-in; "
                "choose a new output directory or explicitly overwrite it."
            )
        return {"replication": replication, "skipped": True, "path": str(output_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate_replication(
        scenario,
        replication,
        master_seed=master_seed,
        burn_in=burn_in,
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.stem}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, output_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {
        "replication": replication,
        "skipped": False,
        "path": str(output_path),
        "metric_rows": len(result["metrics"]),
        "prediction_rows": len(result["predictions"]),
    }
