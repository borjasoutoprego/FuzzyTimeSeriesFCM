"""Descriptive summaries, average ranks, Friedman, Nemenyi, and boxplots."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, studentized_range, t


SCENARIO_KEYS = ("dgp", "T", "sigma", "gamma")
METRICS = ("mae", "rmse", "smape")


def _scenario_key(row):
    return (row["dgp"], int(row["T"]), float(row["sigma"]), row.get("gamma"))


def _scenario_label(key):
    dgp, T, sigma, gamma = key
    gamma_part = f"_gamma{gamma:g}" if gamma not in (None, "") else ""
    return f"{dgp}_T{T}_sigma{sigma:g}{gamma_part}"


def read_core_metrics(input_dir):
    """Read only final test metrics for core procedures from replication JSON."""
    records = []
    for path in sorted(Path(input_dir).glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        for row in result.get("metrics", []):
            if row.get("evaluation_split") != "test" or row.get("result_group") != "core":
                continue
            records.append(row)
    if not records:
        raise ValueError(f"no core test metrics found under {input_dir}.")
    return records


def descriptive_summary(records):
    groups = defaultdict(list)
    for row in records:
        key = (*_scenario_key(row), row["method_variant"])
        for metric in METRICS:
            groups[(*key, metric)].append(float(row[metric]))
    output = []
    for group, values in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        dgp, T, sigma, gamma, method_variant, metric = group
        array = np.asarray(values, dtype=float)
        n = array.size
        sem = float(np.std(array, ddof=1) / math.sqrt(n)) if n > 1 else None
        critical = float(t.ppf(0.975, n - 1)) if n > 1 else None
        half_width = critical * sem if n > 1 else None
        output.append({
            "dgp": dgp,
            "T": T,
            "sigma": sigma,
            "gamma": gamma,
            "method_variant": method_variant,
            "metric": metric,
            "n": int(n),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "std": float(np.std(array, ddof=1)) if n > 1 else None,
            "ci95_low": float(np.mean(array) - half_width) if half_width is not None else None,
            "ci95_high": float(np.mean(array) + half_width) if half_width is not None else None,
        })
    return output


def average_ranks(records):
    grouped = defaultdict(dict)
    for row in records:
        scenario = _scenario_key(row)
        key = (scenario, int(row["replication"]))
        for metric in METRICS:
            grouped[(scenario, metric, int(row["replication"]))][row["method_variant"]] = float(row[metric])

    ranks = defaultdict(list)
    counts = defaultdict(int)
    variants_by_group = defaultdict(set)
    for (scenario, metric, replication), values in grouped.items():
        variants_by_group[(scenario, metric)].update(values)
    for (scenario, metric, replication), values in grouped.items():
        variants = sorted(variants_by_group[(scenario, metric)])
        if set(values) != set(variants):
            raise ValueError(
                f"incomplete method metrics for {_scenario_label(scenario)}, "
                f"{metric}, replication {replication}."
            )
        rank_values = rankdata([values[name] for name in variants], method="average")
        for name, rank in zip(variants, rank_values):
            key = (scenario, metric, name)
            ranks[key].append(float(rank))
            counts[key] += 1
    output = []
    for (scenario, metric, variant), values in sorted(
        ranks.items(), key=lambda item: tuple(str(v) for v in item[0])
    ):
        output.append({
            "dgp": scenario[0],
            "T": scenario[1],
            "sigma": scenario[2],
            "gamma": scenario[3],
            "method_variant": variant,
            "metric": metric,
            "n": counts[(scenario, metric, variant)],
            "average_rank": float(np.mean(values)),
        })
    return output


def selection_frequencies(records):
    """Count selected c, L, and joint (c,L) choices across replications."""
    counts = defaultdict(int)
    totals = defaultdict(set)
    for row in records:
        group = (*_scenario_key(row), row["method_variant"])
        replication = int(row["replication"])
        for parameter in ("selected_c", "selected_L"):
            value = row.get(parameter)
            if value is None:
                continue
            counts[(*group, parameter, str(value))] += 1
            totals[(*group, parameter)].add(replication)
        if row.get("selected_c") is not None and row.get("selected_L") is not None:
            value = f"{row['selected_c']},{row['selected_L']}"
            counts[(*group, "selected_c_L", value)] += 1
            totals[(*group, "selected_c_L")].add(replication)
    output = []
    for key, count in sorted(counts.items(), key=lambda item: tuple(str(v) for v in item[0])):
        dgp, T, sigma, gamma, variant, parameter, value = key
        n = len(totals[key[:-1]])
        output.append({
            "dgp": dgp,
            "T": T,
            "sigma": sigma,
            "gamma": gamma,
            "method_variant": variant,
            "parameter": parameter,
            "value": value,
            "count": count,
            "n": n,
            "proportion": count / n if n else None,
        })
    return output


def friedman_and_nemenyi(records, alpha=0.05):
    """Test paired procedure ranks independently within each design scenario."""
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in records:
        scenario = _scenario_key(row)
        for metric in METRICS:
            grouped[(scenario, metric)][int(row["replication"])][row["method_variant"]] = float(row[metric])

    friedman_rows = []
    nemenyi_rows = []
    for (scenario, metric), replications in sorted(
        grouped.items(), key=lambda item: tuple(str(v) for v in item[0])
    ):
        variants = sorted(set.union(*(set(values) for values in replications.values())))
        complete = [
            (rep, values) for rep, values in sorted(replications.items())
            if all(name in values for name in variants)
        ]
        if len(complete) != len(replications):
            raise ValueError(f"incomplete paired method metrics for {_scenario_label(scenario)} / {metric}.")
        if len(variants) < 3 or len(complete) < 2:
            continue
        samples = [[values[name] for _, values in complete] for name in variants]
        statistic, p_value = friedmanchisquare(*samples)
        friedman_rows.append({
            "dgp": scenario[0],
            "T": scenario[1],
            "sigma": scenario[2],
            "gamma": scenario[3],
            "metric": metric,
            "n": len(complete),
            "k": len(variants),
            "statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": alpha,
            "significant": bool(p_value < alpha),
        })
        if p_value >= alpha:
            continue

        rank_matrix = np.vstack([
            rankdata([values[name] for name in variants], method="average")
            for _, values in complete
        ])
        means = np.mean(rank_matrix, axis=0)
        denominator = math.sqrt(len(variants) * (len(variants) + 1) / (6.0 * len(complete)))
        for left, right in combinations(range(len(variants)), 2):
            q_statistic = abs(means[left] - means[right]) / denominator
            pair_p = float(studentized_range.sf(q_statistic * math.sqrt(2.0), len(variants), np.inf))
            nemenyi_rows.append({
                "dgp": scenario[0],
                "T": scenario[1],
                "sigma": scenario[2],
                "gamma": scenario[3],
                "metric": metric,
                "n": len(complete),
                "method_a": variants[left],
                "method_b": variants[right],
                "mean_rank_a": float(means[left]),
                "mean_rank_b": float(means[right]),
                "q_statistic": float(q_statistic),
                "p_value": pair_p,
                "alpha": alpha,
                "significant": bool(pair_p < alpha),
            })
    return friedman_rows, nemenyi_rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_boxplots(records, output_dir):
    """Write one observed test-error boxplot per scenario and metric."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("boxplots require the already-used matplotlib package.") from error

    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        scenario = _scenario_key(row)
        for metric in METRICS:
            groups[(scenario, metric)][row["method_variant"]].append(float(row[metric]))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for (scenario, metric), values in groups.items():
        names = sorted(values)
        fig, axis = plt.subplots(figsize=(max(8, len(names) * 1.25), 5))
        axis.boxplot([values[name] for name in names], labels=names, showfliers=True)
        axis.set_title(f"{_scenario_label(scenario)} — {metric.upper()}")
        axis.set_ylabel(metric.upper())
        axis.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(output_dir / f"{_scenario_label(scenario)}_{metric}.png", dpi=150)
        plt.close(fig)


def analyze(input_dir, output_dir, *, make_plots=True, alpha=0.05):
    records = read_core_metrics(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = descriptive_summary(records)
    ranks = average_ranks(records)
    frequencies = selection_frequencies(records)
    friedman, nemenyi = friedman_and_nemenyi(records, alpha=alpha)
    write_csv(output_dir / "descriptive_summary.csv", summary)
    write_csv(output_dir / "average_ranks.csv", ranks)
    write_csv(output_dir / "selection_frequencies.csv", frequencies)
    write_csv(output_dir / "friedman.csv", friedman)
    write_csv(output_dir / "nemenyi_posthoc.csv", nemenyi)
    if make_plots:
        write_boxplots(records, output_dir / "figures")
    return {
        "replication_rows": len(records),
        "summary_rows": len(summary),
        "rank_rows": len(ranks),
        "selection_frequency_rows": len(frequencies),
        "friedman_tests": len(friedman),
        "nemenyi_comparisons": len(nemenyi),
    }
