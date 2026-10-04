"""Run selected scenarios from the reproducible fuzzy-TS simulation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from simulation.config import (  # noqa: E402
    DEFAULT_BURN_IN,
    DEFAULT_MASTER_SEED,
    DGP_NAMES,
    GAMMA_VALUES,
    NOISE_STDS,
    SAMPLE_SIZES,
    Scenario,
)
from simulation.experiment import run_and_save_replication  # noqa: E402


def _parse_values(raw, choices, cast, name):
    if raw.lower() == "all":
        return tuple(choices)
    try:
        values = tuple(cast(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid {name}: {raw}") from error
    if not values or any(value not in choices for value in values):
        allowed = ", ".join(str(value) for value in choices)
        raise argparse.ArgumentTypeError(f"{name} must be selected from {allowed}, or be 'all'.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} contains duplicate values.")
    return values


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgp", default="E1", help="E1, E2, E3, E4, or all.")
    parser.add_argument("--T", default="50", help="50, 100, 500, or all.")
    parser.add_argument("--sigma", default="0.25", help="0.25, 0.50, or all.")
    parser.add_argument("--gamma", default="all", help="E4 gamma (1, 5, 20), comma list, or all.")
    parser.add_argument("--replications", type=int, default=1, help="Number of one-based replications.")
    parser.add_argument("--start-replicate", type=int, default=1)
    parser.add_argument("--end-replicate", type=int, help="Optional inclusive end; overrides replications.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Worker processes (replication is the unit).")
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED, help="Master RNG seed.")
    parser.add_argument("--burn-in", type=int, default=DEFAULT_BURN_IN)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "results" / "raw")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing replication files.")
    args = parser.parse_args(argv)

    try:
        args.dgps = _parse_values(args.dgp, DGP_NAMES, str, "dgp")
        args.sample_sizes = _parse_values(args.T, SAMPLE_SIZES, int, "T")
        args.noise_stds = _parse_values(args.sigma, NOISE_STDS, float, "sigma")
        args.gammas = _parse_values(args.gamma, GAMMA_VALUES, float, "gamma")
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.start_replicate < 1:
        parser.error("--start-replicate must be >= 1.")
    if args.end_replicate is not None:
        if args.end_replicate < args.start_replicate:
            parser.error("--end-replicate must be >= --start-replicate.")
        end_replicate = args.end_replicate
    else:
        if args.replications < 1:
            parser.error("--replications must be >= 1.")
        end_replicate = args.start_replicate + args.replications - 1
    if args.n_jobs < 1:
        parser.error("--n-jobs must be >= 1.")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.burn_in < 0:
        parser.error("--burn-in must be non-negative.")
    args.end_index = end_replicate
    args.output_dir = args.output_dir.resolve()
    return args


def build_scenarios(args):
    scenarios = []
    for dgp in args.dgps:
        for T in args.sample_sizes:
            for sigma in args.noise_stds:
                gamma_values = args.gammas if dgp == "E4" else (None,)
                scenarios.extend(Scenario(dgp, T, sigma, gamma) for gamma in gamma_values)
    return scenarios


def _environment_metadata():
    packages = {}
    for package in ("numpy", "scipy", "scikit-fuzzy", "scikit-learn", "pandas", "statsmodels"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
            check=False, capture_output=True, text=True,
        ).stdout.strip() or None
    except OSError:
        commit = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "git_commit": commit,
    }


def _write_metadata(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _scenario_dict(scenario):
    return {"dgp": scenario.dgp, "T": scenario.T, "sigma": scenario.sigma, "gamma": scenario.gamma}


def run(args):
    scenarios = build_scenarios(args)
    replications = range(args.start_replicate, args.end_index + 1)
    tasks = [
        (scenario, replication)
        for scenario in scenarios
        for replication in replications
    ]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir.parent / "experiment_metadata.json"
    now = datetime.now(timezone.utc).isoformat()
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("master_seed") != args.seed:
            raise ValueError(
                f"{metadata_path} records a different master seed; use a new output directory."
            )
        if metadata.get("burn_in") != args.burn_in:
            raise ValueError(
                f"{metadata_path} records a different burn-in; use a new output directory."
            )
    else:
        metadata = {
            "created_at_utc": now,
            "master_seed": args.seed,
            "burn_in": args.burn_in,
            "design": {
                "DGPs": list(DGP_NAMES),
                "T": list(SAMPLE_SIZES),
                "sigma_is_standard_deviation": True,
                "sigma": list(NOISE_STDS),
                "gamma_E4": list(GAMMA_VALUES),
                "m": 2.0,
                "replications_per_scenario": 200,
                "split": [0.6, 0.2, 0.2],
                "cluster_candidates": [2, 3, 4, 5, 6],
                "FCRM_gating": ["fuzzy_centers", "multinomial"],
            },
            "environment": _environment_metadata(),
            "runs": [],
        }
    run_record = {
        "started_at_utc": now,
        "status": "running",
        "selection": [_scenario_dict(scenario) for scenario in scenarios],
        "start_replicate": args.start_replicate,
        "end_replicate": args.end_index,
        "n_jobs": args.n_jobs,
        "output_dir": str(output_dir),
        "overwrite": bool(args.overwrite),
    }
    metadata["runs"].append(run_record)
    _write_metadata(metadata_path, metadata)

    work = [
        {
            "scenario": scenario,
            "replication": replication,
            "master_seed": args.seed,
            "burn_in": args.burn_in,
            "output_dir": output_dir,
            "overwrite": args.overwrite,
        }
        for scenario, replication in tasks
    ]
    started = time.perf_counter()
    completed = 0
    try:
        if args.n_jobs == 1:
            results = []
            for item in work:
                results.append(run_and_save_replication(**item))
                completed += 1
                print(
                    f"[{completed}/{len(work)}] {item['scenario'].dgp} "
                    f"T={item['scenario'].T} sigma={item['scenario'].sigma} "
                    f"gamma={item['scenario'].gamma} rep={item['replication']} "
                    f"{'skipped' if results[-1]['skipped'] else 'saved'}",
                    flush=True,
                )
        else:
            results = []
            with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
                futures = {executor.submit(run_and_save_replication, **item): item for item in work}
                for future in as_completed(futures):
                    item = futures[future]
                    results.append(future.result())
                    completed += 1
                    print(
                        f"[{completed}/{len(work)}] {item['scenario'].dgp} "
                        f"T={item['scenario'].T} sigma={item['scenario'].sigma} "
                        f"gamma={item['scenario'].gamma} rep={item['replication']} "
                        f"{'skipped' if results[-1]['skipped'] else 'saved'}",
                        flush=True,
                    )
    except Exception as error:
        run_record["status"] = "failed"
        run_record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_record["completed_tasks"] = completed
        run_record["error"] = f"{type(error).__name__}: {error}"
        _write_metadata(metadata_path, metadata)
        raise

    elapsed = time.perf_counter() - started
    run_record["status"] = "complete"
    run_record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_record["completed_tasks"] = completed
    run_record["elapsed_seconds"] = elapsed
    run_record["skipped_existing"] = sum(result["skipped"] for result in results)
    _write_metadata(metadata_path, metadata)
    print(f"Finished {completed} replication tasks in {elapsed:.1f} seconds.")
    print(f"Raw files: {output_dir}")
    print(f"Metadata: {metadata_path}")


def main(argv=None):
    run(parse_arguments(argv))


if __name__ == "__main__":
    main()
