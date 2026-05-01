#!/usr/bin/env python3
"""Block I: Signal Nature Analysis.

Tests the nature of the training signal to identify the true property responsible for FLEX's gains:
1. class_centroid_alignment - Align to per-class centroids from client data
2. global_centroid_alignment - Align to single global centroid
3. random_centroid_alignment - Align to random fixed centroids
4. feature_norm_only - L2 normalize features
5. variance_minimization - Minimize intra-batch feature variance
6. flex_full (reference) - Standard cluster prototype alignment
7. fedavg_sgd (baseline)

Output: outputs/failure_mode_coverage/block_I_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from scripts.run_failure_mode_coverage import COVERAGE_DIR

from scripts.phase2_q1_validation import set_seed

# Output paths
RESULTS_JSON = COVERAGE_DIR / "block_I_results.json"
COVERAGE_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
DATASET_NAME = "cifar10"
NUM_CLASSES = 10
NUM_CLIENTS = 10
ROUNDS = 20
LOCAL_EPOCHS = 5
CLUSTER_AWARE_EPOCHS = 2
BATCH_SIZE = 64
LR = 0.001
MAX_SAMPLES = 2000
ALPHA = 0.1
SEEDS = [42, 43, 44]

METHODS = [
    "flex_full",
    "class_centroid_alignment",
    "global_centroid_alignment",
    "random_centroid_alignment",
    "feature_norm_only",
    "variance_minimization",
    "fedavg_sgd",
]


def build_config(method: str, seed: int) -> ExperimentConfig:
    """Build experiment config for a given method and seed."""
    cfg = ExperimentConfig()
    cfg.experiment_name = f"block_i_{method}_s{seed}"
    cfg.dataset_name = DATASET_NAME
    cfg.num_clients = NUM_CLIENTS
    cfg.random_seed = seed
    cfg.partition_mode = "dirichlet"
    cfg.dirichlet_alpha = ALPHA
    cfg.model.num_classes = NUM_CLASSES
    cfg.training.rounds = ROUNDS
    cfg.training.local_epochs = LOCAL_EPOCHS
    cfg.training.cluster_aware_epochs = CLUSTER_AWARE_EPOCHS
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.learning_rate = LR
    cfg.training.max_samples_per_client = MAX_SAMPLES
    cfg.clustering.num_clusters = 5
    cfg.clustering.random_state = seed

    if method == "flex_full":
        cfg.training.alignment_mode = "cluster_prototype"
    elif method == "class_centroid_alignment":
        cfg.training.alignment_mode = "class_centroid"
    elif method == "global_centroid_alignment":
        cfg.training.alignment_mode = "global_centroid"
    elif method == "random_centroid_alignment":
        cfg.training.alignment_mode = "random_centroid"
    elif method == "feature_norm_only":
        cfg.training.alignment_mode = "feature_norm"
    elif method == "variance_minimization":
        cfg.training.alignment_mode = "variance_min"
    elif method == "fedavg_sgd":
        cfg.training.aggregation_mode = "fedavg"
    else:
        raise ValueError(f"Unknown method: {method}")

    cfg.validate()
    return cfg



def run_single_run(method: str, seed: int) -> dict[str, object]:

    """Run a single experiment for Block I."""
    print(f"\n  Running: {method} | seed={seed}")

    try:
        cfg = build_config(method, seed)
        set_seed(seed)
        simulator = FederatedSimulator(workspace_root=PROJECT_ROOT, config=cfg)
        history = simulator.run_experiment()
        report = simulator.build_report(history)

        final_eval = report.get("final_round_metrics", {})
        if not isinstance(final_eval, dict):
            final_eval = {}
        mean_acc = float(final_eval.get("mean_client_accuracy", 0.0))
        worst_acc = float(final_eval.get("worst_client_accuracy", 0.0))
        p10_acc = float(final_eval.get("p10_client_accuracy", 0.0))
        std = float(final_eval.get("std_client_accuracy", 0.0))

        run_summary = report.get("run_summary", {})
        if not isinstance(run_summary, dict):
            run_summary = {}

        result = {
            "method": method,
            "seed": seed,
            "mean_accuracy": mean_acc,
            "worst_accuracy": worst_acc,
            "p10": p10_acc,
            "std": std,
            "run_status": run_summary.get("run_status", "UNKNOWN"),
        }

        print(f"    RESULT: mean={mean_acc:.4f} worst={worst_acc:.4f} p10={p10_acc:.4f}")
        return result

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"    ERROR: {type(exc).__name__}: {exc}")
        return {
            "method": method,
            "seed": seed,
            "mean_accuracy": 0.0,
            "worst_accuracy": 0.0,
            "p10": 0.0,
            "std": 0.0,
            "run_status": f"FAIL: {type(exc).__name__}: {exc}",
        }


def load_existing_results() -> list[dict]:
    """Load existing results from JSON file."""
    if not RESULTS_JSON.exists():
        return []
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def is_run_complete(results: list[dict], method: str, seed: int) -> bool:
    """Check if a specific run is already complete."""
    for r in results:
        if r.get("method") == method and r.get("seed") == seed:
            return True
    return False


def save_results(results: list[dict]) -> None:
    """Save results to JSON file."""
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    print("=" * 70)
    print("  BLOCK I: Signal Nature Analysis")
    print("  Goal: Identify the TRUE PROPERTY behind FLEX's performance gain")
    print("=" * 70)

    results = load_existing_results()
    print(f"\n  Loaded {len(results)} existing results")

    total_start = time.time()
    completed = 0
    skipped = 0

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        for method in METHODS:
            if is_run_complete(results, method, seed):
                print(f"    SKIP: {method} | seed={seed} (already complete)")
                skipped += 1
                continue

            run_start = time.time()
            result = run_single_run(method, seed)
            result["wall_time_seconds"] = round(time.time() - run_start, 2)

            results.append(result)
            save_results(results)
            completed += 1

            print(f"      Time: {result['wall_time_seconds']:.1f}s")

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 70}")
    print(f"  BLOCK I COMPLETE")
    print(f"  Completed: {completed} runs")
    print(f"  Skipped:   {skipped} runs")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Results: {RESULTS_JSON}")
    print(f"{'=' * 70}")

    # Print summary table
    print_summary(results)


def print_summary(results: list[dict]) -> None:
    """Print aggregated summary table."""
    from collections import defaultdict

    aggregated: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        aggregated[r["method"]].append(r)

    print(f"\n{'=' * 70}")
    print("  BLOCK I SUMMARY (Aggregated across seeds)")
    print(f"  {'Method':<30} {'Mean':>8} {'Std':>8} {'Worst':>8} {'P10':>8}")
    print(f"  {'-' * 60}")

    # Get flex_full reference
    flex_full_means = [r["mean_accuracy"] for r in aggregated.get("flex_full", [])]
    flex_full_mean = np.mean(flex_full_means) if flex_full_means else 0.0

    for method in METHODS:
        runs = aggregated[method]
        if not runs:
            continue
        means = [r["mean_accuracy"] for r in runs]
        worsts = [r["worst_accuracy"] for r in runs]
        p10s = [r["p10"] for r in runs]
        mean_acc = np.mean(means)

        std_acc = np.std(means)
        mean_worst = np.mean(worsts)
        mean_p10 = np.mean(p10s)
        drop = flex_full_mean - mean_acc

        print(
            f"  {method:<30} {mean_acc:>8.4f} {std_acc:>8.4f} {mean_worst:>8.4f} {mean_p10:>8.4f}"
            f"  (drop={drop:+.4f})"
        )

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
