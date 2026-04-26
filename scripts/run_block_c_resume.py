#!/usr/bin/env python3
"""Resume Block C: run only missing combinations."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_failure_mode_coverage import (
    COVERAGE_DIR,
    log_run,
    run_fedavg_manual,
    run_flex_simulator,
)


def compute_p10(accuracies: list[float]) -> float:
    if not accuracies:
        return 0.0
    return float(np.percentile(accuracies, 10))


def main():
    print("\n" + "█" * 70)
    print("  BLOCK C: RESUMING MISSING RUNS")
    print("█" * 70)

    # Read existing results
    c_results_path = COVERAGE_DIR / "C_results.jsonl"
    existing = set()
    if c_results_path.exists():
        with open(c_results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    key = (r.get("samples_per_client"), r.get("seed"), r.get("method"))
                    existing.add(key)
                except json.JSONDecodeError:
                    continue

    print(f"  Found {len(existing)} existing runs in C_results.jsonl")

    dataset_name = "cifar10"
    num_classes = 10
    num_clients = 10
    rounds = 20
    local_epochs = 5
    batch_size = 64
    lr = 0.001
    alpha = 0.1

    sample_sizes = [2000, 5000, 10000]
    seeds = [42, 43, 44]

    todo = []
    for samples_per_client in sample_sizes:
        max_samples = samples_per_client * num_clients
        for seed in seeds:
            for method in ["flex_no_extra", "fedavg_sgd"]:
                key = (samples_per_client, seed, method)
                if key not in existing:
                    todo.append((samples_per_client, max_samples, seed, method))

    print(f"  {len(todo)} runs remaining")

    if not todo:
        print("  All runs complete!")
        return

    total = len(todo)
    for idx, (samples_per_client, max_samples, seed, method) in enumerate(todo, 1):
        print(f"\n{'─'*70}")
        print(f"  Run {idx}/{total} | samples={samples_per_client} | seed={seed} | method={method}")
        print(f"{'─'*70}")

        if method == "flex_no_extra":
            print(f"\n  [FLEX] Starting...")
            result = run_flex_simulator(
                dataset_name=dataset_name,
                num_classes=num_classes,
                num_clients=num_clients,
                rounds=rounds,
                local_epochs=local_epochs,
                cluster_aware_epochs=0,
                seed=seed,
                alpha=alpha,
                lr=lr,
                batch_size=batch_size,
                max_samples=max_samples,
            )
            result["block"] = "C"
            result["method"] = "flex_no_extra"
        else:
            print(f"\n  [FedAvg] Starting...")
            result = run_fedavg_manual(
                dataset_name=dataset_name,
                num_classes=num_classes,
                num_clients=num_clients,
                rounds=rounds,
                local_epochs=local_epochs,
                seed=seed,
                alpha=alpha,
                lr=lr,
                batch_size=batch_size,
                max_samples=max_samples,
                optimizer_name="sgd",
            )
            result["block"] = "C"
            result["method"] = "fedavg_sgd"

        result["samples_per_client"] = samples_per_client
        result["seed"] = seed
        acc_values = list(result["client_accuracies"].values())
        result["p10_accuracy"] = compute_p10(acc_values)
        log_run(result)

        print(f"      mean={result['mean_accuracy']:.4f}  "
              f"worst={result['worst_accuracy']:.4f}  "
              f"std={result['std_across_clients']:.4f}  "
              f"p10={result['p10_accuracy']:.4f}")

    print(f"\n{'█'*70}")
    print("  RESUME COMPLETE")
    print(f"{'█'*70}")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
