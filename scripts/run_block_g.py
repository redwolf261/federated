#!/usr/bin/env python3
"""Block G: Mechanism Isolation (Prototype Exchange Causality).

Executes 6 methods × 3 seeds = 18 runs to establish causal link between
prototype exchange and FLEX-Persona's performance gains.

Methods:
- flex_full: Normal prototype extraction + sharing + aggregation
- flex_no_prototype_sharing: No cross-client prototype exchange
- flex_self_only: Server returns client's own prototype only
- flex_shuffled_prototypes: Randomly permuted prototype assignments
- flex_noise_prototypes: Prototypes replaced with random noise
- fedavg_sgd: Baseline reference (FedAvg)

Output: outputs/failure_mode_coverage/G_results.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_failure_mode_coverage import (
    run_fedavg_manual,
    run_flex_simulator,
    COVERAGE_DIR,
)

# Ensure output directory exists
COVERAGE_DIR.mkdir(parents=True, exist_ok=True)

# Fixed configuration per user specification
DATASET_NAME = "cifar10"
NUM_CLASSES = 10
NUM_CLIENTS = 10
ROUNDS = 20
LOCAL_EPOCHS = 5
CLUSTER_AWARE_EPOCHS = 2  # CRITICAL FIX: Enable cluster guidance so prototype exchange is active

BATCH_SIZE = 64
LR = 0.001
MAX_SAMPLES = 20000
ALPHA = 0.1
SEEDS = [42, 43, 44]

METHODS = [
    "flex_full",
    "flex_no_prototype_sharing",
    "flex_self_only",
    "flex_shuffled_prototypes",
    "flex_noise_prototypes",
    "fedavg_sgd",
]


# Use separate file for fixed Block G to avoid mixing with old (broken) results
RESULTS_JSONL = COVERAGE_DIR / "G_results_fixed.jsonl"


def is_run_complete(method: str, seed: int) -> bool:
    """Check if a run already exists in the JSONL file."""
    jsonl_path = RESULTS_JSONL

    if not jsonl_path.exists():
        return False
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("method") == method and entry.get("seed") == seed:
                return True
    return False


def log_run(fields: dict):
    """Append a single run's results to the JSONL file."""
    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(fields) + "\n")



def compute_p10(values: list[float]) -> float:
    """Compute 10th percentile."""
    arr = np.array(values)
    return float(np.percentile(arr, 10))


def run_single(method: str, seed: int) -> dict:
    """Execute a single run and return metrics."""
    print(f"\n    Running: {method} | seed={seed}")

    if method == "fedavg_sgd":
        result = run_fedavg_manual(
            dataset_name=DATASET_NAME,
            num_classes=NUM_CLASSES,
            num_clients=NUM_CLIENTS,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            seed=seed,
            alpha=ALPHA,
            lr=LR,
            batch_size=BATCH_SIZE,
            max_samples=MAX_SAMPLES,
            optimizer_name="adam",
        )
        result["method"] = "fedavg_sgd"
    else:
        # Map method name to ablation_mode
        ablation_mode = method.replace("flex_", "")
        result = run_flex_simulator(
            dataset_name=DATASET_NAME,
            num_classes=NUM_CLASSES,
            num_clients=NUM_CLIENTS,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            cluster_aware_epochs=CLUSTER_AWARE_EPOCHS,
            seed=seed,
            alpha=ALPHA,
            lr=LR,
            batch_size=BATCH_SIZE,
            max_samples=MAX_SAMPLES,
            ablation_mode=ablation_mode,
        )
        result["method"] = method

    result["seed"] = seed
    result["block"] = "G"
    result["p10"] = compute_p10(list(result["client_accuracies"].values()))

    print(f"      mean={result['mean_accuracy']:.4f}  "
          f"worst={result['worst_accuracy']:.4f}  "
          f"std={result['std_across_clients']:.4f}  "
          f"p10={result['p10']:.4f}")

    return result


def validate_alignment_loss() -> bool:
    """Run a quick 1-round test to verify alignment loss is non-zero."""
    print("\n  🔍 VALIDATION: Checking alignment loss magnitude...")

    from flex_persona.config.experiment_config import ExperimentConfig
    from flex_persona.federated.simulator import FederatedSimulator

    cfg = ExperimentConfig(
        experiment_name="block_g_validation",
        dataset_name=DATASET_NAME,
        num_clients=NUM_CLIENTS,
        random_seed=SEEDS[0],
        partition_mode="dirichlet",
        dirichlet_alpha=ALPHA,
    )
    cfg.model.num_classes = NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.aggregation_mode = "prototype"
    cfg.training.rounds = 1
    cfg.training.local_epochs = LOCAL_EPOCHS
    cfg.training.cluster_aware_epochs = CLUSTER_AWARE_EPOCHS
    cfg.training.learning_rate = LR
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.max_samples_per_client = MAX_SAMPLES // NUM_CLIENTS
    cfg.training.ablation_mode = "full"

    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)
    history = sim.run_experiment()

    if not history:
        print("  ❌ ERROR: No history returned from simulator")
        return False

    # Extract alignment loss from the first round's local metrics
    round_state = history[0]
    alignment_losses = []
    for client_id, metrics in round_state.local_metrics.items():
        if "cluster_alignment_loss" in metrics:
            alignment_losses.append(float(metrics["cluster_alignment_loss"]))

    if not alignment_losses:
        print("  ❌ ERROR: No alignment loss found in metrics")
        first_client_metrics = list(round_state.local_metrics.values())[0] if round_state.local_metrics else {}
        print(f"     Available metrics: {list(first_client_metrics.keys())}")
        return False

    mean_alignment_loss = sum(alignment_losses) / len(alignment_losses)
    print(f"  ✅ Alignment loss detected: {mean_alignment_loss:.6f}")
    print(f"     Per-client alignment losses: {[f'{v:.6f}' for v in alignment_losses]}")

    if mean_alignment_loss < 1e-8:
        print("  ❌ ERROR: Alignment loss is approximately zero!")
        print("     Prototype exchange mechanism is NOT active.")
        return False

    print("  ✅ Validation passed — alignment loss is non-zero.")
    return True


def main():
    print("\n" + "█" * 60)

    print("  BLOCK G: MECHANISM ISOLATION")
    print("  Prototype Exchange Causality")
    print("█" * 60)
    print(f"\n  Configuration:")
    print(f"    Dataset: {DATASET_NAME}")
    print(f"    Clients: {NUM_CLIENTS}")
    print(f"    Samples/client: {MAX_SAMPLES // NUM_CLIENTS}")
    print(f"    Partition: Dirichlet (α={ALPHA})")
    print(f"    Rounds: {ROUNDS}")
    print(f"    Local epochs: {LOCAL_EPOCHS}")
    print(f"    Cluster-aware epochs: {CLUSTER_AWARE_EPOCHS}")
    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    Learning rate: {LR}")
    print(f"    Seeds: {SEEDS}")
    print(f"    Methods: {len(METHODS)}")
    print(f"    Total runs: {len(METHODS) * len(SEEDS)}")

    # VALIDATION CHECK: Ensure cluster_aware_epochs is non-zero
    if CLUSTER_AWARE_EPOCHS == 0:
        print("\n  ❌ ERROR: cluster_aware_epochs = 0 — prototype exchange mechanism is INACTIVE!")
        print("     Alignment loss will be zero. Stopping experiment.")
        sys.exit(1)
    else:
        print(f"\n  ✅ VALIDATION: cluster_aware_epochs = {CLUSTER_AWARE_EPOCHS}")
        print(f"     Prototype exchange will be ACTIVE during cluster-aware training.")
        print(f"     Alignment loss should be non-zero.")

    # Run alignment loss validation before full experiment
    if not validate_alignment_loss():
        print("\n  ❌ VALIDATION FAILED — Stopping experiment.")
        sys.exit(1)

    total_start = time.time()

    completed = 0
    skipped = 0

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        for method in METHODS:
            if is_run_complete(method, seed):
                print(f"    SKIP: {method} | seed={seed} (already complete)")
                skipped += 1
                continue

            run_start = time.time()
            result = run_single(method, seed)
            elapsed = time.time() - run_start

            result["wall_time_seconds"] = round(elapsed, 2)
            log_run(result)
            completed += 1

            print(f"      Time: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  BLOCK G COMPLETE")
    print(f"  Completed: {completed} runs")
    print(f"  Skipped:   {skipped} runs")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
