#!/usr/bin/env python3
"""Block H: Mechanism Decomposition (Final Causal Test).

Isolates and quantifies the contribution of:
1. Adapter network
2. Prototype alignment loss
3. Representation geometry

Methods:
- flex_full: Normal system
- flex_no_alignment: lambda_cluster = 0 (alignment loss removed)
- flex_no_adapter: Adapter replaced with identity (backbone → classifier)
- flex_frozen_adapter: Adapter frozen (not trainable)
- flex_random_projection: Adapter replaced with fixed random projection
- flex_noise_alignment: Cluster prototypes replaced with noise
- fedavg_sgd: Baseline

Output: outputs/failure_mode_coverage/block_H_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flex_persona.config.experiment_config import ExperimentConfig
from flex_persona.federated.simulator import FederatedSimulator
from scripts.run_failure_mode_coverage import (
    run_fedavg_manual,
    COVERAGE_DIR,
)
from scripts.phase2_q1_validation import set_seed

# Output paths
RESULTS_JSON = COVERAGE_DIR / "block_H_results.json"
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
MAX_SAMPLES = 20000
ALPHA = 0.1
SEEDS = [42, 43, 44]

METHODS = [
    "flex_full",
    "flex_no_alignment",
    "flex_no_adapter",
    "flex_frozen_adapter",
    "flex_random_projection",
    "flex_noise_alignment",
    "fedavg_sgd",
]


class IdentityAdapter(nn.Module):
    """Adapter that returns input unchanged (no transformation)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.input_dim = dim
        self.shared_dim = dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


def load_existing_results() -> list[dict]:
    """Load existing results from JSON file."""
    if not RESULTS_JSON.exists():
        return []
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: list[dict]) -> None:
    """Save results to JSON file."""
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def is_run_complete(results: list[dict], method: str, seed: int) -> bool:
    """Check if a run already exists in results."""
    for entry in results:
        if entry.get("method") == method and entry.get("seed") == seed:
            return True
    return False


def modify_for_no_adapter(simulator: FederatedSimulator) -> None:
    """Replace adapter with identity for all clients (backbone → classifier directly)."""
    for client in simulator.clients:
        model = client.model
        backbone_dim = model.backbone.output_dim

        # Create identity adapter
        identity = IdentityAdapter(backbone_dim).to(client.device)

        # Replace adapter
        model.adapter = identity

        # Recreate classifier to take backbone_dim input
        old_classifier = model.classifier
        model.classifier = nn.Linear(backbone_dim, model.num_classes).to(client.device)

        # Copy weights if dimensions match, otherwise reinitialize
        if old_classifier.in_features == backbone_dim:
            with torch.no_grad():
                model.classifier.weight.copy_(old_classifier.weight)
                model.classifier.bias.copy_(old_classifier.bias)

        print(f"        [no_adapter] Client {client.client_id}: "
              f"backbone_dim={backbone_dim}, classifier={model.classifier.in_features}→{model.classifier.out_features}")


def modify_for_frozen_adapter(simulator: FederatedSimulator) -> None:
    """Freeze adapter weights for all clients (no gradient updates)."""
    for client in simulator.clients:
        client.model.adapter.requires_grad_(False)
    print("        [frozen_adapter] Adapter frozen for all clients")


def modify_for_random_projection(simulator: FederatedSimulator, seed: int) -> None:
    """Replace adapter with fixed random orthogonal projection."""
    for client in simulator.clients:
        adapter = client.model.adapter
        device = adapter.proj.weight.device
        # Create generator on the same device as the tensor
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        # Initialize with random orthogonal matrix
        nn.init.orthogonal_(adapter.proj.weight, generator=rng)
        nn.init.zeros_(adapter.proj.bias)
        # Freeze so it doesn't learn
        adapter.requires_grad_(False)

    print("        [random_projection] Adapter replaced with random orthogonal projection")


def run_flex_variant(variant: str, seed: int) -> dict:
    """Run a single FLEX variant for Block H."""
    print(f"\n    Running: {variant} | seed={seed}")

    # Base configuration
    cfg = ExperimentConfig(
        experiment_name=f"block_h_{variant}_s{seed}",
        dataset_name=DATASET_NAME,
        num_clients=NUM_CLIENTS,
        random_seed=seed,
        partition_mode="dirichlet",
        dirichlet_alpha=ALPHA,
        output_dir=str(COVERAGE_DIR),
    )
    cfg.model.num_classes = NUM_CLASSES
    cfg.model.client_backbones = ["small_cnn"]
    cfg.training.aggregation_mode = "prototype"
    cfg.training.rounds = ROUNDS
    cfg.training.local_epochs = LOCAL_EPOCHS
    cfg.training.cluster_aware_epochs = CLUSTER_AWARE_EPOCHS
    cfg.training.learning_rate = LR
    cfg.training.batch_size = BATCH_SIZE
    cfg.training.max_samples_per_client = MAX_SAMPLES // NUM_CLIENTS

    # Variant-specific configuration
    if variant == "flex_no_alignment":
        cfg.training.lambda_cluster = 0.0
        cfg.training.lambda_cluster_center = 0.0
        cfg.training.ablation_mode = "full"
    elif variant == "flex_noise_alignment":
        cfg.training.lambda_cluster = 0.1
        cfg.training.lambda_cluster_center = 0.01
        cfg.training.ablation_mode = "noise_prototypes"
    else:
        cfg.training.lambda_cluster = 0.1
        cfg.training.lambda_cluster_center = 0.01
        cfg.training.ablation_mode = "full"

    # Special handling for flex_no_adapter: set shared_dim to backbone output
    if variant == "flex_no_adapter":
        # SmallCNN outputs 128-dim features
        cfg.model.shared_dim = 128

    # Create simulator
    sim = FederatedSimulator(workspace_root=str(PROJECT_ROOT), config=cfg)

    # Apply architectural modifications post-creation
    if variant == "flex_no_adapter":
        modify_for_no_adapter(sim)
    elif variant == "flex_frozen_adapter":
        modify_for_frozen_adapter(sim)
    elif variant == "flex_random_projection":
        modify_for_random_projection(sim, seed)

    # Run experiment
    history = sim.run_experiment()

    # Collect metrics
    client_accs = {c.client_id: c.evaluate_accuracy() for c in sim.clients}
    vals = list(client_accs.values())

    # Extract alignment loss from final round if available
    alignment_loss = None
    if history:
        final_state = history[-1]
        alignments = []
        for client_id, metrics in final_state.local_metrics.items():
            if "cluster_alignment_loss" in metrics:
                alignments.append(metrics["cluster_alignment_loss"])
        if alignments:
            alignment_loss = float(np.mean(alignments))

    result = {
        "method": variant,
        "seed": seed,
        "block": "H",
        "mean_accuracy": float(np.mean(vals)),
        "worst_accuracy": float(min(vals)),
        "std_across_clients": float(np.std(vals)),
        "p10": float(np.percentile(vals, 10)),
        "client_accuracies": {str(k): float(v) for k, v in client_accs.items()},
        "alignment_loss": alignment_loss,
        "config": {
            "lambda_cluster": cfg.training.lambda_cluster,
            "lambda_cluster_center": cfg.training.lambda_cluster_center,
            "cluster_aware_epochs": cfg.training.cluster_aware_epochs,
            "shared_dim": cfg.model.shared_dim,
        },
    }

    print(f"      mean={result['mean_accuracy']:.4f}  "
          f"worst={result['worst_accuracy']:.4f}  "
          f"std={result['std_across_clients']:.4f}  "
          f"p10={result['p10']:.4f}")
    if alignment_loss is not None:
        print(f"      alignment_loss={alignment_loss:.6f}")

    return result


def run_single(method: str, seed: int) -> dict:
    """Execute a single run."""
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
        result["seed"] = seed
        result["block"] = "H"
        result["p10"] = compute_p10(list(result["client_accuracies"].values()))
        print(f"      mean={result['mean_accuracy']:.4f}  "
              f"worst={result['worst_accuracy']:.4f}  "
              f"std={result['std_across_clients']:.4f}  "
              f"p10={result['p10']:.4f}")
    else:
        result = run_flex_variant(method, seed)

    return result


def compute_p10(values: list[float]) -> float:
    """Compute 10th percentile."""
    arr = np.array(values)
    return float(np.percentile(arr, 10))


def main():
    print("\n" + "█" * 70)
    print("  BLOCK H: MECHANISM DECOMPOSITION (FINAL CAUSAL TEST)")
    print("  Isolating adapter, alignment loss, and representation geometry")
    print("█" * 70)
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

    # Load existing results for resume support
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
            result = run_single(method, seed)
            result["wall_time_seconds"] = round(time.time() - run_start, 2)

            results.append(result)
            save_results(results)
            completed += 1

            print(f"      Time: {result['wall_time_seconds']:.1f}s")

    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"  BLOCK H COMPLETE")
    print(f"  Completed: {completed} runs")
    print(f"  Skipped:   {skipped} runs")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}")

    # Print summary table
    print_summary(results)


def print_summary(results: list[dict]) -> None:
    """Print aggregated summary table."""
    from collections import defaultdict

    aggregated: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        aggregated[r["method"]].append(r)

    print(f"\n{'='*70}")
    print("  BLOCK H SUMMARY (Aggregated across seeds)")
    print(f"  {'Method':<25} {'Mean':>8} {'Std':>8} {'Worst':>8} {'P10':>8}")
    print(f"  {'-'*60}")

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

        print(f"  {method:<25} {mean_acc:>8.4f} {std_acc:>8.4f} {mean_worst:>8.4f} {mean_p10:>8.4f}"
              f"  (drop={drop:+.4f})")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
