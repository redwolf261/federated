#!/usr/bin/env python3
"""
Quick audit: Run the 4 critical validation points.

1. Centralized
2. FedAvg (1-client)
3. MOON (μ=0)
4. SCAFFOLD_zero
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add parent to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2_q1_validation import (
    train_centralized, run_fedavg_dirichlet, run_moon, run_scaffold,
    FEMNIST_NUM_CLASSES, set_seed
)

def main():
    print("\n" + "="*70)
    print("  QUICK AUDIT: 4-Point Validation")
    print("="*70 + "\n")

    seed = 42
    num_clients = 1
    rounds = 20
    local_epochs = 3
    alpha = 0.5

    results = {}

    # 1. Centralized
    print("[ 1 / 4 ] Centralized Training...")
    cent_acc = train_centralized("femnist", seed=seed)
    results["centralized"] = cent_acc
    print(f"  ✓ Centralized: {cent_acc:.4f}\n")

    # 2. FedAvg (1-client)
    print("[ 2 / 4 ] FedAvg (1-client equivalence)...")
    fedavg_r = run_fedavg_dirichlet(
        alpha=alpha, num_clients=num_clients, rounds=rounds,
        local_epochs=local_epochs, seed=seed, lr=0.003
    )
    fedavg_acc = fedavg_r["mean_accuracy"]
    results["fedavg_1client"] = fedavg_acc
    delta_fedavg = abs(fedavg_acc - cent_acc)
    print(f"  ✓ FedAvg (1-client): {fedavg_acc:.4f}")
    print(f"    Δ vs Centralized: {delta_fedavg:.4f} ({100*delta_fedavg:.2f}%)\n")

    # 3. MOON (μ=0)
    print("[ 3 / 4 ] MOON (μ=0, contrastive disabled)...")
    moon_r = run_moon(
        dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES,
        num_clients=num_clients, rounds=rounds, local_epochs=local_epochs,
        seed=seed, alpha=alpha, mu=0.0, lr=0.003
    )
    moon_acc = moon_r["mean_accuracy"]
    results["moon_mu0"] = moon_acc
    delta_moon = abs(moon_acc - fedavg_acc)
    print(f"  ✓ MOON (μ=0): {moon_acc:.4f}")
    print(f"    Δ vs FedAvg: {delta_moon:.4f} ({100*delta_moon:.2f}%)")
    print(f"    Δ vs Centralized: {abs(moon_acc - cent_acc):.4f} ({100*abs(moon_acc - cent_acc):.2f}%)\n")

    # 4. SCAFFOLD (zero control variates)
    print("[ 4 / 4 ] SCAFFOLD (control variates zero)...")
    scaffold_r = run_scaffold(
        dataset_name="femnist", num_classes=FEMNIST_NUM_CLASSES,
        num_clients=num_clients, rounds=rounds, local_epochs=local_epochs,
        seed=seed, alpha=alpha, lr=0.003
    )
    scaffold_acc = scaffold_r["mean_accuracy"]
    results["scaffold_zero"] = scaffold_acc
    delta_scaffold = abs(scaffold_acc - fedavg_acc)
    print(f"  ✓ SCAFFOLD (zero): {scaffold_acc:.4f}")
    print(f"    Δ vs FedAvg: {delta_scaffold:.4f} ({100*delta_scaffold:.2f}%)")
    print(f"    Δ vs Centralized: {abs(scaffold_acc - cent_acc):.4f} ({100*abs(scaffold_acc - cent_acc):.2f}%)\n")

    # Summary table
    print("="*70)
    print("  SUMMARY TABLE")
    print("="*70)
    print(f"  Centralized:        {cent_acc:.4f}")
    print(f"  FedAvg (1-client):   {fedavg_acc:.4f}  (Δ = {delta_fedavg:.4f})")
    print(f"  MOON (μ=0):          {moon_acc:.4f}  (Δ = {delta_moon:.4f})")
    print(f"  SCAFFOLD (zero):     {scaffold_acc:.4f}  (Δ = {delta_scaffold:.4f})")
    print("="*70)

    # Verdict
    print("\n  VERDICT:")
    if delta_fedavg < 0.02:
        print(f"  ✅ 1-client equivalence: PASS (<2% gap)")
    elif delta_fedavg < 0.04:
        print(f"  ⚠️  1-client equivalence: BORDERLINE (2-4% gap)")
    else:
        print(f"  ❌ 1-client equivalence: FAIL (>4% gap)")

    if delta_moon < 0.005:
        print(f"  ✅ MOON(μ=0) ≈ FedAvg: PASS (<0.5% gap)")
    elif delta_moon < 0.01:
        print(f"  ✅ MOON(μ=0) ≈ FedAvg: GOOD (<1% gap)")
    elif delta_moon < 0.02:
        print(f"  ⚠️  MOON(μ=0) ≈ FedAvg: BORDERLINE (<2% gap)")
    else:
        print(f"  ❌ MOON(μ=0) ≈ FedAvg: FAIL (>2% gap)")

    if delta_scaffold < 0.01:
        print(f"  ✅ SCAFFOLD(zero) ≈ FedAvg: PASS (<1% gap)")
    elif delta_scaffold < 0.02:
        print(f"  ✅ SCAFFOLD(zero) ≈ FedAvg: GOOD (<2% gap)")
    else:
        print(f"  ⚠️  SCAFFOLD(zero) ≈ FedAvg: BORDERLINE (>2% gap)")

    print("\n" + "="*70 + "\n")

    # Save results
    output_file = Path(__file__).parent.parent / "outputs" / "quick_audit_four.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}\n")

if __name__ == "__main__":
    main()
