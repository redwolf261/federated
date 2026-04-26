#!/usr/bin/env python3
"""
SCAFFOLD LR sensitivity test to explain poor performance.
Tests multiple learning rates to check if SCAFFOLD is hyperparameter-sensitive.
Seed 42, alpha=0.1, 10 rounds (fast).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import time
import numpy as np
from scripts.phase2_q1_validation import run_scaffold, set_seed

OUTPUT_DIR = Path("outputs/locked_cifar10_grid")
RUNS_DIR = OUTPUT_DIR / "runs"

# Test these learning rates for SCAFFOLD
LR_VALUES = [0.001, 0.005, 0.01]
SEED = 42
ALPHA = 0.1
ROUNDS = 10  # Shorter for speed

def main():
    print("=" * 60)
    print("SCAFFOLD LR SENSITIVITY TEST")
    print("Testing if poor SCAFFOLD performance is due to LR mismatch")
    print("=" * 60)
    
    results = []
    
    for lr in LR_VALUES:
        print(f"\n{'-' * 50}")
        print(f"Testing SCAFFOLD with lr={lr}")
        print(f"{'-' * 50}")
        
        set_seed(SEED)
        
        start = time.time()
        result = run_scaffold(
            dataset_name="cifar10",
            num_classes=10,
            num_clients=10,
            rounds=ROUNDS,
            local_epochs=5,
            seed=SEED,
            alpha=ALPHA,
            lr=lr,
            batch_size=64,
            max_samples=20000,
            return_trace=True,
        )
        elapsed = time.time() - start
        
        result["run_id"] = f"scaffold_a{ALPHA}_s{SEED}_lr{lr}"
        result["method"] = "SCAFFOLD"
        result["alpha"] = ALPHA
        result["seed"] = SEED
        result["lr_tested"] = lr
        result["rounds"] = ROUNDS
        result["time_seconds"] = elapsed
        
        print(f"\n[RESULT] lr={lr} | Final acc: {result.get('mean_accuracy', 0):.4f} | Time: {elapsed:.1f}s")
        print(f"  Round accuracies: {result.get('round_accuracy', [])}")
        
        results.append(result)
        
        out_path = RUNS_DIR / f"scaffold_sensitivity_lr{lr}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  [SAVE] Saved to {out_path}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SCAFFOLD SENSITIVITY SUMMARY")
    print(f"{'=' * 60}")
    
    for r in results:
        lr = r["lr_tested"]
        acc = r.get("mean_accuracy", 0)
        print(f"  LR={lr:.4f}: Final accuracy={acc:.4f}")
    
    best = max(results, key=lambda r: r.get("mean_accuracy", 0))
    print(f"\nBest LR: {best['lr_tested']:.4f} → {best.get('mean_accuracy', 0):.4f}")
    
    # Save summary
    summary = {
        "test_name": "SCAFFOLD LR Sensitivity",
        "seed": SEED,
        "alpha": ALPHA,
        "rounds": ROUNDS,
        "results": [
            {
                "lr": r["lr_tested"],
                "final_accuracy": r.get("mean_accuracy", 0),
                "round_accuracy": r.get("round_accuracy", []),
            }
            for r in results
        ],
        "conclusion": "SCAFFOLD shows high sensitivity to learning rate" if len(set(r.get('mean_accuracy', 0) for r in results)) > 0.1 else "SCAFFOLD is robust to LR changes",
    }
    
    summary_path = OUTPUT_DIR / "scaffold_sensitivity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
