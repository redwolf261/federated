#!/usr/bin/env python3
"""
Partial MOON run for defensibility documentation.
5 rounds only (due to time constraints), seed 42, alpha=0.1.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import time
from scripts.phase2_q1_validation import run_moon, set_seed

OUTPUT_DIR = Path("outputs/locked_cifar10_grid")
RUNS_DIR = OUTPUT_DIR / "runs"

def main():
    print("=" * 60)
    print("MOON PARTIAL RUN (5 rounds, for defensibility)")
    print("Full grid excluded due to ~8+ min/round execution time")
    print("=" * 60)
    
    set_seed(42)
    
    start = time.time()
    result = run_moon(
        dataset_name="cifar10",
        num_classes=10,
        num_clients=10,
        rounds=5,  # Partial run
        local_epochs=5,
        seed=42,
        alpha=0.1,
        lr=0.003,
        batch_size=64,
        max_samples=20000,
        mu=1,
        temperature=0.5,
        return_trace=True,
    )
    elapsed = time.time() - start
    
    result["run_id"] = "moon_a0.1_s42_partial"
    result["method"] = "MOON"
    result["alpha"] = 0.1
    result["seed"] = 42
    result["rounds"] = 5
    result["note"] = "Partial run (5/20 rounds) due to computational constraints"
    result["time_seconds"] = elapsed
    
    print(f"\n[RESULT] MOON partial run complete")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Rounds: 5/20")
    print(f"  Final accuracy: {result.get('mean_accuracy', 0):.4f}")
    print(f"  Round accuracies: {result.get('round_accuracy', [])}")
    
    out_path = RUNS_DIR / "moon_a0.1_s42_partial.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVE] Saved to {out_path}")

if __name__ == "__main__":
    main()
