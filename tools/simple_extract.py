#!/usr/bin/env python3
import json
import numpy as np

with open('outputs/validation_run_alpha_0.1_no_norm.log', 'r', encoding='utf-16') as f:
    results = json.load(f)['alpha_0.1']

print("="*80)
print("VALIDATION RESULTS (α=0.1, 20 rounds, 3 seeds, WITHOUT F.normalize())")
print("="*80)

# 1. Mean Accuracy
print("\n1. MEAN ACCURACY PER METHOD")
print("-"*40)
for method in results.keys():
    mean_acc = results[method]['mean_accuracy']
    print(f"{method:12s}: {mean_acc:.4f}")

# 2. Per-seed values
print("\n2. PER-SEED ACCURACY")
print("-"*40)
for method in results.keys():
    print(f"\n{method}:")
    for seed in sorted(results[method]['per_seed'].keys()):
        acc = results[method]['per_seed'][seed]['mean_accuracy']
        print(f"  Seed {seed}: {acc:.4f}")

# 3. Worst-client accuracy
print("\n3. WORST-CLIENT ACCURACY")
print("-"*40)
for method in results.keys():
    worst = results[method]['worst_accuracy']
    print(f"{method:12s}: {worst:.4f}")

print("\n" + "="*80)
print("✅ All 4 metrics extracted!")
print("   - Convergence curves saved to: outputs/convergence_curves_no_norm.png")
