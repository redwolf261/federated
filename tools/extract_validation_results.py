#!/usr/bin/env python3
"""Extract 4 required metrics from validation run."""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('outputs/validation_run_alpha_0.1_no_norm.log', 'r', encoding='utf-16') as f:
    results = json.load(f)['alpha_0.1']

methods = list(results.keys())
print("=" * 80)
print("VALIDATION RESULTS (α=0.1, 20 rounds, 3 seeds, WITHOUT F.normalize())")
print("=" * 80)

# 1. MEAN ACCURACY PER METHOD
print("\n1. MEAN ACCURACY PER METHOD")
print("-" * 40)
for method in methods:
    mean_acc = results[method]['mean_accuracy']
    print(f"  {method:12s}: {mean_acc:.4f}")

# 2. PER-SEED VALUES
print("\n2. PER-SEED ACCURACY VALUES")
print("-" * 40)
for method in methods:
    per_seed = results[method]['per_seed']
    print(f"\n  {method}:")
    for seed in sorted(per_seed.keys()):
        seed_acc = per_seed[seed]['mean_accuracy']
        print(f"    Seed {seed}: {seed_acc:.4f}")

# 3. CONVERGENCE CURVES (one per method)
print("\n3. GENERATING CONVERGENCE CURVES")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, method in enumerate(methods):
    ax = axes[idx]
    round_curves = results[method]['round_curves']
    
    # Each row is a seed curve
    seeds = list(results[method]['per_seed'].keys())
    for seed_idx, seed in enumerate(sorted(seeds)):
        curve = round_curves[seed_idx]
        ax.plot(range(1, 21), curve, marker='o', label=f'Seed {seed}', linewidth=2, markersize=4)
    
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'{method.upper()} Convergence (α=0.1)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('outputs/convergence_curves_no_norm.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: outputs/convergence_curves_no_norm.png")

# 4. WORST-CLIENT ACCURACY
print("\n4. WORST-CLIENT ACCURACY PER METHOD")
print("-" * 40)
for method in methods:
    worst_acc = results[method]['worst_accuracy']
    print(f"  {method:12s}: {worst_acc:.4f}")

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Method':<12} {'Mean Acc':<12} {'Worst-Client':<15} {'Seed Variance':<15}")
print("-" * 80)
for method in methods:
    mean_acc = results[method]['mean_accuracy']
    worst_acc = results[method]['worst_accuracy']
    per_seed = results[method]['per_seed']
    seed_accs = [per_seed[s]['mean_accuracy'] for s in per_seed.keys()]
    variance = np.std(seed_accs)
    print(f"{method:<12} {mean_acc:.4f}         {worst_acc:.4f}         ±{variance:.4f}")

print("\n✅ All 4 metrics extracted successfully!")
print("   1. Mean accuracy: ✓")
print("   2. Per-seed values: ✓")
print("   3. Convergence curves: ✓ (saved as PNG)")
print("   4. Worst-client accuracy: ✓")
