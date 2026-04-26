#!/usr/bin/env python3
"""Analyze behavioral validation results."""
import json
import time
from pathlib import Path

output_file = Path('outputs/short_experiment_alpha_0.1.json')

# Wait for file
max_wait = 1800  # 30 minutes
waited = 0
interval = 10
while not output_file.exists():
    if waited >= max_wait:
        print("⏱ Validation run still executing...")
        exit(0)
    print(f"⏳ Waiting... ({waited}s)", end='\r')
    time.sleep(interval)
    waited += interval

print("\n✅ Results file found! Parsing...\n")

with open(output_file, 'r') as f:
    results = json.load(f)['alpha_0.1']

print("=" * 70)
print("📊 BEHAVIORAL VALIDATION RESULTS (20 rounds, α=0.1)")
print("=" * 70)
print()

# Mean accuracies
methods = ['fedavg', 'moon', 'scaffold', 'flex']
print("MEAN ACCURACY ACROSS 3 SEEDS:")
for m in methods:
    mean = results[m]['mean_accuracy']
    std = results[m]['std'] if 'std' in results[m] else 0
    worst = results[m]['worst_accuracy']
    print(f"  {m.upper():10s}: {mean:.4f} ± {std:.4f}  (worst: {worst:.4f})")

print()

# Per-seed breakdown  
fedavg_mean = results['fedavg']['mean_accuracy']
print("PER-SEED BREAKDOWN:")
for method in methods:
    seed_accs = [
        results[method]['per_seed']['42']['mean_accuracy'],
        results[method]['per_seed']['123']['mean_accuracy'],
        results[method]['per_seed']['456']['mean_accuracy'],
    ]
    print(f"  {method.upper():10s}: {seed_accs[0]:.4f}, {seed_accs[1]:.4f}, {seed_accs[2]:.4f}")

print()
print("=" * 70)
print("✓/✗ BEHAVIORAL CHECKS")
print("=" * 70)

# Check 1: MOON not flat
moon_mean = results['moon']['mean_accuracy']
moon_seeds = [results['moon']['per_seed'][s]['mean_accuracy'] for s in ['42', '123', '456']]
check1 = moon_mean >= fedavg_mean * 0.7 and all(s > 0.1 for s in moon_seeds)
print(f"\n1. MOON not flat?")
print(f"   Mean: {moon_mean:.4f} (FedAvg: {fedavg_mean:.4f})")
print(f"   Per-seed: {moon_seeds}")
print(f"   Status: {'✓ PASS' if check1 else '✗ FAIL'}")
if not check1:
    if moon_mean < fedavg_mean * 0.7:
        print(f"   → Too low vs FedAvg: {moon_mean/fedavg_mean:.2f}x")
    if any(s < 0.1 for s in moon_seeds):
        print(f"   → At least one seed collapsed: {[f'{x:.4f}' for x in moon_seeds]}")

# Check 2: SCAFFOLD stable
scaffold_mean = results['scaffold']['mean_accuracy']
scaffold_seeds = [results['scaffold']['per_seed'][s]['mean_accuracy'] for s in ['42', '123', '456']]
scaffold_var = max(scaffold_seeds) - min(scaffold_seeds)
check2 = scaffold_var < 0.3 and scaffold_mean >= fedavg_mean * 0.5
print(f"\n2. SCAFFOLD stable?")
print(f"   Mean: {scaffold_mean:.4f}")
print(f"   Per-seed: {scaffold_seeds}")
print(f"   Variance (max-min): {scaffold_var:.4f} (target: <0.3)")
print(f"   Status: {'✓ PASS' if check2 else '✗ WARNING'}")
if not check2:
    print(f"   → High variance or too low")

# Check 3: FLEX not unrealistic
flex_mean = results['flex']['mean_accuracy']
flex_advantage = (flex_mean - fedavg_mean) / fedavg_mean
check3 = flex_advantage < 1.0
print(f"\n3. FLEX advantage reasonable?")
print(f"   FLEX: {flex_mean:.4f}, FedAvg: {fedavg_mean:.4f}")
print(f"   Advantage: +{flex_advantage*100:.1f}% (target: <100%)")
print(f"   Status: {'✓ PASS' if check3 else '⚠ HIGH'}")

# Check 4: Fairness
fedavg_gap = results['fedavg']['mean_accuracy'] - results['fedavg']['worst_accuracy']
moon_gap = results['moon']['mean_accuracy'] - results['moon']['worst_accuracy']
scaffold_gap = results['scaffold']['mean_accuracy'] - results['scaffold']['worst_accuracy']
flex_gap = results['flex']['mean_accuracy'] - results['flex']['worst_accuracy']
check4 = flex_gap < 0.2
print(f"\n4. Worst-client fairness?")
print(f"   FedAvg gap: {fedavg_gap:.4f}")
print(f"   MOON gap:   {moon_gap:.4f}")
print(f"   SCAFFOLD gap: {scaffold_gap:.4f}")
print(f"   FLEX gap:   {flex_gap:.4f} (target: <0.2)")
print(f"   Status: {'✓ PASS' if check4 else '⚠ CONCERN'}")

print()
print("=" * 70)
print("🎯 FINAL VERDICT")
print("=" * 70)

all_pass = check1 and check2 and check3 and check4
if all_pass:
    print("✅ EXPERIMENT-READY!")
    print("   All behavioral checks pass.")
    print("   Ready to scale to full experiment suite.")
else:
    print("⚠️  PARTIAL VALIDATION")
    if not check1:
        print("   ❌ MOON still issues")
    if not check2:
        print("   ⚠️  SCAFFOLD high variance")
    if not check3:
        print("   ⚠️  FLEX advantage unusually high")
    print()
    print("   → May need tuning or deeper investigation")
