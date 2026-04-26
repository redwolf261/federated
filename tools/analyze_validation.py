#!/usr/bin/env python3
"""Parse validation run results and check behavioral correctness."""
import json
from pathlib import Path
import numpy as np
import sys
import time

output_file = Path('outputs/validation_run_alpha_0.1.json')

# Wait for file to exist
max_wait = 300  # 5 minutes
waited = 0
while not output_file.exists():
    if waited >= max_wait:
        print("⏱ Validation run still executing... (this can take 20-30 minutes)")
        print(f"Check outputs directory for {output_file.name}")
        sys.exit(0)
    time.sleep(10)
    waited += 10

with open(output_file, 'r') as f:
    data = json.load(f)

print("=" * 70)
print("🔍 BEHAVIORAL VALIDATION RUN RESULTS")
print("=" * 70)
print()

# Extract results per method
results_by_method = {}
for method_name in ['fedavg', 'moon', 'scaffold', 'flex']:
    if method_name in data:
        method_data = data[method_name]
        results_by_method[method_name] = {
            'mean': method_data.get('mean_accuracy', 0),
            'seeds': method_data.get('accuracies_per_seed', []),
            'worst': method_data.get('worst_accuracy', 0),
            'std': method_data.get('std', 0),
        }

print("📊 MEAN ACCURACY (20 rounds)")
print("-" * 70)
for method in ['fedavg', 'moon', 'scaffold', 'flex']:
    if method in results_by_method:
        mean = results_by_method[method]['mean']
        std = results_by_method[method]['std']
        print(f"  {method.upper():10s}: {mean:.4f} ± {std:.4f}")

print()
print("📈 PER-SEED RESULTS")
print("-" * 70)
for method in ['fedavg', 'moon', 'scaffold', 'flex']:
    if method in results_by_method:
        seeds = results_by_method[method]['seeds']
        print(f"  {method.upper():10s}: {seeds}")

print()
print("⚠️  BEHAVIORAL CHECKS")
print("-" * 70)

# Check 1: MOON not flat
moon_mean = results_by_method.get('moon', {}).get('mean', 0)
fedavg_mean = results_by_method.get('fedavg', {}).get('mean', 0)
check1 = moon_mean > 0.15 and moon_mean >= fedavg_mean * 0.8
print(f"1. MOON not flat?: {moon_mean:.4f} (FedAvg: {fedavg_mean:.4f})")
print(f"   Status: {'✓ PASS' if check1 else '✗ FAIL - MOON collapsed'}")

# Check 2: SCAFFOLD stable (not oscillating, not collapsing)
scaffold_mean = results_by_method.get('scaffold', {}).get('mean', 0)
check2 = scaffold_mean > 0.15 and scaffold_mean >= fedavg_mean * 0.7
print(f"2. SCAFFOLD stable?: {scaffold_mean:.4f} (target: >{fedavg_mean*0.7:.4f})")
print(f"   Status: {'✓ PASS' if check2 else '✗ FAIL - SCAFFOLD unstable'}")

# Check 3: FLEX reasonable dominance (not absurd)
flex_mean = results_by_method.get('flex', {}).get('mean', 0)
flex_advantage = (flex_mean - fedavg_mean) / max(fedavg_mean, 0.01)
check3 = flex_advantage < 1.0  # Less than 100% absolute improvement
print(f"3. FLEX dominance?: {flex_mean:.4f} (+{flex_advantage*100:.1f}% vs FedAvg)")
print(f"   Status: {'✓ PASS' if check3 else '✗ WARNING - FLEX suspiciously high'}")

# Check 4: Worst-client tracking (fairness)
print()
print("📉 WORST-CLIENT ACCURACY")
print("-" * 70)
for method in ['fedavg', 'moon', 'scaffold', 'flex']:
    if method in results_by_method:
        worst = results_by_method[method]['worst']
        mean = results_by_method[method]['mean']
        gap = mean - worst
        print(f"  {method.upper():10s}: {worst:.4f} (gap from mean: {gap:.4f})")

print()
print("🚀 VERDICT")
print("=" * 70)
all_pass = check1 and check2 and check3
if all_pass:
    print("✅ EXPERIMENT-READY: All baselines behave correctly.")
    print("   Proceed to full experiment suite (5-10 seeds, 50-100 rounds).")
else:
    print("❌ NOT READY: Issues detected in baseline behavior.")
    if not check1:
        print("   • MOON is still flat—possible contrastive weight issue")
    if not check2:
        print("   • SCAFFOLD is unstable—need deeper investigation")
    if not check3:
        print("   • FLEX advantage is unrealistic—compare vs broken baselines?")
print()
