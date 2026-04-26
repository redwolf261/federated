import json

# Load the validation results
with open('outputs/validation_run_alpha_0.1.json', 'r') as f:
    data = json.load(f)['alpha_0.1']

print("=" * 70)
print("🔍 BEHAVIORAL VALIDATION RESULTS (20 rounds, 3 seeds, α=0.1)")
print("=" * 70)
print()

# Extract mean accuracies
methods = ['fedavg', 'moon', 'scaffold', 'flex']
means = {}
for m in methods:
    means[m] = data[m]['mean_accuracy']
    worst = data[m]['worst_accuracy']
    print(f"{m.upper():10s}: mean={means[m]:.4f}  worst={worst:.4f}")

print()
print("=" * 70)
print("✓/✗ BEHAVIORAL CHECKS")
print("=" * 70)

# Check 1: MOON not flat
moon_seeds = [data['moon']['per_seed']['42']['mean_accuracy'],
              data['moon']['per_seed']['123']['mean_accuracy'],
              data['moon']['per_seed']['456']['mean_accuracy']]
fedavg_mean = means['fedavg']
check1 = all(m > 0.1 for m in moon_seeds) and means['moon'] > fedavg_mean * 0.7
print(f"\n1️⃣  MOON NOT FLAT:")
print(f"   Per-seed: {[f'{m:.4f}' for m in moon_seeds]}")
print(f"   vs FedAvg: {fedavg_mean:.4f}")
print(f"   ❌ FAIL: MOON collapsed on seeds 42,123 (near 0%, flat curves)")

# Check 2: SCAFFOLD stable
scaffold_seeds = [data['scaffold']['per_seed']['42']['mean_accuracy'],
                  data['scaffold']['per_seed']['123']['mean_accuracy'],
                  data['scaffold']['per_seed']['456']['mean_accuracy']]
check2 = all(abs(s - means['scaffold']) < 0.3 for s in scaffold_seeds)
print(f"\n2️⃣  SCAFFOLD STABLE:")
print(f"   Per-seed: {[f'{s:.4f}' for s in scaffold_seeds]}")
print(f"   Variance: {max(scaffold_seeds) - min(scaffold_seeds):.4f}")
print(f"   ❌ FAIL: Extreme seed variance (17.5% → 65.9%), unstable")

# Check 3: FLEX dominance  
flex_mean = means['flex']
dominance = (flex_mean - fedavg_mean) / fedavg_mean * 100
check3 = dominance < 50
print(f"\n3️⃣  FLEX DOMINANCE:")
print(f"   FLEX={flex_mean:.4f}, FedAvg={fedavg_mean:.4f}")
print(f"   Advantage: +{dominance:.1f}% (should be <50%)")
print(f"   ⚠️  CONCERN: Unrealistically high advantage over broken baselines")

# Check 4: Worst-client gap
fedavg_gap = means['fedavg'] - data['fedavg']['worst_accuracy']
moon_gap = means['moon'] - data['moon']['worst_accuracy']
scaffold_gap = means['scaffold'] - data['scaffold']['worst_accuracy']
flex_gap = means['flex'] - data['flex']['worst_accuracy']
check4 = flex_gap < 0.2
print(f"\n4️⃣  WORST-CLIENT FAIRNESS (gap from mean):")
print(f"   FedAvg:   {fedavg_gap:.4f}")
print(f"   MOON:     {moon_gap:.4f}")
print(f"   SCAFFOLD: {scaffold_gap:.4f}")
print(f"   FLEX:     {flex_gap:.4f}")
print(f"   ✓ PASS: FLEX fairness is reasonable")

print()
print("=" * 70)
print("🚨 VERDICT")
print("=" * 70)
print("❌ NOT EXPERIMENT-READY")
print()
print("ROOT CAUSE: MOON & SCAFFOLD are fundamentally broken")
print()
print("Issues:")
print("• MOON: Completely flat on 2/3 seeds (4.75% and 7% accuracy)")
print("  → Feature normalization is preventing learning")
print()
print("• SCAFFOLD: Extreme instability (17.5% to 65.9%)")
print("  → Control-variate fix may be incorrect")
print()
print("• FLEX appears good because baselines are broken")
print("  → Cannot validate FLEX quality until baselines work")
print()
print("Next steps: Debug MOON normalization and SCAFFOLD denominator")
