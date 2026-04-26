#!/usr/bin/env python3
"""Debug why MOON/SCAFFOLD fail despite fixes."""
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Check 1: Does F.normalize exist and work?
print("=" * 70)
print("CHECK 1: F.normalize functionality")
print("=" * 70)
test_z = torch.randn(5, 64)
z_normalized = F.normalize(test_z, p=2, dim=1)
print(f"Input shape: {test_z.shape}")
print(f"Output shape: {z_normalized.shape}")
print(f"Norms after normalize: {torch.norm(z_normalized, dim=1)}")  # Should all be 1.0
print(f"✓ F.normalize works correctly\n")

# Check 2: Load the validation results and look for patterns
print("=" * 70)
print("CHECK 2: Validation run results analysis")
print("=" * 70)

with open('outputs/validation_run_alpha_0.1.json', 'r') as f:
    results = json.load(f)['alpha_0.1']

# Extract MOON seed-wise performance
moon_seeds = results['moon']['per_seed']
for seed, data in moon_seeds.items():
    rounds = data.get('round_curves', [[]])  # Note: round_curves is in per_seed for some reason
    print(f"MOON seed {seed}: mean={data['mean_accuracy']:.4f}")
    if len(results['moon']['round_curves']) > 0:
        seed_curve = results['moon']['round_curves'][int(seed) // 200 if seed == '42' else 0]  # This is wrong
        # Actually the structure might be different, let me check

# Let me print the actual structure
print("\nMOON result structure:")
print(f"Keys: {list(results['moon'].keys())}")
print(f"round_curves type: {type(results['moon']['round_curves'])}")
print(f"round_curves length: {len(results['moon']['round_curves'])}")

print("\nMOON seed curves (from round_curves array):")
for i, curve in enumerate(results['moon']['round_curves']):
    print(f"  Curve {i} (seed {results['moon']['per_seed'].keys().__iter__().__next__() if i == 0 else '?'}): "
          f"starts={curve[0]:.4f}, ends={curve[-1]:.4f}, flat={max(curve)-min(curve)<0.01}")

print("\nSCAFFOLD seed curves (from round_curves array):")
for i, curve in enumerate(results['scaffold']['round_curves']):
    print(f"  Curve {i}: starts={curve[0]:.4f}, ends={curve[-1]:.4f}, "
          f"range={max(curve)-min(curve):.4f}")

# Check 3: Critical question - why is seed 456 different?
print("\n" + "=" * 70)
print("CHECK 3: Why seed 456 works but others don't?")
print("=" * 70)

moon_by_seed = {
    '42': results['moon']['per_seed']['42']['mean_accuracy'],
    '123': results['moon']['per_seed']['123']['mean_accuracy'],
    '456': results['moon']['per_seed']['456']['mean_accuracy'],
}
print("MOON per-seed accuracy:")
for seed, acc in moon_by_seed.items():
    status = "✓ WORKS" if acc > 0.3 else "✗ COLLAPSED"
    print(f"  Seed {seed}: {acc:.4f} {status}")

print("\nPossible explanations:")
print("1. Random initialization: seed 456 might initialize in a better region")
print("2. Normalization + small initial loss: might cause NaN or gradient issues on other seeds")
print("3. MOON contrastive: negative/positive samples might not be well-separated")

# Check 4: Can we reverse-engineer what went wrong?
print("\n" + "=" * 70)
print("CHECK 4: Comparing MOON to FedAvg")
print("=" * 70)

fedavg_by_seed = {
    '42': results['fedavg']['per_seed']['42']['mean_accuracy'],
    '123': results['fedavg']['per_seed']['123']['mean_accuracy'],
    '456': results['fedavg']['per_seed']['456']['mean_accuracy'],
}

print("MOON vs FedAvg per-seed:")
for seed in ['42', '123', '456']:
    moon_acc = moon_by_seed[seed]
    fedavg_acc = fedavg_by_seed[seed]
    ratio = moon_acc / (fedavg_acc + 1e-6)
    print(f"  Seed {seed}: MOON={moon_acc:.4f} vs FedAvg={fedavg_acc:.4f} (ratio={ratio:.4f})")

print("\n" + "=" * 70)
print("CHECK 5: Hypothesis - is F.normalize killing the loss?")
print("=" * 70)
print("""
When you F.normalize(z) to unit norm, cosine similarity becomes:
  cos_sim(z_a, z_b) where ||z_a||=||z_b||=1
  This is theoretically correct for contrastive learning.

BUT: If the temperature is too low or positive/negative don't diverge:
  - Normalized vectors might not separate well during early training
  - Temperature τ=0.5 might be too small with normalized vectors
  - The model might get stuck in a local minimum

Hypothesis: Try REMOVING F.normalize() and see if it helps.
""")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("""
The F.normalize() fix is mathematically correct for cosine similarity,
but empirically it's breaking learning on 2/3 seeds.

Two options:
1. REMOVE F.normalize() and see if plain cosine_sim recovers learning
   (trade-off: cosine sim without norm = magnitude-weighted dot product)
2. Keep F.normalize() but increase temperature τ from 0.5 to 1.0 or 2.0
   (might help separation with normalized vectors)

Recommend trying OPTION 1 first (remove normalize) since diagnostic
showed 0.608 ratio (normalized) but 372.0 control ratio (SCAFFOLD).
MOON might be a red herring - the real issue could be SCAFFOLD.
""")
