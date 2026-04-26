#!/usr/bin/env python3
"""Parse debug_moon_scaffold_fixed.json to see if normalization fix worked"""
import json
from pathlib import Path

output_file = Path('outputs/debug_moon_scaffold_fixed.json')

# Read UTF-16 encoded JSON
with open(output_file, 'r', encoding='utf-16') as f:
    data = json.load(f)

# Extract MOON results
moon_data = data.get('moon_debug', {})
moon_agg = moon_data.get('aggregate', {})

print("=" * 60)
print("🔬 MOON DIAGNOSTIC RESULTS (With Normalization)")
print("=" * 60)
print(f"✓ Feature normalization: {moon_data.get('feature_normalization_present')}")
print(f"  μ (mu) = {moon_data.get('mu')}")
print(f"  τ (temperature) = {moon_data.get('temperature')}")
print()
print("Gradient norms:")
print(f"  CE loss grad norm: {moon_agg.get('ce_grad_norm_mean', 0):.6f}")
print(f"  Contrastive grad norm: {moon_agg.get('contrastive_grad_norm_mean', 0):.6f}")
print(f"  Ratio (contrastive/CE): {moon_agg.get('contrastive_to_ce_ratio_mean', 0):.6f}")
print(f"  Ratio max: {moon_agg.get('contrastive_to_ce_ratio_max', 0):.6f}")
print()

# Extract SCAFFOLD results
scaffold_data = data.get('scaffold_debug', {})
scaffold_agg = scaffold_data.get('aggregate', {})

print("=" * 60)
print("🔬 SCAFFOLD DIAGNOSTIC RESULTS (After Debug Fix)")
print("=" * 60)
print()
print("Control-variate terms:")
print(f"  Gradient norm: {scaffold_agg.get('grad_norm_mean', 0):.6f}")
print(f"  Control term norm: {scaffold_agg.get('control_norm_mean', 0):.6f}")
print(f"  Ratio (control/grad): {scaffold_agg.get('control_to_grad_ratio_mean', 0):.6f}")
print(f"  Ratio max: {scaffold_agg.get('control_to_grad_ratio_max', 0):.6f}")
print()

print("✅ ANALYSIS")
moon_ratio = moon_agg.get('contrastive_to_ce_ratio_mean', 0)
if moon_ratio < 0.5:
    print(f"✓ MOON: Normalization working! Ratio={moon_ratio:.4f} (excellent)")
elif moon_ratio < 1.0:
    print(f"✓ MOON: Acceptable. Ratio={moon_ratio:.4f}")
else:
    print(f"✗ MOON: Still high. Ratio={moon_ratio:.4f}")

scaffold_ratio = scaffold_agg.get('control_to_grad_ratio_mean', 0)
if scaffold_ratio < 2.0:
    print(f"✓ SCAFFOLD: Fixed! Ratio={scaffold_ratio:.4f}")
elif scaffold_ratio < 100:
    print(f"⚠ SCAFFOLD: Improved but not enough. Ratio={scaffold_ratio:.4f}")
else:
    print(f"✗ SCAFFOLD: Still exploding. Ratio={scaffold_ratio:.4f}")
