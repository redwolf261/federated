#!/usr/bin/env python3
"""Parse debug_output.json from debug_moon_scaffold.py"""
import json
import sys
from pathlib import Path

output_file = Path('outputs/debug_moon_scaffold.json')

# Read UTF-16 encoded JSON
with open(output_file, 'r', encoding='utf-16') as f:
    data = json.load(f)

# Extract MOON results
moon_data = data.get('moon_debug', {})
moon_agg = moon_data.get('aggregate', {})

print("=" * 60)
print("🔬 MOON DIAGNOSTIC RESULTS (Post-Fix)")
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
print("🔬 SCAFFOLD DIAGNOSTIC RESULTS (Post-Fix)")
print("=" * 60)
print(f"Update equation: {scaffold_data.get('implemented_equation', 'N/A')}")
print()
print("Control-variate terms:")
print(f"  Gradient norm: {scaffold_agg.get('grad_norm_mean', 0):.6f}")
print(f"  Control term norm: {scaffold_agg.get('control_norm_mean', 0):.6f}")
print(f"  Ratio (control/grad): {scaffold_agg.get('control_to_grad_ratio_mean', 0):.6f} ← KEY METRIC")
print(f"  Ratio max: {scaffold_agg.get('control_to_grad_ratio_max', 0):.6f}")
print(f"  c_global norm (final): {scaffold_agg.get('c_global_norm_final', 0):.6f}")
print()

# Summary
print("=" * 60)
print("✅ VERDICT")
print("=" * 60)
moon_ratio = moon_agg.get('contrastive_to_ce_ratio_mean', 0)
scaffold_ratio = scaffold_agg.get('control_to_grad_ratio_mean', 0)

if moon_ratio < 1.0:
    print(f"✓ MOON normalized: ratio={moon_ratio:.4f} (target: <1.0)")
else:
    print(f"✗ MOON still high: ratio={moon_ratio:.4f}")

if scaffold_ratio < 2.0:
    print(f"✓ SCAFFOLD fixed: ratio={scaffold_ratio:.4f} (target: <2.0)")
else:
    print(f"✗ SCAFFOLD not fixed: ratio={scaffold_ratio:.4f} (should be <2.0)")

print()
