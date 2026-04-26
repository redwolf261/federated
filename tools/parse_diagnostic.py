#!/usr/bin/env python3
"""Parse MOON/SCAFFOLD diagnostic (without normalization)."""
import json
import numpy as np

with open('outputs/debug_moon_scaffold_no_norm.json', 'r', encoding='utf-16') as f:
    data = json.load(f)

print("=" * 70)
print("🔍 MOON DIAGNOSTIC (without F.normalize)")
print("=" * 70)

moon_data = data.get('moon_debug', {})
print(f"Feature normalization present: {moon_data.get('feature_normalization_present', False)}")
print(f"μ = {moon_data.get('mu', 0)}, τ = {moon_data.get('temperature', 0)}")

round_data = moon_data.get('round_debug', [])
if round_data:
    ce_norms = [r['ce_grad_norm_mean'] for r in round_data]
    con_norms = [r['contrastive_grad_norm_mean'] for r in round_data]
    ratios = [r['contrastive_to_ce_ratio'] for r in round_data]
    
    print(f"\nPer-round contrastive/CE ratios:")
    for i, (ce, con, r) in enumerate(zip(ce_norms, con_norms, ratios), 1):
        print(f"  Round {i}: CE={ce:.6f}, Con={con:.6f}, Ratio={r:.6f}")
    
    # Final metrics
    avg_ratio = np.mean(ratios)
    print(f"\nAverage ratio: {avg_ratio:.6f}")
    print(f"Status: {'✓ PASS' if avg_ratio < 1.0 else '✗ FAIL'} (target <1.0)")

print("\n" + "=" * 70)
print("🔍 SCAFFOLD DIAGNOSTIC (denominator = num_clients*K)")
print("=" * 70)

scaffold_data = data.get('scaffold_debug', {})
print(f"Update equation: w_i <- w_i - eta * (grad - c_i + c)")

s_round_data = scaffold_data.get('round_debug', [])
if s_round_data:
    grad_norms = [r['grad_norm_mean'] for r in s_round_data]
    control_norms = [r['control_norm_mean'] for r in s_round_data]
    s_ratios = [r['control_to_grad_ratio'] for r in s_round_data]
    
    print(f"\nPer-round control/gradient ratios:")
    for i, (g, c, r) in enumerate(zip(grad_norms, control_norms, s_ratios), 1):
        print(f"  Round {i}: Grad={g:.6f}, Control={c:.6f}, Ratio={r:.6f}")
    
    avg_s_ratio = np.mean(s_ratios)
    print(f"\nAverage ratio: {avg_s_ratio:.6f}")
    print(f"Status: {'✓ PASS' if avg_s_ratio < 2.0 else '✗ FAIL'} (target <2.0)")

print("\n" + "=" * 70)
print("🚀 VERDICT")
print("=" * 70)
if round_data and s_round_data:
    moon_pass = avg_ratio < 1.0
    scaffold_pass = avg_s_ratio < 2.0
    if moon_pass and scaffold_pass:
        print("✅ Both diagnostics pass!")
        print("   Proceed to full behavioral validation (20 rounds, 3 seeds)")
    else:
        if not moon_pass:
            print(f"❌ MOON still issues: ratio={avg_ratio:.4f} (need <1.0)")
        if not scaffold_pass:
            print(f"❌ SCAFFOLD still issues: ratio={avg_s_ratio:.4f} (need <2.0)")
