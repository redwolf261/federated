#!/usr/bin/env python3
"""Check if MOON learns without F.normalize()."""
import json
import time
from pathlib import Path

output_file = Path('outputs/debug_moon_scaffold_no_norm.json')

# Wait for file
max_wait = 120
waited = 0
while not output_file.exists():
    if waited >= max_wait:
        print("⏱ Still running...")
        exit(0)
    time.sleep(5)
    waited += 5

# Parse and check
with open(output_file, 'r', encoding='utf-16') as f:
    data = json.load(f)

print("=" * 70)
print("🔍 MOON BEHAVIOR (without F.normalize)")
print("=" * 70)

moon_data = data.get('MOON', {})
ce_norm = moon_data.get('CE_grad_norm', 0)
con_norm = moon_data.get('Contrastive_grad_norm', 0)
ratio = moon_data.get('Ratio', 0)

print(f"CE grad norm: {ce_norm:.6f}")
print(f"Contrastive grad norm: {con_norm:.6f}")
print(f"Ratio (contrastive/CE): {ratio:.6f}")
print()
if ratio < 1.0:
    print("✓ MOON ratio good (<1.0)")
else:
    print("✗ MOON ratio bad (≥1.0)")

print("\n" + "=" * 70)
print("🔍 SCAFFOLD BEHAVIOR (denominator = num_clients*K)")
print("=" * 70)

scaffold_data = data.get('SCAFFOLD', {})
grad_norm = scaffold_data.get('Gradient_norm', 0)
control_norm = scaffold_data.get('Control_term_norm', 0)
s_ratio = scaffold_data.get('Ratio', 0)

print(f"Gradient norm: {grad_norm:.6f}")
print(f"Control term norm: {control_norm:.6f}")
print(f"Ratio (control/grad): {s_ratio:.6f}")
print()
if s_ratio < 2.0:
    print("✓ SCAFFOLD ratio good (<2.0)")
else:
    print("✗ SCAFFOLD ratio bad (≥2.0)")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
if ratio < 1.0 and s_ratio < 2.0:
    print("✅ Both metrics pass! Run full validation.")
else:
    print("❌ Still issues - need deeper investigation")
