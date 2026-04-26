#!/usr/bin/env python3
"""Parse SCAFFOLD v3 results"""
import json
from pathlib import Path
import time

output_file = Path('outputs/debug_moon_scaffold_v3.json')

# Wait up to 60 seconds for the file to exist
wait_count = 0
while not output_file.exists() and wait_count < 60:
    wait_count += 1
    time.sleep(1)

try:
    with open(output_file, 'r', encoding='utf-16') as f:
        data = json.load(f)

    moon_data = data.get('moon_debug', {})
    moon_agg = moon_data.get('aggregate', {})
    scaffold_data = data.get('scaffold_debug', {})
    scaffold_agg = scaffold_data.get('aggregate', {})

    print("=" * 60)
    print("🔬 RESULTS: SCAFFOLD v3 (divide by num_clients*K)")
    print("=" * 60)
    print()
    print("MOON:")
    print(f"  CE grad norm: {moon_agg.get('ce_grad_norm_mean', 0):.6f}")
    print(f"  Contrastive grad norm: {moon_agg.get('contrastive_grad_norm_mean', 0):.6f}")
    print(f"  Ratio: {moon_agg.get('contrastive_to_ce_ratio_mean', 0):.6f}")
    print(f"  ✓ Status: {'PASS' if moon_agg.get('contrastive_to_ce_ratio_mean', 0) < 1.0 else 'FAIL'}")
    print()
    print("SCAFFOLD:")
    print(f"  Gradient norm: {scaffold_agg.get('grad_norm_mean', 0):.6f}")
    print(f"  Control term norm: {scaffold_agg.get('control_norm_mean', 0):.6f}")
    print(f"  Ratio (control/grad): {scaffold_agg.get('control_to_grad_ratio_mean', 0):.6f}")
    print()
    ratio = scaffold_agg.get('control_to_grad_ratio_mean', 0)
    if ratio < 2.0:
        print(f"  ✓ PASS: ratio={ratio:.4f} (target: <2.0)")
    elif ratio < 10:
        print(f"  ⚠ MAYBE: ratio={ratio:.4f} (target: <2.0, but much better)")
    else:
        print(f"  ✗ FAIL: ratio={ratio:.4f} (target: <2.0)")

except FileNotFoundError:
    print(f"File {output_file} not found after waiting")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
