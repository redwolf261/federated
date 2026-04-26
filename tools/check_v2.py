#!/usr/bin/env python3
"""Parse SCAFFOLD v2 results"""
import json
from pathlib import Path

output_file = Path('outputs/debug_moon_scaffold_v2.json')

if not output_file.exists():
    print("v2 file doesn't exist yet, waiting...")
    import time
    time.sleep(5)

try:
    with open(output_file, 'r', encoding='utf-16') as f:
        data = json.load(f)

    moon_data = data.get('moon_debug', {})
    moon_agg = moon_data.get('aggregate', {})
    scaffold_data = data.get('scaffold_debug', {})
    scaffold_agg = scaffold_data.get('aggregate', {})

    print("=" * 60)
    print("🔬 SCAFFOLD v2 (with num_clients scaling)")
    print("=" * 60)
    print(f"Gradient norm: {scaffold_agg.get('grad_norm_mean', 0):.6f}")
    print(f"Control term norm: {scaffold_agg.get('control_norm_mean', 0):.6f}")
    print(f"Ratio (control/grad): {scaffold_agg.get('control_to_grad_ratio_mean', 0):.6f}")
    print()
    print("MOON:")
    print(f"CE grad norm: {moon_agg.get('ce_grad_norm_mean', 0):.6f}")
    print(f"Contrastive grad norm: {moon_agg.get('contrastive_grad_norm_mean', 0):.6f}")
    print(f"Ratio: {moon_agg.get('contrastive_to_ce_ratio_mean', 0):.6f}")

except FileNotFoundError:
    print(f"File {output_file} not found yet")
except Exception as e:
    print(f"Error: {e}")
