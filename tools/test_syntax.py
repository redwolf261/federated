#!/usr/bin/env python
import sys
try:
    import scripts.phase2_q1_validation as m
    print("✓ phase2_q1_validation syntax OK")
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)

try:
    import scripts.debug_moon_scaffold as d
    print("✓ debug_moon_scaffold syntax OK")
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)

print("✓ All scripts valid")
