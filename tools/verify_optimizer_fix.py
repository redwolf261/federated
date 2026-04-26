#!/usr/bin/env python3
"""Verify optimizer persistence fix."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def main():
    with open(PROJECT_ROOT / "scripts" / "phase2_q1_validation.py", encoding="utf-8") as f:
        content = f.read()
    
    print("="*60)
    print("VERIFICATION: Optimizer Persistence Fix")
    print("="*60 + "\n")
    
    # Key patterns to check
    checks = [
        ("MOON: persistent client_models dict", "client_models = {}" in content.split("def run_moon")[1]),
        ("MOON: persistent client_optimizers dict", "client_optimizers = {}" in content.split("def run_moon")[1]),
        ("MOON: load_state_dict (not deepcopy)", "client_models[cid].load_state_dict" in content.split("def run_moon")[1]),
        ("MOON: reuse optimizer", "optimizer = client_optimizers[cid]" in content.split("def run_moon")[1]),
        ("SCAFFOLD: persistent client_models dict", "client_models = {}" in content.split("def run_scaffold")[1]),
        ("SCAFFOLD: persistent client_optimizers dict", "client_optimizers = {}" in content.split("def run_scaffold")[1]),
        ("SCAFFOLD: load_state_dict (not deepcopy)", "client_models[cid].load_state_dict" in content.split("def run_scaffold")[1]),
        ("SCAFFOLD: reuse optimizer", "optimizer = client_optimizers[cid]" in content.split("def run_scaffold")[1]),
    ]
    
    all_pass = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_pass = False
    
    print("\n" + "="*60)
    if all_pass:
        print("✅ ALL FIXES VERIFIED - Ready for validation")
        return 0
    else:
        print("❌ SOME FIXES MISSING")
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Direct validation of the optimizer persistence fix without heavy imports.
This just verifies the structure of the changes.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def check_moon_fix():
    """Verify run_moon uses persistent optimizers"""
    with open(PROJECT_ROOT / "scripts" / "phase2_q1_validation.py", encoding="utf-8") as f:
        content = f.read()
    
    # Check for the fixed patterns in run_moon
    checks = {
        "persistent models dict": "client_models = {}" in content,
        "persistent optimizers dict": "client_optimizers = {}" in content,
        "load_state_dict instead of deepcopy": "client_models[cid].load_state_dict(global_model.state_dict())" in content,
        "optimizer reuse": "optimizer = client_optimizers[cid]" in content,
        "NOT using fresh Adam": "optimizer = torch.optim.Adam(local_model.parameters()" not in content.split("def run_moon")[1].split("def run_scaffold")[0],
    }
    
    print("✓ MOON Fix Validation:")
    all_pass = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_pass = False
    
    return all_pass

def check_scaffold_fix():
    """Verify run_scaffold uses persistent optimizers"""
    with open(PROJECT_ROOT / "scripts" / "phase2_q1_validation.py", encoding="utf-8") as f:
        content = f.read()
    
    # Check for the fixed patterns in run_scaffold
    scaffold_section = content.split("def run_scaffold")[1]
    
    checks = {
        "persistent models in scaffold": "client_models[" in scaffold_section and ".load_state_dict" in scaffold_section,
        "persistent optimizers in scaffold": "client_optimizers[cid]" in scaffold_section,
        "NOT recreating optimizer in scaffold": scaffold_section.count("torch.optim.Adam") == 1,  # Should only be in initialization
    }
    
    print("\n✓ SCAFFOLD Fix Validation:")
    all_pass = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_pass = False
    
    return all_pass

if __name__ == "__main__":
    moon_pass = check_moon_fix()
    scaffold_pass = check_scaffold_fix()
    
    print("\n" + "="*60)
    if moon_pass and scaffold_pass:
        print("✅ ALL FIXES VERIFIED")
        sys.exit(0)
    else:
        print("❌ SOME FIXES MISSING")
        sys.exit(1)
