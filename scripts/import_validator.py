#!/usr/bin/env python3
"""Quick import and syntax validation for all experiment scripts."""

import sys
import importlib.util
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_script_imports(script_path: Path) -> bool:
    """Test that a script can be imported without errors."""
    print(f"Testing imports for {script_path.name}...", end=" ")

    try:
        # Use the module name for proper registration
        module_name = f"test_{script_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)

        # Register the module in sys.modules before execution
        import sys
        sys.modules[module_name] = module

        # Import the module (this will execute import statements)
        spec.loader.exec_module(module)

        # Clean up
        if module_name in sys.modules:
            del sys.modules[module_name]

        print("[PASS]")
        return True

    except Exception as e:
        print(f"[FAIL] {e}")
        return False

def main():
    """Test all main experiment scripts."""
    print("IMPORT VALIDATION FOR EXPERIMENT SCRIPTS")
    print("=" * 60)

    scripts_to_test = [
        "scripts/run_10seed_comprehensive.py",
        "scripts/run_ablation_study.py",
        "scripts/experiment_tracker.py",
        "scripts/run_flex_persona.py"
    ]

    results = []

    for script_name in scripts_to_test:
        script_path = project_root / script_name
        if script_path.exists():
            results.append(test_script_imports(script_path))
        else:
            print(f"Skipping {script_name} - file not found")
            results.append(False)

    print("\n" + "=" * 60)
    print("IMPORT VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("[PASS] All imports successful - scripts are syntactically valid")
        return 0
    else:
        print("[FAIL] Some imports failed - check syntax and dependencies")
        return 1

if __name__ == "__main__":
    sys.exit(main())